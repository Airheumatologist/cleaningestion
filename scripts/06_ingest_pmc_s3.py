#!/usr/bin/env python3
"""Directly ingest PMC XML from S3 metadata inventory into configured vector backend."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import (
    EmbeddingProvider,
    append_checkpoint,
    load_checkpoint,
    parse_pmc_xml_bytes,
    reset_pmc_xml_parse_failure_count,
    get_pmc_xml_parse_failure_count,
)
from lancedb_ingestion_sink import BaseIngestionSink, build_ingestion_sink

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SOURCE_PMC_OA = "pmc_oa"
SOURCE_PMC_AUTHOR = "pmc_author_manuscript"
DATASET_TO_SOURCE = {
    "pmc_oa": SOURCE_PMC_OA,
    "author_manuscript": SOURCE_PMC_AUTHOR,
}
RELEASE_MODES = ("all", "baseline", "incremental")
DEFAULT_CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pmc_s3_ingested_ids.txt"
DEFAULT_STATE_FILE = IngestionConfig.DATA_DIR / ".pmc_s3_inventory_state.json"


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_datasets(value: str) -> list[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError("--datasets must include at least one dataset key")
    invalid = [item for item in datasets if item not in DATASET_TO_SOURCE]
    if invalid:
        allowed = ", ".join(DATASET_TO_SOURCE)
        raise argparse.ArgumentTypeError(
            f"Unknown dataset(s): {', '.join(invalid)}. Allowed: {allowed}"
        )
    return datasets


def _checkpoint_key(source: str, metadata_key: str, etag: str) -> str:
    return f"{source}:{metadata_key}:{etag}"


def _select_source_type(metadata: Dict[str, Any], datasets: list[str]) -> Optional[str]:
    if "pmc_oa" in datasets and bool(metadata.get("is_pmc_openaccess")):
        return SOURCE_PMC_OA
    if "author_manuscript" in datasets and bool(metadata.get("is_manuscript")):
        return SOURCE_PMC_AUTHOR
    return None


def _download_xml_bytes(xml_http_url: str, timeout_seconds: int = 120) -> bytes:
    response = requests.get(xml_http_url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.content


def _extract_source_name(xml_url: str) -> str:
    path = urlparse(xml_url).path
    name = Path(path).name
    return name or "s3_object.xml"


def _process_metadata_entry(
    downloader_mod: Any,
    metadata_key: str,
    etag: str,
    datasets: list[str],
) -> Optional[Tuple[Dict[str, Any], str]]:
    try:
        metadata = downloader_mod._download_metadata_json(metadata_key)
        if not isinstance(metadata, dict):
            return None

        source_type = _select_source_type(metadata, datasets)
        if source_type is None:
            return None

        checkpoint_key = _checkpoint_key(source_type, metadata_key, etag)

        xml_url = metadata.get("xml_url")
        if not isinstance(xml_url, str) or not xml_url:
            return None

        xml_http_url = downloader_mod._normalize_s3_or_https_url(xml_url)
        xml_name = _extract_source_name(xml_http_url)
        xml_bytes = _download_xml_bytes(xml_http_url)

        strict_oa = source_type == SOURCE_PMC_OA
        article = parse_pmc_xml_bytes(
            xml_bytes,
            source_name=xml_name,
            require_pmid=strict_oa,
            require_open_access=strict_oa,
            require_commercial_license=strict_oa,
        )
        if not article:
            return None

        article["_source_type"] = source_type
        return article, checkpoint_key
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Failed metadata entry %s: %s", metadata_key, exc)
        return None


def _flush_articles(
    pmc_ingest_mod: Any,
    sink: BaseIngestionSink,
    embedding_provider: EmbeddingProvider,
    articles: list[Dict[str, Any]],
    checkpoint_ids: list[str],
    checkpoint_file: Path,
    processed_ids: set[str],
    checkpoint_lock: threading.Lock,
) -> int:
    if not articles:
        return 0

    points, _chunk_ids = pmc_ingest_mod.build_points(
        articles,
        embedding_provider,
    )
    if not points:
        return 0

    written = sink.write_points(points)
    if written > 0 and checkpoint_ids:
        with checkpoint_lock:
            append_checkpoint(checkpoint_file, checkpoint_ids)
            processed_ids.update(checkpoint_ids)
    return written


def _iter_super_batches(
    entries: list[tuple[str, str, str]],
    super_batch_size: int,
) -> list[list[tuple[str, str, str]]]:
    size = max(1, int(super_batch_size))
    return [entries[i:i + size] for i in range(0, len(entries), size)]


def _run_super_batch_pipeline(
    super_batch_entries: list[tuple[str, str, str]],
    *,
    downloader_mod: Any,
    pmc_ingest_mod: Any,
    sink: BaseIngestionSink,
    embedding_provider: EmbeddingProvider,
    datasets: list[str],
    workers: int,
    processed_ids: set[str],
    checkpoint_file: Path,
    checkpoint_lock: threading.Lock,
    article_queue_size: int,
    points_queue_size: int,
    embed_article_batch_size: int,
    embed_workers: int = 1,
) -> tuple[int, int]:
    """Run one super-batch through a parallel pipeline.

    Architecture:
      - ``workers`` threads fetch & parse metadata entries (I/O bound)
      - ``embed_workers`` threads each embed a batch of articles (CPU+network bound)
      - 1 writer thread writes points to turbopuffer

    The article_queue feeds all embed workers. Each embed worker emits a
    None sentinel to the points_queue when it finishes; the writer thread
    counts N sentinels before terminating.
    """
    effective_article_queue = max(1, article_queue_size)
    effective_points_queue = max(1, points_queue_size * max(1, embed_workers))

    article_queue: queue.Queue[Optional[Tuple[Dict[str, Any], str]]] = queue.Queue(
        maxsize=effective_article_queue
    )
    points_queue: queue.Queue[Optional[Tuple[list[Any], list[str]]]] = queue.Queue(
        maxsize=effective_points_queue
    )
    error_queue: queue.Queue[BaseException] = queue.Queue()
    processed_lock = threading.Lock()
    counters = {"inserted": 0, "skipped": 0}
    embed_done_count = 0
    embed_done_lock = threading.Lock()

    def _enqueue_error(exc: BaseException) -> None:
        if error_queue.empty():
            error_queue.put(exc)

    def _embed_worker(worker_idx: int) -> None:
        """One of ``embed_workers`` parallel embedding threads."""
        batch_articles: list[Dict[str, Any]] = []
        batch_checkpoint_ids: list[str] = []
        try:
            while True:
                try:
                    item = article_queue.get(timeout=300)
                except Exception:
                    break
                if item is None:
                    # Poison pill - put back for the next waiting embed worker
                    article_queue.put(None)
                    break
                article, checkpoint_id = item
                batch_articles.append(article)
                batch_checkpoint_ids.append(checkpoint_id)
                if len(batch_articles) >= embed_article_batch_size:
                    if not error_queue.empty():
                        break
                    try:
                        points, _ = pmc_ingest_mod.build_points(
                            batch_articles,
                            embedding_provider,
                        )
                        points_queue.put((points, batch_checkpoint_ids.copy()))
                    except Exception as exc:
                        _enqueue_error(exc)
                        return
                    finally:
                        batch_articles.clear()
                        batch_checkpoint_ids.clear()

            # Flush remainder
            if batch_articles and error_queue.empty():
                try:
                    points, _ = pmc_ingest_mod.build_points(
                        batch_articles,
                        embedding_provider,
                    )
                    points_queue.put((points, batch_checkpoint_ids.copy()))
                except Exception as exc:
                    _enqueue_error(exc)
        except Exception as exc:
            _enqueue_error(exc)
        finally:
            nonlocal embed_done_count
            with embed_done_lock:
                embed_done_count += 1
                if embed_done_count >= embed_workers:
                    # Last embed worker sends the terminal sentinel to writer
                    points_queue.put(None)

    def _writer_worker() -> None:
        try:
            while True:
                try:
                    payload = points_queue.get(timeout=600)
                except Exception:
                    break
                if payload is None:
                    break  # All embed workers done
                points, checkpoint_ids = payload
                if not points:
                    counters["skipped"] += len(checkpoint_ids)
                    continue
                written = sink.write_points(points)
                counters["inserted"] += written
                if checkpoint_ids:
                    with checkpoint_lock:
                        append_checkpoint(checkpoint_file, checkpoint_ids)
                    with processed_lock:
                        processed_ids.update(checkpoint_ids)
        except Exception as exc:
            _enqueue_error(exc)

    actual_embed_workers = max(1, embed_workers)
    logger.info(
        "Super-batch pipeline: %s download workers, %s embed workers, 1 write worker",
        workers,
        actual_embed_workers,
    )
    embed_threads = [
        threading.Thread(
            target=_embed_worker,
            args=(i,),
            name=f"pmc-s3-embed-{i}",
            daemon=True,
        )
        for i in range(actual_embed_workers)
    ]
    write_thread = threading.Thread(target=_writer_worker, name="pmc-s3-write", daemon=True)

    for t in embed_threads:
        t.start()
    write_thread.start()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_metadata_entry,
                downloader_mod,
                metadata_key,
                etag,
                datasets,
            ): metadata_key
            for metadata_key, _last_modified, etag in super_batch_entries
        }

        for future in as_completed(futures):
            if not error_queue.empty():
                break
            metadata_key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("Worker failed for %s: %s", metadata_key, exc)
                counters["skipped"] += 1
                continue

            if result is None:
                counters["skipped"] += 1
                continue

            article, checkpoint_id = result
            if checkpoint_id in processed_ids:
                counters["skipped"] += 1
                continue
            article_queue.put((article, checkpoint_id))

    # Signal all embed workers to stop via single None (they relay it)
    article_queue.put(None)
    for t in embed_threads:
        t.join()
    write_thread.join()

    if not error_queue.empty():
        raise error_queue.get()
    return counters["inserted"], counters["skipped"]


def run_ingestion_s3(
    datasets: list[str],
    release_mode: str,
    max_files: Optional[int],
    workers: int,
    checkpoint_file: Path,
    state_file: Path,
    precreate_only: bool = False,
    super_batch_size: int = 250,
    article_queue_size: int = 256,
    points_queue_size: int = 32,
    embed_article_batch_size: int = 120,
    embed_workers: int = 1,
) -> None:
    ensure_data_dirs()
    reset_pmc_xml_parse_failure_count()

    scripts_dir = Path(__file__).resolve().parent
    downloader_mod = _load_script_module("download_pmc_unified_mod", scripts_dir / "01_download_pmc_unified.py")
    pmc_ingest_mod = _load_script_module("ingest_pmc_mod", scripts_dir / "06_ingest_pmc.py")

    sink = build_ingestion_sink(namespace_override=IngestionConfig.TURBOPUFFER_NAMESPACE_PMC)
    embedding_provider = EmbeddingProvider()

    processed_ids = load_checkpoint(checkpoint_file)
    checkpoint_lock = threading.Lock()
    logger.info("PMC S3 checkpoint loaded: %s IDs", len(processed_ids))

    state = downloader_mod._load_state(state_file)
    dataset_sig = downloader_mod._dataset_signature(datasets)
    cutoff = downloader_mod._select_cutoff_for_incremental(state, dataset_sig, release_mode)

    eligible_entries: list[tuple[str, str, str]] = []
    max_seen_last_modified: Optional[str] = None
    scanned = 0
    included = 0
    stopped_by_max_files = False

    logger.info("Scanning PMC metadata inventory...")
    for metadata_key, last_modified, etag in downloader_mod._iter_metadata_entries():
        scanned += 1
        if max_seen_last_modified is None or last_modified > max_seen_last_modified:
            max_seen_last_modified = last_modified

        if not downloader_mod._should_include_entry(last_modified, cutoff):
            continue

        included += 1
        if max_files is not None and len(eligible_entries) >= max_files:
            stopped_by_max_files = True
            break

        eligible_entries.append((metadata_key, last_modified, etag))

    logger.info(
        "Inventory scan complete: scanned=%s included=%s eligible=%s",
        scanned,
        included,
        len(eligible_entries),
    )

    if not eligible_entries:
        logger.warning("No eligible metadata entries found.")
        return

    total_inserted = 0
    total_skipped = 0
    start_time = time.time()

    if precreate_only:
        for metadata_key, _last_modified, etag in eligible_entries:
            result = _process_metadata_entry(downloader_mod, metadata_key, etag, datasets)
            if result is None:
                continue
            article, checkpoint_id = result
            if checkpoint_id in processed_ids:
                continue
            written = _flush_articles(
                pmc_ingest_mod=pmc_ingest_mod,
                sink=sink,
                embedding_provider=embedding_provider,
                articles=[article],
                checkpoint_ids=[checkpoint_id],
                checkpoint_file=checkpoint_file,
                processed_ids=processed_ids,
                checkpoint_lock=checkpoint_lock,
            )
            total_inserted += written
            if written > 0:
                logger.info("Precreate-only mode complete after first successful write.")
                break
    else:
        super_batches = _iter_super_batches(eligible_entries, super_batch_size=super_batch_size)
        processed_entries = 0
        for sb_index, sb_entries in enumerate(super_batches, 1):
            inserted_now, skipped_now = _run_super_batch_pipeline(
                sb_entries,
                downloader_mod=downloader_mod,
                pmc_ingest_mod=pmc_ingest_mod,
                sink=sink,
                embedding_provider=embedding_provider,
                datasets=datasets,
                workers=workers,
                processed_ids=processed_ids,
                checkpoint_file=checkpoint_file,
                checkpoint_lock=checkpoint_lock,
                article_queue_size=article_queue_size,
                points_queue_size=points_queue_size,
                embed_article_batch_size=embed_article_batch_size,
                embed_workers=embed_workers,
            )
            total_inserted += inserted_now
            total_skipped += skipped_now
            processed_entries += len(sb_entries)

            elapsed = max(time.time() - start_time, 0.001)
            rate = total_inserted / elapsed
            logger.info(
                "Super-batch %s/%s committed | processed=%s/%s inserted=%s skipped=%s rate=%.2f rows/sec",
                sb_index,
                len(super_batches),
                processed_entries,
                len(eligible_entries),
                total_inserted,
                total_skipped,
                rate,
            )

    if max_seen_last_modified is not None and release_mode == "incremental" and not stopped_by_max_files:
        state.setdefault("last_modified_by_signature", {})[dataset_sig] = max_seen_last_modified
        downloader_mod._save_state(state_file, state)
        logger.info("Updated incremental state for %s to %s", dataset_sig, max_seen_last_modified)

    elapsed = time.time() - start_time
    parse_failures = get_pmc_xml_parse_failure_count()
    logger.info(
        "PMC S3 ingestion complete. Inserted=%s Skipped=%s ParseFailures=%s Time=%.1fs",
        total_inserted,
        total_skipped,
        parse_failures,
        elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct ingest PMC XML from S3 metadata inventory")
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=["pmc_oa", "author_manuscript"],
        help="Comma-separated datasets: pmc_oa,author_manuscript",
    )
    parser.add_argument("--release-mode", choices=RELEASE_MODES, default="all")
    parser.add_argument("--max-files", type=int, default=None, help="Stop after N eligible metadata rows")
    parser.add_argument("--workers", type=int, default=4, help="Parallel metadata/xml workers")
    parser.add_argument(
        "--super-batch-size",
        type=int,
        default=int(os.getenv("PMC_S3_SUPER_BATCH_SIZE", "250")),
        help="Process and commit metadata entries in resumable super-batches.",
    )
    parser.add_argument(
        "--article-queue-size",
        type=int,
        default=int(os.getenv("PMC_S3_ARTICLE_QUEUE_SIZE", "256")),
        help="Queue capacity between parse producers and embed worker.",
    )
    parser.add_argument(
        "--points-queue-size",
        type=int,
        default=int(os.getenv("PMC_S3_POINTS_QUEUE_SIZE", "32")),
        help="Queue capacity between embed worker and write worker.",
    )
    parser.add_argument(
        "--embed-article-batch-size",
        type=int,
        default=int(os.getenv("PMC_S3_EMBED_ARTICLE_BATCH_SIZE", str(IngestionConfig.BATCH_SIZE))),
        help="Number of parsed articles per embedding/write payload batch.",
    )
    parser.add_argument(
        "--embed-workers",
        type=int,
        default=int(os.getenv("EMBEDDING_CONCURRENCY", "1")),
        help=(
            "Number of parallel embedding worker threads. Each worker independently "
            "pulls articles from the queue and calls DeepInfra. "
            "Set to match EMBEDDING_CONCURRENCY for best throughput. "
            "Must stay within DeepInfra rate limits (200 req/sec)."
        ),
    )
    parser.add_argument(
        "--precreate-only",
        action="store_true",
        help="Create target table with first valid ingested row, then stop.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path(os.getenv("PMC_S3_CHECKPOINT_FILE", str(DEFAULT_CHECKPOINT_FILE))),
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path(os.getenv("PMC_S3_STATE_FILE", str(DEFAULT_STATE_FILE))),
    )
    args = parser.parse_args()

    run_ingestion_s3(
        datasets=args.datasets,
        release_mode=args.release_mode,
        max_files=args.max_files,
        workers=args.workers,
        checkpoint_file=args.checkpoint_file,
        state_file=args.state_file,
        precreate_only=args.precreate_only,
        super_batch_size=args.super_batch_size,
        article_queue_size=args.article_queue_size,
        points_queue_size=args.points_queue_size,
        embed_article_batch_size=args.embed_article_batch_size,
        embed_workers=args.embed_workers,
    )


if __name__ == "__main__":
    main()
