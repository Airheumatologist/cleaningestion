#!/usr/bin/env python3
"""Directly ingest PMC XML from S3 metadata inventory into configured vector backend."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import (
    EmbeddingProvider,
    append_checkpoint,
    get_pmc_xml_parse_failure_count,
    load_checkpoint,
    parse_pmc_xml_bytes,
    pop_last_pmc_parse_skip_reason,
    reset_pmc_xml_parse_failure_count,
)
from turbopuffer_ingestion_sink import BaseIngestionSink, build_ingestion_sink

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
_thread_local = threading.local()

SOURCE_PMC_OA = "pmc_oa"
SOURCE_PMC_AUTHOR = "pmc_author_manuscript"
DATASET_TO_SOURCE = {
    "pmc_oa": SOURCE_PMC_OA,
    "author_manuscript": SOURCE_PMC_AUTHOR,
}
RELEASE_MODES = ("all", "baseline", "incremental")
DEFAULT_CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pmc_s3_ingested_ids.txt"
DEFAULT_STATE_FILE = IngestionConfig.DATA_DIR / ".pmc_s3_inventory_state.json"
DEFAULT_XML_FETCH_TIMEOUT_SECONDS = int(os.getenv("PMC_S3_XML_FETCH_TIMEOUT_SECONDS", "120"))
DEFAULT_XML_FETCH_RETRIES = int(os.getenv("PMC_S3_XML_FETCH_RETRIES", "3"))
DEFAULT_XML_FETCH_BACKOFF_FACTOR = float(os.getenv("PMC_S3_XML_FETCH_BACKOFF_FACTOR", "0.5"))
DEFAULT_ARTICLE_QUEUE_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("PMC_S3_ARTICLE_QUEUE_WAIT_TIMEOUT_SECONDS", "300")
)
DEFAULT_POINTS_QUEUE_WAIT_TIMEOUT_SECONDS = float(
    os.getenv("PMC_S3_POINTS_QUEUE_WAIT_TIMEOUT_SECONDS", "600")
)
DEFAULT_NAMESPACE_SHARD_COUNT = int(os.getenv("PMC_S3_NAMESPACE_SHARD_COUNT", "1") or "1")
DEFAULT_NAMESPACE_SHARD_PATTERN = os.getenv(
    "PMC_S3_NAMESPACE_SHARD_PATTERN",
    "{base}_shard_{shard}",
)
FAILED_METADATA_KEYS_ENV = "PMC_S3_FAILED_METADATA_KEYS_FILE"
SKIPPED_CHECKPOINT_SUFFIX = ".skipped.tsv"


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


def _reserve_checkpoint_key(
    processed_ids: set[str],
    inflight_ids: set[str],
    checkpoint_lock: threading.Lock,
    checkpoint_key: str,
) -> bool:
    with checkpoint_lock:
        if checkpoint_key in processed_ids or checkpoint_key in inflight_ids:
            return False
        inflight_ids.add(checkpoint_key)
        return True


def _release_checkpoint_keys(
    inflight_ids: set[str],
    checkpoint_lock: threading.Lock,
    checkpoint_keys: Iterable[str],
) -> None:
    with checkpoint_lock:
        for checkpoint_key in checkpoint_keys:
            inflight_ids.discard(checkpoint_key)


def _finalize_checkpoint_key(
    processed_ids: set[str],
    inflight_ids: set[str],
    checkpoint_lock: threading.Lock,
    checkpoint_key: str,
) -> None:
    append_checkpoint([checkpoint_key])
    with checkpoint_lock:
        processed_ids.add(checkpoint_key)
        inflight_ids.discard(checkpoint_key)


def _select_source_type(metadata: Dict[str, Any], datasets: list[str]) -> Optional[str]:
    if "pmc_oa" in datasets and bool(metadata.get("is_pmc_openaccess")):
        return SOURCE_PMC_OA
    if "author_manuscript" in datasets and bool(metadata.get("is_manuscript")):
        return SOURCE_PMC_AUTHOR
    return None


def _derive_skipped_checkpoint_file(checkpoint_file: Path) -> Path:
    if checkpoint_file.suffix:
        return checkpoint_file.with_name(f"{checkpoint_file.stem}{SKIPPED_CHECKPOINT_SUFFIX}")
    return checkpoint_file.with_name(f"{checkpoint_file.name}{SKIPPED_CHECKPOINT_SUFFIX}")


def _load_skipped_checkpoint(skip_checkpoint_file: Path) -> set[str]:
    if not skip_checkpoint_file.exists():
        return set()

    skipped_ids: set[str] = set()
    try:
        for raw_line in skip_checkpoint_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            checkpoint_key = line.split("\t", 1)[0].strip()
            if checkpoint_key:
                skipped_ids.add(checkpoint_key)
    except Exception as exc:
        logger.warning("Failed to load skipped checkpoint from %s: %s", skip_checkpoint_file, exc)
        return set()
    return skipped_ids


def _metadata_non_open_reason(metadata: Dict[str, Any], source_type: str) -> Optional[str]:
    if source_type != SOURCE_PMC_OA:
        return None

    bool_fields = (
        "is_pmc_openaccess",
        "is_open_access",
        "open_access",
        "openaccess",
        "is_commercial_license",
        "commercial_license",
        "allows_commercial_use",
    )
    for field in bool_fields:
        if field not in metadata:
            continue
        value = metadata.get(field)
        if isinstance(value, bool):
            if not value:
                if field in {"is_commercial_license", "commercial_license", "allows_commercial_use"}:
                    return f"metadata field {field} is false"
                return f"metadata field {field} is false"
            continue
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"false", "0", "no", "n"}:
                if field in {"is_commercial_license", "commercial_license", "allows_commercial_use"}:
                    return f"metadata field {field} is false"
                return f"metadata field {field} is false"

    license_fields = (
        "license",
        "license_type",
        "license-name",
        "license_name",
        "license_text",
    )
    non_commercial_markers = (
        "cc-by-nc",
        "by-nc",
        "non-commercial",
        "noncommercial",
    )
    for field in license_fields:
        value = metadata.get(field)
        if not isinstance(value, str):
            continue
        lowered = value.strip().lower()
        if lowered and any(marker in lowered for marker in non_commercial_markers):
            return f"metadata field {field} marks a non-commercial license ({value})"

    return None


def _append_skipped_checkpoint(
    skip_checkpoint_file: Optional[Path],
    skip_checkpoint_lock: Optional[threading.Lock],
    checkpoint_key: str,
    stage: str,
    reason: str,
) -> None:
    if skip_checkpoint_file is None:
        return

    safe_stage = " ".join(stage.split())
    safe_reason = " ".join(reason.split())
    skip_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    lock = skip_checkpoint_lock or threading.Lock()
    with lock:
        with skip_checkpoint_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{checkpoint_key}\t{safe_stage}\t{safe_reason}\n")


def _finalize_skipped_checkpoint_key(
    completed_ids: set[str],
    skipped_ids: set[str],
    inflight_ids: set[str],
    checkpoint_lock: threading.Lock,
    checkpoint_key: str,
    *,
    skip_checkpoint_file: Optional[Path],
    skip_checkpoint_lock: Optional[threading.Lock],
    stage: str,
    reason: str,
) -> None:
    _append_skipped_checkpoint(
        skip_checkpoint_file,
        skip_checkpoint_lock,
        checkpoint_key,
        stage,
        reason,
    )
    with checkpoint_lock:
        completed_ids.add(checkpoint_key)
        skipped_ids.add(checkpoint_key)
        inflight_ids.discard(checkpoint_key)


def _compute_namespace_shards(
    *,
    base_namespace: str,
    shard_count: int,
    pattern: str,
) -> list[str]:
    count = max(1, int(shard_count))
    if count == 1:
        return [base_namespace]

    template = (pattern or "").strip() or "{base}_shard_{shard}"
    if "{base}" not in template:
        template = "{base}_" + template
    if "{shard}" not in template:
        template = template + "_{shard}"

    namespaces: list[str] = []
    for shard in range(count):
        namespace = template.format(base=base_namespace, shard=shard)
        namespaces.append(namespace)
    return namespaces


def _stable_shard_index(token: str, shard_count: int) -> int:
    if shard_count <= 1:
        return 0
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % shard_count


def _extract_doc_id_from_point(point: Any) -> str:
    payload = getattr(point, "payload", None)
    if isinstance(payload, dict):
        doc_id = payload.get("doc_id")
        if doc_id is not None:
            value = str(doc_id).strip()
            if value:
                return value
    return ""


def _extract_point_id(point: Any) -> str:
    point_id = getattr(point, "id", None)
    if point_id is None:
        return ""
    value = str(point_id).strip()
    return value


def _dedupe_points_for_upsert(points: list[Any]) -> list[Any]:
    if not points:
        return points

    deduped_points: list[Any] = []
    seen_ids: set[str] = set()
    duplicates_removed = 0

    for point in points:
        point_id = _extract_point_id(point)
        if point_id and point_id in seen_ids:
            duplicates_removed += 1
            continue
        if point_id:
            seen_ids.add(point_id)
        deduped_points.append(point)

    if duplicates_removed:
        logger.warning(
            "Dropped %s duplicate point IDs from write batch before upsert",
            duplicates_removed,
        )
    return deduped_points


def _write_points(
    *,
    points: list[Any],
    checkpoint_ids: list[str],
    sink: BaseIngestionSink,
    shard_sinks: Optional[list[BaseIngestionSink]],
) -> int:
    points = _dedupe_points_for_upsert(points)
    if not points:
        return 0
    if not shard_sinks or len(shard_sinks) <= 1:
        return sink.write_points(points)

    shard_count = len(shard_sinks)
    shard_to_points: dict[int, list[Any]] = {i: [] for i in range(shard_count)}
    fallback_token = checkpoint_ids[0] if checkpoint_ids else "pmc_s3"
    for index, point in enumerate(points):
        doc_id = _extract_doc_id_from_point(point)
        route_token = doc_id or f"{fallback_token}:{index}"
        shard_idx = _stable_shard_index(route_token, shard_count)
        shard_to_points[shard_idx].append(point)

    written = 0
    for shard_idx in sorted(shard_to_points):
        shard_points = shard_to_points[shard_idx]
        if not shard_points:
            continue
        written += shard_sinks[shard_idx].write_points(shard_points)
    return written


def _get_xml_download_session(
    *,
    max_retries: int,
    backoff_factor: float,
    pool_size: int = 64,
) -> requests.Session:
    session_cache = getattr(_thread_local, "xml_download_sessions", None)
    if session_cache is None:
        session_cache = {}
        _thread_local.xml_download_sessions = session_cache

    cache_key = (max_retries, backoff_factor, pool_size)
    session = session_cache.get(cache_key)
    if session is not None:
        return session

    retry = Retry(
        total=max(0, max_retries),
        connect=max(0, max_retries),
        read=max(0, max_retries),
        status=max(0, max_retries),
        backoff_factor=max(0.0, backoff_factor),
        allowed_methods=frozenset({"GET"}),
        status_forcelist=(408, 429, 500, 502, 503, 504),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=retry,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session_cache[cache_key] = session
    return session


def _download_xml_bytes(
    xml_http_url: str,
    timeout_seconds: int = DEFAULT_XML_FETCH_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_XML_FETCH_RETRIES,
    backoff_factor: float = DEFAULT_XML_FETCH_BACKOFF_FACTOR,
) -> bytes:
    session = _get_xml_download_session(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
    )
    response = session.get(xml_http_url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.content


def _extract_source_name(xml_url: str) -> str:
    path = urlparse(xml_url).path
    name = Path(path).name
    return name or "s3_object.xml"


def _format_failure_reason(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _append_failed_metadata_key(
    failed_metadata_keys_file: Optional[Path],
    failed_metadata_keys_lock: Optional[threading.Lock],
    metadata_key: str,
    stage: str,
    reason: str,
) -> None:
    if failed_metadata_keys_file is None:
        return

    safe_reason = " ".join(reason.split())
    failed_metadata_keys_file.parent.mkdir(parents=True, exist_ok=True)
    lock = failed_metadata_keys_lock or threading.Lock()
    with lock:
        with failed_metadata_keys_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{metadata_key}\t{stage}\t{safe_reason}\n")


def _record_metadata_entry_issue(
    metadata_key: str,
    stage: str,
    reason: str,
    *,
    failed_metadata_keys_file: Optional[Path],
    failed_metadata_keys_lock: Optional[threading.Lock],
    level: int,
) -> None:
    logger.log(level, "Skipping metadata entry %s at %s stage: %s", metadata_key, stage, reason)
    _append_failed_metadata_key(
        failed_metadata_keys_file,
        failed_metadata_keys_lock,
        metadata_key,
        stage,
        reason,
    )


def _process_metadata_entry(
    downloader_mod: Any,
    metadata_key: str,
    etag: str,
    datasets: list[str],
    processed_ids: set[str],
    skipped_ids: set[str],
    inflight_ids: set[str],
    checkpoint_lock: threading.Lock,
    xml_fetch_timeout_seconds: int = DEFAULT_XML_FETCH_TIMEOUT_SECONDS,
    xml_fetch_retries: int = DEFAULT_XML_FETCH_RETRIES,
    xml_fetch_backoff_factor: float = DEFAULT_XML_FETCH_BACKOFF_FACTOR,
    failed_metadata_keys_file: Optional[Path] = None,
    failed_metadata_keys_lock: Optional[threading.Lock] = None,
    skip_checkpoint_file: Optional[Path] = None,
    skip_checkpoint_lock: Optional[threading.Lock] = None,
) -> Optional[Tuple[Dict[str, Any], str]]:
    checkpoint_key = ""
    reserved = False
    try:
        try:
            metadata = downloader_mod._download_metadata_json(metadata_key)
        except Exception as exc:
            _record_metadata_entry_issue(
                metadata_key,
                "metadata fetch",
                _format_failure_reason(exc),
                failed_metadata_keys_file=failed_metadata_keys_file,
                failed_metadata_keys_lock=failed_metadata_keys_lock,
                level=logging.WARNING,
            )
            return None

        if not isinstance(metadata, dict):
            _record_metadata_entry_issue(
                metadata_key,
                "metadata fetch",
                "metadata payload unavailable or invalid",
                failed_metadata_keys_file=failed_metadata_keys_file,
                failed_metadata_keys_lock=failed_metadata_keys_lock,
                level=logging.WARNING,
            )
            return None

        source_type = _select_source_type(metadata, datasets)
        if source_type is None:
            _record_metadata_entry_issue(
                metadata_key,
                "source filter",
                f"metadata does not match requested datasets={','.join(datasets)}",
                failed_metadata_keys_file=failed_metadata_keys_file,
                failed_metadata_keys_lock=failed_metadata_keys_lock,
                level=logging.INFO,
            )
            return None

        checkpoint_key = _checkpoint_key(source_type, metadata_key, etag)
        if checkpoint_key in skipped_ids:
            return None
        if not _reserve_checkpoint_key(processed_ids, inflight_ids, checkpoint_lock, checkpoint_key):
            return None
        reserved = True

        early_skip_reason = _metadata_non_open_reason(metadata, source_type)
        if early_skip_reason is not None:
            _record_metadata_entry_issue(
                metadata_key,
                "metadata filter",
                early_skip_reason,
                failed_metadata_keys_file=failed_metadata_keys_file,
                failed_metadata_keys_lock=failed_metadata_keys_lock,
                level=logging.INFO,
            )
            _finalize_skipped_checkpoint_key(
                processed_ids,
                skipped_ids,
                inflight_ids,
                checkpoint_lock,
                checkpoint_key,
                skip_checkpoint_file=skip_checkpoint_file,
                skip_checkpoint_lock=skip_checkpoint_lock,
                stage="metadata filter",
                reason=early_skip_reason,
            )
            reserved = False
            return None

        xml_url = metadata.get("xml_url")
        if not isinstance(xml_url, str) or not xml_url:
            _record_metadata_entry_issue(
                metadata_key,
                "xml_url missing",
                "metadata is missing xml_url",
                failed_metadata_keys_file=failed_metadata_keys_file,
                failed_metadata_keys_lock=failed_metadata_keys_lock,
                level=logging.INFO,
            )
            return None

        try:
            xml_http_url = downloader_mod._normalize_s3_or_https_url(xml_url)
            xml_bytes = _download_xml_bytes(
                xml_http_url,
                timeout_seconds=xml_fetch_timeout_seconds,
                max_retries=xml_fetch_retries,
                backoff_factor=xml_fetch_backoff_factor,
            )
        except Exception as exc:
            _record_metadata_entry_issue(
                metadata_key,
                "xml download",
                _format_failure_reason(exc),
                failed_metadata_keys_file=failed_metadata_keys_file,
                failed_metadata_keys_lock=failed_metadata_keys_lock,
                level=logging.WARNING,
            )
            return None

        xml_name = _extract_source_name(xml_http_url)

        try:
            article = parse_pmc_xml_bytes(
                xml_bytes,
                source_name=xml_name,
                require_pmid=False,
                require_open_access=False,
                require_commercial_license=True,
            )
        except Exception as exc:
            _record_metadata_entry_issue(
                metadata_key,
                "parse",
                _format_failure_reason(exc),
                failed_metadata_keys_file=failed_metadata_keys_file,
                failed_metadata_keys_lock=failed_metadata_keys_lock,
                level=logging.WARNING,
            )
            _finalize_skipped_checkpoint_key(
                processed_ids,
                skipped_ids,
                inflight_ids,
                checkpoint_lock,
                checkpoint_key,
                skip_checkpoint_file=skip_checkpoint_file,
                skip_checkpoint_lock=skip_checkpoint_lock,
                stage="parse",
                reason=_format_failure_reason(exc),
            )
            reserved = False
            return None
        if not article:
            skip_reason = pop_last_pmc_parse_skip_reason() or "PMC XML parser returned no article"
            _record_metadata_entry_issue(
                metadata_key,
                "parse",
                skip_reason,
                failed_metadata_keys_file=failed_metadata_keys_file,
                failed_metadata_keys_lock=failed_metadata_keys_lock,
                level=logging.WARNING,
            )
            _finalize_skipped_checkpoint_key(
                processed_ids,
                skipped_ids,
                inflight_ids,
                checkpoint_lock,
                checkpoint_key,
                skip_checkpoint_file=skip_checkpoint_file,
                skip_checkpoint_lock=skip_checkpoint_lock,
                stage="parse",
                reason=skip_reason,
            )
            reserved = False
            return None

        article["_source_type"] = source_type
        reserved = False
        return article, checkpoint_key
    finally:
        if reserved:
            _release_checkpoint_keys(inflight_ids, checkpoint_lock, [checkpoint_key])


def _flush_articles(
    pmc_ingest_mod: Any,
    sink: BaseIngestionSink,
    shard_sinks: Optional[list[BaseIngestionSink]],
    embedding_provider: EmbeddingProvider,
    articles: list[Dict[str, Any]],
    checkpoint_ids: list[str],
    checkpoint_file: Path,
    processed_ids: set[str],
    inflight_ids: set[str],
    checkpoint_lock: threading.Lock,
) -> int:
    if not articles:
        return 0

    try:
        points, _chunk_ids = pmc_ingest_mod.build_points(
            articles,
            embedding_provider,
        )
    except Exception:
        _release_checkpoint_keys(inflight_ids, checkpoint_lock, checkpoint_ids)
        raise

    if not points:
        _release_checkpoint_keys(inflight_ids, checkpoint_lock, checkpoint_ids)
        return 0

    try:
        written = _write_points(
            points=points,
            checkpoint_ids=checkpoint_ids,
            sink=sink,
            shard_sinks=shard_sinks,
        )
    except Exception:
        _release_checkpoint_keys(inflight_ids, checkpoint_lock, checkpoint_ids)
        raise
    if written > 0 and checkpoint_ids:
        try:
            append_checkpoint(checkpoint_file, checkpoint_ids)
            with checkpoint_lock:
                processed_ids.update(checkpoint_ids)
        finally:
            _release_checkpoint_keys(inflight_ids, checkpoint_lock, checkpoint_ids)
    else:
        _release_checkpoint_keys(inflight_ids, checkpoint_lock, checkpoint_ids)
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
    shard_sinks: Optional[list[BaseIngestionSink]],
    embedding_provider: EmbeddingProvider,
    datasets: list[str],
    workers: int,
    processed_ids: set[str],
    skipped_ids: set[str],
    inflight_ids: set[str],
    checkpoint_file: Path,
    checkpoint_lock: threading.Lock,
    skip_checkpoint_file: Optional[Path],
    skip_checkpoint_lock: threading.Lock,
    article_queue_size: int,
    points_queue_size: int,
    embed_article_batch_size: int,
    embed_workers: int = 1,
    write_workers: int = 1,
    xml_fetch_timeout_seconds: int = DEFAULT_XML_FETCH_TIMEOUT_SECONDS,
    xml_fetch_retries: int = DEFAULT_XML_FETCH_RETRIES,
    xml_fetch_backoff_factor: float = DEFAULT_XML_FETCH_BACKOFF_FACTOR,
    article_queue_wait_timeout_seconds: float = DEFAULT_ARTICLE_QUEUE_WAIT_TIMEOUT_SECONDS,
    points_queue_wait_timeout_seconds: float = DEFAULT_POINTS_QUEUE_WAIT_TIMEOUT_SECONDS,
    failed_metadata_keys_file: Optional[Path] = None,
    failed_metadata_keys_lock: Optional[threading.Lock] = None,
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
    counters = {"inserted": 0, "skipped": 0}
    counters_lock = threading.Lock()
    actual_write_workers = max(1, write_workers)
    embed_done_count = 0
    embed_done_lock = threading.Lock()

    def _enqueue_error(exc: BaseException) -> None:
        if error_queue.empty():
            error_queue.put(exc)

    def _embed_worker(worker_idx: int) -> None:
        """One of ``embed_workers`` parallel embedding threads."""
        batch_articles: list[Dict[str, Any]] = []
        batch_checkpoint_ids: list[str] = []

        def _release_pending_batch() -> None:
            if batch_checkpoint_ids:
                _release_checkpoint_keys(inflight_ids, checkpoint_lock, batch_checkpoint_ids)
                batch_checkpoint_ids.clear()
            batch_articles.clear()

        try:
            while True:
                try:
                    item = article_queue.get(timeout=article_queue_wait_timeout_seconds)
                except Exception:
                    _release_pending_batch()
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
                        _release_pending_batch()
                        break
                    try:
                        points, _ = pmc_ingest_mod.build_points(
                            batch_articles,
                            embedding_provider,
                        )
                        points_queue.put((points, batch_checkpoint_ids.copy()))
                    except Exception as exc:
                        _release_pending_batch()
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
                    _release_pending_batch()
                    _enqueue_error(exc)
            elif batch_articles:
                _release_pending_batch()
        except Exception as exc:
            _release_pending_batch()
            _enqueue_error(exc)
        finally:
            nonlocal embed_done_count
            with embed_done_lock:
                embed_done_count += 1
                if embed_done_count >= embed_workers:
                    # Last embed worker sends one sentinel per write worker
                    for _ in range(actual_write_workers):
                        points_queue.put(None)

    def _writer_worker(worker_idx: int) -> None:
        try:
            while True:
                try:
                    payload = points_queue.get(timeout=points_queue_wait_timeout_seconds)
                except Exception:
                    break
                if payload is None:
                    break  # This writer's sentinel received
                points, checkpoint_ids = payload
                if not points:
                    with counters_lock:
                        counters["skipped"] += len(checkpoint_ids)
                    _release_checkpoint_keys(inflight_ids, checkpoint_lock, checkpoint_ids)
                    continue
                try:
                    written = _write_points(
                        points=points,
                        checkpoint_ids=checkpoint_ids,
                        sink=sink,
                        shard_sinks=shard_sinks,
                    )
                except Exception as exc:
                    _release_checkpoint_keys(inflight_ids, checkpoint_lock, checkpoint_ids)
                    raise
                with counters_lock:
                    counters["inserted"] += written
                if checkpoint_ids and written > 0:
                    try:
                        append_checkpoint(checkpoint_file, checkpoint_ids)
                        with checkpoint_lock:
                            processed_ids.update(checkpoint_ids)
                    finally:
                        _release_checkpoint_keys(inflight_ids, checkpoint_lock, checkpoint_ids)
                elif checkpoint_ids:
                    _release_checkpoint_keys(inflight_ids, checkpoint_lock, checkpoint_ids)
        except Exception as exc:
            _enqueue_error(exc)

    actual_embed_workers = max(1, embed_workers)
    logger.info(
        "Super-batch pipeline: %s download workers, %s embed workers, %s write workers",
        workers,
        actual_embed_workers,
        actual_write_workers,
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
    write_threads = [
        threading.Thread(
            target=_writer_worker,
            args=(i,),
            name=f"pmc-s3-write-{i}",
            daemon=True,
        )
        for i in range(actual_write_workers)
    ]

    for t in embed_threads:
        t.start()
    for t in write_threads:
        t.start()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_metadata_entry,
                downloader_mod,
                metadata_key,
                etag,
                datasets,
                processed_ids,
                skipped_ids,
                inflight_ids,
                checkpoint_lock,
                xml_fetch_timeout_seconds,
                xml_fetch_retries,
                xml_fetch_backoff_factor,
                failed_metadata_keys_file,
                failed_metadata_keys_lock,
                skip_checkpoint_file,
                skip_checkpoint_lock,
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
            should_skip = False
            with checkpoint_lock:
                if checkpoint_id in processed_ids:
                    should_skip = True
            if should_skip:
                _release_checkpoint_keys(inflight_ids, checkpoint_lock, [checkpoint_id])
                counters["skipped"] += 1
                continue
            article_queue.put((article, checkpoint_id))

    # Signal all embed workers to stop via single None (they relay it)
    article_queue.put(None)
    for t in embed_threads:
        t.join()
    for t in write_threads:
        t.join()

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
    write_workers: int = 1,
    xml_fetch_timeout_seconds: int = DEFAULT_XML_FETCH_TIMEOUT_SECONDS,
    xml_fetch_retries: int = DEFAULT_XML_FETCH_RETRIES,
    xml_fetch_backoff_factor: float = DEFAULT_XML_FETCH_BACKOFF_FACTOR,
    article_queue_wait_timeout_seconds: float = DEFAULT_ARTICLE_QUEUE_WAIT_TIMEOUT_SECONDS,
    points_queue_wait_timeout_seconds: float = DEFAULT_POINTS_QUEUE_WAIT_TIMEOUT_SECONDS,
    failed_metadata_keys_file: Optional[Path] = None,
    namespace_shard_count: int = DEFAULT_NAMESPACE_SHARD_COUNT,
    namespace_shard_pattern: str = DEFAULT_NAMESPACE_SHARD_PATTERN,
) -> None:
    ensure_data_dirs()
    reset_pmc_xml_parse_failure_count()

    scripts_dir = Path(__file__).resolve().parent
    downloader_mod = _load_script_module("download_pmc_unified_mod", scripts_dir / "01_download_pmc_unified.py")
    pmc_ingest_mod = _load_script_module("ingest_pmc_mod", scripts_dir / "06_ingest_pmc.py")

    base_namespace = IngestionConfig.TURBOPUFFER_NAMESPACE_PMC
    shard_namespaces = _compute_namespace_shards(
        base_namespace=base_namespace,
        shard_count=namespace_shard_count,
        pattern=namespace_shard_pattern,
    )
    sink = build_ingestion_sink(namespace_override=base_namespace)
    shard_sinks: Optional[list[BaseIngestionSink]] = None
    if len(shard_namespaces) > 1:
        shard_sinks = [build_ingestion_sink(namespace_override=ns) for ns in shard_namespaces]
        logger.info(
            "PMC S3 namespace sharding enabled: count=%s pattern=%s namespaces=%s",
            len(shard_namespaces),
            namespace_shard_pattern,
            ",".join(shard_namespaces),
        )
    embedding_provider = EmbeddingProvider()

    processed_ids = load_checkpoint(checkpoint_file)
    skip_checkpoint_file = _derive_skipped_checkpoint_file(checkpoint_file)
    skipped_ids = _load_skipped_checkpoint(skip_checkpoint_file)
    processed_ids.update(skipped_ids)
    inflight_ids: set[str] = set()
    checkpoint_lock = threading.Lock()
    skip_checkpoint_lock = threading.Lock()
    failed_metadata_keys_lock = threading.Lock()
    logger.info(
        "PMC S3 checkpoint loaded: %s processed IDs, %s skipped IDs",
        len(processed_ids) - len(skipped_ids),
        len(skipped_ids),
    )

    state = downloader_mod._load_state(state_file)
    dataset_sig = downloader_mod._dataset_signature(datasets)
    cutoff = downloader_mod._select_cutoff_for_incremental(state, dataset_sig, release_mode)

    completed_metadata_keys = set()
    for cp_key in processed_ids | skipped_ids:
        parts = cp_key.split(":")
        if len(parts) >= 2:
            completed_metadata_keys.add(parts[1])
        else:
            completed_metadata_keys.add(cp_key)

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

        if metadata_key in completed_metadata_keys:
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
            result = _process_metadata_entry(
                downloader_mod,
                metadata_key,
                etag,
                datasets,
                processed_ids,
                skipped_ids,
                inflight_ids,
                checkpoint_lock,
                xml_fetch_timeout_seconds,
                xml_fetch_retries,
                xml_fetch_backoff_factor,
                failed_metadata_keys_file,
                failed_metadata_keys_lock,
                skip_checkpoint_file,
                skip_checkpoint_lock,
            )
            if result is None:
                continue
            article, checkpoint_id = result
            should_skip = False
            with checkpoint_lock:
                if checkpoint_id in processed_ids:
                    should_skip = True
            if should_skip:
                _release_checkpoint_keys(inflight_ids, checkpoint_lock, [checkpoint_id])
                total_skipped += 1
                continue
            written = _flush_articles(
                pmc_ingest_mod=pmc_ingest_mod,
                sink=sink,
                shard_sinks=shard_sinks,
                embedding_provider=embedding_provider,
                articles=[article],
                checkpoint_ids=[checkpoint_id],
                checkpoint_file=checkpoint_file,
                processed_ids=processed_ids,
                inflight_ids=inflight_ids,
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
                shard_sinks=shard_sinks,
                embedding_provider=embedding_provider,
                datasets=datasets,
                workers=workers,
                processed_ids=processed_ids,
                skipped_ids=skipped_ids,
                inflight_ids=inflight_ids,
                checkpoint_file=checkpoint_file,
                checkpoint_lock=checkpoint_lock,
                skip_checkpoint_file=skip_checkpoint_file,
                skip_checkpoint_lock=skip_checkpoint_lock,
                article_queue_size=article_queue_size,
                points_queue_size=points_queue_size,
                embed_article_batch_size=embed_article_batch_size,
                embed_workers=embed_workers,
                write_workers=write_workers,
                xml_fetch_timeout_seconds=xml_fetch_timeout_seconds,
                xml_fetch_retries=xml_fetch_retries,
                xml_fetch_backoff_factor=xml_fetch_backoff_factor,
                article_queue_wait_timeout_seconds=article_queue_wait_timeout_seconds,
                points_queue_wait_timeout_seconds=points_queue_wait_timeout_seconds,
                failed_metadata_keys_file=failed_metadata_keys_file,
                failed_metadata_keys_lock=failed_metadata_keys_lock,
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
        "--write-workers",
        type=int,
        default=int(os.getenv("PMC_S3_WRITE_WORKERS", "1")),
        help=(
            "Number of parallel write worker threads. Each worker independently "
            "pulls embedded point batches from the queue and writes to Turbopuffer. "
            "Set higher to saturate TURBOPUFFER_MAX_CONCURRENT_WRITES slots."
        ),
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
    parser.add_argument(
        "--xml-fetch-timeout-seconds",
        type=int,
        default=DEFAULT_XML_FETCH_TIMEOUT_SECONDS,
        help="Timeout in seconds for XML downloads from PMC S3/HTTPS URLs.",
    )
    parser.add_argument(
        "--xml-fetch-retries",
        type=int,
        default=DEFAULT_XML_FETCH_RETRIES,
        help="Retry attempts for transient XML download failures.",
    )
    parser.add_argument(
        "--xml-fetch-backoff-factor",
        type=float,
        default=DEFAULT_XML_FETCH_BACKOFF_FACTOR,
        help="Exponential backoff factor for transient XML download retries.",
    )
    parser.add_argument(
        "--article-queue-wait-timeout-seconds",
        type=float,
        default=DEFAULT_ARTICLE_QUEUE_WAIT_TIMEOUT_SECONDS,
        help="How long embed workers wait for parsed articles before stopping.",
    )
    parser.add_argument(
        "--points-queue-wait-timeout-seconds",
        type=float,
        default=DEFAULT_POINTS_QUEUE_WAIT_TIMEOUT_SECONDS,
        help="How long the write worker waits for embedded points before stopping.",
    )
    parser.add_argument(
        "--failed-metadata-keys-file",
        type=Path,
        default=Path(os.getenv(FAILED_METADATA_KEYS_ENV))
        if os.getenv(FAILED_METADATA_KEYS_ENV)
        else None,
        help="Optional append-only TSV file for skipped/failed metadata keys and reasons.",
    )
    parser.add_argument(
        "--namespace-shard-count",
        type=int,
        default=DEFAULT_NAMESPACE_SHARD_COUNT,
        help="Number of turbopuffer namespaces to shard PMC S3 writes across (default: 1).",
    )
    parser.add_argument(
        "--namespace-shard-pattern",
        type=str,
        default=DEFAULT_NAMESPACE_SHARD_PATTERN,
        help=(
            "Namespace naming pattern for PMC S3 sharding. "
            "Supports {base} and {shard}; example: {base}_shard_{shard}."
        ),
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
        write_workers=args.write_workers,
        xml_fetch_timeout_seconds=args.xml_fetch_timeout_seconds,
        xml_fetch_retries=args.xml_fetch_retries,
        xml_fetch_backoff_factor=args.xml_fetch_backoff_factor,
        article_queue_wait_timeout_seconds=args.article_queue_wait_timeout_seconds,
        points_queue_wait_timeout_seconds=args.points_queue_wait_timeout_seconds,
        failed_metadata_keys_file=args.failed_metadata_keys_file,
        namespace_shard_count=args.namespace_shard_count,
        namespace_shard_pattern=args.namespace_shard_pattern,
    )


if __name__ == "__main__":
    main()
