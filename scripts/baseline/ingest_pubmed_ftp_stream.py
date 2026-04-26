#!/usr/bin/env python3
"""
Direct PubMed baseline ingestion from NCBI FTP stream into turbopuffer.

No local baseline XML download/staging is used. Each remote .xml.gz file is
streamed, parsed incrementally, filtered, embedded, and ingested.
"""

from __future__ import annotations

import argparse
import ftplib
import gzip
import hashlib
import importlib.util
import io
import logging
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import EmbeddingProvider
from turbopuffer_ingestion_sink import build_ingestion_sink

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


FTP_HOST = "ftp.ncbi.nlm.nih.gov"
FTP_BASELINE_DIR = "pubmed/baseline"
DEFAULT_MIN_YEAR = 2023
FILES_CHECKPOINT = IngestionConfig.DATA_DIR / "pubmed_baseline_ftp_processed_files.txt"
FAILED_FILES_CHECKPOINT = IngestionConfig.DATA_DIR / "pubmed_baseline_ftp_failed_files.txt"
MD5_PATTERN = re.compile(r"\b([0-9a-fA-F]{32})\b")
FTP_TIMEOUT_SECONDS = 120


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _load_processed_files(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _append_processed_file(path: Path, file_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(file_name.strip() + "\n")


def _load_failed_files(path: Path) -> set[str]:
    if not path.exists():
        return set()
    failed_files: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            failed_files.add(value.split("\t", 1)[0].strip())
    return failed_files


def _append_failed_file(path: Path, file_name: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_reason = " ".join(str(reason).split())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{file_name.strip()}\t{cleaned_reason}\n")


def _list_ftp_baseline_files() -> list[str]:
    files: list[str] = []
    ftp = ftplib.FTP(FTP_HOST, timeout=FTP_TIMEOUT_SECONDS)
    ftp.login()
    ftp.cwd(FTP_BASELINE_DIR)
    for name in ftp.nlst():
        if name.endswith(".xml.gz") and name.startswith("pubmed"):
            files.append(name)
    ftp.quit()
    files.sort()
    return files


def _iter_articles_from_ftp_gzip(
    file_name: str,
    min_year: int,
    extract_article_data: Any,
    include_all_articles: bool,
):
    compressed_bytes = _download_verified_gzip_bytes(file_name)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed_bytes)) as gz:
        context = ET.iterparse(gz, events=("end",))
        for _event, elem in context:
            if elem.tag != "PubmedArticle":
                continue
            article = extract_article_data(
                elem,
                min_year,
                apply_publication_type_filter=not include_all_articles,
                apply_abstract_text_filter=not include_all_articles,
            )
            if article:
                yield article
            elem.clear()


def _read_ftp_bytes(remote_name: str, timeout: int = FTP_TIMEOUT_SECONDS) -> bytes:
    ftp = ftplib.FTP(FTP_HOST, timeout=timeout)
    payload = bytearray()
    try:
        ftp.login()
        ftp.cwd(FTP_BASELINE_DIR)
        ftp.retrbinary(f"RETR {remote_name}", payload.extend, blocksize=1024 * 1024)
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()
    return bytes(payload)


def _get_expected_md5(file_name: str) -> str:
    md5_text = _read_ftp_bytes(f"{file_name}.md5", timeout=FTP_TIMEOUT_SECONDS).decode(
        "utf-8", errors="replace"
    )
    match = MD5_PATTERN.search(md5_text)
    if match is None:
        raise ValueError(f"Could not parse MD5 checksum for {file_name}: {md5_text.strip()!r}")
    return match.group(1).lower()


def _download_verified_gzip_bytes(file_name: str) -> bytes:
    expected_md5 = _get_expected_md5(file_name)
    payload = _read_ftp_bytes(file_name, timeout=FTP_TIMEOUT_SECONDS)
    md5 = hashlib.md5()
    md5.update(payload)
    actual_md5 = md5.hexdigest().lower()
    if actual_md5 != expected_md5:
        raise ValueError(
            f"MD5 mismatch for {file_name}: expected={expected_md5} actual={actual_md5}"
        )
    return bytes(payload)


def _run_file_ingestion(
    file_name: str,
    min_year: int,
    batch_size: int,
    max_workers: int,
    extract_article_data: Any,
    ingest_mod: ModuleType,
    embedding_provider: EmbeddingProvider,
    sink: Any,
    processed_ids: set[str],
    processed_lock: threading.Lock,
    include_all_articles: bool,
) -> int:
    total_written = 0
    current_batch: list[dict[str, Any]] = []
    futures: list[Any] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for article in _iter_articles_from_ftp_gzip(
            file_name,
            min_year,
            extract_article_data,
            include_all_articles,
        ):
            pmid = str(article.get("pmid") or "").strip()
            if not pmid:
                continue
            checkpoint_id = ingest_mod._checkpoint_id(pmid)
            with processed_lock:
                if checkpoint_id in processed_ids:
                    continue

            current_batch.append(article)
            if len(current_batch) < batch_size:
                continue

            futures.append(
                executor.submit(
                    ingest_mod.process_batch,
                    None,
                    list(current_batch),
                    embedding_provider,
                    validate_chunks=bool(getattr(ingest_mod, "ENHANCED_UTILS_AVAILABLE", False)),
                    dedup_chunks=bool(getattr(ingest_mod, "ENHANCED_UTILS_AVAILABLE", False)),
                    processed_ids=processed_ids,
                    processed_lock=processed_lock,
                    sink=sink,
                )
            )
            current_batch.clear()

        if current_batch:
            futures.append(
                executor.submit(
                    ingest_mod.process_batch,
                    None,
                    list(current_batch),
                    embedding_provider,
                    validate_chunks=bool(getattr(ingest_mod, "ENHANCED_UTILS_AVAILABLE", False)),
                    dedup_chunks=bool(getattr(ingest_mod, "ENHANCED_UTILS_AVAILABLE", False)),
                    processed_ids=processed_ids,
                    processed_lock=processed_lock,
                    sink=sink,
                )
            )

        for future in as_completed(futures):
            total_written += int(future.result())

    return total_written


def _run_file_ingestion_with_retry(
    *,
    file_name: str,
    min_year: int,
    batch_size: int,
    max_workers: int,
    extract_article_data: Any,
    ingest_mod: ModuleType,
    embedding_provider: EmbeddingProvider,
    sink: Any,
    processed_ids: set[str],
    processed_lock: threading.Lock,
    max_attempts: int,
    retry_backoff_seconds: float,
    include_all_articles: bool,
) -> int:
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            return _run_file_ingestion(
                file_name=file_name,
                min_year=min_year,
                batch_size=batch_size,
                max_workers=max_workers,
                extract_article_data=extract_article_data,
                ingest_mod=ingest_mod,
                embedding_provider=embedding_provider,
                sink=sink,
                processed_ids=processed_ids,
                processed_lock=processed_lock,
                include_all_articles=include_all_articles,
            )
        except Exception as exc:
            if attempt >= max(1, max_attempts):
                logger.error(
                    "File ingestion failed after %d/%d attempts: file=%s error=%s",
                    attempt,
                    max_attempts,
                    file_name,
                    exc,
                )
                raise
            wait_seconds = max(0.0, retry_backoff_seconds) * attempt
            logger.warning(
                "File ingestion failed (attempt %d/%d), retrying file=%s in %.1fs: %s",
                attempt,
                max_attempts,
                file_name,
                wait_seconds,
                exc,
            )
            if wait_seconds > 0:
                time.sleep(wait_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct PubMed baseline FTP stream ingestion")
    parser.add_argument("--namespace", default=IngestionConfig.TURBOPUFFER_NAMESPACE_PUBMED)
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR)
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for files to process")
    parser.add_argument(
        "--oldest-first",
        action="store_true",
        help="Process baseline files oldest-to-newest (default is newest-first for faster recent-yield ingestion)",
    )
    parser.add_argument("--files-checkpoint", type=Path, default=FILES_CHECKPOINT)
    parser.add_argument("--failed-files-checkpoint", type=Path, default=FAILED_FILES_CHECKPOINT)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=max(1, int(IngestionConfig.BATCH_SIZE)),
        help="Article batch size for ingestion worker submit",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(IngestionConfig.MAX_WORKERS)),
        help="Ingestion worker budget (distributed across file workers when --file-workers > 1)",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=1,
        help="Number of FTP files to process concurrently",
    )
    parser.add_argument(
        "--file-retries",
        type=int,
        default=3,
        help="Attempts per FTP file before failing the overall run",
    )
    parser.add_argument(
        "--file-retry-backoff",
        type=float,
        default=10.0,
        help="Base backoff in seconds between whole-file retries",
    )
    parser.add_argument(
        "--include-all-articles",
        action="store_true",
        default=True,
        help="Disable publication-type and abstract-length filtering and ingest all PubMed articles from the selected year onward",
    )
    parser.add_argument(
        "--skip-failed-files",
        action="store_true",
        default=True,
        help="Record permanently failed FTP files and continue with the remaining queue",
    )
    args = parser.parse_args()

    ensure_data_dirs()
    if not IngestionConfig.TURBOPUFFER_API_KEY:
        logger.error("TURBOPUFFER_API_KEY is required")
        return 2
    if not (IngestionConfig.EMBEDDING_PROVIDER or "").strip():
        logger.error("EMBEDDING_PROVIDER is required")
        return 2

    scripts_dir = Path(__file__).parent
    baseline_mod = _load_module(scripts_dir / "20_download_pubmed_baseline.py", "pubmed_baseline_download_mod")
    ingest_mod = _load_module(scripts_dir / "21_ingest_pubmed_abstracts.py", "pubmed_ingest_mod")

    extract_article_data = getattr(baseline_mod, "extract_article_data", None)
    if extract_article_data is None:
        logger.error("Could not load extract_article_data from 20_download_pubmed_baseline.py")
        return 2

    processed_files = _load_processed_files(args.files_checkpoint)
    failed_files = _load_failed_files(args.failed_files_checkpoint)
    all_files = _list_ftp_baseline_files()
    pending_files = [f for f in all_files if f not in processed_files and f not in failed_files]
    if not args.oldest_first:
        pending_files.sort(reverse=True)
    if args.max_files is not None and args.max_files > 0:
        pending_files = pending_files[: args.max_files]

    file_workers = max(1, int(args.file_workers))
    worker_budget = max(1, int(args.workers))
    workers_per_file = max(1, worker_budget // file_workers) if file_workers > 1 else worker_budget

    logger.info(
        "PubMed FTP stream ingestion starting: namespace=%s files_total=%d already_processed=%d pending=%d file_workers=%d worker_budget=%d workers_per_file=%d batch_size=%d",
        args.namespace,
        len(all_files),
        len(processed_files),
        len(pending_files),
        file_workers,
        worker_budget,
        workers_per_file,
        args.batch_size,
    )

    embedding_provider = EmbeddingProvider()
    sink = build_ingestion_sink(namespace_override=args.namespace)

    processed_ids = ingest_mod.load_checkpoint_namespaced(ingest_mod.CHECKPOINT_FILE)
    processed_lock = threading.Lock()
    logger.info("Loaded PubMed article checkpoint IDs=%d from %s", len(processed_ids), ingest_mod.CHECKPOINT_FILE)

    total_written = 0
    files_done = 0
    files_failed = 0
    progress_lock = threading.Lock()
    files_checkpoint_lock = threading.Lock()
    start = time.time()
    if file_workers == 1:
        for idx, file_name in enumerate(pending_files, start=1):
            logger.info("Processing FTP file %d/%d: %s", idx, len(pending_files), file_name)
            try:
                file_written = _run_file_ingestion_with_retry(
                    file_name=file_name,
                    min_year=args.min_year,
                    batch_size=max(1, args.batch_size),
                    max_workers=workers_per_file,
                    extract_article_data=extract_article_data,
                    ingest_mod=ingest_mod,
                    embedding_provider=embedding_provider,
                    sink=sink,
                    processed_ids=processed_ids,
                    processed_lock=processed_lock,
                    max_attempts=max(1, args.file_retries),
                    retry_backoff_seconds=max(0.0, args.file_retry_backoff),
                    include_all_articles=bool(args.include_all_articles),
                )
                total_written += file_written
                files_done += 1
                _append_processed_file(args.files_checkpoint, file_name)
                elapsed = max(0.001, time.time() - start)
                logger.info(
                    "File complete: %s written=%d total_written=%d files_done=%d elapsed=%.1fs rate=%.2f points/s",
                    file_name,
                    file_written,
                    total_written,
                    files_done,
                    elapsed,
                    total_written / elapsed,
                )
            except Exception as exc:
                if not args.skip_failed_files:
                    logger.error("Failed processing file=%s error=%s", file_name, exc)
                    return 1
                files_failed += 1
                _append_failed_file(args.failed_files_checkpoint, file_name, str(exc))
                logger.error("Skipping failed file=%s error=%s", file_name, exc)
    else:
        def _file_task(file_name: str) -> tuple[str, int]:
            written = _run_file_ingestion_with_retry(
                file_name=file_name,
                min_year=args.min_year,
                batch_size=max(1, args.batch_size),
                max_workers=workers_per_file,
                extract_article_data=extract_article_data,
                ingest_mod=ingest_mod,
                embedding_provider=embedding_provider,
                sink=sink,
                processed_ids=processed_ids,
                processed_lock=processed_lock,
                max_attempts=max(1, args.file_retries),
                retry_backoff_seconds=max(0.0, args.file_retry_backoff),
                include_all_articles=bool(args.include_all_articles),
            )
            return file_name, written

        logger.info(
            "Starting parallel FTP file processing: file_workers=%d workers_per_file=%d",
            file_workers,
            workers_per_file,
        )
        try:
            with ThreadPoolExecutor(max_workers=file_workers) as file_executor:
                future_to_file = {
                    file_executor.submit(_file_task, file_name): file_name for file_name in pending_files
                }
                for future in as_completed(future_to_file):
                    file_name = future_to_file[future]
                    try:
                        done_file, file_written = future.result()
                    except Exception as exc:
                        if not args.skip_failed_files:
                            logger.error("Failed processing file=%s error=%s", file_name, exc)
                            return 1
                        with progress_lock:
                            files_failed += 1
                        with files_checkpoint_lock:
                            _append_failed_file(args.failed_files_checkpoint, file_name, str(exc))
                        logger.error("Skipping failed file=%s error=%s", file_name, exc)
                        continue
                    with progress_lock:
                        total_written += int(file_written)
                        files_done += 1
                        elapsed = max(0.001, time.time() - start)
                        rate = total_written / elapsed
                    with files_checkpoint_lock:
                        _append_processed_file(args.files_checkpoint, done_file)
                    logger.info(
                        "File complete: %s written=%d total_written=%d files_done=%d/%d elapsed=%.1fs rate=%.2f points/s",
                        done_file,
                        file_written,
                        total_written,
                        files_done,
                        len(pending_files),
                        elapsed,
                        rate,
                    )
        except Exception as exc:
            logger.error("Parallel file processing failed: %s", exc)
            return 1

    elapsed = max(0.001, time.time() - start)
    logger.info(
        "PubMed FTP stream ingestion complete: namespace=%s files_done=%d files_failed=%d total_written=%d elapsed=%.1fs rate=%.2f points/s",
        args.namespace,
        files_done,
        files_failed,
        total_written,
        elapsed,
        total_written / elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
