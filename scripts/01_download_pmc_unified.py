#!/usr/bin/env python3
"""Download PMC OA and Author Manuscript XML files from PMC Cloud Service (AWS S3)."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Literal, Tuple
from urllib.parse import quote, urlparse
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config_ingestion import IngestionConfig, ensure_data_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Thread-local storage for HTTP sessions with connection pooling
_thread_local = threading.local()


def _get_session(pool_size: int = 64) -> requests.Session:
    """Get or create a thread-local requests.Session with connection pooling."""
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=retry,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _thread_local.session = session
    return _thread_local.session
logger = logging.getLogger(__name__)

S3_BUCKET = "pmc-oa-opendata"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.amazonaws.com"
INVENTORY_ROOT_PREFIX = f"inventory-reports/{S3_BUCKET}/metadata/"
DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}Z/$")

REMOTE_DIRS = {
    "pmc_oa": "pmc_oa",
    "author_manuscript": "author_manuscript",
}

# Kept for backward compatibility with existing CLI usage.
RELEASE_MODES = ("all", "baseline", "incremental")
ReleaseMode = Literal["all", "baseline", "incremental"]

STATE_FILE_NAME = ".pmc_s3_inventory_state.json"
MARKER_DIR_NAME = ".pmc_s3_markers"


def _s3_key_to_https(key: str) -> str:
    return f"{S3_BASE_URL}/{quote(key, safe='/')}"


def _normalize_s3_or_https_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        query = f"?{parsed.query}" if parsed.query else ""
        return f"https://{bucket}.s3.amazonaws.com/{quote(key, safe='/')}{query}"
    if parsed.scheme in {"http", "https"}:
        return value
    raise ValueError(f"Unsupported URL scheme in xml_url: {value}")


def _download_file_http(url: str, local_path: Path, chunk_size: int = 1024 * 1024, session: requests.Session | None = None) -> bool:
    temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    _session = session or _get_session()
    try:
        logger.debug("Downloading: %s", url)
        with _session.get(url, stream=True, timeout=120) as response:
            if response.status_code != 200:
                logger.warning("HTTP missing (status %s): %s", response.status_code, url)
                return False

            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)

        temp_path.replace(local_path)
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("HTTP download failed for %s: %s", url, exc)
        return False
    finally:
        temp_path.unlink(missing_ok=True)


def _list_common_prefixes(prefix: str) -> list[str]:
    prefixes: list[str] = []
    continuation_token: str | None = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    while True:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "delimiter": "/",
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token

        response = requests.get(f"{S3_BASE_URL}/", params=params, timeout=120)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        for entry in root.findall("s3:CommonPrefixes/s3:Prefix", ns):
            if entry.text:
                prefixes.append(entry.text)

        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=ns).lower() == "true"
        continuation_token = root.findtext("s3:NextContinuationToken", default=None, namespaces=ns)
        if not is_truncated:
            break

    return prefixes


def _latest_inventory_prefix() -> str:
    common_prefixes = _list_common_prefixes(INVENTORY_ROOT_PREFIX)
    version_dirs: list[str] = []

    for full_prefix in common_prefixes:
        suffix = full_prefix[len(INVENTORY_ROOT_PREFIX) :]
        if DATE_PREFIX_RE.match(suffix):
            version_dirs.append(full_prefix)

    if not version_dirs:
        raise RuntimeError("No dated inventory versions found under inventory-reports/")

    return sorted(version_dirs)[-1]


def _inventory_csv_keys_for_latest_version() -> list[str]:
    latest_prefix = _latest_inventory_prefix()
    manifest_key = f"{latest_prefix}manifest.json"
    manifest_url = _s3_key_to_https(manifest_key)

    response = requests.get(manifest_url, timeout=120)
    response.raise_for_status()
    manifest = response.json()

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"Inventory manifest has no files: {manifest_url}")

    csv_keys = [entry["key"] for entry in files if isinstance(entry, dict) and entry.get("key")]
    if not csv_keys:
        raise RuntimeError(f"Inventory manifest files list is empty: {manifest_url}")

    logger.info("Using inventory version prefix: %s", latest_prefix)
    logger.info("Inventory CSV partitions: %s", len(csv_keys))
    return csv_keys


def _iter_inventory_entries(csv_keys: list[str]) -> Iterable[Tuple[str, str, str]]:
    """Yield (metadata_key, last_modified_utc, etag) rows from inventory CSV files."""
    for csv_key in csv_keys:
        csv_url = _s3_key_to_https(csv_key)
        logger.info("Reading inventory CSV: %s", csv_key)

        with requests.get(csv_url, stream=True, timeout=180) as response:
            response.raise_for_status()
            with gzip.open(response.raw, mode="rt", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if len(row) < 4:
                        continue
                    _bucket, key, last_modified, etag = row[0], row[1], row[2], row[3]
                    if not key.startswith("metadata/") or not key.endswith(".json"):
                        continue
                    yield key, last_modified, etag.strip('"')


def _iter_metadata_entries_via_list_api() -> Iterable[Tuple[str, str, str]]:
    """Yield (metadata_key, last_modified_utc, etag) by listing metadata/ directly."""
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    continuation_token: str | None = None
    page = 0

    while True:
        page += 1
        params = {
            "list-type": "2",
            "prefix": "metadata/",
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token

        response = requests.get(f"{S3_BASE_URL}/", params=params, timeout=180)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        yielded = 0
        for content in root.findall("s3:Contents", ns):
            key = content.findtext("s3:Key", default="", namespaces=ns)
            if not key.startswith("metadata/") or not key.endswith(".json"):
                continue
            last_modified = content.findtext("s3:LastModified", default="", namespaces=ns)
            etag = content.findtext("s3:ETag", default="", namespaces=ns).strip('"')
            if not last_modified:
                continue
            yielded += 1
            yield key, last_modified, etag

        logger.info("Listed metadata page %s (yielded=%s)", page, yielded)

        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=ns).lower() == "true"
        continuation_token = root.findtext("s3:NextContinuationToken", default=None, namespaces=ns)
        if not is_truncated:
            break


def _iter_metadata_entries() -> Iterable[Tuple[str, str, str]]:
    """Prefer inventory CSV when available, else fall back to listing metadata/."""
    try:
        csv_keys = _inventory_csv_keys_for_latest_version()
    except Exception as exc:
        logger.warning(
            "Inventory listing unavailable (%s). Falling back to ListObjectsV2 on metadata/.",
            exc,
        )
        yield from _iter_metadata_entries_via_list_api()
        return

    yield from _iter_inventory_entries(csv_keys)


def _metadata_matches_dataset(metadata: dict, dataset_key: str) -> bool:
    if dataset_key == "pmc_oa":
        return bool(metadata.get("is_pmc_openaccess"))
    if dataset_key == "author_manuscript":
        return bool(metadata.get("is_manuscript"))
    return False


def _dataset_signature(datasets: List[str]) -> str:
    return ",".join(sorted(datasets))


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"last_modified_by_signature": {}}

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"last_modified_by_signature": {}}
        if not isinstance(data.get("last_modified_by_signature"), dict):
            data["last_modified_by_signature"] = {}
        return data
    except Exception:
        return {"last_modified_by_signature": {}}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _marker_path(marker_root: Path, dataset_sig: str, metadata_key: str, etag: str) -> Path:
    safe_key = metadata_key.replace("/", "__")
    safe_sig = dataset_sig.replace(",", "__")
    safe_etag = etag.replace("/", "_")
    return marker_root / safe_sig / f".{safe_key}.{safe_etag}.done"


def _download_metadata_json(metadata_key: str, session: requests.Session | None = None) -> dict | None:
    metadata_url = _s3_key_to_https(metadata_key)
    _session = session or _get_session()
    try:
        response = _session.get(metadata_url, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Failed to read metadata %s: %s", metadata_key, exc)
    return None


def _process_metadata_entry(
    output_dir: Path,
    datasets: List[str],
    dataset_sig: str,
    metadata_key: str,
    etag: str,
    session: requests.Session | None = None,
) -> tuple[int, int]:
    marker_root = output_dir / MARKER_DIR_NAME
    marker_path = _marker_path(marker_root, dataset_sig, metadata_key, etag)

    if marker_path.exists():
        return 0, 0

    _session = session or _get_session()
    metadata = _download_metadata_json(metadata_key, session=_session)
    if metadata is None:
        return 0, 0

    target_datasets = [dataset for dataset in datasets if _metadata_matches_dataset(metadata, dataset)]
    if not target_datasets:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()
        return 0, 0

    xml_url = metadata.get("xml_url")
    if not isinstance(xml_url, str) or not xml_url:
        logger.debug("Skipping metadata without xml_url: %s", metadata_key)
        return 0, 0

    try:
        xml_http_url = _normalize_s3_or_https_url(xml_url)
    except Exception as exc:
        logger.warning("Invalid xml_url in %s: %s", metadata_key, exc)
        return 0, 0

    xml_name = Path(urlparse(xml_http_url).path).name
    if not xml_name.endswith((".xml", ".nxml")):
        logger.debug("Skipping non-XML target for %s: %s", metadata_key, xml_name)
        return 0, 0

    downloaded_now = 0

    for dataset in target_datasets:
        local_path = output_dir / dataset / xml_name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and local_path.stat().st_size > 0:
            continue

        if _download_file_http(xml_http_url, local_path, session=_session):
            downloaded_now += 1
        else:
            logger.error("Failed XML download for %s (%s)", metadata_key, dataset)
            return downloaded_now, downloaded_now

    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.touch()
    return downloaded_now, downloaded_now


def _select_cutoff_for_incremental(state: dict, dataset_sig: str, release_mode: ReleaseMode) -> str | None:
    if release_mode != "incremental":
        if release_mode == "baseline":
            logger.warning("--release-mode baseline is not available on PMC Cloud Service; treating as full scan.")
        return None

    cutoff = state.get("last_modified_by_signature", {}).get(dataset_sig)
    if cutoff:
        logger.info("Incremental mode enabled with cutoff last_modified > %s", cutoff)
    else:
        logger.info("Incremental mode enabled, but no prior state exists; processing all inventory entries.")
    return cutoff


def _should_include_entry(last_modified: str, cutoff: str | None) -> bool:
    if cutoff is None:
        return True
    return last_modified > cutoff


def download_pmc(
    output_dir: Path,
    datasets: List[str] | None = None,
    release_mode: ReleaseMode = "all",
    max_files: int | None = None,
    workers: int = 64,
) -> int:
    """Download requested datasets from PMC Cloud Service and return total files downloaded.

    Uses a thread pool to download multiple files concurrently for much faster throughput.
    Each thread gets its own HTTP session with connection pooling for maximum performance.
    """
    selected = datasets or list(REMOTE_DIRS.keys())
    invalid = [value for value in selected if value not in REMOTE_DIRS]
    if invalid:
        raise ValueError(f"Unknown dataset keys: {', '.join(invalid)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in selected:
        dataset_output = output_dir / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        (dataset_output / ".source").write_text(dataset, encoding="utf-8")

    dataset_sig = _dataset_signature(selected)
    state_path = output_dir / STATE_FILE_NAME
    state = _load_state(state_path)
    cutoff = _select_cutoff_for_incremental(state, dataset_sig, release_mode)

    if max_files is not None and max_files < 0:
        raise ValueError("--max-files must be >= 0")

    if max_files == 0:
        logger.info("--max-files is 0; nothing to process.")
        return 0

    # Thread-safe counters
    lock = threading.Lock()
    counters = {
        "total_downloaded": 0,
        "total_extracted": 0,
        "processed_entries": 0,
        "skipped_marker": 0,
    }
    start_time = time.monotonic()
    last_log_time = [start_time]  # mutable for closure

    def _process_entry_wrapper(entry_tuple):
        """Wrapper for thread pool execution."""
        metadata_key, etag = entry_tuple
        session = _get_session(pool_size=workers)
        downloaded_now, extracted_now = _process_metadata_entry(
            output_dir=output_dir,
            datasets=selected,
            dataset_sig=dataset_sig,
            metadata_key=metadata_key,
            etag=etag,
            session=session,
        )
        with lock:
            counters["total_downloaded"] += downloaded_now
            counters["total_extracted"] += extracted_now
            counters["processed_entries"] += 1
            if downloaded_now == 0 and extracted_now == 0:
                counters["skipped_marker"] += 1
            now = time.monotonic()
            # Log progress every 30 seconds instead of every N entries
            if now - last_log_time[0] >= 30:
                elapsed = now - start_time
                rate = counters["total_downloaded"] / elapsed if elapsed > 0 else 0
                logger.info(
                    "Progress: processed=%s downloaded=%s skipped=%s rate=%.1f files/sec elapsed=%.0fs (workers=%s)",
                    counters["processed_entries"],
                    counters["total_downloaded"],
                    counters["skipped_marker"],
                    rate,
                    elapsed,
                    workers,
                )
                last_log_time[0] = now
        return downloaded_now

    # Phase 1: Collect eligible entries from inventory (this is fast, just scanning metadata)
    logger.info("Phase 1: Scanning inventory for eligible metadata entries...")
    eligible_entries = []
    scanned_entries = 0
    included_entries = 0
    max_seen_last_modified: str | None = None
    stopped_by_max_files = False

    for metadata_key, last_modified, etag in _iter_metadata_entries():
        scanned_entries += 1
        if max_seen_last_modified is None or last_modified > max_seen_last_modified:
            max_seen_last_modified = last_modified

        if not _should_include_entry(last_modified, cutoff):
            continue

        included_entries += 1

        if max_files is not None and len(eligible_entries) >= max_files:
            logger.info("Reached --max-files=%s; stopping scan early.", max_files)
            stopped_by_max_files = True
            break

        eligible_entries.append((metadata_key, etag))

        if scanned_entries % 100_000 == 0:
            logger.info(
                "Scanning inventory: scanned=%s included=%s",
                scanned_entries,
                included_entries,
            )

    logger.info(
        "Phase 1 complete: scanned=%s included=%s eligible=%s",
        scanned_entries,
        included_entries,
        len(eligible_entries),
    )

    # Phase 2: Download in parallel
    logger.info("Phase 2: Downloading with %s parallel workers (connection pooling enabled)...", workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_entry_wrapper, entry): entry
            for entry in eligible_entries
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                entry = futures[future]
                logger.warning("Worker error for %s: %s", entry[0], exc)

    elapsed_total = time.monotonic() - start_time
    rate = counters["total_downloaded"] / elapsed_total if elapsed_total > 0 else 0

    if max_seen_last_modified is not None and release_mode == "incremental" and not stopped_by_max_files:
        state.setdefault("last_modified_by_signature", {})[dataset_sig] = max_seen_last_modified
        _save_state(state_path, state)
        logger.info("Updated incremental state for %s to %s", dataset_sig, max_seen_last_modified)
    elif release_mode == "incremental" and stopped_by_max_files:
        logger.info("Skipped incremental state update because run ended early due to --max-files.")

    logger.info("=" * 70)
    logger.info("PMC Cloud download complete in %.1f min (%.1f files/sec)", elapsed_total / 60, rate)
    logger.info("Scanned metadata rows: %s", scanned_entries)
    logger.info("Rows included by release mode/cutoff: %s", included_entries)
    logger.info("Metadata rows processed: %s", counters["processed_entries"])
    logger.info("Downloaded XML files: %s (skipped: %s)", counters["total_downloaded"], counters["skipped_marker"])
    return counters["total_downloaded"]


def _parse_datasets(value: str) -> List[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    if not datasets:
        raise argparse.ArgumentTypeError("--datasets must include at least one dataset key")
    invalid = [item for item in datasets if item not in REMOTE_DIRS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown dataset(s): {', '.join(invalid)}. Allowed: {', '.join(REMOTE_DIRS.keys())}"
        )
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PMC OA + Author Manuscript XML files from PMC Cloud Service")
    parser.add_argument("--output-dir", type=Path, default=IngestionConfig.PMC_XML_DIR)
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=list(REMOTE_DIRS.keys()),
        help="Comma-separated dataset keys: pmc_oa,author_manuscript",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Stop after processing N eligible metadata rows")
    parser.add_argument(
        "--release-mode",
        choices=RELEASE_MODES,
        default="all",
        help=(
            "all=full scan, incremental=only inventory rows newer than previous run, "
            "baseline=alias of all (not available in PMC Cloud layout)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of parallel download workers (default: 64)",
    )
    args = parser.parse_args()

    ensure_data_dirs()
    download_pmc(
        output_dir=args.output_dir,
        datasets=args.datasets,
        release_mode=args.release_mode,
        max_files=args.max_files,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
