#!/usr/bin/env python3
"""Stream DailyMed daily update ZIPs directly into TurboPuffer."""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import tempfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import requests
import turbopuffer as tpuf

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config_ingestion import IngestionConfig
from ingestion_utils import EmbeddingProvider, append_checkpoint as append_checkpoint_file, get_chunker
from turbopuffer_ingestion_sink import build_ingestion_sink
import dailymed_ingest_lib as dailymed_ingest

logger = logging.getLogger(__name__)

DAILYMED_BASE_URL = "https://dailymed-data.nlm.nih.gov/public-release-files"


class SeekableHTTPFile(io.IOBase):
    """Seekable file facade backed by HTTP Range requests."""

    def __init__(self, url: str, timeout: int = 300):
        self.url = url
        self.timeout = timeout
        self._pos = 0
        self._session = requests.Session()
        response = self._session.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        if not length:
            raise RuntimeError(f"DailyMed did not return Content-Length for {url}")
        self._length = int(length)

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._length + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._length - self._pos
        if size <= 0:
            return b""
        end = min(self._length - 1, self._pos + size - 1)
        response = self._session.get(
            self.url,
            headers={"Range": f"bytes={self._pos}-{end}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.content
        self._pos += len(data)
        return data


def daily_update_filename(day: date) -> str:
    return f"dm_spl_daily_update_{day:%m%d%Y}.zip"


def url_exists(url: str, timeout: int = 30) -> bool:
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        if response.status_code == 200:
            return True
        if response.status_code not in (403, 405):
            return False
    except requests.RequestException:
        pass

    try:
        response = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout)
        return response.status_code in (200, 206)
    except requests.RequestException:
        return False


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def get_namespace_last_write_date(namespace: str) -> date | None:
    client = tpuf.Turbopuffer(
        api_key=IngestionConfig.TURBOPUFFER_API_KEY,
        region=IngestionConfig.TURBOPUFFER_REGION,
        timeout=IngestionConfig.TURBOPUFFER_TIMEOUT_SECONDS,
    )
    metadata = client.namespace(namespace).metadata()
    last_write_at = getattr(metadata, "last_write_at", None)
    if last_write_at is None:
        return None
    if isinstance(last_write_at, datetime):
        return last_write_at.astimezone(timezone.utc).date()
    text = str(last_write_at).replace("Z", "+00:00")
    text = re.sub(r"(\.\d{6})\d+(\+00:00)$", r"\1\2", text)
    return datetime.fromisoformat(text).astimezone(timezone.utc).date()


def resolve_daily_update_urls(start_date: date, end_date: date) -> list[tuple[date, str]]:
    if end_date < start_date:
        return []

    urls: list[tuple[date, str]] = []
    current = start_date
    while current <= end_date:
        filename = daily_update_filename(current)
        url = f"{DAILYMED_BASE_URL}/{filename}"
        if url_exists(url):
            urls.append((current, url))
        else:
            logger.info("DailyMed daily update not published or unavailable: %s", filename)
        current += timedelta(days=1)
    return urls


def iter_zip_xml(url: str) -> Iterator[tuple[str, bytes]]:
    """Yield ``(set_id, xml_bytes)`` from a DailyMed outer ZIP URL."""
    with zipfile.ZipFile(SeekableHTTPFile(url)) as outer:
        direct_xml = [name for name in outer.namelist() if name.lower().endswith(".xml")]
        if direct_xml:
            for xml_name in direct_xml:
                set_id = Path(xml_name).stem.strip().lower()
                yield set_id, outer.read(xml_name)
            return

        nested_zips = [name for name in outer.namelist() if name.lower().endswith(".zip")]
        logger.info("Streaming %d nested ZIP members from %s", len(nested_zips), Path(url).name)
        for nested_zip_name in nested_zips:
            nested_set_id = Path(nested_zip_name).stem.strip().lower()
            with outer.open(nested_zip_name) as nested_data:
                with zipfile.ZipFile(io.BytesIO(nested_data.read())) as nested:
                    for xml_name in nested.namelist():
                        if not xml_name.lower().endswith(".xml"):
                            continue
                        set_id = Path(xml_name).stem.strip().lower() or nested_set_id
                        yield set_id, nested.read(xml_name)


def parse_xml_bytes(xml_bytes: bytes, source_name: str) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=True) as tmp:
        tmp.write(xml_bytes)
        tmp.flush()
        drug, status = dailymed_ingest._parse_spl_xml_with_status(Path(tmp.name))
    if status != "ok":
        logger.debug("Skipping %s status=%s", source_name, status)
        return None
    return drug


def run_direct_update(
    *,
    namespace: str,
    checkpoint_file: Path,
    since_date: date | None = None,
    through_date: date | None = None,
    max_labels: int | None = None,
) -> int:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    if since_date is None:
        last_write_date = get_namespace_last_write_date(namespace)
        if last_write_date is None:
            raise RuntimeError("Could not determine DailyMed namespace last_write_at; pass --since-date")
        since_date = last_write_date + timedelta(days=1)
        logger.info("Derived DailyMed update start date from namespace last_write_at: %s", since_date)

    through_date = through_date or datetime.now(timezone.utc).date()
    urls = resolve_daily_update_urls(since_date, through_date)
    logger.info(
        "DailyMed direct update resolved %d daily ZIP(s) from %s through %s",
        len(urls),
        since_date,
        through_date,
    )
    if not urls:
        return 0

    processed_ids = dailymed_ingest.load_checkpoint_namespaced(checkpoint_file)
    embedding_provider = EmbeddingProvider()
    chunker = get_chunker(
        chunker_class=dailymed_ingest.CHUNKER_CLASS,
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
    )
    sink = build_ingestion_sink(namespace_override=namespace, disable_backpressure=True)

    labels_written = 0
    rows_written = 0
    for update_day, url in urls:
        logger.info("Processing DailyMed daily update date=%s url=%s", update_day, url)
        for archive_set_id, xml_bytes in iter_zip_xml(url):
            archive_checkpoint_id = dailymed_ingest._checkpoint_id(archive_set_id)
            if archive_checkpoint_id in processed_ids:
                continue
            drug = parse_xml_bytes(xml_bytes, f"{Path(url).name}/{archive_set_id}.xml")
            if drug is None:
                continue
            set_id = str(drug.get("set_id") or archive_set_id).strip().lower()
            checkpoint_id = dailymed_ingest._checkpoint_id(set_id)
            if checkpoint_id in processed_ids:
                continue
            chunks = dailymed_ingest.create_chunks(
                drug,
                chunker,
                validate_chunks=dailymed_ingest.ENHANCED_UTILS_AVAILABLE,
            )
            points, _chunk_ids = dailymed_ingest.build_points(
                chunks,
                embedding_provider,
                validate_chunks=dailymed_ingest.ENHANCED_UTILS_AVAILABLE,
                dedup_chunks=dailymed_ingest.ENHANCED_UTILS_AVAILABLE,
            )
            if not points:
                continue
            written = sink.write_points(points)
            if written <= 0:
                continue
            append_checkpoint_file(checkpoint_file, [checkpoint_id])
            processed_ids.add(checkpoint_id)
            labels_written += 1
            rows_written += written
            logger.info("Ingested DailyMed set_id=%s rows=%d", drug.get("set_id", set_id), written)
            if max_labels is not None and labels_written >= max_labels:
                logger.info(
                    "DailyMed direct update stopped at max_labels=%d rows_written=%d",
                    max_labels,
                    rows_written,
                )
                return rows_written

    logger.info(
        "DailyMed direct update complete labels_written=%d rows_written=%d",
        labels_written,
        rows_written,
    )
    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=IngestionConfig.TURBOPUFFER_NAMESPACE_DAILYMED)
    parser.add_argument("--checkpoint", type=Path, default=IngestionConfig.DAILYMED_CHECKPOINT_FILE)
    parser.add_argument("--since-date", type=parse_date, default=None, help="First DailyMed daily update date, YYYY-MM-DD")
    parser.add_argument("--through-date", type=parse_date, default=None, help="Last DailyMed daily update date, YYYY-MM-DD")
    parser.add_argument("--max-labels", type=int, default=None, help="Stop after ingesting this many RX labels")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not IngestionConfig.TURBOPUFFER_API_KEY:
        raise RuntimeError("TURBOPUFFER_API_KEY is required")
    if max_labels := args.max_labels:
        if max_labels <= 0:
            raise ValueError("--max-labels must be greater than 0")
    run_direct_update(
        namespace=args.namespace,
        checkpoint_file=args.checkpoint,
        since_date=args.since_date,
        through_date=args.through_date,
        max_labels=args.max_labels,
    )


if __name__ == "__main__":
    main()
