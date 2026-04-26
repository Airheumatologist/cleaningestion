#!/usr/bin/env python3
"""
Direct DailyMed ingestion from HTTPS ZIP parts into turbopuffer.
Processes nested ZIPs in-memory using HTTP Range requests for seekable ZIP access.
"""

import io
import logging
import sys
import zipfile
import requests
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from config_ingestion import IngestionConfig, ensure_data_dirs
    from ingestion_utils import EmbeddingProvider, get_chunker
    from turbopuffer_ingestion_sink import build_ingestion_sink
    import dailymed_ingest_lib as dailymed_ingest
    import lxml.etree as ET
except ImportError as e:
    logger.error(f"Failed to import dependencies: {e}")
    sys.exit(1)

# DailyMed URLs
DAILYMED_BASE_URL = "https://dailymed-data.nlm.nih.gov/public-release-files"
HUMAN_RX_ZIPS = [
    "dm_spl_release_human_rx_part1.zip",
    "dm_spl_release_human_rx_part2.zip",
    "dm_spl_release_human_rx_part3.zip",
    "dm_spl_release_human_rx_part4.zip",
    "dm_spl_release_human_rx_part5.zip",
    "dm_spl_release_human_rx_part6.zip",
]

class SeekableHTTPFile(io.IOBase):
    """File-like object that supports seeking and reading from an HTTP URL using Range requests."""
    def __init__(self, url: str):
        self.url = url
        self._pos = 0
        self._length = None
        self._session = requests.Session()
        
        # Get content length
        resp = self._session.head(url, allow_redirects=True)
        resp.raise_for_status()
        self._length = int(resp.headers.get('Content-Length', 0))
        if not self._length:
            raise ValueError(f"Could not determine content length for {url}")

    def seekable(self) -> bool:
        return True

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

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = self._length - self._pos
        if size <= 0:
            return b""
        
        end = self._pos + size - 1
        headers = {'Range': f'bytes={self._pos}-{end}'}
        resp = self._session.get(self.url, headers=headers, stream=True)
        resp.raise_for_status()
        data = resp.content
        self._pos += len(data)
        return data

def parse_xml_content(xml_content: bytes, source_name: str) -> Optional[Dict[str, Any]]:
    """Parse XML content using the logic from 07_ingest_dailymed.py."""
    try:
        # We need to hack ET.parse(str(xml_path)) in the original script or just
        # use the core logic directly.
        # Since _parse_spl_xml_with_status uses ET.parse(str(xml_path)),
        # we can pass an io.BytesIO object if we modify it, or just replicate the important parts.
        
        # Replicating the core parsing from 07_ingest_dailymed for safety
        root = ET.fromstring(xml_content)
        
        # Use existing namespace and filter logic
        NS = {"hl7": "urn:hl7-org:v3"}
        from dailymed_rx_filters import extract_document_label_type, is_human_prescription_label
        
        label_type_code, label_type_display = extract_document_label_type(root, namespaces=NS)
        if not is_human_prescription_label(label_type_code, label_type_display):
            return None

        # Call the existing logic by mocking the tree object if needed, 
        # but here we can just use the functions from dailymed_ingest 
        # if we modify them to accept a root element.
        
        # For now, let's use a temporary file to keep it simple and compatible with 07_ingest_dailymed.py
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=True) as tmp:
            tmp.write(xml_content)
            tmp.flush()
            drug, status = dailymed_ingest._parse_spl_xml_with_status(Path(tmp.name))
            if status == "ok":
                return drug
    except Exception as e:
        logger.warning(f"Failed to parse XML from {source_name}: {e}")
    return None

def process_remote_zip(url: str, processed_ids: set[str], processed_lock: threading.Lock, chunker, embedding_provider, sink):
    """Open a remote ZIP and process its members."""
    logger.info(f"Opening remote ZIP: {url}")
    try:
        http_file = SeekableHTTPFile(url)
        with zipfile.ZipFile(http_file) as zf:
            inner_zips = [m for m in zf.namelist() if m.lower().endswith(".zip")]
            logger.info(f"Found {len(inner_zips)} inner ZIPs in {url.split('/')[-1]}")
            
            for inner_zip_name in inner_zips:
                set_id = Path(inner_zip_name).stem.strip().lower()
                checkpoint_id = f"dailymed:{set_id}"
                
                with processed_lock:
                    if checkpoint_id in processed_ids:
                        continue
                
                try:
                    # Fetch inner zip data
                    with zf.open(inner_zip_name) as inner_zf_data:
                        inner_zip_bytes = inner_zf_data.read()
                        
                        # Open inner zip
                        with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zf:
                            xml_files = [m for m in inner_zf.namelist() if m.lower().endswith(".xml")]
                            for xml_name in xml_files:
                                with inner_zf.open(xml_name) as xml_file:
                                    xml_content = xml_file.read()
                                    drug = parse_xml_content(xml_content, f"{inner_zip_name}/{xml_name}")
                                    
                                    if drug:
                                        # Process drug data (chunk and ingest)
                                        chunks = dailymed_ingest.create_chunks(drug, chunker)
                                        points, chunk_ids = dailymed_ingest.build_points(chunks, embedding_provider)
                                        
                                        if points:
                                            sink.write_points(points)
                                            with processed_lock:
                                                processed_ids.add(checkpoint_id)
                                                # Persistent checkpoint update
                                                with open(dailymed_ingest.CHECKPOINT_FILE, "a") as f:
                                                    f.write(f"{set_id}\n")
                                            logger.info(f"Ingested {set_id} ({len(points)} points)")
                                            
                except Exception as e:
                    logger.error(f"Error processing {inner_zip_name}: {e}")
    except Exception as e:
        logger.error(f"Failed to process remote ZIP {url}: {e}")

def process_inner_zip(url, inner_zip_name, processed_ids, processed_lock, checkpoint_path, chunker, embedding_provider, sink):
    """Fetch and process a single inner ZIP from the outer ZIP, using a fresh connection for thread safety."""
    set_id = Path(inner_zip_name).stem.strip().lower()
    checkpoint_id = f"dailymed:{set_id}"
    
    with processed_lock:
        if checkpoint_id in processed_ids:
            return 0
    
    try:
        # Create a private connection for this worker to allow parallel fetching
        http_file = SeekableHTTPFile(url)
        with zipfile.ZipFile(http_file) as zf:
            # Fetch inner zip data
            with zf.open(inner_zip_name) as inner_zf_data:
                inner_zip_bytes = inner_zf_data.read()
                
                # Open inner zip
                with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zf:
                    xml_files = [m for m in inner_zf.namelist() if m.lower().endswith(".xml")]
                    labels_processed = 0
                    for xml_name in xml_files:
                        with inner_zf.open(xml_name) as xml_file:
                            xml_content = xml_file.read()
                            drug = parse_xml_content(xml_content, f"{inner_zip_name}/{xml_name}")
                            
                            if drug:
                                # Process drug data (chunk and ingest)
                                chunks = dailymed_ingest.create_chunks(drug, chunker)
                                points, chunk_ids = dailymed_ingest.build_points(chunks, embedding_provider)
                                
                                if points:
                                    sink.write_points(points)
                                    with processed_lock:
                                        processed_ids.add(checkpoint_id)
                                        # Persistent checkpoint update
                                        with open(checkpoint_path, "a") as f:
                                            f.write(f"{set_id}\n")
                                    labels_processed += 1
                                    logger.info(f"Ingested {set_id} ({len(points)} points)")
                    return labels_processed
    except Exception as e:
        logger.error(f"Error processing {inner_zip_name}: {e}")
        return 0

def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    parser = argparse.ArgumentParser(description="Direct DailyMed ingestion from HTTPS ZIP parts")
    parser.add_argument("--max-labels", type=int, default=None, help="Maximum number of labels to process")
    parser.add_argument("--parts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="Part numbers to process (1-6)")
    parser.add_argument("--checkpoint", type=Path, default=dailymed_ingest.CHECKPOINT_FILE, help="Path to checkpoint file")
    parser.add_argument("--workers", type=int, default=IngestionConfig.MAX_WORKERS, help="Number of parallel workers")
    args = parser.parse_args()

    ensure_data_dirs()
    
    # Load existing checkpoints
    checkpoint_path = Path(args.checkpoint)
    processed_ids = dailymed_ingest.load_checkpoint_namespaced(checkpoint_path)
    processed_lock = threading.Lock()
    logger.info(f"Loaded {len(processed_ids)} processed IDs from checkpoint: {checkpoint_path}")
    
    embedding_provider = EmbeddingProvider()
    chunker = get_chunker()
    sink = build_ingestion_sink(namespace_override=IngestionConfig.TURBOPUFFER_NAMESPACE_DAILYMED)
    
    labels_count = 0
    for part_num in args.parts:
        if args.max_labels and labels_count >= args.max_labels:
            break
            
        zip_name = f"dm_spl_release_human_rx_part{part_num}.zip"
        url = f"{DAILYMED_BASE_URL}/{zip_name}"
        
        logger.info(f"Opening remote ZIP: {url} with {args.workers} workers")
        try:
            http_file = SeekableHTTPFile(url)
            with zipfile.ZipFile(http_file) as zf:
                inner_zips = [m for m in zf.namelist() if m.lower().endswith(".zip")]
                logger.info(f"Found {len(inner_zips)} inner ZIPs in {zip_name}")
                
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = []
                    for inner_zip_name in inner_zips:
                        if args.max_labels and labels_count >= args.max_labels:
                            break
                        
                        # Preliminary check before submitting to executor to avoid queueing skipped items
                        set_id = Path(inner_zip_name).stem.strip().lower()
                        if f"dailymed:{set_id}" in processed_ids:
                            continue
                            
                        futures.append(executor.submit(
                            process_inner_zip, 
                            url, inner_zip_name, processed_ids, processed_lock, 
                            checkpoint_path, chunker, embedding_provider, sink
                        ))
                        
                    for future in as_completed(futures):
                        try:
                            labels_count += future.result()
                            if args.max_labels and labels_count >= args.max_labels:
                                # Note: as_completed might return more futures than needed
                                pass
                        except Exception as e:
                            logger.error(f"Worker failed: {e}")
                            
        except Exception as e:
            logger.error(f"Failed to process remote ZIP {url}: {e}")
        
    logger.info(f"Direct ingestion complete! Processed {labels_count} labels.")

if __name__ == "__main__":
    main()
