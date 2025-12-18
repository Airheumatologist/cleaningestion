#!/usr/bin/env python3
"""
Download DailyMed Drug Data.

Downloads Structured Product Labeling (SPL) files from DailyMed.
Supports both XML and JSON formats.
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import zipfile
import io
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
# API base (used only for small tests now)
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed"
DAILYMED_DOWNLOAD_URL = "https://dailymed.nlm.nih.gov/dailymed/download.cfm"

# Official bulk public-release files:
# See: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
# Full human RX labels are split into multiple ZIPs, e.g.:
#   dm_spl_release_human_rx_part1.zip ... dm_spl_release_human_rx_part5.zip
# plus additional ZIPs for OTC, homeopathic, animal, remainder, etc.
# The HTTPS links on that page resolve under:
#   https://dailymed-data.nlm.nih.gov/public-release-files/
DAILYMED_PUBLIC_RELEASE_BASE = "https://dailymed-data.nlm.nih.gov/public-release-files"

# Human RX full release parts (5 ZIPs)
HUMAN_RX_ZIPS = [
    "dm_spl_release_human_rx_part1.zip",
    "dm_spl_release_human_rx_part2.zip",
    "dm_spl_release_human_rx_part3.zip",
    "dm_spl_release_human_rx_part4.zip",
    "dm_spl_release_human_rx_part5.zip",
]

OUTPUT_DIR = Path(__file__).parent / "dailymed"
OUTPUT_DIR.mkdir(exist_ok=True)


def download_dailymed_metadata() -> List[Dict]:
    """Download DailyMed metadata (list of available drugs)."""
    logger.info("Fetching DailyMed metadata...")
    
    # DailyMed provides a list of set IDs via their API
    # This is a simplified version - actual implementation would use their API
    
    metadata_url = f"{DAILYMED_BASE_URL}/services/v2/spls.json"
    
    try:
        response = requests.get(metadata_url, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return data.get("data", [])
    
    except Exception as e:
        logger.error(f"Error fetching metadata: {e}")
        logger.info("Falling back to manual download method...")
        return []


def download_spl_file(set_id: str, output_dir: Path) -> Optional[Path]:
    """Download SPL file for a given set ID."""
    # DailyMed SPL files are available via:
    # https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml
    
    xml_url = f"{DAILYMED_BASE_URL}/services/v2/spls/{set_id}.xml"
    
    try:
        response = requests.get(xml_url, timeout=60)
        response.raise_for_status()
        
        output_file = output_dir / f"{set_id}.xml"
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        logger.debug(f"Downloaded {set_id}")
        return output_file
    
    except Exception as e:
        logger.debug(f"Error downloading {set_id}: {e}")
        return None


def download_bulk_dailymed(output_dir: Path, limit: Optional[int] = None):
    """Download DailyMed files via API (small / demo mode).

    NOTE: DailyMed's v2 SPL JSON endpoint is not the recommended way to fetch
    the full corpus. It typically returns a limited window of SPLs.
    For a full production pipeline, prefer the public-release ZIP method
    implemented in `download_public_release_spl` below.
    """
    logger.info("Downloading DailyMed SPL files...")
    
    # Get metadata
    metadata = download_dailymed_metadata()
    
    if not metadata:
        logger.warning("No metadata available, using alternative download method")
        # Alternative: Download from data.gov or use FTP
        return
    
    if limit:
        metadata = metadata[:limit]
    
    logger.info(f"Found {len(metadata)} SPL files to download")
    
    downloaded = 0
    failed = 0
    
    for item in tqdm(metadata, desc="Downloading SPL files"):
        set_id = item.get("setid") or item.get("set_id")
        if not set_id:
            continue
        
        result = download_spl_file(set_id, output_dir)
        if result:
            downloaded += 1
        else:
            failed += 1
        
        # Progress update
        if (downloaded + failed) % 100 == 0:
            logger.info(f"Progress: {downloaded} downloaded, {failed} failed")
    
    logger.info(f"✅ Downloaded {downloaded} SPL files, {failed} failed")


def download_file_stream(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream-download a large file with progress logging."""
    logger.info(f"Downloading {url} → {dest_path}")

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        downloaded = 0

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB "
                                f"({pct:.1f}% of {total / (1024*1024):.1f} MB)")
                else:
                    logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB")

    logger.info(f"✅ Finished downloading {dest_path}")


def extract_zip_xml(zip_path: Path, output_dir: Path) -> None:
    """Extract all XML files from a ZIP into output_dir (flat layout).
    
    Handles nested ZIP files: DailyMed ZIPs contain nested ZIPs which contain XML files.
    """
    logger.info(f"Extracting SPL XML files from {zip_path} into {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        # First, check if there are direct XML files
        xml_members = [m for m in zf.namelist() if m.lower().endswith(".xml")]
        if xml_members:
            logger.info(f"Found {len(xml_members)} direct XML entries in archive {zip_path.name}")
            for name in tqdm(xml_members, desc=f"Extracting XML from {zip_path.name}"):
                target_name = os.path.basename(name)
                if not target_name:
                    continue
                dest_path = output_dir / target_name
                if dest_path.exists():
                    continue
                with zf.open(name) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                extracted_count += 1
        else:
            # Look for nested ZIP files
            nested_zips = [m for m in zf.namelist() if m.lower().endswith(".zip")]
            logger.info(f"Found {len(nested_zips)} nested ZIP files in archive {zip_path.name}")
            
            for nested_zip_name in tqdm(nested_zips, desc=f"Processing nested ZIPs from {zip_path.name}"):
                try:
                    # Extract nested ZIP to temp location
                    with zf.open(nested_zip_name) as nested_zip_data:
                        # Read nested ZIP into memory
                        nested_zip_bytes = nested_zip_data.read()
                        
                        # Open nested ZIP
                        with zipfile.ZipFile(io.BytesIO(nested_zip_bytes), "r") as nested_zf:
                            # Find XML files in nested ZIP
                            xml_files = [m for m in nested_zf.namelist() if m.lower().endswith(".xml")]
                            
                            for xml_name in xml_files:
                                target_name = os.path.basename(xml_name)
                                if not target_name:
                                    continue
                                dest_path = output_dir / target_name
                                if dest_path.exists():
                                    continue
                                
                                # Extract XML from nested ZIP
                                with nested_zf.open(xml_name) as xml_src, open(dest_path, "wb") as dst:
                                    dst.write(xml_src.read())
                                extracted_count += 1
                except Exception as e:
                    logger.debug(f"Error processing nested ZIP {nested_zip_name}: {e}")
                    continue
    
    logger.info(f"✅ Extracted {extracted_count} XML files from {zip_path.name}")


def download_public_release_human_rx(output_dir: Path) -> None:
    """Download and extract the full human RX SPL release (5 ZIP parts).

    Uses the set of ZIPs listed under "HUMAN PRESCRIPTION LABELS" on:
    https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for zip_name in HUMAN_RX_ZIPS:
        zip_url = f"{DAILYMED_PUBLIC_RELEASE_BASE}/{zip_name}"
        local_zip = output_dir / zip_name

        if local_zip.exists() and local_zip.stat().st_size > 0:
            logger.info(f"ZIP already exists, reusing: {local_zip}")
        else:
            download_file_stream(zip_url, local_zip)

        extract_zip_xml(local_zip, output_dir)

    logger.info("✅ Finished downloading and extracting all human RX SPL XML files")


def download_from_datagov(output_dir: Path):
    """Download DailyMed data from data.gov (alternative method)."""
    logger.info("Attempting to download from data.gov...")
    
    # data.gov provides DailyMed dataset
    # URL: https://catalog.data.gov/dataset/dailymed
    # This would require accessing the dataset API or FTP
    
    logger.info("data.gov download not fully implemented - use API method")


def main():
    """Main download function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download DailyMed drug data")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of files to download")
    parser.add_argument(
        "--method",
        choices=["api", "public_rx", "datagov"],
        default="public_rx",
        help="Download method: 'public_rx' (human RX bulk ZIPs), 'api' (limited), or 'datagov' (placeholder)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("💊 Downloading DailyMed Drug Data")
    print("=" * 70)
    
    args.output_dir.mkdir(exist_ok=True)
    
    if args.method == "public_rx":
        # Bulk public-release human RX ZIPs (5 parts)
        download_public_release_human_rx(args.output_dir)
    elif args.method == "api":
        # Legacy / demo method (limited SPLs via JSON API)
        download_bulk_dailymed(args.output_dir, args.limit)
    else:
        download_from_datagov(args.output_dir)
    
    print(f"\n✅ Download complete! Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

