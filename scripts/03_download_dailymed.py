#!/usr/bin/env python3
"""
Download DailyMed Drug Labels.

Downloads Structured Product Labeling (SPL) files from DailyMed FDA database.
Uses the official bulk public-release ZIP files.

Data Source:
    https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm

IMPORTANT: DailyMed ZIPs contain NESTED ZIPs, not XML directly.
Each nested ZIP contains one drug's SPL XML file.

Usage:
    # Server deployment (uses DAILYMED_XML_DIR from .env)
    python scripts/03_download_dailymed.py
    
    # Or specify custom path
    python scripts/03_download_dailymed.py --output-dir /data/ingestion/dailymed/xml

Expected Duration: 30-60 minutes
"""

import os
import io
import sys
import zipfile
import logging
import argparse
import requests
from pathlib import Path

# Setup logging BEFORE any imports that might fail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    logger.info("Installing tqdm...")
    os.system("pip3 install tqdm requests --quiet")
    from tqdm import tqdm

# Import centralized config to ensure path consistency
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from config_ingestion import IngestionConfig, ensure_data_dirs
    HAS_CONFIG = True
except Exception as e:
    logger.warning("Could not import config_ingestion: %s", e)
    HAS_CONFIG = False

# DailyMed bulk download configuration
DAILYMED_PUBLIC_RELEASE_BASE = "https://dailymed-data.nlm.nih.gov/public-release-files"

# Human prescription drug labels (5 ZIP parts)
HUMAN_RX_ZIPS = [
    "dm_spl_release_human_rx_part1.zip",
    "dm_spl_release_human_rx_part2.zip",
    "dm_spl_release_human_rx_part3.zip",
    "dm_spl_release_human_rx_part4.zip",
    "dm_spl_release_human_rx_part5.zip",
]

# Optional: Additional categories
HUMAN_OTC_ZIPS = [
    "dm_spl_release_human_otc_part1.zip",
    "dm_spl_release_human_otc_part2.zip",
    "dm_spl_release_human_otc_part3.zip",
    "dm_spl_release_human_otc_part4.zip",
]

# Default output dir - uses centralized config if available, otherwise fallback
if HAS_CONFIG:
    DEFAULT_OUTPUT_DIR = IngestionConfig.DAILYMED_XML_DIR
else:
    DEFAULT_OUTPUT_DIR = Path(os.getenv("DAILYMED_XML_DIR", "/data/ingestion/dailymed/xml"))


def download_file_stream(url: str, dest_path: Path, chunk_size: int = 1024 * 1024):
    """Stream-download a large file with progress logging."""
    logger.info(f"Downloading: {url}")
    logger.info(f"Destination: {dest_path}")
    
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            downloaded = 0
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dest_path, "wb") as f:
                with tqdm(total=total, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✅ Downloaded: {dest_path.name} ({downloaded / (1024*1024):.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


def extract_nested_zip_xml(zip_path: Path, output_dir: Path) -> int:
    """
    Extract XML files from a DailyMed ZIP.
    
    DailyMed ZIPs contain nested ZIPs, not XML directly:
    - outer.zip → contains many inner.zip files
    - inner.zip → contains the actual XML file
    
    Returns number of XML files extracted.
    """
    logger.info(f"Extracting: {zip_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Check for direct XML files first
        xml_members = [m for m in zf.namelist() if m.lower().endswith(".xml")]
        
        if xml_members:
            # Direct XML files (unlikely but handle it)
            logger.info(f"  Found {len(xml_members)} direct XML files")
            for name in tqdm(xml_members, desc=f"Extracting {zip_path.name}"):
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
            # Nested ZIPs (standard DailyMed format)
            nested_zips = [m for m in zf.namelist() if m.lower().endswith(".zip")]
            logger.info(f"  Found {len(nested_zips)} nested ZIP files")
            
            for nested_zip_name in tqdm(nested_zips, desc=f"Processing {zip_path.name}"):
                try:
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
                    logger.debug(f"Error processing {nested_zip_name}: {e}")
                    continue
    
    logger.info(f"✅ Extracted {extracted_count:,} XML files from {zip_path.name}")
    return extracted_count


def download_dailymed(output_dir: Path, include_otc: bool = False):
    """
    Download and extract DailyMed drug labels.
    
    Args:
        output_dir: Directory to store extracted XML files
        include_otc: Also download OTC (over-the-counter) drugs
    """
    logger.info("=" * 70)
    logger.info("💊 DailyMed Drug Labels Download")
    logger.info("=" * 70)
    
    # Ensure all data directories exist (creates parent dirs as needed)
    if HAS_CONFIG:
        ensure_data_dirs()
    
    # Also ensure the specific output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which ZIPs to download
    zip_files = HUMAN_RX_ZIPS.copy()
    if include_otc:
        zip_files.extend(HUMAN_OTC_ZIPS)
    
    logger.info(f"Downloading {len(zip_files)} ZIP files...")
    logger.info(f"Destination: {output_dir}")
    
    total_extracted = 0
    
    for zip_name in zip_files:
        zip_url = f"{DAILYMED_PUBLIC_RELEASE_BASE}/{zip_name}"
        local_zip = output_dir / zip_name
        
        # Download if not exists
        if local_zip.exists() and local_zip.stat().st_size > 0:
            logger.info(f"ZIP already exists, reusing: {zip_name}")
        else:
            success = download_file_stream(zip_url, local_zip)
            if not success:
                logger.error(f"Failed to download {zip_name}, skipping...")
                continue
        
        # Extract XML files
        extracted = extract_nested_zip_xml(local_zip, output_dir)
        total_extracted += extracted
        
        # Optional: Remove ZIP to save space
        # local_zip.unlink()
    
    # Count total XML files
    xml_count = len(list(output_dir.glob("*.xml")))
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ DailyMed Download Complete!")
    logger.info("=" * 70)
    logger.info(f"   Total XML files: {xml_count:,}")
    logger.info(f"   Location: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download DailyMed drug labels")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for XML files"
    )
    parser.add_argument(
        "--include-otc",
        action="store_true",
        help="Also download OTC (over-the-counter) drug labels"
    )
    
    args = parser.parse_args()
    download_dailymed(args.output_dir, args.include_otc)


if __name__ == "__main__":
    main()

