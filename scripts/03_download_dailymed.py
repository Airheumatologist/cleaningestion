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
    
    # Weekly incremental update (recommended for monthly refresh)
    python scripts/03_download_dailymed.py --update-type weekly

Expected Duration: 30-60 minutes for full, 2-5 minutes for weekly updates
"""

import os
import io
import sys
import zipfile
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Set, List
from dailymed_rx_filters import (
    INCREMENTAL_ALLOWED_NESTED_ROOTS,
    select_nested_zip_members,
)

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


def _fallback_dailymed_xml_dir() -> Path:
    """
    Mirror config_ingestion default path construction when config import is unavailable.
    """
    default_data_dir = Path(os.getenv("DATA_DIR", "/data/ingestion"))
    default_xml_dir = default_data_dir / "dailymed" / "xml"
    return Path(os.getenv("DAILYMED_XML_DIR", str(default_xml_dir)))


def _fallback_set_id_manifest() -> Path:
    """Fallback manifest path when centralized config is unavailable."""
    default_data_dir = Path(os.getenv("DATA_DIR", "/data/ingestion"))
    default_state_dir = default_data_dir / "dailymed" / "state"
    return Path(
        os.getenv(
            "DAILYMED_SET_ID_MANIFEST",
            str(default_state_dir / "dailymed_last_update_set_ids.txt"),
        )
    )

# Human prescription drug labels (6 ZIP parts)
HUMAN_RX_ZIPS = [
    "dm_spl_release_human_rx_part1.zip",
    "dm_spl_release_human_rx_part2.zip",
    "dm_spl_release_human_rx_part3.zip",
    "dm_spl_release_human_rx_part4.zip",
    "dm_spl_release_human_rx_part5.zip",
    "dm_spl_release_human_rx_part6.zip",
]

# Optional: Additional categories (11 ZIP parts)
HUMAN_OTC_ZIPS = [
    "dm_spl_release_human_otc_part1.zip",
    "dm_spl_release_human_otc_part2.zip",
    "dm_spl_release_human_otc_part3.zip",
    "dm_spl_release_human_otc_part4.zip",
    "dm_spl_release_human_otc_part5.zip",
    "dm_spl_release_human_otc_part6.zip",
    "dm_spl_release_human_otc_part7.zip",
    "dm_spl_release_human_otc_part8.zip",
    "dm_spl_release_human_otc_part9.zip",
    "dm_spl_release_human_otc_part10.zip",
    "dm_spl_release_human_otc_part11.zip",
]


def get_weekly_update_filename(weeks_ago: int = 0) -> Optional[str]:
    """
    Generate the filename for a weekly update based on weeks ago.
    
    DailyMed weekly files are named: dm_spl_weekly_update_MMDDYY_MMDDYY.zip
    Weeks start on Monday and end on Friday (based on observed patterns).
    
    Args:
        weeks_ago: Number of weeks ago (0 = most recently completed week, 1 = week before, etc.)
    
    Returns:
        Filename string or None if date would be invalid
    """
    # Calculate the target week
    today = datetime.now()
    
    # Find the most recent Friday (end of week)
    # weekday(): Monday=0, Tuesday=1, ..., Friday=4, Saturday=5, Sunday=6
    days_since_friday = (today.weekday() - 4) % 7
    latest_friday = today - timedelta(days=days_since_friday)
    
    # Calculate target week
    target_friday = latest_friday - timedelta(weeks=weeks_ago)
    target_monday = target_friday - timedelta(days=4)
    
    # Format: dm_spl_weekly_update_MMDDYY_MMDDYY.zip
    monday_str = target_monday.strftime("%m%d%y")
    friday_str = target_friday.strftime("%m%d%y")
    
    return f"dm_spl_weekly_update_{monday_str}_{friday_str}.zip"


def _url_exists(url: str, timeout: int = 30) -> bool:
    """
    Return True when an HTTP resource exists.

    Tries HEAD first, then falls back to a streamed GET for servers that do not
    support HEAD reliably.
    """
    try:
        head = requests.head(url, allow_redirects=True, timeout=timeout)
        if head.status_code == 200:
            return True
        if head.status_code not in (403, 405):
            return False
    except Exception:
        # Fall back to GET below.
        pass

    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            return resp.status_code == 200
    except Exception:
        return False


def resolve_weekly_update_filenames(
    weeks_back: int,
    probe_window_weeks: int = 8,
) -> List[str]:
    """
    Resolve the newest published DailyMed weekly update ZIP files.

    DailyMed weekly publication can lag; this probes a range of candidate week
    filenames and keeps only URLs that currently exist.
    """
    wanted = max(1, weeks_back)
    max_probe = max(wanted + probe_window_weeks, wanted)
    existing: List[str] = []

    for weeks_ago in range(max_probe):
        filename = get_weekly_update_filename(weeks_ago=weeks_ago)
        if not filename:
            continue
        zip_url = f"{DAILYMED_PUBLIC_RELEASE_BASE}/{filename}"
        if _url_exists(zip_url):
            existing.append(filename)
            if len(existing) >= wanted:
                break
        else:
            logger.info("Weekly ZIP not yet published, skipping candidate: %s", filename)

    if len(existing) < wanted:
        logger.warning(
            "Requested %d weekly update(s), but only %d currently published in probe window=%d weeks",
            wanted,
            len(existing),
            max_probe,
        )
    return existing


def get_monthly_update_filename(months_ago: int = 0) -> Optional[str]:
    """
    Generate the filename for a monthly update based on months ago.
    
    DailyMed monthly files are named: dm_spl_monthly_update_monYYYY.zip
    (e.g., dm_spl_monthly_update_jan2026.zip)
    
    Args:
        months_ago: Number of months ago (0 = current month, 1 = last month, etc.)
    
    Returns:
        Filename string or None if date would be invalid
    """
    today = datetime.now()
    
    # Calculate target month
    target_month = today.month - months_ago
    target_year = today.year
    
    while target_month <= 0:
        target_month += 12
        target_year -= 1
    
    month_abbr = datetime(target_year, target_month, 1).strftime("%b").lower()
    return f"dm_spl_monthly_update_{month_abbr}{target_year}.zip"


def get_daily_update_filename(days_ago: int = 0) -> Optional[str]:
    """
    Generate the filename for a daily update based on days ago.
    
    DailyMed daily files are named: dm_spl_daily_update_MMDDYYYY.zip
    
    Args:
        days_ago: Number of days ago (0 = today, 1 = yesterday, etc.)
    
    Returns:
        Filename string or None if date would be invalid
    """
    target_date = datetime.now() - timedelta(days=days_ago)
    date_str = target_date.strftime("%m%d%Y")
    return f"dm_spl_daily_update_{date_str}.zip"


def _set_id_from_xml_name(xml_name: str) -> str:
    """Derive normalized set_id from XML filename."""
    return Path(xml_name).stem.strip().lower()


def write_set_id_manifest(set_ids: Set[str], manifest_path: Path) -> None:
    """Persist updated set_ids for downstream checkpoint preparation."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    lines = [f"{set_id}\n" for set_id in sorted(set_ids)]
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    os.replace(temp_path, manifest_path)

# Default output dir - uses centralized config if available, otherwise fallback
if HAS_CONFIG:
    DEFAULT_OUTPUT_DIR = IngestionConfig.DAILYMED_XML_DIR
    DEFAULT_SET_ID_MANIFEST = IngestionConfig.DAILYMED_SET_ID_MANIFEST
else:
    DEFAULT_OUTPUT_DIR = _fallback_dailymed_xml_dir()
    DEFAULT_SET_ID_MANIFEST = _fallback_set_id_manifest()


def _is_valid_zip(path: Path) -> bool:
    """Return True when a ZIP file can be opened and indexed."""
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.namelist()
        return True
    except Exception:
        return False


def download_file_stream(url: str, dest_path: Path, chunk_size: int = 1024 * 1024):
    """Stream-download a large file with progress logging."""
    logger.info(f"Downloading: {url}")
    logger.info(f"Destination: {dest_path}")
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length") or 0)
            downloaded = 0
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, "wb") as f:
                with tqdm(total=total, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))

        if downloaded <= 0:
            raise RuntimeError("Downloaded file is empty")

        if not _is_valid_zip(temp_path):
            raise RuntimeError(f"Downloaded archive is invalid: {dest_path.name}")

        os.replace(temp_path, dest_path)
        
        logger.info(f"✅ Downloaded: {dest_path.name} ({downloaded / (1024*1024):.1f} MB)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def extract_nested_zip_xml(
    zip_path: Path,
    output_dir: Path,
    refresh: bool = False,
    allowed_nested_roots: Optional[Set[str]] = None,
) -> tuple[int, Set[str]]:
    """
    Extract XML files from a DailyMed ZIP.
    
    DailyMed ZIPs contain nested ZIPs, not XML directly:
    - outer.zip → contains many inner.zip files
    - inner.zip → contains the actual XML file
    
    Args:
        zip_path: Outer DailyMed ZIP file path.
        output_dir: Destination directory for extracted XML files.
        refresh: Whether to overwrite existing XMLs.
        allowed_nested_roots: Optional top-level member roots to include for nested ZIPs.

    Returns:
        Tuple of (number of XML files extracted, set_ids extracted)
    """
    logger.info(f"Extracting: {zip_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    extracted_set_ids: Set[str] = set()
    
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
                if dest_path.exists() and not refresh:
                    continue
                with zf.open(name) as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
                extracted_count += 1
                extracted_set_ids.add(_set_id_from_xml_name(target_name))
        else:
            # Nested ZIPs (standard DailyMed format)
            nested_zips = [m for m in zf.namelist() if m.lower().endswith(".zip")]
            logger.info(f"  Found {len(nested_zips)} nested ZIP files")

            selected_nested_zips, root_counts, used_fallback = select_nested_zip_members(
                nested_zips,
                allowed_roots=allowed_nested_roots,
            )
            if allowed_nested_roots:
                roots_str = ", ".join(f"{root}={count}" for root, count in sorted(root_counts.items()))
                logger.info(f"  Nested ZIP roots: {roots_str}")
                if used_fallback:
                    allowlist = ", ".join(sorted(r.lower() for r in allowed_nested_roots))
                    logger.warning(
                        "  No nested ZIPs matched allowed roots [%s]; falling back to all nested ZIPs",
                        allowlist,
                    )
                else:
                    excluded = len(nested_zips) - len(selected_nested_zips)
                    logger.info(
                        "  Applied nested ZIP root filter: selected=%d excluded=%d",
                        len(selected_nested_zips),
                        excluded,
                    )
            
            for nested_zip_name in tqdm(selected_nested_zips, desc=f"Processing {zip_path.name}"):
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
                                if dest_path.exists() and not refresh:
                                    continue
                                
                                # Extract XML from nested ZIP
                                with nested_zf.open(xml_name) as xml_src, open(dest_path, "wb") as dst:
                                    dst.write(xml_src.read())
                                extracted_count += 1
                                extracted_set_ids.add(_set_id_from_xml_name(target_name))
                                
                except Exception as e:
                    logger.debug(f"Error processing {nested_zip_name}: {e}")
                    continue
    
    logger.info(f"✅ Extracted {extracted_count:,} XML files from {zip_path.name}")
    return extracted_count, extracted_set_ids


def download_dailymed(
    output_dir: Path, 
    include_otc: bool = False, 
    refresh: bool = False,
    update_type: Optional[str] = None,
    weeks_back: int = 1,
    weekly_probe_window: int = 8,
    days_back: int = 7,
    set_id_manifest: Optional[Path] = None
):
    """
    Download and extract DailyMed drug labels.
    
    Args:
        output_dir: Directory to store extracted XML files
        include_otc: Also download OTC (over-the-counter) drugs (full refresh only)
        refresh: If True, re-download and re-extract all files (full release)
        update_type: Type of update - 'daily', 'weekly', 'monthly', or None for full release
        weeks_back: Number of weekly files to download, starting from most recent published week
        weekly_probe_window: Additional older weeks to probe when newest week is not yet published
        days_back: Number of days to go back for daily updates (default: 7 = last 7 days)
        set_id_manifest: Path to persist updated set_ids for incremental re-ingestion
    """
    logger.info("=" * 70)
    logger.info("💊 DailyMed Drug Labels Download")
    logger.info("=" * 70)
    
    # Ensure all data directories exist (creates parent dirs as needed)
    if HAS_CONFIG:
        ensure_data_dirs()
    
    # Also ensure the specific output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which ZIPs to download based on update type
    if update_type == "weekly":
        zip_files = resolve_weekly_update_filenames(
            weeks_back=weeks_back,
            probe_window_weeks=weekly_probe_window,
        )
        logger.info(f"📅 Weekly update mode: downloading {len(zip_files)} weekly update(s)")
        
    elif update_type == "daily":
        zip_files = []
        for i in range(days_back):
            filename = get_daily_update_filename(days_ago=i)
            if filename:
                zip_files.append(filename)
        logger.info(f"📅 Daily update mode: downloading {len(zip_files)} daily update(s)")
        
    elif update_type == "monthly":
        filename = get_monthly_update_filename(months_ago=0)
        zip_files = [filename] if filename else []
        logger.info(f"📅 Monthly update mode: downloading {len(zip_files)} monthly update(s)")
        
    else:
        # Full release (default)
        zip_files = HUMAN_RX_ZIPS.copy()
        if include_otc:
            zip_files.extend(HUMAN_OTC_ZIPS)
        logger.info(f"📦 Full release mode: downloading {len(zip_files)} ZIP files...")
    
    logger.info(f"Destination: {output_dir}")
    
    total_extracted = 0
    downloaded_count = 0
    updated_set_ids: Set[str] = set()
    
    for zip_name in zip_files:
        zip_url = f"{DAILYMED_PUBLIC_RELEASE_BASE}/{zip_name}"
        local_zip = output_dir / zip_name
        
        # Download if not exists (or refresh). Validate any reused local ZIP first.
        if not refresh and _is_valid_zip(local_zip):
            logger.info(f"ZIP already exists, reusing: {zip_name}")
        else:
            if local_zip.exists():
                if refresh:
                    logger.info(f"Refresh mode: re-downloading {zip_name}")
                else:
                    logger.warning(f"Existing ZIP is invalid/incomplete, re-downloading: {zip_name}")
                local_zip.unlink(missing_ok=True)
            success = download_file_stream(zip_url, local_zip)
            if not success:
                logger.warning(f"Failed to download {zip_name}, it may not exist yet...")
                continue
            downloaded_count += 1
        
        # Extract XML files
        # For incremental updates, always overwrite existing XMLs to get latest versions
        extract_refresh = True if update_type else refresh
        allowed_nested_roots = INCREMENTAL_ALLOWED_NESTED_ROOTS if update_type in ("daily", "weekly", "monthly") else None
        extracted, extracted_ids = extract_nested_zip_xml(
            local_zip,
            output_dir,
            refresh=extract_refresh,
            allowed_nested_roots=allowed_nested_roots,
        )
        total_extracted += extracted
        updated_set_ids.update(extracted_ids)
        
        # For incremental updates, clean up ZIP files after extraction to save space
        if update_type in ("daily", "weekly", "monthly"):
            local_zip.unlink(missing_ok=True)
            logger.info(f"🗑️  Cleaned up: {zip_name}")
    
    # Count total XML files
    xml_count = len(list(output_dir.glob("*.xml")))

    manifest_path = set_id_manifest
    if update_type:
        if manifest_path is None:
            manifest_path = DEFAULT_SET_ID_MANIFEST
        write_set_id_manifest(updated_set_ids, manifest_path)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ DailyMed Download Complete!")
    logger.info("=" * 70)
    logger.info(f"   Downloaded files: {downloaded_count}")
    logger.info(f"   Total XML files extracted this run: {total_extracted:,}")
    logger.info(f"   Total XML files in directory: {xml_count:,}")
    if update_type and manifest_path is not None:
        logger.info(f"   Updated set_ids in manifest: {len(updated_set_ids):,}")
        logger.info(f"   Manifest: {manifest_path}")
    logger.info(f"   Location: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download DailyMed drug labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full baseline download (all labels)
  python scripts/03_download_dailymed.py
  
  # Weekly incremental update (recommended for regular updates)
  python scripts/03_download_dailymed.py --update-type weekly
  
  # Multiple weeks of updates
  python scripts/03_download_dailymed.py --update-type weekly --weeks-back 4
  
  # Daily updates (last 7 days)
  python scripts/03_download_dailymed.py --update-type daily
  
  # Monthly update
  python scripts/03_download_dailymed.py --update-type monthly
        """
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for XML files"
    )
    parser.add_argument(
        "--include-otc",
        action="store_true",
        help="Also download OTC (over-the-counter) drug labels (full release only)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download and re-extract all files (for full release)"
    )
    parser.add_argument(
        "--update-type",
        choices=["daily", "weekly", "monthly"],
        default=None,
        help="Type of incremental update to download (default: full release)"
    )
    parser.add_argument(
        "--weeks-back",
        type=int,
        default=1,
        help="Number of weekly files to download for weekly updates (default: 1)"
    )
    parser.add_argument(
        "--weekly-probe-window",
        type=int,
        default=8,
        help="Extra older weeks to probe when the newest weekly ZIP is not yet published (default: 8)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="Number of days to download for daily updates (default: 7)"
    )
    parser.add_argument(
        "--set-id-manifest",
        type=Path,
        default=DEFAULT_SET_ID_MANIFEST,
        help="Path to write updated set_ids (incremental modes only)"
    )
    
    args = parser.parse_args()
    download_dailymed(
        args.output_dir, 
        args.include_otc, 
        args.refresh,
        args.update_type,
        args.weeks_back,
        args.weekly_probe_window,
        args.days_back,
        args.set_id_manifest
    )


if __name__ == "__main__":
    main()
