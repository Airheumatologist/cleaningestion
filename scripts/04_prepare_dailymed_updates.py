#!/usr/bin/env python3
"""
Prepare DailyMed incremental updates for ingestion.

This script extracts set_ids from weekly/daily update files and removes them
from the checkpoint file, ensuring updated labels get re-ingested.

Usage:
    python scripts/04_prepare_dailymed_updates.py --xml-dir /data/ingestion/dailymed/xml
"""

import os
import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import config
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config_ingestion import IngestionConfig
    CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "dailymed_ingested_ids.txt"
except Exception:
    CHECKPOINT_FILE = Path("/data/ingestion/dailymed_ingested_ids.txt")


def extract_set_id_from_filename(xml_filename: str) -> str:
    """Extract set_id from XML filename.
    
    DailyMed XML files are typically named: {set_id}.xml
    """
    # Remove .xml extension
    return xml_filename.replace(".xml", "").lower()


def get_update_set_ids(xml_dir: Path) -> Set[str]:
    """Get all set_ids from XML files in the directory."""
    set_ids = set()
    
    if not xml_dir.exists():
        logger.warning(f"XML directory does not exist: {xml_dir}")
        return set_ids
    
    for xml_file in xml_dir.glob("*.xml"):
        set_id = extract_set_id_from_filename(xml_file.name)
        if set_id:
            set_ids.add(set_id)
    
    return set_ids


def clear_checkpoint_entries(set_ids: Set[str], checkpoint_path: Path) -> int:
    """Remove set_ids from checkpoint file so they get re-ingested.
    
    Returns number of entries cleared.
    """
    if not checkpoint_path.exists():
        logger.info(f"Checkpoint file does not exist: {checkpoint_path}")
        return 0
    
    # Read existing checkpoint
    with open(checkpoint_path, 'r') as f:
        lines = f.readlines()
    
    # Filter out lines containing any of the set_ids
    original_count = len(lines)
    filtered_lines = []
    cleared_count = 0
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            filtered_lines.append(line)
            continue
        
        # Check if this line contains any of the set_ids to clear
        should_keep = True
        for set_id in set_ids:
            # Handle both namespaced (dailymed:set_id) and plain set_id formats
            if set_id in line_stripped.lower():
                should_keep = False
                cleared_count += 1
                break
        
        if should_keep:
            filtered_lines.append(line)
    
    # Write back filtered checkpoint
    with open(checkpoint_path, 'w') as f:
        f.writelines(filtered_lines)
    
    return cleared_count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DailyMed updates for ingestion by clearing checkpoint entries"
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=IngestionConfig.DAILYMED_XML_DIR if 'IngestionConfig' in dir() else "/data/ingestion/dailymed/xml",
        help="Directory containing XML files"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_FILE,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleared without modifying checkpoint"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("🔄 Preparing DailyMed Updates for Ingestion")
    logger.info("=" * 70)
    
    # Get set_ids from XML files
    logger.info(f"Scanning XML files in: {args.xml_dir}")
    update_set_ids = get_update_set_ids(args.xml_dir)
    logger.info(f"Found {len(update_set_ids):,} XML files")
    
    if not update_set_ids:
        logger.info("No XML files found - nothing to do")
        return
    
    # Clear checkpoint entries
    if args.dry_run:
        logger.info(f"[DRY RUN] Would clear {len(update_set_ids):,} entries from checkpoint")
    else:
        cleared = clear_checkpoint_entries(update_set_ids, args.checkpoint)
        logger.info(f"✅ Cleared {cleared:,} entries from checkpoint")
        logger.info(f"   Checkpoint file: {args.checkpoint}")
    
    logger.info("\nNext step: Run the ingestion script")
    logger.info(f"   python scripts/07_ingest_dailymed.py --xml-dir {args.xml_dir}")


if __name__ == "__main__":
    main()
