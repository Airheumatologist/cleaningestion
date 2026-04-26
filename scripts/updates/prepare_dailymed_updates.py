#!/usr/bin/env python3
"""
Prepare DailyMed incremental updates for ingestion.

This script loads updated set_ids (preferably from a per-run manifest written
by the downloader) and removes those IDs from the checkpoint file so only
updated labels get re-ingested.

Usage:
    python scripts/updates/prepare_dailymed_updates.py --set-id-manifest /data/ingestion/dailymed/state/dailymed_last_update_set_ids.txt
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fallback_data_dir() -> Path:
    """Mirror config_ingestion default DATA_DIR when config import is unavailable."""
    return Path(os.getenv("DATA_DIR", "/data/ingestion"))


FALLBACK_DAILYMED_XML_DIR = Path(
    os.getenv("DAILYMED_XML_DIR", str(_fallback_data_dir() / "dailymed" / "xml"))
)
FALLBACK_SET_ID_MANIFEST = Path(
    os.getenv(
        "DAILYMED_SET_ID_MANIFEST",
        str(_fallback_data_dir() / "dailymed" / "state" / "dailymed_last_update_set_ids.txt"),
    )
)
FALLBACK_CHECKPOINT_FILE = Path(
    os.getenv("DAILYMED_CHECKPOINT_FILE", str(_fallback_data_dir() / "dailymed_ingested_ids.txt"))
)

try:
    from config_ingestion import IngestionConfig
    DEFAULT_DAILYMED_XML_DIR = IngestionConfig.DAILYMED_XML_DIR
    CHECKPOINT_FILE = IngestionConfig.DAILYMED_CHECKPOINT_FILE
    DEFAULT_SET_ID_MANIFEST = IngestionConfig.DAILYMED_SET_ID_MANIFEST
except Exception:
    DEFAULT_DAILYMED_XML_DIR = FALLBACK_DAILYMED_XML_DIR
    CHECKPOINT_FILE = FALLBACK_CHECKPOINT_FILE
    DEFAULT_SET_ID_MANIFEST = FALLBACK_SET_ID_MANIFEST


def normalize_set_id(value: str) -> str:
    """Normalize set_id for robust matching."""
    return value.strip().lower()


def extract_set_id_from_filename(xml_filename: str) -> str:
    """Extract set_id from XML filename.
    
    DailyMed XML files are typically named: {set_id}.xml
    """
    # Remove .xml extension
    return normalize_set_id(Path(xml_filename).stem)


def load_set_ids_from_manifest(manifest_path: Path) -> Set[str]:
    """Load set_ids from manifest written by incremental downloader."""
    set_ids: Set[str] = set()
    if not manifest_path.exists():
        logger.warning("Set-id manifest not found: %s", manifest_path)
        return set_ids

    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = normalize_set_id(line)
            if value:
                set_ids.add(value)

    return set_ids


def get_update_set_ids_from_xml_dir(xml_dir: Path) -> Set[str]:
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


def parse_checkpoint_line_set_id(line: str) -> Optional[str]:
    """
    Extract set_id from a checkpoint line.

    Supports:
    - dailymed:{set_id}
    - legacy plain {set_id}
    """
    value = line.strip()
    if not value:
        return None

    namespace = "dailymed:"
    if value.lower().startswith(namespace):
        return normalize_set_id(value[len(namespace):])

    # Ignore foreign namespaces (e.g., pmc:...)
    if ":" in value:
        return None

    return normalize_set_id(value)


def clear_checkpoint_entries(set_ids: Set[str], checkpoint_path: Path, dry_run: bool = False) -> int:
    """Remove set_ids from checkpoint file so they get re-ingested.
    
    Returns number of entries cleared.
    """
    if not checkpoint_path.exists():
        logger.info(f"Checkpoint file does not exist: {checkpoint_path}")
        return 0
    
    # Read existing checkpoint
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    target_set_ids = {normalize_set_id(set_id) for set_id in set_ids if normalize_set_id(set_id)}
    if not target_set_ids:
        return 0

    # Filter out exact set_id matches
    filtered_lines = []
    cleared_count = 0
    
    for line in lines:
        if not line.strip():
            filtered_lines.append(line)
            continue

        line_set_id = parse_checkpoint_line_set_id(line)
        if line_set_id is None:
            filtered_lines.append(line)
            continue

        if line_set_id in target_set_ids:
            cleared_count += 1
            continue

        filtered_lines.append(line)
    
    if dry_run:
        return cleared_count

    # Atomic write-back to avoid checkpoint corruption
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)
    temp_path.replace(checkpoint_path)
    
    return cleared_count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DailyMed updates for ingestion by clearing checkpoint entries"
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=DEFAULT_DAILYMED_XML_DIR,
        help="Directory containing XML files (used only with --scan-xml-dir)"
    )
    parser.add_argument(
        "--set-id-manifest",
        type=Path,
        default=DEFAULT_SET_ID_MANIFEST,
        help="Path to set_id manifest generated by incremental downloader"
    )
    parser.add_argument(
        "--scan-xml-dir",
        action="store_true",
        help="Fallback mode: derive set_ids by scanning all XML files in --xml-dir"
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
    
    # Prefer manifest (safe, run-scoped). XML scan is explicit fallback only.
    if args.scan_xml_dir:
        logger.info("Scanning XML files in: %s", args.xml_dir)
        update_set_ids = get_update_set_ids_from_xml_dir(args.xml_dir)
        logger.info("Found %s set_ids from XML directory scan", f"{len(update_set_ids):,}")
    else:
        logger.info("Loading update set_ids from manifest: %s", args.set_id_manifest)
        update_set_ids = load_set_ids_from_manifest(args.set_id_manifest)
        logger.info("Found %s set_ids in manifest", f"{len(update_set_ids):,}")
    
    if not update_set_ids:
        logger.info("No update set_ids found - nothing to clear")
        return
    
    # Clear checkpoint entries
    cleared = clear_checkpoint_entries(update_set_ids, args.checkpoint, dry_run=args.dry_run)
    if args.dry_run:
        logger.info(f"[DRY RUN] Would clear {cleared:,} checkpoint entries")
    else:
        logger.info(f"✅ Cleared {cleared:,} entries from checkpoint")
        logger.info(f"   Checkpoint file: {args.checkpoint}")
    
    logger.info("\nNext step: Run the ingestion script")
    logger.info(f"   python scripts/dailymed_ingest_lib.py --xml-dir {args.xml_dir}")


if __name__ == "__main__":
    main()
