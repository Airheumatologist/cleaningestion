#!/usr/bin/env python3
"""
Download NIH Author Manuscripts from AWS S3.

NIH Author Manuscripts are peer-reviewed manuscripts deposited under the 
NIH Public Access Policy. They are available for text mining and commercial use.

These are NOT included in the PMC OA subset (oa_comm), so this is additional content.

Data Source:
    s3://pmc-oa-opendata/author_manuscript/xml/all/

Usage:
    python 11_download_author_manuscripts.py [--output-dir /data/author_manuscripts/xml]

Expected Duration: 2-4 hours (depending on network speed)
Expected Size: ~400K manuscripts, ~50-80 GB
"""

import os
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('author_manuscripts_download.log')
    ]
)
logger = logging.getLogger(__name__)

# AWS S3 Configuration
# author_manuscript = NIH Public Access Policy manuscripts (commercially usable)
AUTHOR_MANUSCRIPT_S3 = "s3://pmc-oa-opendata/author_manuscript/xml/all/"

DEFAULT_OUTPUT_DIR = Path("/data/author_manuscripts/xml")


def check_aws_cli():
    """Check if AWS CLI is installed."""
    try:
        result = subprocess.run(
            ["aws", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"AWS CLI: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    logger.error("AWS CLI not found. Please install: pip install awscli")
    return False


def download_author_manuscripts(output_dir: Path, dry_run: bool = False):
    """
    Download NIH Author Manuscripts using AWS CLI sync.
    
    Args:
        output_dir: Local directory to store XML files
        dry_run: If True, only show what would be downloaded
    """
    
    logger.info("=" * 70)
    logger.info("📚 NIH Author Manuscripts Download")
    logger.info("=" * 70)
    logger.info(f"Source: {AUTHOR_MANUSCRIPT_S3}")
    logger.info(f"Destination: {output_dir}")
    logger.info("License: Available for text mining (commercially usable)")
    logger.info("=" * 70)
    
    if not check_aws_cli():
        return 1
    
    # Create destination directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # AWS CLI sync command
    # --no-sign-request: No credentials needed for public bucket
    # --only-show-errors: Reduce output verbosity (or remove for progress)
    cmd = [
        "aws", "s3", "sync",
        AUTHOR_MANUSCRIPT_S3,
        str(output_dir),
        "--no-sign-request",
    ]
    
    if dry_run:
        cmd.append("--dryrun")
        logger.info("DRY RUN - showing what would be downloaded...")
    
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info("This may take 2-4 hours depending on network speed...")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    files_synced = 0
    for line in process.stdout:
        line = line.strip()
        if line:
            # Count files being synced
            if line.startswith("download:") or line.startswith("copy:"):
                files_synced += 1
                if files_synced % 1000 == 0:
                    logger.info(f"Progress: {files_synced:,} files synced...")
            elif "error" in line.lower():
                logger.warning(line)
            # Print progress periodically
            if files_synced % 10000 == 0:
                print(line)
    
    process.wait()
    
    elapsed = datetime.now() - start_time
    
    if process.returncode == 0:
        # Count downloaded files
        xml_count = len(list(output_dir.glob("*.xml")))
        tar_count = len(list(output_dir.glob("*.tar.gz")))
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ NIH Author Manuscripts Download Complete!")
        logger.info("=" * 70)
        logger.info(f"   XML files: {xml_count:,}")
        if tar_count > 0:
            logger.info(f"   TAR.GZ files: {tar_count:,}")
        logger.info(f"   Duration: {elapsed}")
        logger.info(f"   Location: {output_dir}")
        logger.info("=" * 70)
        
        # Calculate size
        try:
            result = subprocess.run(
                ["du", "-sh", str(output_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                size = result.stdout.split()[0]
                logger.info(f"   Total size: {size}")
        except:
            pass
        
        logger.info("\n📌 Next steps:")
        logger.info("   1. Process manuscripts: python 14_process_author_manuscripts.py")
        logger.info("   2. Ingest into Qdrant with content_type='author_manuscript'")
        
    else:
        logger.error(f"❌ Download failed with code {process.returncode}")
    
    return process.returncode


def verify_download(output_dir: Path):
    """Verify downloaded files and show statistics."""
    logger.info("\n" + "=" * 70)
    logger.info("📊 Verifying Download")
    logger.info("=" * 70)
    
    xml_files = list(output_dir.glob("*.xml"))
    tar_files = list(output_dir.glob("*.tar.gz"))
    
    logger.info(f"XML files found: {len(xml_files):,}")
    logger.info(f"TAR.GZ files found: {len(tar_files):,}")
    
    if xml_files:
        # Sample a few files
        logger.info("\nSample files:")
        for f in xml_files[:5]:
            size_kb = f.stat().st_size / 1024
            logger.info(f"  - {f.name} ({size_kb:.1f} KB)")
    
    # Check for any obviously corrupted files (too small)
    small_files = [f for f in xml_files if f.stat().st_size < 100]
    if small_files:
        logger.warning(f"\n⚠️  Found {len(small_files)} suspiciously small files (<100 bytes)")
    
    return len(xml_files)


def main():
    parser = argparse.ArgumentParser(
        description="Download NIH Author Manuscripts from AWS S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all author manuscripts
    python 11_download_author_manuscripts.py --output-dir /data/author_manuscripts/xml
    
    # Dry run (see what would be downloaded)
    python 11_download_author_manuscripts.py --dry-run
    
    # Verify existing download
    python 11_download_author_manuscripts.py --verify-only
        """
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for XML files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing download, don't download"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_download(args.output_dir)
    else:
        result = download_author_manuscripts(args.output_dir, args.dry_run)
        if result == 0 and not args.dry_run:
            verify_download(args.output_dir)
        return result


if __name__ == "__main__":
    exit(main() or 0)

