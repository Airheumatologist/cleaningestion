#!/usr/bin/env python3
"""
Download PMC Open Access Articles from AWS S3.

PMC Open Access articles are available via AWS S3 (no authentication required).
This script downloads full-text articles in JATS XML format.

Data Source:
    s3://pmc-oa-opendata/oa_comm/xml/all/

Usage:
    python 01_download_pmc.py [--output-dir /data/pmc_fulltext/xml]

Expected Duration: 4-8 hours (depending on network speed)
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
        logging.FileHandler('pmc_download.log')
    ]
)
logger = logging.getLogger(__name__)

# PMC S3 Configuration
# oa_comm = Commercial-friendly Open Access (CC BY, CC0) with FULL TEXT
# oa_noncomm = Non-commercial Open Access with FULL TEXT
# oa_other = Other OA licenses with FULL TEXT
PMC_S3_BUCKETS = {
    "oa_comm": "s3://pmc-oa-opendata/oa_comm/xml/all/",      # Recommended
    "oa_noncomm": "s3://pmc-oa-opendata/oa_noncomm/xml/all/",
    "oa_other": "s3://pmc-oa-opendata/oa_other/xml/all/",
}

DEFAULT_OUTPUT_DIR = Path("/data/pmc_fulltext/xml")


def download_pmc(output_dir: Path, subset: str = "oa_comm"):
    """
    Download PMC articles using AWS CLI sync.
    
    Args:
        output_dir: Local directory to store XML files
        subset: Which OA subset to download (oa_comm, oa_noncomm, oa_other)
    """
    
    if subset not in PMC_S3_BUCKETS:
        logger.error(f"Invalid subset: {subset}. Valid options: {list(PMC_S3_BUCKETS.keys())}")
        return 1
    
    s3_path = PMC_S3_BUCKETS[subset]
    
    logger.info("=" * 70)
    logger.info("📚 PMC Open Access Download")
    logger.info("=" * 70)
    logger.info(f"Source: {s3_path}")
    logger.info(f"Destination: {output_dir}")
    logger.info(f"Subset: {subset} (full-text articles)")
    logger.info("=" * 70)
    
    # Create destination directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # AWS CLI sync command
    # --no-sign-request: No credentials needed for public bucket
    # --only-show-errors: Reduce output verbosity
    cmd = [
        "aws", "s3", "sync",
        s3_path,
        str(output_dir),
        "--no-sign-request",
        "--only-show-errors"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info("This may take 4-8 hours depending on network speed...")
    
    start_time = datetime.now()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    elapsed = datetime.now() - start_time
    
    if process.returncode == 0:
        # Count downloaded files
        xml_count = len(list(output_dir.glob("*.xml")))
        logger.info("\n" + "=" * 70)
        logger.info("✅ PMC Download Complete!")
        logger.info("=" * 70)
        logger.info(f"   XML files: {xml_count:,}")
        logger.info(f"   Duration: {elapsed}")
        logger.info(f"   Location: {output_dir}")
    else:
        logger.error(f"❌ Download failed with code {process.returncode}")
    
    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Download PMC Open Access articles from AWS S3"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for XML files"
    )
    parser.add_argument(
        "--subset",
        choices=["oa_comm", "oa_noncomm", "oa_other", "all"],
        default="oa_comm",
        help="Which OA subset to download (default: oa_comm)"
    )
    
    args = parser.parse_args()
    
    if args.subset == "all":
        # Download all subsets
        for subset in PMC_S3_BUCKETS.keys():
            download_pmc(args.output_dir, subset)
    else:
        download_pmc(args.output_dir, args.subset)


if __name__ == "__main__":
    main()

