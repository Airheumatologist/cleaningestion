#!/usr/bin/env python3
"""
Monthly Incremental Update Script.

Downloads and ingests new articles from PMC and DailyMed
that were published in the last 30 days.

PMC provides incremental updates via:
    s3://pmc-oa-opendata/oa_comm/xml/incr/YYYY-MM-DD/

Usage:
    python 08_monthly_update.py [--days-back 30]

Automation:
    - Run on 1st of each month via cron or GitHub Actions
    - Start EC2 → Run update → Stop EC2
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

# Load environment
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PMC_INCR_BASE = "s3://pmc-oa-opendata/oa_comm/xml/incr/"
LOCAL_INCR_DIR = Path("/data/pmc_incremental")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pmc_medical_rag_fulltext")


def download_pmc_incremental(days_back: int = 30) -> list:
    """
    Download PMC incremental updates for the past N days.
    
    Returns list of downloaded XML files.
    """
    
    logger.info("=" * 70)
    logger.info("📥 Downloading PMC Incremental Updates")
    logger.info("=" * 70)
    
    LOCAL_INCR_DIR.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now()
    downloaded_files = []
    
    for i in range(days_back):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        s3_path = f"{PMC_INCR_BASE}{date_str}/"
        local_path = LOCAL_INCR_DIR / date_str
        
        logger.info(f"Checking {date_str}...")
        
        cmd = [
            "aws", "s3", "sync",
            s3_path,
            str(local_path),
            "--no-sign-request",
            "--quiet"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if local_path.exists():
                xml_files = list(local_path.glob("*.xml"))
                if xml_files:
                    logger.info(f"  ✅ {len(xml_files)} new articles for {date_str}")
                    downloaded_files.extend(xml_files)
        except subprocess.TimeoutExpired:
            logger.warning(f"  ⚠️ Timeout for {date_str}")
        except Exception as e:
            logger.debug(f"  No updates for {date_str}")
    
    logger.info(f"\n📊 Total new XML files: {len(downloaded_files):,}")
    return downloaded_files


def process_and_ingest(xml_files: list) -> int:
    """
    Process new XML files and ingest to Qdrant.
    
    Returns count of ingested articles.
    """
    
    if not xml_files:
        logger.info("No new files to process")
        return 0
    
    logger.info("\n" + "=" * 70)
    logger.info("📤 Processing and Ingesting New Articles")
    logger.info("=" * 70)
    
    # Import from other scripts
    from extract_pmc import parse_xml_file  # Script 02
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, Document
    except ImportError:
        os.system("pip3 install qdrant-client --quiet")
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, Document
    
    import uuid
    
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,
        cloud_inference=True
    )
    
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"✅ Connected to Qdrant: {COLLECTION_NAME}")
    logger.info(f"   Current points: {info.points_count:,}")
    
    # Process files
    ingested = 0
    errors = 0
    batch = []
    BATCH_SIZE = 50
    EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
    
    for xml_file in tqdm(xml_files, desc="Processing"):
        try:
            article = parse_xml_file(xml_file)
            if not article:
                continue
            
            # Create embedding text
            title = article.get("title", "") or ""
            abstract = article.get("abstract", "") or ""
            full_text = article.get("full_text", "") or ""
            embedding_text = f"{title}. {abstract}"[:1500]
            if full_text:
                embedding_text += f"\n\n{full_text[:500]}"
            embedding_text = embedding_text[:2000]
            
            if len(embedding_text) < 20:
                continue
            
            # Create point
            doc_id = article.get("pmcid") or article.get("pmid")
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))
            
            point = PointStruct(
                id=point_id,
                vector=Document(text=embedding_text, model=EMBEDDING_MODEL),
                payload={
                    "pmcid": article.get("pmcid"),
                    "pmid": article.get("pmid"),
                    "doi": article.get("doi"),
                    "title": article.get("title", "")[:300],
                    "abstract": article.get("abstract", "")[:1000],
                    "full_text": article.get("full_text", "")[:10000],
                    "year": article.get("year"),
                    "journal": article.get("journal", ""),
                    "article_type": article.get("article_type", ""),
                    "evidence_grade": article.get("evidence_grade"),
                    "country": article.get("country"),
                    "source": "pmc",
                    "has_full_text": article.get("has_full_text", False),
                }
            )
            
            batch.append(point)
            
            if len(batch) >= BATCH_SIZE:
                try:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=batch,
                        wait=False
                    )
                    ingested += len(batch)
                except Exception as e:
                    errors += len(batch)
                    logger.error(f"Batch error: {e}")
                batch = []
                
        except Exception as e:
            errors += 1
            logger.debug(f"Error processing {xml_file}: {e}")
    
    # Final batch
    if batch:
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            ingested += len(batch)
        except Exception as e:
            errors += len(batch)
    
    return ingested


def main():
    parser = argparse.ArgumentParser(description="Monthly RAG Update")
    parser.add_argument("--days-back", type=int, default=30, help="Days to look back")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use existing files")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("📅 Monthly RAG Update")
    logger.info(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 70)
    
    # Check environment
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    # Download incremental updates
    if args.skip_download:
        xml_files = list(LOCAL_INCR_DIR.rglob("*.xml"))
        logger.info(f"Using existing files: {len(xml_files):,}")
    else:
        xml_files = download_pmc_incremental(args.days_back)
    
    # Process and ingest
    if xml_files:
        count = process_and_ingest(xml_files)
        logger.info(f"\n✅ Monthly update complete: {count:,} new articles added")
    else:
        logger.info("\n✅ No new articles to add")
    
    logger.info("\n📊 Cleanup tip: Stop EC2 to save costs!")


if __name__ == "__main__":
    main()

