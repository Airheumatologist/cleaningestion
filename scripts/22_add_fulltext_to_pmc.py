#!/usr/bin/env python3
"""
Add Full Text to Existing PMC Articles in Qdrant.

Updates existing PMC article payloads with full_text from the JSONL file.
Uses set_payload to efficiently add/update just the full_text field.

Usage:
    python 22_add_fulltext_to_pmc.py --jsonl /data/pmc_fulltext/pmc_articles.jsonl

Expected Duration: ~2-3 hours for 1.2M articles
"""

import os
import sys
import json
import logging
import argparse
import time
import uuid
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList
except ImportError:
    os.system("pip3 install qdrant-client --quiet")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('add_fulltext.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pmc_medical_rag_fulltext")

BATCH_SIZE = 100  # Points to update per batch
MAX_FULL_TEXT_LENGTH = 50000  # Store full article text
MAX_RETRIES = 3
CHECKPOINT_FILE = Path("add_fulltext_checkpoint.txt")
PROGRESS_LOG_INTERVAL = 100


def get_point_id(pmcid: str) -> str:
    """Generate the same point ID used during ingestion."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, pmcid))


def load_checkpoint() -> set:
    """Load checkpoint of already updated PMCIDs."""
    if CHECKPOINT_FILE.exists():
        try:
            return set(CHECKPOINT_FILE.read_text().strip().split('\n'))
        except:
            return set()
    return set()


def save_checkpoint(pmcids: list):
    """Append PMCIDs to checkpoint file."""
    with open(CHECKPOINT_FILE, 'a') as f:
        for pmcid in pmcids:
            f.write(f"{pmcid}\n")


def build_fulltext_mapping(jsonl_path: Path, updated_set: set) -> Dict[str, str]:
    """
    Build mapping of PMCID -> full_text from JSONL file.
    Only includes articles not yet updated.
    """
    logger.info(f"Building full_text mapping from {jsonl_path}...")
    mapping = {}
    
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f, desc="Loading JSONL"):
            if not line.strip():
                continue
            try:
                article = json.loads(line)
                pmcid = article.get("pmcid")
                if not pmcid:
                    continue
                
                # Skip if already updated
                if pmcid in updated_set:
                    continue
                
                full_text = article.get("full_text", "")
                if full_text and len(full_text) > 100:
                    # Truncate to save storage
                    mapping[pmcid] = full_text[:MAX_FULL_TEXT_LENGTH]
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(mapping):,} articles with full_text to update")
    return mapping


def update_batch(client: QdrantClient, batch: list) -> tuple:
    """
    Update a batch of points with full_text.
    Returns (success_count, failed_pmcids)
    """
    success = 0
    failed = []
    
    for pmcid, full_text in batch:
        point_id = get_point_id(pmcid)
        
        for attempt in range(MAX_RETRIES):
            try:
                # Use set_payload to add/update just the full_text field
                client.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={"full_text": full_text},
                    points=[point_id],
                    wait=False
                )
                success += 1
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    failed.append(pmcid)
                    logger.debug(f"Failed to update {pmcid}: {e}")
                else:
                    time.sleep(0.5 * (attempt + 1))
    
    return success, failed


def run_update(jsonl_path: Path, workers: int = 4):
    """Run the full text update process."""
    
    logger.info("=" * 70)
    logger.info("📝 Add Full Text to PMC Articles")
    logger.info("=" * 70)
    
    # Validate configuration
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    if not jsonl_path.exists():
        logger.error(f"❌ JSONL file not found: {jsonl_path}")
        sys.exit(1)
    
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=300
    )
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to Qdrant: {QDRANT_URL}")
        logger.info(f"   Collection: {COLLECTION_NAME}")
        logger.info(f"   Total points: {info.points_count:,}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    # Load checkpoint
    updated_set = load_checkpoint()
    logger.info(f"   Already updated: {len(updated_set):,}")
    
    # Build mapping from JSONL
    fulltext_mapping = build_fulltext_mapping(jsonl_path, updated_set)
    
    if not fulltext_mapping:
        logger.info("✅ All articles already updated!")
        return
    
    # Process updates
    logger.info(f"\n🚀 Starting updates with {workers} workers...")
    start_time = time.time()
    
    total_success = 0
    total_failed = 0
    batch_num = 0
    
    # Convert to list for batching
    items = list(fulltext_mapping.items())
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i:i + BATCH_SIZE]
            future = executor.submit(update_batch, client, batch)
            futures.append((future, [pmcid for pmcid, _ in batch]))
        
        with tqdm(total=len(items), desc="Updating") as pbar:
            for future, pmcids in futures:
                try:
                    success, failed = future.result(timeout=300)
                    total_success += success
                    total_failed += len(failed)
                    
                    # Save checkpoint for successful updates
                    successful_pmcids = [p for p in pmcids if p not in failed]
                    if successful_pmcids:
                        save_checkpoint(successful_pmcids)
                    
                    pbar.update(len(pmcids))
                    batch_num += 1
                    
                    if batch_num % PROGRESS_LOG_INTERVAL == 0:
                        elapsed = time.time() - start_time
                        rate = total_success / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Progress: {total_success:,} updated, "
                            f"{total_failed:,} failed, {rate:.1f}/sec"
                        )
                except Exception as e:
                    logger.error(f"Batch error: {e}")
                    total_failed += len(pmcids)
    
    # Final stats
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Full Text Update Complete!")
    logger.info("=" * 70)
    logger.info(f"📊 Results:")
    logger.info(f"   Updated: {total_success:,}")
    logger.info(f"   Failed: {total_failed:,}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes")
    logger.info(f"   Rate: {total_success/elapsed:.1f}/sec")


def main():
    parser = argparse.ArgumentParser(description="Add full_text to PMC articles in Qdrant")
    parser.add_argument("--jsonl", type=Path, required=True, help="JSONL file with full text")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    
    args = parser.parse_args()
    run_update(args.jsonl, args.workers)


if __name__ == "__main__":
    main()
