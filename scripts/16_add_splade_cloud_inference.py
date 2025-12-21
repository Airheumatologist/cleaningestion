#!/usr/bin/env python3
"""
Add SPLADE Sparse Vectors via Qdrant Cloud Inference.

This script adds SPLADE sparse vectors to existing points using Qdrant's
Cloud Inference service, which is MUCH faster than local CPU-based SPLADE.

Cloud Inference Model: prithivida/splade_pp_en_v1
Cost: $0.06 per 1M tokens

Usage:
    export QDRANT_API_KEY="your_key"
    python 16_add_splade_cloud_inference.py

Expected Speed: ~100-200 points/sec (vs ~5/sec on CPU)
"""

import os
import sys
import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointVectors, Document
except ImportError:
    os.system("pip3 install qdrant-client --quiet")
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointVectors, Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/splade_cloud_inference.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://cf6c28ca-8a2a-43fa-9424-1f2af9e9a5f3.us-east-1-1.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pmc_medical_rag_fulltext"

# SPLADE Cloud Inference Model
SPLADE_MODEL = "prithivida/splade_pp_en_v1"

# Processing settings
SCROLL_BATCH_SIZE = 100  # Points to fetch at once
UPDATE_BATCH_SIZE = 50   # Points to update at once
PARALLEL_WORKERS = 4
MAX_RETRIES = 3
MAX_TEXT_LENGTH = 128  # SPLADE context window limit
CHECKPOINT_FILE = Path("/data/splade_cloud_checkpoint.txt")
PROGRESS_LOG_INTERVAL = 500

# ============================================================================


class Counters:
    """Thread-safe counters."""
    
    def __init__(self):
        self.success = 0
        self.errors = 0
        self.skipped = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def increment(self, field: str, count: int = 1):
        with self.lock:
            setattr(self, field, getattr(self, field) + count)
    
    def get_rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.success / elapsed if elapsed > 0 else 0


def get_checkpoint() -> int:
    """Get last processed offset."""
    if CHECKPOINT_FILE.exists():
        try:
            return int(CHECKPOINT_FILE.read_text().strip())
        except:
            return 0
    return 0


def save_checkpoint(offset: int):
    """Save checkpoint."""
    CHECKPOINT_FILE.write_text(str(offset))


def create_splade_text(payload: Dict[str, Any]) -> str:
    """Create text for SPLADE embedding from payload."""
    title = payload.get("title", "") or ""
    abstract = payload.get("abstract", "") or ""
    
    # For DailyMed drugs
    if payload.get("source") == "dailymed":
        drug_name = payload.get("drug_name", "") or ""
        indications = payload.get("indications", "") or ""
        return f"{drug_name}. {indications}"[:MAX_TEXT_LENGTH * 4]
    
    # For articles
    combined = f"{title}. {abstract}"
    return combined[:MAX_TEXT_LENGTH * 4]  # Allow more, will be truncated by model


def update_batch_with_splade(
    client: QdrantClient,
    point_ids: List[str],
    texts: List[str],
    counters: Counters
) -> bool:
    """Update a batch of points with SPLADE vectors via Cloud Inference."""
    
    for attempt in range(MAX_RETRIES):
        try:
            # Create update vectors with Cloud Inference SPLADE
            point_vectors = [
                PointVectors(
                    id=pid,
                    vector={
                        "sparse": Document(
                            text=text[:MAX_TEXT_LENGTH * 4],
                            model=SPLADE_MODEL
                        )
                    }
                )
                for pid, text in zip(point_ids, texts)
            ]
            
            client.update_vectors(
                collection_name=COLLECTION_NAME,
                points=point_vectors,
                wait=False
            )
            
            counters.increment('success', len(point_ids))
            return True
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES}: {str(e)[:100]}")
                time.sleep(wait_time)
            else:
                logger.error(f"Batch failed: {str(e)[:200]}")
                counters.increment('errors', len(point_ids))
                return False
    
    return False


def has_sparse_vector(record) -> bool:
    """Check if a record already has a sparse vector."""
    # If vectors were fetched, check for sparse
    if hasattr(record, 'vector') and record.vector:
        if isinstance(record.vector, dict) and 'sparse' in record.vector:
            return True
    return False


def run_splade_update(start_offset: int = None, only_new: bool = True):
    """Add SPLADE vectors to points missing them."""
    
    logger.info("=" * 70)
    logger.info("🔬 Adding SPLADE Sparse Vectors via Cloud Inference")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    # Connect with Cloud Inference enabled
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,
        cloud_inference=True
    )
    
    # Get collection info
    info = client.get_collection(COLLECTION_NAME)
    total_points = info.points_count
    
    logger.info(f"✅ Connected to Qdrant")
    logger.info(f"   Collection: {COLLECTION_NAME}")
    logger.info(f"   Total points: {total_points:,}")
    logger.info(f"   SPLADE model: {SPLADE_MODEL}")
    
    # Get checkpoint
    counters = Counters()
    checkpoint = start_offset if start_offset is not None else get_checkpoint()
    logger.info(f"   Starting from offset: {checkpoint:,}")
    
    if only_new:
        # For new documents, we filter by source
        logger.info("   Mode: Only new documents (author_manuscript, pubmed_gov)")
    
    logger.info(f"\n🚀 Starting SPLADE update...")
    logger.info(f"   Batch size: {UPDATE_BATCH_SIZE}")
    logger.info(f"   Parallel workers: {PARALLEL_WORKERS}")
    
    offset = None
    processed = 0
    batch_for_update = []
    batch_ids = []
    batch_texts = []
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        pending_futures = []
        
        with tqdm(total=total_points, initial=checkpoint, desc="Adding SPLADE") as pbar:
            while True:
                # Scroll through points
                try:
                    records, next_offset = client.scroll(
                        collection_name=COLLECTION_NAME,
                        offset=offset,
                        limit=SCROLL_BATCH_SIZE,
                        with_payload=True,
                        with_vectors=False  # Don't fetch vectors to save bandwidth
                    )
                except Exception as e:
                    logger.error(f"Scroll error: {e}")
                    time.sleep(5)
                    continue
                
                if not records:
                    break
                
                for record in records:
                    processed += 1
                    
                    # Filter for new documents only if requested
                    if only_new:
                        source = record.payload.get("source", "")
                        if source not in ["pmc_author_manuscript", "pubmed_gov"]:
                            counters.increment('skipped')
                            continue
                    
                    # Create text for SPLADE
                    text = create_splade_text(record.payload)
                    if len(text) < 20:
                        counters.increment('skipped')
                        continue
                    
                    batch_ids.append(record.id)
                    batch_texts.append(text)
                    
                    # Submit batch when full
                    if len(batch_ids) >= UPDATE_BATCH_SIZE:
                        future = executor.submit(
                            update_batch_with_splade,
                            client,
                            batch_ids.copy(),
                            batch_texts.copy(),
                            counters
                        )
                        pending_futures.append(future)
                        batch_ids = []
                        batch_texts = []
                        
                        # Cleanup completed futures
                        pending_futures = [f for f in pending_futures if not f.done()]
                
                pbar.update(len(records))
                
                # Save checkpoint periodically
                if processed % 1000 == 0:
                    save_checkpoint(processed + checkpoint)
                    
                    if processed % (PROGRESS_LOG_INTERVAL * SCROLL_BATCH_SIZE) == 0:
                        rate = counters.get_rate()
                        logger.info(
                            f"Progress: {counters.success:,} updated, "
                            f"{counters.skipped:,} skipped, "
                            f"{counters.errors:,} errors, "
                            f"{rate:.1f}/sec"
                        )
                
                offset = next_offset
                if offset is None:
                    break
            
            # Final batch
            if batch_ids:
                future = executor.submit(
                    update_batch_with_splade,
                    client,
                    batch_ids,
                    batch_texts,
                    counters
                )
                pending_futures.append(future)
        
        # Wait for pending
        logger.info(f"Waiting for {len(pending_futures)} pending updates...")
        for future in as_completed(pending_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Future error: {e}")
    
    # Save final checkpoint
    save_checkpoint(processed + checkpoint)
    
    # Final stats
    elapsed = time.time() - counters.start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ SPLADE Cloud Inference Update Complete!")
    logger.info("=" * 70)
    logger.info(f"📊 Results:")
    logger.info(f"   Updated: {counters.success:,}")
    logger.info(f"   Skipped: {counters.skipped:,}")
    logger.info(f"   Errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    logger.info(f"   Rate: {counters.get_rate():.1f} points/sec")
    logger.info("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add SPLADE vectors via Cloud Inference")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start from beginning"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all documents, not just new ones"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Start from specific offset"
    )
    
    args = parser.parse_args()
    
    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint reset")
    
    run_splade_update(
        start_offset=args.offset,
        only_new=not args.all
    )


if __name__ == "__main__":
    main()

