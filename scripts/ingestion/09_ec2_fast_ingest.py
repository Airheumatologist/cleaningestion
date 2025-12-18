#!/usr/bin/env python3
"""
Fast EC2 Ingestion Script - Dense Vectors Only (Cloud Inference).

This script is optimized to run on EC2 for maximum throughput:
- Uses Qdrant Cloud Inference for embeddings (no local GPU needed)
- Streams data to avoid memory issues
- Parallel upserts to Qdrant (4 shards = 4 parallel streams)
- No SPLADE (can be added later as separate pass)

Run on EC2:
    python3 09_ec2_fast_ingest.py --articles-file /data/pmc_fulltext/pmc_articles.jsonl

Optimal settings based on Qdrant docs:
- Batch size: 1,000-10,000 (we use 5,000)
- Parallel workers: Match shard count (4)
- Network: EC2 to Qdrant Cloud is fast in same region
"""

import os
import sys
import json
import logging
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Check for tqdm
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

# Check for qdrant-client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Document
except ImportError:
    print("Installing qdrant-client...")
    os.system("pip3 install qdrant-client --quiet")
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Update these values!
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://cf6c28ca-8a2a-43fa-9424-1f2af9e9a5f3.us-east-1-1.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # Set via environment variable
COLLECTION_NAME = "pmc_medical_rag_fulltext"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# Optimal batch settings for Cloud Inference
# Cloud Inference has rate limits - smaller batches with retries work best
BATCH_SIZE = 50  # Cloud Inference has strict limits, use smaller batches
PARALLEL_WORKERS = 4  # Fewer workers to avoid overwhelming inference
CHECKPOINT_FILE = Path("/data/ingest_checkpoint.txt")
PROGRESS_LOG_INTERVAL = 50  # Log progress every N batches
MAX_RETRIES = 3  # Retry failed batches

# ============================================================================

# Thread-safe counters
class Counters:
    def __init__(self):
        self.success = 0
        self.errors = 0
        self.skipped = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def increment_success(self, count=1):
        with self.lock:
            self.success += count
    
    def increment_errors(self, count=1):
        with self.lock:
            self.errors += count
    
    def increment_skipped(self, count=1):
        with self.lock:
            self.skipped += count
    
    def get_rate(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.success / elapsed
        return 0

counters = Counters()


def get_checkpoint() -> set:
    """Get set of already ingested IDs."""
    if CHECKPOINT_FILE.exists():
        try:
            return set(CHECKPOINT_FILE.read_text().strip().split('\n'))
        except:
            return set()
    return set()


def save_checkpoint(ids: List[str]):
    """Append IDs to checkpoint file (thread-safe)."""
    checkpoint_lock = threading.Lock()
    with checkpoint_lock:
        with open(CHECKPOINT_FILE, 'a') as f:
            for id_val in ids:
                f.write(f"{id_val}\n")


def create_embedding_text(article: Dict[str, Any]) -> str:
    """Create text for embedding (title + abstract + excerpt of full text).
    
    Note: Keeping text shorter for Cloud Inference stability.
    Max ~2000 chars to stay within inference limits.
    """
    title = article.get("title", "") or ""
    abstract = article.get("abstract", "") or ""
    full_text = article.get("full_text", "") or ""
    
    # Build combined text, prioritizing title and abstract
    combined = f"{title}. {abstract}"
    
    # Add excerpt of full text if available
    if full_text and len(combined) < 1500:
        remaining = 2000 - len(combined)
        combined = f"{combined}\n\n{full_text[:remaining]}"
    
    return combined[:2000]  # Strict limit for Cloud Inference


def create_points_batch(articles: List[Dict[str, Any]]) -> tuple[List[PointStruct], List[str]]:
    """Create PointStruct objects for a batch of articles."""
    points = []
    pmcids = []
    
    for article in articles:
        pmcid = article.get("pmcid") or article.get("pmid") or article.get("set_id")
        if not pmcid:
            continue
        
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(pmcid)))
        embedding_text = create_embedding_text(article)
        
        if not embedding_text.strip():
            continue
        
        # Prepare payload with metadata for retrieval and reranking
        payload = {
            # Identifiers
            "pmcid": article.get("pmcid"),
            "pmid": article.get("pmid"),
            "doi": article.get("doi"),
            
            # Content
            "title": article.get("title", "")[:300],
            "abstract": article.get("abstract", "")[:1000],
            
            # Publication info
            "year": article.get("year"),
            "journal": article.get("journal", ""),
            "article_type": article.get("article_type", ""),
            "publication_type": article.get("publication_type_list", [])[:5],  # For evidence hierarchy
            
            # Evidence & quality signals for reranking
            "evidence_grade": article.get("evidence_grade"),  # A, B, C, D
            "country": article.get("country"),  # Country of publication for reranking
            "institutions": article.get("institutions", [])[:5],  # Research institutions
            
            # Subject classification
            "keywords": article.get("keywords", [])[:10],
            "mesh_terms": article.get("mesh_terms", [])[:15],
            
            # Authorship
            "authors": article.get("authors", [])[:5],
            "first_author": article.get("first_author"),
            "author_count": article.get("author_count", len(article.get("authors", []))),
            
            # Structure signals
            "source": article.get("source", "pmc"),
            "has_full_text": bool(article.get("full_text")),
            "has_methods": article.get("has_methods", False),
            "has_results": article.get("has_results", False),
            "table_count": len(article.get("tables", [])),
            "figure_count": article.get("figure_count", 0),
        }
        
        # Create point with Cloud Inference document
        point = PointStruct(
            id=point_id,
            vector=Document(
                text=embedding_text,
                model=EMBEDDING_MODEL,
            ),
            payload=payload
        )
        
        points.append(point)
        pmcids.append(pmcid)
    
    return points, pmcids


def upsert_batch(client: QdrantClient, points: List[PointStruct], pmcids: List[str]) -> bool:
    """Upsert a batch of points to Qdrant with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=False  # Don't wait for indexing (faster)
            )
            save_checkpoint(pmcids)
            counters.increment_success(len(points))
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} after {sleep_time}s: {str(e)[:100]}")
                time.sleep(sleep_time)
            else:
                logger.error(f"Batch upsert failed after {MAX_RETRIES} attempts: {str(e)[:200]}")
                counters.increment_errors(len(points))
                return False
    return False


def run_ingestion(articles_file: Path):
    """Run the ingestion process."""
    
    logger.info("=" * 70)
    logger.info("🚀 Fast EC2 Ingestion with Cloud Inference")
    logger.info("=" * 70)
    
    # Check API key
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY environment variable not set!")
        logger.error("   Run: export QDRANT_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,  # Long timeout for large batches
        cloud_inference=True
    )
    
    # Verify connection
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to Qdrant: {QDRANT_URL}")
        logger.info(f"   Collection: {COLLECTION_NAME}")
        logger.info(f"   Current points: {collection_info.points_count:,}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        sys.exit(1)
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"   Batch size: {BATCH_SIZE:,}")
    logger.info(f"   Parallel workers: {PARALLEL_WORKERS}")
    
    # Get checkpoint
    ingested = get_checkpoint()
    logger.info(f"   Already ingested: {len(ingested):,}")
    
    # Count total articles
    logger.info(f"\n📤 Counting articles in {articles_file}...")
    total_articles = sum(1 for line in open(articles_file) if line.strip())
    logger.info(f"   Total articles: {total_articles:,}")
    
    # Process in streaming batches
    logger.info(f"\n🚀 Starting ingestion...")
    
    current_batch = []
    batch_num = 0
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        pending_futures = []
        
        with open(articles_file, 'r') as f:
            with tqdm(total=total_articles, desc="Ingesting", unit="article") as pbar:
                for line in f:
                    if not line.strip():
                        pbar.update(1)
                        continue
                    
                    try:
                        article = json.loads(line)
                        pmcid = article.get("pmcid") or article.get("pmid") or article.get("set_id")
                        
                        if pmcid and str(pmcid) not in ingested:
                            current_batch.append(article)
                        else:
                            counters.increment_skipped(1)
                        
                        pbar.update(1)
                        
                        # When batch is full, submit for processing
                        if len(current_batch) >= BATCH_SIZE:
                            # Create points
                            points, pmcids = create_points_batch(current_batch)
                            
                            if points:
                                # Submit upsert
                                future = executor.submit(upsert_batch, client, points, pmcids)
                                pending_futures.append(future)
                            
                            current_batch = []
                            batch_num += 1
                            
                            # Clean up completed futures
                            pending_futures = [f for f in pending_futures if not f.done()]
                            
                            # Progress logging
                            if batch_num % PROGRESS_LOG_INTERVAL == 0:
                                rate = counters.get_rate()
                                logger.info(
                                    f"Progress: {counters.success:,} ingested, "
                                    f"{counters.errors:,} errors, "
                                    f"{rate:.1f} articles/sec"
                                )
                    
                    except json.JSONDecodeError:
                        continue
                
                # Process remaining batch
                if current_batch:
                    points, pmcids = create_points_batch(current_batch)
                    if points:
                        future = executor.submit(upsert_batch, client, points, pmcids)
                        pending_futures.append(future)
        
        # Wait for all pending futures
        logger.info(f"Waiting for {len(pending_futures)} pending uploads...")
        for future in as_completed(pending_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in pending upload: {e}")
    
    # Final stats
    elapsed = time.time() - counters.start_time
    rate = counters.success / elapsed if elapsed > 0 else 0
    
    collection_info = client.get_collection(COLLECTION_NAME)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"📊 Results:")
    logger.info(f"   Ingested: {counters.success:,}")
    logger.info(f"   Skipped: {counters.skipped:,}")
    logger.info(f"   Errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed/3600:.1f} hours")
    logger.info(f"   Rate: {rate:.1f} articles/sec")
    logger.info(f"   Collection points: {collection_info.points_count:,}")


def main():
    global BATCH_SIZE, PARALLEL_WORKERS
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast EC2 Ingestion with Cloud Inference")
    parser.add_argument("--articles-file", type=Path, required=True, help="JSONL file with articles")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--workers", type=int, default=PARALLEL_WORKERS, help="Parallel workers")
    
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch_size
    PARALLEL_WORKERS = args.workers
    
    if not args.articles_file.exists():
        logger.error(f"Articles file not found: {args.articles_file}")
        sys.exit(1)
    
    run_ingestion(args.articles_file)


if __name__ == "__main__":
    main()

