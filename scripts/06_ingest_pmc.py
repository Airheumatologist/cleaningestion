#!/usr/bin/env python3
"""
Ingest PMC Articles to Qdrant.

Fast ingestion script using Qdrant Cloud Inference for embeddings.
Optimized for EC2 to Qdrant Cloud throughput.

Features:
- Cloud Inference for embeddings (no local GPU needed)
- Streaming ingestion for memory efficiency
- Checkpoint/resume support
- Parallel upserts matching shard count
- Retry logic with exponential backoff

Optimal Settings:
- Batch size: 50 (Cloud Inference limit)
- Parallel workers: 4 (matches shard count)
- Max text length: 2000 chars (Cloud Inference limit)

Usage:
    python 06_ingest_pmc.py --articles-file /data/pmc_articles.jsonl

Expected Duration: ~60 minutes for 1.2M articles
"""

import os
import sys
import json
import logging
import uuid
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Document
except ImportError:
    os.system("pip3 install qdrant-client --quiet")
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Document

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pmc_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pmc_medical_rag_fulltext")
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# Optimal settings for Cloud Inference
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_TEXT_LENGTH = 2000  # Cloud Inference limit
CHECKPOINT_FILE = Path("pmc_ingest_checkpoint.txt")
PROGRESS_LOG_INTERVAL = 50

# =============================================================================


class Counters:
    """Thread-safe counters for progress tracking."""
    
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
        return self.success / elapsed if elapsed > 0 else 0


def get_checkpoint() -> set:
    """Load checkpoint of already ingested IDs."""
    if CHECKPOINT_FILE.exists():
        try:
            return set(CHECKPOINT_FILE.read_text().strip().split('\n'))
        except:
            return set()
    return set()


def save_checkpoint(ids: List[str]):
    """Append IDs to checkpoint file."""
    with open(CHECKPOINT_FILE, 'a') as f:
        for id_val in ids:
            f.write(f"{id_val}\n")


def create_embedding_text(article: Dict[str, Any]) -> str:
    """
    Create text for embedding from article data.
    
    Includes title + abstract + beginning of full text,
    limited to 2000 chars for Cloud Inference.
    """
    title = article.get("title", "") or ""
    abstract = article.get("abstract", "") or ""
    full_text = article.get("full_text", "") or ""
    
    # Combine title + abstract
    combined = f"{title}. {abstract}"
    
    # Add as much full text as fits within limit
    if full_text and len(combined) < MAX_TEXT_LENGTH - 100:
        remaining = MAX_TEXT_LENGTH - len(combined) - 10
        combined = f"{combined}\n\n{full_text[:remaining]}"
    
    return combined[:MAX_TEXT_LENGTH]


def create_payload(article: Dict[str, Any]) -> Dict[str, Any]:
    """Create payload with metadata for reranking."""
    return {
        # Identifiers
        "pmcid": article.get("pmcid"),
        "pmid": article.get("pmid"),
        "doi": article.get("doi"),
        
        # Content (truncated for payload size)
        "title": article.get("title", "")[:300],
        "abstract": article.get("abstract", "")[:1000],
        "full_text": article.get("full_text", "")[:10000],  # Store for RAG context
        
        # Publication info
        "year": article.get("year"),
        "journal": article.get("journal", ""),
        "article_type": article.get("article_type", ""),
        "publication_type": article.get("publication_type_list", [])[:5],
        
        # Evidence signals for reranking
        "evidence_grade": article.get("evidence_grade"),
        "evidence_level": article.get("evidence_level"),
        "country": article.get("country"),
        "institutions": article.get("institutions", [])[:5],
        
        # Subject classification
        "keywords": article.get("keywords", [])[:10],
        "mesh_terms": article.get("mesh_terms", [])[:15],
        
        # Authors
        "authors": article.get("authors", [])[:5],
        "first_author": article.get("first_author"),
        "author_count": article.get("author_count", 0),
        
        # Structure signals
        "source": article.get("source", "pmc"),
        "has_full_text": article.get("has_full_text", False),
        "has_methods": article.get("has_methods", False),
        "has_results": article.get("has_results", False),
        "table_count": article.get("table_count", 0),
        "figure_count": article.get("figure_count", 0),
    }


def create_points_batch(articles: List[Dict[str, Any]]) -> tuple:
    """Create PointStruct objects for a batch."""
    points = []
    ids = []
    
    for article in articles:
        doc_id = article.get("pmcid") or article.get("pmid")
        if not doc_id:
            continue
        
        embedding_text = create_embedding_text(article)
        if not embedding_text.strip() or len(embedding_text) < 20:
            continue
        
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))
        payload = create_payload(article)
        
        point = PointStruct(
            id=point_id,
            vector=Document(text=embedding_text, model=EMBEDDING_MODEL),
            payload=payload
        )
        
        points.append(point)
        ids.append(str(doc_id))
    
    return points, ids


def upsert_batch(client: QdrantClient, points: List[PointStruct], ids: List[str], counters: Counters) -> bool:
    """Upsert batch with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=False
            )
            save_checkpoint(ids)
            counters.increment_success(len(points))
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} after {sleep_time}s: {str(e)[:100]}")
                time.sleep(sleep_time)
            else:
                logger.error(f"Batch failed: {str(e)[:200]}")
                counters.increment_errors(len(points))
                return False
    return False


def run_ingestion(articles_file: Path):
    """Run the ingestion process."""
    
    logger.info("=" * 70)
    logger.info("🚀 PMC Article Ingestion with Cloud Inference")
    logger.info("=" * 70)
    
    # Validate configuration
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY environment variable not set!")
        sys.exit(1)
    
    if not articles_file.exists():
        logger.error(f"❌ Articles file not found: {articles_file}")
        sys.exit(1)
    
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,
        cloud_inference=True
    )
    
    # Verify connection
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to Qdrant: {QDRANT_URL}")
        logger.info(f"   Collection: {COLLECTION_NAME}")
        logger.info(f"   Current points: {info.points_count:,}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Parallel workers: {PARALLEL_WORKERS}")
    
    # Get checkpoint
    counters = Counters()
    ingested = get_checkpoint()
    logger.info(f"   Already ingested: {len(ingested):,}")
    
    # Count total articles
    logger.info(f"\n📤 Counting articles...")
    total = sum(1 for line in open(articles_file) if line.strip())
    logger.info(f"   Total articles: {total:,}")
    
    # Process
    logger.info(f"\n🚀 Starting ingestion...")
    
    current_batch = []
    batch_num = 0
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        pending = []
        
        with open(articles_file, 'r') as f:
            with tqdm(total=total, desc="Ingesting", unit="article") as pbar:
                for line in f:
                    if not line.strip():
                        pbar.update(1)
                        continue
                    
                    try:
                        article = json.loads(line)
                        doc_id = article.get("pmcid") or article.get("pmid")
                        
                        if doc_id and str(doc_id) not in ingested:
                            current_batch.append(article)
                        else:
                            counters.increment_skipped(1)
                        
                        pbar.update(1)
                        
                        if len(current_batch) >= BATCH_SIZE:
                            points, ids = create_points_batch(current_batch)
                            if points:
                                future = executor.submit(upsert_batch, client, points, ids, counters)
                                pending.append(future)
                            
                            current_batch = []
                            batch_num += 1
                            pending = [f for f in pending if not f.done()]
                            
                            if batch_num % PROGRESS_LOG_INTERVAL == 0:
                                rate = counters.get_rate()
                                logger.info(
                                    f"Progress: {counters.success:,} ingested, "
                                    f"{counters.errors:,} errors, {rate:.1f}/sec"
                                )
                    
                    except json.JSONDecodeError:
                        continue
                
                # Final batch
                if current_batch:
                    points, ids = create_points_batch(current_batch)
                    if points:
                        future = executor.submit(upsert_batch, client, points, ids, counters)
                        pending.append(future)
        
        # Wait for pending
        logger.info(f"Waiting for {len(pending)} pending uploads...")
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error: {e}")
    
    # Final stats
    elapsed = time.time() - counters.start_time
    info = client.get_collection(COLLECTION_NAME)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ PMC Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"📊 Results:")
    logger.info(f"   Ingested: {counters.success:,}")
    logger.info(f"   Skipped: {counters.skipped:,}")
    logger.info(f"   Errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes")
    logger.info(f"   Rate: {counters.get_rate():.1f} articles/sec")
    logger.info(f"   Collection total: {info.points_count:,}")


def main():
    parser = argparse.ArgumentParser(description="Ingest PMC articles to Qdrant")
    parser.add_argument("--articles-file", type=Path, required=True, help="JSONL file with articles")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--workers", type=int, default=PARALLEL_WORKERS, help="Parallel workers")
    
    args = parser.parse_args()
    
    global BATCH_SIZE, PARALLEL_WORKERS
    BATCH_SIZE = args.batch_size
    PARALLEL_WORKERS = args.workers
    
    run_ingestion(args.articles_file)


if __name__ == "__main__":
    main()

