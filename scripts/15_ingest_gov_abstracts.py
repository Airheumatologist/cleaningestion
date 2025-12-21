#!/usr/bin/env python3
"""
Ingest Government PubMed Abstracts to Qdrant.

Ingests public domain abstracts from US government-authored articles
(NIH, CDC, FDA, etc.) into the existing Qdrant collection.

Features:
- Dense vectors via Qdrant Cloud Inference
- All metadata for filtering and reranking
- content_type='abstract' for filtering
- source='pubmed_gov' for identification

Usage:
    export QDRANT_API_KEY="your_key"
    python 15_ingest_gov_abstracts.py --jsonl-file /data/pubmed_gov/gov_abstracts/gov_abstracts.jsonl

Expected Duration: 4-6 hours for ~3.5M abstracts
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
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/gov_abstracts_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://cf6c28ca-8a2a-43fa-9424-1f2af9e9a5f3.us-east-1-1.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pmc_medical_rag_fulltext"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# Ingestion settings
BATCH_SIZE = 50  # Cloud Inference limit
PARALLEL_WORKERS = 4  # Match shard count
MAX_TEXT_LENGTH = 2000
MAX_RETRIES = 3
CHECKPOINT_FILE = Path("/data/gov_abstracts_ingest_checkpoint.txt")
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


def get_checkpoint() -> set:
    """Load checkpoint of ingested PMIDs."""
    if CHECKPOINT_FILE.exists():
        try:
            return set(CHECKPOINT_FILE.read_text().strip().split('\n'))
        except:
            return set()
    return set()


def save_checkpoint(ids: List[str]):
    """Append IDs to checkpoint."""
    with open(CHECKPOINT_FILE, 'a') as f:
        for id_val in ids:
            f.write(f"{id_val}\n")


def create_embedding_text(article: Dict[str, Any]) -> str:
    """Create text for embedding from abstract data."""
    title = article.get("title", "") or ""
    abstract = article.get("abstract", "") or ""
    
    combined = f"{title}. {abstract}"
    return combined[:MAX_TEXT_LENGTH]


def detect_evidence_grade(abstract: str) -> str:
    """Detect evidence grade from abstract content."""
    abstract_lower = abstract.lower() if abstract else ""
    
    # Look for evidence indicators
    if any(term in abstract_lower for term in ["meta-analysis", "systematic review", "cochrane"]):
        return "A"
    if any(term in abstract_lower for term in ["randomized", "randomised", "rct", "clinical trial"]):
        return "B"
    if any(term in abstract_lower for term in ["cohort", "prospective", "longitudinal"]):
        return "C"
    if any(term in abstract_lower for term in ["case report", "case series"]):
        return "D"
    
    return "C"  # Default


def create_payload(article: Dict[str, Any]) -> Dict[str, Any]:
    """Create payload matching existing PMC schema."""
    abstract = article.get("abstract", "") or ""
    evidence_grade = detect_evidence_grade(abstract)
    
    return {
        # Identifiers
        "pmcid": None,  # Abstracts don't have PMCID
        "pmid": article.get("pmid"),
        "doi": article.get("doi"),
        
        # Content
        "title": article.get("title", "")[:300],
        "abstract": abstract[:2000],
        "full_text": "",  # Abstracts only
        
        # Publication info
        "year": article.get("year"),
        "journal": article.get("journal", "")[:200],
        "article_type": "research_article",
        "publication_type": [],
        
        # Evidence signals
        "evidence_grade": evidence_grade,
        "evidence_level": {"A": 1, "B": 2, "C": 3, "D": 4}.get(evidence_grade, 3),
        "country": None,
        "institutions": article.get("affiliations", [])[:3],
        
        # Subject classification
        "keywords": [],
        "mesh_terms": article.get("mesh_terms", [])[:15],
        
        # Authors
        "authors": article.get("authors", [])[:5],
        "first_author": article.get("authors", [None])[0] if article.get("authors") else None,
        "author_count": len(article.get("authors", [])),
        
        # Structure signals
        "source": "pubmed_gov",
        "content_type": "abstract",
        "has_full_text": False,
        "has_methods": False,
        "has_results": False,
        "table_count": 0,
        "figure_count": 0,
    }


def upsert_batch(
    client: QdrantClient,
    points: List[PointStruct],
    ids: List[str],
    counters: Counters
) -> bool:
    """Upsert batch with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=False
            )
            save_checkpoint(ids)
            counters.increment('success', len(points))
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES}: {str(e)[:100]}")
                time.sleep(wait_time)
            else:
                logger.error(f"Batch failed: {str(e)[:200]}")
                counters.increment('errors', len(points))
                return False
    return False


def run_ingestion(jsonl_file: Path, limit: Optional[int] = None, batch_size: int = 50, workers: int = 4):
    """Run the ingestion process."""
    
    global BATCH_SIZE, PARALLEL_WORKERS
    BATCH_SIZE = batch_size
    PARALLEL_WORKERS = workers
    
    logger.info("=" * 70)
    logger.info("🏛️  Government Abstracts Ingestion")
    logger.info("   Public Domain Content via Cloud Inference")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    if not jsonl_file.exists():
        logger.error(f"❌ JSONL file not found: {jsonl_file}")
        sys.exit(1)
    
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,
        cloud_inference=True
    )
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to Qdrant")
        logger.info(f"   Collection: {COLLECTION_NAME}")
        logger.info(f"   Current points: {info.points_count:,}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    # Get checkpoint
    counters = Counters()
    ingested = get_checkpoint()
    logger.info(f"   Already ingested: {len(ingested):,}")
    
    # Count total
    logger.info(f"\n📂 Counting articles in: {jsonl_file}")
    total = sum(1 for _ in open(jsonl_file))
    logger.info(f"   Total articles: {total:,}")
    
    if limit:
        logger.info(f"   Limited to: {limit:,}")
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Parallel workers: {PARALLEL_WORKERS}")
    
    logger.info(f"\n🚀 Starting ingestion...")
    
    current_batch = []
    batch_num = 0
    processed = 0
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        pending_futures = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            with tqdm(total=min(total, limit) if limit else total, desc="Ingesting", unit="doc") as pbar:
                for line in f:
                    if limit and processed >= limit:
                        break
                    
                    processed += 1
                    pbar.update(1)
                    
                    if not line.strip():
                        continue
                    
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    pmid = article.get("pmid")
                    if not pmid:
                        continue
                    
                    if str(pmid) in ingested:
                        counters.increment('skipped')
                        continue
                    
                    current_batch.append(article)
                    
                    if len(current_batch) >= BATCH_SIZE:
                        # Create points
                        points = []
                        ids = []
                        
                        for art in current_batch:
                            doc_id = art.get("pmid")
                            if not doc_id:
                                continue
                            
                            embedding_text = create_embedding_text(art)
                            if len(embedding_text) < 50:
                                continue
                            
                            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pmid:{doc_id}"))
                            payload = create_payload(art)
                            
                            point = PointStruct(
                                id=point_id,
                                vector=Document(text=embedding_text, model=EMBEDDING_MODEL),
                                payload=payload
                            )
                            points.append(point)
                            ids.append(str(doc_id))
                        
                        if points:
                            future = executor.submit(upsert_batch, client, points, ids, counters)
                            pending_futures.append(future)
                        
                        current_batch = []
                        batch_num += 1
                        
                        # Cleanup completed
                        pending_futures = [f for f in pending_futures if not f.done()]
                        
                        if batch_num % PROGRESS_LOG_INTERVAL == 0:
                            rate = counters.get_rate()
                            logger.info(
                                f"Progress: {counters.success:,} ingested, "
                                f"{counters.errors:,} errors, "
                                f"{rate:.1f}/sec"
                            )
                
                # Final batch
                if current_batch:
                    points = []
                    ids = []
                    
                    for art in current_batch:
                        doc_id = art.get("pmid")
                        if not doc_id:
                            continue
                        
                        embedding_text = create_embedding_text(art)
                        if len(embedding_text) < 50:
                            continue
                        
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pmid:{doc_id}"))
                        payload = create_payload(art)
                        
                        point = PointStruct(
                            id=point_id,
                            vector=Document(text=embedding_text, model=EMBEDDING_MODEL),
                            payload=payload
                        )
                        points.append(point)
                        ids.append(str(doc_id))
                    
                    if points:
                        future = executor.submit(upsert_batch, client, points, ids, counters)
                        pending_futures.append(future)
        
        # Wait for pending
        logger.info(f"Waiting for {len(pending_futures)} pending uploads...")
        for future in as_completed(pending_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Future error: {e}")
    
    # Final stats
    elapsed = time.time() - counters.start_time
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        final_count = info.points_count
    except:
        final_count = "N/A"
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Government Abstracts Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"📊 Results:")
    logger.info(f"   Ingested: {counters.success:,}")
    logger.info(f"   Skipped: {counters.skipped:,}")
    logger.info(f"   Errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    logger.info(f"   Rate: {counters.get_rate():.1f} docs/sec")
    logger.info(f"   Collection total: {final_count:,}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Ingest Government Abstracts to Qdrant")
    parser.add_argument(
        "--jsonl-file",
        type=Path,
        default=Path("/data/pubmed_gov/gov_abstracts/gov_abstracts.jsonl"),
        help="JSONL file with government abstracts"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number to process (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=PARALLEL_WORKERS,
        help="Parallel workers"
    )
    
    args = parser.parse_args()
    run_ingestion(args.jsonl_file, args.limit, args.batch_size, args.workers)


if __name__ == "__main__":
    main()

