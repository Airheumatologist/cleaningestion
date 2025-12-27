#!/usr/bin/env python3
"""
Ingest PubMed Abstracts to Qdrant with Hybrid Vectors.

Ingests high-value PubMed abstracts (Reviews, Meta-Analyses, Practice Guidelines)
with both dense and sparse vectors in a single pass.

Features:
- Dense vectors: mixedbread-ai/mxbai-embed-large-v1 (Cloud Inference)
- Sparse vectors: prithivida/splade_pp_en_v1 (Cloud Inference)
- article_type values compatible with reranker evidence tiers
- PMID-based point IDs for automatic deduplication
- Binary quantization enabled on collection for memory efficiency
- Checkpoint/resume support

Usage:
    python 21_ingest_pubmed_abstracts.py --input /data/pubmed_baseline/filtered/pubmed_abstracts.jsonl
    
    # With limits for testing
    python 21_ingest_pubmed_abstracts.py --input /data/pubmed_baseline/filtered/pubmed_abstracts.jsonl --limit 10000

Expected Duration: 15-20 hours for ~1.8M articles at ~2000 docs/min
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
        logging.FileHandler('/data/pubmed_abstracts_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://cf6c28ca-8a2a-43fa-9424-1f2af9e9a5f3.us-east-1-1.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pmc_medical_rag_fulltext"

# Cloud Inference models (used via Document())
DENSE_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
SPARSE_MODEL = "prithivida/splade_pp_en_v1"

# Ingestion settings
BATCH_SIZE = 50  # Cloud Inference limit
PARALLEL_WORKERS = 4  # Match shard count
MAX_DENSE_TEXT_LENGTH = 2000  # Dense embedding limit
MAX_SPARSE_TEXT_LENGTH = 512  # SPLADE context window
MAX_RETRIES = 3
CHECKPOINT_FILE = Path("/data/pubmed_abstracts_checkpoint.txt")
PROGRESS_LOG_INTERVAL = 500

# Evidence grades mapping based on article_type
EVIDENCE_GRADES = {
    "practice_guideline": "A",
    "meta_analysis": "A",
    "systematic_review": "A",
    "review": "B",
}

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


def create_dense_text(article: Dict[str, Any]) -> str:
    """Create text for dense embedding (max 2000 chars)."""
    title = article.get("title", "") or ""
    abstract = article.get("abstract", "") or ""
    
    combined = f"{title}. {abstract}"
    return combined[:MAX_DENSE_TEXT_LENGTH]


def create_sparse_text(article: Dict[str, Any]) -> str:
    """Create text for SPLADE sparse embedding (max 512 chars)."""
    title = article.get("title", "") or ""
    abstract = article.get("abstract", "") or ""
    
    # For sparse, use title + beginning of abstract
    combined = f"{title}. {abstract}"
    return combined[:MAX_SPARSE_TEXT_LENGTH * 4]  # Model will truncate


def create_payload(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create payload matching existing PMC schema.
    
    Maps article_type values for reranker evidence hierarchy:
    - practice_guideline, meta_analysis, systematic_review → TIER_1 (1.80x)
    - review → TIER_2 (1.25x)
    """
    article_type = article.get("article_type", "review")
    evidence_grade = EVIDENCE_GRADES.get(article_type, "C")
    
    return {
        # Identifiers
        "pmcid": None,  # Abstracts don't have PMCID
        "pmid": article.get("pmid"),
        "doi": article.get("doi"),
        
        # Content
        "title": article.get("title", "")[:300],
        "abstract": article.get("abstract", "")[:2000],
        "full_text": "",  # Abstracts only
        
        # Publication info
        "year": article.get("year"),
        "journal": article.get("journal", "")[:200],
        "article_type": article_type,  # Mapped for reranker tiers
        "publication_type": article.get("publication_types", [])[:5],
        
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
        "source": "pubmed_abstract",  # Distinguishes from pmc_oa, dailymed
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


def run_ingestion(
    jsonl_file: Path,
    limit: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    workers: int = PARALLEL_WORKERS,
    include_sparse: bool = True
):
    """Run the ingestion process with hybrid vectors."""
    
    logger.info("=" * 70)
    logger.info("📚 PubMed Abstracts Ingestion")
    logger.info("   High-Value Articles via Cloud Inference")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    if not jsonl_file.exists():
        logger.error(f"❌ JSONL file not found: {jsonl_file}")
        sys.exit(1)
    
    # Connect to Qdrant with Cloud Inference
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
    logger.info(f"   Dense model: {DENSE_MODEL}")
    logger.info(f"   Sparse model: {SPARSE_MODEL if include_sparse else 'DISABLED'}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Parallel workers: {workers}")
    
    logger.info(f"\n🚀 Starting ingestion...")
    
    current_batch = []
    batch_num = 0
    processed = 0
    
    # Track article type distribution
    type_counts = {}
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
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
                    
                    # Track type distribution
                    at = article.get("article_type", "unknown")
                    type_counts[at] = type_counts.get(at, 0) + 1
                    
                    current_batch.append(article)
                    
                    if len(current_batch) >= batch_size:
                        # Create points with hybrid vectors
                        points = []
                        ids = []
                        
                        for art in current_batch:
                            doc_id = art.get("pmid")
                            if not doc_id:
                                continue
                            
                            dense_text = create_dense_text(art)
                            if len(dense_text) < 50:
                                continue
                            
                            # Create point ID using PMID for deduplication
                            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pmid:{doc_id}"))
                            payload = create_payload(art)
                            
                            # Create hybrid vector with named vectors
                            if include_sparse:
                                sparse_text = create_sparse_text(art)
                                vector = {
                                    "dense": Document(text=dense_text, model=DENSE_MODEL),
                                    "sparse": Document(text=sparse_text, model=SPARSE_MODEL),
                                }
                            else:
                                # Dense only (for faster testing)
                                vector = Document(text=dense_text, model=DENSE_MODEL)
                            
                            point = PointStruct(
                                id=point_id,
                                vector=vector,
                                payload=payload
                            )
                            points.append(point)
                            ids.append(str(doc_id))
                        
                        if points:
                            future = executor.submit(upsert_batch, client, points, ids, counters)
                            pending_futures.append(future)
                        
                        current_batch = []
                        batch_num += 1
                        
                        # Cleanup completed futures
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
                        
                        dense_text = create_dense_text(art)
                        if len(dense_text) < 50:
                            continue
                        
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pmid:{doc_id}"))
                        payload = create_payload(art)
                        
                        if include_sparse:
                            sparse_text = create_sparse_text(art)
                            vector = {
                                "dense": Document(text=dense_text, model=DENSE_MODEL),
                                "sparse": Document(text=sparse_text, model=SPARSE_MODEL),
                            }
                        else:
                            vector = Document(text=dense_text, model=DENSE_MODEL)
                        
                        point = PointStruct(
                            id=point_id,
                            vector=vector,
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
    logger.info("✅ PubMed Abstracts Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"📊 Results:")
    logger.info(f"   Ingested: {counters.success:,}")
    logger.info(f"   Skipped (already in DB): {counters.skipped:,}")
    logger.info(f"   Errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    logger.info(f"   Rate: {counters.get_rate():.1f} docs/sec")
    logger.info(f"   Collection total: {final_count:,}")
    
    logger.info(f"\n📈 Article Type Distribution:")
    for at, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"   {at}: {count:,}")
    
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Ingest PubMed Abstracts to Qdrant")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/pubmed_baseline/filtered/pubmed_abstracts.jsonl"),
        help="JSONL file with filtered abstracts"
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
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Skip SPLADE sparse vectors (faster but less accurate)"
    )
    
    args = parser.parse_args()
    run_ingestion(
        args.input, 
        args.limit, 
        args.batch_size, 
        args.workers,
        include_sparse=not args.no_sparse
    )


if __name__ == "__main__":
    main()
