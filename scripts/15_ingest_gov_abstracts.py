#!/usr/bin/env python3
"""
Ingest Government PubMed Abstracts to Qdrant.

Ingests public domain abstracts from US government-authored articles
(NIH, CDC, FDA, etc.) into the existing Qdrant collection.

Features:
- Dense vectors via DeepInfra API (Qwen/Qwen3-Embedding-0.6B-batch)
- BM25 sparse vectors for hybrid search
- Token-based chunking (2048 tokens)
- All metadata for filtering and reranking
- content_type='abstract' for filtering
- source='pubmed_gov' for identification

Usage:
    export DEEPINFRA_API_KEY="your_key"
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
import hashlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for shared config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import Chunker as BaseChunker, EmbeddingProvider, upsert_with_retry

# Import enhanced utilities for semantic chunking and validation
try:
    from ingestion_utils_enhanced import SemanticChunker, QualityValidator, ContentDeduplicator
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False

# Import BM25SparseEncoder
spec = importlib.util.find_spec("src.bm25_sparse")
if spec is not None:
    from src.bm25_sparse import BM25SparseEncoder
else:
    BM25SparseEncoder = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
except ImportError:
    os.system("pip3 install qdrant-client --quiet")
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

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
# CONFIGURATION - Uses centralized IngestionConfig
# ============================================================================
QDRANT_URL = IngestionConfig.QDRANT_URL
QDRANT_API_KEY = IngestionConfig.QDRANT_API_KEY or os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = IngestionConfig.COLLECTION_NAME

# Use centralized settings with fallback defaults
BATCH_SIZE = IngestionConfig.BATCH_SIZE
PARALLEL_WORKERS = IngestionConfig.MAX_WORKERS  # Standardized to use config value
MAX_RETRIES = IngestionConfig.MAX_RETRIES
CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "gov_abstracts_ingested_ids.txt"
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


def load_checkpoint() -> set[str]:
    """Load checkpoint of ingested PMIDs."""
    if CHECKPOINT_FILE.exists():
        try:
            return set(CHECKPOINT_FILE.read_text().strip().split('\n'))
        except:
            return set()
    return set()


def append_checkpoint(ids: List[str]) -> None:
    """Append IDs to checkpoint."""
    with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
        for id_val in ids:
            f.write(f"{id_val}\n")


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


# generate_section_id is imported from ingestion_utils


def create_chunks_from_article(article: Dict[str, Any], chunker: Any, validate_chunks: bool = True) -> List[Dict[str, Any]]:
    """
    Create chunks from article with token-aware chunking.
    For gov abstracts, we chunk the title + abstract.
    
    Args:
        article: The parsed article dictionary
        chunker: The chunker instance (BaseChunker or SemanticChunker)
        validate_chunks: Whether to validate chunk quality before returning
    """
    chunks = []
    
    pmid = str(article.get("pmid", ""))
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    
    if not title and not abstract:
        return []
    
    # Create full content text
    full_text = f"Title: {title}\n\nAbstract: {abstract}" if title or abstract else ""
    
    if len(full_text) < 50:
        return []
    
    # Base metadata for all chunks
    base_metadata = {
        "title": title,
        "abstract": abstract,
        "journal": article.get("journal", ""),
        "year": article.get("year"),
        "pmid": pmid,
        "doi": article.get("doi"),
        "mesh_terms": article.get("mesh_terms", []),
        "authors": article.get("authors", []),
    }
    
    # Generate section ID for abstract
    section_id = generate_section_id(pmid, "Abstract")
    
    # Chunk the content
    text_chunks = chunker.chunk_text(full_text)
    
    for j, chunk_data in enumerate(text_chunks):
        chunk_id = f"{pmid}_abstract_part{j}" if len(text_chunks) > 1 else f"{pmid}_abstract"
        
        # Extract chunk abstract (remove title prefix if present)
        chunk_text = chunk_data["text"]
        chunk_abstract = chunk_text
        if chunk_text.startswith(f"Title: {title}\n\nAbstract: "):
            chunk_abstract = chunk_text[len(f"Title: {title}\n\nAbstract: "):]
        elif chunk_text.startswith("Title: "):
            # Fallback for continuation chunks
            chunk_abstract = chunk_text[chunk_text.find("\n\nAbstract: ") + 12:]
        
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": pmid,
            "text": chunk_text,
            "abstract": chunk_abstract,
            "section_type": "abstract",
            "section_title": f"Abstract (Part {j+1}/{len(text_chunks)})" if len(text_chunks) > 1 else "Abstract",
            "full_section_text": full_text,
            "section_id": section_id,
            "section_weight": 1.0 - (j * 0.05),  # Slight decay for later chunks
            "chunk_index": j,
            "total_chunks": len(text_chunks),
            **base_metadata,
        })
    
    # Validate chunks if enabled
    if validate_chunks and ENHANCED_UTILS_AVAILABLE:
        valid_chunks = []
        for chunk in chunks:
            is_valid, issues = QualityValidator.validate_chunk(
                chunk["text"],
                {k: chunk.get(k) for k in ["doc_id", "chunk_id", "title", "pmid", "year"]}
            )
            if is_valid:
                valid_chunks.append(chunk)
            else:
                logger.debug("Skipping invalid chunk %s: %s", chunk.get("chunk_id"), issues)
        chunks = valid_chunks
        if len(valid_chunks) < len(chunks):
            logger.debug("Validated chunks: %d valid out of %d total", len(valid_chunks), len(chunks))
    
    return chunks


def build_points(
    articles: List[Dict[str, Any]], 
    embedding_provider: EmbeddingProvider,
    chunker: Chunker,
    sparse_encoder: Optional[BM25SparseEncoder]
) -> Tuple[List[PointStruct], List[str]]:
    """
    Build Qdrant points from articles with chunking and embeddings.
    """
    points: List[PointStruct] = []
    doc_ids: List[str] = []
    
    # Collect all chunks for batch embedding
    all_chunks: List[Dict[str, Any]] = []
    
    for article in articles:
        chunks = create_chunks_from_article(article, chunker)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        return [], []
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in all_chunks]
    
    # Generate embeddings
    try:
        vectors = embedding_provider.embed_batch(texts)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return [], []
    
    # Create points
    for chunk, vector in zip(all_chunks, vectors):
        chunk_id = chunk["chunk_id"]
        doc_id = chunk["doc_id"]
        
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
        
        # Create payload
        payload = {
            # Core identifiers
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "pmid": doc_id,
            "pmcid": None,
            "doi": chunk.get("doi"),
            
            # CRITICAL: Full text for retriever
            "page_content": chunk["text"],
            
            # Article metadata
            "title": chunk.get("title", ""),
            "abstract": chunk.get("abstract", ""),
            "journal": chunk.get("journal", ""),
            "year": chunk.get("year"),
            "mesh_terms": chunk.get("mesh_terms", []),
            "authors": chunk.get("authors", []),
            "first_author": chunk.get("authors", [None])[0] if chunk.get("authors") else None,
            "author_count": len(chunk.get("authors", [])),
            
            # Section information
            "section_type": chunk["section_type"],
            "section_title": chunk["section_title"],
            
            # Parent-child indexing fields
            "full_section_text": chunk.get("full_section_text", chunk["text"]),
            "section_id": chunk.get("section_id", ""),
            "section_weight": chunk.get("section_weight", 1.0),
            
            # Chunking metadata
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 1),
            
            # Source and type
            "source": "pubmed_gov",
            "article_type": "research_article",
            "content_type": "abstract",
            "has_full_text": False,
            "has_methods": False,
            "has_results": False,
            "table_count": 0,
            "figure_count": 0,
            
            # Evidence
            "evidence_grade": detect_evidence_grade(chunk.get("abstract", "")),
            "evidence_level": {"A": 1, "B": 2, "C": 3, "D": 4}.get(
                detect_evidence_grade(chunk.get("abstract", "")), 3
            ),
            
            # Preview for dashboard
            "text_preview": chunk["text"][:500],
        }
        
        # Create vector data
        vector_data: Any = {"dense": vector}
        if sparse_encoder is not None:
            sparse_vector = sparse_encoder.encode_document(chunk["text"])
            vector_data["sparse"] = sparse_vector
        
        # Create deterministic point ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"gov:{chunk_id}"))
        
        points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))
    
    return points, doc_ids


def upsert_batch(
    client: QdrantClient,
    points: List[PointStruct],
    ids: List[str],
    counters: Counters
) -> bool:
    """Upsert batch with retry logic."""
    try:
        upsert_with_retry(client, points)
        append_checkpoint(ids)
        counters.increment('success', len(points))
        return True
    except Exception as e:
        logger.error(f"Batch failed: {str(e)[:200]}")
        counters.increment('errors', len(points))
        return False


def run_ingestion(jsonl_file: Path, limit: Optional[int] = None, batch_size: int = 50, workers: int = 4):
    """Run the ingestion process with DeepInfra embeddings."""
    
    global BATCH_SIZE, PARALLEL_WORKERS
    BATCH_SIZE = batch_size
    PARALLEL_WORKERS = workers
    
    logger.info("=" * 70)
    logger.info("🏛️  Government Abstracts Ingestion")
    logger.info("   Dense Vectors via DeepInfra API")
    logger.info("   BM25 Sparse Vectors for Hybrid Search")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    if not os.getenv("DEEPINFRA_API_KEY"):
        logger.error("❌ DEEPINFRA_API_KEY not set!")
        logger.info("   Set: export DEEPINFRA_API_KEY='your_key'")
        sys.exit(1)
    
    if not jsonl_file.exists():
        logger.error(f"❌ JSONL file not found: {jsonl_file}")
        sys.exit(1)
    
    # Initialize embedding provider
    embedding_provider = EmbeddingProvider()
    
    # Initialize chunker (uses SemanticChunker if available)
    ChunkerClass = SemanticChunker if ENHANCED_UTILS_AVAILABLE else BaseChunker
    chunker = ChunkerClass(
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS
    )
    logger.info(f"Using {ChunkerClass.__name__} for chunking")
    
    # Initialize sparse encoder
    sparse_encoder = None
    if IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
        if BM25SparseEncoder is not None:
            sparse_encoder = BM25SparseEncoder(
                max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
                max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
                min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
                remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
            )
            logger.info("✅ BM25 sparse encoder initialized")
        else:
            logger.warning("⚠️ BM25SparseEncoder not available")
    
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,
        prefer_grpc=IngestionConfig.USE_GRPC,
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
    ingested = load_checkpoint()
    logger.info(f"   Already ingested: {len(ingested):,}")
    
    # Count total
    logger.info(f"\n📂 Counting articles in: {jsonl_file}")
    total = sum(1 for _ in open(jsonl_file))
    logger.info(f"   Total articles: {total:,}")
    
    if limit:
        logger.info(f"   Limited to: {limit:,}")
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Embedding provider: DeepInfra")
    logger.info(f"   Embedding model: {IngestionConfig.EMBEDDING_MODEL}")
    logger.info(f"   Chunk size: {IngestionConfig.CHUNK_SIZE_TOKENS} tokens")
    logger.info(f"   Chunk overlap: {IngestionConfig.CHUNK_OVERLAP_TOKENS} tokens")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Parallel workers: {PARALLEL_WORKERS}")
    logger.info(f"   Data directory: {IngestionConfig.DATA_DIR}")
    
    logger.info(f"\n🚀 Starting ingestion...")
    
    current_batch: List[Dict[str, Any]] = []
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
                        # Build points with embeddings
                        points, ids = build_points(current_batch, embedding_provider, chunker, sparse_encoder)
                        
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
                                f"Progress: {counters.success:,} points ingested, "
                                f"{counters.errors:,} errors, "
                                f"{rate:.1f} pts/sec"
                            )
                
                # Final batch
                if current_batch:
                    points, ids = build_points(current_batch, embedding_provider, chunker, sparse_encoder)
                    
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
    logger.info(f"   Ingested: {counters.success:,} points")
    logger.info(f"   Skipped: {counters.skipped:,}")
    logger.info(f"   Errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    logger.info(f"   Rate: {counters.get_rate():.1f} pts/sec")
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
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=PARALLEL_WORKERS,
        help=f"Parallel workers (default: {PARALLEL_WORKERS})"
    )
    
    args = parser.parse_args()
    run_ingestion(args.jsonl_file, args.limit, args.batch_size, args.workers)


if __name__ == "__main__":
    main()
