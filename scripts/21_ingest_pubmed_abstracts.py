#!/usr/bin/env python3
"""
Ingest PubMed Abstracts to Qdrant with Hybrid Vectors.

Ingests high-value PubMed abstracts (Reviews, Meta-Analyses, Practice Guidelines)
matching the shared architecture of PMC/DailyMed ingestion.

Features:
- Embedding: Uses centralized EmbeddingProvider (Cohere/Local)
- Sparse: BM25 (if enabled in config)
- Chunking: Title + Abstract
- Deduplication: PMID-based UUIDs
- Consistency: Shares config with other ingestion scripts

Usage:
    python 21_ingest_pubmed_abstracts.py --input /data/pubmed_baseline/filtered/pubmed_abstracts.jsonl
"""

import argparse
import importlib.util
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import EmbeddingProvider, upsert_with_retry

# Import BM25SparseEncoder
spec = importlib.util.find_spec("src.bm25_sparse")
if spec is not None:
    from src.bm25_sparse import BM25SparseEncoder
else:
    BM25SparseEncoder = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pubmed_ingested_ids.txt"

# Evidence grades mapping based on article_type
EVIDENCE_GRADES = {
    "practice_guideline": "A",
    "meta_analysis": "A",
    "systematic_review": "A",
    "review": "B",
}


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return {line.strip() for line in CHECKPOINT_FILE.read_text().splitlines() if line.strip()}
    return set()


def append_checkpoint(ids: List[str]) -> None:
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        for value in ids:
            f.write(f"{value}\n")


def create_payload(article: Dict[str, Any]) -> Dict[str, Any]:
    """Create payload matching existing schema."""
    article_type = article.get("article_type", "review")
    evidence_grade = EVIDENCE_GRADES.get(article_type, "C")
    
    return {
        # Identifiers
        "doc_id": article.get("pmid"),
        "pmcid": None,
        "pmid": article.get("pmid"),
        "doi": article.get("doi"),
        
        # Content
        "title": (article.get("title") or "")[:300],
        "abstract": (article.get("abstract") or "")[:2000],
        "full_text": "", 
        
        # Publication info
        "year": article.get("year"),
        "journal": (article.get("journal") or "")[:200],
        "article_type": article_type,
        "publication_type": article.get("publication_types", [])[:5],
        
        # Evidence
        "evidence_grade": evidence_grade,
        "evidence_level": {"A": 1, "B": 2, "C": 3, "D": 4}.get(evidence_grade, 3),
        "country": None,
        "affiliations": article.get("affiliations", [])[:3],
        
        # Classification
        "keywords": [],
        "mesh_terms": article.get("mesh_terms", [])[:15],
        
        # Authors
        "authors": article.get("authors", [])[:5],
        
        # Metadata
        "source": "pubmed_abstract",
        "content_type": "abstract",
        "has_full_text": False,
        "table_count": 0,
    }


def build_points(batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider, sparse_encoder: Optional[Any]) -> tuple[List[PointStruct], List[str]]:
    points: List[PointStruct] = []
    processed_pmids: List[str] = []
    texts_to_embed: List[str] = []
    text_metas: List[Dict[str, Any]] = []

    for article in batch:
        pmid = article.get("pmid")
        if not pmid:
            continue
            
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        
        if not title and not abstract:
            continue
            
        # Create text for embedding
        text = f"{title}. {abstract}"[:2000] 
        texts_to_embed.append(text)
        
        # Metadata for point creation
        text_metas.append({
            "pmid": pmid,
            "payload": create_payload(article),
            "text": text
        })

    if not texts_to_embed:
        return [], []

    # Embed batch
    try:
        vectors = embedding_provider.embed_batch(texts_to_embed)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return [], []

    # Create points
    for meta, vector in zip(text_metas, vectors):
        pmid = meta["pmid"]
        
        # Sparse vector
        vector_data: Any = vector
        if sparse_encoder:
            sparse_vector = sparse_encoder.encode_document(meta["text"])
            vector_data = {"": vector, "sparse": sparse_vector}
            
        # Point ID (deterministic from PMID)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pmid:{pmid}"))
        
        points.append(PointStruct(id=point_id, vector=vector_data, payload=meta["payload"]))
        processed_pmids.append(str(pmid))

    return points, processed_pmids





def run_ingestion(input_file: Path, limit: Optional[int], embedding_provider: EmbeddingProvider) -> None:
    if not input_file.exists():
        logger.error("Input file not found: %s", input_file)
        return

    ensure_data_dirs()
    
    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    # Initialize Sparse Encoder if enabled
    sparse_encoder = None
    use_sparse = IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25"
    if use_sparse and BM25SparseEncoder is not None:
        logger.info("Initializing BM25 sparse encoder...")
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )

    processed_ids = load_checkpoint()
    logger.info("Loaded checkpoint with %d IDs", len(processed_ids))

    batch: List[Dict[str, Any]] = []
    total_processed = 0
    start_time = time.time()

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if limit and total_processed >= limit:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                continue

            pmid = str(article.get("pmid"))
            if pmid in processed_ids:
                continue

            batch.append(article)
            
            if len(batch) >= IngestionConfig.BATCH_SIZE:
                points, ids = build_points(batch, embedding_provider, sparse_encoder)
                if points:
                    upsert_with_retry(client, points)
                    append_checkpoint(ids)
                    processed_ids.update(ids)
                    total_processed += len(points)
                    
                    if total_processed % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = total_processed / elapsed
                        logger.info("Ingested: %d (%.1f docs/sec)", total_processed, rate)
                
                batch.clear()

    # Final batch
    if batch:
        points, ids = build_points(batch, embedding_provider, sparse_encoder)
        if points:
            upsert_with_retry(client, points)
            append_checkpoint(ids)
            total_processed += len(points)

    logger.info("Start-to-finish ingestion complete. Total: %d", total_processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PubMed Abstracts")
    parser.add_argument("--input", type=Path, required=True, help="Path to pubmed_abstracts.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs")
    
    args = parser.parse_args()
    
    provider = EmbeddingProvider()
    run_ingestion(args.input, args.limit, provider)
