#!/usr/bin/env python3
"""
Ingest PubMed Abstracts to Qdrant with Hybrid Vectors and Word-Based Chunking.

Ingests high-value PubMed abstracts (Reviews, Meta-Analyses, Practice Guidelines)
matching the shared architecture of PMC/DailyMed ingestion.

Features:
- Embedding: Uses centralized EmbeddingProvider (DeepInfra)
- Sparse: BM25 (if enabled in config)
- Chunking: Word-based chunking sized by estimated token limits
- Deduplication: PMID-based UUIDs
- Consistency: Shares config with other ingestion scripts
- Supports full structured data from whitelist extraction (MeSH with qualifiers,
  structured abstracts, detailed journal info)

Payload Structure:
- Identifiers: pmid, doi, pmc, pii, other_ids
- Content: title, abstract (no limits), abstract_structured, page_content
- Journal: journal, journal_full (ISSN, volume, issue, etc.)
- Dates: year, publication_date
- Classification: mesh_terms, mesh_terms_full, keywords, keywords_full, publication_types
- Evidence: article_type, evidence_grade, evidence_level
- Chunking: chunk_index, total_chunks, parent_section_id

Usage:
    python 21_ingest_pubmed_abstracts.py
    python 21_ingest_pubmed_abstracts.py --input /data/ingestion/pubmed_baseline/filtered/pubmed_abstracts.jsonl
"""

import argparse
import importlib.util
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import sys
# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import (
    Chunker as BaseChunker,
    EmbeddingProvider,
    upsert_with_retry,
    validate_qdrant_collection_schema,
    classify_evidence_metadata,
    get_chunker as get_shared_chunker,
    load_checkpoint as load_checkpoint_file,
    append_checkpoint as append_checkpoint_file,
    generate_section_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import enhanced utilities for semantic chunking and validation
try:
    from ingestion_utils_enhanced import SemanticChunker, QualityValidator, ContentDeduplicator
    ENHANCED_UTILS_AVAILABLE = True
    logger.info("Using SemanticChunker for improved chunking")
except ImportError:
    SemanticChunker = None  # type: ignore
    QualityValidator = None  # type: ignore
    ContentDeduplicator = None  # type: ignore
    ENHANCED_UTILS_AVAILABLE = False
    logger.warning("Enhanced utils not available")

CHUNKER_CLASS = SemanticChunker if ENHANCED_UTILS_AVAILABLE else BaseChunker

# Import BM25SparseEncoder
spec = importlib.util.find_spec("src.bm25_sparse")
if spec is not None:
    from src.bm25_sparse import BM25SparseEncoder
else:
    BM25SparseEncoder = None  # type: ignore

CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pubmed_ingested_ids.txt"
DEFAULT_INPUT_FILE = IngestionConfig.PUBMED_ABSTRACTS_FILE

# Namespace for PubMed checkpoint IDs to prevent collision with PMC/DailyMed
PUBMED_CHECKPOINT_NAMESPACE = "pubmed"


def _checkpoint_id(pmid: str) -> str:
    """Generate namespaced checkpoint ID for PubMed abstracts."""
    return f"{PUBMED_CHECKPOINT_NAMESPACE}:{pmid.strip()}"


def _resolve_checkpoint_line(line: str) -> str | None:
    """Resolve checkpoint line to namespaced ID, handling legacy plain PMIDs."""
    value = line.strip()
    if not value:
        return None
    
    # Already namespaced
    if value.startswith(f"{PUBMED_CHECKPOINT_NAMESPACE}:"):
        return value
    
    # Legacy format (plain PMID) - convert to namespaced
    return _checkpoint_id(value)


def load_checkpoint_namespaced(path: Path) -> set[str]:
    """Load checkpoint with legacy format support."""
    return {
        resolved
        for line in load_checkpoint_file(path)
        if (resolved := _resolve_checkpoint_line(line)) is not None
    }

# Token counting is handled by the shared Chunker class from ingestion_utils.
# Counts are estimated from words for consistency across ingestion scripts.


def count_tokens(text: str) -> int:
    """Estimate tokens using the shared Chunker's word-based approximation."""
    chunker = get_shared_chunker(
        chunker_class=CHUNKER_CLASS,
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
    )
    return chunker.count_tokens(text)


def create_payloads_with_chunking(article: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create one or more payloads with word-based chunking.
    
    If the content exceeds CHUNK_SIZE_TOKENS (estimated from words), it will be
    split into multiple chunks with overlap. Each chunk becomes a separate point
    in Qdrant.
    
    Merged PubMed pipeline: includes both high-value article types AND
    government affiliation detection (from former gov pipeline).
    
    Returns:
        List of payload dictionaries (one per chunk)
    """
    article_type = article.get("article_type", "review")
    
    pmid = article.get("pmid")
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    
    # Government affiliation fields (merged from gov pipeline)
    is_gov_affiliated = article.get("is_gov_affiliated", False)
    gov_agencies = article.get("gov_agencies", [])
    
    # Create full content text
    full_text = f"Title: {title}\n\nAbstract: {abstract}" if title or abstract else ""
    
    # Get chunking config
    chunk_size = IngestionConfig.CHUNK_SIZE_TOKENS
    chunk_overlap = IngestionConfig.CHUNK_OVERLAP_TOKENS
    
    # Count total tokens
    total_tokens = count_tokens(full_text)
    
    # Get structured abstract sections if available
    abstract_structured = article.get("abstract_structured", [])
    has_structured = article.get("has_structured_abstract", False)
    
    # Get journal info
    journal_full = article.get("journal_full", {})
    pub_date = article.get("publication_date", {})
    nlm_unique_id = journal_full.get("nlm_unique_id") if journal_full else None
    
    # Build mesh_terms
    mesh_terms = article.get("mesh_terms", [])
    if mesh_terms and isinstance(mesh_terms[0], dict):
        mesh_flat = [m["descriptor"] for m in mesh_terms]
    else:
        mesh_flat = mesh_terms if mesh_terms else []
        mesh_terms = []
    
    # Build keywords
    keywords = article.get("keywords", [])
    if keywords and isinstance(keywords[0], dict):
        keywords_flat = [k["keyword"] for k in keywords]
    else:
        keywords_flat = keywords if keywords else []
        keywords = []
    
    # Get publication types
    pub_types_data = article.get("publication_types", [])
    if pub_types_data and isinstance(pub_types_data[0], dict):
        pub_types_flat = [p["type"] for p in pub_types_data]
    else:
        pub_types_flat = pub_types_data if pub_types_data else []

    evidence = classify_evidence_metadata(
        article_type=article_type,
        pub_types=pub_types_flat,
        abstract=abstract,
    )
    evidence_grade = evidence["grade"]
    evidence_level = evidence["level_1_4"]
    evidence_term = evidence["matched_term"]
    evidence_source = evidence["matched_from"]
    
    # Base section ID (for single chunk or parent reference)
    base_section_id = generate_section_id(pmid, "Abstract")
    
    payloads = []
    
    # Use shared Chunker for consistent chunking across all ingestion scripts
    chunker = get_shared_chunker(
        chunker_class=CHUNKER_CLASS,
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
    )
    
    # Determine if chunking is needed
    if total_tokens <= chunk_size:
        # Single chunk - no splitting needed
        payload = {
            # Identifiers
            "doc_id": pmid,
            "chunk_id": f"{pmid}_abstract",
            "pmcid": article.get("pmc"),
            "pmid": pmid,
            "doi": article.get("doi"),
            "pii": article.get("pii"),
            "other_ids": article.get("other_ids", {}),
            
            # Content
            "page_content": full_text,
            "title": title,
            "abstract": abstract,
            
            # Chunking metadata
            "chunk_index": 0,
            "total_chunks": 1,
            "token_count": total_tokens,
            
            # Abstract structure
            "abstract_structured": abstract_structured,
            "has_structured_abstract": has_structured,
            "full_text": "",
            
            # Publication info
            "year": article.get("year"),
            "journal": article.get("journal", ""),
            "journal_full": journal_full,
            "nlm_unique_id": nlm_unique_id,
            "publication_date": pub_date,
            "article_type": article_type,
            "publication_type": pub_types_flat,
            "publication_types_full": pub_types_data,
            
            # Evidence
            "evidence_grade": evidence_grade,
            "evidence_level": evidence_level,
            "evidence_term": evidence_term,
            "evidence_source": evidence_source,
            
            # Classification
            "mesh_terms": mesh_flat,
            "mesh_terms_full": mesh_terms,
            "keywords": keywords_flat,
            "keywords_full": keywords,
            
            # Parent-child fields
            "full_section_text": full_text,
            "section_id": base_section_id,
            "section_weight": 1.0,
            "section_type": "abstract",
            "section_title": "Abstract",
            
            # Metadata
            "source": "pubmed_abstract",
            "source_family": "pubmed",
            "content_type": "abstract",
            "has_full_text": False,
            "table_count": 0,
            "text_preview": full_text[:500],
            
            # Government affiliation (merged from gov pipeline)
            "is_gov_affiliated": is_gov_affiliated,
            "gov_agencies": gov_agencies,
        }
        payloads.append(payload)
    else:
        # Multi-chunk: split content using shared Chunker
        chunk_results = chunker.chunk_text(full_text)
        
        for i, chunk_data in enumerate(chunk_results):
            # Create sub-section ID for this chunk
            chunk_section_id = generate_section_id(pmid, f"Abstract_{i}")
            
            # Determine chunk title
            if i == 0:
                chunk_title = title
            else:
                chunk_title = f"{title} (cont.)" if title else f"Abstract (Part {i+1})"
            
            # Extract chunk abstract (remove title prefix if present)
            chunk_text = chunk_data['text']
            chunk_abstract = chunk_text
            if chunk_text.startswith(f"Title: {title}\n\nAbstract: "):
                chunk_abstract = chunk_text[len(f"Title: {title}\n\nAbstract: "):]
            elif chunk_text.startswith("Title: "):
                # Fallback for continuation chunks
                chunk_abstract = chunk_text[chunk_text.find("\n\nAbstract: ") + 12:]
            
            payload = {
                # Identifiers
                "doc_id": pmid,
                "chunk_id": f"{pmid}_abstract_chunk_{i}",
                "pmcid": article.get("pmc"),
                "pmid": pmid,
                "doi": article.get("doi"),
                "pii": article.get("pii"),
                "other_ids": article.get("other_ids", {}),
                
                # Content
                "page_content": chunk_text,
                "title": chunk_title,
                "abstract": chunk_abstract,
                
                # Chunking metadata
                "chunk_index": i,
                "total_chunks": len(chunk_results),
                "token_count": chunk_data['token_count'],
                
                # Abstract structure (only in first chunk)
                "abstract_structured": abstract_structured if i == 0 else [],
                "has_structured_abstract": has_structured if i == 0 else False,
                "full_text": "",
                
                # Publication info
                "year": article.get("year"),
                "journal": article.get("journal", ""),
                "journal_full": journal_full,
                "nlm_unique_id": nlm_unique_id,
                "publication_date": pub_date,
                "article_type": article_type,
                "publication_type": pub_types_flat,
                "publication_types_full": pub_types_data if i == 0 else [],
                
                # Evidence
                "evidence_grade": evidence_grade,
                "evidence_level": evidence_level,
                "evidence_term": evidence_term,
                "evidence_source": evidence_source,
                
                # Classification (only in first chunk to save space)
                "mesh_terms": mesh_flat if i == 0 else [],
                "mesh_terms_full": mesh_terms if i == 0 else [],
                "keywords": keywords_flat if i == 0 else [],
                "keywords_full": keywords if i == 0 else [],
                
                # Parent-child fields
                "full_section_text": full_text,
                "section_id": chunk_section_id,
                "parent_section_id": base_section_id,
                "section_weight": 1.0 - (i * 0.05),  # Slight decay for later chunks
                "section_type": "abstract",
                "section_title": f"Abstract (Part {i+1}/{len(chunk_results)})",
                
                # Metadata
                "source": "pubmed_abstract",
                "source_family": "pubmed",
                "content_type": "abstract",
                "has_full_text": False,
                "table_count": 0,
                "text_preview": chunk_text[:500],
                
                # Government affiliation (merged from gov pipeline)
                "is_gov_affiliated": is_gov_affiliated,
                "gov_agencies": gov_agencies if i == 0 else [],
            }
            payloads.append(payload)
    
    return payloads


def build_points(batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider, sparse_encoder: Optional[Any], 
                 validate_chunks: bool = True, dedup_chunks: bool = True) -> tuple[List[PointStruct], List[str]]:
    """
    Build Qdrant points from article batch with word-based chunking.
    
    Each article may generate 1 or multiple points depending on estimated token count.
    
    Args:
        batch: List of articles to process
        embedding_provider: Provider for generating embeddings
        sparse_encoder: Optional sparse encoder for hybrid search
        validate_chunks: Whether to validate chunk quality before ingestion
        dedup_chunks: Whether to deduplicate chunks within batch
    """
    points: List[PointStruct] = []
    processed_pmids: List[str] = []
    texts_to_embed: List[str] = []
    text_metas: List[Dict[str, Any]] = []
    total_validated = 0
    total_rejected = 0

    # Initialize deduplicator if enabled
    dedup = None
    if dedup_chunks and ContentDeduplicator is not None:
        dedup = ContentDeduplicator()

    for article in batch:
        pmid = article.get("pmid")
        if not pmid:
            continue
            
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        
        if not title and not abstract:
            continue
        
        # Create payloads with chunking
        payloads = create_payloads_with_chunking(article)
        
        # Validate chunks if enabled
        if validate_chunks and ENHANCED_UTILS_AVAILABLE:
            valid_payloads = []
            for payload in payloads:
                chunk_text = payload.get("page_content", "")
                is_valid, issues = QualityValidator.validate_chunk(
                    chunk_text,
                    {
                        k: payload.get(k)
                        for k in ["doc_id", "chunk_id", "title", "pmid", "year", "page_content"]
                    },
                )
                if is_valid:
                    valid_payloads.append(payload)
                    total_validated += 1
                else:
                    logger.debug("Skipping invalid chunk %s: %s", payload.get("chunk_id"), issues)
                    total_rejected += 1
            payloads = valid_payloads
        
        # Deduplicate chunks if enabled
        if dedup:
            unique_payloads = []
            for payload in payloads:
                metadata = {
                    "doc_id": payload.get("doc_id", ""),
                    "chunk_id": payload.get("chunk_id", "")
                }
                if not dedup.is_duplicate(payload["page_content"], metadata):
                    unique_payloads.append(payload)
            payloads = unique_payloads
        
        for payload in payloads:
            text = payload["page_content"]
            texts_to_embed.append(text)
            
            text_metas.append({
                "pmid": pmid,
                "payload": payload,
                "text": text
            })
    
    if validate_chunks and ENHANCED_UTILS_AVAILABLE:
        logger.debug("Chunk validation: %d valid, %d rejected", total_validated, total_rejected)

    if not texts_to_embed:
        return [], []

    # Embed batch
    try:
        vectors = embedding_provider.embed_batch(texts_to_embed)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return [], []

    sparse_vectors: List[Optional[Any]] = []
    if sparse_encoder:
        try:
            if hasattr(sparse_encoder, "encode_batch"):
                sparse_vectors = sparse_encoder.encode_batch(texts_to_embed)
            else:
                sparse_vectors = [sparse_encoder.encode_document(text) for text in texts_to_embed]
        except Exception as e:
            logger.warning("Sparse encoding failed for batch: %s", e)
            sparse_vectors = [None] * len(texts_to_embed)

    # Create points
    for idx, (meta, vector) in enumerate(zip(text_metas, vectors)):
        pmid = meta["pmid"]
        
        # Sparse vector
        vector_data: Any = {"dense": vector}
        if sparse_encoder and idx < len(sparse_vectors) and sparse_vectors[idx] is not None:
            vector_data["sparse"] = sparse_vectors[idx]
            
        # Point ID (deterministic from chunk_id for uniqueness)
        chunk_id = meta["payload"].get("chunk_id", f"{pmid}_abstract")
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed:{chunk_id}"))
        
        points.append(PointStruct(id=point_id, vector=vector_data, payload=meta["payload"]))
        
        # Track unique PMIDs for checkpoint
        if str(pmid) not in processed_pmids:
            processed_pmids.append(str(pmid))

    return points, processed_pmids


def process_batch(client: QdrantClient, batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider, 
                  sparse_encoder: Optional[Any], validate_chunks: bool = True, dedup_chunks: bool = True) -> int:
    """Process a single batch in a worker thread."""
    if not batch:
        return 0
    
    try:
        points, ids = build_points(batch, embedding_provider, sparse_encoder, 
                                   validate_chunks=validate_chunks, dedup_chunks=dedup_chunks)
        if points:
            upsert_with_retry(client, points)
            # Convert plain PMIDs to namespaced checkpoint IDs
            checkpoint_ids = [_checkpoint_id(pmid) for pmid in ids]
            append_checkpoint_file(CHECKPOINT_FILE, checkpoint_ids)
            return len(points)
    except Exception as e:
        logger.error("Batch processing failed: %s", e)
    
    return 0


def run_ingestion(input_file: Path, limit: Optional[int], embedding_provider: EmbeddingProvider) -> None:
    if not input_file.exists():
        logger.error("Input file not found: %s", input_file)
        return

    ensure_data_dirs()
    
    # Preload shared chunker
    logger.info("Preloading shared chunker...")
    try:
        get_shared_chunker(
            chunker_class=CHUNKER_CLASS,
            chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
            overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
        )
        logger.info("Shared chunker ready.")
    except Exception as e:
        logger.warning(f"Shared chunker preload warning: {e}")
    
    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    try:
        info = client.get_collection(IngestionConfig.COLLECTION_NAME)
        logger.info("Connected to %s points=%s", IngestionConfig.COLLECTION_NAME, info.points_count)
        validate_qdrant_collection_schema(client, IngestionConfig.COLLECTION_NAME)
    except Exception as e:
         logger.error("Failed to connect to collection: %s", e)
         return

    # Initialize Sparse Encoder if enabled
    sparse_encoder = None
    use_sparse = IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25"
    if use_sparse and BM25SparseEncoder is not None:
        logger.info("Initializing BM25 sparse encoder...")
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )

    processed_ids = load_checkpoint_namespaced(CHECKPOINT_FILE)
    logger.info("Loaded checkpoint with %d IDs", len(processed_ids))
    logger.info("Chunking config: size=%d, overlap=%d, chunker=%s, validation=%s, dedup=%s", 
                IngestionConfig.CHUNK_SIZE_TOKENS, 
                IngestionConfig.CHUNK_OVERLAP_TOKENS,
                CHUNKER_CLASS.__name__,
                ENHANCED_UTILS_AVAILABLE,
                ContentDeduplicator is not None)

    current_batch: List[Dict[str, Any]] = []
    total_processed = 0
    total_chunks = 0
    articles_chunked = 0
    start_time = time.time()
    
    MAX_WORKERS = IngestionConfig.MAX_WORKERS
    BATCH_SIZE = IngestionConfig.BATCH_SIZE
    
    futures = []
    
    logger.info("Starting ingestion with %d workers...", MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
                checkpoint_id = _checkpoint_id(pmid)
                if checkpoint_id in processed_ids:
                    continue

                current_batch.append(article)
                
                # Check if batch full
                if len(current_batch) >= BATCH_SIZE:
                    # Count chunks in this batch for stats
                    for art in current_batch:
                        text = f"Title: {art.get('title', '')}\n\nAbstract: {art.get('abstract', '')}"
                        tokens = count_tokens(text)
                        if tokens > IngestionConfig.CHUNK_SIZE_TOKENS:
                            # Improved chunk count calculation: ceil((tokens - overlap) / (chunk_size - overlap))
                            effective_size = IngestionConfig.CHUNK_SIZE_TOKENS - IngestionConfig.CHUNK_OVERLAP_TOKENS
                            remaining_tokens = tokens - IngestionConfig.CHUNK_OVERLAP_TOKENS
                            chunks_needed = (remaining_tokens + effective_size - 1) // effective_size  # Ceiling division
                            chunks_needed = max(1, chunks_needed)  # Ensure at least 1
                            total_chunks += chunks_needed
                            articles_chunked += 1
                        else:
                            total_chunks += 1
                    
                    # Submit job
                    future = executor.submit(process_batch, client, list(current_batch), embedding_provider, sparse_encoder,
                                            validate_chunks=ENHANCED_UTILS_AVAILABLE, dedup_chunks=ENHANCED_UTILS_AVAILABLE)
                    futures.append(future)
                    current_batch.clear()
                    
                    # Periodic maintenance
                    if len(futures) >= MAX_WORKERS * 2:
                        pending = []
                        for fut in futures:
                            if fut.done():
                                try:
                                    total_processed += fut.result()
                                except Exception as e:
                                    logger.error("Future failed: %s", e)
                            else:
                                pending.append(fut)
                        
                        futures = pending
                        
                        elapsed = time.time() - start_time
                        if elapsed > 0 and total_processed > 0:
                             rate = total_processed / elapsed
                             logger.info("Ingested: %d chunks (%.1f chunks/sec, %d articles chunked)", 
                                       total_processed, rate, articles_chunked)

        # Submit remaining
        if current_batch:
            future = executor.submit(process_batch, client, list(current_batch), embedding_provider, sparse_encoder,
                                    validate_chunks=ENHANCED_UTILS_AVAILABLE, dedup_chunks=ENHANCED_UTILS_AVAILABLE)
            futures.append(future)
            
        # Wait for all remaining
        for fut in as_completed(futures):
            try:
                total_processed += fut.result()
            except Exception as e:
                logger.error("Future failed: %s", e)

    logger.info("Ingestion complete. Total points: %d (from %d articles)", total_processed, len(processed_ids))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PubMed Abstracts with word-based chunking")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to pubmed_abstracts.jsonl (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs")
    
    args = parser.parse_args()
    
    provider = EmbeddingProvider()
    run_ingestion(args.input, args.limit, provider)
