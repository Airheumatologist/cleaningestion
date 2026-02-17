#!/usr/bin/env python3
"""Ingest PMC content into self-hosted Qdrant with improved section and table handling."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Document, PointStruct

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import SectionFilter, EmbeddingProvider, upsert_with_retry
from ingestion_utils import Chunker as BaseChunker  # Fallback

# Import enhanced utilities if available
try:
    from ingestion_utils_enhanced import (
        SemanticChunker, QualityValidator, 
        ContentDeduplicator, enhance_pmc_parsing
    )
    ENHANCED_UTILS_AVAILABLE = True
    Chunker = SemanticChunker  # Use semantic chunking by default
    logger.info("Using SemanticChunker for improved chunking")
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False
    Chunker = BaseChunker  # Fallback to base chunker
    logger.warning("Enhanced ingestion utils not available, using base Chunker")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pmc_ingested_ids.txt"





import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import BM25SparseEncoder
spec = importlib.util.find_spec("src.bm25_sparse")
if spec is not None:
    from src.bm25_sparse import BM25SparseEncoder
else:
    BM25SparseEncoder = None  # type: ignore


def create_payload(article: Dict[str, Any]) -> Dict[str, Any]:
    """Create base payload for a PMC article."""
    # Handle PRD-compliant structure
    metadata = article.get("metadata", {})
    content = article.get("content", {})
    identifiers = metadata.get("identifiers", {}) if metadata else {}
    publication = metadata.get("publication", {}) if metadata else {}
    content_flags = metadata.get("content_flags", {}) if metadata else {}
    classification = metadata.get("classification", {}) if metadata else {}
    
    pmcid = str(identifiers.get("pmcid") or article.get("pmcid") or "")
    pmid = str(identifiers.get("pmid") or article.get("pmid") or "")
    doi = identifiers.get("doi") or article.get("doi", "")
    title = content.get("title", "") if content else article.get("title", "")
    journal_info = publication.get("journal", {}) if publication else {}
    journal = journal_info.get("title", "") if journal_info else article.get("journal", "")
    year = publication.get("year") if publication else article.get("year")
    country = publication.get("country", "") if publication else article.get("country", "")
    
    # Get tables for count
    tables = content.get("tables", []) if content else article.get("tables", [])
    
    return {
        "doc_id": pmcid or pmid,
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi,
        "title": title[:300],
        "journal": journal[:100],
        "year": year,

        "keywords": classification.get("keywords", []) if classification else article.get("keywords", [])[:20],
        "mesh_terms": classification.get("mesh_terms", []) if classification else article.get("mesh_terms", [])[:30],
        "article_type": metadata.get("article_type", "") if metadata else article.get("article_type", "")[:50],
        "evidence_grade": article.get("evidence_grade", ""),
        "evidence_level": article.get("evidence_level"),
        "country": country,
        "table_count": len(tables),
        "has_full_text": content_flags.get("has_full_text") if content_flags else article.get("has_full_text", False),
        "is_open_access": content_flags.get("is_open_access") if content_flags else article.get("is_open_access", False),
        "source": "pmc",
    }


import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lock for file operations (checkpointing)
checkpoint_lock = threading.Lock()

def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return {line.strip() for line in CHECKPOINT_FILE.read_text().splitlines() if line.strip()}
    return set()


def append_checkpoint(ids: Iterable[str]) -> None:
    with checkpoint_lock:
        with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
            for value in ids:
                f.write(f"{value}\n")


def create_chunks_from_article(article: Dict[str, Any], chunker: Chunker) -> List[Dict[str, Any]]:
    """
    Create multiple chunks from an article: sections + tables with token-aware chunking.
    Updated for PRD v1.0 compliant article structure.
    Includes full metadata propagation and parent-child ready fields.
    """
    from ingestion_utils import generate_section_id, get_section_weight
    
    chunks = []
    
    # Handle both PRD-compliant structure and legacy structure
    metadata = article.get("metadata", {})
    content = article.get("content", {})
    
    # Extract identifiers
    identifiers = metadata.get("identifiers", {}) if metadata else {}
    doc_id = str(identifiers.get("pmcid") or identifiers.get("pmid") or article.get("pmcid") or article.get("pmid") or "")
    pmid = identifiers.get("pmid") or article.get("pmid", "")
    
    # Extract content
    title = content.get("title", "") if content else article.get("title", "")
    abstract_text = article.get("abstract", "")
    
    # Extract publication info
    publication = metadata.get("publication", {}) if metadata else {}
    journal = publication.get("journal", {}).get("title", "") if publication else article.get("journal", "")
    year = publication.get("year") if publication else article.get("year")
    country = publication.get("country", "") if publication else article.get("country", "")
    
    # Extract classification
    classification = metadata.get("classification", {}) if metadata else {}
    keywords = classification.get("keywords", []) if classification else article.get("keywords", [])
    mesh_terms = classification.get("mesh_terms", []) if classification else article.get("mesh_terms", [])
    
    # Common metadata for all chunks (full metadata propagation)
    # Note: Author information excluded per requirements
    base_metadata = {
        "title": title,
        "abstract": abstract_text,
        "journal": journal,
        "year": year,
        "pmcid": identifiers.get("pmcid") if identifiers else article.get("pmcid"),
        "pmid": pmid,
        "country": country,
        "keywords": keywords,
        "mesh_terms": mesh_terms,
    }
    
    # 1. Abstract chunk (most important) - usually fits in single chunk
    if abstract_text and len(abstract_text) > 50:
        abstract_full_text = f"Title: {title}\n\nAbstract: {abstract_text}"
        section_id = generate_section_id(doc_id, "Abstract")
        
        chunks.append({
            "chunk_id": f"{doc_id}_abstract",
            "doc_id": doc_id,
            "text": abstract_full_text,
            "section_type": "abstract",
            "section_title": "Abstract",
            "is_table": False,
            # Parent-child ready fields
            "full_section_text": abstract_full_text,
            "section_id": section_id,
            "section_weight": get_section_weight("abstract"),
            **base_metadata,
        })
    
    # 2. Section chunks with token-aware splitting
    # Use PRD-compliant sections structure
    sections = content.get("sections", []) if content else article.get("structured_sections", [])
    for i, section in enumerate(sections):
        # Skip excluded sections (references, acknowledgments, etc.)
        if SectionFilter.should_exclude(section):
            continue
            
        # PRD structure uses "content" field, legacy uses "text"
        sec_text = section.get("content", "") or section.get("text", "")
        if len(sec_text) < 50:
            continue
        
        sec_title = section.get("title", "Body")
        sec_type = section.get("type", "other")
        
        # Generate section ID for parent-child indexing
        section_id = generate_section_id(doc_id, sec_title)
        
        # Create context-rich prefix
        prefix = f"Title: {title}\n\nSection: {sec_title}\n\n"
        full_section_text = prefix + sec_text
        
        # Use Chunker for token-aware chunking
        section_chunks = chunker.chunk_text(full_section_text)
        
        for j, chunk_data in enumerate(section_chunks):
            chunk_id = f"{doc_id}_sec{i}_part{j}" if len(section_chunks) > 1 else f"{doc_id}_sec{i}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_data["text"],
                "section_type": sec_type,
                "section_title": sec_title,
                "is_table": False,
                # Parent-child ready fields
                "full_section_text": full_section_text,
                "section_id": section_id,
                "section_weight": get_section_weight(sec_type),
                **base_metadata,
            })
    
    # 3. Table chunks with token-aware chunking for large tables
    tables = content.get("tables", []) if content else article.get("tables", [])
    for i, table in enumerate(tables):
        # PRD-compliant table structure
        if isinstance(table, dict):
            if "content" in table:  # PRD structure
                table_text = table.get("content", "")
                caption = table.get("caption_title", "")
                table_label = table.get("label", "")
                table_id = table.get("id", f"table-{i}")
                full_caption = f"{table_label}: {caption}".strip(": ")
            else:  # Legacy structure
                table_text = table.get("row_by_row", "") or table.get("markdown", "")
                full_caption = table.get("caption", "")
                table_id = table.get("id", f"table-{i}")
        else:
            continue
        
        if not table_text or len(table_text) < 20:
            continue
        
        # Generate section ID for table
        section_title = f"Table: {full_caption[:100]}" if full_caption else f"Table {i}"
        section_id = generate_section_id(doc_id, section_title)
        
        # Create context-rich table text
        table_context = f"Title: {title}\n\n{section_title}\n\n{table_text}"
        
        # Use Chunker for token-aware chunking of large tables
        table_chunks = chunker.chunk_text(table_context)
        
        for j, chunk_data in enumerate(table_chunks):
            chunk_id = f"{doc_id}_table{i}_part{j}" if len(table_chunks) > 1 else f"{doc_id}_table{i}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": chunk_data["text"],
                "section_type": "table",
                "section_title": section_title,
                "is_table": True,
                "table_caption": full_caption,
                "table_id": table_id,
                # Parent-child ready fields
                "full_section_text": table_context,
                "section_id": section_id,
                "section_weight": get_section_weight("table"),
                **base_metadata,
            })
    
    return chunks


def build_points(batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider, sparse_encoder=None, 
                validate_chunks: bool = True, dedup_chunks: bool = True) -> tuple[List[PointStruct], List[str]]:
    """
    Build Qdrant points from article batch.
    
    Args:
        batch: List of parsed articles
        embedding_provider: Provider for generating embeddings
        sparse_encoder: Optional sparse encoder for hybrid search
        validate_chunks: Whether to validate chunk quality before ingestion
        dedup_chunks: Whether to deduplicate chunks within batch
    """
    # Use SemanticChunker if available, otherwise fallback to base Chunker
    chunker = Chunker(
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS
    )
    
    points: List[PointStruct] = []
    all_chunk_ids: List[str] = []
    
    # Initialize deduplicator if enabled
    dedup = None
    if dedup_chunks and ENHANCED_UTILS_AVAILABLE:
        dedup = ContentDeduplicator()
    
    # Collect all chunks for batch embedding
    all_chunks: List[Dict[str, Any]] = []
    all_texts: List[str] = []
    
    for article in batch:
        chunks = create_chunks_from_article(article, chunker)
        
        # Validate chunks if enabled
        if validate_chunks and ENHANCED_UTILS_AVAILABLE:
            valid_chunks = []
            for chunk in chunks:
                is_valid, issues = QualityValidator.validate_chunk(
                    chunk["text"],
                    {k: chunk.get(k) for k in ["doc_id", "chunk_id", "title", "authors", "year"]}
                )
                if is_valid:
                    valid_chunks.append(chunk)
                else:
                    logger.debug("Skipping invalid chunk %s: %s", chunk.get("chunk_id"), issues)
            chunks = valid_chunks
        
        # Deduplicate chunks if enabled
        if dedup:
            unique_chunks = []
            for chunk in chunks:
                metadata = {
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_id": chunk.get("chunk_id", "")
                }
                if not dedup.is_duplicate(chunk["text"], metadata):
                    unique_chunks.append(chunk)
            chunks = unique_chunks
        
        all_chunks.extend(chunks)
        all_texts.extend([c["text"] for c in chunks])
    
    if not all_texts:
        return [], []
    
    logger.info("Embedding %s chunks (validated: %s, deduped: %s)", 
                len(all_texts), validate_chunks, dedup_chunks)
    
    # Embed all texts in batch
    try:
        vectors = embedding_provider.embed_batch(all_texts)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return [], []
    
    # Create points
    for chunk, vector in zip(all_chunks, vectors):
        chunk_id = chunk["chunk_id"]
        all_chunk_ids.append(chunk_id)
        
        # Create sparse vector if enabled
        vector_data: Any = {"dense": vector}
        if sparse_encoder is not None:
            sparse_vector = sparse_encoder.encode_document(chunk["text"])
            vector_data["sparse"] = sparse_vector
        
        # Create payload with FULL metadata (CRITICAL: page_content for retriever)
        payload = {
            # Core identifiers
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk_id,
            "pmcid": chunk.get("pmcid", ""),
            "pmid": chunk.get("pmid", ""),
            
            # CRITICAL: Full text for retriever (was missing!)
            "page_content": chunk["text"],
            
            # Article metadata for citations and filtering
            "title": chunk.get("title", ""),
            "abstract": chunk.get("abstract", ""),
            "journal": chunk.get("journal", ""),
            "year": chunk.get("year"),
            "country": chunk.get("country", ""),
            "keywords": chunk.get("keywords", []),
            "mesh_terms": chunk.get("mesh_terms", []),
            
            # Section information (PRD-compliant types)
            "section_type": chunk["section_type"],
            "section_title": chunk["section_title"],
            "is_table": chunk.get("is_table", False),
            "table_caption": chunk.get("table_caption", ""),
            
            # Parent-child indexing fields
            "full_section_text": chunk.get("full_section_text", chunk["text"]),
            "section_id": chunk.get("section_id", ""),
            "section_weight": chunk.get("section_weight", 0.5),
            
            # Source and type
            "source": "pmc",
            "article_type": chunk.get("section_type", "other"),
            "has_full_text": True,
            
            # Preview for dashboard browsing (keep for UI)
            "text_preview": chunk["text"][:500],
        }
        
        # Create deterministic point ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pmc:{chunk_id}"))
        
        points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))
    
    return points, all_chunk_ids

def _create_sparse_encoder():
    """Create a BM25SparseEncoder instance if enabled."""
    use_sparse = IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25"
    if use_sparse and BM25SparseEncoder is not None:
        return BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )
    return None

def process_batch(client: QdrantClient, batch_files: List[Path], embedding_provider: EmbeddingProvider, processed_ids: set[str], processed_lock: threading.Lock, sparse_encoder=None) -> tuple[int, int]:
    """Process a batch of XML files in a single thread."""
    from scripts.ingestion_utils import parse_pmc_xml
    
    articles = []
    new_ids = []
    
    # 1. Parse all files in this batch
    for xml_path in batch_files:
        try:
            # Optimization: Fast skip based on filename
            # PMC files are usually named PMCxxxxxx.xml
            stem_id = xml_path.stem
            if xml_path.name.endswith(".xml.gz"):
                stem_id = stem_id.replace(".xml", "")
                
            # Check against checkpoint relative to lock - but `processed_ids` is a set, so generic read is thread-safe enough for hit check
            # We strictly only need lock for updates, or if we want perfect consistency. 
            # Given the speed requirement, direct read is better. "set" lookup is atomic in Python CPython.
            if stem_id in processed_ids:
                continue

            article = parse_pmc_xml(xml_path)
            if not article:
                continue
                
            source_id = str(article.get("pmcid") or article.get("pmid") or "").strip()
            
            # Double check with exact ID from content
            if not source_id:
                continue
                
            if source_id != stem_id and source_id in processed_ids:
                 continue

            # If we get here, it's a new article
            articles.append(article)
            new_ids.append(source_id)
        except Exception as e:
            logger.error("Failed to parse %s: %s", xml_path.name, e)

    if not articles:
        return 0, len(batch_files)

    # 2. Build points (includes embedding)
    points, chunk_ids = build_points(articles, embedding_provider, sparse_encoder=sparse_encoder)
    
    if not points:
        return 0, len(batch_files)

    # 3. Upsert to Qdrant
    try:
        upsert_with_retry(client, points)
        
        # 4. Update checkpoint
        append_checkpoint(new_ids)
        with processed_lock:
            processed_ids.update(new_ids)
            
        return len(points), len(batch_files) - len(articles) # inserted, skipped
    except Exception as e:
        logger.error("Failed to upsert batch: %s", e)
        return 0, 0


def run_ingestion(xml_dir: Path, articles_file: Optional[Path], embedding_provider: EmbeddingProvider) -> None:
    ensure_data_dirs()
    
    # Preload tokenizer once in main thread to avoid race conditions in workers
    logger.info("Preloading tokenizer...")
    try:
        from scripts.ingestion_utils import Chunker
        Chunker() # This triggers _load_tokenizer with the global lock
        logger.info("Tokenizer preloaded successfully.")
    except Exception as e:
        logger.warning("Tokenizer preload warning: %s", e)
    
    # Qdrant client is thread-safe (connection pooling)
    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    try:
        info = client.get_collection(IngestionConfig.COLLECTION_NAME)
        logger.info("Connected to %s (points=%s)", IngestionConfig.COLLECTION_NAME, info.points_count)
    except Exception:
        logger.warning("Collection %s not found or not accessible", IngestionConfig.COLLECTION_NAME)

    processed_ids = load_checkpoint()
    processed_lock = threading.Lock()
    logger.info("Already ingested from checkpoint: %s", len(processed_ids))

    all_xml_files = sorted(list(xml_dir.rglob("*.xml*")))
    if not all_xml_files:
        logger.warning("No XML files found in %s", xml_dir)
        return
        
    logger.info("Found %s XML files to process", len(all_xml_files))

    # Create shared sparse encoder (thread-safe, stateless)
    sparse_encoder = _create_sparse_encoder()

    # Calculate batches
    THREAD_BATCH_SIZE = IngestionConfig.BATCH_SIZE # e.g. 25
    MAX_WORKERS = IngestionConfig.MAX_WORKERS  # Respect configuration from .env
    
    # Create batches generator/list
    file_batches = [all_xml_files[i:i + THREAD_BATCH_SIZE] for i in range(0, len(all_xml_files), THREAD_BATCH_SIZE)]
    total_batches = len(file_batches)
    
    total_inserted = 0
    total_skipped = 0
    completed_batches = 0
    
    start_time = time.time()
    
    # Process in "super-batches" to avoid overloading the executor queue with 100k+ tasks
    # Submit 1000 batches at a time
    SUPER_BATCH_SIZE = 1000 
    
    logger.info("Processing in super-batches of %s (total batches: %s)", SUPER_BATCH_SIZE, total_batches)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(0, total_batches, SUPER_BATCH_SIZE):
            current_super_batch = file_batches[i : i + SUPER_BATCH_SIZE]
            
            future_to_batch = {
                executor.submit(process_batch, client, batch, embedding_provider, processed_ids, processed_lock, sparse_encoder): batch 
                for batch in current_super_batch
            }
            
            for future in as_completed(future_to_batch):
                inserted, skipped = future.result()
                total_inserted += inserted
                total_skipped += skipped
                completed_batches += 1
                
                if completed_batches % 10 == 0:
                     elapsed = time.time() - start_time
                     rate = total_inserted / elapsed if elapsed > 0 else 0
                     progress = (completed_batches / total_batches) * 100
                     logger.info("Progress: %.1f%% | Inserted: %s | Skipped: %s | Rate: %.2f docs/sec", 
                                 progress, total_inserted, total_skipped, rate)
            
            # Optional: explicit garbage collection if memory is tight
            future_to_batch.clear()

    elapsed = time.time() - start_time
    logger.info("PMC ingestion complete. Total Inserted: %s, Total Skipped: %s, Time: %.1fs", total_inserted, total_skipped, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PMC into self-hosted Qdrant with improved section/table handling")
    parser.add_argument("--articles-file", type=Path, default=None, help="Path to pre-extracted JSONL")
    parser.add_argument("--xml-dir", type=Path, default=IngestionConfig.PMC_XML_DIR, help="Directory with .xml/.xml.gz")
    parser.add_argument("--delete-source", action="store_true", help="Delete XML file after successful ingestion")
    args = parser.parse_args()

    provider = EmbeddingProvider()
    run_ingestion(args.xml_dir, args.articles_file, provider)
