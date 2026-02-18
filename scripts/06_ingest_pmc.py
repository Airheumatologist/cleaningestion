#!/usr/bin/env python3
"""Ingest PMC content into self-hosted Qdrant with improved section and table handling."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Document, PointStruct

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import (
    SectionFilter,
    EmbeddingProvider,
    upsert_with_retry,
    validate_qdrant_collection_schema,
    get_chunker as get_shared_chunker,
    load_checkpoint as load_checkpoint_file,
    append_checkpoint as append_checkpoint_file,
)
from ingestion_utils import Chunker as BaseChunker  # Fallback

# Initialize logging FIRST before any imports that might use logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import enhanced utilities if available
ENHANCED_UTILS_AVAILABLE = False
try:
    from ingestion_utils_enhanced import (
        SemanticChunker, QualityValidator, 
        ContentDeduplicator, enhance_pmc_parsing
    )
    ENHANCED_UTILS_AVAILABLE = True
    logger.info("Using SemanticChunker for improved chunking")
except ImportError:
    SemanticChunker = None  # type: ignore
    QualityValidator = None  # type: ignore
    ContentDeduplicator = None  # type: ignore
    enhance_pmc_parsing = None  # type: ignore
    logger.warning("Enhanced ingestion utils not available, using base Chunker")

CHUNKER_CLASS = SemanticChunker if ENHANCED_UTILS_AVAILABLE else BaseChunker


import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import BM25SparseEncoder
spec = importlib.util.find_spec("src.bm25_sparse")
if spec is not None:
    from src.bm25_sparse import BM25SparseEncoder
else:
    BM25SparseEncoder = None  # type: ignore


SOURCE_PMC_OA = "pmc_oa"
SOURCE_PMC_AUTHOR = "pmc_author_manuscript"
SOURCE_MARKER_NAME = ".source"
CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pmc_ingested_ids.txt"
LEGACY_AUTHOR_CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "author_manuscript_ingested_ids.txt"
SUPPORTED_PMC_FILE_SUFFIXES = (".xml", ".nxml", ".xml.gz", ".nxml.gz")
SOURCE_ALIAS_MAP = {
    "pmc": SOURCE_PMC_OA,
    "pmc_oa": SOURCE_PMC_OA,
    "author_manuscript": SOURCE_PMC_AUTHOR,
    "pmc_author_manuscript": SOURCE_PMC_AUTHOR,
}

def _normalize_source_type(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    return SOURCE_ALIAS_MAP.get(normalized, SOURCE_PMC_OA)


def _checkpoint_id(source_type: str, doc_id: str) -> str:
    return f"{_normalize_source_type(source_type)}:{doc_id.strip()}"


def _resolve_legacy_checkpoint_line(line: str) -> str | None:
    value = line.strip()
    if not value:
        return None

    if ":" in value:
        prefix, doc_id = value.split(":", 1)
        if not doc_id.strip():
            return None
        return _checkpoint_id(prefix, doc_id)

    # Legacy format from historical pmc_ingested_ids.txt
    return _checkpoint_id(SOURCE_PMC_OA, value)


def _load_checkpoint_file(path: Path) -> set[str]:
    return {
        resolved
        for line in load_checkpoint_file(path)
        if (resolved := _resolve_legacy_checkpoint_line(line)) is not None
    }


def load_checkpoint() -> set[str]:
    ids = _load_checkpoint_file(CHECKPOINT_FILE)

    if LEGACY_AUTHOR_CHECKPOINT_FILE.exists():
        legacy_author_ids = {
            resolved
            for line in load_checkpoint_file(LEGACY_AUTHOR_CHECKPOINT_FILE)
            if (
                resolved := _resolve_legacy_checkpoint_line(
                    line if ":" in line else f"{SOURCE_PMC_AUTHOR}:{line}"
                )
            )
            is not None
        }
        if legacy_author_ids:
            logger.info(
                "Loaded %s legacy author manuscript checkpoint IDs from %s",
                len(legacy_author_ids),
                LEGACY_AUTHOR_CHECKPOINT_FILE,
            )
            ids.update(legacy_author_ids)

    return ids


def append_checkpoint(ids: Iterable[str]) -> None:
    append_checkpoint_file(CHECKPOINT_FILE, ids)


def _extract_stem_id(xml_path: Path) -> str:
    filename = xml_path.name
    lowered = filename.lower()

    if lowered.endswith(".xml.gz"):
        return filename[: -len(".xml.gz")].strip()
    if lowered.endswith(".nxml.gz"):
        return filename[: -len(".nxml.gz")].strip()
    if lowered.endswith(".xml"):
        return filename[: -len(".xml")].strip()
    if lowered.endswith(".nxml"):
        return filename[: -len(".nxml")].strip()

    return xml_path.stem.strip()


def _is_supported_pmc_file(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(SUPPORTED_PMC_FILE_SUFFIXES)


def _detect_source_type(xml_path: Path, xml_root: Path) -> str:
    current = xml_path.parent.resolve()
    root = xml_root.resolve()

    while True:
        marker_path = current / SOURCE_MARKER_NAME
        if marker_path.exists():
            try:
                marker_value = marker_path.read_text(encoding="utf-8").strip()
                return _normalize_source_type(marker_value)
            except Exception:
                logger.debug("Unable to parse source marker at %s", marker_path)

        if current == root or current.parent == current:
            break
        current = current.parent

    lowered_parts = {part.lower() for part in xml_path.parts}
    if "author_manuscript" in lowered_parts or "manuscript" in lowered_parts:
        return SOURCE_PMC_AUTHOR
    return SOURCE_PMC_OA


def create_payload(article: Dict[str, Any], source_type: str) -> Dict[str, Any]:
    """Create base payload for a PMC article with source differentiation."""
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
    
    normalized_source = _normalize_source_type(source_type)
    identifiers_nihms_id = identifiers.get("nihms_id")
    article_nihms_id = article.get("nihms_id")
    nihms_id = identifiers_nihms_id or article_nihms_id

    payload = {
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
        "evidence_term": article.get("evidence_term"),
        "evidence_source": article.get("evidence_source"),
        "country": country,
        "table_count": len(tables),
        "has_full_text": content_flags.get("has_full_text") if content_flags else article.get("has_full_text", False),
        "is_open_access": content_flags.get("is_open_access") if content_flags else article.get("is_open_access", False),
        "source": normalized_source,
        "source_family": "pmc",
        "content_type": "full_text" if normalized_source == SOURCE_PMC_OA else "author_manuscript",
    }

    if normalized_source == SOURCE_PMC_AUTHOR:
        payload["is_author_manuscript"] = True
        if nihms_id:
            payload["nihms_id"] = nihms_id
    else:
        payload["is_author_manuscript"] = False

    return payload


from concurrent.futures import ThreadPoolExecutor, as_completed


def create_chunks_from_article(article: Dict[str, Any], chunker: Any) -> List[Dict[str, Any]]:
    """
    Create multiple chunks from an article: sections + tables with word-based chunking.
    Updated for PRD v1.0 compliant article structure.
    Includes full metadata propagation and parent-child ready fields.
    """
    from ingestion_utils import generate_section_id, get_section_weight
    
    chunks = []
    
    # Handle both PRD-compliant structure and legacy structure
    metadata = article.get("metadata", {})
    content = article.get("content", {})
    
    source_type = _normalize_source_type(article.get("_source_type"))
    article_payload = create_payload(article, source_type)

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
        "article_type": article_payload.get("article_type", ""),
        "evidence_grade": article_payload.get("evidence_grade", ""),
        "evidence_level": article_payload.get("evidence_level"),
        "evidence_term": article_payload.get("evidence_term"),
        "evidence_source": article_payload.get("evidence_source"),
        "source": article_payload["source"],
        "source_family": article_payload["source_family"],
        "content_type": article_payload["content_type"],
        "is_author_manuscript": article_payload["is_author_manuscript"],
        "nihms_id": article_payload.get("nihms_id"),
        "has_full_text": article_payload["has_full_text"],
        "is_open_access": article_payload["is_open_access"],
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
    
    # 2. Section chunks with word-based splitting
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
        
        # Use Chunker for word-based chunking
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
    
    # 3. Table chunks with word-based chunking for large tables
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
        
        # Use Chunker for word-based chunking of large tables
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
    # Use shared chunker instance for efficiency
    chunker = get_shared_chunker(
        chunker_class=CHUNKER_CLASS,
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
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

    sparse_vectors: List[Optional[Any]] = []
    if sparse_encoder is not None:
        try:
            if hasattr(sparse_encoder, "encode_batch"):
                sparse_vectors = sparse_encoder.encode_batch(all_texts)
            else:
                sparse_vectors = [sparse_encoder.encode_document(text) for text in all_texts]
        except Exception as e:
            logger.warning("Sparse encoding failed for batch: %s", e)
            sparse_vectors = [None] * len(all_texts)
    
    # Create points
    for idx, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
        chunk_id = chunk["chunk_id"]
        all_chunk_ids.append(chunk_id)
        source_type = _normalize_source_type(chunk.get("source"))
        
        # Create sparse vector if enabled
        vector_data: Any = {"dense": vector}
        if sparse_encoder is not None and idx < len(sparse_vectors) and sparse_vectors[idx] is not None:
            vector_data["sparse"] = sparse_vectors[idx]
        
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
            "evidence_grade": chunk.get("evidence_grade", ""),
            "evidence_level": chunk.get("evidence_level"),
            "evidence_term": chunk.get("evidence_term"),
            "evidence_source": chunk.get("evidence_source"),
            
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
            "source": source_type,
            "source_family": chunk.get("source_family", "pmc"),
            "content_type": chunk.get(
                "content_type",
                "author_manuscript" if source_type == SOURCE_PMC_AUTHOR else "full_text",
            ),
            "is_author_manuscript": bool(chunk.get("is_author_manuscript", False)),
            "nihms_id": chunk.get("nihms_id"),
            "article_type": chunk.get("article_type", "other"),
            "has_full_text": chunk.get("has_full_text", True),
            "is_open_access": chunk.get("is_open_access"),
            
            # Preview for dashboard browsing (keep for UI)
            "text_preview": chunk["text"][:500],
        }
        
        # Create deterministic point ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_type}:{chunk_id}"))
        
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

def process_batch(
    client: QdrantClient,
    batch_files: List[Path],
    embedding_provider: EmbeddingProvider,
    processed_ids: set[str],
    processed_lock: threading.Lock,
    xml_root: Path,
    sparse_encoder=None,
    delete_source: bool = False,
) -> tuple[int, int]:
    """Process a batch of XML files in a single thread."""
    from scripts.ingestion_utils import parse_pmc_xml
    
    articles: List[Dict[str, Any]] = []
    new_ids: List[str] = []
    processed_files: List[Path] = []  # Track files for potential deletion
    
    # 1. Parse all files in this batch
    for xml_path in batch_files:
        try:
            source_type = _detect_source_type(xml_path, xml_root)
            stem_id = _extract_stem_id(xml_path)
            candidate_ids = set()
            if stem_id:
                candidate_ids.add(_checkpoint_id(source_type, stem_id))
            if stem_id and not stem_id.upper().startswith("PMC"):
                candidate_ids.add(_checkpoint_id(source_type, f"PMC{stem_id}"))
            if any(value in processed_ids for value in candidate_ids):
                continue

            strict_oa = source_type == SOURCE_PMC_OA
            article = parse_pmc_xml(
                xml_path,
                require_pmid=strict_oa,
                require_open_access=strict_oa,
            )
            if not article:
                continue

            article["_source_type"] = source_type
                
            source_id = str(article.get("pmcid") or article.get("pmid") or "").strip()
            
            if not source_id:
                continue

            namespaced_id = _checkpoint_id(source_type, source_id)
            if namespaced_id in processed_ids:
                continue

            # If we get here, it's a new article
            articles.append(article)
            new_ids.append(namespaced_id)
            processed_files.append(xml_path)  # Track for potential deletion
        except Exception as e:
            logger.error("Failed to parse %s: %s", xml_path.name, e)

    if not articles:
        return 0, len(batch_files)

    # 2. Build points (includes embedding)
    points, _chunk_ids = build_points(articles, embedding_provider, sparse_encoder=sparse_encoder)
    
    if not points:
        return 0, len(batch_files)

    # 3. Upsert to Qdrant
    try:
        upsert_with_retry(client, points)
        
        # 4. Update checkpoint
        checkpoint_ids = list(dict.fromkeys(new_ids))
        append_checkpoint(checkpoint_ids)
        with processed_lock:
            processed_ids.update(checkpoint_ids)
        
        # 5. Delete source files if requested (after successful ingestion)
        if delete_source:
            for xml_path in processed_files:
                try:
                    xml_path.unlink()
                    logger.debug("Deleted source file: %s", xml_path)
                except OSError as e:
                    logger.warning("Failed to delete %s: %s", xml_path, e)
            
        return len(points), len(batch_files) - len(articles) # inserted, skipped
    except Exception as e:
        logger.error("Failed to upsert batch: %s", e)
        return 0, 0


def run_ingestion(xml_dir: Path, embedding_provider: EmbeddingProvider, delete_source: bool = False) -> None:
    ensure_data_dirs()
    
    if delete_source:
        logger.warning("Source file deletion enabled: XML/NXML files will be deleted after successful ingestion")
    
    # Preload shared chunker once in main thread to avoid race conditions in workers
    logger.info("Preloading shared chunker...")
    try:
        chunker = get_shared_chunker(
            chunker_class=CHUNKER_CLASS,
            chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
            overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
        )
        logger.info("Shared chunker ready (using %s).", chunker.__class__.__name__)
    except Exception as e:
        logger.warning("Shared chunker preload warning: %s", e)
    
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
        validate_qdrant_collection_schema(client, IngestionConfig.COLLECTION_NAME)
    except Exception as e:
        logger.error("Failed Qdrant schema preflight for %s: %s", IngestionConfig.COLLECTION_NAME, e)
        return

    processed_ids = load_checkpoint()
    processed_lock = threading.Lock()
    logger.info("Already ingested from checkpoint: %s", len(processed_ids))

    all_xml_files = sorted(
        {
            path
            for pattern in ("*.xml*", "*.nxml*")
            for path in xml_dir.rglob(pattern)
            if _is_supported_pmc_file(path)
        }
    )
    if not all_xml_files:
        logger.warning("No XML/NXML files found in %s", xml_dir)
        return
        
    logger.info("Found %s XML/NXML files to process", len(all_xml_files))

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
                executor.submit(
                    process_batch,
                    client,
                    batch,
                    embedding_provider,
                    processed_ids,
                    processed_lock,
                    xml_dir,
                    sparse_encoder,
                    delete_source,
                ): batch
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
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=IngestionConfig.PMC_XML_DIR,
        help="Directory with .xml/.nxml (optionally .gz)",
    )
    parser.add_argument("--delete-source", action="store_true", help="Delete XML/NXML file after successful ingestion")
    args = parser.parse_args()

    provider = EmbeddingProvider()
    run_ingestion(args.xml_dir, provider, delete_source=args.delete_source)
