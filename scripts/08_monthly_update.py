#!/usr/bin/env python3
"""Monthly incremental updater for self-hosted Medical RAG."""

from __future__ import annotations

import argparse
import ftplib
import gzip
import json
import logging
import os
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config_ingestion import IngestionConfig, ensure_data_dirs
from src.bm25_sparse import BM25SparseEncoder
from ingestion_utils import (
    Chunker as BaseChunker, 
    EmbeddingProvider, 
    upsert_with_retry,
    generate_section_id,
    get_section_weight,
    get_chunker as get_shared_chunker,
    classify_evidence_metadata,
    get_evidence_level,
    extract_gov_affiliations_from_pubmed_xml,
)
from pubmed_publication_filters import (
    is_target_article,
    map_publication_type,
)

# Import enhanced utilities for semantic chunking
try:
    from ingestion_utils_enhanced import SemanticChunker, QualityValidator
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    SemanticChunker = None  # type: ignore
    QualityValidator = None  # type: ignore
    ENHANCED_UTILS_AVAILABLE = False

CHUNKER_CLASS = SemanticChunker if ENHANCED_UTILS_AVAILABLE else BaseChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PUBMED_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
PUBMED_UPDATE_DIR = "/pubmed/updatefiles/"
PROCESSED_TRACKER = IngestionConfig.DATA_DIR / "processed_updates.json"
CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pubmed_ingested_ids.txt"
UPDATE_DOWNLOAD_DIR = IngestionConfig.DATA_DIR / "pubmed_updatefiles"
PUBMED_CHECKPOINT_NAMESPACE = "pubmed"

# PubMed filtering constants
DEFAULT_MIN_YEAR = 2015

def load_processed_files() -> Set[str]:
    if not PROCESSED_TRACKER.exists():
        return set()
    try:
        return set(json.loads(PROCESSED_TRACKER.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_processed_files(processed: Set[str]) -> None:
    PROCESSED_TRACKER.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_TRACKER.write_text(json.dumps(sorted(processed), indent=2), encoding="utf-8")


def _resolve_checkpoint_pmid(line: str) -> Optional[str]:
    """Resolve checkpoint line to PMID, handling namespaced and legacy formats."""
    value = line.strip()
    if not value:
        return None

    # Namespaced format (current baseline ingestion)
    if ":" in value:
        namespace, pmid = value.split(":", 1)
        if namespace != PUBMED_CHECKPOINT_NAMESPACE:
            return None
        value = pmid

    value = value.strip()
    return value if value else None


def load_baseline_pmids(path: Path) -> Set[str]:
    """Load baseline PubMed checkpoint PMIDs for cross-pipeline deduplication."""
    if not path.exists():
        return set()

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.warning("Failed to load baseline checkpoint %s: %s", path, exc)
        return set()

    return {pmid for line in lines if (pmid := _resolve_checkpoint_pmid(line)) is not None}


def list_pubmed_update_files(max_files: Optional[int]) -> List[str]:
    ftp = ftplib.FTP(PUBMED_FTP_HOST, timeout=120)
    ftp.login()
    ftp.cwd(PUBMED_UPDATE_DIR)
    files = sorted(f for f in ftp.nlst() if f.endswith(".xml.gz"))
    ftp.quit()
    if max_files:
        files = files[-max_files:]
    return files


def download_pubmed_file(file_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / file_name

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    ftp = ftplib.FTP(PUBMED_FTP_HOST, timeout=120)
    ftp.login()
    ftp.cwd(PUBMED_UPDATE_DIR)
    with local_path.open("wb") as f:
        ftp.retrbinary(f"RETR {file_name}", f.write, blocksize=1024 * 1024)
    ftp.quit()
    return local_path


def parse_pubmed_update_xml(
    xml_gz_path: Path, min_year: int = DEFAULT_MIN_YEAR
) -> Iterable[Dict[str, Any]]:
    with gzip.open(xml_gz_path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for event, elem in context:
            if elem.tag != "PubmedArticle":
                continue

            pmid_elem = elem.find(".//PMID")
            title_elem = elem.find(".//ArticleTitle")
            abstract_parts = elem.findall(".//Abstract/AbstractText")

            pmid = (pmid_elem.text or "").strip() if pmid_elem is not None else ""
            title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
            abstract = " ".join("".join(part.itertext()).strip() for part in abstract_parts if part is not None).strip()

            # Basic presence check (line 132 equivalent)
            if not pmid or not title or not abstract:
                elem.clear()
                continue

            # Filter: Abstract length check (matches baseline line 748-749)
            if len(abstract) < 50:
                elem.clear()
                continue

            # Extract publication types early for filtering
            pub_types = []
            for pt in elem.findall(".//PublicationType"):
                if pt.text:
                    pub_types.append(pt.text.strip())

            # Filter: Target publication types (matches baseline line 726)
            if not is_target_article(pub_types):
                elem.clear()
                continue

            # Extract year from PubDate
            year = None
            year_elem = elem.find(".//PubDate/Year")
            if year_elem is not None and year_elem.text and year_elem.text.isdigit():
                year = int(year_elem.text)

            # Try MedlineDate if no Year element (matches baseline behavior)
            if year is None:
                medline_date_elem = elem.find(".//PubDate/MedlineDate")
                if medline_date_elem is not None and medline_date_elem.text:
                    import re
                    match = re.search(r'(\d{4})', medline_date_elem.text)
                    if match:
                        year = int(match.group(1))

            # Filter: Minimum year check (matches baseline line 740)
            if year is None or year < min_year:
                elem.clear()
                continue

            journal_elem = elem.find(".//Journal/Title")
            journal = (journal_elem.text or "").strip() if journal_elem is not None and journal_elem.text else ""

            doi = None
            for article_id in elem.findall(".//ArticleId"):
                if article_id.attrib.get("IdType") == "doi" and article_id.text:
                    doi = article_id.text.strip()
                    break

            article_type = map_publication_type(pub_types)
            if not article_type:
                elem.clear()
                continue
            evidence = classify_evidence_metadata(
                article_type=article_type,
                pub_types=pub_types,
                abstract=abstract,
            )
            evidence_grade = evidence["grade"]
            evidence_level = evidence["level_1_4"]
            evidence_term = evidence["matched_term"]
            evidence_source = evidence["matched_from"]

            # Extract government affiliations
            is_gov_affiliated, gov_agencies = extract_gov_affiliations_from_pubmed_xml(elem)

            yield {
                "pmid": pmid,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "article_type": article_type,
                "publication_type": pub_types[:5],
                "evidence_grade": evidence_grade,
                "evidence_level": evidence_level,
                "evidence_term": evidence_term,
                "evidence_source": evidence_source,
                "source": "pubmed_abstract",
                "has_full_text": False,
                "is_gov_affiliated": is_gov_affiliated,
                "gov_agencies": gov_agencies,
            }

            elem.clear()


def build_points(batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider, 
                 sparse_encoder: Optional[Any] = None,
                 validate_chunks: bool = True, dedup_chunks: bool = True) -> tuple[List[PointStruct], List[str]]:
    """Build Qdrant points from article batch with semantic chunking.
    
    Args:
        batch: List of parsed articles
        embedding_provider: Provider for generating embeddings
        sparse_encoder: Optional sparse encoder for hybrid search (created once and reused)
        validate_chunks: Whether to validate chunk quality before ingestion
        dedup_chunks: Whether to deduplicate chunks within batch
    """
    # Use shared chunker instance (SemanticChunker if available)
    chunker = get_shared_chunker(
        chunker_class=CHUNKER_CLASS,
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
    )

    points: List[PointStruct] = []
    pmids: List[str] = []
    
    # Batch collection
    all_chunks_text: List[str] = []
    chunk_metadata: List[Dict[str, Any]] = []

    for article in batch:
        pmid = str(article.get("pmid") or "").strip()
        if not pmid:
            continue
            
        pmids.append(pmid)
        
        title = article.get("title", "")
        abstract = article.get("abstract", "")
        
        # Combine title + abstract for chunking with proper format
        text = f"Title: {title}\n\nAbstract: {abstract}" if title or abstract else ""
        if len(text) < 50:
            continue
            
        # Generate section ID for abstract
        section_id = generate_section_id(pmid, "Abstract")
        
        # Chunk the content using shared chunker
        text_chunks = chunker.chunk_text(text)
        
        for i, chunk in enumerate(text_chunks):
            chunk_text = chunk["text"]
            all_chunks_text.append(chunk_text)
            
            # Extract chunk abstract (remove title prefix if present)
            chunk_abstract = chunk_text
            if chunk_text.startswith(f"Title: {title}\n\nAbstract: "):
                chunk_abstract = chunk_text[len(f"Title: {title}\n\nAbstract: "):]
            elif chunk_text.startswith("Title: "):
                chunk_abstract = chunk_text[chunk_text.find("\n\nAbstract: ") + 12:]
            
            # Build enhanced payload consistent with other ingestion scripts
            payload = {
                # Core identifiers
                "doc_id": pmid,
                "chunk_id": f"{pmid}_abstract_chunk_{i}" if len(text_chunks) > 1 else f"{pmid}_abstract",
                "pmid": pmid,
                "pmcid": None,
                "doi": article.get("doi"),
                
                # CRITICAL: Full text for retriever
                "page_content": chunk_text,
                
                # Content
                "title": title if i == 0 else f"{title} (cont.)" if title else f"Abstract (Part {i+1})",
                "abstract": chunk_abstract,
                
                # Chunking metadata
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "token_count": chunk.get("token_count", 0),
                
                # Publication info
                "year": article.get("year"),
                "journal": article.get("journal", ""),
                "article_type": article.get("article_type", "research_article"),
                "publication_type": article.get("publication_type", []),
                
                # Evidence
                "evidence_grade": article.get("evidence_grade", "C"),
                "evidence_level": article.get("evidence_level", get_evidence_level(article.get("evidence_grade", "C"))),
                "evidence_term": article.get("evidence_term"),
                "evidence_source": article.get("evidence_source", "fallback"),
                
                # Government Affiliation
                "is_gov_affiliated": article.get("is_gov_affiliated", False),
                "gov_agencies": article.get("gov_agencies", []),
                
                # Parent-child indexing fields
                "section_title": f"Abstract (Part {i+1}/{len(text_chunks)})" if len(text_chunks) > 1 else "Abstract",
                "section_type": "abstract",
                "section_id": section_id,
                "section_weight": get_section_weight("abstract") - (i * 0.05),  # Slight decay for later chunks
                "full_section_text": text,
                
                # Source and type
                "source": article.get("source", "pubmed_abstract"),
                "has_full_text": False,
                "content_type": "abstract",
                
                # Preview for dashboard
                "text_preview": chunk_text[:500],
            }
            
            chunk_metadata.append(payload)

    if not all_chunks_text:
        return [], pmids

    # Embed all chunks using shared embedding provider
    vectors = embedding_provider.embed_batch(all_chunks_text)
    
    # Batch encode sparse vectors (if enabled)
    sparse_vectors = []
    if sparse_encoder:
        try:
            sparse_vectors = sparse_encoder.encode_batch(all_chunks_text)
        except Exception as e:
            logger.warning("Sparse encoding failed for batch: %s", e)
            sparse_vectors = [None] * len(all_chunks_text)
    
    for i, vector in enumerate(vectors):
        payload = chunk_metadata[i]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed:{payload['chunk_id']}"))
        vector_data: Any = {"dense": vector}
        
        # Add sparse vector if available
        if sparse_vectors and i < len(sparse_vectors) and sparse_vectors[i]:
            vector_data["sparse"] = sparse_vectors[i].model_dump() if hasattr(sparse_vectors[i], 'model_dump') else sparse_vectors[i]
        
        points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))

    return points, pmids


def ingest_pubmed_updates(
    client: QdrantClient, 
    embedding_provider: EmbeddingProvider, 
    max_files: Optional[int],
    min_year: int = DEFAULT_MIN_YEAR
) -> int:
    processed = load_processed_files()
    baseline_pmids = load_baseline_pmids(CHECKPOINT_FILE)
    logger.info("Loaded baseline PubMed checkpoint IDs=%d from %s", len(baseline_pmids), CHECKPOINT_FILE)

    all_files = list_pubmed_update_files(max_files=max_files)
    new_files = [f for f in all_files if f not in processed]

    logger.info("PubMed update files total=%s new=%s", len(all_files), len(new_files))
    
    # Log chunking configuration
    chunker = get_shared_chunker(
        chunker_class=CHUNKER_CLASS,
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
    )
    logger.info("Chunking config: size=%d, overlap=%d, chunker=%s",
                IngestionConfig.CHUNK_SIZE_TOKENS,
                IngestionConfig.CHUNK_OVERLAP_TOKENS,
                chunker.__class__.__name__)
    
    # Create sparse encoder once (reused for all batches)
    sparse_encoder: Optional[Any] = None
    if IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )
        logger.info("BM25 sparse encoder initialized for hybrid search")

    inserted = 0
    for file_name in new_files:
        logger.info("Processing update file: %s", file_name)
        local_file = download_pubmed_file(file_name, UPDATE_DOWNLOAD_DIR)

        batch: List[Dict[str, Any]] = []
        file_inserted = 0
        skipped_from_baseline = 0
        for article in parse_pubmed_update_xml(local_file, min_year=min_year):
            pmid = str(article.get("pmid") or "").strip()
            if pmid in baseline_pmids:
                skipped_from_baseline += 1
                continue

            batch.append(article)
            if len(batch) < IngestionConfig.BATCH_SIZE:
                continue
            points, _ = build_points(batch, embedding_provider, sparse_encoder,
                                     validate_chunks=ENHANCED_UTILS_AVAILABLE, dedup_chunks=ENHANCED_UTILS_AVAILABLE)
            if points:
                upsert_with_retry(client, points)
                inserted += len(points)
                file_inserted += len(points)
                time.sleep(0.5)  # Throttle: allow RocksDB compaction
            batch.clear()

        if batch:
            points, _ = build_points(batch, embedding_provider, sparse_encoder,
                                     validate_chunks=ENHANCED_UTILS_AVAILABLE, dedup_chunks=ENHANCED_UTILS_AVAILABLE)
            if points:
                upsert_with_retry(client, points)
                inserted += len(points)
                file_inserted += len(points)

        processed.add(file_name)
        save_processed_files(processed)
        local_file.unlink(missing_ok=True)
        logger.info("Completed file: %s inserted=%d skipped_baseline=%d", file_name, file_inserted, skipped_from_baseline)

    return inserted


def run_dailymed_refresh() -> None:
    """Run DailyMed incremental update using weekly updates."""
    logger.info("Starting DailyMed incremental refresh (weekly updates)")
    set_id_manifest = IngestionConfig.DAILYMED_XML_DIR / "dailymed_last_update_set_ids.txt"
    # Download last 4 weeks of updates to catch any missed changes
    subprocess.run([
        sys.executable, "scripts/03_download_dailymed.py", 
        "--output-dir", str(IngestionConfig.DAILYMED_XML_DIR),
        "--update-type", "weekly",
        "--weeks-back", "4",
        "--set-id-manifest", str(set_id_manifest),
    ], check=True)
    # Clear checkpoint entries for updated labels so they get re-ingested
    subprocess.run([
        sys.executable, "scripts/04_prepare_dailymed_updates.py",
        "--xml-dir", str(IngestionConfig.DAILYMED_XML_DIR),
        "--set-id-manifest", str(set_id_manifest),
    ], check=True)
    # Ingest - updated labels will now be processed
    subprocess.run([
        sys.executable, "scripts/07_ingest_dailymed.py", 
        "--xml-dir", str(IngestionConfig.DAILYMED_XML_DIR)
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly incremental update for self-hosted RAG")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N newest PubMed update files")
    parser.add_argument("--skip-pubmed", action="store_true")
    parser.add_argument("--skip-dailymed", action="store_true")
    parser.add_argument(
        "--min-year",
        type=int,
        default=DEFAULT_MIN_YEAR,
        help=f"Minimum publication year (default: {DEFAULT_MIN_YEAR})"
    )
    args = parser.parse_args()
    min_year = args.min_year

    ensure_data_dirs()

    # Validate required API keys
    if not os.getenv("DEEPINFRA_API_KEY"):
        logger.error("DEEPINFRA_API_KEY not set!")
        sys.exit(1)
    
    if not IngestionConfig.QDRANT_API_KEY:
        logger.error("QDRANT_API_KEY not set!")
        sys.exit(1)

    embedding_provider = EmbeddingProvider()
    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    started = time.time()
    total_inserted = 0

    if not args.skip_pubmed:
        total_inserted += ingest_pubmed_updates(client, embedding_provider, max_files=args.max_files, min_year=min_year)

    if not args.skip_dailymed:
        run_dailymed_refresh()

    info = client.get_collection(IngestionConfig.COLLECTION_NAME)
    logger.info("Monthly update complete inserted=%s total_points=%s elapsed=%.1fs", total_inserted, info.points_count, time.time() - started)


if __name__ == "__main__":
    main()
