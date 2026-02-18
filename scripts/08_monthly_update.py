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
    detect_evidence_grade,
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
UPDATE_DOWNLOAD_DIR = IngestionConfig.DATA_DIR / "pubmed_updatefiles"

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


def infer_article_type(pub_types: List[str]) -> str:
    lowered = [p.lower() for p in pub_types]
    if any("guideline" in p for p in lowered):
        return "guideline"
    if any("meta-analysis" in p or "systematic review" in p for p in lowered):
        return "systematic_review"
    if any("clinical trial" in p or "randomized" in p for p in lowered):
        return "clinical_trial"
    if any("review" in p for p in lowered):
        return "review_article"
    if any("case report" in p for p in lowered):
        return "case_report"
    return "research_article"


def parse_pubmed_update_xml(xml_gz_path: Path) -> Iterable[Dict[str, Any]]:
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

            if not pmid or not title or not abstract:
                elem.clear()
                continue

            year = None
            year_elem = elem.find(".//PubDate/Year")
            if year_elem is not None and year_elem.text and year_elem.text.isdigit():
                year = int(year_elem.text)

            journal_elem = elem.find(".//Journal/Title")
            journal = (journal_elem.text or "").strip() if journal_elem is not None and journal_elem.text else ""

            pub_types = []
            for pt in elem.findall(".//PublicationType"):
                if pt.text:
                    pub_types.append(pt.text.strip())

            doi = None
            for article_id in elem.findall(".//ArticleId"):
                if article_id.attrib.get("IdType") == "doi" and article_id.text:
                    doi = article_id.text.strip()
                    break

            article_type = infer_article_type(pub_types)
            evidence_grade = detect_evidence_grade(
                article_type=article_type,
                pub_types=pub_types,
                abstract=abstract,
            )

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
                "source": "pubmed_update",
                "has_full_text": False,
            }

            elem.clear()


def build_points(batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider) -> tuple[List[PointStruct], List[str]]:
    """Build Qdrant points from article batch with semantic chunking."""
    # Use shared chunker instance (SemanticChunker if available)
    chunker = get_shared_chunker(
        chunker_class=CHUNKER_CLASS,
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
    )
    
    sparse_encoder = None
    if IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )

    points: List[PointStruct] = []
    pmids: List[str] = []
    
    # Batch collection
    all_chunks_text: List[str] = []
    all_chunks_sparse = []
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
            all_chunks_sparse.append(sparse_encoder.encode_document(chunk_text) if sparse_encoder is not None else None)
            
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
                "evidence_level": {"A": 1, "B": 2, "C": 3, "D": 4}.get(article.get("evidence_grade", "C"), 3),
                
                # Parent-child indexing fields
                "section_title": f"Abstract (Part {i+1}/{len(text_chunks)})" if len(text_chunks) > 1 else "Abstract",
                "section_type": "abstract",
                "section_id": section_id,
                "section_weight": get_section_weight("abstract") - (i * 0.05),  # Slight decay for later chunks
                "full_section_text": text,
                
                # Source and type
                "source": article.get("source", "pubmed_update"),
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
    for i, vector in enumerate(vectors):
        payload = chunk_metadata[i]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed_update:{payload['chunk_id']}"))
        sparse_vector = all_chunks_sparse[i]
        vector_data: Any = {"dense": vector}
        if sparse_vector is not None:
            vector_data["sparse"] = sparse_vector
        
        points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))

    return points, pmids


def ingest_pubmed_updates(client: QdrantClient, embedding_provider: EmbeddingProvider, max_files: Optional[int]) -> int:
    processed = load_processed_files()
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

    inserted = 0
    for file_name in new_files:
        logger.info("Processing update file: %s", file_name)
        local_file = download_pubmed_file(file_name, UPDATE_DOWNLOAD_DIR)

        batch: List[Dict[str, Any]] = []
        for article in parse_pubmed_update_xml(local_file):
            batch.append(article)
            if len(batch) < IngestionConfig.BATCH_SIZE:
                continue
            points, _ = build_points(batch, embedding_provider)
            if points:
                upsert_with_retry(client, points)
                inserted += len(points)
                time.sleep(0.5)  # Throttle: allow RocksDB compaction
            batch.clear()

        if batch:
            points, _ = build_points(batch, embedding_provider)
            if points:
                upsert_with_retry(client, points)
                inserted += len(points)

        processed.add(file_name)
        save_processed_files(processed)
        local_file.unlink(missing_ok=True)

    return inserted


def run_dailymed_refresh() -> None:
    logger.info("Starting DailyMed refresh")
    subprocess.run([sys.executable, "scripts/03_download_dailymed.py", "--output-dir", str(IngestionConfig.DAILYMED_XML_DIR)], check=True)
    subprocess.run([sys.executable, "scripts/07_ingest_dailymed.py", "--xml-dir", str(IngestionConfig.DAILYMED_XML_DIR)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly incremental update for self-hosted RAG")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N newest PubMed update files")
    parser.add_argument("--skip-pubmed", action="store_true")
    parser.add_argument("--skip-dailymed", action="store_true")
    args = parser.parse_args()

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
        total_inserted += ingest_pubmed_updates(client, embedding_provider, max_files=args.max_files)

    if not args.skip_dailymed:
        run_dailymed_refresh()

    info = client.get_collection(IngestionConfig.COLLECTION_NAME)
    logger.info("Monthly update complete inserted=%s total_points=%s elapsed=%.1fs", total_inserted, info.points_count, time.time() - started)


if __name__ == "__main__":
    main()
