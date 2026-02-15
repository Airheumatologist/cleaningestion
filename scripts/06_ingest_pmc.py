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
from ingestion_utils import Chunker, SectionFilter, EmbeddingProvider, upsert_with_retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pmc_ingested_improved_ids.txt"





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
    pmcid = str(article.get("pmcid") or "")
    pmid = str(article.get("pmid") or "")
    
    return {
        "doc_id": pmcid or pmid,
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": article.get("doi", ""),
        "title": (article.get("title") or "")[:300],
        "journal": article.get("journal", "")[:100],
        "year": article.get("year"),
        "authors": article.get("authors", [])[:20],
        "keywords": article.get("keywords", [])[:20],
        "mesh_terms": article.get("mesh_terms", [])[:30],
        "article_type": article.get("article_type", "")[:50],
        "evidence_grade": article.get("evidence_grade", ""),
        "evidence_level": article.get("evidence_level"),
        "country": article.get("country", ""),
        "affiliations": article.get("affiliations", [])[:10],
        "table_count": article.get("table_count", 0),
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

# ... (create_payload, create_chunks_from_article, build_points remain unchanged) ...

def process_batch(client: QdrantClient, batch_files: List[Path], embedding_provider: EmbeddingProvider, processed_ids: set[str], processed_lock: threading.Lock) -> tuple[int, int]:
    """Process a batch of XML files in a single thread."""
    from scripts.ingestion_utils import parse_pmc_xml
    
    articles = []
    new_ids = []
    
    # 1. Parse all files in this batch
    for xml_path in batch_files:
        try:
            article = parse_pmc_xml(xml_path)
            if not article:
                continue
                
            source_id = str(article.get("pmcid") or article.get("pmid") or "").strip()
            
            # Check against global processed set (with lock for safety if needed, though read overlaps are fine)
            # strictly speaking, we should check before parsing, but parsing is fast enough
            with processed_lock:
                if not source_id or source_id in processed_ids:
                    continue
            
            articles.append(article)
            new_ids.append(source_id)
        except Exception as e:
            logger.error("Failed to parse %s: %s", xml_path.name, e)

    if not articles:
        return 0, len(batch_files)

    # 2. Build points (includes embedding)
    points, chunk_ids = build_points(articles, embedding_provider)
    
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

    # Calculate batches
    THREAD_BATCH_SIZE = IngestionConfig.BATCH_SIZE # e.g. 25
    MAX_WORKERS = 4
    
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
                executor.submit(process_batch, client, batch, embedding_provider, processed_ids, processed_lock): batch 
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
