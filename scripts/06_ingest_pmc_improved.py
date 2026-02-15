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
from ingestion_utils import Chunker, SectionFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pmc_ingested_improved_ids.txt"


class EmbeddingProvider:
    """Support for local, Cohere, and cloud inference embeddings."""
    
    def __init__(self) -> None:
        self.provider = IngestionConfig.EMBEDDING_PROVIDER.lower().strip()
        self.model = IngestionConfig.EMBEDDING_MODEL
        self.local_encoder = None
        self.cohere_client = None

        if self.provider == "cohere":
            import cohere
            api_key = IngestionConfig.COHERE_API_KEY
            if not api_key:
                raise ValueError("COHERE_API_KEY not set - required for cohere embedding provider")
            self.cohere_client = cohere.ClientV2(api_key=api_key)
            logger.info("✅ Cohere embedding provider initialized (model: %s)", self.model)
        elif self.provider == "local":
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model: %s", self.model)
            self.local_encoder = SentenceTransformer(self.model)

    def use_cloud(self) -> bool:
        return self.provider == "qdrant_cloud_inference"

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "cohere":
            return self._embed_cohere(texts)
        elif self.local_encoder is not None:
            vectors = self.local_encoder.encode(texts, normalize_embeddings=True, batch_size=IngestionConfig.EMBEDDING_BATCH_SIZE)
            return [v.tolist() for v in vectors]
        else:
            raise RuntimeError("No embedding provider available")

    def _embed_cohere(self, texts: List[str]) -> List[List[float]]:
        """Embed using Cohere API."""
        try:
            response = self.cohere_client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document",
                embedding_types=["float"]
            )
            return [embedding for embedding in response.embeddings.float]
        except Exception as e:
            logger.error("Cohere embedding failed: %s", e)
            raise


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


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return {line.strip() for line in CHECKPOINT_FILE.read_text().splitlines() if line.strip()}
    return set()


def append_checkpoint(ids: Iterable[str]) -> None:
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        for value in ids:
            f.write(f"{value}\n")


def create_chunks_from_article(article: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create multiple chunks from an article: sections + tables."""
    chunks = []
    doc_id = str(article.get("pmcid") or article.get("pmid") or "")
    title = article.get("title", "")
    
    # 1. Abstract chunk (most important)
    abstract = article.get("abstract", "")
    if abstract and len(abstract) > 50:
        chunks.append({
            "chunk_id": f"{doc_id}_abstract",
            "doc_id": doc_id,
            "text": f"Title: {title}\n\nAbstract: {abstract}",
            "section_type": "abstract",
            "section_title": "Abstract",
            "is_table": False,
        })
    
    # 2. Section chunks
    sections = article.get("structured_sections", [])
    for i, section in enumerate(sections):
        sec_text = section.get("text", "")
        if len(sec_text) < 50:
            continue
            
        # Section context with title
        section_context = f"Title: {title}\n\nSection: {section.get('title', 'Body')}\n\n{sec_text}"
        
        # Split long sections into smaller chunks (5000 chars each)
        max_chunk_size = 5000
        if len(section_context) > max_chunk_size:
            for j in range(0, len(section_context), max_chunk_size):
                chunk_text = section_context[j:j+max_chunk_size]
                chunks.append({
                    "chunk_id": f"{doc_id}_sec{i}_part{j//max_chunk_size}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "section_type": section.get("type", "body"),
                    "section_title": section.get("title", "Body"),
                    "is_table": False,
                })
        else:
            chunks.append({
                "chunk_id": f"{doc_id}_sec{i}",
                "doc_id": doc_id,
                "text": section_context,
                "section_type": section.get("type", "body"),
                "section_title": section.get("title", "Body"),
                "is_table": False,
            })
    
    # 3. Table chunks (NEW - embed each table separately)
    tables = article.get("tables", [])
    for i, table in enumerate(tables):
        # Use row-by-row format for better semantic search
        table_text = table.get("row_by_row", "")
        caption = table.get("caption", "")
        
        if not table_text and table.get("markdown"):
            # Fallback to markdown if row_by_row not available
            table_text = table.get("markdown", "")
        
        if table_text and len(table_text) > 20:
            # Create context-rich table text
            table_context = f"Title: {title}\n\nTable: {caption}\n\n{table_text}"
            
            chunks.append({
                "chunk_id": f"{doc_id}_table{i}",
                "doc_id": doc_id,
                "text": table_context[:8000],  # Tables can be long
                "section_type": "table",
                "section_title": f"Table: {caption[:100]}",
                "is_table": True,
                "table_caption": caption,
                "table_id": table.get("id", f"table-{i}"),
            })
    
    return chunks


def build_points(batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider) -> tuple[List[PointStruct], List[str]]:
    chunker = Chunker(
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS
    )
    sparse_encoder = None
    use_sparse = IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25"
    if use_sparse and BM25SparseEncoder is not None:
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )
    
    points: List[PointStruct] = []
    all_chunk_ids: List[str] = []
    
    # Collect all chunks for batch embedding
    all_chunks: List[Dict[str, Any]] = []
    all_texts: List[str] = []
    
    for article in batch:
        chunks = create_chunks_from_article(article)
        all_chunks.extend(chunks)
        all_texts.extend([c["text"] for c in chunks])
    
    if not all_texts:
        return [], []
    
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
        vector_data: Any = vector
        if sparse_encoder is not None:
            sparse_vector = sparse_encoder.encode_document(chunk["text"])
            vector_data = {"": vector, "sparse": sparse_vector}
        
        # Create payload
        payload = {
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk_id,
            "section_type": chunk["section_type"],
            "section_title": chunk["section_title"],
            "is_table": chunk.get("is_table", False),
            "table_caption": chunk.get("table_caption", ""),
            "source": "pmc",
            "article_type": "research_article",
            "text_preview": chunk["text"][:500],
        }
        
        # Create deterministic point ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pmc:{chunk_id}"))
        
        points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))
    
    return points, all_chunk_ids


def upsert_with_retry(client: QdrantClient, points: List[PointStruct]) -> None:
    for attempt in range(IngestionConfig.MAX_RETRIES):
        try:
            client.upsert(
                collection_name=IngestionConfig.COLLECTION_NAME,
                points=points,
                wait=True,  # Changed to True for reliability
            )
            return
        except Exception as exc:
            if attempt == IngestionConfig.MAX_RETRIES - 1:
                raise
            wait_for = 2**attempt
            logger.warning("Upsert retry %s/%s after error: %s", attempt + 1, IngestionConfig.MAX_RETRIES, str(exc)[:200])
            time.sleep(wait_for)


def iter_articles(xml_dir: Path, articles_file: Optional[Path] = None):
    """Iterate articles from JSONL file or XML directory."""
    if articles_file and articles_file.exists():
        logger.info("Reading from JSONL: %s", articles_file)
        with open(articles_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        # Fallback to XML processing
        from scripts.ingestion_utils import parse_pmc_xml
        logger.info("Reading from XML directory: %s", xml_dir)
        for xml_path in sorted(xml_dir.rglob("*.xml*")):
            article = parse_pmc_xml(xml_path)
            if article:
                yield article


def run_ingestion(xml_dir: Path, articles_file: Optional[Path], embedding_provider: EmbeddingProvider) -> None:
    ensure_data_dirs()
    
    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    info = client.get_collection(IngestionConfig.COLLECTION_NAME)
    logger.info("Connected to %s (points=%s)", IngestionConfig.COLLECTION_NAME, info.points_count)

    processed_ids = load_checkpoint()
    logger.info("Already ingested from checkpoint: %s", len(processed_ids))

    batch: List[Dict[str, Any]] = []
    inserted = 0
    skipped = 0
    start = time.time()

    article_stream = iter_articles(xml_dir, articles_file)

    for article in article_stream:
        source_id = str(article.get("pmcid") or article.get("pmid") or "").strip()
        if not source_id or source_id in processed_ids:
            skipped += 1
            continue

        batch.append(article)
        if len(batch) < IngestionConfig.BATCH_SIZE:
            continue

        points, ids = build_points(batch, embedding_provider)
        if points:
            upsert_with_retry(client, points)
            append_checkpoint(ids)
            processed_ids.update(ids)
            inserted += len(points)
        batch.clear()

        if inserted and inserted % (IngestionConfig.BATCH_SIZE * 10) == 0:
            elapsed = time.time() - start
            rate = inserted / elapsed if elapsed > 0 else 0
            logger.info("Inserted=%s skipped=%s rate=%.2f docs/sec", inserted, skipped, rate)

    if batch:
        points, ids = build_points(batch, embedding_provider)
        if points:
            upsert_with_retry(client, points)
            append_checkpoint(ids)
            inserted += len(points)

    elapsed = time.time() - start
    logger.info("PMC ingestion complete inserted=%s skipped=%s elapsed=%.1fs", inserted, skipped, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PMC into self-hosted Qdrant with improved section/table handling")
    parser.add_argument("--articles-file", type=Path, default=None, help="Path to pre-extracted JSONL")
    parser.add_argument("--xml-dir", type=Path, default=IngestionConfig.PMC_XML_DIR, help="Directory with .xml/.xml.gz")
    parser.add_argument("--delete-source", action="store_true", help="Delete XML file after successful ingestion")
    args = parser.parse_args()

    provider = EmbeddingProvider()
    run_ingestion(args.xml_dir, args.articles_file, provider)
