#!/usr/bin/env python3
"""Ingest PMC content into self-hosted Qdrant with checkpoint support."""

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pmc_ingested_ids.txt"


class EmbeddingProvider:
    def __init__(self) -> None:
        self.provider = IngestionConfig.EMBEDDING_PROVIDER.lower().strip()
        self.model = IngestionConfig.EMBEDDING_MODEL
        self.local_encoder = None
        self.cohere_client = None

        if self.provider == "cohere":
            import cohere
            api_key = IngestionConfig.COHERE_API_KEY
            if not api_key:
                raise ValueError("COHERE_API_KEY not set — required for cohere embedding provider")
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
        if self.local_encoder is None:
            raise RuntimeError("Local encoder not initialized")
        vectors = self.local_encoder.encode(texts, normalize_embeddings=True, batch_size=IngestionConfig.EMBEDDING_BATCH_SIZE)
        return [v.tolist() for v in vectors]

    def _embed_cohere(self, texts: List[str]) -> List[List[float]]:
        """Embed texts via Cohere API with batching (max 96 per call)."""
        all_vectors: List[List[float]] = []
        batch_size = min(IngestionConfig.EMBEDDING_BATCH_SIZE, 96)  # Cohere API limit

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            response = self.cohere_client.embed(
                texts=chunk,
                model=self.model,
                input_type="search_document",
                embedding_types=["float"],
            )
            all_vectors.extend([list(v) for v in response.embeddings.float_])

        return all_vectors



def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return {line.strip() for line in CHECKPOINT_FILE.read_text().splitlines() if line.strip()}
    return set()



def append_checkpoint(ids: Iterable[str]) -> None:
    if not ids:
        return
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        for item in ids:
            f.write(f"{item}\n")



def create_embedding_text(article: Dict[str, Any], max_chars: int = 2000) -> str:
    title = article.get("title", "") or ""
    abstract = article.get("abstract", "") or ""
    full_text = article.get("full_text", "") or ""
    combined = f"{title}. {abstract}".strip()
    if full_text and len(combined) < max_chars - 128:
        remaining = max_chars - len(combined) - 2
        combined = f"{combined}\n{full_text[:remaining]}"
    return combined[:max_chars]



def create_payload(article: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pmcid": article.get("pmcid"),
        "pmid": article.get("pmid"),
        "doi": article.get("doi"),
        "title": (article.get("title") or "")[:300],
        "abstract": (article.get("abstract") or "")[:2000],
        "full_text": (article.get("full_text") or "")[:12000],
        "year": article.get("year"),
        "journal": article.get("journal", ""),
        "article_type": article.get("article_type", "research_article"),
        "publication_type": (article.get("publication_type_list") or [])[:5],
        "evidence_grade": article.get("evidence_grade"),
        "evidence_level": article.get("evidence_level"),
        "country": article.get("country"),
        "institutions": (article.get("institutions") or [])[:5],
        "keywords": (article.get("keywords") or [])[:15],
        "mesh_terms": (article.get("mesh_terms") or [])[:20],
        "authors": (article.get("authors") or [])[:8],
        "first_author": article.get("first_author"),
        "author_count": article.get("author_count", 0),
        "source": article.get("source", "pmc"),
        "has_full_text": bool(article.get("full_text")),
        "has_methods": bool(article.get("has_methods")),
        "has_results": bool(article.get("has_results")),
        "table_count": article.get("table_count", 0),
        "figure_count": article.get("figure_count", 0),
    }



def upsert_with_retry(client: QdrantClient, points: List[PointStruct]) -> None:
    for attempt in range(IngestionConfig.MAX_RETRIES):
        try:
            client.upsert(
                collection_name=IngestionConfig.COLLECTION_NAME,
                points=points,
                wait=False,
            )
            return
        except Exception as exc:
            if attempt == IngestionConfig.MAX_RETRIES - 1:
                raise
            wait_for = 2**attempt
            logger.warning("Upsert retry %s/%s after error: %s", attempt + 1, IngestionConfig.MAX_RETRIES, str(exc)[:200])
            time.sleep(wait_for)



from ingestion_utils import Chunker, SectionFilter
from src.bm25_sparse import BM25SparseEncoder

def build_points(batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider) -> tuple[List[PointStruct], List[str]]:
    chunker = Chunker(
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS
    )
    sparse_encoder = None
    use_sparse = IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25"
    if use_sparse:
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )
    
    points: List[PointStruct] = []
    source_ids: List[str] = [] # Doc IDs processed
    
    # Batch collection for embeddings
    all_chunks_text: List[str] = []
    all_chunks_sparse = []
    chunk_metadata: List[Dict[str, Any]] = []

    for article in batch:
        doc_id = str(article.get("pmcid") or article.get("pmid") or "").strip()
        if not doc_id:
            continue
            
        source_ids.append(doc_id)
        
        # Base payload (article metadata)
        base_payload = create_payload(article)
        
        # Iterating over sections
        sections = article.get("sections", [])
        if not sections:
            # Fallback for old extraction or empty body
            full_text = article.get("full_text", "")
            if full_text:
                sections = [{"title": "Body", "text": full_text, "type": "body"}]
            else:
                # Abstract fallback
                abstract = article.get("abstract", "")
                if abstract:
                    sections = [{"title": "Abstract", "text": abstract, "type": "abstract"}]
        
        chunk_index = 0
        for section in sections:
            # Filter Backmatter
            if IngestionConfig.EMBED_FILTER_ENABLED and SectionFilter.should_exclude(section):
                continue
                
            # Create Chunks
            text_chunks = chunker.chunk_text(section.get("text", ""))
            
            for chunk in text_chunks:
                text = chunk["text"]
                if len(text) < 20:
                    continue
                    
                # Store text for embedding
                all_chunks_text.append(text)
                if sparse_encoder is not None:
                    all_chunks_sparse.append(sparse_encoder.encode_document(text))
                else:
                    all_chunks_sparse.append(None)
                
                # Create Payload
                payload = base_payload.copy()
                payload.update({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{chunk_index}",
                    "chunk_index": chunk_index,
                    "chunk_token_count": chunk["token_count"],
                    "section_title": section.get("title", ""),
                    "section_type": section.get("type", "body"),
                    "page_content": text, # Store chunk text
                    "is_backmatter_excluded": False,
                    "full_text": "" # Clear full_text to save space (rely on chunk text)
                })
                chunk_metadata.append(payload)
                chunk_index += 1

    if not all_chunks_text:
        return [], source_ids

    # Embed all chunks
    if embedding_provider.use_cloud():
        for i, text in enumerate(all_chunks_text):
            payload = chunk_metadata[i]
            # ID is deterministic based on chunk_id (pmcid_index)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk:{payload['chunk_id']}"))
            sparse_vector = all_chunks_sparse[i]
            vector_data: Any = Document(text=text, model=embedding_provider.model)
            if sparse_vector is not None:
                vector_data = {"": vector_data, "sparse": sparse_vector}
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector_data,
                    payload=payload,
                )
            )
    else:
        vectors = embedding_provider.embed_batch(all_chunks_text)
        for i, vector in enumerate(vectors):
            payload = chunk_metadata[i]
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk:{payload['chunk_id']}"))
            sparse_vector = all_chunks_sparse[i]
            vector_data: Any = vector
            if sparse_vector is not None:
                vector_data = {"": vector, "sparse": sparse_vector}
            
            points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))

    return points, source_ids



def _load_parse_xml_file():
    script_path = Path(__file__).with_name("02_extract_pmc.py")
    spec = importlib.util.spec_from_file_location("extract_pmc_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load parser from 02_extract_pmc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_xml_file



def iter_articles_from_jsonl(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue



def iter_articles_from_xml(xml_dir: Path, delete_source: bool = False) -> Iterable[Dict[str, Any]]:
    parse_xml_file = _load_parse_xml_file()
    xml_files = sorted(list(xml_dir.rglob("*.xml")) + list(xml_dir.rglob("*.xml.gz")))

    for path in xml_files:
        article = parse_xml_file(path)
        if article:
            yield article
            if delete_source:
                try:
                    path.unlink(missing_ok=True)
                except Exception as exc:
                    logger.debug("Delete failed for %s: %s", path, exc)



def run_ingestion(
    articles_file: Optional[Path],
    xml_dir: Optional[Path],
    delete_source: bool,
    embedding_provider: EmbeddingProvider,
) -> None:
    ensure_data_dirs()
    IngestionConfig.DATA_DIR.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        cloud_inference=embedding_provider.use_cloud(),
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    info = client.get_collection(IngestionConfig.COLLECTION_NAME)
    logger.info("Connected to %s (points=%s)", IngestionConfig.COLLECTION_NAME, info.points_count)

    processed_ids = load_checkpoint()
    logger.info("Already ingested from checkpoint: %s", len(processed_ids))

    if articles_file:
        article_stream = iter_articles_from_jsonl(articles_file)
    else:
        article_stream = iter_articles_from_xml(xml_dir or IngestionConfig.PMC_XML_DIR, delete_source=delete_source)

    batch: List[Dict[str, Any]] = []
    inserted = 0
    skipped = 0
    start = time.time()

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
    parser = argparse.ArgumentParser(description="Ingest PMC into self-hosted Qdrant")
    parser.add_argument("--articles-file", type=Path, default=None, help="Path to pre-extracted JSONL")
    parser.add_argument("--xml-dir", type=Path, default=IngestionConfig.PMC_XML_DIR, help="Directory with .xml/.xml.gz")
    parser.add_argument("--delete-source", action="store_true", help="Delete XML file after successful ingestion")
    args = parser.parse_args()

    if args.articles_file is None and not args.xml_dir.exists():
        raise SystemExit(f"XML directory not found: {args.xml_dir}")

    embedding_provider = EmbeddingProvider()
    run_ingestion(args.articles_file, args.xml_dir, args.delete_source, embedding_provider=embedding_provider)
