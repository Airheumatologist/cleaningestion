#!/usr/bin/env python3
"""Ingest DailyMed XML files into self-hosted Qdrant with section-aware chunking and table extraction."""

from __future__ import annotations

import argparse
import logging
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Document, PointStruct

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config_ingestion import IngestionConfig, ensure_data_dirs
from src.bm25_sparse import BM25SparseEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "dailymed_ingested_improved_ids.txt"
NS = {"hl7": "urn:hl7-org:v3"}

SECTION_CODES = {
    "34067-9": ("indications", "Indications and Usage"),
    "34070-3": ("contraindications", "Contraindications"),
    "34068-7": ("dosage", "Dosage and Administration"),
    "34084-4": ("adverse_reactions", "Adverse Reactions"),
    "34073-7": ("interactions", "Drug Interactions"),
    "34071-1": ("warnings", "Warnings and Precautions"),
    "43685-7": ("warnings_precautions", "Warnings and Precautions"),
    "34089-3": ("description", "Description"),
    "34066-1": ("mechanism", "Mechanism of Action"),
    "34090-1": ("pharmacokinetics", "Pharmacokinetics"),
    "34091-9": ("clinical_studies", "Clinical Studies"),
    "42229-5": ("patient_info", "Patient Information"),
    "34072-9": ("storage", "Storage and Handling"),
    "51727-6": ("supply", "Supply Information"),
}


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


def get_text(element: Optional[ET.Element]) -> str:
    """Extract all text from element."""
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()


def parse_table(table_elem: ET.Element) -> str:
    """Parse an HTML table to a readable text format."""
    if table_elem is None:
        return ""
    
    rows = []
    for tr in table_elem.findall(".//hl7:tr", NS):
        row = []
        for cell in tr.findall("hl7:th|hl7:td", NS):
            cell_text = " ".join(cell.itertext()).strip()
            row.append(cell_text)
        if row:
            rows.append(" | ".join(row))
    
    return "\n".join(rows)


def parse_spl_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse DailyMed SPL XML file with full section and table extraction."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        set_id_elem = root.find(".//hl7:setId", NS)
        set_id = set_id_elem.get("root", "") if set_id_elem is not None else ""
        if not set_id:
            set_id = xml_path.stem

        title = get_text(root.find(".//hl7:title", NS))

        # Drug name
        drug_name = title
        name_elem = root.find(".//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:name", NS)
        if name_elem is not None and name_elem.text:
            drug_name = name_elem.text.strip()

        # Active ingredients
        active_ingredients: List[str] = []
        for ingredient in root.findall('.//hl7:ingredient[@classCode="ACTIB"]', NS):
            ing_name = ingredient.find('.//hl7:ingredientSubstance/hl7:name', NS)
            if ing_name is not None and ing_name.text:
                active_ingredients.append(ing_name.text.strip())

        # Manufacturer
        manufacturer = ""
        org_name = root.find(".//hl7:representedOrganization/hl7:name", NS)
        if org_name is not None and org_name.text:
            manufacturer = org_name.text.strip()

        # Parse sections with tables
        sections: Dict[str, Dict[str, Any]] = {}
        for section in root.findall(".//hl7:section", NS):
            code_elem = section.find("hl7:code", NS)
            if code_elem is None:
                continue
            
            code = code_elem.get("code", "")
            if code not in SECTION_CODES:
                continue
            
            section_key, section_title = SECTION_CODES[code]
            text_elem = section.find("hl7:text", NS)
            
            if text_elem is None:
                continue
            
            # Get section text
            section_text = get_text(text_elem)
            
            # Parse tables within this section
            tables = []
            for i, table in enumerate(text_elem.findall(".//hl7:table", NS)):
                table_content = parse_table(table)
                if table_content:
                    tables.append({
                        "index": i,
                        "content": table_content,
                        "type": f"table_{i}"
                    })
            
            sections[section_key] = {
                "title": section_title,
                "text": section_text[:10000],  # Store more text
                "tables": tables,
                "has_tables": len(tables) > 0
            }

        return {
            "set_id": set_id,
            "drug_name": drug_name,
            "title": title,
            "active_ingredients": active_ingredients,
            "manufacturer": manufacturer,
            "sections": sections,
            "source": "dailymed",
            "article_type": "drug_label",
        }
    except Exception as e:
        logger.warning("Failed to parse %s: %s", xml_path, e)
        return None


def create_chunks(drug: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create multiple chunks per drug - one per section + one per table."""
    chunks = []
    set_id = drug.get("set_id", "")
    drug_name = drug.get("drug_name", "")
    ingredients = ", ".join(drug.get("active_ingredients", []))
    
    # 1. Drug overview chunk (summary)
    overview_text = f"Drug: {drug_name}. "
    if ingredients:
        overview_text += f"Active ingredients: {ingredients}. "
    
    # Add key sections to overview
    sections = drug.get("sections", {})
    if "indications" in sections:
        overview_text += f"Indications: {sections['indications']['text'][:500]}. "
    if "dosage" in sections:
        overview_text += f"Dosage: {sections['dosage']['text'][:300]}. "
    
    chunks.append({
        "chunk_id": f"{set_id}_overview",
        "set_id": set_id,
        "drug_name": drug_name,
        "section_type": "overview",
        "section_title": "Drug Overview",
        "text": overview_text[:1500],
        "manufacturer": drug.get("manufacturer", ""),
        "active_ingredients": drug.get("active_ingredients", []),
    })
    
    # 2. Individual section chunks (for detailed search)
    for section_key, section_data in sections.items():
        section_text = section_data["text"]
        section_title = section_data["title"]
        
        # Skip very short sections
        if len(section_text) < 50:
            continue
        
        # Create chunk with drug context
        chunk_text = f"{drug_name} - {section_title}: {section_text}"
        
        chunks.append({
            "chunk_id": f"{set_id}_{section_key}",
            "set_id": set_id,
            "drug_name": drug_name,
            "section_type": section_key,
            "section_title": section_title,
            "text": chunk_text[:2000],
            "has_tables": section_data.get("has_tables", False),
            "manufacturer": drug.get("manufacturer", ""),
            "active_ingredients": drug.get("active_ingredients", []),
        })
        
        # 3. Table chunks (if present)
        for table in section_data.get("tables", []):
            table_text = f"{drug_name} - {section_title} Table: {table['content']}"
            chunks.append({
                "chunk_id": f"{set_id}_{section_key}_table_{table['index']}",
                "set_id": set_id,
                "drug_name": drug_name,
                "section_type": "table",
                "section_title": f"{section_title} Table",
                "table_type": section_key,
                "text": table_text[:2000],
                "manufacturer": drug.get("manufacturer", ""),
                "active_ingredients": drug.get("active_ingredients", []),
            })
    
    return chunks


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return {line.strip() for line in CHECKPOINT_FILE.read_text().splitlines() if line.strip()}
    return set()


def append_checkpoint(ids: Iterable[str]) -> None:
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as f:
        for value in ids:
            f.write(f"{value}\n")


def build_points(chunks: List[Dict[str, Any]], embedding_provider: EmbeddingProvider) -> Tuple[List[PointStruct], List[str]]:
    """Build Qdrant points from chunks."""
    chunk_ids: List[str] = []
    texts: List[str] = []
    sparse_vectors = []
    payloads: List[Dict[str, Any]] = []
    
    sparse_encoder = None
    if IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        text = chunk.get("text", "")
        
        if len(text) < 20:
            continue

        chunk_ids.append(chunk_id)
        texts.append(text)
        sparse_vectors.append(sparse_encoder.encode_document(text) if sparse_encoder is not None else None)
        
        payloads.append({
            "doc_id": chunk.get("set_id", ""),
            "chunk_id": chunk_id,
            "drug_name": chunk.get("drug_name", "")[:200],
            "section_type": chunk.get("section_type", ""),
            "section_title": chunk.get("section_title", "")[:100],
            "table_type": chunk.get("table_type", ""),
            "active_ingredients": chunk.get("active_ingredients", [])[:10],
            "manufacturer": chunk.get("manufacturer", ""),
            "source": "dailymed",
            "article_type": "drug_label",
            "text_preview": text[:500],
        })

    points: List[PointStruct] = []
    if not texts:
        return points, []

    if embedding_provider.use_cloud():
        for chunk_id, text, sparse_vector, payload in zip(chunk_ids, texts, sparse_vectors, payloads):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            vector_data: Any = Document(text=text, model=embedding_provider.model)
            if sparse_vector is not None:
                vector_data = {"": vector_data, "sparse": sparse_vector}
            points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))
    else:
        vectors = embedding_provider.embed_batch(texts)
        for chunk_id, vector, sparse_vector, payload in zip(chunk_ids, vectors, sparse_vectors, payloads):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            vector_data: Any = vector
            if sparse_vector is not None:
                vector_data = {"": vector, "sparse": sparse_vector}
            points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))

    return points, chunk_ids


def upsert_with_retry(client: QdrantClient, points: List[PointStruct]) -> None:
    for attempt in range(IngestionConfig.MAX_RETRIES):
        try:
            client.upsert(
                collection_name=IngestionConfig.COLLECTION_NAME, 
                points=points, 
                wait=True  # Changed to wait=True for reliability
            )
            return
        except Exception as exc:
            if attempt == IngestionConfig.MAX_RETRIES - 1:
                raise
            wait_time = 2 ** attempt
            logger.warning("Upsert retry %d/%d after error: %s", attempt + 1, IngestionConfig.MAX_RETRIES, str(exc)[:200])
            time.sleep(wait_time)


def run_ingestion(xml_dir: Path, embedding_provider: EmbeddingProvider) -> None:
    ensure_data_dirs()
    xml_files = sorted(xml_dir.rglob("*.xml"))

    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        cloud_inference=embedding_provider.use_cloud(),
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    info = client.get_collection(IngestionConfig.COLLECTION_NAME)
    logger.info("Connected to %s points=%s", IngestionConfig.COLLECTION_NAME, info.points_count)

    seen = load_checkpoint()
    logger.info("DailyMed checkpoint size=%s", len(seen))

    inserted = 0
    skipped = 0
    started = time.time()

    for xml_file in xml_files:
        drug = parse_spl_xml(xml_file)
        if not drug:
            continue

        # Create multiple chunks per drug
        chunks = create_chunks(drug)
        
        # Filter out already processed chunks
        new_chunks = [c for c in chunks if c.get("chunk_id") not in seen]
        if not new_chunks:
            skipped += 1
            continue

        # Build and upsert points
        points, chunk_ids = build_points(new_chunks, embedding_provider)
        if points:
            upsert_with_retry(client, points)
            append_checkpoint(chunk_ids)
            seen.update(chunk_ids)
            inserted += len(points)
            
            if inserted % 100 == 0:
                elapsed = time.time() - started
                logger.info("DailyMed inserted=%s skipped=%s elapsed=%.1fs", inserted, skipped, elapsed)

    logger.info(
        "DailyMed ingestion complete inserted=%s skipped=%s elapsed=%.1fs",
        inserted,
        skipped,
        time.time() - started,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DailyMed into self-hosted Qdrant with improved section handling")
    parser.add_argument("--xml-dir", type=Path, default=IngestionConfig.DAILYMED_XML_DIR)
    args = parser.parse_args()

    if not args.xml_dir.exists():
        raise SystemExit(f"Directory not found: {args.xml_dir}")

    provider = EmbeddingProvider()
    run_ingestion(args.xml_dir, provider)
