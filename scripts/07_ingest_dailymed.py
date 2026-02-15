#!/usr/bin/env python3
"""Ingest DailyMed XML files into self-hosted Qdrant with section-aware chunking and table extraction."""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import lxml.etree as ET
from qdrant_client import QdrantClient
from qdrant_client.models import Document, PointStruct

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import upsert_with_retry
from src.bm25_sparse import BM25SparseEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "dailymed_ingested_improved_ids.txt"
NS = {"hl7": "urn:hl7-org:v3"}

# Standard sections we want to identify specifically
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
    return " ".join(element.xpath(".//text()")).strip()


def parse_table_to_markdown(table_elem: ET.Element) -> str:
    """Parse an HTML table to Markdown format, handling colspans and rowspans."""
    if table_elem is None:
        return ""

    rows = []
    # Find all rows (thead and tbody)
    # Note: SPL tables often use thead/tbody, but sometimes just trs directly
    tr_elements = table_elem.xpath(".//hl7:tr", namespaces=NS)
    
    if not tr_elements:
        return ""

    # Calculate max columns to normalize table width
    max_cols = 0
    parsed_rows = []

    for tr in tr_elements:
        row_cells = []
        cells = tr.xpath("hl7:th|hl7:td", namespaces=NS)
        for cell in cells:
            cell_text = get_text(cell).replace("\n", " ").strip()
            # Handle colspan (default to 1)
            colspan = int(cell.get("colspan", "1"))
            # We don't perfectly handle rowspan in markdown, but we can fill empty cells or just repeat
            # For simplicity in RAG, repeating the value or leaving blank is often okay.
            # Here we just expand colspan.
            row_cells.append(cell_text)
            for _ in range(colspan - 1):
                row_cells.append("") # Empty cell for skipped col
        
        parsed_rows.append(row_cells)
        max_cols = max(max_cols, len(row_cells))

    # Build Markdown
    md_lines = []
    
    # Header
    if parsed_rows:
        header_row = parsed_rows[0]
        # Ensure header has max_cols
        header_row.extend([""] * (max_cols - len(header_row)))
        md_lines.append("| " + " | ".join(header_row) + " |")
        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        
        # Body
        for row in parsed_rows[1:]:
            row.extend([""] * (max_cols - len(row)))
            md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def parse_spl_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse DailyMed SPL XML file with full section and table extraction using lxml."""
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        set_id_elem = root.xpath(".//hl7:setId", namespaces=NS)
        set_id = set_id_elem[0].get("root", "") if set_id_elem else ""
        if not set_id:
            set_id = xml_path.stem

        title_elem = root.xpath(".//hl7:title", namespaces=NS)
        title = get_text(title_elem[0]) if title_elem else ""

        # Drug name
        drug_name = title
        name_elem = root.xpath(".//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:name", namespaces=NS)
        if name_elem:
            drug_name = get_text(name_elem[0]).strip()

        # Active ingredients
        active_ingredients: List[str] = []
        ingredients = root.xpath('.//hl7:ingredient[@classCode="ACTIB"]', namespaces=NS)
        for ingredient in ingredients:
            ing_name = ingredient.xpath('.//hl7:ingredientSubstance/hl7:name', namespaces=NS)
            if ing_name:
                active_ingredients.append(get_text(ing_name[0]))

        # Manufacturer
        manufacturer = ""
        org_name = root.xpath(".//hl7:representedOrganization/hl7:name", namespaces=NS)
        if org_name:
            manufacturer = get_text(org_name[0])

        # Parse ALL sections
        sections: Dict[str, Dict[str, Any]] = {}
        
        # Find all sections that have codes or titles
        section_elements = root.xpath(".//hl7:section", namespaces=NS)
        
        for section in section_elements:
            code_elem = section.xpath("hl7:code", namespaces=NS)
            title_elem = section.xpath("hl7:title", namespaces=NS)
            text_elem = section.xpath("hl7:text", namespaces=NS)
            
            # Determine section key and title
            section_key = ""
            section_title = ""
            
            if code_elem:
                code = code_elem[0].get("code", "")
                if code in SECTION_CODES:
                    section_key, section_title = SECTION_CODES[code]
                else:
                    # Non-standard section, use code as fallback key or generate one
                    section_key = code or "unknown_section"
            
            if not section_title and title_elem:
                section_title = get_text(title_elem[0])
            
            if not section_key and section_title:
                # Slugify title for key
                section_key = section_title.lower().replace(" ", "_").replace("/", "_")[:30]

            if not section_key:
                section_key = f"section_{uuid.uuid4().hex[:8]}"

            # Get section text
            section_text = get_text(text_elem[0]) if text_elem else ""
            
            # Parse tables within this section
            tables = []
            if text_elem:
                table_elements = text_elem[0].xpath(".//hl7:table", namespaces=NS)
                for i, table in enumerate(table_elements):
                    table_content = parse_table_to_markdown(table)
                    if table_content:
                        tables.append({
                            "index": i,
                            "content": table_content,
                            "type": f"table_{i}"
                        })
            
            # Only add if interesting
            if section_text or tables:
                # If we already have this section (e.g. repeated codes), append
                if section_key in sections:
                    existing = sections[section_key]
                    existing["text"] += "\n\n" + section_text
                    existing["tables"].extend(tables)
                    existing["has_tables"] = existing["has_tables"] or (len(tables) > 0)
                else:
                    sections[section_key] = {
                        "title": section_title,
                        "text": section_text[:20000],  # Increased limit
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
        
        # Skip very short sections without tables
        if len(section_text) < 50 and not section_data.get("has_tables"):
            continue
        
        # Create chunk with drug context
        # Use markdown header for section title
        chunk_text = f"# {drug_name} - {section_title}\n\n{section_text}"
        
        chunks.append({
            "chunk_id": f"{set_id}_{section_key}",
            "set_id": set_id,
            "drug_name": drug_name,
            "section_type": section_key,
            "section_title": section_title,
            "text": chunk_text[:3000], # Increased limit
            "has_tables": section_data.get("has_tables", False),
            "manufacturer": drug.get("manufacturer", ""),
            "active_ingredients": drug.get("active_ingredients", []),
        })
        
        # 3. Table chunks (if present)
        for table in section_data.get("tables", []):
            table_text = f"# {drug_name} - {section_title} Table\n\n{table['content']}"
            chunks.append({
                "chunk_id": f"{set_id}_{section_key}_table_{table['index']}",
                "set_id": set_id,
                "drug_name": drug_name,
                "section_type": "table",
                "section_title": f"{section_title} Table",
                "table_type": section_key,
                "text": table_text[:2500],
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


def build_points(chunks: List[Dict[str, Any]], embedding_provider: EmbeddingProvider, sparse_encoder: Optional[BM25SparseEncoder]) -> Tuple[List[PointStruct], List[str]]:
    """Build Qdrant points from chunks."""
    chunk_ids: List[str] = []
    texts: List[str] = []
    sparse_vectors = []
    payloads: List[Dict[str, Any]] = []
    
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





def process_file_batch(
    xml_files: List[Path], 
    seen: set,
    embedding_provider: EmbeddingProvider,
    sparse_encoder: Optional[BM25SparseEncoder],
    client: QdrantClient
) -> Tuple[int, int]:
    """Process a batch of files (to be run in loop)."""
    inserted = 0
    skipped = 0
    
    for xml_file in xml_files:
        drug = parse_spl_xml(xml_file)
        if not drug:
            continue

        chunks = create_chunks(drug)
        new_chunks = [c for c in chunks if c.get("chunk_id") not in seen]
        
        if not new_chunks:
            skipped += 1
            continue

        # Build and upsert points
        try:
            points, chunk_ids = build_points(new_chunks, embedding_provider, sparse_encoder)
            if points:
                upsert_with_retry(client, points)
                append_checkpoint(chunk_ids)
                # We don't update 'seen' in real-time across processes easily, but this is a batch
                inserted += len(points)
        except Exception as e:
            logger.error("Error processing %s: %s", xml_file, e)

    return inserted, skipped


def run_ingestion(xml_dir: Path, embedding_provider: EmbeddingProvider) -> None:
    ensure_data_dirs()
    # Use rglob to find all XMLs
    xml_files = sorted(xml_dir.rglob("*.xml"))
    
    if not xml_files:
        logger.warning("No XML files found in %s", xml_dir)
        return

    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        cloud_inference=embedding_provider.use_cloud(),
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    try:
        info = client.get_collection(IngestionConfig.COLLECTION_NAME)
        logger.info("Connected to %s points=%s", IngestionConfig.COLLECTION_NAME, info.points_count)
    except Exception as e:
         logger.error("Failed to connect to collection: %s", e)
         return

    seen = load_checkpoint()
    logger.info("DailyMed checkpoint size=%s", len(seen))

    sparse_encoder = None
    if IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )

    # Simple sequential for now to avoid pickling issues with QdrantClient/EmbeddingProvider in ProcessPool
    # unless we architect it differently (e.g. init client in worker).
    # Correct parallel approach: distribute file paths, init resources in worker.
    # For now, let's Stick to sequential to ensure Stability with shared resources OR use threads.
    # Given EmbeddingProvider might be not thread safe or picklable (local encoder), sequential/threading.
    # Qdrant client is thread safe.
    # But Python GIL limits threads for parsing.
    # Let's use simple sequential loop as in original but optimized, unless dataset is huge.
    # If huge, we process in chunks.
    
    inserted = 0
    skipped = 0
    started = time.time()

    total_files = len(xml_files)
    logger.info("Starting ingestion of %d files...", total_files)

    for i, xml_file in enumerate(xml_files):
        try:
            drug = parse_spl_xml(xml_file)
            if not drug:
                continue

            chunks = create_chunks(drug)
            new_chunks = [c for c in chunks if c.get("chunk_id") not in seen]
            
            if not new_chunks:
                skipped += 1
                continue

            points, chunk_ids = build_points(new_chunks, embedding_provider, sparse_encoder)
            if points:
                upsert_with_retry(client, points)
                append_checkpoint(chunk_ids)
                seen.update(chunk_ids)
                inserted += len(points)
                
            if (i + 1) % 10 == 0:
                 elapsed = time.time() - started
                 rate = (i + 1) / elapsed
                 logger.info("Processed %d/%d files. Inserted points: %d. Rate: %.2f files/s", i + 1, total_files, inserted, rate)

        except Exception as e:
            logger.error("Failed to process file %s: %s", xml_file, e)

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
        # Just warn, don't exit, might be mounting issue
        logger.warning(f"Directory not found: {args.xml_dir}")

    try:
        provider = EmbeddingProvider()
        run_ingestion(args.xml_dir, provider)
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        sys.exit(1)

