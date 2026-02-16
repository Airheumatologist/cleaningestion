#!/usr/bin/env python3
"""Ingest DailyMed XML files into self-hosted Qdrant with section-aware chunking and table extraction."""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Initialize logger FIRST before any imports that might fail
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import lxml.etree as ET
    from qdrant_client import QdrantClient
    from qdrant_client.models import Document, PointStruct

    sys.path.insert(0, str(Path(__file__).parent))         # Add scripts/ to path
    sys.path.insert(0, str(Path(__file__).parent.parent))  # Add root to path for src
    from config_ingestion import IngestionConfig, ensure_data_dirs
    from ingestion_utils import EmbeddingProvider, upsert_with_retry
    
    # Import BM25SparseEncoder
    import importlib.util
    spec = importlib.util.find_spec("src.bm25_sparse")
    if spec is not None:
        from src.bm25_sparse import BM25SparseEncoder
    else:
        BM25SparseEncoder = None  # type: ignore
except Exception as import_err:
    logger.error("Failed to import required modules: %s", import_err)
    logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# XML Namespaces for SPL (Structured Product Labeling)
NS = {"hl7": "urn:hl7-org:v3"}

# LOINC Codes for Common Drug Label Sections
SECTION_CODES = {
    "34067-9": ("indications", "Indications & Usage"),
    "34068-7": ("dosage", "Dosage & Administration"), 
    "43683-2": ("recent_major_changes", "Recent Major Changes"),
    "34070-3": ("contraindications", "Contraindications"),
    "34071-1": ("warnings", "Warnings & Precautions"), 
    "34072-9": ("adverse_reactions", "Adverse Reactions"),
    "34073-7": ("interactions", "Drug Interactions"),
    "34066-1": ("boxed_warning", "Boxed Warning"),
    "34074-5": ("use_in_specific_populations", "Use in Specific Populations"),
    "34076-0": ("overdosage", "Overdosage"),
    "34069-5": ("description", "Description"),
    "34089-3": ("clinical_pharmacology", "Clinical Pharmacology"),
    "34090-1": ("nonclinical_toxicology", "Nonclinical Toxicology"),
    "34091-9": ("clinical_studies", "Clinical Studies"),
    "34092-7": ("references", "References"),
    "34093-5": ("supply", "How Supplied/Storage"),
    "34075-2": ("patient_counseling", "Patient Counseling Information"),
    "42232-9": ("precautions", "Precautions"),
    "42229-5": ("spl_unclassified", "Unclassified Section"),
    "51945-4": ("package_insert", "Package Insert"),
}

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




def build_points(chunks: List[Dict[str, Any]], embedding_provider: EmbeddingProvider, sparse_encoder: Optional[BM25SparseEncoder]) -> Tuple[List[PointStruct], List[str]]:
    """Convert chunks into Qdrant PointStructs with embeddings."""
    if not chunks:
        return [], []
        
    texts = [chunk["text"] for chunk in chunks]
    
    # 1. Generate Dense Embeddings
    try:
        embeddings = embedding_provider.embed_batch(texts)
    except Exception as e:
        logger.error("Failed to generate embeddings: %s", e)
        return [], []
        
    points = []
    chunk_ids = []
    
    # 2. Generate Sparse Vectors (if enabled)
    sparse_vectors = []
    if sparse_encoder:
        try:
            sparse_vectors = sparse_encoder.encode_batch(texts)
        except Exception as e:
            logger.warning("Sparse parsing failed batch: %s", e)
            # Fallback to empty if failed, or skip?
            # Let's fill with empty to keep alignment
            sparse_vectors = [None] * len(texts)
    
    # 3. Build Points
    for i, chunk in enumerate(chunks):
        try:
            # Generate UUID from chunk_id for consistent ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
            
            vector = {"dense": embeddings[i]}
            
            if sparse_encoder and i < len(sparse_vectors) and sparse_vectors[i]:
                vector["sparse"] = sparse_vectors[i].model_dump()
            
            # Add timestamp
            chunk["ingestion_timestamp"] = time.time()
            
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk
            ))
            chunk_ids.append(chunk["chunk_id"])
            
        except Exception as e:
            logger.error("Failed to build point for chunk %s: %s", chunk.get("chunk_id"), e)
            
    return points, chunk_ids


# Checkpoint file for DailyMed ingestion
CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "dailymed_ingested_ids.txt"

# Checkpoint file for DailyMed ingestion

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


def process_batch(client: QdrantClient, batch_files: List[Path], embedding_provider: EmbeddingProvider, sparse_encoder: Optional[BM25SparseEncoder], processed_ids: set[str], processed_lock: threading.Lock) -> Tuple[int, int]:
    """Process a batch of DailyMed XML files in a single thread."""
    inserted = 0
    skipped = 0
    
    drugs = []
    new_ids = []
    
    # 1. Parse and filter
    for xml_file in batch_files:
        try:
            # Fast skip: check filename (stem) against checkpoint
            # parse_spl_xml uses stem if set_id is missing, so this is a safe heuristic
            stem_id = xml_file.stem
            if stem_id in processed_ids:
                skipped += 1
                continue

            drug = parse_spl_xml(xml_file)
            if not drug:
                continue
            
            set_id = drug.get("set_id")
            if not set_id:
                continue
                
            # Double check exact ID
            if set_id != stem_id and set_id in processed_ids:
                skipped += 1
                continue
            
            drugs.append(drug)
            new_ids.append(set_id)
            
        except Exception as e:
            logger.warning("Failed to parse %s: %s", xml_file, e)

    if not drugs:
        return 0, len(batch_files)

    # 2. Chunk and Embedding
    all_chunks = []
    for drug in drugs:
        all_chunks.extend(create_chunks(drug))
        
    if not all_chunks:
        return 0, len(batch_files)

    try:
        points, chunk_ids = build_points(all_chunks, embedding_provider, sparse_encoder)
        
        if points:
            upsert_with_retry(client, points)
            
            # Update checkpoints
            append_checkpoint(new_ids)
            with processed_lock:
                processed_ids.update(new_ids)
                
            inserted = len(points)
    except Exception as e:
        logger.error("Batch upsert failed: %s", e)
        
    return inserted, len(batch_files) - len(drugs) # skipped count approximation for stats


def run_ingestion(xml_dir: Path, embedding_provider: EmbeddingProvider) -> None:
    ensure_data_dirs()
    
    # Preload tokenizer
    logger.info("Preloading tokenizer...")
    try:
        from ingestion_utils import Chunker
        Chunker()
        logger.info("Tokenizer preloaded.")
    except Exception as e:
        logger.warning("Tokenizer preload failed: %s", e)

    # Use rglob to find all XMLs
    xml_files = sorted(xml_dir.rglob("*.xml"))
    
    if not xml_files:
        logger.warning("No XML files found in %s", xml_dir)
        return

    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    try:
        info = client.get_collection(IngestionConfig.COLLECTION_NAME)
        logger.info("Connected to %s points=%s", IngestionConfig.COLLECTION_NAME, info.points_count)
    except Exception as e:
         logger.error("Failed to connect to collection: %s", e)
         return

    processed_ids = load_checkpoint()
    processed_lock = threading.Lock()
    logger.info("DailyMed checkpoint size=%s", len(processed_ids))

    sparse_encoder = None
    if IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )

    logger.info("Starting ingestion of %d files...", len(xml_files))
    
    # Batching config
    THREAD_BATCH_SIZE = IngestionConfig.BATCH_SIZE
    MAX_WORKERS = IngestionConfig.MAX_WORKERS # Should be 64 now
    
    file_batches = [xml_files[i:i + THREAD_BATCH_SIZE] for i in range(0, len(xml_files), THREAD_BATCH_SIZE)]
    total_batches = len(file_batches)
    
    total_inserted = 0
    total_skipped = 0
    completed_batches = 0
    start_time = time.time()
    
    SUPER_BATCH_SIZE = 1000

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(0, total_batches, SUPER_BATCH_SIZE):
            current_super_batch = file_batches[i : i + SUPER_BATCH_SIZE]
            
            future_to_batch = {
                executor.submit(process_batch, client, batch, embedding_provider, sparse_encoder, processed_ids, processed_lock): batch 
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
                     logger.info("Progress: %.1f%% | Inserted: %s | Skipped: %s | Rate: %.2f pts/sec", 
                                 progress, total_inserted, total_skipped, rate)
            
            future_to_batch.clear()

    logger.info(
        "DailyMed ingestion complete inserted=%s skipped=%s elapsed=%.1fs",
        total_inserted,
        total_skipped,
        time.time() - start_time,
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

