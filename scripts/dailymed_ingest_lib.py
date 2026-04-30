#!/usr/bin/env python3
"""Shared DailyMed SPL parsing, chunking, and point-building utilities."""

from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Initialize logger FIRST before any imports that might fail
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import lxml.etree as ET
    from qdrant_client.models import PointStruct

    sys.path.insert(0, str(Path(__file__).parent))         # Add scripts/ to path
    sys.path.insert(0, str(Path(__file__).parent.parent))  # Add root to path for src
    from dailymed_rx_filters import extract_document_label_type, is_human_prescription_label
    from config_ingestion import IngestionConfig
    from ingestion_utils import (
        EmbeddingProvider,
        load_checkpoint as load_checkpoint_file,
    )
    from ingestion_utils import Chunker as BaseChunker
    
    # Import enhanced utilities for semantic chunking and validation
    try:
        from ingestion_utils_enhanced import SemanticChunker, QualityValidator, ContentDeduplicator
        ENHANCED_UTILS_AVAILABLE = True
        logger.info("Using SemanticChunker for improved chunking")
    except ImportError:
        SemanticChunker = None  # type: ignore
        QualityValidator = None  # type: ignore
        ContentDeduplicator = None  # type: ignore
        ENHANCED_UTILS_AVAILABLE = False
        logger.warning("Enhanced utils not available, using base Chunker")
except Exception as import_err:
    logger.error("Failed to import required modules: %s", import_err)
    logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

CHUNKER_CLASS = SemanticChunker if ENHANCED_UTILS_AVAILABLE else BaseChunker

# XML Namespaces for SPL (Structured Product Labeling)
NS = {"hl7": "urn:hl7-org:v3"}

# LOINC Codes for Common Drug Label Sections
# Source: https://www.fda.gov/industry/structured-product-labeling-resources/section-headings-loinc
SECTION_CODES = {
    # --- HIGH PRIORITY SECTIONS (Included) ---
    "34066-3": ("highlights", "Highlights of Prescribing Information"),
    "34067-9": ("indications", "Indications & Usage"),
    "34068-7": ("dosage", "Dosage & Administration"), 
    "34070-3": ("contraindications", "Contraindications"),
    # Warnings can be coded as 34071-1 (Warnings) or 43685-7 (Warnings and Precautions)
    "34071-1": ("warnings", "Warnings"),
    "43685-7": ("warnings", "Warnings & Precautions"),
    "34084-4": ("adverse_reactions", "Adverse Reactions"),  # CORRECT code for Adverse Reactions (34072-9 is an old code for General Precautions, see below)
    "34073-7": ("interactions", "Drug Interactions"),
    "34066-1": ("boxed_warning", "Boxed Warning"),
    # Use in Specific Populations can be 34074-5 or parent 43684-0
    "34074-5": ("use_in_specific_populations", "Use in Specific Populations"),
    "43684-0": ("use_in_specific_populations", "Use in Specific Populations"),
    "34092-7": ("clinical_studies", "Clinical Studies"),  # CORRECT: was 34091-9
    
    # --- EXCLUDED SECTIONS ---
    "43678-2": ("dosage_forms", "Dosage Forms & Strengths"),  # EXCLUDED per requirements
    "34076-0": ("overdosage", "Overdosage"),  # EXCLUDED per requirements
    "34069-5": ("description", "Description"),  # EXCLUDED - inactive ingredients
    "34089-3": ("clinical_pharmacology", "Clinical Pharmacology"),  # EXCLUDED
    "34090-1": ("nonclinical_toxicology", "Nonclinical Toxicology"),  # EXCLUDED
    "34093-5": ("references", "References"),  # EXCLUDED
    "43683-2": ("recent_major_changes", "Recent Major Changes"),  # EXCLUDED - metadata about label updates
    "44425-7": ("storage", "Storage and Handling"),  # EXCLUDED
    "34075-2": ("patient_counseling", "Patient Counseling Information"),  # EXCLUDED
    "88436-1": ("patient_counseling_new", "Patient Counseling Information (New Code)"),  # EXCLUDED
    "42232-9": ("precautions", "Precautions"),  # EXCLUDED - covered in warnings
    "42229-5": ("spl_unclassified", "Unclassified Section"),  # EXCLUDED
    "51945-4": ("package_insert", "Package Insert"),  # EXCLUDED
    "48780-1": ("spl_product_data", "SPL Product Data Elements"),  # EXCLUDED
    # Old/incorrect codes that might appear in some documents
    "34072-9": ("precautions_old", "General Precautions (Old)"),  # EXCLUDED - use warnings instead
    "34091-9": ("animal_pharmacology", "Animal Pharmacology"),  # EXCLUDED
    "34088-5": ("overdosage_alt", "Overdosage (Alternate)"),  # EXCLUDED
    "34086-9": ("abuse", "Abuse Section"),  # EXCLUDED
    "34087-7": ("dependence", "Dependence Section"),  # EXCLUDED
    "34085-1": ("controlled_substance", "Controlled Substance"),  # EXCLUDED
    "43679-0": ("mechanism_of_action", "Mechanism of Action"),  # EXCLUDED
    "43681-6": ("pharmacodynamics", "Pharmacodynamics"),  # EXCLUDED
    "43682-4": ("pharmacokinetics", "Pharmacokinetics"),  # EXCLUDED
    "34083-6": ("carcinogenesis", "Carcinogenesis & Mutagenesis"),  # EXCLUDED
    "34079-4": ("labor_delivery", "Labor & Delivery"),  # EXCLUDED
    "34077-8": ("teratogenic_effects", "Teratogenic Effects"),  # EXCLUDED
    "34078-6": ("nonteratogenic_effects", "Nonteratogenic Effects"),  # EXCLUDED
    "34080-2": ("nursing_mothers", "Nursing Mothers (Old)"),  # EXCLUDED - covered in Use in Specific Populations
    "77290-5": ("lactation", "Lactation"),  # EXCLUDED - covered in Use in Specific Populations
    "77291-3": ("reproductive_potential", "Females & Males of Reproductive Potential"),  # EXCLUDED - covered in Use in Specific Populations
    "34081-0": ("pediatric_use", "Pediatric Use"),  # EXCLUDED - covered in Use in Specific Populations
    "34082-8": ("geriatric_use", "Geriatric Use"),  # EXCLUDED - covered in Use in Specific Populations
    "88828-9": ("renal_impairment", "Renal Impairment"),  # EXCLUDED - covered in Use in Specific Populations
    "88829-7": ("hepatic_impairment", "Hepatic Impairment"),  # EXCLUDED - covered in Use in Specific Populations
}

# Whitelist of LOINC codes to include (strict filtering)
WHITELIST_LOINC_CODES = {
    "34066-3",  # Highlights
    "34067-9",  # Indications & Usage
    "34068-7",  # Dosage & Administration
    "34070-3",  # Contraindications
    "34071-1",  # Warnings (older format)
    "43685-7",  # Warnings and Precautions (PLR format)
    "34084-4",  # Adverse Reactions (CORRECTED from 34072-9)
    "34073-7",  # Drug Interactions
    "34074-5",  # Use in Specific Populations
    "43684-0",  # Use in Specific Populations (parent section)
    "34092-7",  # Clinical Studies (CORRECTED from 34091-9)
    "34066-1",  # Boxed Warning (if present)
}

# Section weights for retrieval priority
SECTION_WEIGHTS = {
    "highlights": 1.0,
    "boxed_warning": 1.0,
    "contraindications": 0.95,
    "warnings": 0.9,
    "indications": 0.85,
    "dosage": 0.85,
    "adverse_reactions": 0.8,
    "interactions": 0.8,
    "use_in_specific_populations": 0.75,
    "clinical_studies": 0.7,
}

def get_text(element: Optional[ET.Element]) -> str:
    """Extract all text from element."""
    if element is None:
        return ""
    return " ".join(element.xpath(".//text()")).strip()


def parse_table_to_markdown(table_elem: ET.Element) -> str:
    """
    Parse an HTML table to Markdown format, handling colspans and rowspans.
    Uses shared parsing logic from ingestion_utils.
    """
    if table_elem is None:
        return ""
    
    # Import shared parsing function
    from ingestion_utils import _parse_table_to_rows
    
    parsed_rows, max_cols = _parse_table_to_rows(table_elem, ns=NS)
    
    if not parsed_rows:
        return ""
    
    # Build Markdown
    md_lines = []
    
    # Header
    header_row = parsed_rows[0]
    header_row.extend([""] * (max_cols - len(header_row)))
    md_lines.append("| " + " | ".join(header_row) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    
    # Body
    for row in parsed_rows[1:]:
        row.extend([""] * (max_cols - len(row)))
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def _parse_spl_xml_with_status(xml_path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Parse DailyMed SPL XML file with STRICT section filtering.
    
    Only includes whitelisted high-value clinical sections:
    - Highlights, Indications, Dosage, Contraindications
    - Warnings, Adverse Reactions, Drug Interactions
    - Use in Specific Populations, Clinical Studies
    
    Excludes: Dosage Forms, Description, References, Package Insert, etc.
    
    Note: For parent sections (Dosage, Warnings, etc.) that contain subsections,
    we aggregate text from all child subsections since the parent <text> is often empty.
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        label_type_code, label_type_display = extract_document_label_type(root, namespaces=NS)
        if not is_human_prescription_label(label_type_code, label_type_display):
            return None, "non_rx"

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

        # Parse ONLY whitelisted sections
        sections: Dict[str, Dict[str, Any]] = {}
        
        # First pass: collect all whitelisted sections and their content
        section_elements = root.xpath(".//hl7:section", namespaces=NS)
        
        for section in section_elements:
            code_elem = section.xpath("hl7:code", namespaces=NS)
            title_elem = section.xpath("hl7:title", namespaces=NS)
            text_elem = section.xpath("hl7:text", namespaces=NS)
            
            # STRICT FILTERING: Must have a whitelisted LOINC code
            if not code_elem:
                continue
                
            code = code_elem[0].get("code", "")
            
            # Skip if not in whitelist
            if code not in WHITELIST_LOINC_CODES:
                continue
            
            # Get section key and title from SECTION_CODES
            section_key, section_title = SECTION_CODES[code]
            
            # Get section text (direct content)
            section_text = get_text(text_elem[0]) if text_elem else ""
            table_text_elements: List[ET.Element] = []
            if text_elem:
                table_text_elements.append(text_elem[0])
            
            # IMPORTANT: For parent sections (Dosage, Warnings, etc.), also aggregate
            # text from child subsections, since parent <text> is often empty
            child_sections = section.xpath("hl7:component/hl7:section", namespaces=NS)
            child_texts = []
            for child in child_sections:
                child_text_elem = child.xpath("hl7:text", namespaces=NS)
                if child_text_elem:
                    table_text_elements.append(child_text_elem[0])
                    child_text = get_text(child_text_elem[0])
                    if child_text:
                        child_title = child.xpath("hl7:title/text()", namespaces=NS)
                        if child_title:
                            child_texts.append(f"{child_title[0]}: {child_text}")
                        else:
                            child_texts.append(child_text)
            
            # Combine parent text with child subsection texts
            all_text_parts = []
            if section_text:
                all_text_parts.append(section_text)
            if child_texts:
                all_text_parts.extend(child_texts)
            
            combined_text = "\n\n".join(all_text_parts) if all_text_parts else ""
            
            # Parse tables from the same text elements used for prose aggregation.
            # This keeps section prose/table context aligned for chunking.
            tables = []
            table_index = 0
            for te in table_text_elements:
                table_elements = te.xpath(".//hl7:table", namespaces=NS)
                for table in table_elements:
                    table_content = parse_table_to_markdown(table)
                    if table_content:
                        tables.append({
                            "index": table_index,
                            "content": table_content,
                            "type": f"table_{table_index}"
                        })
                        table_index += 1
            
            # Only add if has content
            if combined_text or tables:
                # If we already have this section (e.g. repeated codes), append
                if section_key in sections:
                    existing = sections[section_key]
                    if combined_text:
                        existing["text"] += "\n\n" + combined_text
                    existing["tables"].extend(tables)
                    existing["has_tables"] = existing["has_tables"] or (len(tables) > 0)
                else:
                    sections[section_key] = {
                        "title": section_title,
                        "text": combined_text,
                        "tables": tables,
                        "has_tables": len(tables) > 0
                    }
        
        # Extract Highlights from excerpts (special handling)
        # Highlights are stored in <excerpt><highlight> within whitelisted sections
        highlights_texts = []
        for section in section_elements:
            code_elem = section.xpath("hl7:code", namespaces=NS)
            if not code_elem:
                continue
            code = code_elem[0].get("code", "")
            if code not in WHITELIST_LOINC_CODES:
                continue
            
            for excerpt in section.xpath(".//hl7:excerpt//hl7:highlight", namespaces=NS):
                highlight_text = get_text(excerpt)
                if highlight_text and len(highlight_text) > 50:
                    highlights_texts.append(highlight_text)
        
        # Only add highlights section if we found content
        if highlights_texts:
            # Avoid duplicate content that's already in main sections
            combined_highlights = "\n\n".join(highlights_texts)
            sections["highlights"] = {
                "title": "Highlights of Prescribing Information",
                "text": combined_highlights[:15000],  # Limit size
                "tables": [],
                "has_tables": False
            }

        return {
            "set_id": set_id,
            "drug_name": drug_name,
            "title": title,
            "active_ingredients": active_ingredients,
            "manufacturer": manufacturer,
            "label_type_code": label_type_code,
            "label_type_display": label_type_display,
            "sections": sections,
            "source": "dailymed",
        }, "ok"
    except Exception as e:
        logger.warning("Failed to parse %s: %s", xml_path, e)
        return None, "parse_error"


def parse_spl_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Compatibility wrapper for callers that expect only parsed data."""
    drug, _status = _parse_spl_xml_with_status(xml_path)
    return drug


def create_chunks(drug: Dict[str, Any], chunker, validate_chunks: bool = True) -> List[Dict[str, Any]]:
    """
    Create chunks for whitelisted sections only.
    Uses SECTION_WEIGHTS for retrieval priority.
    """
    import hashlib
    
    chunks = []
    raw_chunks = []  # Collect chunks before validation
    set_id = drug.get("set_id", "")
    drug_name = drug.get("drug_name", "")
    label_title = drug.get("title", "") or drug_name
    manufacturer = drug.get("manufacturer", "")
    active_ingredients = drug.get("active_ingredients", [])
    sections = drug.get("sections", {})
    
    # Helper to generate section_id
    def generate_section_id(doc_id: str, section_title: str) -> str:
        content = f"{doc_id}:{section_title.lower().strip()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    base_metadata = {
        "set_id": set_id,
        "title": label_title,
        "source": "dailymed",
        "label_type_code": drug.get("label_type_code", ""),
        "label_type_display": drug.get("label_type_display", ""),
    }
    
    # Process ONLY whitelisted sections
    for section_key, section_data in sections.items():
        section_text = section_data["text"]
        section_title = section_data["title"]
        
        # Skip very short sections without tables
        if len(section_text) < 50 and not section_data.get("has_tables"):
            continue
        
        # Get section weight (default 0.7 for tables, 0.5 otherwise)
        section_weight = SECTION_WEIGHTS.get(section_key, 0.7 if section_data.get("has_tables") else 0.5)
        
        # Generate section_id
        section_id = generate_section_id(set_id, section_title)
        
        # Create chunk with drug context
        full_section_text = f"# {drug_name} - {section_title}\n\n{section_text}"
        
        # Use chunker for word-based splitting
        section_chunks = chunker.chunk_text(full_section_text)
        
        for j, chunk_data in enumerate(section_chunks):
            chunk_id = f"{set_id}_{section_key}_part{j}" if len(section_chunks) > 1 else f"{set_id}_{section_key}"
            chunk_text = chunk_data["text"]
            chunk_section_id = (
                generate_section_id(set_id, f"{section_title} Part {j}")
                if len(section_chunks) > 1
                else section_id
            )
            chunks.append({
                **base_metadata,
                "chunk_id": chunk_id,
                "drug_name": drug_name,
                "table_type": "",
                "is_table": False,
                "section_type": section_key,
                "section_title": section_title,
                "text": chunk_text,
                "has_tables": section_data.get("has_tables", False),
                "table_id": "",
                "manufacturer": manufacturer,
                "active_ingredients": active_ingredients,
                "table_count": len(section_data.get("tables", [])),
                # Chunking metadata for consistency
                "chunk_index": j,
                "total_chunks": len(section_chunks),
                "token_count": chunk_data.get("token_count", 0),
                # Parent-child ready fields
                "section_id": chunk_section_id,
                "parent_section_id": section_id if len(section_chunks) > 1 else None,
                "section_weight": section_weight,
                "page_content": chunk_text,
            })
        
        # Table chunks (only for whitelisted sections - already filtered in parse)
        for table in section_data.get("tables", []):
            table_full_text = f"# {drug_name} - {section_title} Table\n\n{table['content']}"
            table_section_id = generate_section_id(set_id, f"{section_title} Table")
            
            # Use chunker for table chunking if large
            table_chunks = chunker.chunk_text(table_full_text)
            
            for j, chunk_data in enumerate(table_chunks):
                chunk_id = f"{set_id}_{section_key}_table_{table['index']}_part{j}" if len(table_chunks) > 1 else f"{set_id}_{section_key}_table_{table['index']}"
                chunk_text = chunk_data["text"]
                chunk_section_id = (
                    generate_section_id(set_id, f"{section_title} Table {table['index']} Part {j}")
                    if len(table_chunks) > 1
                    else table_section_id
                )
                raw_chunks.append({
                    **base_metadata,
                    "chunk_id": chunk_id,
                    "drug_name": drug_name,
                    "is_table": True,
                    "section_type": "table",
                    "section_title": f"{section_title} Table",
                    "table_type": section_key,
                    "table_id": f"{section_key}_table_{table['index']}",
                    "text": chunk_text,
                    "has_tables": True,
                    "manufacturer": manufacturer,
                    "active_ingredients": active_ingredients,
                    "table_count": 1,
                    # Chunking metadata for consistency
                    "chunk_index": j,
                    "total_chunks": len(table_chunks),
                    "token_count": chunk_data.get("token_count", 0),
                    # Parent-child ready fields
                    "section_id": chunk_section_id,
                    "parent_section_id": table_section_id if len(table_chunks) > 1 else None,
                    "section_weight": 0.75,  # Tables get slightly higher weight
                    "page_content": chunk_text,
                })
    
    # Combine all chunks
    all_chunks = chunks + raw_chunks
    
    # Validate chunks if enabled (note: deduplication is now done in build_points for consistency)
    if validate_chunks and ENHANCED_UTILS_AVAILABLE:
        valid_chunks = []
        for chunk in all_chunks:
            is_valid, issues = QualityValidator.validate_chunk(
                chunk["text"],
                {k: chunk.get(k) for k in ["set_id", "chunk_id", "drug_name", "section_title"]}
            )
            if is_valid:
                valid_chunks.append(chunk)
            else:
                logger.debug("Skipping invalid chunk %s: %s", chunk.get("chunk_id"), issues)
        all_chunks = valid_chunks
        logger.debug("Validated chunks: %d valid out of %d total", len(all_chunks), len(chunks) + len(raw_chunks))
    
    return all_chunks




def build_points(chunks: List[Dict[str, Any]], embedding_provider: EmbeddingProvider,
                 validate_chunks: bool = True, dedup_chunks: bool = True) -> Tuple[List[PointStruct], List[str]]:
    """Convert chunks into Qdrant PointStructs with embeddings.
    
    Args:
        chunks: List of chunk dictionaries
        embedding_provider: Provider for generating embeddings
        validate_chunks: Whether to validate chunk quality before ingestion
        dedup_chunks: Whether to deduplicate chunks within batch
    """
    if not chunks:
        return [], []
    
    # Initialize deduplicator if enabled
    dedup = None
    if dedup_chunks and ContentDeduplicator is not None:
        dedup = ContentDeduplicator()
    
    # Deduplicate chunks (validation already done in create_chunks)
    filtered_chunks = []
    for chunk in chunks:
        # Deduplicate if enabled
        if dedup:
            metadata = {
                "set_id": chunk.get("set_id", ""),
                "chunk_id": chunk.get("chunk_id", "")
            }
            if dedup.is_duplicate(chunk["text"], metadata):
                logger.debug("Skipping duplicate chunk %s", chunk.get("chunk_id"))
                continue
        
        filtered_chunks.append(chunk)
    
    chunks = filtered_chunks
    texts = [chunk["text"] for chunk in chunks]
    
    # 1. Generate Dense Embeddings
    try:
        embeddings = embedding_provider.embed_batch(texts)
    except Exception as e:
        logger.error("Failed to generate embeddings: %s", e)
        return [], []
        
    points = []
    chunk_ids = []
    
    # 2. Build Points
    for i, chunk in enumerate(chunks):
        try:
            # Generate UUID from chunk_id for consistent ID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"dailymed:{chunk['chunk_id']}"))
            
            vector = {"dense": embeddings[i]}
            
            # Build the DailyMed payload with only the fields we want to retain
            payload = {
                # Core identifiers
                "chunk_id": chunk["chunk_id"],
                "set_id": chunk.get("set_id", ""),
                
                # Content
                "page_content": chunk.get("page_content", chunk["text"]),
                "title": chunk.get("title", chunk.get("drug_name", "")),
                
                # Section metadata
                "is_table": chunk.get("is_table", False),
                "section_type": chunk.get("section_type", ""),
                "section_title": chunk.get("section_title", ""),
                "table_type": chunk.get("table_type", ""),
                "table_id": chunk.get("table_id", ""),
                "has_tables": chunk.get("has_tables", False),
                "section_id": chunk.get("section_id", ""),
                "parent_section_id": chunk.get("parent_section_id"),
                "section_weight": chunk.get("section_weight", 0.0),
                
                # Drug label metadata
                "drug_name": chunk.get("drug_name", ""),
                "manufacturer": chunk.get("manufacturer", ""),
                "active_ingredients": chunk.get("active_ingredients", []),
                "label_type_code": chunk.get("label_type_code", ""),
                "label_type_display": chunk.get("label_type_display", ""),

                # Chunk metadata
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "token_count": chunk.get("token_count", 0),
                
                # Source info
                "source": chunk.get("source", "dailymed"),
                "table_count": chunk.get("table_count", 0),
                
                # Ingestion metadata
                "ingestion_timestamp": time.time(),
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
            chunk_ids.append(chunk["chunk_id"])
            
        except Exception as e:
            logger.error("Failed to build point for chunk %s: %s", chunk.get("chunk_id"), e)
            
    return points, chunk_ids


# Checkpoint file for DailyMed ingestion
CHECKPOINT_FILE = IngestionConfig.DAILYMED_CHECKPOINT_FILE

# Namespace for DailyMed checkpoint IDs to prevent collision with PMC/PubMed
DAILYMED_CHECKPOINT_NAMESPACE = "dailymed"

def _checkpoint_id(set_id: str) -> str:
    """Generate namespaced checkpoint ID for DailyMed labels."""
    return f"{DAILYMED_CHECKPOINT_NAMESPACE}:{set_id.strip()}"


def _resolve_checkpoint_line(line: str) -> str | None:
    """Resolve checkpoint line to namespaced ID, handling legacy plain set_ids."""
    value = line.strip()
    if not value:
        return None
    
    # Already namespaced
    if value.startswith(f"{DAILYMED_CHECKPOINT_NAMESPACE}:"):
        return value
    
    # Legacy format (plain set_id) - convert to namespaced
    return _checkpoint_id(value)


def load_checkpoint_namespaced(path: Path) -> set[str]:
    """Load checkpoint with legacy format support."""
    return {
        resolved
        for line in load_checkpoint_file(path)
        if (resolved := _resolve_checkpoint_line(line)) is not None
    }
