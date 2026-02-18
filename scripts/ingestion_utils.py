"""Shared utilities for ingestion scripts."""

from __future__ import annotations

import gzip
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime

import lxml.etree as ET


from config_ingestion import IngestionConfig, ensure_data_dirs
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import grpc
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# PMC XML Parsing - PRD v1.0 Compliant
# ============================================================================

# Section type normalization map per PRD
SECTION_TYPE_MAP = {
    "intro": "introduction",
    "introduction": "introduction",
    "methods": "methods",
    "materials": "methods",
    "materials|methods": "methods",
    "methodology": "methods",
    "results": "results",
    "disc": "discussion",
    "discussion": "discussion",
    "concl": "conclusions",
    "conclusions": "conclusions",
    "conclusion": "conclusions",
    "case-report": "case_report",
    "case-study": "case_report",
    "ack": "acknowledgments",
    "acknowledgments": "acknowledgments",
    "abstract": "abstract_text",
}

# ISO 3166-1 alpha-3 country codes (common ones for validation)
ISO_COUNTRY_CODES = {
    "USA", "GBR", "CHN", "DEU", "FRA", "JPN", "CAN", "AUS", "ITA", "ESP",
    "NLD", "BRA", "IND", "KOR", "SWE", "CHE", "AUT", "BEL", "DNK", "FIN",
    "GRC", "IRL", "MEX", "NOR", "NZL", "POL", "PRT", "RUS", "SGP", "ZAF",
    "TUR", "ARG", "CHL", "CZE", "HUN", "ISR", "MYS", "PHL", "ROU", "THA",
    "VNM", "UKR", "COL", "EGY", "IDN", "PAK", "BGD", "NGA", "ETH", "KEN",
    "TZA", "UGA", "ZWE", "ZMB", "MAR", "DZA", "TUN", "LBY", "SDN", "SOM",
    "NER", "MLI", "BFA", "SEN", "GIN", "GHA", "CIV", "TGO", "BEN", "NER",
    "CMR", "CAF", "TCD", "COG", "GAB", "GNQ", "STP", "BDI", "RWA", "COD",
}

class EmbeddingProvider:
    """Support for local, Cohere, and cloud inference embeddings."""
    
    def __init__(self) -> None:
        self.provider = IngestionConfig.EMBEDDING_PROVIDER.lower().strip()
        self.model = IngestionConfig.EMBEDDING_MODEL
        self.local_encoder = None
        self.openai_client = None

        if self.provider == "deepinfra":
            from openai import OpenAI
            api_key = os.getenv("DEEPINFRA_API_KEY")
            if not api_key:
                raise ValueError("DEEPINFRA_API_KEY not set - required for deepinfra embedding provider")
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepinfra.com/v1/openai"
            )
            logger.info("✅ DeepInfra embedding provider initialized (model: %s)", self.model)
        elif self.provider == "local":
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model: %s", self.model)
            self.local_encoder = SentenceTransformer(self.model)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "deepinfra":
            return self._embed_deepinfra(texts)
        elif self.local_encoder is not None:
            vectors = self.local_encoder.encode(texts, normalize_embeddings=True, batch_size=IngestionConfig.EMBEDDING_BATCH_SIZE)
            return [v.tolist() for v in vectors]
        else:
            raise RuntimeError("No embedding provider available")

    def _embed_deepinfra(self, texts: List[str]) -> List[List[float]]:
        """Embed using DeepInfra OpenAI-compatible API with sub-batching."""
        batch_size = IngestionConfig.EMBEDDING_BATCH_SIZE  # 64
        
        # Small enough to send in one call
        if len(texts) <= batch_size:
            return self._embed_deepinfra_single(texts)
        
        # Sub-batch large requests to avoid 500 errors
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            sub_batch = texts[i:i + batch_size]
            embeddings = self._embed_deepinfra_single(sub_batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _embed_deepinfra_single(self, texts: List[str]) -> List[List[float]]:
        """Send a single embedding request to DeepInfra."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error("DeepInfra embedding failed: %s", e)
            raise


# Evidence hierarchy for article types
EVIDENCE_HIERARCHY = {
    'meta-analysis': ('A', 1),
    'systematic-review': ('A', 1),
    'systematic review': ('A', 1),
    'practice-guideline': ('A', 1),
    'guideline': ('A', 1),
    'randomized-controlled-trial': ('A', 2),
    'randomized controlled trial': ('A', 2),
    'clinical-trial': ('A', 2),
    'clinical trial': ('A', 2),
    'cohort-study': ('B', 3),
    'cohort study': ('B', 3),
    'review': ('B', 3),
    'case-control': ('B', 4),
    'cross-sectional': ('B', 4),
    'case-report': ('C', 5),
    'case report': ('C', 5),
    'case-series': ('C', 5),
    'editorial': ('D', 6),
    'letter': ('D', 6),
    'comment': ('D', 6),
}

class SectionFilter:
    EXCLUDED_TYPES = {
        "references", "bibliography", "acknowledgments", "funding",
        "conflict", "disclosure", "author_contributions", "supplementary"
    }
    
    EXCLUDED_TITLES = {
        "references", "bibliography", "literature cited", "acknowledgments",
        "acknowledgements", "funding", "financial support", "conflict of interest",
        "author contributions", "disclosure", "supplementary data"
    }

    @classmethod
    def should_exclude(cls, section: Dict[str, Any]) -> bool:
        # Check type
        sec_type = section.get("type", "").lower()
        if sec_type in cls.EXCLUDED_TYPES:
            return True
            
        # Check title
        title = section.get("title", "").lower()
        if any(ex in title for ex in cls.EXCLUDED_TITLES):
            return True
            
        return False


import threading

_GLOBAL_TOKENIZER = None
_TOKENIZER_LOCK = threading.Lock()

class Chunker:
    def __init__(self, chunk_size: int = None, overlap: int = None):
        # Read defaults from IngestionConfig to ensure consistency
        # This prevents hardcoded values from getting out of sync with .env
        self.chunk_size = chunk_size if chunk_size is not None else IngestionConfig.CHUNK_SIZE_TOKENS
        self.overlap = overlap if overlap is not None else IngestionConfig.CHUNK_OVERLAP_TOKENS
        self._load_tokenizer()
            
    def _load_tokenizer(self):
        global _GLOBAL_TOKENIZER
        
        # Double-check locking pattern
        if _GLOBAL_TOKENIZER is not None:
            self.tokenizer = _GLOBAL_TOKENIZER
            return

        with _TOKENIZER_LOCK:
            if _GLOBAL_TOKENIZER is not None:
                self.tokenizer = _GLOBAL_TOKENIZER
                return

            try:
                from transformers import AutoTokenizer
                # Use Qwen3-Embedding-0.6B tokenizer for consistency with embedding model
                # This ensures token counts are accurate for chunking
                logger.info("Loading tokenizer (Qwen/Qwen3-Embedding-0.6B)...")
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3-Embedding-0.6B", 
                    trust_remote_code=True,
                    padding_side='left'  # Critical for correct last-token pooling
                )
                _GLOBAL_TOKENIZER = tokenizer
                self.tokenizer = tokenizer
                logger.info("✅ Tokenizer loaded with padding_side='left'")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer (Qwen/Qwen3-Embedding-0.6B): {e}")
                self.tokenizer = None
            
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the loaded tokenizer."""
        if not text:
            return 0
        
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception:
                pass  # Fall through to approximation
        
        # Fallback: approximate with word count (1 token ≈ 0.75 words)
        return int(len(text.split()) / 0.75)
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
            
        # Simple word-based fallback if tokenizer fails
        if not self.tokenizer:
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunks.append({
                    "text": " ".join(chunk_words),
                    "token_count": len(chunk_words)
                })
            return chunks
            
        # Token-aware chunking
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_tokens)
            })
            
        return chunks


def get_text(element: Optional[ET.Element], default: str = "") -> str:
    """Extract all text from an element recursively."""
    if element is None:
        return default
    return " ".join(element.xpath(".//text()")).strip()


def get_text_excluding_children(element: ET.Element, excluded_tags: List[str]) -> str:
    """Extract text from element but exclude specific child tags (e.g. nested sections)."""
    text = element.text or ""
    for child in element:
        if child.tag not in excluded_tags:
            text += get_text_excluding_children(child, excluded_tags)
        text += child.tail or ""
    return " ".join(text.split())

def _parse_table_to_rows(table_elem: ET.Element, ns: Optional[Dict[str, str]] = None) -> Tuple[List[List[str]], int]:
    """
    Parse an HTML/XML table element into rows with rowspan and colspan support.
    
    Args:
        table_elem: The table element (from PMC or DailyMed)
        ns: Optional namespace dict for DailyMed HL7 tables
        
    Returns:
        Tuple of (parsed_rows, max_cols)
    """
    # XPath with or without namespace
    if ns:
        all_trs = table_elem.xpath(".//hl7:tr", namespaces=ns)
    else:
        all_trs = table_elem.xpath(".//tr")
    
    if not all_trs:
        return [], 0
    
    parsed_rows = []
    max_cols = 0
    rowspan_tracker: Dict[int, Tuple[int, str]] = {}
    
    for tr in all_trs:
        row_cells = []
        # XPath for cells with or without namespace
        if ns:
            cells = tr.xpath("hl7:th|hl7:td", namespaces=ns)
        else:
            cells = tr.xpath("th|td")
        
        cell_idx = 0
        
        for cell in cells:
            # Skip columns that have active rowspan
            while cell_idx in rowspan_tracker and rowspan_tracker[cell_idx][0] > 0:
                remaining, carried_text = rowspan_tracker[cell_idx]
                row_cells.append(carried_text)
                rowspan_tracker[cell_idx] = (remaining - 1, carried_text)
                cell_idx += 1
            
            cell_text = get_text(cell).replace("|", "\\|").replace("\n", " ").strip()
            colspan = int(cell.get("colspan", "1"))
            rowspan = int(cell.get("rowspan", "1"))
            
            row_cells.append(cell_text)
            
            # Set up rowspan tracker for future rows
            if rowspan > 1:
                rowspan_tracker[cell_idx] = (rowspan - 1, cell_text)
            
            # Handle colspan - add empty cells
            for _ in range(colspan - 1):
                cell_idx += 1
                row_cells.append("")  # Empty cell for skipped col
                # Also need to handle rowspan for spanned columns
                if rowspan > 1:
                    rowspan_tracker[cell_idx] = (rowspan - 1, cell_text)
            
            cell_idx += 1
        
        # Fill any remaining active rowspans at the end of the row
        while cell_idx in rowspan_tracker and rowspan_tracker[cell_idx][0] > 0:
            remaining, carried_text = rowspan_tracker[cell_idx]
            row_cells.append(carried_text)
            rowspan_tracker[cell_idx] = (remaining - 1, carried_text)
            cell_idx += 1
        
        if row_cells:
            parsed_rows.append(row_cells)
            max_cols = max(max_cols, len(row_cells))
    
    return parsed_rows, max_cols


def parse_table_markdown(table_wrap: ET.Element, ns: Optional[Dict[str, str]] = None) -> str:
    """
    Extract table as markdown format with improved colspan and rowspan support.
    
    Args:
        table_wrap: The table-wrap element (PMC) or section element (DailyMed)
        ns: Optional namespace dict for DailyMed HL7 tables
    """
    # Extract caption
    if ns:
        caption_elem = table_wrap.xpath(".//hl7:caption", namespaces=ns)
    else:
        caption_elem = table_wrap.xpath(".//caption")
    caption = get_text(caption_elem[0]) if caption_elem else ""
    
    # Extract footnotes (PMC only)
    footnotes = ""
    if not ns:
        footnotes_elem = table_wrap.xpath(".//table-wrap-foot")
        if footnotes_elem:
            footnote_texts = []
            for fn in footnotes_elem[0].xpath(".//p|.//fn"):
                fn_text = get_text(fn).strip()
                if fn_text:
                    footnote_texts.append(fn_text)
            if footnote_texts:
                footnotes = "\n\nFootnotes: " + "; ".join(footnote_texts)
    
    # Find table element
    if ns:
        table = table_wrap.xpath(".//hl7:table", namespaces=ns)
    else:
        table = table_wrap.xpath(".//table")
    
    if not table:
        # Image-only table - return caption as fallback
        if caption:
            return f"**{caption}**{footnotes}"
        return ""
    
    table_elem = table[0]
    parsed_rows, max_cols = _parse_table_to_rows(table_elem, ns)
    
    if not parsed_rows:
        return f"**{caption}**{footnotes}" if caption else ""
    
    # Build Markdown
    rows = []
    
    # Assume first row is header
    header_row = parsed_rows[0]
    header_row.extend([""] * (max_cols - len(header_row)))
    rows.append("| " + " | ".join(header_row) + " |")
    rows.append("| " + " | ".join(["---"] * max_cols) + " |")
    
    for row in parsed_rows[1:]:
        row.extend([""] * (max_cols - len(row)))
        rows.append("| " + " | ".join(row) + " |")
    
    markdown = "\n".join(rows) + footnotes
    if caption:
        markdown = f"**{caption}**\n\n{markdown}"
    
    return markdown

def parse_table_row_by_row(table_wrap: ET.Element, ns: Optional[Dict[str, str]] = None) -> str:
    """
    Extract table in row-by-row format with rowspan support.
    
    Args:
        table_wrap: The table-wrap element (PMC) or section element (DailyMed)
        ns: Optional namespace dict for DailyMed HL7 tables
    """
    # Extract caption
    if ns:
        caption_elem = table_wrap.xpath(".//hl7:caption", namespaces=ns)
    else:
        caption_elem = table_wrap.xpath(".//caption")
    caption = get_text(caption_elem[0]) if caption_elem else ""
    
    # Extract footnotes (PMC only)
    footnotes_text = ""
    if not ns:
        footnotes_elem = table_wrap.xpath(".//table-wrap-foot")
        if footnotes_elem:
            footnote_texts = []
            for fn in footnotes_elem[0].xpath(".//p|.//fn"):
                fn_text = get_text(fn).strip()
                if fn_text:
                    footnote_texts.append(fn_text)
            if footnote_texts:
                footnotes_text = "\n\nFootnotes: " + "; ".join(footnote_texts)
    
    # Find table element
    if ns:
        table = table_wrap.xpath(".//hl7:table", namespaces=ns)
    else:
        table = table_wrap.xpath(".//table")
    
    if not table:
        # Image-only table - return caption as fallback
        if caption:
            return f"Table: {caption}{footnotes_text}"
        return ""
    
    table_elem = table[0]
    parsed_rows, max_cols = _parse_table_to_rows(table_elem, ns)
    
    if not parsed_rows or len(parsed_rows) < 2:
        return f"Table: {caption}{footnotes_text}" if caption else ""
    
    # First row is headers
    headers = parsed_rows[0]
    
    # Build row-by-row format
    rows = []
    for i, row_cells in enumerate(parsed_rows[1:], 1):
        row_parts = []
        for j, cell_text in enumerate(row_cells):
            if cell_text:  # Only include non-empty cells
                header = headers[j] if j < len(headers) else f"Col{j+1}"
                row_parts.append(f"{header}: {cell_text}")
        
        if row_parts:
            rows.append(f"Row {i}: " + ", ".join(row_parts))
    
    result = f"Table: {caption}. " if caption else "Table: "
    result += ". ".join(rows) + footnotes_text
    return result


def extract_tables(root: ET.Element) -> List[Dict[str, Any]]:
    """Extract tables with support for alternatives, image-only tables, and footnotes."""
    tables = []
    for i, table_wrap in enumerate(root.xpath(".//table-wrap")[:15], 1):  # Increased limit to 15 tables
        # Check for <alternatives> wrapper - prefer structured table inside
        alternatives = table_wrap.xpath(".//alternatives")
        if alternatives:
            # Use the first table found within alternatives
            alt_table = alternatives[0].xpath(".//table")
            if alt_table:
                # Create a temporary wrapper for parsing
                temp_wrap = ET.Element("table-wrap")
                caption = table_wrap.xpath(".//caption")
                if caption:
                    temp_wrap.append(caption[0])
                temp_wrap.append(alt_table[0])
                # Also copy footnotes if present
                footnotes = table_wrap.xpath(".//table-wrap-foot")
                if footnotes:
                    temp_wrap.append(footnotes[0])
                table_wrap = temp_wrap
        
        # Check for image-only table (graphic without table)
        table_elem = table_wrap.xpath(".//table")
        graphic_elem = table_wrap.xpath(".//graphic")
        
        caption = get_text(table_wrap.xpath(".//caption")[0]) if table_wrap.xpath(".//caption") else ""
        
        if not table_elem and graphic_elem:
            # Image-only table - use caption + footnotes as fallback
            footnotes_elem = table_wrap.xpath(".//table-wrap-foot")
            footnotes = ""
            if footnotes_elem:
                footnote_texts = []
                for fn in footnotes_elem[0].xpath(".//p|.//fn"):
                    fn_text = get_text(fn).strip()
                    if fn_text:
                        footnote_texts.append(fn_text)
                if footnote_texts:
                    footnotes = "\n\nFootnotes: " + "; ".join(footnote_texts)
            
            tables.append({
                "id": f"table-{i}",
                "caption": caption,
                "markdown": f"**{caption}** (Image-based table){footnotes}",
                "row_by_row": f"Table: {caption} (Image-based table){footnotes}",
                "is_image_only": True
            })
        else:
            tables.append({
                "id": f"table-{i}",
                "caption": caption,
                "markdown": parse_table_markdown(table_wrap),
                "row_by_row": parse_table_row_by_row(table_wrap),
                "is_image_only": False
            })
    return tables

def classify_evidence_grade(article_type: str, pub_types: List[str]) -> tuple:
    article_type_lower = article_type.lower().replace("_", "-").replace(" ", "-")
    pub_types_lower = [pt.lower() for pt in pub_types]
    
    for pattern, (grade, level) in EVIDENCE_HIERARCHY.items():
        if pattern in article_type_lower:
            return grade, level
        for pt in pub_types_lower:
            if pattern in pt:
                return grade, level
    return "B", 3

def parse_pmc_xml(xml_path: Path, require_pmid: bool = True, require_open_access: bool = True) -> Optional[Dict[str, Any]]:
    """
    Parse PMC XML using lxml with PRD v1.0 compliant extraction.
    
    Args:
        xml_path: Path to the XML file
        require_pmid: If True, skip articles without PMID (PRD Hard Fail rule)
        require_open_access: If True, skip non-open-access articles
    
    Returns:
        Article dict following PRD JSON structure, or None if invalid/skipped.
    """
    try:
        # Parse XML
        if xml_path.name.endswith(".xml.gz"):
            with gzip.open(xml_path, "rb") as f:
                root = ET.fromstring(f.read())
            pmcid_from_file = xml_path.stem.replace(".xml", "")
        else:
            root = ET.parse(str(xml_path)).getroot()
            pmcid_from_file = xml_path.stem

        # Get article and article-meta elements
        article_elem = root.find(".//article")
        if article_elem is None:
            logger.debug(f"No article element found in {xml_path}")
            return None
        
        article_meta = article_elem.find("front/article-meta")
        if article_meta is None:
            logger.debug(f"No article-meta found in {xml_path}")
            return None

        # === 1. Document Metadata (PRD Section 1.1) ===
        article_type = _extract_article_type(root)
        language = _extract_language(root)
        identifiers = _extract_identifiers(article_meta)
        
        # Hard Fail: PMID must exist
        if require_pmid and not identifiers.get("pmid"):
            logger.debug(f"Skipping {xml_path}: No PMID found")
            return None
        
        # Use PMCID from file if not in XML
        if not identifiers.get("pmcid"):
            identifiers["pmcid"] = pmcid_from_file

        # === 2. Publication Metadata (PRD Section 1.2) ===
        pub_date, year = _extract_publication_date(article_meta)
        journal_info = _extract_journal_info(root)
        country_code, country_name = _extract_country(root)

        # === 3. Content Structure (PRD Section 1.3) ===
        article_title = _extract_article_title(article_meta)
        if not article_title:
            logger.debug(f"Skipping {xml_path}: No article title")
            return None

        # Check for body and determine has_full_text
        body_elem = article_elem.find("body")
        has_full_text = False
        body_text = ""
        if body_elem is not None:
            body_text = get_text(body_elem).strip()
            has_full_text = len(body_text) > 100

        # Open Access Check (PRD Section 1.3)
        is_open_access = _extract_open_access(article_meta)
        if require_open_access and not is_open_access:
            logger.debug(f"Skipping {xml_path}: Not open access")
            return None

        # === 4. Controlled Vocabulary (PRD Section 1.4) ===
        mesh_terms, keywords = _extract_keywords(article_meta)

        # === 5. Full Text Sections (PRD Section 1.5) ===
        sections = _extract_full_text_sections(body_elem)
        
        # Update has_full_text based on sections
        if sections:
            has_full_text = True

        # === 6. Tables (PRD Section 1.6) ===
        tables = _extract_tables(article_elem)
        tables_in_floats = sum(1 for t in tables if t.get("location") == "floating")

        # === 7. Evidence Grade (existing functionality) ===
        pub_types = [get_text(s) for s in root.xpath(".//article-categories//subject")]
        evidence_grade, evidence_level = classify_evidence_grade(article_type, pub_types)

        # === Build Output Structure per PRD Section 3 ===
        output = {
            # Document ID for indexing
            "document_id": f"pmid_{identifiers['pmid']}" if identifiers.get("pmid") else f"pmcid_{identifiers['pmcid']}",
            
            "metadata": {
                "article_type": article_type,
                "language": language,
                "identifiers": {
                    "pmid": identifiers.get("pmid"),
                    "doi": identifiers.get("doi"),
                    "pmcid": identifiers.get("pmcid"),
                    "publisher_id": identifiers.get("publisher_id"),
                },
                "publication": {
                    "date": pub_date,
                    "year": year,
                    "journal": {
                        "title": journal_info["title"],
                        "abbreviation": journal_info["abbreviation"],
                        "publisher": journal_info["publisher"],
                        "issn": journal_info["issn_electronic"],
                    },
                    "country": country_code,
                    "country_name": country_name,
                },
                "content_flags": {
                    "has_full_text": has_full_text,
                    "is_open_access": is_open_access,
                },
                "classification": {
                    "mesh_terms": mesh_terms,
                    "keywords": keywords,
                }
            },
            "content": {
                "title": article_title,
                "sections": sections,
                "tables": tables,
            },
            "extraction_metadata": {
                "schema_version": "1.0",
                "processing_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "empty_body_detected": not has_full_text,
                "tables_in_floats_group": tables_in_floats,
            },
            
            # Legacy fields for backward compatibility
            "pmcid": identifiers.get("pmcid"),
            "pmid": identifiers.get("pmid"),
            "doi": identifiers.get("doi"),
            "title": article_title,
            "abstract": _extract_abstract(article_meta),
            "full_text": body_text if has_full_text else "",
            "structured_sections": sections,  # Renamed for clarity
            "section_titles": [s["title"] for s in sections],
            "tables": [
                {
                    "id": t["id"],
                    "caption": f"{t['label']}: {t['caption_title']}".strip(": "),
                    "markdown": t["content"],
                    "row_by_row": t["content"],  # Same for now
                    "is_image_only": False,
                } for t in tables
            ],
            "year": year,
            "journal": journal_info["title"],
            "keywords": keywords,
            "mesh_terms": mesh_terms,
            "country": country_code or country_name,
            "evidence_grade": evidence_grade,
            "evidence_level": evidence_level,
            "article_type": article_type,
            "publication_type_list": pub_types,
            "is_open_access": is_open_access,
            "has_full_text": has_full_text,
        }

        return output

    except Exception as e:
        logger.debug(f"Error parsing {xml_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def upsert_with_retry(client: QdrantClient, points: List[PointStruct]) -> None:
    """Upsert points with robust retry logic.
    
    Uses wait=False for throughput — Qdrant's optimizer can cause NOT_FOUND
    errors during wait=True verification even though the data IS written.
    """
    for attempt in range(IngestionConfig.MAX_RETRIES):
        try:
            client.upsert(
                collection_name=IngestionConfig.COLLECTION_NAME,
                points=points,
                wait=False,  # Don't block on optimizer — avoids NOT_FOUND errors
            )
            return
        except Exception as exc:
            error_str = str(exc)
            
            # NOT_FOUND during upsert means Qdrant's optimizer moved/merged
            # segments between write and confirmation. Data IS written.
            # Treat as success — do NOT retry.
            if "NOT_FOUND" in error_str or "Not found: No point with id" in error_str:
                logger.debug("NOT_FOUND during upsert (optimizer race) — points likely written, skipping retry")
                return
            
            # Check for specific "Too many open files" error (RocksDB/GRPC)
            is_file_limit = "Too many open files" in error_str
            
            if attempt == IngestionConfig.MAX_RETRIES - 1:
                if is_file_limit:
                    logger.critical("Failed after retries due to file limit. Server ulimits need increasing.")
                raise
            
            # Backoff strategy
            if is_file_limit:
                # Aggressive backoff for file/resource limits
                wait_for = 30 * (attempt + 1)
                logger.warning("File limit hit during upsert. Waiting %ss for cleanup... (Attempt %s/%s)", 
                             wait_for, attempt + 1, IngestionConfig.MAX_RETRIES)
            else:
                # Standard exponential backoff
                wait_for = 2**attempt
                logger.warning("Upsert retry %s/%s after error: %s", 
                             attempt + 1, IngestionConfig.MAX_RETRIES, error_str[:200])
            
            time.sleep(wait_for)

import hashlib

def generate_section_id(doc_id: str, section_title: str) -> str:
    """
    Generate a unique section ID for parent-child indexing.
    
    This creates a deterministic hash that can be used to group chunks
    belonging to the same section.
    
    Args:
        doc_id: The document ID (PMCID or PMID)
        section_title: The section title
        
    Returns:
        A unique section ID hash
    """
    content = f"{doc_id}:{section_title.lower().strip()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_section_weight(section_type: str) -> float:
    """
    Get the importance weight for a section type.
    
    Higher weights indicate more clinically relevant sections.
    Used for ranking/search relevance boosting.
    
    Args:
        section_type: The type of section (e.g., 'abstract', 'methods', 'results')
        
    Returns:
        Weight value between 0.0 and 1.0
    """
    weights = {
        "abstract": 1.0,
        "abstract_text": 1.0,
        "title": 1.0,
        "introduction": 0.7,
        "methods": 0.6,
        "results": 0.9,
        "discussion": 0.8,
        "conclusions": 0.85,
        "conclusion": 0.85,
        "case_report": 0.8,
        "table": 0.75,
        "figure": 0.6,
        "references": 0.1,
        "acknowledgments": 0.1,
        "funding": 0.1,
        "body": 0.5,
        "other": 0.5,
    }
    return weights.get(section_type.lower(), 0.5)


# ============================================================================
# PRD v1.0 Compliant PMC XML Parsing Functions
# ============================================================================

def _extract_identifiers(article_meta: ET.Element) -> Dict[str, Optional[str]]:
    """Extract document identifiers per PRD Section 1.1."""
    identifiers = {
        "pmid": None,
        "doi": None,
        "pmcid": None,
        "publisher_id": None,
    }
    
    for aid in article_meta.xpath(".//article-id"):
        id_type = aid.get("pub-id-type")
        text = (aid.text or "").strip()
        if not text:
            continue
        if id_type == "pmid":
            identifiers["pmid"] = text
        elif id_type == "doi":
            identifiers["doi"] = text
        elif id_type == "pmcid":
            identifiers["pmcid"] = text
        elif id_type == "publisher-id":
            identifiers["publisher_id"] = text
    
    return identifiers


def _extract_article_type(root: ET.Element) -> str:
    """Extract article type per PRD Section 1.1."""
    article_elem = root.xpath(".//article")
    if article_elem:
        art_type = article_elem[0].get("article-type", "")
        if art_type:
            return art_type.lower()
    return "other"


def _extract_language(root: ET.Element) -> str:
    """Extract language per PRD Section 1.1."""
    article_elem = root.xpath(".//article")
    if article_elem:
        lang = article_elem[0].get("{http://www.w3.org/XML/1998/namespace}lang")
        if lang:
            return lang.lower()
        # Try without namespace
        lang = article_elem[0].get("xml:lang")
        if lang:
            return lang.lower()
    return "en"


def _extract_publication_date(article_meta: ET.Element) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract publication date per PRD Section 1.2 and 4.2.
    Returns (iso_date, year_int).
    """
    dates = article_meta.findall("pub-date")
    iso_date = None
    year = None
    
    # Priority 1: epub with iso-8601-date
    for d in dates:
        if d.get("pub-type") == "epub":
            iso = d.get("iso-8601-date")
            if iso and re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
                iso_date = iso
                year = int(iso[:4])
                return iso_date, year
    
    # Priority 2: any date with iso-8601-date
    for d in dates:
        iso = d.get("iso-8601-date")
        if iso and re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
            iso_date = iso
            year = int(iso[:4])
            return iso_date, year
    
    # Priority 3: construct from year-month-day
    for d in dates:
        if d.get("pub-type") in ["epub", "collection", "ppub"]:
            year_elem = d.find("year")
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                    month_elem = d.find("month")
                    day_elem = d.find("day")
                    month = month_elem.text if month_elem is not None and month_elem.text else "01"
                    day = day_elem.text if day_elem is not None and day_elem.text else "01"
                    # Normalize month/day to 2 digits
                    iso_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    return iso_date, year
                except (ValueError, TypeError):
                    continue
    
    # Last resort: just get any year
    for d in dates:
        year_elem = d.find("year")
        if year_elem is not None and year_elem.text:
            try:
                year = int(year_elem.text)
                iso_date = f"{year}-01-01"
                return iso_date, year
            except (ValueError, TypeError):
                continue
    
    return iso_date, year


def _extract_journal_info(root: ET.Element) -> Dict[str, str]:
    """Extract journal metadata per PRD Section 1.2."""
    journal_meta = root.find(".//journal-meta")
    info = {
        "title": "",
        "abbreviation": "",
        "publisher": "",
        "issn_electronic": "",
    }
    
    if journal_meta is None:
        return info
    
    # Journal title
    title_group = journal_meta.find(".//journal-title-group")
    if title_group is not None:
        title_elem = title_group.find("journal-title")
        if title_elem is not None and title_elem.text:
            info["title"] = title_elem.text
    
    # NLM abbreviation
    for jid in journal_meta.findall("journal-id"):
        if jid.get("journal-id-type") == "nlm-ta":
            info["abbreviation"] = (jid.text or "").strip()
            break
    
    # Publisher
    publisher = journal_meta.find(".//publisher/publisher-name")
    if publisher is not None and publisher.text:
        info["publisher"] = publisher.text
    
    # Electronic ISSN
    for issn in journal_meta.findall("issn"):
        if issn.get("pub-type") == "epub":
            info["issn_electronic"] = (issn.text or "").strip()
            break
    
    return info


def _extract_country(root: ET.Element) -> Tuple[str, str]:
    """
    Extract country per PRD Section 1.2 and 4.1.
    Returns (country_code, country_name).
    """
    article_meta = root.find(".//article-meta")
    if article_meta is None:
        return "", ""
    
    country_code = ""
    country_name = ""
    
    # Priority 1: Corresponding author affiliation
    corresp = article_meta.find("corresp")
    if corresp is not None:
        country_elem = corresp.find(".//country")
        if country_elem is not None:
            code = country_elem.get("country")
            if code:
                country_code = code.upper()
            country_name = get_text(country_elem)
    
    # Priority 2: First contrib-group affiliation
    if not country_code:
        contrib_group = article_meta.find("contrib-group")
        if contrib_group is not None:
            aff = contrib_group.find(".//aff")
            if aff is not None:
                country_elem = aff.find("country")
                if country_elem is not None:
                    code = country_elem.get("country")
                    if code:
                        country_code = code.upper()
                    country_name = get_text(country_elem)
    
    # Priority 3: Any affiliation
    if not country_code:
        for aff in root.xpath(".//aff"):
            country_elem = aff.find("country")
            if country_elem is not None:
                code = country_elem.get("country")
                if code:
                    country_code = code.upper()
                    country_name = get_text(country_elem)
                    break
    
    return country_code, country_name


def _extract_open_access(article_meta: ET.Element) -> bool:
    """
    Extract open access status per PRD Section 1.3.
    Returns True if article has open-access or CC license.
    """
    permissions = article_meta.find("permissions")
    if permissions is None:
        return False
    
    for license_elem in permissions.findall("license"):
        license_type = license_elem.get("license-type", "").lower()
        
        # Check for open-access indicators in license-type
        if any(indicator in license_type for indicator in ["open-access", "cc-", "cc0"]):
            return True
        
        # Check license text content for CC licenses
        license_text = (license_elem.text or "").lower()
        for elem in license_elem.iter():
            license_text += (elem.text or "").lower() + (elem.tail or "").lower()
        
        if any(indicator in license_text for indicator in ["creativecommons", "cc by", "cc-by", "cc0"]):
            return True
    
    return False


def _extract_keywords(article_meta: ET.Element) -> Tuple[List[str], List[str]]:
    """
    Extract MeSH terms and author keywords per PRD Section 1.4.
    Returns (mesh_terms, keywords).
    """
    mesh_terms = []
    keywords = []
    
    for kwd_group in article_meta.findall("kwd-group"):
        group_type = kwd_group.get("kwd-group-type", "").lower()
        
        if group_type == "mesh":
            for kwd in kwd_group.findall("kwd"):
                term = get_text(kwd)
                if term:
                    # Handle qualifiers (split on /)
                    mesh_terms.append(term)
        elif group_type in ["author-keywords", "author keywords"]:
            for kwd in kwd_group.findall("kwd"):
                term = get_text(kwd)
                if term:
                    keywords.append(term)
        elif not group_type:
            # Untyped as fallback for keywords
            for kwd in kwd_group.findall("kwd"):
                term = get_text(kwd)
                if term:
                    keywords.append(term)
    
    return mesh_terms, keywords


def _extract_article_title(article_meta: ET.Element) -> str:
    """Extract article title per PRD Section 1.3."""
    title_group = article_meta.find("title-group")
    if title_group is None:
        return ""
    
    title_elem = title_group.find("article-title")
    if title_elem is None:
        return ""
    
    # Use xpath string() to get all text content recursively
    # This handles inline tags like <sub>, <sup> correctly
    title_text = title_elem.xpath("string()")
    return " ".join(title_text.split())


def _normalize_section_type(sec_type: Optional[str]) -> str:
    """Normalize section type per PRD Section 4.4."""
    if not sec_type:
        return "other"
    return SECTION_TYPE_MAP.get(sec_type.lower(), "other")


def _extract_full_text_sections(body: ET.Element) -> List[Dict[str, Any]]:
    """
    Extract full text sections per PRD Section 1.5.
    Only extracts from /article/body, excludes /article/back.
    """
    sections = []
    
    if body is None:
        return sections
    
    # Direct paragraphs not in section (rare but possible)
    direct_paras = []
    for child in body:
        if child.tag == "p":
            para_text = get_text(child)
            if para_text:
                direct_paras.append(para_text)
    
    if direct_paras:
        sections.append({
            "type": "introduction",
            "id": "intro-direct",
            "title": "Introduction",
            "content": "\n\n".join(direct_paras)
        })
    
    # Sections
    for i, sec in enumerate(body.xpath("./sec")):
        sec_id = sec.get("id", f"sec{i+1}")
        sec_type_attr = sec.get("sec-type", "")
        
        title_elem = sec.find("title")
        sec_title = get_text(title_elem) if title_elem is not None else "Section"
        
        # Get content from direct child paragraphs only
        content_parts = []
        for p in sec.xpath("./p"):
            p_text = get_text(p)
            if p_text:
                content_parts.append(p_text)
        
        if not content_parts:
            continue
        
        # Normalize section type
        sec_type = _normalize_section_type(sec_type_attr)
        if sec_type == "other" and sec_title:
            # Infer from title
            t_low = sec_title.lower()
            if "method" in t_low:
                sec_type = "methods"
            elif "result" in t_low:
                sec_type = "results"
            elif "discuss" in t_low:
                sec_type = "discussion"
            elif "conclusion" in t_low:
                sec_type = "conclusions"
            elif "intro" in t_low:
                sec_type = "introduction"
            elif "case" in t_low:
                sec_type = "case_report"
        
        sections.append({
            "type": sec_type,
            "id": sec_id,
            "title": sec_title,
            "content": "\n\n".join(content_parts)
        })
    
    return sections


def _table_to_markdown(table_elem: ET.Element) -> str:
    """Convert XHTML table to Markdown per PRD Section 4.3."""
    rows, max_cols = _parse_table_to_rows(table_elem)
    
    if not rows:
        return ""
    
    md_lines = []
    
    # Header row
    header = rows[0]
    header.extend([""] * (max_cols - len(header)))
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join([":---"] * max_cols) + " |")
    
    # Data rows
    for row in rows[1:]:
        row.extend([""] * (max_cols - len(row)))
        md_lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(md_lines)


def _extract_table_footnotes(table_wrap: ET.Element) -> List[str]:
    """Extract table footnotes per PRD Section 1.6."""
    footnotes = []
    foot_elem = table_wrap.find("table-wrap-foot")
    
    if foot_elem is not None:
        for fn in foot_elem.findall("fn"):
            fn_text = get_text(fn).strip()
            if fn_text:
                footnotes.append(fn_text)
        # Also check for direct p elements
        for p in foot_elem.findall("p"):
            p_text = get_text(p).strip()
            if p_text and p_text not in footnotes:
                footnotes.append(p_text)
    
    return footnotes


def _extract_tables(root: ET.Element) -> List[Dict[str, Any]]:
    """
    Extract tables per PRD Section 1.6 - Dual Location Strategy.
    Searches both /article/body and /article/floats-group.
    """
    tables = []
    seen_ids = set()
    
    # Search in body (inline tables)
    body = root.find("body")
    if body is not None:
        for table_wrap in body.xpath(".//table-wrap"):
            table_id = table_wrap.get("id", "")
            position = table_wrap.get("position", "")
            
            # Skip if duplicate
            if table_id and table_id in seen_ids:
                logger.debug(f"Duplicate table ID {table_id}, preferring body version")
                continue
            
            if table_id:
                seen_ids.add(table_id)
            
            # Extract caption parts
            label = ""
            caption_title = ""
            caption_text = ""
            
            label_elem = table_wrap.find("label")
            if label_elem is not None and label_elem.text:
                label = label_elem.text
            
            caption_elem = table_wrap.find("caption")
            if caption_elem is not None:
                cap_title = caption_elem.find("title")
                if cap_title is not None:
                    caption_title = get_text(cap_title)
                
                cap_paras = []
                for p in caption_elem.findall("p"):
                    p_text = get_text(p)
                    if p_text:
                        cap_paras.append(p_text)
                caption_text = " ".join(cap_paras)
            
            # Extract table content
            table_elem = table_wrap.find("table")
            content = ""
            if table_elem is not None:
                content = _table_to_markdown(table_elem)
            
            # Extract footnotes
            footnotes = _extract_table_footnotes(table_wrap)
            
            tables.append({
                "id": table_id or f"table-inline-{len(tables)+1}",
                "location": "inline",
                "label": label,
                "caption_title": caption_title,
                "caption_text": caption_text,
                "content": content,
                "footnotes": footnotes,
            })
    
    # Search in floats-group (floating tables)
    floats_group = root.find("floats-group")
    if floats_group is not None:
        for table_wrap in floats_group.xpath(".//table-wrap"):
            table_id = table_wrap.get("id", "")
            
            # Skip if already in body (prefer body version)
            if table_id and table_id in seen_ids:
                logger.debug(f"Table {table_id} conflict, using body version")
                continue
            
            if table_id:
                seen_ids.add(table_id)
            
            # Extract same as above
            label = ""
            caption_title = ""
            caption_text = ""
            
            label_elem = table_wrap.find("label")
            if label_elem is not None and label_elem.text:
                label = label_elem.text
            
            caption_elem = table_wrap.find("caption")
            if caption_elem is not None:
                cap_title = caption_elem.find("title")
                if cap_title is not None:
                    caption_title = get_text(cap_title)
                
                cap_paras = []
                for p in caption_elem.findall("p"):
                    p_text = get_text(p)
                    if p_text:
                        cap_paras.append(p_text)
                caption_text = " ".join(cap_paras)
            
            table_elem = table_wrap.find("table")
            content = ""
            if table_elem is not None:
                content = _table_to_markdown(table_elem)
            
            footnotes = _extract_table_footnotes(table_wrap)
            
            tables.append({
                "id": table_id or f"table-float-{len(tables)+1}",
                "location": "floating",
                "label": label,
                "caption_title": caption_title,
                "caption_text": caption_text,
                "content": content,
                "footnotes": footnotes,
            })
    
    return tables


def _extract_abstract(article_meta: ET.Element) -> str:
    """Extract abstract text."""
    abstract_parts = []
    
    for abstract in article_meta.findall("abstract"):
        for p in abstract.findall("p"):
            p_text = get_text(p)
            if p_text:
                abstract_parts.append(p_text)
        
        # Handle structured abstracts (sec elements)
        for sec in abstract.findall("sec"):
            sec_title = sec.find("title")
            if sec_title is not None:
                abstract_parts.append(get_text(sec_title) + ":")
            for p in sec.findall("p"):
                p_text = get_text(p)
                if p_text:
                    abstract_parts.append(p_text)
    
    return "\n\n".join(abstract_parts)

