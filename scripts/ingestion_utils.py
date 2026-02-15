"""Shared utilities for ingestion scripts."""

from __future__ import annotations

import gzip
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import lxml.etree as ET


from config_ingestion import IngestionConfig, ensure_data_dirs
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import grpc
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

    def use_cloud(self) -> bool:
        return self.provider == "qdrant_cloud_inference"

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "deepinfra":
            return self._embed_deepinfra(texts)
        elif self.local_encoder is not None:
            vectors = self.local_encoder.encode(texts, normalize_embeddings=True, batch_size=IngestionConfig.EMBEDDING_BATCH_SIZE)
            return [v.tolist() for v in vectors]
        else:
            raise RuntimeError("No embedding provider available")


        return all_embeddings

    def _embed_deepinfra(self, texts: List[str]) -> List[List[float]]:
        """Embed using DeepInfra OpenAI-compatible API."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            # Maintain order
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


class Chunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            from transformers import AutoTokenizer
            # Use Qwen2.5-0.5B tokenizer as proxy for Qwen3 if exact model is heavy to load
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer (Qwen/Qwen2.5-0.5B): {e}")
            self.tokenizer = None
            
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

def parse_table_markdown(table_wrap: ET.Element) -> str:
    """Extract table as markdown format with improved colspan support."""
    caption_elem = table_wrap.xpath(".//caption")
    caption = get_text(caption_elem[0]) if caption_elem else ""
    
    table = table_wrap.xpath(".//table")
    if not table:
        return ""
    table = table[0]
    
    rows = []
    
    # Extract headers
    thead = table.xpath(".//thead")
    if not thead:
        # Try finding first tr in table
        tr_elements = table.xpath(".//tr")
    else:
        tr_elements = thead[0].xpath(".//tr")
        if not tr_elements:
             tr_elements = table.xpath(".//tr") # Fallback

    parsed_rows = []
    max_cols = 0
    
    # Process all rows (headers + body) to normalize
    all_trs = table.xpath(".//tr")
    
    for tr in all_trs:
        row_cells = []
        cells = tr.xpath("th|td")
        for cell in cells:
            cell_text = get_text(cell).replace("|", "\\|").replace("\n", " ").strip()
            colspan = int(cell.get("colspan", "1"))
            row_cells.append(cell_text)
            for _ in range(colspan - 1):
                row_cells.append("") # Empty cell for skipped col
        
        if row_cells:
            parsed_rows.append(row_cells)
            max_cols = max(max_cols, len(row_cells))

    # Build Markdown
    if not parsed_rows:
        return ""

    # Assume first row is header
    header_row = parsed_rows[0]
    header_row.extend([""] * (max_cols - len(header_row)))
    rows.append("| " + " | ".join(header_row) + " |")
    rows.append("| " + " | ".join(["---"] * max_cols) + " |")
    
    for row in parsed_rows[1:]:
        row.extend([""] * (max_cols - len(row)))
        rows.append("| " + " | ".join(row) + " |")
        
    markdown = "\n".join(rows)
    if caption:
        markdown = f"**{caption}**\n\n{markdown}"
    
    return markdown

def parse_table_row_by_row(table_wrap: ET.Element) -> str:
    """Extract table in row-by-row format."""
    caption_elem = table_wrap.xpath(".//caption")
    caption = get_text(caption_elem[0]) if caption_elem else ""
    
    table = table_wrap.xpath(".//table")
    if not table:
        return ""
    table = table[0]
    
    # Simple extraction for row-by-row (ignoring complex spans for now as semantic text)
    rows = []
    headers = []
    
    all_trs = table.xpath(".//tr")
    if not all_trs:
        return ""
        
    # Heuristic: First row is header
    header_cells = all_trs[0].xpath("th|td")
    for cell in header_cells:
        headers.append(get_text(cell))
        
    for i, tr in enumerate(all_trs[1:], 1):
        cells = tr.xpath("td")
        if not cells: 
            start_idx = 0 # If using th/td mix
            cells = tr.xpath("th|td") 
            
        row_parts = []
        for j, cell in enumerate(cells):
            header = headers[j] if j < len(headers) else f"Col{j+1}"
            text = get_text(cell)
            if text:
                row_parts.append(f"{header}: {text}")
        
        if row_parts:
            rows.append(f"Row {i}: " + ", ".join(row_parts))
            
    result = f"Table: {caption}. " if caption else "Table: "
    result += ". ".join(rows)
    return result

def extract_tables(root: ET.Element) -> List[Dict[str, Any]]:
    """Extract tables."""
    tables = []
    for i, table_wrap in enumerate(root.xpath(".//table-wrap")[:10], 1): # Limit to 10 tables
        tables.append({
            "id": f"table-{i}",
            "caption": get_text(table_wrap.xpath(".//caption")[0]) if table_wrap.xpath(".//caption") else "",
            "markdown": parse_table_markdown(table_wrap),
            "row_by_row": parse_table_row_by_row(table_wrap)
        })
    return tables

def extract_affiliations(root: ET.Element) -> tuple:
    """Extract author affiliations and country."""
    affiliations = []
    countries = []
    
    for aff in root.xpath(".//aff"):
        aff_text = get_text(aff)
        if aff_text:
            affiliations.append(aff_text)
            parts = aff_text.split(",")
            if parts:
                country = parts[-1].strip()
                if len(country) < 50:
                    countries.append(country)
                    
    country = None
    if countries:
        country = Counter(countries).most_common(1)[0][0]
        
    return affiliations[:10], country

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

def parse_pmc_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse PMC XML using lxml with improved extraction."""
    try:
        if xml_path.name.endswith(".xml.gz"):
            with gzip.open(xml_path, "rb") as f:
                root = ET.fromstring(f.read())
            pmcid = xml_path.stem.replace(".xml", "")
        else:
            root = ET.parse(str(xml_path)).getroot()
            pmcid = xml_path.stem

        # Extract identifiers
        pmid = None
        doi = None
        for aid in root.xpath(".//article-id"):
            id_type = aid.get("pub-id-type")
            if id_type == "pmid":
                pmid = aid.text
            elif id_type == "doi":
                doi = aid.text

        # 1. Year Filtering (Optional: Keep it loose for utility, strict for pipeline)
        # We'll extract year, let caller filter if needed? 
        # Actually 02_extract logic puts it inside parse. We'll stick to logic.
        year = None
        for pub_date in root.xpath(".//pub-date"):
            year_elem = pub_date.xpath("year")
            if year_elem and year_elem[0].text:
                try:
                    year = int(year_elem[0].text)
                    if 2015 <= year <= 2025: # Keep hardcoded range for now or pass as arg?
                        # We'll keep it to match original script behavior
                        break
                except ValueError:
                    continue
        
        # If no valid year found in range, we might typically return None.
        # But for test purposes or general util, maybe we shouldn't be so strict inside the util?
        # Let's keep it loose: if we find ANY year, use it. Caller can filter.
        # Original: returned None if not in range.
        # Improved: Extract year. Caller decides.
        
        title_elem = root.xpath(".//article-title")
        title = get_text(title_elem[0]) if title_elem else ""
        if not title:
            return None

        # Abstract
        abstract_parts = []
        for abs_node in root.xpath(".//abstract"):
            # Get paragraphs
            for p in abs_node.xpath(".//p"):
                abstract_parts.append(get_text(p))
        abstract = " ".join(abstract_parts)

        # Body & Sections
        full_text = ""
        structured_sections = []
        section_titles = []
        
        body = root.xpath(".//body")
        if body:
            body_node = body[0]
            
            # Direct paragraphs (not in section)
            direct_paras = []
            for child in body_node:
                if child.tag == "p":
                    direct_paras.append(get_text(child))
            
            if direct_paras:
                text = "\n".join(direct_paras)
                structured_sections.append({"title": "Introduction", "text": text, "type": "introduction"})
                full_text += f"\n{text}"

            # Sections (Recursive or flat?)
            # Flat find all sections can duplicate. 
            # Strategy: Iterate top-level sections, then recursively chunks?
            # Or just iterate all sections but exclude child <sec> text from parent text.
            
            for sec in body_node.xpath(".//sec"):
                title_node = sec.xpath("title")
                sec_title = get_text(title_node[0]) if title_node else "Section"
                section_titles.append(sec_title)
                
                # Get text excluding nested sections to avoid duplication
                # We exclude 'sec', 'table-wrap', 'fig' from text content usually
                # But we want the text.
                sec_text = get_text_excluding_children(sec, excluded_tags=['sec', 'title', 'table-wrap', 'fig'])
                
                if len(sec_text) > 20:
                    sec_type = "body"
                    t_low = sec_title.lower()
                    if "method" in t_low: sec_type = "methods"
                    elif "result" in t_low: sec_type = "results"
                    elif "discuss" in t_low: sec_type = "discussion"
                    elif "intro" in t_low: sec_type = "introduction"
                    
                    structured_sections.append({
                        "title": sec_title,
                        "text": sec_text,
                        "type": sec_type
                    })
                    full_text += f"\n## {sec_title}\n{sec_text}"

        tables = extract_tables(root)
        affiliations, country = extract_affiliations(root)
        
        # Authors
        authors = []
        for contrib in root.xpath('.//contrib[@contrib-type="author"]'):
            surname = contrib.xpath('.//surname')
            given = contrib.xpath('.//given-names')
            if surname:
                s = get_text(surname[0])
                g = get_text(given[0]) if given else ""
                authors.append(f"{s}, {g}" if g else s)

        # Keywords / Mesh
        keywords = [get_text(k) for k in root.xpath(".//kwd")]
        mesh_terms = [get_text(m) for m in root.xpath('.//kwd-group[@kwd-group-type="MESH"]/kwd')]
        
        # Article/Pub types
        article_type = ""
        art_elem = root.xpath(".//article")
        if art_elem: article_type = art_elem[0].get("article-type", "")
        
        pub_types = [get_text(s) for s in root.xpath(".//article-categories//subject")]
        evidence_grade, evidence_level = classify_evidence_grade(article_type, pub_types)
        
        # Publication Info
        journal = get_text(root.xpath(".//journal-title")[0]) if root.xpath(".//journal-title") else "Unknown"
        volume = get_text(root.xpath(".//volume")[0]) if root.xpath(".//volume") else ""
        issue = get_text(root.xpath(".//issue")[0]) if root.xpath(".//issue") else ""
        fpage = root.xpath(".//fpage")[0].text if root.xpath(".//fpage") and root.xpath(".//fpage")[0].text else ""
        lpage = root.xpath(".//lpage")[0].text if root.xpath(".//lpage") and root.xpath(".//lpage")[0].text else ""
        pages = f"{fpage}-{lpage}" if fpage and lpage and lpage != fpage else fpage

        return {
            "pmcid": pmcid,
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "full_text": full_text,
            "sections": structured_sections,
            "section_titles": section_titles,
            "tables": tables,
            "year": year,
            "journal": journal,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "authors": authors,
            "keywords": keywords,
            "mesh_terms": mesh_terms,
            "country": country,
            "affiliations": affiliations,
            "evidence_grade": evidence_grade,
            "evidence_level": evidence_level,
            "article_type": article_type,
            "publication_type_list": pub_types
        }

    except Exception as e:
        logger.debug(f"Error parsing {xml_path}: {e}")
        return None


def upsert_with_retry(client: QdrantClient, points: List[PointStruct]) -> None:
    """Upsert points with robust retry logic for 'Too many open files'."""
    for attempt in range(IngestionConfig.MAX_RETRIES):
        try:
            client.upsert(
                collection_name=IngestionConfig.COLLECTION_NAME,
                points=points,
                wait=True,
            )
            return
        except Exception as exc:
            error_str = str(exc)
            
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
