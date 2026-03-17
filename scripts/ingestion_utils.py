"""Shared utilities for ingestion scripts."""

from __future__ import annotations

import gzip
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, deque
from datetime import datetime

import lxml.etree as ET


from config_ingestion import IngestionConfig, ensure_data_dirs
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import grpc
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Track hard parse failures (malformed/unreadable XML) across a run.
_PMC_PARSE_FAILURE_COUNT = 0
_PMC_PARSE_FAILURE_LOCK = threading.Lock()


def reset_pmc_xml_parse_failure_count() -> None:
    """Reset the global PMC XML parse failure counter."""
    global _PMC_PARSE_FAILURE_COUNT
    with _PMC_PARSE_FAILURE_LOCK:
        _PMC_PARSE_FAILURE_COUNT = 0


def get_pmc_xml_parse_failure_count() -> int:
    """Return the number of PMC XML files skipped due to parse exceptions."""
    with _PMC_PARSE_FAILURE_LOCK:
        return _PMC_PARSE_FAILURE_COUNT


def _increment_pmc_xml_parse_failure_count() -> None:
    global _PMC_PARSE_FAILURE_COUNT
    with _PMC_PARSE_FAILURE_LOCK:
        _PMC_PARSE_FAILURE_COUNT += 1

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
    """Support for DeepInfra embeddings with concurrent parallel requests."""

    def __init__(self) -> None:
        self.provider = IngestionConfig.EMBEDDING_PROVIDER.lower().strip()
        self.model = IngestionConfig.EMBEDDING_MODEL
        self.openai_client = None
        self._rate_limiter: Optional[_RequestRateLimiter] = None
        self._request_log_lock = threading.Lock()
        self._request_timestamps: deque[float] = deque()
        self._rate_log_enabled = os.getenv("EMBEDDING_RATE_LOG_ENABLED", "true").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._rate_log_window_seconds = max(
            1.0, float(os.getenv("EMBEDDING_RATE_LOG_WINDOW_SECONDS", "10"))
        )
        self._rate_log_every_seconds = max(
            1.0, float(os.getenv("EMBEDDING_RATE_LOG_EVERY_SECONDS", "10"))
        )
        self._next_rate_log_at = time.monotonic() + self._rate_log_every_seconds
        # Number of concurrent embedding request threads
        self._embed_concurrency = max(
            1, int(os.getenv("EMBEDDING_CONCURRENCY", "8"))
        )
        # Max retries per individual embed request
        self._embed_max_retries = max(
            1, int(os.getenv("EMBEDDING_REQUEST_MAX_RETRIES", "6"))
        )

        if self.provider != "deepinfra":
            raise ValueError(
                f"Unsupported embedding provider '{self.provider}'. "
                "Only 'deepinfra' is supported."
            )

        from openai import OpenAI

        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY not set - required for deepinfra embedding provider")
        deepinfra_base_url = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
        self.openai_client = OpenAI(
            api_key=api_key,
            base_url=deepinfra_base_url,
            # Allow many concurrent connections from threadpool
            max_retries=0,  # We handle retries ourselves with better backoff
            timeout=120.0,
        )
        max_requests_per_sec = int(os.getenv("EMBEDDING_MAX_REQUESTS_PER_SEC", "0") or "0")
        if max_requests_per_sec > 0:
            self._rate_limiter = _RequestRateLimiter(max_requests_per_sec=max_requests_per_sec)
            logger.info(
                "Embedding request limiter enabled: max %s req/sec",
                max_requests_per_sec,
            )
        logger.info(
            "✅ DeepInfra embedding provider initialized (model: %s, concurrency: %s)",
            self.model,
            self._embed_concurrency,
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a flat list of texts. Sub-batches serially."""
        return self._embed_deepinfra(texts)

    def embed_batches_concurrent(
        self,
        text_batches: List[List[str]],
    ) -> List[List[List[float]]]:
        """Embed multiple pre-split batches concurrently.

        Args:
            text_batches: List of text lists. Each inner list becomes one
                          sub-batched embedding call.

        Returns:
            List of embedding lists, one per input text_batch, in the same order.
        """
        if not text_batches:
            return []
        if len(text_batches) == 1:
            return [self._embed_deepinfra(text_batches[0])]

        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

        results: List[Optional[List[List[float]]]] = [None] * len(text_batches)
        errors: List[Optional[Exception]] = [None] * len(text_batches)

        with ThreadPoolExecutor(
            max_workers=min(self._embed_concurrency, len(text_batches)),
            thread_name_prefix="embed-concurrent",
        ) as pool:
            future_to_idx = {
                pool.submit(self._embed_deepinfra, batch): idx
                for idx, batch in enumerate(text_batches)
            }
            for future in _as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    errors[idx] = exc

        first_error = next((e for e in errors if e is not None), None)
        if first_error is not None:
            raise first_error

        return results  # type: ignore[return-value]

    def _embed_deepinfra(self, texts: List[str]) -> List[List[float]]:
        """Embed using DeepInfra OpenAI-compatible API with sub-batching."""
        batch_size = IngestionConfig.EMBEDDING_BATCH_SIZE

        if len(texts) <= batch_size:
            return self._embed_deepinfra_single(texts)

        # Sub-batch large requests
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            sub_batch = texts[i:i + batch_size]
            embeddings = self._embed_deepinfra_single(sub_batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _embed_deepinfra_single(self, texts: List[str]) -> List[List[float]]:
        """Send a single embedding request to DeepInfra with retry/backoff."""
        import random as _random

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._embed_max_retries + 1):
            try:
                self._before_embedding_request()
                response = self.openai_client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float",
                )
                return [data.embedding for data in response.data]
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                is_rate_limit = (
                    "429" in exc_str
                    or "rate limit" in exc_str.lower()
                    or "too many requests" in exc_str.lower()
                )
                is_transient = (
                    "500" in exc_str
                    or "502" in exc_str
                    or "503" in exc_str
                    or "504" in exc_str
                    or "connection" in exc_str.lower()
                    or "timeout" in exc_str.lower()
                    or "reset" in exc_str.lower()
                )

                if attempt >= self._embed_max_retries:
                    logger.error(
                        "DeepInfra embedding failed after %s attempts: %s",
                        attempt, exc_str[:300],
                    )
                    raise

                if is_rate_limit:
                    # Rate-limited: longer wait, then retry
                    base_wait = min(30.0, 2.0 * (2 ** (attempt - 1)))
                    jitter = _random.uniform(0.5, 1.5)
                    wait = base_wait * jitter
                    logger.warning(
                        "DeepInfra rate-limited (attempt %s/%s), pausing %.1fs: %s",
                        attempt, self._embed_max_retries, wait, exc_str[:120],
                    )
                    time.sleep(wait)
                elif is_transient:
                    # Transient 5xx: shorter backoff
                    base_wait = min(8.0, 1.0 * (2 ** (attempt - 1)))
                    jitter = _random.uniform(0.25, 1.25)
                    wait = base_wait * jitter
                    logger.warning(
                        "DeepInfra transient error (attempt %s/%s), retrying in %.1fs: %s",
                        attempt, self._embed_max_retries, wait, exc_str[:120],
                    )
                    time.sleep(wait)
                else:
                    # Non-retriable error
                    logger.error("DeepInfra non-retriable error: %s", exc_str[:300])
                    raise

        # Should be unreachable
        if last_exc is not None:
            raise last_exc
        return []

    def _before_embedding_request(self) -> None:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        self._record_request_for_telemetry()

    def _record_request_for_telemetry(self) -> None:
        if not self._rate_log_enabled:
            return

        now = time.monotonic()
        with self._request_log_lock:
            self._request_timestamps.append(now)
            cutoff = now - self._rate_log_window_seconds
            while self._request_timestamps and self._request_timestamps[0] < cutoff:
                self._request_timestamps.popleft()

            if now < self._next_rate_log_at:
                return

            window_count = len(self._request_timestamps)
            rate = window_count / self._rate_log_window_seconds
            logger.info(
                "Embedding request rate: %.2f req/sec over last %.0fs",
                rate,
                self._rate_log_window_seconds,
            )
            self._next_rate_log_at = now + self._rate_log_every_seconds


class _RequestRateLimiter:
    """Thread-safe token-bucket rate limiter for outbound embedding API calls."""

    def __init__(self, max_requests_per_sec: int):
        self.max_requests_per_sec = max(1, int(max_requests_per_sec))
        self._lock = threading.Lock()
        self._recent_requests: deque[float] = deque()

    def acquire(self) -> None:
        """Block until a request token is available."""
        while True:
            now = time.monotonic()
            with self._lock:
                cutoff = now - 1.0
                while self._recent_requests and self._recent_requests[0] <= cutoff:
                    self._recent_requests.popleft()

                if len(self._recent_requests) < self.max_requests_per_sec:
                    self._recent_requests.append(now)
                    return

                earliest = self._recent_requests[0]
                sleep_seconds = max(0.001, (earliest + 1.0) - now)

            time.sleep(min(sleep_seconds, 0.02))


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

EVIDENCE_GRADE_LABELS = {
    "A": "Highest evidence",
    "B": "High evidence",
    "C": "Moderate evidence",
    "D": "Lower evidence",
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
    WORDS_PER_TOKEN = 0.75

    def __init__(self, chunk_size: int = None, overlap: int = None):
        # Read defaults from IngestionConfig to ensure consistency
        # This prevents hardcoded values from getting out of sync with .env
        self.chunk_size = chunk_size if chunk_size is not None else IngestionConfig.CHUNK_SIZE_TOKENS
        self.overlap = overlap if overlap is not None else IngestionConfig.CHUNK_OVERLAP_TOKENS
        self.chunk_size_words = max(1, int(self.chunk_size * self.WORDS_PER_TOKEN))
        self.overlap_words = max(0, int(self.overlap * self.WORDS_PER_TOKEN))
        if self.overlap_words >= self.chunk_size_words:
            self.overlap_words = max(0, self.chunk_size_words - 1)
            
    def count_tokens(self, text: str) -> int:
        """Estimate tokens from word count (1 token ≈ 0.75 words)."""
        if not text:
            return 0

        return int(len(text.split()) / self.WORDS_PER_TOKEN)
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        words = text.split()
        chunks = []
        stride = max(1, self.chunk_size_words - self.overlap_words)

        for i in range(0, len(words), stride):
            chunk_words = words[i:i + self.chunk_size_words]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "token_count": int(len(chunk_words) / self.WORDS_PER_TOKEN)
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
            try:
                colspan = int(cell.get("colspan", "1"))
            except (ValueError, TypeError):
                colspan = 1
            try:
                rowspan = int(cell.get("rowspan", "1"))
            except (ValueError, TypeError):
                rowspan = 1
            
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
    """Backward-compatible wrapper returning (grade, normalized_level_1_4)."""
    result = classify_evidence_metadata(article_type=article_type, pub_types=pub_types, abstract=None)
    return result["grade"], result["level_1_4"]

def _base_name_from_archive_name(name: str, lowered: str, fallback_stem: str) -> str:
    if lowered.endswith(".xml.gz"):
        return name[: -len(".xml.gz")]
    if lowered.endswith(".nxml.gz"):
        return name[: -len(".nxml.gz")]
    if lowered.endswith(".xml"):
        return name[: -len(".xml")]
    if lowered.endswith(".nxml"):
        return name[: -len(".nxml")]
    return fallback_stem


def _parse_pmc_xml_root(
    root: ET.Element,
    source_label: str,
    pmcid_from_file: str,
    require_pmid: bool = True,
    require_open_access: bool = True,
    require_commercial_license: bool = False,
) -> Optional[Dict[str, Any]]:
    # Get article and article-meta elements
    # Handle both cases: root IS the article, or article is nested
    if root.tag == "article" or root.tag.endswith("}article"):
        article_elem = root
    else:
        article_elem = root.find(".//article")
    if article_elem is None:
        logger.debug("No article element found in %s", source_label)
        return None

    article_meta = article_elem.find("front/article-meta")
    if article_meta is None:
        logger.debug("No article-meta found in %s", source_label)
        return None

    # === 1. Document Metadata (PRD Section 1.1) ===
    article_type = _extract_article_type(root)
    language = _extract_language(root)
    identifiers = _extract_identifiers(article_meta)

    # Hard Fail: PMID must exist
    if require_pmid and not identifiers.get("pmid"):
        logger.debug("Skipping %s: No PMID found", source_label)
        return None

    # Use PMCID from filename/url if not in XML
    if not identifiers.get("pmcid"):
        identifiers["pmcid"] = pmcid_from_file

    # === 2. Publication Metadata (PRD Section 1.2) ===
    pub_date, year = _extract_publication_date(article_meta)
    journal_info = _extract_journal_info(root)
    country_code, country_name = _extract_country(root)

    # === 3. Content Structure (PRD Section 1.3) ===
    article_title = _extract_article_title(article_meta)
    if not article_title:
        logger.debug("Skipping %s: No article title", source_label)
        return None

    # Check for body and determine has_full_text
    body_elem = article_elem.find("body")
    has_full_text = False
    body_text = ""
    if body_elem is not None:
        body_text = get_text(body_elem).strip()
        has_full_text = len(body_text) > 100

    # Open Access and Commercial License Check (PRD Section 1.3)
    is_open_access = _extract_open_access(article_meta)
    license_type = _extract_license_type(article_meta)

    if require_open_access and not is_open_access:
        logger.info("Skipping %s: Not open access", source_label)
        return None

    if require_commercial_license and not _is_commercial_license(license_type):
        logger.info("Skipping %s: Non-commercial license (%s)", source_label, license_type)
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
    abstract_text = _extract_abstract(article_meta)
    evidence = classify_evidence_metadata(
        article_type=article_type,
        pub_types=pub_types,
        abstract=abstract_text,
    )
    evidence_grade = evidence["grade"]
    evidence_level = evidence["level_1_4"]
    evidence_term = evidence["matched_term"]
    evidence_source = evidence["matched_from"]

    # === Build Output Structure per PRD Section 3 ===
    output = {
        # Document ID for indexing
        "document_id": (
            f"pmid_{identifiers['pmid']}"
            if identifiers.get("pmid")
            else f"pmcid_{identifiers['pmcid']}"
        ),
        "metadata": {
            "article_type": article_type,
            "language": language,
            "identifiers": {
                "pmid": identifiers.get("pmid"),
                "doi": identifiers.get("doi"),
                "pmcid": identifiers.get("pmcid"),
                "publisher_id": identifiers.get("publisher_id"),
                "nihms_id": identifiers.get("nihms_id"),
            },
            "publication": {
                "date": pub_date,
                "year": year,
                "journal": {
                    "title": journal_info["title"],
                    "abbreviation": journal_info["abbreviation"],
                    "publisher": journal_info["publisher"],
                    "issn": journal_info["issn_electronic"],
                    "nlm_unique_id": journal_info["nlm_unique_id"],
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
            },
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
        "nihms_id": identifiers.get("nihms_id"),
        "title": article_title,
        "abstract": abstract_text,
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
            }
            for t in tables
        ],
        "year": year,
        "journal": journal_info["title"],
        "keywords": keywords,
        "mesh_terms": mesh_terms,
        "country": country_code or country_name,
        "evidence_grade": evidence_grade,
        "evidence_level": evidence_level,
        "evidence_term": evidence_term,
        "evidence_source": evidence_source,
        "article_type": article_type,
        "publication_type_list": pub_types,
        "is_open_access": is_open_access,
        "has_full_text": has_full_text,
        "license": license_type,
    }
    return output


def parse_pmc_xml(
    xml_path: Path,
    require_pmid: bool = True,
    require_open_access: bool = True,
    require_commercial_license: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Parse PMC XML using lxml with PRD v1.0 compliant extraction.

    Args:
        xml_path: Path to the XML file
        require_pmid: If True, skip articles without PMID (PRD Hard Fail rule)
        require_open_access: If True, skip non-open-access articles
        require_commercial_license: If True, skip non-commercial licenses (CC-BY-NC, etc.)

    Returns:
        Article dict following PRD JSON structure, or None if invalid/skipped.
    """
    try:
        xml_path = Path(xml_path)  # Accept both str and Path
        file_name = xml_path.name
        lowered_name = file_name.lower()
        pmcid_from_file = _base_name_from_archive_name(file_name, lowered_name, xml_path.stem)

        # Parse XML
        if lowered_name.endswith(".gz"):
            with gzip.open(xml_path, "rb") as f:
                root = ET.fromstring(f.read())
        else:
            root = ET.parse(str(xml_path)).getroot()

        return _parse_pmc_xml_root(
            root=root,
            source_label=str(xml_path),
            pmcid_from_file=pmcid_from_file,
            require_pmid=require_pmid,
            require_open_access=require_open_access,
            require_commercial_license=require_commercial_license,
        )

    except Exception as e:
        _increment_pmc_xml_parse_failure_count()
        logger.warning("Error parsing %s: %s", xml_path, e)
        logger.debug("PMC XML parse traceback for %s", xml_path, exc_info=True)
        return None


def parse_pmc_xml_bytes(
    xml_bytes: bytes,
    source_name: str = "s3_object.xml",
    require_pmid: bool = True,
    require_open_access: bool = True,
    require_commercial_license: bool = False,
) -> Optional[Dict[str, Any]]:
    """Parse PMC XML from in-memory bytes with the same behavior as parse_pmc_xml()."""
    try:
        lowered_name = source_name.lower()
        pmcid_from_file = _base_name_from_archive_name(source_name, lowered_name, Path(source_name).stem)

        payload = xml_bytes
        if lowered_name.endswith(".gz") or xml_bytes.startswith(b"\x1f\x8b"):
            payload = gzip.decompress(xml_bytes)

        root = ET.fromstring(payload)
        return _parse_pmc_xml_root(
            root=root,
            source_label=source_name,
            pmcid_from_file=pmcid_from_file,
            require_pmid=require_pmid,
            require_open_access=require_open_access,
            require_commercial_license=require_commercial_license,
        )
    except Exception as e:
        _increment_pmc_xml_parse_failure_count()
        logger.warning("Error parsing %s: %s", source_name, e)
        logger.debug("PMC XML parse traceback for %s", source_name, exc_info=True)
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


def validate_qdrant_collection_schema(
    client: QdrantClient,
    collection_name: Optional[str] = None,
    require_sparse: Optional[bool] = None,
) -> None:
    """
    Validate collection schema before ingestion writes any points.

    Expected schema:
    - Named dense vector key: "dense"
    - Optional sparse vector key: "sparse" when BM25 sparse is enabled
    """
    target_collection = collection_name or IngestionConfig.COLLECTION_NAME
    sparse_required = (
        require_sparse
        if require_sparse is not None
        else (IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25")
    )

    info = client.get_collection(target_collection)
    params = info.config.params
    vectors_config = params.vectors

    if not isinstance(vectors_config, dict) or "dense" not in vectors_config:
        raise RuntimeError(
            "Qdrant schema mismatch: expected named dense vector key 'dense'. "
            "Run scripts/05_setup_qdrant.py --recreate before ingestion."
        )

    dense_cfg = vectors_config["dense"]
    expected_size = IngestionConfig.get_vector_size()
    if dense_cfg.size != expected_size:
        raise RuntimeError(
            f"Qdrant dense vector size mismatch: expected {expected_size}, got {dense_cfg.size}."
        )

    sparse_vectors_config = params.sparse_vectors or {}
    if sparse_required and (not isinstance(sparse_vectors_config, dict) or "sparse" not in sparse_vectors_config):
        raise RuntimeError(
            "Qdrant schema mismatch: expected sparse vector key 'sparse' for BM25 hybrid ingestion."
        )

    logger.info(
        "Qdrant schema preflight passed (dense=%s, sparse=%s)",
        "present",
        "present" if sparse_required else "not-required",
    )

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
        "nihms_id": None,
    }
    
    for aid in article_meta.xpath(".//article-id"):
        id_type = (aid.get("pub-id-type") or "").strip().lower().replace("_", "-")
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
        elif id_type in {"nihms", "nihms-id", "nihmsid", "manuscript-id", "manuscript"}:
            identifiers["nihms_id"] = text
        elif identifiers.get("nihms_id") is None and text.upper().startswith("NIHMS"):
            identifiers["nihms_id"] = text
    
    return identifiers


def _extract_article_type(root: ET.Element) -> str:
    """Extract article type per PRD Section 1.1."""
    # Handle root being the article element itself
    if root.tag == "article" or root.tag.endswith("}article"):
        art_type = root.get("article-type", "")
        if art_type:
            return art_type.lower()
    else:
        article_elem = root.xpath(".//article")
        if article_elem:
            art_type = article_elem[0].get("article-type", "")
            if art_type:
                return art_type.lower()
    return "other"


def _extract_language(root: ET.Element) -> str:
    """Extract language per PRD Section 1.1."""
    # Handle root being the article element itself
    if root.tag == "article" or root.tag.endswith("}article"):
        elem = root
    else:
        article_elems = root.xpath(".//article")
        elem = article_elems[0] if article_elems else None
    if elem is not None:
        lang = elem.get("{http://www.w3.org/XML/1998/namespace}lang")
        if lang:
            return lang.lower()
        lang = elem.get("xml:lang")
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
        "nlm_unique_id": "",
    }
    
    if journal_meta is None:
        return info
    
    # Journal title
    title_group = journal_meta.find(".//journal-title-group")
    if title_group is not None:
        title_elem = title_group.find("journal-title")
        if title_elem is not None and title_elem.text:
            info["title"] = title_elem.text
    
    # NLM abbreviation and NLM Unique ID
    for jid in journal_meta.findall("journal-id"):
        if jid.get("journal-id-type") == "nlm-ta":
            info["abbreviation"] = (jid.text or "").strip()
        if jid.get("journal-id-type") == "nlm-id":
            info["nlm_unique_id"] = (jid.text or "").strip()
    
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



def _extract_license_type(article_meta: ET.Element) -> str:
    """
    Extract and normalize the license type from PMC XML.
    Returns normalized strings like 'cc-by', 'cc-by-nc', 'cc0', etc.
    """
    permissions = article_meta.find("permissions")
    if permissions is None:
        return "unknown"
    
    for license_elem in permissions.findall("license"):
        # 1. Check license-type attribute
        license_type = license_elem.get("license-type", "").lower()
        if "cc0" in license_type:
            return "cc0"
        if "cc-by-nc" in license_type:
            return "cc-by-nc"
        if "cc-by-sa" in license_type:
            return "cc-by-sa"
        if "cc-by-nd" in license_type:
            return "cc-by-nd"
        if "cc-by" in license_type:
            return "cc-by"
        if "open-access" in license_type:
            # Generic open-access tag — treat as CC-BY unless text reveals otherwise
            return "cc-by"

        # 2. Check href/xlink:href attribute
        # Namespaces can be tricky, check plain 'href' and with common prefixes
        href = ""
        for attr_key, attr_val in license_elem.attrib.items():
            if "href" in attr_key.lower():
                href = attr_val.lower()
                break
        
        if href:
            if "cc0" in href: return "cc0"
            if "by-nc" in href: return "cc-by-nc"
            if "by-sa" in href: return "cc-by-sa"
            if "by-nd" in href: return "cc-by-nd"
            if "by" in href: return "cc-by"

        # 3. Fallback: Check license text content
        license_text = (license_elem.text or "").lower()
        for elem in license_elem.iter():
            license_text += (elem.text or "").lower() + (elem.tail or "").lower()
        
        if "cc0" in license_text: return "cc0"
        if "by-nc" in license_text: return "cc-by-nc"
        if "by-sa" in license_text: return "cc-by-sa"
        if "by-nd" in license_text: return "cc-by-nd"
        if "cc by" in license_text or "cc-by" in license_text: return "cc-by"
        if "public domain" in license_text: return "cc0"

    return "unknown"


def _is_commercial_license(license_type: str) -> bool:
    """Return True if the normalized license type allows commercial use.
    
    Uses a denylist approach: only explicitly NC (non-commercial) licenses
    are rejected. Unknown/unparseable licenses are trusted because this
    function is only called for files from the PMC OA Commercial Use Subset,
    where PMC already curates articles to be commercially licensed.
    """
    # Explicitly non-commercial → reject
    if "nc" in (license_type or ""):
        return False
    
    # Unknown/unparseable → trust the PMC OA source curation
    # (pmc_oa files are guaranteed commercially usable by PMC)
    return True


def _extract_open_access(article_meta: ET.Element) -> bool:
    """
    Extract open access status per PRD Section 1.3.
    Returns True if article has open-access or CC license.
    """
    permissions = article_meta.find("permissions")
    if permissions is None:
        return False
    
    # Check for <free-to-read/> tag (often used for gold OA/author manuscripts)
    if permissions.find("free-to-read") is not None:
        return True

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


# ============================================================================
# Shared Ingestion Utilities - Consolidated from duplicate implementations
# across ingestion scripts (06, 07, 21, 08_monthly_update, 12, etc.)
# ============================================================================

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Global singleton for chunker instance
_CHUNKER_INSTANCE: Optional[Any] = None
_CHUNKER_LOCK = threading.Lock()


def get_chunker(chunker_class: Optional[type] = None, chunk_size: Optional[int] = None, 
                overlap: Optional[int] = None) -> Any:
    """
    Get or initialize the shared Chunker instance (singleton pattern).
    
    This function provides a centralized way to get a chunker instance across
    all ingestion scripts, eliminating duplicate _get_chunker() implementations.
    
    Args:
        chunker_class: Optional chunker class to use (defaults to Chunker or SemanticChunker)
        chunk_size: Optional chunk size override (defaults to IngestionConfig.CHUNK_SIZE_TOKENS)
        overlap: Optional overlap override (defaults to IngestionConfig.CHUNK_OVERLAP_TOKENS)
        
    Returns:
        Chunker instance (singleton)
    """
    global _CHUNKER_INSTANCE
    
    if _CHUNKER_INSTANCE is not None:
        return _CHUNKER_INSTANCE
    
    with _CHUNKER_LOCK:
        if _CHUNKER_INSTANCE is not None:
            return _CHUNKER_INSTANCE
        
        # Determine which chunker class to use
        if chunker_class is None:
            # Try to use SemanticChunker if available, fallback to base Chunker
            try:
                from ingestion_utils_enhanced import SemanticChunker
                chunker_class = SemanticChunker
                logger.info("Using SemanticChunker for improved chunking")
            except ImportError:
                chunker_class = Chunker
                logger.info("Using base Chunker for chunking")
        
        # Get config values if not provided
        if chunk_size is None or overlap is None:
            try:
                from config_ingestion import IngestionConfig
                chunk_size = chunk_size or IngestionConfig.CHUNK_SIZE_TOKENS
                overlap = overlap or IngestionConfig.CHUNK_OVERLAP_TOKENS
            except ImportError:
                chunk_size = chunk_size or 2048
                overlap = overlap or 256
        
        _CHUNKER_INSTANCE = chunker_class(chunk_size=chunk_size, overlap=overlap)
        return _CHUNKER_INSTANCE


def reset_chunker() -> None:
    """Reset the chunker singleton (useful for testing)."""
    global _CHUNKER_INSTANCE
    with _CHUNKER_LOCK:
        _CHUNKER_INSTANCE = None


# Global checkpoint lock for thread-safe file operations
_CHECKPOINT_LOCK = threading.Lock()


def load_checkpoint(checkpoint_file: Path) -> Set[str]:
    """
    Load checkpoint of already processed IDs from file.
    
    This function provides a centralized way to load checkpoints across
    all ingestion scripts, eliminating duplicate load_checkpoint() implementations.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        
    Returns:
        Set of processed IDs (strings)
    """
    if checkpoint_file.exists():
        try:
            content = checkpoint_file.read_text(encoding="utf-8")
            return {line.strip() for line in content.splitlines() if line.strip()}
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {checkpoint_file}: {e}")
            return set()
    return set()


def append_checkpoint(checkpoint_file: Path, ids: Iterable[str]) -> None:
    """
    Append IDs to checkpoint file in a thread-safe manner.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        ids: Iterable of IDs to append
    """
    with _CHECKPOINT_LOCK:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_file.open("a", encoding="utf-8") as f:
            for id_val in ids:
                f.write(f"{id_val}\n")


def _build_evidence_result(grade: str, matched_term: Optional[str], matched_from: str) -> Dict[str, Any]:
    """Build normalized evidence metadata shared by ingestion and serving paths."""
    return {
        "grade": grade,
        "level_1_4": get_evidence_level(grade),
        "matched_term": matched_term,
        "matched_from": matched_from,
    }


def classify_evidence_metadata(article_type: str, pub_types: Optional[List[str]] = None,
                               abstract: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify evidence with normalized metadata for storage and frontend display.

    Returns:
        Dict with keys:
            - grade: Evidence grade ('A'|'B'|'C'|'D')
            - level_1_4: Normalized level (1=A, 2=B, 3=C, 4=D)
            - matched_term: Source term that triggered classification (or None)
            - matched_from: 'article_type'|'publication_type'|'abstract'|'fallback'
    """
    article_type_lower = (article_type or "").lower().replace("_", "-").replace(" ", "-")
    pub_types_lower = [p.lower() for p in (pub_types or [])]
    abstract_lower = (abstract or "").lower()

    # Priority 1: Check EVIDENCE_HIERARCHY for article_type patterns
    for pattern, (grade, _) in EVIDENCE_HIERARCHY.items():
        pattern_normalized = pattern.lower().replace("_", "-")
        if pattern_normalized in article_type_lower:
            return _build_evidence_result(grade, pattern, "article_type")

    # Priority 2: Check publication types
    for pub_type in pub_types_lower:
        for pattern, (grade, _) in EVIDENCE_HIERARCHY.items():
            pattern_normalized = pattern.lower().replace("_", "-")
            if pattern_normalized in pub_type:
                return _build_evidence_result(grade, pattern, "publication_type")

    # Priority 3: Content-based detection from abstract
    if abstract_lower:
        if any(x in abstract_lower for x in ["systematic review", "meta-analysis", "meta analysis"]):
            return _build_evidence_result("A", "systematic review", "abstract")
        if "guideline" in abstract_lower:
            return _build_evidence_result("A", "guideline", "abstract")

        if any(x in abstract_lower for x in ["randomized controlled trial", "randomised controlled trial"]):
            return _build_evidence_result("B", "randomized controlled trial", "abstract")
        if "rct" in abstract_lower:
            return _build_evidence_result("B", "rct", "abstract")
        if "randomized trial" in abstract_lower and "phase" in abstract_lower:
            return _build_evidence_result("B", "randomized trial + phase", "abstract")

        if any(x in abstract_lower for x in ["cohort study", "prospective study", "observational study"]):
            return _build_evidence_result("C", "cohort study", "abstract")

    # Priority 4: Article type fallbacks
    if article_type_lower in ("systematic-review", "meta-analysis", "guideline", "practice-guideline"):
        return _build_evidence_result("A", article_type_lower or None, "fallback")
    if article_type_lower in ("clinical-trial", "randomized-controlled-trial", "review-article", "review"):
        return _build_evidence_result("B", article_type_lower or None, "fallback")
    if article_type_lower in ("research-article", "journal-article", "article"):
        return _build_evidence_result("C", article_type_lower or None, "fallback")
    if article_type_lower in ("case-report", "case-series", "editorial", "letter", "news"):
        return _build_evidence_result("D", article_type_lower or None, "fallback")

    return _build_evidence_result("C", article_type_lower or None, "fallback")


def detect_evidence_grade(article_type: str, pub_types: Optional[List[str]] = None,
                          abstract: Optional[str] = None) -> str:
    """Backward-compatible wrapper that returns only evidence grade."""
    return classify_evidence_metadata(article_type=article_type, pub_types=pub_types, abstract=abstract)["grade"]


def get_evidence_level(grade: str) -> int:
    """
    Convert evidence grade to numeric level.
    
    Args:
        grade: Evidence grade ('A', 'B', 'C', or 'D')
        
    Returns:
        Numeric level (1=A, 2=B, 3=C, 4=D)
    """
    return {"A": 1, "B": 2, "C": 3, "D": 4}.get(grade.upper(), 3)


def get_evidence_hierarchy_levels() -> Dict[str, Any]:
    """
    Build frontend-facing evidence hierarchy payload from canonical term mapping.

    Returns:
        {
            "levels": [
                {"grade": "A", "level": 1, "label": "...", "terms": [...]},
                ...
            ]
        }
    """
    terms_by_grade: Dict[str, List[str]] = {"A": [], "B": [], "C": [], "D": []}
    for term, (grade, _) in EVIDENCE_HIERARCHY.items():
        if grade in terms_by_grade and term not in terms_by_grade[grade]:
            terms_by_grade[grade].append(term)

    levels = []
    for grade in ("A", "B", "C", "D"):
        levels.append({
            "grade": grade,
            "level": get_evidence_level(grade),
            "label": EVIDENCE_GRADE_LABELS[grade],
            "terms": terms_by_grade[grade],
        })

    return {"levels": levels}


# ============================================================================
# Government Affiliation Extraction
# ============================================================================

# Government affiliation patterns for detecting US government-authored articles
GOV_AFFILIATION_PATTERNS = [
    # NIH and institutes
    "national institutes of health",
    "national institute of",
    "nih,",
    "(nih)",
    " nih ",
    # CDC
    "centers for disease control",
    "cdc,",
    "(cdc)",
    " cdc ",
    # FDA
    "food and drug administration",
    "fda,",
    "(fda)",
    " fda ",
    # Other federal
    "veterans affairs",
    "va medical",
    "va hospital",
    "department of health and human services",
    "hhs,",
    "walter reed",
    "uniformed services university",
    "national library of medicine",
    "national cancer institute",
    "national heart, lung, and blood",
    "national institute of allergy",
    "national institute of mental health",
    "national eye institute",
    "national institute of diabetes",
    "national institute on aging",
    "national institute of child health",
    "national institute of neurological",
    "national human genome research",
    "agency for healthcare research and quality",
    "ahrq",
    # Location-based
    "bethesda, md",
    "bethesda, maryland",
    "bethesda md",
    "atlanta, ga",
    "atlanta, georgia",
    "silver spring, md",
    "silver spring, maryland",
    "rockville, md",
    "rockville, maryland",
]


def extract_agency_name_from_pattern(pattern: str) -> Optional[str]:
    """Extract standardized agency name from matched pattern."""
    pattern_lower = pattern.lower()
    
    # Map patterns to standardized agency names
    agency_map = {
        "nih": "NIH",
        "national institutes of health": "NIH",
        "national institute of": "NIH",
        "national cancer institute": "NCI",
        "national heart": "NHLBI",
        "national institute of allergy": "NIAID",
        "national institute of mental health": "NIMH",
        "national eye institute": "NEI",
        "national institute of diabetes": "NIDDK",
        "national institute on aging": "NIA",
        "national institute of child health": "NICHD",
        "national institute of neurological": "NINDS",
        "national human genome research": "NHGRI",
        "national library of medicine": "NLM",
        "cdc": "CDC",
        "centers for disease control": "CDC",
        "fda": "FDA",
        "food and drug administration": "FDA",
        "veterans affairs": "VA",
        "va medical": "VA",
        "va hospital": "VA",
        "uniformed services university": "USUHS",
        "walter reed": "Walter Reed",
        "ahrq": "AHRQ",
        "agency for healthcare research": "AHRQ",
    }
    
    for key, agency in agency_map.items():
        if key in pattern_lower:
            return agency
    
    return None


def extract_gov_affiliations_from_pubmed_xml(article_elem: ET.Element) -> tuple[bool, list[str]]:
    """
    Extract government affiliations from PubMed XML article element.
    
    This function checks Affiliation elements, AffiliationInfo elements (newer format),
    and GrantList for NIH/government funding indicators.
    
    Args:
        article_elem: The PubmedArticle XML element from ElementTree parsing
        
    Returns:
        Tuple of (is_gov_affiliated, list_of_matched_agencies)
    """
    matched_agencies: set[str] = set()
    
    # Check all affiliation elements
    for aff in article_elem.findall(".//Affiliation"):
        if aff.text:
            aff_text = aff.text.lower()
            for pattern in GOV_AFFILIATION_PATTERNS:
                if pattern.lower() in aff_text:
                    # Extract agency name from matched pattern
                    agency = extract_agency_name_from_pattern(pattern)
                    if agency:
                        matched_agencies.add(agency)
    
    # Check AffiliationInfo elements (newer format)
    for aff_info in article_elem.findall(".//AffiliationInfo/Affiliation"):
        if aff_info.text:
            aff_text = aff_info.text.lower()
            for pattern in GOV_AFFILIATION_PATTERNS:
                if pattern.lower() in aff_text:
                    agency = extract_agency_name_from_pattern(pattern)
                    if agency:
                        matched_agencies.add(agency)
    
    # Check grant list for NIH/government grants
    for grant in article_elem.findall(".//Grant"):
        agency = grant.find("Agency")
        if agency is not None and agency.text:
            agency_lower = agency.text.lower()
            if any(g in agency_lower for g in ["nih", "national institutes", "cdc", "fda", "va ", "veterans", "ahrq"]):
                matched_agencies.add(agency.text.strip())
    
    is_gov = len(matched_agencies) > 0
    return is_gov, sorted(list(matched_agencies))
