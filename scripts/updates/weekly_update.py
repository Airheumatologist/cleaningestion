#!/usr/bin/env python3
"""Weekly incremental updater for self-hosted Medical RAG.

Replaces the monthly cadence (08_monthly_update.py) with a weekly schedule
that matches the daily-update frequency of our upstream datasets (PubMed and
DailyMed both publish new files every weekday).

Run via cron on Saturday at 01:00; backup.sh is chained in the cron entry and
runs immediately after this script exits successfully.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ftplib
import gzip
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from qdrant_client.models import PointStruct

# Add scripts root to path so sibling shared modules import cleanly
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import (
    Chunker as BaseChunker,
    EmbeddingProvider,
    append_checkpoint as append_checkpoint_file,
    generate_section_id,
    get_section_weight,
    get_chunker as get_shared_chunker,
    classify_evidence_metadata,
    get_evidence_level,
    extract_gov_affiliations_from_pubmed_xml,
)
from turbopuffer_ingestion_sink import build_ingestion_sink
from updates.ingest_dailymed_updates_direct import run_direct_update as run_dailymed_direct_update
from pubmed_publication_filters import (
    is_target_article,
    map_publication_type,
)
import turbopuffer as tpuf

# Import enhanced utilities for semantic chunking
try:
    from ingestion_utils_enhanced import SemanticChunker, QualityValidator
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    SemanticChunker = None  # type: ignore
    QualityValidator = None  # type: ignore
    ENHANCED_UTILS_AVAILABLE = False

CHUNKER_CLASS = SemanticChunker if ENHANCED_UTILS_AVAILABLE else BaseChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PUBMED_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
PUBMED_UPDATE_DIR = "/pubmed/updatefiles/"
PROCESSED_TRACKER = IngestionConfig.DATA_DIR / "processed_updates.json"
# Shared checkpoint with the baseline ingestion script (21_ingest_pubmed_abstracts.py).
# Format: "pubmed:{pmid}" per line — consistent with PUBMED_CHECKPOINT_NAMESPACE below.
CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pubmed_ingested_ids.txt"
PUBMED_CHECKPOINT_NAMESPACE = "pubmed"

# PubMed filtering constants
DEFAULT_MIN_YEAR = 2015
PUBMED_LEAN_DROP_FIELDS = {
    "token_count",
    "source_family",
    "full_section_text",
    "full_text",
    "text_preview",
}

def _checkpoint_id(pmid: str) -> str:
    """Generate namespaced checkpoint ID for PubMed abstracts."""
    return f"{PUBMED_CHECKPOINT_NAMESPACE}:{pmid.strip()}"

def load_processed_files() -> Set[str]:
    if not PROCESSED_TRACKER.exists():
        return set()
    try:
        return set(json.loads(PROCESSED_TRACKER.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_processed_files(processed: Set[str]) -> None:
    PROCESSED_TRACKER.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_TRACKER.write_text(json.dumps(sorted(processed), indent=2), encoding="utf-8")


def _resolve_checkpoint_pmid(line: str) -> Optional[str]:
    """Resolve checkpoint line to PMID, handling namespaced and legacy formats."""
    value = line.strip()
    if not value:
        return None

    # Namespaced format (current baseline ingestion)
    if ":" in value:
        namespace, pmid = value.split(":", 1)
        if namespace != PUBMED_CHECKPOINT_NAMESPACE:
            return None
        value = pmid

    value = value.strip()
    return value if value else None


def load_baseline_pmids(path: Path) -> Set[str]:
    """Load baseline PubMed checkpoint PMIDs for cross-pipeline deduplication."""
    if not path.exists():
        return set()

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.warning("Failed to load baseline checkpoint %s: %s", path, exc)
        return set()

    return {pmid for line in lines if (pmid := _resolve_checkpoint_pmid(line)) is not None}


def get_pubmed_namespace_last_write_at() -> Optional[datetime]:
    """Return PubMed namespace last_write_at timestamp from TurboPuffer metadata."""
    try:
        client = tpuf.Turbopuffer(
            api_key=IngestionConfig.TURBOPUFFER_API_KEY,
            region=IngestionConfig.TURBOPUFFER_REGION,
        )
        metadata = client.namespace(IngestionConfig.TURBOPUFFER_NAMESPACE_PUBMED).metadata()
        last_write_at = getattr(metadata, "last_write_at", None)
        if not last_write_at:
            return None
        return datetime.fromisoformat(str(last_write_at).replace("Z", "+00:00"))
    except Exception as exc:
        logger.warning("Failed to fetch PubMed namespace last_write_at: %s", exc)
        return None


def _parse_ftp_modify(value: str) -> Optional[datetime]:
    """Parse MLSD modify timestamps (YYYYMMDDHHMMSS[.sss])."""
    if not value:
        return None
    raw = value.split(".", 1)[0]
    try:
        return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def list_pubmed_update_files(max_files: Optional[int], since: Optional[datetime] = None) -> List[str]:
    last_error: Optional[Exception] = None
    for attempt in range(1, 6):
        ftp: Optional[ftplib.FTP] = None
        try:
            ftp = ftplib.FTP(PUBMED_FTP_HOST, timeout=120)
            ftp.login()
            ftp.cwd(PUBMED_UPDATE_DIR)
            files: List[tuple[str, Optional[datetime]]] = []
            try:
                for name, facts in ftp.mlsd(facts=["modify"]):
                    if not name.endswith(".xml.gz"):
                        continue
                    modified = _parse_ftp_modify(facts.get("modify", ""))
                    files.append((name, modified))
            except Exception:
                logger.warning("FTP MLSD listing unavailable; falling back to NLST without timestamp filtering")
                files = [(name, None) for name in ftp.nlst() if name.endswith(".xml.gz")]

            files = sorted(files, key=lambda item: item[0])
            if since is not None:
                since_utc = since.astimezone(timezone.utc)
                files = [(name, modified) for name, modified in files if modified is None or modified > since_utc]
            selected = [name for name, _ in files]
            if max_files:
                selected = selected[-max_files:]
            return selected
        except Exception as exc:
            last_error = exc
            if attempt < 5:
                wait = min(2 ** (attempt - 1), 8)
                logger.warning("PubMed FTP file listing failed attempt %d/5: %s (retrying in %ss)", attempt, exc, wait)
                time.sleep(wait)
        finally:
            if ftp is not None:
                try:
                    ftp.quit()
                except Exception:
                    pass
    raise RuntimeError(f"Failed to list PubMed update files after retries: {last_error}")


def _iter_pubmed_articles_from_ftp(file_name: str, min_year: int = DEFAULT_MIN_YEAR) -> Iterable[Dict[str, Any]]:
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        ftp: Optional[ftplib.FTP] = None
        conn = None
        try:
            ftp = ftplib.FTP(PUBMED_FTP_HOST, timeout=120)
            ftp.login()
            ftp.cwd(PUBMED_UPDATE_DIR)
            conn = ftp.transfercmd(f"RETR {file_name}")
            with conn.makefile("rb") as stream, gzip.GzipFile(fileobj=stream, mode="rb") as handle:
                context = ET.iterparse(handle, events=("end",))
                for _, elem in context:
                    if elem.tag != "PubmedArticle":
                        continue
                    for article in parse_pubmed_article_element(elem, min_year=min_year):
                        yield article
                    elem.clear()
            ftp.voidresp()
            return
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                wait = min(2 ** (attempt - 1), 6)
                logger.warning(
                    "PubMed FTP stream failed file=%s attempt %d/3: %s (retrying in %ss)",
                    file_name,
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            if ftp is not None:
                try:
                    ftp.quit()
                except Exception:
                    try:
                        ftp.close()
                    except Exception:
                        pass
    raise RuntimeError(f"Failed to stream PubMed FTP file after retries: {file_name} ({last_error})")


def parse_pubmed_date(pub_date_elem: Optional[ET.Element]) -> Dict[str, Any]:
    """Parse PubDate into a normalized dict used by baseline ingestion."""
    result = {
        "year": None,
        "month": None,
        "day": None,
        "medline_date": None,
        "date_str": None,
    }

    if pub_date_elem is None:
        return result

    year_elem = pub_date_elem.find("Year")
    month_elem = pub_date_elem.find("Month")
    day_elem = pub_date_elem.find("Day")

    if year_elem is not None and year_elem.text:
        result["year"] = year_elem.text.strip()
    if month_elem is not None and month_elem.text:
        result["month"] = month_elem.text.strip()
    if day_elem is not None and day_elem.text:
        result["day"] = day_elem.text.strip()

    if result["year"] is None:
        medline_date_elem = pub_date_elem.find("MedlineDate")
        if medline_date_elem is not None and medline_date_elem.text:
            result["medline_date"] = medline_date_elem.text.strip()
            match = re.search(r"(\d{4})", result["medline_date"])
            if match:
                result["year"] = match.group(1)

    if result["year"]:
        parts = [result["year"]]
        if result["month"]:
            parts.append(result["month"])
            if result["day"]:
                parts.append(result["day"])
        result["date_str"] = " ".join(parts)
    elif result["medline_date"]:
        result["date_str"] = result["medline_date"]

    return result


def extract_abstract(article_elem: ET.Element) -> Dict[str, Any]:
    """Extract plain + structured abstract sections."""
    result: Dict[str, Any] = {
        "abstract_text": "",
        "abstract_structured": [],
        "has_structured_abstract": False,
    }

    abstract_elem = article_elem.find(".//Abstract")
    if abstract_elem is None:
        return result

    abstract_parts: List[str] = []
    structured_parts: List[Dict[str, Any]] = []
    has_structured = False

    for abstract_text in abstract_elem.findall("AbstractText"):
        label = abstract_text.get("Label", "")
        nlm_category = abstract_text.get("NlmCategory", "")
        text = "".join(abstract_text.itertext()).strip()
        if not text:
            continue

        if label:
            has_structured = True
            structured_parts.append(
                {
                    "label": label,
                    "nlm_category": nlm_category,
                    "text": text,
                }
            )
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)

    result["abstract_text"] = " ".join(abstract_parts).strip()
    result["abstract_structured"] = structured_parts
    result["has_structured_abstract"] = has_structured
    return result


def extract_journal_info(article_elem: ET.Element) -> Dict[str, Any]:
    """Extract journal info fields compatible with baseline ingestion."""
    result: Dict[str, Any] = {
        "title": None,
        "iso_abbreviation": None,
        "issn": None,
        "issn_type": None,
        "volume": None,
        "issue": None,
        "pub_date": {},
        "cited_medium": None,
        "nlm_unique_id": None,
    }

    journal_elem = article_elem.find(".//Journal")
    if journal_elem is None:
        return result

    issn_elem = journal_elem.find("ISSN")
    if issn_elem is not None:
        result["issn"] = issn_elem.text.strip() if issn_elem.text else None
        result["issn_type"] = issn_elem.get("IssnType")

    # NLM Unique ID (from MedlineJournalInfo)
    nlm_id_elem = article_elem.find(".//MedlineJournalInfo/NlmUniqueID")
    if nlm_id_elem is not None and nlm_id_elem.text:
        result["nlm_unique_id"] = nlm_id_elem.text.strip()

    title_elem = journal_elem.find("Title")
    if title_elem is not None and title_elem.text:
        result["title"] = title_elem.text.strip()

    iso_elem = journal_elem.find("ISOAbbreviation")
    if iso_elem is not None and iso_elem.text:
        result["iso_abbreviation"] = iso_elem.text.strip()

    issue_elem = journal_elem.find("JournalIssue")
    if issue_elem is not None:
        result["cited_medium"] = issue_elem.get("CitedMedium")

        volume_elem = issue_elem.find("Volume")
        if volume_elem is not None and volume_elem.text:
            result["volume"] = volume_elem.text.strip()

        issue_num_elem = issue_elem.find("Issue")
        if issue_num_elem is not None and issue_num_elem.text:
            result["issue"] = issue_num_elem.text.strip()

        result["pub_date"] = parse_pubmed_date(issue_elem.find("PubDate"))

    return result


def extract_article_ids(article_elem: ET.Element) -> Dict[str, Any]:
    """Extract PMID/DOI/PMCID/PII and other IDs."""
    result: Dict[str, Any] = {
        "pmid": None,
        "doi": None,
        "pmc": None,
        "pii": None,
        "other_ids": {},
    }

    # Only use the article's canonical PMID, not PMIDs from references.
    pmid_elem = article_elem.find("./MedlineCitation/PMID")
    if pmid_elem is not None and pmid_elem.text:
        result["pmid"] = pmid_elem.text.strip()

    # Only use IDs from PubmedData/ArticleIdList for the current article.
    for article_id in article_elem.findall("./PubmedData/ArticleIdList/ArticleId"):
        id_type = article_id.get("IdType", "").lower()
        id_value = article_id.text.strip() if article_id.text else None
        if not id_value:
            continue

        if id_type == "doi":
            result["doi"] = id_value
        elif id_type == "pubmed":
            result["pmid"] = id_value
        elif id_type in {"pmc", "pmcid"}:
            result["pmc"] = id_value
        elif id_type == "pii":
            result["pii"] = id_value
        else:
            result["other_ids"][id_type] = id_value

    # ELocationID belongs to MedlineCitation/Article for the current article.
    for eloc_id in article_elem.findall("./MedlineCitation/Article/ELocationID"):
        id_type = eloc_id.get("EIdType", "").lower()
        id_value = eloc_id.text.strip() if eloc_id.text else None
        if not id_value:
            continue
        if id_type == "doi" and not result["doi"]:
            result["doi"] = id_value
        elif id_type == "pii" and not result["pii"]:
            result["pii"] = id_value

    return result


def extract_mesh_terms(article_elem: ET.Element) -> List[Dict[str, Any]]:
    """Extract structured MeSH terms including qualifiers."""
    mesh_terms: List[Dict[str, Any]] = []
    for mesh_heading in article_elem.findall(".//MeshHeading"):
        descriptor = mesh_heading.find("DescriptorName")
        if descriptor is None or not descriptor.text:
            continue

        term: Dict[str, Any] = {
            "descriptor": descriptor.text.strip(),
            "descriptor_ui": descriptor.get("UI"),
            "major_topic": descriptor.get("MajorTopicYN") == "Y",
            "type": descriptor.get("Type"),
            "qualifiers": [],
        }
        qualifiers: List[Dict[str, Any]] = []
        for qualifier in mesh_heading.findall("QualifierName"):
            if qualifier.text:
                qualifiers.append(
                    {
                        "name": qualifier.text.strip(),
                        "ui": qualifier.get("UI"),
                        "major_topic": qualifier.get("MajorTopicYN") == "Y",
                    }
                )
        term["qualifiers"] = qualifiers
        mesh_terms.append(term)
    return mesh_terms


def extract_keywords(article_elem: ET.Element) -> List[Dict[str, Any]]:
    """Extract structured keyword entries."""
    keywords: List[Dict[str, Any]] = []
    keyword_list = article_elem.find(".//KeywordList")
    if keyword_list is None:
        return keywords

    owner = keyword_list.get("Owner", "")
    for keyword in keyword_list.findall("Keyword"):
        if keyword.text:
            keywords.append(
                {
                    "keyword": keyword.text.strip(),
                    "major_topic": keyword.get("MajorTopicYN") == "Y",
                    "owner": owner,
                }
            )
    return keywords


def extract_publication_types(article_elem: ET.Element) -> List[Dict[str, Any]]:
    """Extract publication type entries."""
    pub_types: List[Dict[str, Any]] = []
    for pub_type in article_elem.findall(".//PublicationType"):
        if pub_type.text:
            pub_types.append({"type": pub_type.text.strip(), "ui": pub_type.get("UI")})
    return pub_types


def parse_pubmed_article_element(elem: ET.Element, min_year: int = DEFAULT_MIN_YEAR) -> Iterable[Dict[str, Any]]:
    article_ids = extract_article_ids(elem)
    pmid = str(article_ids.get("pmid") or "").strip()
    if not pmid:
        return

    title_elem = elem.find(".//ArticleTitle")
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
    abstract_data = extract_abstract(elem)
    abstract = str(abstract_data.get("abstract_text") or "").strip()

    if not pmid or not title or not abstract:
        return
    if len(abstract) < 50:
        return

    pub_types_data = extract_publication_types(elem)
    pub_types_flat = [pt["type"] for pt in pub_types_data]
    if not is_target_article(pub_types_flat):
        return

    journal_info = extract_journal_info(elem)
    pub_date = journal_info.get("pub_date", {})

    year: Optional[int] = None
    year_str = pub_date.get("year")
    if year_str is not None:
        try:
            year = int(str(year_str))
        except ValueError:
            year = None
    if year is None or year < min_year:
        return

    journal = journal_info.get("title") or journal_info.get("iso_abbreviation") or ""
    article_type = map_publication_type(pub_types_flat)
    if not article_type:
        return
    evidence = classify_evidence_metadata(
        article_type=article_type,
        pub_types=pub_types_flat,
        abstract=abstract,
    )
    mesh_terms = extract_mesh_terms(elem)
    keywords = extract_keywords(elem)
    is_gov_affiliated, gov_agencies = extract_gov_affiliations_from_pubmed_xml(elem)

    yield {
        "pmid": pmid,
        "doi": article_ids.get("doi"),
        "pmc": article_ids.get("pmc"),
        "pii": article_ids.get("pii"),
        "other_ids": article_ids.get("other_ids", {}),
        "title": title,
        "abstract": abstract,
        "abstract_structured": abstract_data.get("abstract_structured", []),
        "has_structured_abstract": abstract_data.get("has_structured_abstract", False),
        "year": year,
        "journal": journal,
        "journal_full": {
            "title": journal_info.get("title"),
            "iso_abbreviation": journal_info.get("iso_abbreviation"),
            "issn": journal_info.get("issn"),
            "issn_type": journal_info.get("issn_type"),
            "volume": journal_info.get("volume"),
            "issue": journal_info.get("issue"),
            "cited_medium": journal_info.get("cited_medium"),
            "nlm_unique_id": journal_info.get("nlm_unique_id"),
        },
        "publication_date": pub_date,
        "mesh_terms": mesh_terms,
        "mesh_terms_flat": [m["descriptor"] for m in mesh_terms],
        "keywords": keywords,
        "keywords_flat": [k["keyword"] for k in keywords],
        "publication_types": pub_types_data,
        "publication_types_flat": pub_types_flat,
        "article_type": article_type,
        "evidence_grade": evidence["grade"],
        "evidence_level": evidence["level_1_4"],
        "evidence_term": evidence["matched_term"],
        "evidence_source": evidence["matched_from"],
        "source": "pubmed_abstract",
        "has_full_text": False,
        "content_type": "abstract",
        "is_gov_affiliated": is_gov_affiliated,
        "gov_agencies": gov_agencies,
    }


def build_points(
    batch: List[Dict[str, Any]],
    embedding_provider: EmbeddingProvider,
    drop_fields: Optional[Set[str]] = None,
) -> tuple[List[PointStruct], List[str]]:
    """Build Qdrant points from article batch with semantic chunking.

    Point IDs are generated as uuid5(NAMESPACE_DNS, "pubmed:{chunk_id}"), which
    is identical to the scheme used by 21_ingest_pubmed_abstracts.py so that
    upserts from this script overwrite — rather than duplicate — baseline points.

    Args:
        batch: List of parsed articles
        embedding_provider: Provider for generating embeddings
    """
    # Use shared chunker instance (SemanticChunker if available)
    chunker = get_shared_chunker(
        chunker_class=CHUNKER_CLASS,
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
    )

    points: List[PointStruct] = []
    processed_pmids: List[str] = []
    seen_pmids: Set[str] = set()

    # Batch collection
    all_chunks_text: List[str] = []
    chunk_metadata: List[Dict[str, Any]] = []

    for article in batch:
        pmid = str(article.get("pmid") or "").strip()
        if not pmid:
            continue

        title = article.get("title", "")
        abstract = article.get("abstract", "")

        # Combine title + abstract for chunking with proper format
        text = f"Title: {title}\n\nAbstract: {abstract}" if title or abstract else ""
        if len(text) < 50:
            continue

        abstract_structured = article.get("abstract_structured", [])
        has_structured_abstract = bool(article.get("has_structured_abstract", False))
        journal_full = article.get("journal_full", {})
        nlm_unique_id = journal_full.get("nlm_unique_id") if journal_full else None
        publication_date = article.get("publication_date", {})

        mesh_terms_data = article.get("mesh_terms", [])
        mesh_terms_full = (
            mesh_terms_data
            if isinstance(mesh_terms_data, list) and mesh_terms_data and isinstance(mesh_terms_data[0], dict)
            else []
        )
        mesh_terms_flat = article.get("mesh_terms_flat", [])
        if not mesh_terms_flat:
            if mesh_terms_full:
                mesh_terms_flat = [m.get("descriptor", "") for m in mesh_terms_full if m.get("descriptor")]
            elif isinstance(mesh_terms_data, list):
                mesh_terms_flat = [str(m).strip() for m in mesh_terms_data if m]

        keywords_data = article.get("keywords", [])
        keywords_full = (
            keywords_data
            if isinstance(keywords_data, list) and keywords_data and isinstance(keywords_data[0], dict)
            else []
        )
        keywords_flat = article.get("keywords_flat", [])
        if not keywords_flat:
            if keywords_full:
                keywords_flat = [k.get("keyword", "") for k in keywords_full if k.get("keyword")]
            elif isinstance(keywords_data, list):
                keywords_flat = [str(k).strip() for k in keywords_data if k]

        publication_types_data = article.get("publication_types", [])
        publication_types_full = (
            publication_types_data
            if isinstance(publication_types_data, list)
            and publication_types_data
            and isinstance(publication_types_data[0], dict)
            else []
        )
        publication_types_flat = article.get("publication_types_flat", [])
        if not publication_types_flat:
            if publication_types_full:
                publication_types_flat = [p.get("type", "") for p in publication_types_full if p.get("type")]
            elif isinstance(publication_types_data, list):
                publication_types_flat = [str(p).strip() for p in publication_types_data if p]
        if not publication_types_flat:
            legacy_pub_types = article.get("publication_type", [])
            if isinstance(legacy_pub_types, list):
                publication_types_flat = [str(p).strip() for p in legacy_pub_types if p]
            elif legacy_pub_types:
                publication_types_flat = [str(legacy_pub_types).strip()]

        # Base section ID for abstract
        base_section_id = generate_section_id(pmid, "Abstract")

        # Chunk the content using shared chunker
        text_chunks = chunker.chunk_text(text)

        for i, chunk in enumerate(text_chunks):
            chunk_text = chunk["text"]
            all_chunks_text.append(chunk_text)

            chunk_section_id = (
                base_section_id if len(text_chunks) == 1 else generate_section_id(pmid, f"Abstract_{i}")
            )

            # Extract chunk abstract (remove title prefix if present)
            chunk_abstract = chunk_text
            abstract_prefix = "\n\nAbstract: "
            prefixed_title = f"Title: {title}{abstract_prefix}"
            if chunk_text.startswith(prefixed_title):
                chunk_abstract = chunk_text[len(prefixed_title):]
            elif chunk_text.startswith("Title: ") and abstract_prefix in chunk_text:
                chunk_abstract = chunk_text.split(abstract_prefix, 1)[1]

            # Build enhanced payload consistent with other ingestion scripts
            payload = {
                # Core identifiers
                "doc_id": pmid,
                "chunk_id": f"{pmid}_abstract_chunk_{i}" if len(text_chunks) > 1 else f"{pmid}_abstract",
                "pmid": pmid,
                "pmcid": article.get("pmc"),
                "doi": article.get("doi"),
                "pii": article.get("pii"),
                "other_ids": article.get("other_ids", {}),

                # CRITICAL: Full text for retriever
                "page_content": chunk_text,

                # Content
                "title": title if i == 0 else f"{title} (cont.)" if title else f"Abstract (Part {i+1})",
                "abstract": chunk_abstract,
                "abstract_structured": abstract_structured if i == 0 else [],
                "has_structured_abstract": has_structured_abstract if i == 0 else False,
                "full_text": "",

                # Chunking metadata
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "token_count": chunk.get("token_count", 0),

                # Publication info
                "year": article.get("year"),
                "journal": article.get("journal", ""),
                "journal_full": journal_full,
                "nlm_unique_id": nlm_unique_id,
                "publication_date": publication_date,
                "article_type": article.get("article_type", "research_article"),
                "publication_type": publication_types_flat,
                "publication_types_full": publication_types_full if i == 0 else [],

                # Evidence
                "evidence_grade": article.get("evidence_grade", "C"),
                "evidence_level": article.get("evidence_level", get_evidence_level(article.get("evidence_grade", "C"))),
                "evidence_term": article.get("evidence_term"),
                "evidence_source": article.get("evidence_source", "fallback"),

                # Classification
                "mesh_terms": mesh_terms_flat if i == 0 else [],
                "mesh_terms_full": mesh_terms_full if i == 0 else [],
                "keywords": keywords_flat if i == 0 else [],
                "keywords_full": keywords_full if i == 0 else [],

                # Government Affiliation
                "is_gov_affiliated": article.get("is_gov_affiliated", False),
                "gov_agencies": article.get("gov_agencies", []) if i == 0 else [],

                # Parent-child indexing fields
                "section_title": f"Abstract (Part {i+1}/{len(text_chunks)})" if len(text_chunks) > 1 else "Abstract",
                "section_type": "abstract",
                "section_id": chunk_section_id,
                "parent_section_id": base_section_id if len(text_chunks) > 1 else None,
                "section_weight": get_section_weight("abstract") - (i * 0.05),  # Slight decay for later chunks
                "full_section_text": text,

                # Source and type
                "source": article.get("source", "pubmed_abstract"),
                "has_full_text": False,
                "content_type": "abstract",
                "table_count": 0,

                # Preview for dashboard
                "text_preview": chunk_text[:500],
            }
            if drop_fields:
                for field in drop_fields:
                    payload.pop(field, None)

            chunk_metadata.append(payload)

    if not all_chunks_text:
        return [], []

    # Embed all chunks using shared embedding provider
    vectors = embedding_provider.embed_batch(all_chunks_text)

    for i, vector in enumerate(vectors):
        payload = chunk_metadata[i]
        # Deterministic UUID matching 21_ingest_pubmed_abstracts.py so upserts
        # overwrite baseline points rather than creating duplicates.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed:{payload['chunk_id']}"))
        vector_data: Any = {"dense": vector}

        points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))
        point_pmid = str(payload.get("pmid") or "").strip()
        if point_pmid and point_pmid not in seen_pmids:
            seen_pmids.add(point_pmid)
            processed_pmids.append(point_pmid)

    return points, processed_pmids


def ingest_pubmed_updates(
    sink: Any,
    embedding_provider: EmbeddingProvider,
    max_files: Optional[int],
    min_year: int = DEFAULT_MIN_YEAR,
    batch_size: Optional[int] = None,
    throttle_seconds: Optional[float] = None,
    file_workers: int = 1,
    checkpoint_flush_size: int = 200,
    since_override: Optional[datetime] = None,
    lean_payload: bool = False,
) -> int:
    effective_batch_size = (
        batch_size
        if batch_size is not None and batch_size > 0
        else (
            IngestionConfig.WEEKLY_UPDATE_BATCH_SIZE
            if IngestionConfig.WEEKLY_UPDATE_BATCH_SIZE > 0
            else IngestionConfig.BATCH_SIZE
        )
    )
    effective_throttle = (
        throttle_seconds
        if throttle_seconds is not None and throttle_seconds >= 0
        else max(0.0, IngestionConfig.WEEKLY_UPDATE_THROTTLE_SECONDS)
    )

    logger.info(
        "Weekly PubMed QoS settings: batch_size=%d throttle_seconds=%.3f file_workers=%d checkpoint_flush_size=%d lean_payload=%s",
        effective_batch_size,
        effective_throttle,
        max(1, file_workers),
        max(1, checkpoint_flush_size),
        lean_payload,
    )

    processed = load_processed_files()
    baseline_pmids = load_baseline_pmids(CHECKPOINT_FILE)
    logger.info(
        "Loaded baseline PubMed checkpoint IDs=%d from %s (used for checkpoint dedupe only)",
        len(baseline_pmids),
        CHECKPOINT_FILE,
    )

    namespace_last_write_at = since_override if since_override is not None else get_pubmed_namespace_last_write_at()
    if namespace_last_write_at is not None:
        logger.info(
            "PubMed namespace last_write_at=%s (UTC) - selecting FTP updates newer than this timestamp",
            namespace_last_write_at.astimezone(timezone.utc).isoformat(),
        )
    all_files = list_pubmed_update_files(max_files=max_files, since=namespace_last_write_at)
    new_files = [f for f in all_files if f not in processed]
    drop_fields = PUBMED_LEAN_DROP_FIELDS if lean_payload else None

    logger.info("PubMed update files total=%s new=%s", len(all_files), len(new_files))

    # Log chunking configuration
    chunker = get_shared_chunker(
        chunker_class=CHUNKER_CLASS,
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS,
    )
    logger.info("Chunking config: size=%d, overlap=%d, chunker=%s",
                IngestionConfig.CHUNK_SIZE_TOKENS,
                IngestionConfig.CHUNK_OVERLAP_TOKENS,
                chunker.__class__.__name__)

    inserted = 0
    total_articles_processed = 0
    total_batches_upserted = 0
    checkpoint_buffer: List[str] = []
    state_lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    pubmed_started = time.time()

    def flush_checkpoint_buffer(force: bool = False) -> None:
        ids_to_flush: List[str] = []
        with checkpoint_lock:
            if not checkpoint_buffer:
                return
            if not force and len(checkpoint_buffer) < max(1, checkpoint_flush_size):
                return
            ids_to_flush = list(checkpoint_buffer)
            checkpoint_buffer.clear()
        append_checkpoint_file(CHECKPOINT_FILE, ids_to_flush)

    def process_single_file(file_name: str) -> tuple[int, int, int]:
        file_started = time.time()
        logger.info("Processing update file: %s", file_name)
        batch: List[Dict[str, Any]] = []
        file_inserted = 0
        file_articles_processed = 0
        file_batches_upserted = 0

        def process_article_batch(articles: List[Dict[str, Any]]) -> None:
            nonlocal file_inserted, file_batches_upserted
            nonlocal inserted, total_batches_upserted
            points, ingested_pmids = build_points(
                articles,
                embedding_provider,
                drop_fields=drop_fields,
            )
            if not points:
                return
            sink.write_points(points)
            with state_lock:
                new_checkpoint_pmids = [pmid for pmid in ingested_pmids if pmid not in baseline_pmids]
                checkpoint_ids = [_checkpoint_id(pmid) for pmid in new_checkpoint_pmids]
                if checkpoint_ids:
                    with checkpoint_lock:
                        checkpoint_buffer.extend(checkpoint_ids)
                baseline_pmids.update(ingested_pmids)
                inserted += len(points)
                total_batches_upserted += 1
                file_inserted += len(points)
                file_batches_upserted += 1
            flush_checkpoint_buffer(force=False)
            if effective_throttle > 0:
                time.sleep(effective_throttle)

        for article in _iter_pubmed_articles_from_ftp(file_name, min_year=min_year):
            batch.append(article)
            file_articles_processed += 1
            if len(batch) < effective_batch_size:
                continue
            process_article_batch(batch)
            batch.clear()

        if batch:
            process_article_batch(batch)

        with state_lock:
            total_articles_processed += file_articles_processed
            processed.add(file_name)
            save_processed_files(processed)
        file_elapsed = max(0.001, time.time() - file_started)
        logger.info(
            "Completed file: %s inserted=%d articles=%d batches=%d elapsed=%.1fs throughput_points_per_s=%.2f",
            file_name,
            file_inserted,
            file_articles_processed,
            file_batches_upserted,
            file_elapsed,
            file_inserted / file_elapsed,
        )
        return file_inserted, file_articles_processed, file_batches_upserted

    workers = max(1, int(file_workers))
    failed_files: List[str] = []
    if workers == 1:
        for file_name in new_files:
            try:
                process_single_file(file_name)
            except Exception:
                failed_files.append(file_name)
                logger.exception("Failed processing PubMed update file: %s", file_name)
    else:
        logger.info("PubMed file-level parallelism enabled workers=%d", workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_file, name): name for name in new_files}
            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    future.result()
                except Exception:
                    failed_files.append(file_name)
                    logger.exception("Failed processing PubMed update file: %s", file_name)

    flush_checkpoint_buffer(force=True)

    if failed_files:
        logger.warning("PubMed files failed=%d files=%s", len(failed_files), ",".join(sorted(failed_files)))

    pubmed_elapsed = max(0.001, time.time() - pubmed_started)
    logger.info(
        "Weekly PubMed update summary: inserted=%d articles=%d batches=%d elapsed=%.1fs throughput_points_per_s=%.2f",
        inserted,
        total_articles_processed,
        total_batches_upserted,
        pubmed_elapsed,
        inserted / pubmed_elapsed,
    )

    return inserted


def run_dailymed_refresh() -> None:
    """Run DailyMed incremental update by streaming daily ZIPs into TurboPuffer.

    The start date is derived from TurboPuffer namespace ``last_write_at`` so the
    job catches up every DailyMed daily update published after the most recent
    successful write.
    """
    logger.info(
        "Starting DailyMed direct incremental refresh (namespace=%s)",
        IngestionConfig.TURBOPUFFER_NAMESPACE_DAILYMED,
    )
    rows_written = run_dailymed_direct_update(
        namespace=IngestionConfig.TURBOPUFFER_NAMESPACE_DAILYMED,
        checkpoint_file=IngestionConfig.DAILYMED_CHECKPOINT_FILE,
    )
    logger.info("DailyMed direct incremental refresh wrote %d rows", rows_written)


def run_pmc_refresh() -> None:
    """Run PMC OA + Author Manuscript incremental update.

    Uses the S3 inventory last-modified cutoff stored in
    ``.pmc_s3_inventory_state.json`` inside PMC_XML_DIR to download only
    articles that are new or updated since the previous run.  On the very
    first call there is no cutoff, so the full inventory is scanned —
    identical to a baseline download, which is expected behaviour.

    Step 1 — download: 01_download_pmc_unified.py --release-mode incremental
        Fetches new/updated XML files for both pmc_oa and author_manuscript
        and advances the state file so the next weekly run starts from here.

    Step 2 — ingest: 06_ingest_pmc.py --xml-dir ... --delete-source
        Processes only files not already in pmc_ingested_ids.txt and removes
        each XML immediately after successful ingestion to keep disk usage
        bounded (full-text XMLs can be large).
    """
    logger.info("Starting PMC incremental refresh (pmc_oa + author_manuscript)")
    scripts_dir = PROJECT_ROOT / "scripts"
    # Step 1: download new/updated articles from PMC Cloud Service (AWS S3)
    subprocess.run([
        sys.executable, str(scripts_dir / "01_download_pmc_unified.py"),
        "--output-dir", str(IngestionConfig.PMC_XML_DIR),
        "--datasets", "pmc_oa,author_manuscript",
        "--release-mode", "incremental",
    ], check=True, cwd=PROJECT_ROOT)
    # Step 2: ingest — checkpoint skips already-ingested IDs; delete after
    subprocess.run([
        sys.executable, str(scripts_dir / "06_ingest_pmc.py"),
        "--xml-dir", str(IngestionConfig.PMC_XML_DIR),
        "--delete-source",
    ], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly incremental update for self-hosted Medical RAG")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N newest PubMed update files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override ingestion data directory (defaults to DATA_DIR env / config).",
    )
    parser.add_argument("--skip-pubmed", action="store_true")
    parser.add_argument("--skip-dailymed", action="store_true")
    parser.add_argument("--skip-pmc", action="store_true")
    parser.add_argument(
        "--min-year",
        type=int,
        default=DEFAULT_MIN_YEAR,
        help=f"Minimum publication year (default: {DEFAULT_MIN_YEAR})"
    )
    parser.add_argument(
        "--since-date",
        type=str,
        default=None,
        help="Override PubMed update start date in YYYY-MM-DD (UTC); bypasses namespace last_write_at.",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=IngestionConfig.WEEKLY_UPDATE_THROTTLE_SECONDS,
        help=(
            "Sleep duration after each PubMed upsert batch (seconds). "
            f"Default from env WEEKLY_UPDATE_THROTTLE_SECONDS={IngestionConfig.WEEKLY_UPDATE_THROTTLE_SECONDS}."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=IngestionConfig.WEEKLY_UPDATE_BATCH_SIZE,
        help=(
            "Override PubMed weekly ingestion batch size. "
            "Set <=0 to use BATCH_SIZE."
        ),
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=1,
        help="Parallel PubMed file workers (default: 1). Increase to improve throughput.",
    )
    parser.add_argument(
        "--checkpoint-flush-size",
        type=int,
        default=200,
        help="Number of checkpoint IDs to buffer before appending to disk (default: 200).",
    )
    parser.add_argument(
        "--lean-pubmed-payload",
        action="store_true",
        help=(
            "Drop non-essential PubMed payload fields for lower write volume: "
            f"{', '.join(sorted(PUBMED_LEAN_DROP_FIELDS))}"
        ),
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        IngestionConfig.DATA_DIR = data_dir
        IngestionConfig.PMC_XML_DIR = Path(str(data_dir / "pmc_xml"))
        IngestionConfig.DAILYMED_CHECKPOINT_FILE = Path(str(data_dir / "dailymed_ingested_ids.txt"))
        IngestionConfig.PUBMED_BASELINE_DIR = Path(str(data_dir / "pubmed_baseline"))
        IngestionConfig.PUBMED_ABSTRACTS_FILE = Path(
            str(IngestionConfig.PUBMED_BASELINE_DIR / "filtered" / "pubmed_abstracts.jsonl")
        )
        global PROCESSED_TRACKER, CHECKPOINT_FILE
        PROCESSED_TRACKER = IngestionConfig.DATA_DIR / "processed_updates.json"
        CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "pubmed_ingested_ids.txt"
        logger.info("Overrode data directory for weekly update: %s", IngestionConfig.DATA_DIR)

    ensure_data_dirs()

    # Validate required API keys
    if not os.getenv("DEEPINFRA_API_KEY"):
        logger.error("DEEPINFRA_API_KEY not set!")
        sys.exit(1)

    if not IngestionConfig.TURBOPUFFER_API_KEY:
        logger.error("TURBOPUFFER_API_KEY not set!")
        sys.exit(1)

    embedding_provider = EmbeddingProvider()
    sink = build_ingestion_sink(namespace_override=IngestionConfig.TURBOPUFFER_NAMESPACE_PUBMED)
    since_override_dt: Optional[datetime] = None
    if args.since_date:
        try:
            since_override_dt = datetime.strptime(args.since_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error("Invalid --since-date value %r (expected YYYY-MM-DD)", args.since_date)
            sys.exit(2)

    started = time.time()
    total_inserted = 0

    if not args.skip_pubmed:
        total_inserted += ingest_pubmed_updates(
            sink, embedding_provider,
            max_files=args.max_files,
            min_year=args.min_year,
            batch_size=args.batch_size,
            throttle_seconds=args.throttle_seconds,
            file_workers=args.file_workers,
            checkpoint_flush_size=args.checkpoint_flush_size,
            since_override=since_override_dt,
            lean_payload=args.lean_pubmed_payload,
        )

    if not args.skip_dailymed:
        run_dailymed_refresh()

    if not args.skip_pmc:
        run_pmc_refresh()

    logger.info(
        "Weekly update complete inserted=%s elapsed=%.1fs",
        total_inserted, time.time() - started,
    )


if __name__ == "__main__":
    main()
