#!/usr/bin/env python3
"""Weekly incremental updater for self-hosted Medical RAG.

Replaces the monthly cadence (08_monthly_update.py) with a weekly schedule
that matches the daily-update frequency of our upstream datasets (PubMed and
DailyMed both publish new files every weekday).

Run via cron on Sunday at 01:00; backup.sh is chained in the cron entry and
runs immediately after this script exits successfully.
"""

from __future__ import annotations

import argparse
import ftplib
import gzip
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import (
    Chunker as BaseChunker,
    EmbeddingProvider,
    upsert_with_retry,
    append_checkpoint as append_checkpoint_file,
    generate_section_id,
    get_section_weight,
    get_chunker as get_shared_chunker,
    classify_evidence_metadata,
    get_evidence_level,
    extract_gov_affiliations_from_pubmed_xml,
)
from pubmed_publication_filters import (
    is_target_article,
    map_publication_type,
)

# Import BM25SparseEncoder (optional)
spec = importlib.util.find_spec("src.bm25_sparse")
if spec is not None:
    from src.bm25_sparse import BM25SparseEncoder
else:
    BM25SparseEncoder = None  # type: ignore

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
UPDATE_DOWNLOAD_DIR = IngestionConfig.DATA_DIR / "pubmed_updatefiles"
PUBMED_CHECKPOINT_NAMESPACE = "pubmed"

# PubMed filtering constants
DEFAULT_MIN_YEAR = 2015

# Default number of DailyMed weekly files to download per run.
# 1 = only the most recently completed week (appropriate for a weekly job).
# Increase to 2-4 if a run is skipped and you need to catch up.
DEFAULT_DAILYMED_WEEKS_BACK = 1


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


def list_pubmed_update_files(max_files: Optional[int]) -> List[str]:
    ftp = ftplib.FTP(PUBMED_FTP_HOST, timeout=120)
    ftp.login()
    ftp.cwd(PUBMED_UPDATE_DIR)
    files = sorted(f for f in ftp.nlst() if f.endswith(".xml.gz"))
    ftp.quit()
    if max_files:
        files = files[-max_files:]
    return files


def download_pubmed_file(file_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / file_name

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    ftp = ftplib.FTP(PUBMED_FTP_HOST, timeout=120)
    ftp.login()
    ftp.cwd(PUBMED_UPDATE_DIR)
    with local_path.open("wb") as f:
        ftp.retrbinary(f"RETR {file_name}", f.write, blocksize=1024 * 1024)
    ftp.quit()
    return local_path


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
    }

    journal_elem = article_elem.find(".//Journal")
    if journal_elem is None:
        return result

    issn_elem = journal_elem.find("ISSN")
    if issn_elem is not None:
        result["issn"] = issn_elem.text.strip() if issn_elem.text else None
        result["issn_type"] = issn_elem.get("IssnType")

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

    pmid_elem = article_elem.find(".//PMID")
    if pmid_elem is not None and pmid_elem.text:
        result["pmid"] = pmid_elem.text.strip()

    for article_id in article_elem.findall(".//ArticleId"):
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

    for eloc_id in article_elem.findall(".//ELocationID"):
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


def parse_pubmed_update_xml(
    xml_gz_path: Path, min_year: int = DEFAULT_MIN_YEAR
) -> Iterable[Dict[str, Any]]:
    with gzip.open(xml_gz_path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for event, elem in context:
            if elem.tag != "PubmedArticle":
                continue

            article_ids = extract_article_ids(elem)
            pmid = str(article_ids.get("pmid") or "").strip()
            if not pmid:
                elem.clear()
                continue

            title_elem = elem.find(".//ArticleTitle")
            title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
            abstract_data = extract_abstract(elem)
            abstract = str(abstract_data.get("abstract_text") or "").strip()

            # Basic presence check (line 132 equivalent)
            if not pmid or not title or not abstract:
                elem.clear()
                continue

            # Filter: Abstract length check (matches baseline line 748-749)
            if len(abstract) < 50:
                elem.clear()
                continue

            # Extract publication types early for filtering
            pub_types_data = extract_publication_types(elem)
            pub_types_flat = [pt["type"] for pt in pub_types_data]

            # Filter: Target publication types (matches baseline line 726)
            if not is_target_article(pub_types_flat):
                elem.clear()
                continue

            journal_info = extract_journal_info(elem)
            pub_date = journal_info.get("pub_date", {})

            year: Optional[int] = None
            year_str = pub_date.get("year")
            if year_str is not None:
                try:
                    year = int(str(year_str))
                except ValueError:
                    year = None

            # Filter: Minimum year check (matches baseline line 740)
            if year is None or year < min_year:
                elem.clear()
                continue

            journal = (
                journal_info.get("title")
                or journal_info.get("iso_abbreviation")
                or ""
            )

            article_type = map_publication_type(pub_types_flat)
            if not article_type:
                elem.clear()
                continue
            evidence = classify_evidence_metadata(
                article_type=article_type,
                pub_types=pub_types_flat,
                abstract=abstract,
            )
            evidence_grade = evidence["grade"]
            evidence_level = evidence["level_1_4"]
            evidence_term = evidence["matched_term"]
            evidence_source = evidence["matched_from"]

            mesh_terms = extract_mesh_terms(elem)
            keywords = extract_keywords(elem)

            # Extract government affiliations
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
                },
                "publication_date": pub_date,
                "mesh_terms": mesh_terms,
                "mesh_terms_flat": [m["descriptor"] for m in mesh_terms],
                "keywords": keywords,
                "keywords_flat": [k["keyword"] for k in keywords],
                "publication_types": pub_types_data,
                "publication_types_flat": pub_types_flat,
                "article_type": article_type,
                "evidence_grade": evidence_grade,
                "evidence_level": evidence_level,
                "evidence_term": evidence_term,
                "evidence_source": evidence_source,
                "source": "pubmed_abstract",
                "has_full_text": False,
                "content_type": "abstract",
                "is_gov_affiliated": is_gov_affiliated,
                "gov_agencies": gov_agencies,
            }

            elem.clear()


def build_points(batch: List[Dict[str, Any]], embedding_provider: EmbeddingProvider,
                 sparse_encoder: Optional[Any] = None,
                 validate_chunks: bool = True, dedup_chunks: bool = True) -> tuple[List[PointStruct], List[str]]:
    """Build Qdrant points from article batch with semantic chunking.

    Point IDs are generated as uuid5(NAMESPACE_DNS, "pubmed:{chunk_id}"), which
    is identical to the scheme used by 21_ingest_pubmed_abstracts.py so that
    upserts from this script overwrite — rather than duplicate — baseline points.

    Args:
        batch: List of parsed articles
        embedding_provider: Provider for generating embeddings
        sparse_encoder: Optional sparse encoder for hybrid search (created once and reused)
        validate_chunks: Whether to validate chunk quality before ingestion
        dedup_chunks: Whether to deduplicate chunks within batch
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
                "full_section_text": chunk_text,

                # Source and type
                "source": article.get("source", "pubmed_abstract"),
                "has_full_text": False,
                "content_type": "abstract",
                "table_count": 0,

                # Preview for dashboard
                "text_preview": chunk_text[:500],
            }

            chunk_metadata.append(payload)

    if not all_chunks_text:
        return [], []

    # Embed all chunks using shared embedding provider
    vectors = embedding_provider.embed_batch(all_chunks_text)

    # Batch encode sparse vectors (if enabled)
    sparse_vectors = []
    if sparse_encoder:
        try:
            sparse_vectors = sparse_encoder.encode_batch(all_chunks_text)
        except Exception as e:
            logger.warning("Sparse encoding failed for batch: %s", e)
            sparse_vectors = [None] * len(all_chunks_text)

    for i, vector in enumerate(vectors):
        payload = chunk_metadata[i]
        # Deterministic UUID matching 21_ingest_pubmed_abstracts.py so upserts
        # overwrite baseline points rather than creating duplicates.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pubmed:{payload['chunk_id']}"))
        vector_data: Any = {"dense": vector}

        # Add sparse vector if available
        if sparse_vectors and i < len(sparse_vectors) and sparse_vectors[i]:
            vector_data["sparse"] = sparse_vectors[i].model_dump() if hasattr(sparse_vectors[i], 'model_dump') else sparse_vectors[i]

        points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))
        point_pmid = str(payload.get("pmid") or "").strip()
        if point_pmid and point_pmid not in seen_pmids:
            seen_pmids.add(point_pmid)
            processed_pmids.append(point_pmid)

    return points, processed_pmids


def ingest_pubmed_updates(
    client: QdrantClient,
    embedding_provider: EmbeddingProvider,
    max_files: Optional[int],
    min_year: int = DEFAULT_MIN_YEAR
) -> int:
    processed = load_processed_files()
    baseline_pmids = load_baseline_pmids(CHECKPOINT_FILE)
    logger.info("Loaded baseline PubMed checkpoint IDs=%d from %s", len(baseline_pmids), CHECKPOINT_FILE)

    all_files = list_pubmed_update_files(max_files=max_files)
    new_files = [f for f in all_files if f not in processed]

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

    # Create sparse encoder once (reused for all batches)
    sparse_encoder: Optional[Any] = None
    if (
        IngestionConfig.SPARSE_ENABLED
        and IngestionConfig.SPARSE_MODE == "bm25"
        and BM25SparseEncoder is not None
    ):
        sparse_encoder = BM25SparseEncoder(
            max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
            max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
            min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
            remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
        )
        logger.info("BM25 sparse encoder initialized for hybrid search")
    elif IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
        logger.warning("BM25 sparse encoder unavailable; continuing with dense-only indexing")

    inserted = 0
    for file_name in new_files:
        logger.info("Processing update file: %s", file_name)
        local_file = download_pubmed_file(file_name, UPDATE_DOWNLOAD_DIR)

        batch: List[Dict[str, Any]] = []
        file_inserted = 0
        skipped_from_baseline = 0
        for article in parse_pubmed_update_xml(local_file, min_year=min_year):
            pmid = str(article.get("pmid") or "").strip()
            if pmid in baseline_pmids:
                skipped_from_baseline += 1
                continue

            batch.append(article)
            if len(batch) < IngestionConfig.BATCH_SIZE:
                continue
            points, ingested_pmids = build_points(
                batch,
                embedding_provider,
                sparse_encoder,
                validate_chunks=ENHANCED_UTILS_AVAILABLE,
                dedup_chunks=ENHANCED_UTILS_AVAILABLE,
            )
            if points:
                upsert_with_retry(client, points)
                checkpoint_ids = [_checkpoint_id(pmid) for pmid in ingested_pmids]
                if checkpoint_ids:
                    append_checkpoint_file(CHECKPOINT_FILE, checkpoint_ids)
                    baseline_pmids.update(ingested_pmids)
                inserted += len(points)
                file_inserted += len(points)
                time.sleep(0.5)  # Throttle: allow RocksDB compaction
            batch.clear()

        if batch:
            points, ingested_pmids = build_points(
                batch,
                embedding_provider,
                sparse_encoder,
                validate_chunks=ENHANCED_UTILS_AVAILABLE,
                dedup_chunks=ENHANCED_UTILS_AVAILABLE,
            )
            if points:
                upsert_with_retry(client, points)
                checkpoint_ids = [_checkpoint_id(pmid) for pmid in ingested_pmids]
                if checkpoint_ids:
                    append_checkpoint_file(CHECKPOINT_FILE, checkpoint_ids)
                    baseline_pmids.update(ingested_pmids)
                inserted += len(points)
                file_inserted += len(points)

        processed.add(file_name)
        save_processed_files(processed)
        local_file.unlink(missing_ok=True)
        logger.info("Completed file: %s inserted=%d skipped_baseline=%d", file_name, file_inserted, skipped_from_baseline)

    return inserted


def run_dailymed_refresh(weeks_back: int = DEFAULT_DAILYMED_WEEKS_BACK) -> None:
    """Run DailyMed incremental update using the most recent weekly update file(s).

    Uses DailyMed's weekly update ZIPs (dm_spl_weekly_update_*.zip) rather than
    the monthly ZIP so that label changes are picked up within the same week they
    are published.  For a weekly job, ``weeks_back=1`` (the default) is sufficient.
    Pass a higher value (e.g. 2–4) after a missed run to catch up.
    """
    logger.info("Starting DailyMed incremental refresh (weeks_back=%d)", weeks_back)
    set_id_manifest = IngestionConfig.DAILYMED_SET_ID_MANIFEST
    scripts_dir = PROJECT_ROOT / "scripts"
    subprocess.run([
        sys.executable, str(scripts_dir / "03_download_dailymed.py"),
        "--output-dir", str(IngestionConfig.DAILYMED_XML_DIR),
        "--update-type", "weekly",
        "--weeks-back", str(weeks_back),
        "--set-id-manifest", str(set_id_manifest),
    ], check=True, cwd=PROJECT_ROOT)
    # Clear checkpoint entries for updated labels so they get re-ingested
    subprocess.run([
        sys.executable, str(scripts_dir / "04_prepare_dailymed_updates.py"),
        "--xml-dir", str(IngestionConfig.DAILYMED_XML_DIR),
        "--set-id-manifest", str(set_id_manifest),
    ], check=True, cwd=PROJECT_ROOT)
    # Ingest - updated labels will now be processed
    subprocess.run([
        sys.executable, str(scripts_dir / "07_ingest_dailymed.py"),
        "--xml-dir", str(IngestionConfig.DAILYMED_XML_DIR)
    ], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly incremental update for self-hosted Medical RAG")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N newest PubMed update files")
    parser.add_argument("--skip-pubmed", action="store_true")
    parser.add_argument("--skip-dailymed", action="store_true")
    parser.add_argument(
        "--min-year",
        type=int,
        default=DEFAULT_MIN_YEAR,
        help=f"Minimum publication year (default: {DEFAULT_MIN_YEAR})"
    )
    parser.add_argument(
        "--dailymed-weeks-back",
        type=int,
        default=DEFAULT_DAILYMED_WEEKS_BACK,
        help=(
            f"Number of DailyMed weekly update ZIPs to download (default: {DEFAULT_DAILYMED_WEEKS_BACK}). "
            "Increase to 2-4 after a missed run to catch up."
        ),
    )
    args = parser.parse_args()

    ensure_data_dirs()

    # Validate required API keys
    if not os.getenv("DEEPINFRA_API_KEY"):
        logger.error("DEEPINFRA_API_KEY not set!")
        sys.exit(1)

    if not IngestionConfig.QDRANT_API_KEY:
        logger.error("QDRANT_API_KEY not set!")
        sys.exit(1)

    embedding_provider = EmbeddingProvider()
    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    started = time.time()
    total_inserted = 0

    if not args.skip_pubmed:
        total_inserted += ingest_pubmed_updates(
            client, embedding_provider,
            max_files=args.max_files,
            min_year=args.min_year,
        )

    if not args.skip_dailymed:
        run_dailymed_refresh(weeks_back=args.dailymed_weeks_back)

    info = client.get_collection(IngestionConfig.COLLECTION_NAME)
    logger.info(
        "Weekly update complete inserted=%s total_points=%s elapsed=%.1fs",
        total_inserted, info.points_count, time.time() - started,
    )


if __name__ == "__main__":
    main()
