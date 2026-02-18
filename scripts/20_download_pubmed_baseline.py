#!/usr/bin/env python3
"""
Download PubMed Baseline and Filter for High-Value Article Types.

Downloads full PubMed baseline via FTP (faster than E-utilities) and filters
locally for Reviews, Meta-Analyses, Practice Guidelines, and Systematic Reviews.

Features:
- Parallel rsync download (~2-3 hours for 31GB)
- Memory-efficient iterative XML parsing
- Multi-threaded filtering
- Checkpoint/resume support
- Maps publication types to reranker-compatible article_type values
- WHITELIST extraction: Only extracts specified PubMed XML fields

Extracted Fields (Whitelist):
- Identifiers: PMID, DOI, PMC, PII, other ArticleIds
- Article Title (full text, no character limit)
- Journal Info: Title, ISO Abbreviation, ISSN (with type), Volume, Issue, CitedMedium
- Publication Date: Year, Month, Day or MedlineDate
- Abstract: Full text (no limit) + structured sections (Label, NlmCategory)
- MeSH Terms: DescriptorName (UI, MajorTopicYN) + QualifierName (UI, MajorTopicYN)
- Keywords: Keyword (with MajorTopicYN, Owner)
- Publication Types: Type (with UI)

Usage:
    # Full pipeline (download + filter)
    python 20_download_pubmed_baseline.py --output-dir /data/ingestion/pubmed_baseline

    # Filter only (if baseline already downloaded)
    python 20_download_pubmed_baseline.py --output-dir /data/ingestion/pubmed_baseline --filter-only
    
    # Limit to year range
    python 20_download_pubmed_baseline.py --output-dir /data/ingestion/pubmed_baseline --min-year 2020

Expected Duration:
    Download: 2-3 hours (31GB compressed)
    Filter: 2-3 hours (~1.8M articles extracted)

Output:
    /data/ingestion/pubmed_baseline/filtered/pubmed_abstracts.jsonl
"""

import os
import sys
import gzip
import json
import time
import logging
import argparse
import subprocess
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set
import xml.etree.ElementTree as ET
from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import extract_gov_affiliations_from_pubmed_xml
from pubmed_publication_filters import (
    TARGET_PUBLICATION_TYPES,
    is_target_article,
    map_publication_type,
)

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

logger = logging.getLogger(__name__)

# Configuration
PUBMED_FTP_BASELINE = "ftp.ncbi.nlm.nih.gov::pubmed/baseline/"
DEFAULT_OUTPUT_DIR = IngestionConfig.PUBMED_BASELINE_DIR
DEFAULT_MIN_YEAR = 2015


def configure_logging(output_dir: Path) -> None:
    """Configure logging to console and an output-dir-local log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pubmed_baseline_download.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ],
        force=True,
    )

class ProgressTracker:
    """Track and persist progress across stages."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_file = output_dir / "progress.json"
        self.stats = {
            "stage": "init",
            "files_downloaded": 0,
            "files_processed": 0,
            "total_articles_scanned": 0,
            "filtered_articles": 0,
            "start_time": None,
            "last_update": None,
            "processed_files": []
        }
        self.load()
    
    def load(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    saved = json.load(f)
                    self.stats.update(saved)
            except:
                pass
    
    def save(self):
        self.stats["last_update"] = datetime.now().isoformat()
        with open(self.progress_file, "w") as f:
            json.dump(self.stats, f, indent=2)
    
    def update(self, **kwargs):
        self.stats.update(kwargs)
        self.save()
    
    def get_processed_files(self) -> Set[str]:
        return set(self.stats.get("processed_files", []))
    
    def add_processed_file(self, filename: str):
        if "processed_files" not in self.stats:
            self.stats["processed_files"] = []
        self.stats["processed_files"].append(filename)


def download_baseline(output_dir: Path) -> int:
    """Download PubMed baseline files via rsync."""
    baseline_dir = output_dir / "xml"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("📥 Downloading PubMed Baseline via rsync")
    logger.info("=" * 70)
    logger.info(f"Source: {PUBMED_FTP_BASELINE}")
    logger.info(f"Destination: {baseline_dir}")
    logger.info("This may take 2-3 hours depending on network speed...")
    logger.info("=" * 70)

    def rsync_source_to_ftp_url(source: str) -> str:
        """
        Convert rsync daemon syntax (host::module/path) to ftp://host/module/path.
        """
        source = source.strip()
        if source.startswith(("ftp://", "http://", "https://")):
            return source if source.endswith("/") else source + "/"
        if source.startswith("rsync://"):
            ftp_url = "ftp://" + source[len("rsync://"):]
            return ftp_url if ftp_url.endswith("/") else ftp_url + "/"
        if "::" in source:
            host, path = source.split("::", 1)
            ftp_url = f"ftp://{host}/{path.lstrip('/')}"
            return ftp_url if ftp_url.endswith("/") else ftp_url + "/"
        ftp_url = f"ftp://{source.lstrip('/')}"
        return ftp_url if ftp_url.endswith("/") else ftp_url + "/"

    def run_wget_fallback() -> bool:
        fallback_url = rsync_source_to_ftp_url(PUBMED_FTP_BASELINE)
        logger.info(f"Trying wget fallback from: {fallback_url}")
        wget_cmd = [
            "wget", "-r", "-np", "-nd", "-c", "-N",
            "-A", "*.xml.gz",
            "-P", str(baseline_dir),
            fallback_url,
        ]
        result = subprocess.run(wget_cmd, check=False)
        if result.returncode != 0:
            logger.error(f"wget fallback failed with code {result.returncode}")
            return False
        return True
    
    # Use rsync for efficient download with resume capability
    cmd = [
        "rsync",
        "-Pav",           # Progress, archive mode, verbose
        "--compress",     # Compress during transfer
        f"{PUBMED_FTP_BASELINE}*.xml.gz",
        str(baseline_dir) + "/"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            # Log progress periodically
            if "%" in line or "sent" in line.lower() or "total" in line.lower():
                print(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            logger.warning(f"rsync exited with code {process.returncode}")
            if not run_wget_fallback():
                return 0
            
    except FileNotFoundError:
        logger.warning("rsync not found; using wget fallback instead")
        if not run_wget_fallback():
            logger.error("Download failed. Please install rsync or wget and retry.")
            return 0
    except Exception as e:
        logger.error(f"Download error: {e}")
        return 0
    
    file_count = len(list(baseline_dir.glob("*.xml.gz")))
    logger.info(f"✅ Downloaded {file_count} baseline files")
    return file_count


def parse_pubmed_date(pub_date_elem: ET.Element) -> Dict[str, Any]:
    """
    Parse PubDate element which can contain:
    - Year, Month, Day elements
    - Or MedlineDate string (e.g., "2024 Spring", "2024 Jan-Feb")
    """
    result = {
        "year": None,
        "month": None,
        "day": None,
        "medline_date": None,
        "date_str": None
    }
    
    if pub_date_elem is None:
        return result
    
    # Try structured date first
    year_elem = pub_date_elem.find("Year")
    month_elem = pub_date_elem.find("Month")
    day_elem = pub_date_elem.find("Day")
    
    if year_elem is not None and year_elem.text:
        result["year"] = year_elem.text.strip()
        
    if month_elem is not None and month_elem.text:
        result["month"] = month_elem.text.strip()
        
    if day_elem is not None and day_elem.text:
        result["day"] = day_elem.text.strip()
    
    # If no Year, try MedlineDate
    if result["year"] is None:
        medline_date = pub_date_elem.find("MedlineDate")
        if medline_date is not None and medline_date.text:
            result["medline_date"] = medline_date.text.strip()
            # Try to extract year from MedlineDate (usually starts with 4-digit year)
            match = re.search(r'(\d{4})', result["medline_date"])
            if match:
                result["year"] = match.group(1)
    
    # Build date string
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
    """
    Extract abstract with support for structured abstracts.
    Returns both full text and structured sections.
    """
    result = {
        "abstract_text": "",
        "abstract_structured": [],
        "has_structured_abstract": False
    }
    
    abstract_elem = article_elem.find(".//Abstract")
    if abstract_elem is None:
        return result
    
    abstract_parts = []
    
    for abstract_text in abstract_elem.findall("AbstractText"):
        label = abstract_text.get("Label", "")
        nlm_category = abstract_text.get("NlmCategory", "")
        text = "".join(abstract_text.itertext()).strip()
        
        if text:
            if label:
                # Structured abstract section
                result["has_structured_abstract"] = True
                result["abstract_structured"].append({
                    "label": label,
                    "nlm_category": nlm_category,
                    "text": text
                })
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
    
    result["abstract_text"] = " ".join(abstract_parts)
    return result


def extract_journal_info(article_elem: ET.Element) -> Dict[str, Any]:
    """Extract journal information including Title, JournalIssue, ISSN, NLM Unique ID."""
    result = {
        "title": None,
        "iso_abbreviation": None,
        "issn": None,
        "issn_type": None,
        "volume": None,
        "issue": None,
        "pub_date": {},
        "cited_medium": None,
        "nlm_unique_id": None
    }
    
    journal_elem = article_elem.find(".//Journal")
    if journal_elem is None:
        return result
    
    # ISSN
    issn_elem = journal_elem.find("ISSN")
    if issn_elem is not None:
        result["issn"] = issn_elem.text.strip() if issn_elem.text else None
        result["issn_type"] = issn_elem.get("IssnType")
    
    # NLM Unique ID (from MedlineJournalInfo)
    nlm_id_elem = article_elem.find(".//MedlineJournalInfo/NlmUniqueID")
    if nlm_id_elem is not None and nlm_id_elem.text:
        result["nlm_unique_id"] = nlm_id_elem.text.strip()
    
    # Journal Title
    title_elem = journal_elem.find("Title")
    if title_elem is not None and title_elem.text:
        result["title"] = title_elem.text.strip()
    
    # ISO Abbreviation
    iso_elem = journal_elem.find("ISOAbbreviation")
    if iso_elem is not None and iso_elem.text:
        result["iso_abbreviation"] = iso_elem.text.strip()
    
    # Journal Issue
    issue_elem = journal_elem.find("JournalIssue")
    if issue_elem is not None:
        result["cited_medium"] = issue_elem.get("CitedMedium")
        
        volume_elem = issue_elem.find("Volume")
        if volume_elem is not None and volume_elem.text:
            result["volume"] = volume_elem.text.strip()
        
        issue_num_elem = issue_elem.find("Issue")
        if issue_num_elem is not None and issue_num_elem.text:
            result["issue"] = issue_num_elem.text.strip()
        
        # Publication Date
        pub_date_elem = issue_elem.find("PubDate")
        result["pub_date"] = parse_pubmed_date(pub_date_elem)
    
    return result


def extract_article_ids(article_elem: ET.Element) -> Dict[str, Any]:
    """
    Extract article identifiers from ArticleIdList.
    Handles doi, pubmed, pmc, pii, etc.
    """
    result = {
        "pmid": None,
        "doi": None,
        "pmc": None,
        "pii": None,
        "other_ids": {}
    }
    
    # PMID from MedlineCitation (top-level article only, not references)
    pmid_elem = article_elem.find("./MedlineCitation/PMID")
    if pmid_elem is not None and pmid_elem.text:
        result["pmid"] = pmid_elem.text.strip()
    
    # ArticleIdList from PubmedData (top-level article only, not references)
    for article_id in article_elem.findall("./PubmedData/ArticleIdList/ArticleId"):
        id_type = article_id.get("IdType", "").lower()
        id_value = article_id.text.strip() if article_id.text else None
        
        if id_value:
            if id_type == "doi":
                result["doi"] = id_value
            elif id_type == "pubmed":
                result["pmid"] = id_value  # Confirm PMID
            elif id_type == "pmc" or id_type == "pmcid":
                result["pmc"] = id_value
            elif id_type == "pii":
                result["pii"] = id_value
            else:
                result["other_ids"][id_type] = id_value
    
    # Also check ELocationID in the top-level Article
    for eloc_id in article_elem.findall("./MedlineCitation/Article/ELocationID"):
        id_type = eloc_id.get("EIdType", "").lower()
        id_value = eloc_id.text.strip() if eloc_id.text else None
        
        if id_value:
            if id_type == "doi" and not result["doi"]:
                result["doi"] = id_value
            elif id_type == "pii" and not result["pii"]:
                result["pii"] = id_value
    
    return result


def extract_mesh_terms(article_elem: ET.Element) -> List[Dict[str, Any]]:
    """
    Extract MeSH headings with DescriptorName and QualifierName.
    """
    mesh_terms = []
    
    for mesh_heading in article_elem.findall(".//MeshHeading"):
        descriptor = mesh_heading.find("DescriptorName")
        if descriptor is not None and descriptor.text:
            term = {
                "descriptor": descriptor.text.strip(),
                "descriptor_ui": descriptor.get("UI"),
                "major_topic": descriptor.get("MajorTopicYN") == "Y",
                "type": descriptor.get("Type"),
                "qualifiers": []
            }
            
            for qualifier in mesh_heading.findall("QualifierName"):
                if qualifier.text:
                    term["qualifiers"].append({
                        "name": qualifier.text.strip(),
                        "ui": qualifier.get("UI"),
                        "major_topic": qualifier.get("MajorTopicYN") == "Y"
                    })
            
            mesh_terms.append(term)
    
    return mesh_terms


def extract_keywords(article_elem: ET.Element) -> List[Dict[str, Any]]:
    """
    Extract keywords from KeywordList.
    """
    keywords = []
    
    keyword_list = article_elem.find(".//KeywordList")
    if keyword_list is not None:
        owner = keyword_list.get("Owner", "")
        
        for keyword in keyword_list.findall("Keyword"):
            if keyword.text:
                keywords.append({
                    "keyword": keyword.text.strip(),
                    "major_topic": keyword.get("MajorTopicYN") == "Y",
                    "owner": owner
                })
    
    return keywords


def extract_publication_types(article_elem: ET.Element) -> List[Dict[str, Any]]:
    """
    Extract publication types.
    """
    pub_types = []
    
    for pub_type in article_elem.findall(".//PublicationType"):
        if pub_type.text:
            pub_types.append({
                "type": pub_type.text.strip(),
                "ui": pub_type.get("UI")
            })
    
    return pub_types


def extract_article_title(article_elem: ET.Element) -> str:
    """Extract article title, handling any nested elements."""
    title_elem = article_elem.find(".//ArticleTitle")
    if title_elem is not None:
        return "".join(title_elem.itertext()).strip()
    return ""


def extract_article_data(article_elem: ET.Element, min_year: int) -> Optional[Dict]:
    """
    Extract article data from a PubmedArticle XML element.
    Uses whitelist approach - only extracts specified fields.
    
    Whitelist:
    - PMID
    - Data Links (ArticleIdList: doi, pubmed, pmc)
    - Article Title
    - Journal Info (Title, JournalIssue, ISSN)
    - Publication Date (PubDate)
    - Abstract (with structured abstract support)
    - Keywords (KeywordList)
    - MeSH Terms (DescriptorName, QualifierName)
    - Publication Types
    - Government Affiliations (NEW: merged from gov pipeline)
    """
    try:
        # Get ArticleIdList data (PMID, DOI, PMC, etc.)
        article_ids = extract_article_ids(article_elem)
        if not article_ids["pmid"]:
            return None
        
        # Get publication types (needed for filtering)
        pub_types_data = extract_publication_types(article_elem)
        pub_types = [pt["type"] for pt in pub_types_data]
        
        # Check if target article type
        if not is_target_article(pub_types):
            return None
        
        # Get Journal Info (includes publication date)
        journal_info = extract_journal_info(article_elem)
        
        # Get year for filtering
        year_str = journal_info["pub_date"].get("year")
        try:
            year = int(year_str) if year_str else None
        except ValueError:
            year = None
        
        # Filter by year
        if year is None or year < min_year:
            return None
        
        # Get Abstract
        abstract_data = extract_abstract(article_elem)
        
        # Filter out articles without abstracts or with very short abstracts
        abstract_text = abstract_data["abstract_text"]
        if not abstract_text or len(abstract_text) < 50:
            return None
        
        # Get Article Title
        article_title = extract_article_title(article_elem)
        
        # Get MeSH Terms
        mesh_terms = extract_mesh_terms(article_elem)
        
        # Get Keywords
        keywords = extract_keywords(article_elem)
        
        # Map to article_type for reranker
        article_type = map_publication_type(pub_types)
        
        # Extract government affiliations (merged from gov pipeline)
        is_gov_affiliated, gov_agencies = extract_gov_affiliations_from_pubmed_xml(article_elem)
        
        # Build result - NO CHARACTER LIMITS on any field
        return {
            # Identifiers (from ArticleIdList)
            "pmid": article_ids["pmid"],
            "doi": article_ids["doi"],
            "pmc": article_ids["pmc"],
            "pii": article_ids["pii"],
            "other_ids": article_ids["other_ids"],
            
            # Article Title
            "title": article_title,
            
            # Abstract
            "abstract": abstract_text,
            "abstract_structured": abstract_data["abstract_structured"],
            "has_structured_abstract": abstract_data["has_structured_abstract"],
            
            # Journal Info
            "journal": journal_info["title"] or journal_info["iso_abbreviation"] or "",
            "journal_full": {
                "title": journal_info["title"],
                "iso_abbreviation": journal_info["iso_abbreviation"],
                "issn": journal_info["issn"],
                "issn_type": journal_info["issn_type"],
                "volume": journal_info["volume"],
                "issue": journal_info["issue"],
                "cited_medium": journal_info["cited_medium"],
                "nlm_unique_id": journal_info["nlm_unique_id"],
            },
            
            # Publication Date
            "year": year,
            "publication_date": journal_info["pub_date"],
            
            # Classification
            "mesh_terms": mesh_terms,  # Full MeSH structure with qualifiers
            "mesh_terms_flat": [m["descriptor"] for m in mesh_terms],  # Simple list for compatibility
            "keywords": keywords,  # Full keyword structure
            "keywords_flat": [k["keyword"] for k in keywords],  # Simple list for compatibility
            "publication_types": pub_types_data,
            "publication_types_flat": pub_types,  # Simple list for compatibility
            "article_type": article_type,
            
            # Government Affiliation (merged from gov pipeline)
            "is_gov_affiliated": is_gov_affiliated,
            "gov_agencies": gov_agencies,
            
            # Source info - unified source
            "source": "pubmed_abstract",
            "content_type": "abstract",
            "has_full_text": False,
        }
        
    except Exception as e:
        logger.debug(f"Error extracting article: {e}")
        return None


def process_baseline_file(gz_file: Path, min_year: int) -> List[Dict]:
    """Process a single baseline .xml.gz file and extract target articles."""
    articles = []
    
    try:
        with gzip.open(gz_file, 'rt', encoding='utf-8', errors='replace') as f:
            context = ET.iterparse(f, events=('end',))
            
            for event, elem in context:
                if elem.tag == 'PubmedArticle':
                    article_data = extract_article_data(elem, min_year)
                    if article_data:
                        articles.append(article_data)
                    
                    # Clear processed elements to free memory
                    elem.clear()
                    
    except ET.ParseError as e:
        logger.warning(f"XML parse error in {gz_file.name}: {e}")
    except Exception as e:
        logger.error(f"Error processing {gz_file.name}: {e}")
    
    return articles


def filter_baseline(
    output_dir: Path,
    min_year: int = DEFAULT_MIN_YEAR,
    max_workers: int = 4
) -> int:
    """Filter baseline files for target article types."""
    baseline_dir = output_dir / "xml"
    filtered_dir = output_dir / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    
    gz_files = sorted(baseline_dir.glob("*.xml.gz"))
    if not gz_files:
        logger.error(f"No .xml.gz files found in {baseline_dir}")
        return 0
    
    tracker = ProgressTracker(output_dir)
    processed_files = tracker.get_processed_files()
    
    # Filter out already processed files
    pending_files = [f for f in gz_files if f.name not in processed_files]
    
    logger.info("=" * 70)
    logger.info("🔍 Filtering for High-Value Article Types")
    logger.info("=" * 70)
    logger.info(f"Total baseline files: {len(gz_files)}")
    logger.info(f"Already processed: {len(processed_files)}")
    logger.info(f"Pending: {len(pending_files)}")
    logger.info(f"Min year: {min_year}")
    logger.info(f"Target types: {list(TARGET_PUBLICATION_TYPES.keys())}")
    logger.info(f"Workers: {max_workers}")
    logger.info("=" * 70)
    
    tracker.update(stage="filtering")
    
    total_filtered = tracker.stats.get("filtered_articles", 0)
    output_file = filtered_dir / "pubmed_abstracts.jsonl"
    
    # Open in append mode to support resume
    mode = 'a' if output_file.exists() else 'w'
    
    with open(output_file, mode, encoding='utf-8') as out_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_baseline_file, gz_file, min_year): gz_file
                for gz_file in pending_files
            }
            
            with tqdm(total=len(pending_files), desc="Filtering files", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    gz_file = future_to_file[future]
                    try:
                        articles = future.result()
                        for article in articles:
                            out_f.write(json.dumps(article, ensure_ascii=False) + "\n")
                        total_filtered += len(articles)
                        
                        # Update tracker
                        tracker.add_processed_file(gz_file.name)
                        tracker.update(
                            files_processed=len(tracker.stats["processed_files"]),
                            filtered_articles=total_filtered
                        )
                        
                        pbar.set_postfix({
                            "filtered": f"{total_filtered:,}",
                            "rate": f"{len(articles)}/file"
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {gz_file.name}: {e}")
                    
                    pbar.update(1)
    
    tracker.update(stage="complete", filtered_articles=total_filtered)
    
    logger.info(f"\n✅ Filtered {total_filtered:,} high-value articles")
    logger.info(f"   Output: {output_file}")
    
    return total_filtered


def verify_output(output_dir: Path):
    """Verify and print statistics about the output."""
    jsonl_file = output_dir / "filtered" / "pubmed_abstracts.jsonl"
    
    if not jsonl_file.exists():
        logger.error(f"Output file not found: {jsonl_file}")
        return
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 Output Verification")
    logger.info("=" * 70)
    
    total_count = 0
    type_counts = {}
    year_counts = {}
    sample_articles = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_count += 1
            
            if i < 5:
                article = json.loads(line)
                sample_articles.append(article)
            
            # Sample every 1000th for stats
            if i % 1000 == 0:
                article = json.loads(line)
                at = article.get("article_type", "unknown")
                type_counts[at] = type_counts.get(at, 0) + 1
                year = article.get("year")
                if year:
                    year_counts[year] = year_counts.get(year, 0) + 1
    
    logger.info(f"Total articles: {total_count:,}")
    logger.info(f"File size: {jsonl_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Article type distribution (extrapolated)
    logger.info("\nArticle type distribution (sampled):")
    for at, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        estimated = count * 1000
        logger.info(f"  {at}: ~{estimated:,}")
    
    # Recent years
    recent_years = sorted([y for y in year_counts.keys() if y and y >= 2020])
    if recent_years:
        logger.info(f"\nRecent years (sampled, x1000): {dict((y, year_counts[y]) for y in recent_years[-5:])}")
    
    # Sample articles
    logger.info("\nSample articles:")
    for i, article in enumerate(sample_articles[:3], 1):
        logger.info(f"  {i}. [{article['article_type']}] PMID {article['pmid']}: {article['title'][:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Download PubMed baseline and filter for high-value article types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (download + filter)
    python 20_download_pubmed_baseline.py --output-dir /data/ingestion/pubmed_baseline
    
    # Filter only (if already downloaded)
    python 20_download_pubmed_baseline.py --output-dir /data/ingestion/pubmed_baseline --filter-only
    
    # Custom year range
    python 20_download_pubmed_baseline.py --output-dir /data/ingestion/pubmed_baseline --min-year 2020
        """
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Path to existing baseline XML files (e.g., /data/pubmed_gov/baseline). If set, skips download."
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=DEFAULT_MIN_YEAR,
        help=f"Minimum publication year (default: {DEFAULT_MIN_YEAR})"
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Skip download, only run filtering on existing data"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for filtering (default: 4)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output after completion"
    )
    
    args = parser.parse_args()

    ensure_data_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir)
    
    start_time = time.time()
    
    # Handle --baseline-dir: use existing baseline instead of downloading
    if args.baseline_dir:
        if not args.baseline_dir.exists():
            logger.error(f"❌ Baseline directory not found: {args.baseline_dir}")
            sys.exit(1)
        
        # Check for xml.gz files
        gz_files = list(args.baseline_dir.glob("*.xml.gz"))
        if not gz_files:
            logger.error(f"❌ No .xml.gz files found in {args.baseline_dir}")
            sys.exit(1)
        
        logger.info(f"✅ Using existing baseline: {args.baseline_dir}")
        logger.info(f"   Found {len(gz_files)} .xml.gz files")
        
        # Create symlink or copy reference
        xml_dir = args.output_dir / "xml"
        baseline_dir_resolved = args.baseline_dir.resolve()
        
        if xml_dir.exists():
            if xml_dir.is_symlink():
                current_target = xml_dir.readlink()
                if current_target != baseline_dir_resolved:
                    logger.error(f"❌ Symlink {xml_dir} already exists but points to {current_target}")
                    logger.error(f"   Expected: {baseline_dir_resolved}")
                    logger.error(f"   Remove the existing symlink or use the correct --baseline-dir")
                    sys.exit(1)
                # Symlink already points to the correct target, no action needed
            else:
                logger.error(f"❌ Directory {xml_dir} already exists and is not a symlink")
                logger.error(f"   Cannot create symlink to {baseline_dir_resolved}")
                logger.error(f"   Remove the existing directory to use --baseline-dir")
                sys.exit(1)
        else:
            xml_dir.symlink_to(baseline_dir_resolved)
            logger.info(f"   Created symlink: {xml_dir} -> {args.baseline_dir}")
        
        args.filter_only = True  # Force filter-only mode
    
    # Step 1: Download (unless filter-only or using existing baseline)
    if not args.filter_only:
        file_count = download_baseline(args.output_dir)
        if file_count == 0:
            logger.error("No files downloaded, aborting")
            sys.exit(1)
    
    # Step 2: Filter
    total_filtered = filter_baseline(
        args.output_dir,
        min_year=args.min_year,
        max_workers=args.workers
    )
    
    # Step 3: Verify (optional)
    if args.verify:
        verify_output(args.output_dir)
    
    elapsed = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ PubMed Baseline Processing Complete!")
    logger.info("=" * 70)
    logger.info(f"   Filtered articles: {total_filtered:,}")
    logger.info(f"   Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    logger.info(f"   Output: {args.output_dir}/filtered/pubmed_abstracts.jsonl")
    logger.info("\n   Next step: Run ingestion script")
    logger.info(f"   python 21_ingest_pubmed_abstracts.py --input {args.output_dir}/filtered/pubmed_abstracts.jsonl")


if __name__ == "__main__":
    main()
