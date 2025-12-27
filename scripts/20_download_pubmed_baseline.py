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

Usage:
    # Full pipeline (download + filter)
    python 20_download_pubmed_baseline.py --output-dir /data/pubmed_baseline

    # Filter only (if baseline already downloaded)
    python 20_download_pubmed_baseline.py --output-dir /data/pubmed_baseline --filter-only
    
    # Limit to year range
    python 20_download_pubmed_baseline.py --output-dir /data/pubmed_baseline --min-year 2020

Expected Duration:
    Download: 2-3 hours (31GB compressed)
    Filter: 2-3 hours (~1.8M articles extracted)

Output:
    /data/pubmed_baseline/filtered/pubmed_abstracts.jsonl
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
from typing import Dict, List, Optional, Set
import xml.etree.ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pubmed_baseline_download.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PUBMED_FTP_BASELINE = "ftp.ncbi.nlm.nih.gov::pubmed/baseline/"
DEFAULT_OUTPUT_DIR = Path("/data/pubmed_baseline")
DEFAULT_MIN_YEAR = 2015

# Target publication types (case-insensitive matching)
# These are mapped to article_type values compatible with reranker.py tiers
TARGET_PUBLICATION_TYPES = {
    # TIER_1 types (1.80x boost in reranker)
    "practice guideline": "practice_guideline",
    "meta-analysis": "meta_analysis",
    "systematic review": "systematic_review",
    # TIER_2 types (1.25x boost in reranker)
    "review": "review",
}


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
            # Try wget as fallback
            logger.info("Trying wget as fallback...")
            wget_cmd = [
                "wget", "-r", "-np", "-nd", "-c", "-N",
                "-A", "*.xml.gz",
                "-P", str(baseline_dir),
                f"ftp://{PUBMED_FTP_BASELINE}"
            ]
            subprocess.run(wget_cmd, check=False)
            
    except FileNotFoundError:
        logger.error("rsync not found. Please install: sudo apt install rsync")
        return 0
    except Exception as e:
        logger.error(f"Download error: {e}")
        return 0
    
    file_count = len(list(baseline_dir.glob("*.xml.gz")))
    logger.info(f"✅ Downloaded {file_count} baseline files")
    return file_count


def map_publication_type(pub_types: List[str]) -> str:
    """
    Map PubMed publication types to reranker-compatible article_type.
    
    Priority order ensures highest-value type is returned:
    1. Practice Guideline (TIER_1)
    2. Meta-Analysis (TIER_1)
    3. Systematic Review (TIER_1)
    4. Review (TIER_2)
    """
    pub_types_lower = [pt.lower().strip() for pt in pub_types]
    
    # Check in priority order
    if "practice guideline" in pub_types_lower:
        return "practice_guideline"
    if "meta-analysis" in pub_types_lower:
        return "meta_analysis"
    if "systematic review" in pub_types_lower:
        return "systematic_review"
    if "review" in pub_types_lower:
        return "review"
    
    return None  # Not a target type


def is_target_article(pub_types: List[str]) -> bool:
    """Check if article has any target publication type."""
    pub_types_lower = [pt.lower().strip() for pt in pub_types]
    return any(target in pub_types_lower for target in TARGET_PUBLICATION_TYPES.keys())


def extract_article_data(article: ET.Element, min_year: int) -> Optional[Dict]:
    """Extract article data from a PubmedArticle XML element."""
    try:
        # Get PMID
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            return None
        pmid = pmid_elem.text.strip()
        
        # Get publication types
        pub_type_elems = article.findall(".//PublicationType")
        pub_types = [pt.text.strip() for pt in pub_type_elems if pt.text]
        
        # Check if target article type
        if not is_target_article(pub_types):
            return None
        
        # Get year
        year = None
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                except ValueError:
                    pass
            if year is None:
                medline_date = pub_date.find("MedlineDate")
                if medline_date is not None and medline_date.text:
                    match = re.search(r'(\d{4})', medline_date.text)
                    if match:
                        year = int(match.group(1))
        
        # Filter by year
        if year is None or year < min_year:
            return None
        
        # Get abstract
        abstract_parts = article.findall(".//Abstract/AbstractText")
        if not abstract_parts:
            return None
        
        if len(abstract_parts) > 1:
            # Structured abstract
            abstract_sections = []
            for part in abstract_parts:
                label = part.get("Label", "")
                text = "".join(part.itertext()).strip()
                if text:
                    if label:
                        abstract_sections.append(f"{label}: {text}")
                    else:
                        abstract_sections.append(text)
            abstract_text = " ".join(abstract_sections)
        else:
            abstract_text = "".join(abstract_parts[0].itertext()).strip()
        
        if not abstract_text or len(abstract_text) < 50:
            return None
        
        # Get title
        title_elem = article.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
        
        # Get journal
        journal = ""
        journal_elem = article.find(".//Journal/Title")
        if journal_elem is not None and journal_elem.text:
            journal = journal_elem.text.strip()
        else:
            iso_elem = article.find(".//Journal/ISOAbbreviation")
            if iso_elem is not None and iso_elem.text:
                journal = iso_elem.text.strip()
        
        # Get authors (limit to first 10)
        authors = []
        for author in article.findall(".//Author")[:10]:
            lastname = author.find("LastName")
            forename = author.find("ForeName")
            if lastname is not None and lastname.text:
                name = lastname.text
                if forename is not None and forename.text:
                    name = f"{forename.text} {name}"
                authors.append(name)
        
        # Get affiliations (limit to first 5)
        affiliations = []
        seen_affs = set()
        for aff in article.findall(".//Affiliation")[:10]:
            if aff.text and aff.text not in seen_affs:
                affiliations.append(aff.text.strip())
                seen_affs.add(aff.text)
                if len(affiliations) >= 5:
                    break
        
        # Get DOI
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break
        
        # Get MeSH terms
        mesh_terms = []
        for mesh in article.findall(".//MeshHeading/DescriptorName")[:20]:
            if mesh.text:
                mesh_terms.append(mesh.text)
        
        # Map to article_type for reranker
        article_type = map_publication_type(pub_types)
        
        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract_text,
            "year": year,
            "journal": journal,
            "authors": authors,
            "affiliations": affiliations,
            "doi": doi,
            "mesh_terms": mesh_terms,
            "publication_types": pub_types,
            "article_type": article_type,
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
    python 20_download_pubmed_baseline.py --output-dir /data/pubmed_baseline
    
    # Filter only (if already downloaded)
    python 20_download_pubmed_baseline.py --output-dir /data/pubmed_baseline --filter-only
    
    # Custom year range
    python 20_download_pubmed_baseline.py --output-dir /data/pubmed_baseline --min-year 2020
        """
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: /data/pubmed_baseline)"
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
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
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
        if not xml_dir.exists():
            xml_dir.symlink_to(args.baseline_dir.resolve())
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
