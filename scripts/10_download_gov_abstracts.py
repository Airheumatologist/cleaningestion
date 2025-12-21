#!/usr/bin/env python3
"""
Download PubMed Government Abstracts (Public Domain).

Downloads abstracts from US government-authored PubMed articles (NIH, CDC, FDA, etc.)
which are in the public domain and can be used commercially.

Two modes:
1. FTP_BASELINE (recommended): Download full PubMed baseline, filter locally
2. EUTILS: Use E-utilities API to fetch only government articles (slower)

Data Sources:
    FTP: ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
    API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/

Usage:
    # Fast mode (recommended) - downloads baseline, filters locally
    python 10_download_gov_abstracts.py --mode ftp --output-dir /data/pubmed_gov/
    
    # API mode (slower but targeted)
    python 10_download_gov_abstracts.py --mode eutils --api-key YOUR_KEY --output-dir /data/pubmed_gov/
    
    # Filter only (if baseline already downloaded)
    python 10_download_gov_abstracts.py --mode ftp --filter-only --output-dir /data/pubmed_gov/

Expected Duration:
    FTP mode: 4-6 hours download + 2-3 hours filtering
    E-utilities mode: 1-2 days (rate limited)

Output:
    /data/pubmed_gov/gov_abstracts/gov_abstracts.jsonl
    
    Each line is a JSON object with:
    - pmid, title, abstract, year, journal, authors, affiliations
    - source: "pubmed_gov"
    - content_type: "abstract"
    - has_full_text: false
"""

import os
import sys
import gzip
import json
import time
import logging
import argparse
import asyncio
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Set
import subprocess
import re

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    os.system("pip3 install aiohttp --quiet")
    import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gov_abstracts_download.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PUBMED_FTP_BASELINE = "ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
PUBMED_FTP_UPDATEFILES = "ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DEFAULT_OUTPUT_DIR = Path("/data/pubmed_gov")

# Government affiliation patterns for filtering (case-insensitive)
GOV_AFFILIATIONS = [
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
    # Location-based
    "bethesda, md",
    "bethesda, maryland",
    "bethesda md",
    "atlanta, ga",  # CDC headquarters
    "silver spring, md",  # FDA headquarters
]

# E-utilities search query for government articles
GOV_SEARCH_QUERY = '''
    "NIH"[gr] OR "NIH"[ad] OR 
    "CDC"[ad] OR "Centers for Disease Control"[ad] OR
    "FDA"[ad] OR "Food and Drug Administration"[ad] OR
    "Veterans Affairs"[ad] OR "VA Medical"[ad] OR
    "National Institutes of Health"[ad] OR
    "National Cancer Institute"[ad] OR
    "National Heart Lung and Blood"[ad] OR
    "National Library of Medicine"[ad] OR
    "National Institute of Allergy"[ad] OR
    "National Institute of Mental Health"[ad]
'''


class ProgressTracker:
    """Track and display progress across multiple stages."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_file = output_dir / "progress.json"
        self.stats = {
            "stage": "init",
            "files_downloaded": 0,
            "files_processed": 0,
            "total_articles_scanned": 0,
            "gov_articles_found": 0,
            "start_time": None,
            "last_update": None
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


class RateLimiter:
    """Async rate limiter for API calls."""
    
    def __init__(self, rate: float = 10.0):
        self.rate = rate
        self.interval = 1.0 / rate
        self.last_call = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            now = time.time()
            wait_time = self.last_call + self.interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = time.time()


class EutilsDownloader:
    """Download government abstracts via E-utilities API."""
    
    def __init__(
        self,
        api_key: str,
        email: str,
        output_dir: Path,
        batch_size: int = 500,
        max_concurrent: int = 5
    ):
        self.api_key = api_key
        self.email = email
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(rate=9.0)
        self.checkpoint_file = output_dir / "eutils_checkpoint.json"
        
    async def search_gov_articles(self, session: aiohttp.ClientSession) -> tuple:
        """Search for government-authored articles."""
        logger.info("Searching for government-authored articles...")
        
        params = {
            "db": "pubmed",
            "term": GOV_SEARCH_QUERY,
            "usehistory": "y",
            "retmode": "json",
            "retmax": 0,
            "api_key": self.api_key,
            "email": self.email,
            "tool": "medical_rag_pipeline"
        }
        
        await self.rate_limiter.acquire()
        async with session.get(f"{EUTILS_BASE}esearch.fcgi", params=params) as resp:
            data = await resp.json()
            
        result = data["esearchresult"]
        count = int(result["count"])
        webenv = result["webenv"]
        query_key = result["querykey"]
        
        logger.info(f"Found {count:,} government-authored articles")
        return count, webenv, query_key
    
    async def fetch_batch(
        self,
        session: aiohttp.ClientSession,
        webenv: str,
        query_key: str,
        retstart: int,
        retmax: int
    ) -> Optional[str]:
        """Fetch a batch of abstracts."""
        params = {
            "db": "pubmed",
            "query_key": query_key,
            "WebEnv": webenv,
            "retstart": retstart,
            "retmax": retmax,
            "rettype": "abstract",
            "retmode": "xml",
            "api_key": self.api_key,
            "email": self.email,
            "tool": "medical_rag_pipeline"
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self.rate_limiter.acquire()
                async with session.get(
                    f"{EUTILS_BASE}efetch.fcgi",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status == 429:
                        wait = 2 ** attempt * 5
                        logger.warning(f"Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    return await resp.text()
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed batch at {retstart}: {e}")
                    return None
        return None
    
    def save_checkpoint(self, completed_batches: Set[int], total_count: int):
        checkpoint = {
            "completed_batches": list(completed_batches),
            "total_count": total_count,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self) -> tuple:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
            return set(data["completed_batches"]), data["total_count"]
        return set(), 0
    
    async def download_all(self):
        """Download all government abstracts via API."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        completed_batches, saved_count = self.load_checkpoint()
        if completed_batches:
            logger.info(f"Resuming from checkpoint: {len(completed_batches)} batches completed")
        
        async with aiohttp.ClientSession() as session:
            total_count, webenv, query_key = await self.search_gov_articles(session)
            
            if saved_count and saved_count != total_count:
                logger.warning(f"Article count changed ({saved_count} -> {total_count}), restarting...")
                completed_batches = set()
            
            total_batches = (total_count + self.batch_size - 1) // self.batch_size
            pending_batches = [i for i in range(total_batches) if i not in completed_batches]
            
            logger.info(f"Total batches: {total_batches}, Pending: {len(pending_batches)}")
            
            xml_dir = self.output_dir / "xml_batches"
            xml_dir.mkdir(exist_ok=True)
            
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def process_batch(batch_idx: int):
                async with semaphore:
                    retstart = batch_idx * self.batch_size
                    xml_content = await self.fetch_batch(
                        session, webenv, query_key, retstart, self.batch_size
                    )
                    if xml_content:
                        batch_file = xml_dir / f"batch_{batch_idx:06d}.xml"
                        with open(batch_file, "w", encoding="utf-8") as f:
                            f.write(xml_content)
                        return batch_idx
                    return None
            
            with tqdm(total=len(pending_batches), desc="Downloading batches") as pbar:
                for i in range(0, len(pending_batches), self.max_concurrent * 2):
                    chunk = pending_batches[i:i + self.max_concurrent * 2]
                    tasks = [process_batch(idx) for idx in chunk]
                    results = await asyncio.gather(*tasks)
                    
                    for result in results:
                        if result is not None:
                            completed_batches.add(result)
                            pbar.update(1)
                    
                    if len(completed_batches) % 100 == 0:
                        self.save_checkpoint(completed_batches, total_count)
            
            self.save_checkpoint(completed_batches, total_count)
            
        logger.info(f"✅ Download complete: {len(completed_batches)} batches")
        return len(completed_batches) * self.batch_size


class FTPBaselineDownloader:
    """Download PubMed baseline via FTP and filter for government articles."""
    
    def __init__(self, output_dir: Path, max_workers: int = 4, include_updates: bool = False):
        self.output_dir = output_dir
        self.baseline_dir = output_dir / "baseline"
        self.filtered_dir = output_dir / "gov_abstracts"
        self.max_workers = max_workers
        self.include_updates = include_updates
        self.tracker = ProgressTracker(output_dir)
        
        # Compile regex patterns for faster matching
        self.gov_patterns = [
            re.compile(re.escape(pattern), re.IGNORECASE) 
            for pattern in GOV_AFFILIATIONS
        ]
    
    def download_baseline(self) -> int:
        """Download PubMed baseline files via FTP using wget."""
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info("📥 Downloading PubMed Baseline via FTP")
        logger.info("=" * 70)
        logger.info(f"Source: {PUBMED_FTP_BASELINE}")
        logger.info(f"Destination: {self.baseline_dir}")
        logger.info("This may take 4-6 hours depending on network speed...")
        logger.info("=" * 70)
        
        self.tracker.update(stage="downloading", start_time=datetime.now().isoformat())
        
        # Use wget for efficient FTP download with resume capability
        cmd = [
            "wget",
            "-r",           # Recursive
            "-np",          # No parent directories  
            "-nd",          # No directory structure (flat)
            "-c",           # Continue/resume interrupted downloads
            "-N",           # Only download newer files
            "-A", "*.xml.gz",  # Only .xml.gz files
            "--progress=bar:force",
            "-P", str(self.baseline_dir),
            PUBMED_FTP_BASELINE
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
                if "%" in line or "saved" in line.lower():
                    print(line.strip())
            
            process.wait()
            
            if process.returncode != 0:
                logger.warning(f"wget exited with code {process.returncode}")
                
        except FileNotFoundError:
            logger.error("wget not found. Please install: sudo apt install wget")
            logger.info("Alternatively, download manually from: " + PUBMED_FTP_BASELINE)
            return 0
        except Exception as e:
            logger.error(f"Download error: {e}")
            return 0
        
        file_count = len(list(self.baseline_dir.glob("*.xml.gz")))
        self.tracker.update(files_downloaded=file_count)
        logger.info(f"✅ Downloaded {file_count} baseline files")
        
        # Optionally download update files
        if self.include_updates:
            logger.info("Downloading update files...")
            cmd[len(cmd)-1] = PUBMED_FTP_UPDATEFILES
            try:
                subprocess.run(cmd, check=False)
            except:
                pass
            file_count = len(list(self.baseline_dir.glob("*.xml.gz")))
            self.tracker.update(files_downloaded=file_count)
        
        return file_count
    
    def is_gov_article(self, article: ET.Element) -> bool:
        """Check if article has government affiliation."""
        # Check all affiliation elements
        for aff in article.findall(".//Affiliation"):
            if aff.text:
                aff_text = aff.text.lower()
                for pattern in self.gov_patterns:
                    if pattern.search(aff_text):
                        return True
        
        # Check AffiliationInfo elements (newer format)
        for aff_info in article.findall(".//AffiliationInfo/Affiliation"):
            if aff_info.text:
                aff_text = aff_info.text.lower()
                for pattern in self.gov_patterns:
                    if pattern.search(aff_text):
                        return True
        
        # Check grant list for NIH/government grants
        for grant in article.findall(".//Grant"):
            agency = grant.find("Agency")
            if agency is not None and agency.text:
                agency_lower = agency.text.lower()
                if any(g in agency_lower for g in ["nih", "national institutes", "cdc", "fda", "va ", "veterans"]):
                    return True
        
        return False
    
    def extract_abstract_data(self, article: ET.Element) -> Optional[Dict]:
        """Extract abstract and metadata from a PubmedArticle element."""
        try:
            # Get PMID
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                return None
            pmid = pmid_elem.text.strip()
            
            # Get abstract - handle structured abstracts
            abstract_parts = article.findall(".//Abstract/AbstractText")
            if not abstract_parts:
                return None
            
            if len(abstract_parts) > 1:
                # Structured abstract with labels
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
            
            # Get publication date
            year = None
            pub_date = article.find(".//PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None and year_elem.text:
                    try:
                        year = int(year_elem.text)
                    except ValueError:
                        pass
                # Try MedlineDate if Year not found
                if year is None:
                    medline_date = pub_date.find("MedlineDate")
                    if medline_date is not None and medline_date.text:
                        match = re.search(r'(\d{4})', medline_date.text)
                        if match:
                            year = int(match.group(1))
            
            # Get journal
            journal = ""
            journal_elem = article.find(".//Journal/Title")
            if journal_elem is not None and journal_elem.text:
                journal = journal_elem.text.strip()
            else:
                # Try ISOAbbreviation
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
            
            # Get DOI if available
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
                "source": "pubmed_gov",
                "content_type": "abstract",
                "has_full_text": False
            }
            
        except Exception as e:
            logger.debug(f"Error extracting article: {e}")
            return None
    
    def process_baseline_file(self, gz_file: Path) -> List[Dict]:
        """Process a single baseline .xml.gz file and extract gov articles."""
        gov_articles = []
        articles_scanned = 0
        
        try:
            with gzip.open(gz_file, 'rt', encoding='utf-8', errors='replace') as f:
                # Use iterparse for memory-efficient processing
                context = ET.iterparse(f, events=('end',))
                
                for event, elem in context:
                    if elem.tag == 'PubmedArticle':
                        articles_scanned += 1
                        
                        if self.is_gov_article(elem):
                            article_data = self.extract_abstract_data(elem)
                            if article_data:
                                gov_articles.append(article_data)
                        
                        # Clear processed elements to free memory
                        elem.clear()
                            
        except ET.ParseError as e:
            logger.warning(f"XML parse error in {gz_file.name}: {e}")
        except Exception as e:
            logger.error(f"Error processing {gz_file.name}: {e}")
        
        return gov_articles
    
    def filter_baseline(self) -> int:
        """Filter baseline files for government articles using parallel processing."""
        self.filtered_dir.mkdir(parents=True, exist_ok=True)
        
        gz_files = sorted(self.baseline_dir.glob("*.xml.gz"))
        if not gz_files:
            logger.error(f"No .xml.gz files found in {self.baseline_dir}")
            return 0
        
        logger.info("=" * 70)
        logger.info("🔍 Filtering for Government Articles")
        logger.info("=" * 70)
        logger.info(f"Processing {len(gz_files)} baseline files...")
        logger.info(f"Using {self.max_workers} parallel workers")
        logger.info("=" * 70)
        
        self.tracker.update(stage="filtering")
        
        total_gov = 0
        processed_files = 0
        output_file = self.filtered_dir / "gov_abstracts.jsonl"
        
        # Process files in parallel, write results sequentially
        with open(output_file, 'w', encoding='utf-8') as out_f:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self.process_baseline_file, gz_file): gz_file
                    for gz_file in gz_files
                }
                
                with tqdm(total=len(gz_files), desc="Filtering files", unit="file") as pbar:
                    for future in as_completed(future_to_file):
                        gz_file = future_to_file[future]
                        try:
                            articles = future.result()
                            for article in articles:
                                out_f.write(json.dumps(article, ensure_ascii=False) + "\n")
                            total_gov += len(articles)
                            processed_files += 1
                            
                            pbar.set_postfix({
                                "gov_found": f"{total_gov:,}",
                                "rate": f"{total_gov/max(processed_files,1):.0f}/file"
                            })
                            
                        except Exception as e:
                            logger.error(f"Error processing {gz_file.name}: {e}")
                        
                        pbar.update(1)
                        
                        # Update tracker periodically
                        if processed_files % 50 == 0:
                            self.tracker.update(
                                files_processed=processed_files,
                                gov_articles_found=total_gov
                            )
        
        self.tracker.update(
            stage="complete",
            files_processed=processed_files,
            gov_articles_found=total_gov
        )
        
        logger.info(f"\n✅ Found {total_gov:,} government articles")
        logger.info(f"   Output: {output_file}")
        
        return total_gov
    
    def download_and_filter(self) -> int:
        """Full pipeline: download baseline then filter for gov articles."""
        # Step 1: Download baseline
        file_count = self.download_baseline()
        if file_count == 0:
            logger.error("No files downloaded, aborting")
            return 0
        
        # Step 2: Filter for government articles
        return self.filter_baseline()


def verify_output(output_dir: Path):
    """Verify and print statistics about the output."""
    jsonl_file = output_dir / "gov_abstracts" / "gov_abstracts.jsonl"
    
    if not jsonl_file.exists():
        logger.error(f"Output file not found: {jsonl_file}")
        return
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 Output Verification")
    logger.info("=" * 70)
    
    total_count = 0
    sample_articles = []
    year_counts = {}
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_count += 1
            if i < 3:
                article = json.loads(line)
                sample_articles.append(article)
            
            # Count by year (sample every 1000th)
            if i % 1000 == 0:
                article = json.loads(line)
                year = article.get("year")
                if year:
                    year_counts[year] = year_counts.get(year, 0) + 1
    
    logger.info(f"Total articles: {total_count:,}")
    logger.info(f"File size: {jsonl_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Year distribution (sampled)
    if year_counts:
        recent_years = sorted([y for y in year_counts.keys() if y and y >= 2020])
        if recent_years:
            logger.info(f"Recent years (sampled): {dict((y, year_counts[y]*1000) for y in recent_years[-5:])}")
    
    # Sample articles
    logger.info("\nSample articles:")
    for i, article in enumerate(sample_articles, 1):
        logger.info(f"  {i}. PMID {article['pmid']}: {article['title'][:80]}...")
        logger.info(f"     Year: {article.get('year')}, Journal: {article.get('journal', 'N/A')[:40]}")


def main():
    parser = argparse.ArgumentParser(
        description="Download PubMed government abstracts (public domain)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download baseline and filter (recommended)
    python 10_download_gov_abstracts.py --mode ftp --output-dir /data/pubmed_gov/
    
    # Filter only (if baseline already downloaded)
    python 10_download_gov_abstracts.py --mode ftp --filter-only
    
    # Use E-utilities API (slower)
    python 10_download_gov_abstracts.py --mode eutils --api-key YOUR_KEY
        """
    )
    parser.add_argument(
        "--mode",
        choices=["ftp", "eutils"],
        default="ftp",
        help="Download mode: 'ftp' (faster, recommended) or 'eutils' (API-based)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: /data/pubmed_gov)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("NCBI_API_KEY"),
        help="NCBI API key (required for eutils mode, get from ncbi.nlm.nih.gov/account)"
    )
    parser.add_argument(
        "--email",
        type=str,
        default=os.environ.get("NCBI_EMAIL", "user@example.com"),
        help="Email for NCBI API identification"
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Only filter existing baseline files (skip download)"
    )
    parser.add_argument(
        "--include-updates",
        action="store_true",
        help="Also download daily update files (in addition to baseline)"
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
        help="Verify output after processing"
    )
    
    args = parser.parse_args()
    
    # Print banner
    logger.info("=" * 70)
    logger.info("🏛️  PubMed Government Abstracts Download")
    logger.info("    Public Domain Content for Commercial Use")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Workers: {args.workers}")
    if args.filter_only:
        logger.info("Filter only: YES (skipping download)")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    count = 0
    
    try:
        if args.mode == "ftp":
            downloader = FTPBaselineDownloader(
                output_dir=args.output_dir,
                max_workers=args.workers,
                include_updates=args.include_updates
            )
            
            if args.filter_only:
                count = downloader.filter_baseline()
            else:
                count = downloader.download_and_filter()
        
        elif args.mode == "eutils":
            if not args.api_key:
                logger.error("❌ API key required for eutils mode")
                logger.info("   Set --api-key or NCBI_API_KEY environment variable")
                logger.info("   Get your API key at: https://www.ncbi.nlm.nih.gov/account/settings/")
                sys.exit(1)
            
            downloader = EutilsDownloader(
                api_key=args.api_key,
                email=args.email,
                output_dir=args.output_dir
            )
            count = asyncio.run(downloader.download_all())
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user. Progress has been saved.")
        logger.info("   Re-run with same arguments to resume.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise
    
    elapsed = datetime.now() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Processing Complete!")
    logger.info("=" * 70)
    logger.info(f"   Government abstracts: {count:,}")
    logger.info(f"   Duration: {elapsed}")
    logger.info(f"   Output: {args.output_dir / 'gov_abstracts' / 'gov_abstracts.jsonl'}")
    logger.info("=" * 70)
    
    # Verify output
    if args.verify or count > 0:
        verify_output(args.output_dir)
    
    logger.info("\n📌 Next steps:")
    logger.info("   1. Ingest into Qdrant: python 11_ingest_gov_abstracts.py")
    logger.info("   2. Update retriever to include new content_type field")


if __name__ == "__main__":
    main()

