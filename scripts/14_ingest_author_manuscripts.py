#!/usr/bin/env python3
"""
Ingest NIH Author Manuscripts to Qdrant with Hybrid Retrieval Support.

Features:
- Dense vectors via Qdrant Cloud Inference (mixedbread-ai/mxbai-embed-large-v1)
- Full text storage for RAG context
- All metadata for filtering and reranking
- Compatible with existing SPLADE sparse vector pipeline

The SPLADE sparse vectors are added in a second pass using 11_add_splade_vectors.py
to enable hybrid retrieval with binary quantization.

Collection Setup:
- Dense: 1024-dim with Binary Quantization
- Sparse: SPLADE vectors (added separately)
- 4 shards for parallel processing

Usage:
    # Set environment variables
    export QDRANT_API_KEY="your_api_key"
    
    # Run ingestion
    python 14_ingest_author_manuscripts.py --xml-dir /data/author_manuscripts/xml/
    
    # After ingestion, add SPLADE vectors:
    python 11_add_splade_vectors.py

Expected Duration: 2-4 hours for ~900K manuscripts
"""

import os
import sys
import json
import logging
import uuid
import time
import argparse
import threading
import re
import gzip
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Document
except ImportError:
    os.system("pip3 install qdrant-client --quiet")
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/author_manuscript_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Matches existing PMC ingestion setup
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://cf6c28ca-8a2a-43fa-9424-1f2af9e9a5f3.us-east-1-1.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pmc_medical_rag_fulltext"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# Cloud Inference optimal settings
BATCH_SIZE = 50  # Cloud Inference limit per request
PARALLEL_WORKERS = 4  # Match shard count
MAX_TEXT_LENGTH = 2000  # Cloud Inference embedding text limit
MAX_RETRIES = 3
CHECKPOINT_FILE = Path("/data/author_manuscript_checkpoint.txt")
PROGRESS_LOG_INTERVAL = 100

# Evidence classification for reranking
ARTICLE_TYPE_MAPPING = {
    "research-article": "research_article",
    "review-article": "review_article",
    "case-report": "case_report",
    "editorial": "editorial",
    "letter": "letter",
    "brief-report": "brief_report",
    "methods": "methods_article",
    "protocol": "protocol",
}

# ============================================================================


class Counters:
    """Thread-safe counters for progress tracking."""
    
    def __init__(self):
        self.success = 0
        self.errors = 0
        self.skipped = 0
        self.parse_errors = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def increment(self, field: str, count: int = 1):
        with self.lock:
            setattr(self, field, getattr(self, field) + count)
    
    def get_rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.success / elapsed if elapsed > 0 else 0


def get_text(element: Optional[ET.Element], default: str = "") -> str:
    """Safely extract all text from XML element including nested elements."""
    if element is None:
        return default
    return " ".join("".join(element.itertext()).split()).strip() or default


def detect_article_type(root: ET.Element) -> str:
    """Detect article type from XML."""
    # Check article-type attribute
    article_type = root.get("article-type", "")
    if article_type:
        return ARTICLE_TYPE_MAPPING.get(article_type, "research_article")
    
    # Check subject groups
    for subject in root.findall(".//subject"):
        subj_text = get_text(subject).lower()
        if "review" in subj_text:
            return "review_article"
        if "case" in subj_text:
            return "case_report"
        if "trial" in subj_text:
            return "clinical_trial"
        if "meta-analysis" in subj_text:
            return "meta_analysis"
        if "systematic" in subj_text:
            return "systematic_review"
    
    return "research_article"


def detect_evidence_grade(article_type: str, abstract: str) -> str:
    """Assign evidence grade based on article type and content."""
    abstract_lower = abstract.lower() if abstract else ""
    
    # Highest evidence
    if article_type in ["systematic_review", "meta_analysis"]:
        return "A"
    if article_type == "guideline":
        return "A"
    
    # High evidence
    if article_type in ["clinical_trial", "randomized_controlled_trial"]:
        return "B"
    if "randomized" in abstract_lower and "trial" in abstract_lower:
        return "B"
    if "rct" in abstract_lower or "randomised" in abstract_lower:
        return "B"
    
    # Moderate evidence
    if article_type in ["cohort_study", "review_article"]:
        return "C"
    if "cohort" in abstract_lower or "prospective" in abstract_lower:
        return "C"
    
    # Lower evidence
    if article_type in ["case_report", "case_series"]:
        return "D"
    
    return "C"  # Default to moderate


def parse_author_manuscript_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a JATS XML author manuscript file.
    
    Extracts all metadata and content matching existing PMC schema.
    """
    try:
        # Handle both regular and gzipped files
        if str(xml_path).endswith('.gz'):
            with gzip.open(xml_path, 'rt', encoding='utf-8', errors='replace') as f:
                tree = ET.parse(f)
        else:
            tree = ET.parse(xml_path)
        
        root = tree.getroot()
        
        # Extract PMCID from filename
        pmcid = xml_path.stem.replace('.xml', '').replace('.gz', '')
        if not pmcid.startswith('PMC'):
            pmcid = f"PMC{pmcid}"
        
        # Get front matter
        front = root.find('.//front')
        if front is None:
            front = root
        
        article_meta = front.find('.//article-meta')
        if article_meta is None:
            article_meta = front
        
        # === IDENTIFIERS ===
        pmid = None
        doi = None
        for article_id in article_meta.findall('.//article-id'):
            id_type = article_id.get('pub-id-type')
            if id_type == 'pmid' and article_id.text:
                pmid = article_id.text.strip()
            elif id_type == 'doi' and article_id.text:
                doi = article_id.text.strip()
        
        # === TITLE ===
        title = ""
        title_group = article_meta.find('.//title-group')
        if title_group is not None:
            title_elem = title_group.find('article-title')
            if title_elem is not None:
                title = get_text(title_elem)
        if not title:
            title_elem = article_meta.find('.//article-title')
            title = get_text(title_elem)
        
        # === ABSTRACT ===
        abstract = ""
        abstract_elem = article_meta.find('.//abstract')
        if abstract_elem is not None:
            # Handle structured abstracts
            sections = abstract_elem.findall('.//sec')
            if sections:
                abstract_parts = []
                for sec in sections:
                    sec_title = get_text(sec.find('title'))
                    paragraphs = [get_text(p) for p in sec.findall('.//p')]
                    sec_text = " ".join(paragraphs)
                    if sec_title and sec_text:
                        abstract_parts.append(f"{sec_title}: {sec_text}")
                    elif sec_text:
                        abstract_parts.append(sec_text)
                abstract = " ".join(abstract_parts)
            else:
                # Simple abstract
                paragraphs = abstract_elem.findall('.//p')
                if paragraphs:
                    abstract = " ".join(get_text(p) for p in paragraphs)
                else:
                    abstract = get_text(abstract_elem)
        
        # === PUBLICATION DATE ===
        year = None
        pub_date = article_meta.find('.//pub-date')
        if pub_date is not None:
            year_elem = pub_date.find('year')
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text.strip())
                except ValueError:
                    pass
        
        # === JOURNAL ===
        journal = ""
        journal_meta = front.find('.//journal-meta')
        if journal_meta is not None:
            for elem_name in ['journal-title', 'journal-id', 'abbrev-journal-title']:
                elem = journal_meta.find(f'.//{elem_name}')
                if elem is not None and elem.text:
                    journal = elem.text.strip()
                    break
        
        # === AUTHORS ===
        authors = []
        first_author = None
        institutions = set()
        country = None
        
        for contrib in article_meta.findall('.//contrib[@contrib-type="author"]'):
            name = contrib.find('name')
            if name is not None:
                surname = get_text(name.find('surname'))
                given = get_text(name.find('given-names'))
                if surname:
                    full_name = f"{given} {surname}".strip() if given else surname
                    authors.append(full_name)
                    if first_author is None:
                        first_author = full_name
            
            # Get affiliations
            for aff in contrib.findall('.//aff'):
                inst = get_text(aff.find('institution'))
                if inst:
                    institutions.add(inst)
                addr = aff.find('.//country')
                if addr is not None and addr.text:
                    country = addr.text.strip()
        
        # Also check standalone affiliations
        for aff in article_meta.findall('.//aff'):
            inst = get_text(aff.find('institution'))
            if inst:
                institutions.add(inst)
            if not country:
                addr = aff.find('.//country')
                if addr is not None and addr.text:
                    country = addr.text.strip()
        
        # === KEYWORDS ===
        keywords = []
        for kwd in article_meta.findall('.//kwd'):
            kw = get_text(kwd)
            if kw and len(kw) > 2:
                keywords.append(kw)
        
        # === MESH TERMS ===
        mesh_terms = []
        for mesh in root.findall('.//MeshHeading/DescriptorName'):
            term = get_text(mesh)
            if term:
                mesh_terms.append(term)
        
        # === FULL TEXT (BODY) ===
        full_text = ""
        has_methods = False
        has_results = False
        body = root.find('.//body')
        if body is not None:
            paragraphs = []
            for sec in body.findall('.//sec'):
                sec_type = sec.get('sec-type', '').lower()
                sec_title = get_text(sec.find('title')).lower()
                
                if 'method' in sec_type or 'method' in sec_title:
                    has_methods = True
                if 'result' in sec_type or 'result' in sec_title:
                    has_results = True
                
                for p in sec.findall('.//p'):
                    text = get_text(p)
                    if text and len(text) > 30:
                        paragraphs.append(text)
            
            # Also get paragraphs not in sections
            for p in body.findall('./p'):
                text = get_text(p)
                if text and len(text) > 30:
                    paragraphs.append(text)
            
            full_text = " ".join(paragraphs[:100])  # Limit paragraphs
        
        # === TABLES AND FIGURES ===
        table_count = len(body.findall('.//table-wrap')) if body is not None else 0
        figure_count = len(body.findall('.//fig')) if body is not None else 0
        
        # === ARTICLE TYPE & EVIDENCE ===
        article_type = detect_article_type(root)
        evidence_grade = detect_evidence_grade(article_type, abstract)
        
        # === PUBLICATION TYPES ===
        publication_types = []
        for pub_type in article_meta.findall('.//article-categories//subj-group/subject'):
            pt = get_text(pub_type)
            if pt:
                publication_types.append(pt)
        
        # Skip if no meaningful content
        if not title and not abstract and not full_text:
            return None
        
        return {
            # Identifiers
            "pmcid": pmcid,
            "pmid": pmid,
            "doi": doi,
            
            # Content (stored for RAG context)
            "title": title[:500] if title else "",
            "abstract": abstract[:5000] if abstract else "",
            "full_text": full_text[:20000] if full_text else "",
            
            # Publication metadata
            "year": year,
            "journal": journal[:200] if journal else "",
            "article_type": article_type,
            "publication_type": publication_types[:5],
            
            # Evidence signals for reranking
            "evidence_grade": evidence_grade,
            "evidence_level": {"A": 1, "B": 2, "C": 3, "D": 4}.get(evidence_grade, 3),
            
            # Authorship
            "authors": authors[:10],
            "first_author": first_author,
            "author_count": len(authors),
            "institutions": list(institutions)[:5],
            "country": country,
            
            # Subject classification
            "keywords": keywords[:15],
            "mesh_terms": mesh_terms[:20],
            
            # Structure signals
            "source": "pmc_author_manuscript",
            "content_type": "author_manuscript",
            "has_full_text": bool(full_text and len(full_text) > 100),
            "has_methods": has_methods,
            "has_results": has_results,
            "table_count": table_count,
            "figure_count": figure_count,
        }
        
    except ET.ParseError as e:
        logger.debug(f"XML parse error in {xml_path.name}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Error processing {xml_path.name}: {e}")
        return None


def create_embedding_text(article: Dict[str, Any]) -> str:
    """
    Create text for dense embedding (max 2000 chars for Cloud Inference).
    
    Combines title + abstract + beginning of full text.
    """
    title = article.get("title", "") or ""
    abstract = article.get("abstract", "") or ""
    full_text = article.get("full_text", "") or ""
    
    # Start with title and abstract
    combined = f"{title}. {abstract}"
    
    # Add as much full text as fits
    if full_text and len(combined) < MAX_TEXT_LENGTH - 100:
        remaining = MAX_TEXT_LENGTH - len(combined) - 10
        combined = f"{combined}\n\n{full_text[:remaining]}"
    
    return combined[:MAX_TEXT_LENGTH]


def create_payload(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create payload with all metadata for filtering and reranking.
    
    Matches existing PMC payload schema for consistency.
    """
    return {
        # Identifiers
        "pmcid": article.get("pmcid"),
        "pmid": article.get("pmid"),
        "doi": article.get("doi"),
        
        # Content (truncated for payload size limits)
        "title": article.get("title", "")[:300],
        "abstract": article.get("abstract", "")[:2000],
        "full_text": article.get("full_text", "")[:15000],  # Store for RAG context
        
        # Publication info
        "year": article.get("year"),
        "journal": article.get("journal", ""),
        "article_type": article.get("article_type", "research_article"),
        "publication_type": article.get("publication_type", [])[:5],
        
        # Evidence signals for reranking
        "evidence_grade": article.get("evidence_grade"),
        "evidence_level": article.get("evidence_level"),
        "country": article.get("country"),
        "institutions": article.get("institutions", [])[:5],
        
        # Subject classification
        "keywords": article.get("keywords", [])[:10],
        "mesh_terms": article.get("mesh_terms", [])[:15],
        
        # Authors
        "authors": article.get("authors", [])[:5],
        "first_author": article.get("first_author"),
        "author_count": article.get("author_count", 0),
        
        # Structure signals
        "source": "pmc_author_manuscript",
        "content_type": "author_manuscript",
        "has_full_text": article.get("has_full_text", True),
        "has_methods": article.get("has_methods", False),
        "has_results": article.get("has_results", False),
        "table_count": article.get("table_count", 0),
        "figure_count": article.get("figure_count", 0),
    }


def get_checkpoint() -> set:
    """Load checkpoint of ingested PMCIDs."""
    if CHECKPOINT_FILE.exists():
        try:
            return set(CHECKPOINT_FILE.read_text().strip().split('\n'))
        except:
            return set()
    return set()


def save_checkpoint(ids: List[str]):
    """Append IDs to checkpoint."""
    with open(CHECKPOINT_FILE, 'a') as f:
        for id_val in ids:
            f.write(f"{id_val}\n")


def upsert_batch(
    client: QdrantClient,
    points: List[PointStruct],
    ids: List[str],
    counters: Counters
) -> bool:
    """Upsert batch with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=False  # Async for throughput
            )
            save_checkpoint(ids)
            counters.increment('success', len(points))
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES}: {str(e)[:100]}")
                time.sleep(wait_time)
            else:
                logger.error(f"Batch failed after {MAX_RETRIES} attempts: {str(e)[:200]}")
                counters.increment('errors', len(points))
                return False
    return False


def process_xml_batch(xml_files: List[Path], ingested: set, counters: Counters) -> List[Dict[str, Any]]:
    """Process a batch of XML files into articles."""
    articles = []
    for xml_path in xml_files:
        pmcid = xml_path.stem.replace('.xml', '').replace('.gz', '')
        if not pmcid.startswith('PMC'):
            pmcid = f"PMC{pmcid}"
        
        if pmcid in ingested:
            counters.increment('skipped')
            continue
        
        article = parse_author_manuscript_xml(xml_path)
        if article:
            articles.append(article)
        else:
            counters.increment('parse_errors')
    
    return articles


def run_ingestion(xml_dir: Path, limit: Optional[int] = None, batch_size: int = 50, workers: int = 4):
    """Run the ingestion process with Cloud Inference."""
    
    # Use provided values
    global BATCH_SIZE, PARALLEL_WORKERS
    BATCH_SIZE = batch_size
    PARALLEL_WORKERS = workers
    
    logger.info("=" * 70)
    logger.info("🚀 NIH Author Manuscript Ingestion")
    logger.info("   Dense Vectors via Qdrant Cloud Inference")
    logger.info("   (SPLADE sparse vectors added separately)")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        logger.info("   Set: export QDRANT_API_KEY='your_key'")
        sys.exit(1)
    
    if not xml_dir.exists():
        logger.error(f"❌ XML directory not found: {xml_dir}")
        sys.exit(1)
    
    # Connect to Qdrant with Cloud Inference enabled
    # Enable Cloud Inference for server-side embeddings
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,
        cloud_inference=True  # Required for Document() vector type
    )
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to Qdrant")
        logger.info(f"   URL: {QDRANT_URL}")
        logger.info(f"   Collection: {COLLECTION_NAME}")
        logger.info(f"   Current points: {info.points_count:,}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Parallel workers: {PARALLEL_WORKERS}")
    logger.info(f"   Max text length: {MAX_TEXT_LENGTH}")
    
    # Get checkpoint
    counters = Counters()
    ingested = get_checkpoint()
    logger.info(f"   Already ingested: {len(ingested):,}")
    
    # Find XML files
    logger.info(f"\n📂 Scanning XML directory: {xml_dir}")
    xml_files = list(xml_dir.glob("*.xml"))
    if not xml_files:
        xml_files = list(xml_dir.glob("**/*.xml"))
    
    total_files = len(xml_files)
    logger.info(f"   Found {total_files:,} XML files")
    
    if limit:
        xml_files = xml_files[:limit]
        logger.info(f"   Limited to {limit:,} files")
    
    # Skip already ingested
    remaining_files = []
    for f in xml_files:
        pmcid = f.stem.replace('.xml', '').replace('.gz', '')
        if not pmcid.startswith('PMC'):
            pmcid = f"PMC{pmcid}"
        if pmcid not in ingested:
            remaining_files.append(f)
    
    logger.info(f"   Remaining to process: {len(remaining_files):,}")
    
    if not remaining_files:
        logger.info("✅ All files already ingested!")
        return
    
    logger.info(f"\n🚀 Starting ingestion...")
    
    # Process in chunks for memory efficiency
    PARSE_BATCH_SIZE = 200
    current_batch = []
    batch_num = 0
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        pending_futures = []
        
        with tqdm(total=len(remaining_files), desc="Ingesting", unit="file") as pbar:
            for i in range(0, len(remaining_files), PARSE_BATCH_SIZE):
                # Parse a chunk of XML files
                chunk = remaining_files[i:i + PARSE_BATCH_SIZE]
                articles = process_xml_batch(chunk, ingested, counters)
                
                pbar.update(len(chunk))
                
                if not articles:
                    continue
                
                # Add to current batch
                current_batch.extend(articles)
                
                # Submit batches when we have enough
                while len(current_batch) >= BATCH_SIZE:
                    batch = current_batch[:BATCH_SIZE]
                    current_batch = current_batch[BATCH_SIZE:]
                    
                    # Create points with Cloud Inference embedding
                    points = []
                    ids = []
                    for article in batch:
                        doc_id = article.get("pmcid") or article.get("pmid")
                        if not doc_id:
                            continue
                        
                        embedding_text = create_embedding_text(article)
                        if len(embedding_text) < 50:
                            continue
                        
                        # Generate deterministic UUID from doc_id
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))
                        payload = create_payload(article)
                        
                        # Use Document for Cloud Inference
                        point = PointStruct(
                            id=point_id,
                            vector=Document(text=embedding_text, model=EMBEDDING_MODEL),
                            payload=payload
                        )
                        points.append(point)
                        ids.append(str(doc_id))
                    
                    if points:
                        future = executor.submit(upsert_batch, client, points, ids, counters)
                        pending_futures.append(future)
                    
                    batch_num += 1
                    
                    # Clean up completed futures
                    pending_futures = [f for f in pending_futures if not f.done()]
                    
                    if batch_num % PROGRESS_LOG_INTERVAL == 0:
                        rate = counters.get_rate()
                        logger.info(
                            f"Progress: {counters.success:,} ingested, "
                            f"{counters.errors:,} errors, "
                            f"{counters.parse_errors:,} parse errors, "
                            f"{rate:.1f}/sec"
                        )
            
            # Process remaining batch
            if current_batch:
                points = []
                ids = []
                for article in current_batch:
                    doc_id = article.get("pmcid") or article.get("pmid")
                    if not doc_id:
                        continue
                    
                    embedding_text = create_embedding_text(article)
                    if len(embedding_text) < 50:
                        continue
                    
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))
                    payload = create_payload(article)
                    
                    point = PointStruct(
                        id=point_id,
                        vector=Document(text=embedding_text, model=EMBEDDING_MODEL),
                        payload=payload
                    )
                    points.append(point)
                    ids.append(str(doc_id))
                
                if points:
                    future = executor.submit(upsert_batch, client, points, ids, counters)
                    pending_futures.append(future)
        
        # Wait for all pending uploads
        logger.info(f"Waiting for {len(pending_futures)} pending uploads...")
        for future in as_completed(pending_futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Future error: {e}")
    
    # Final stats
    elapsed = time.time() - counters.start_time
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        final_count = info.points_count
    except:
        final_count = "N/A"
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Author Manuscript Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"📊 Results:")
    logger.info(f"   Ingested: {counters.success:,}")
    logger.info(f"   Skipped (already done): {counters.skipped:,}")
    logger.info(f"   Parse errors: {counters.parse_errors:,}")
    logger.info(f"   Upload errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes")
    logger.info(f"   Rate: {counters.get_rate():.1f} docs/sec")
    logger.info(f"   Collection total: {final_count:,}")
    logger.info("=" * 70)
    
    logger.info("\n📌 Next steps:")
    logger.info("   1. Add SPLADE sparse vectors: python3 11_add_splade_vectors.py")
    logger.info("   2. Verify with test query")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest NIH Author Manuscripts to Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full ingestion
    python 14_ingest_author_manuscripts.py --xml-dir /data/author_manuscripts/xml/
    
    # Test with 1000 files
    python 14_ingest_author_manuscripts.py --xml-dir /data/author_manuscripts/xml/ --limit 1000
    
    # Adjust batch size
    python 14_ingest_author_manuscripts.py --batch-size 25 --workers 2
        """
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=Path("/data/author_manuscripts/xml"),
        help="Directory containing XML files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for upserts (default: 50)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=PARALLEL_WORKERS,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    run_ingestion(
        xml_dir=args.xml_dir, 
        limit=args.limit,
        batch_size=args.batch_size,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
