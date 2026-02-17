#!/usr/bin/env python3
"""
Ingest NIH Author Manuscripts to Qdrant with Hybrid Retrieval Support.

Features:
- Dense vectors via DeepInfra API (Qwen/Qwen3-Embedding-0.6B-batch)
- BM25 sparse vectors for hybrid search
- Token-based chunking (2048 tokens)
- Full metadata for filtering and reranking

Usage:
    # Set environment variables
    export DEEPINFRA_API_KEY="your_key"
    export QDRANT_API_KEY="your_key"
    
    # Run ingestion
    python 14_ingest_author_manuscripts.py --xml-dir /data/author_manuscripts/xml/

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
import hashlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET

# Add project root to path for shared config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config_ingestion import IngestionConfig, ensure_data_dirs
from ingestion_utils import Chunker as BaseChunker, EmbeddingProvider, upsert_with_retry, generate_section_id, get_section_weight

# Import enhanced utilities for semantic chunking and validation
try:
    from ingestion_utils_enhanced import SemanticChunker, QualityValidator, ContentDeduplicator
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False

# Import BM25SparseEncoder
spec = importlib.util.find_spec("src.bm25_sparse")
if spec is not None:
    from src.bm25_sparse import BM25SparseEncoder
else:
    BM25SparseEncoder = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
except ImportError:
    os.system("pip3 install qdrant-client --quiet")
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

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
# CONFIGURATION - Uses centralized IngestionConfig
# ============================================================================
QDRANT_URL = IngestionConfig.QDRANT_URL
QDRANT_API_KEY = IngestionConfig.QDRANT_API_KEY or os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = IngestionConfig.COLLECTION_NAME

# Use centralized settings with fallback defaults
BATCH_SIZE = IngestionConfig.BATCH_SIZE
PARALLEL_WORKERS = IngestionConfig.MAX_WORKERS  # Standardized to use config value
MAX_RETRIES = IngestionConfig.MAX_RETRIES
CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "author_manuscript_ingested_ids.txt"
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


# generate_section_id and get_section_weight are imported from ingestion_utils


def create_chunks_from_article(article: Dict[str, Any], chunker: Any, validate_chunks: bool = True) -> List[Dict[str, Any]]:
    """
    Create chunks from article with token-aware chunking.
    Returns list of chunk dicts with metadata.
    
    Args:
        article: The parsed article dictionary
        chunker: The chunker instance (BaseChunker or SemanticChunker)
        validate_chunks: Whether to validate chunk quality before returning
    """
    chunks = []
    
    pmcid = article.get("pmcid", "")
    pmid = article.get("pmid")
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    full_text = article.get("full_text", "")
    
    # Base metadata for all chunks
    base_metadata = {
        "title": title,
        "abstract": abstract,
        "journal": article.get("journal", ""),
        "year": article.get("year"),
        "pmcid": pmcid,
        "pmid": pmid,
        "country": article.get("country"),
        "keywords": article.get("keywords", []),
        "mesh_terms": article.get("mesh_terms", []),
        "authors": article.get("authors", []),
        "first_author": article.get("first_author"),
        "author_count": article.get("author_count", 0),
    }
    
    # 1. Abstract chunk (most important)
    if abstract and len(abstract) > 50:
        abstract_full_text = f"Title: {title}\n\nAbstract: {abstract}"
        section_id = generate_section_id(pmcid, "Abstract")
        
        # Chunk the abstract if needed
        abstract_chunks = chunker.chunk_text(abstract_full_text)
        
        for j, chunk_data in enumerate(abstract_chunks):
            chunk_id = f"{pmcid}_abstract_part{j}" if len(abstract_chunks) > 1 else f"{pmcid}_abstract"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": pmcid,
                "text": chunk_data["text"],
                "section_type": "abstract",
                "section_title": "Abstract",
                "full_section_text": abstract_full_text,
                "section_id": section_id,
                "section_weight": get_section_weight("abstract"),
                **base_metadata,
            })
    
    # 2. Full text chunks if available
    if full_text and len(full_text) > 100:
        full_text_with_title = f"Title: {title}\n\n{full_text}"
        section_id = generate_section_id(pmcid, "Full Text")
        
        text_chunks = chunker.chunk_text(full_text_with_title)
        
        for j, chunk_data in enumerate(text_chunks):
            chunk_id = f"{pmcid}_text_part{j}" if len(text_chunks) > 1 else f"{pmcid}_text"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": pmcid,
                "text": chunk_data["text"],
                "section_type": "body",
                "section_title": "Full Text",
                "full_section_text": full_text_with_title,
                "section_id": section_id,
                "section_weight": get_section_weight("body"),
                **base_metadata,
            })
    
    # 3. Validate chunks if enabled
    if validate_chunks and ENHANCED_UTILS_AVAILABLE:
        original_count = len(chunks)
        valid_chunks = []
        for chunk in chunks:
            is_valid, issues = QualityValidator.validate_chunk(
                chunk["text"],
                {k: chunk.get(k) for k in ["doc_id", "chunk_id", "title", "pmid", "year"]}
            )
            if is_valid:
                valid_chunks.append(chunk)
            else:
                logger.debug("Skipping invalid chunk %s: %s", chunk.get("chunk_id"), issues)
        chunks = valid_chunks
        if len(chunks) < original_count:
            logger.debug("Validated chunks: %d valid out of %d total", len(chunks), original_count)
    
    return chunks


def build_points(
    articles: List[Dict[str, Any]], 
    embedding_provider: EmbeddingProvider,
    chunker: Chunker,
    sparse_encoder: Optional[BM25SparseEncoder]
) -> Tuple[List[PointStruct], List[str]]:
    """
    Build Qdrant points from articles with chunking and embeddings.
    """
    points: List[PointStruct] = []
    doc_ids: List[str] = []
    
    # Collect all chunks for batch embedding
    all_chunks: List[Dict[str, Any]] = []
    
    for article in articles:
        chunks = create_chunks_from_article(article, chunker)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        return [], []
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in all_chunks]
    
    # Generate embeddings
    try:
        vectors = embedding_provider.embed_batch(texts)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return [], []
    
    # Create points
    for chunk, vector in zip(all_chunks, vectors):
        chunk_id = chunk["chunk_id"]
        doc_id = chunk["doc_id"]
        
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
        
        # Create payload
        payload = {
            # Core identifiers
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "pmcid": chunk.get("pmcid"),
            "pmid": chunk.get("pmid"),
            "doi": chunk.get("doi"),
            
            # CRITICAL: Full text for retriever
            "page_content": chunk["text"],
            
            # Article metadata
            "title": chunk.get("title", ""),
            "abstract": chunk.get("abstract", ""),
            "journal": chunk.get("journal", ""),
            "year": chunk.get("year"),
            "country": chunk.get("country", ""),
            "keywords": chunk.get("keywords", []),
            "mesh_terms": chunk.get("mesh_terms", []),
            "authors": chunk.get("authors", []),
            "first_author": chunk.get("first_author"),
            "author_count": chunk.get("author_count", 0),
            
            # Section information
            "section_type": chunk["section_type"],
            "section_title": chunk["section_title"],
            
            # Parent-child indexing fields
            "full_section_text": chunk.get("full_section_text", chunk["text"]),
            "section_id": chunk.get("section_id", ""),
            "section_weight": chunk.get("section_weight", 0.5),
            
            # Source and type
            "source": "pmc_author_manuscript",
            "article_type": chunk.get("section_type", "other"),
            "has_full_text": chunk.get("has_full_text", True),
            
            # Preview for dashboard
            "text_preview": chunk["text"][:500],
        }
        
        # Create vector data
        vector_data: Any = {"dense": vector}
        if sparse_encoder is not None:
            sparse_vector = sparse_encoder.encode_document(chunk["text"])
            vector_data["sparse"] = sparse_vector
        
        # Create deterministic point ID
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"author_ms:{chunk_id}"))
        
        points.append(PointStruct(id=point_id, vector=vector_data, payload=payload))
    
    return points, doc_ids


def load_checkpoint() -> set[str]:
    """Load checkpoint of ingested PMCIDs."""
    if CHECKPOINT_FILE.exists():
        try:
            return set(CHECKPOINT_FILE.read_text().strip().split('\n'))
        except:
            return set()
    return set()


def append_checkpoint(ids: List[str]) -> None:
    """Append IDs to checkpoint."""
    with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
        for id_val in ids:
            f.write(f"{id_val}\n")


def upsert_batch(
    client: QdrantClient,
    points: List[PointStruct],
    ids: List[str],
    counters: Counters
) -> bool:
    """Upsert batch with retry logic."""
    try:
        upsert_with_retry(client, points)
        append_checkpoint(ids)
        counters.increment('success', len(points))
        return True
    except Exception as e:
        logger.error(f"Batch failed: {str(e)[:200]}")
        counters.increment('errors', len(points))
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
    """Run the ingestion process with DeepInfra embeddings."""
    
    # Use provided values
    global BATCH_SIZE, PARALLEL_WORKERS
    BATCH_SIZE = batch_size
    PARALLEL_WORKERS = workers
    
    logger.info("=" * 70)
    logger.info("🚀 NIH Author Manuscript Ingestion")
    logger.info("   Dense Vectors via DeepInfra API")
    logger.info("   BM25 Sparse Vectors for Hybrid Search")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        logger.info("   Set: export QDRANT_API_KEY='your_key'")
        sys.exit(1)
    
    if not os.getenv("DEEPINFRA_API_KEY"):
        logger.error("❌ DEEPINFRA_API_KEY not set!")
        logger.info("   Set: export DEEPINFRA_API_KEY='your_key'")
        sys.exit(1)
    
    if not xml_dir.exists():
        logger.error(f"❌ XML directory not found: {xml_dir}")
        sys.exit(1)
    
    # Initialize embedding provider
    embedding_provider = EmbeddingProvider()
    
    # Initialize chunker (uses SemanticChunker if available)
    ChunkerClass = SemanticChunker if ENHANCED_UTILS_AVAILABLE else BaseChunker
    chunker = ChunkerClass(
        chunk_size=IngestionConfig.CHUNK_SIZE_TOKENS,
        overlap=IngestionConfig.CHUNK_OVERLAP_TOKENS
    )
    logger.info(f"Using {ChunkerClass.__name__} for chunking")
    
    # Initialize sparse encoder
    sparse_encoder = None
    if IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
        if BM25SparseEncoder is not None:
            sparse_encoder = BM25SparseEncoder(
                max_terms_doc=IngestionConfig.SPARSE_MAX_TERMS_DOC,
                max_terms_query=IngestionConfig.SPARSE_MAX_TERMS_QUERY,
                min_token_len=IngestionConfig.SPARSE_MIN_TOKEN_LEN,
                remove_stopwords=IngestionConfig.SPARSE_REMOVE_STOPWORDS,
            )
            logger.info("✅ BM25 sparse encoder initialized")
        else:
            logger.warning("⚠️ BM25SparseEncoder not available")
    
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,
        prefer_grpc=IngestionConfig.USE_GRPC,
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
    logger.info(f"   Embedding provider: DeepInfra")
    logger.info(f"   Embedding model: {IngestionConfig.EMBEDDING_MODEL}")
    logger.info(f"   Chunk size: {IngestionConfig.CHUNK_SIZE_TOKENS} tokens")
    logger.info(f"   Chunk overlap: {IngestionConfig.CHUNK_OVERLAP_TOKENS} tokens")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Parallel workers: {PARALLEL_WORKERS}")
    logger.info(f"   Data directory: {IngestionConfig.DATA_DIR}")
    
    # Get checkpoint
    counters = Counters()
    ingested = load_checkpoint()
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
    current_batch: List[Dict[str, Any]] = []
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
                    
                    # Build points with embeddings
                    points, ids = build_points(batch, embedding_provider, chunker, sparse_encoder)
                    
                    if points:
                        future = executor.submit(upsert_batch, client, points, ids, counters)
                        pending_futures.append(future)
                    
                    batch_num += 1
                    
                    # Clean up completed futures
                    pending_futures = [f for f in pending_futures if not f.done()]
                    
                    if batch_num % PROGRESS_LOG_INTERVAL == 0:
                        rate = counters.get_rate()
                        logger.info(
                            f"Progress: {counters.success:,} points ingested, "
                            f"{counters.errors:,} errors, "
                            f"{counters.parse_errors:,} parse errors, "
                            f"{rate:.1f} pts/sec"
                        )
            
            # Process remaining batch
            if current_batch:
                points, ids = build_points(current_batch, embedding_provider, chunker, sparse_encoder)
                
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
    logger.info(f"   Ingested: {counters.success:,} points")
    logger.info(f"   Skipped (already done): {counters.skipped:,}")
    logger.info(f"   Parse errors: {counters.parse_errors:,}")
    logger.info(f"   Upload errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes")
    logger.info(f"   Rate: {counters.get_rate():.1f} pts/sec")
    logger.info(f"   Collection total: {final_count:,}")
    logger.info("=" * 70)


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
        help=f"Batch size for upserts (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=PARALLEL_WORKERS,
        help=f"Number of parallel workers (default: {PARALLEL_WORKERS})"
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
