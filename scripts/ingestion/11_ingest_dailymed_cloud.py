#!/usr/bin/env python3
"""
DailyMed Cloud Ingestion - Dense + Sparse via Qdrant Cloud Inference.

Uses Cloud Inference for BOTH embedding types:
- Dense: mixedbread-ai/mxbai-embed-large-v1 (1024-d)
- Sparse: prithivida/splade_pp_en_v1

Extracts 8 sections with markdown table support (40K chars each):
- highlights, indications, dosage, contraindications, 
- warnings, adverse_reactions, clinical_pharmacology, clinical_studies

Run on EC2:
    python3 11_ingest_dailymed_cloud.py --xml-dir /data/dailymed/xml
"""

import os
import sys
import logging
import uuid
import time
import re
from lxml import etree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal
from io import StringIO

# Dependencies
try:
    from tqdm import tqdm
except ImportError:
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    os.system("pip3 install pandas lxml tabulate --quiet")
    import pandas as pd

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
        logging.FileHandler('dailymed_cloud_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://cf6c28ca-8a2a-43fa-9424-1f2af9e9a5f3.us-east-1-1.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pmc_medical_rag_fulltext"

# Cloud Inference models
DENSE_MODEL = "mixedbread-ai/mxbai-embed-large-v1"  # For default/unnamed dense vector
SPARSE_MODEL = "prithivida/splade_pp_en_v1"  # For named "sparse" vector

# Section limits (40K each as requested)
SECTION_CHAR_LIMIT = 40000

# Batch settings
BATCH_SIZE = 10  # Reduced for stability with Cloud Inference
PARALLEL_WORKERS = 1  # Single worker to reduce memory pressure
MAX_RETRIES = 3
FILE_TIMEOUT_SECONDS = 30  # Skip files that take too long to parse
CHECKPOINT_FILE = Path("dailymed_cloud_checkpoint.txt")

# HL7 SPL namespace
NS = {'hl7': 'urn:hl7-org:v3'}

# LOINC codes for ALL 8 sections
SECTION_CODES = {
    '34066-1': 'boxed_warning',         # BOXED WARNING
    '48780-1': 'highlights',            # HIGHLIGHTS OF PRESCRIBING INFORMATION
    '34067-9': 'indications',           # INDICATIONS & USAGE
    '34068-7': 'dosage',                # DOSAGE & ADMINISTRATION
    '34070-3': 'contraindications',     # CONTRAINDICATIONS
    '34071-1': 'warnings',              # WARNINGS
    '43685-7': 'warnings_precautions',  # WARNINGS AND PRECAUTIONS (alternative)
    '34084-4': 'adverse_reactions',     # ADVERSE REACTIONS
    '34090-1': 'clinical_pharmacology', # CLINICAL PHARMACOLOGY
    '34092-7': 'clinical_studies',      # CLINICAL STUDIES
}

# ============================================================================
# THREAD-SAFE COUNTERS
# ============================================================================
class Counters:
    def __init__(self):
        self.success = 0
        self.errors = 0
        self.skipped = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def increment_success(self, count=1):
        with self.lock:
            self.success += count
    
    def increment_errors(self, count=1):
        with self.lock:
            self.errors += count
    
    def get_rate(self):
        elapsed = time.time() - self.start_time
        return self.success / elapsed if elapsed > 0 else 0

counters = Counters()

# ============================================================================
# TABLE EXTRACTION USING PANDAS (ROBUST)
# ============================================================================
def html_table_to_markdown(html_table: str) -> str:
    """
    Convert HTML table to markdown using pandas.
    
    Handles:
    - Variable column counts
    - colspan (fills merged columns)
    - rowspan (fills merged rows)
    - Empty cells
    - MultiIndex headers (flattened)
    """
    try:
        dfs = pd.read_html(StringIO(html_table), flavor='lxml')
        if not dfs:
            return ""
        
        df = dfs[0]
        if df.empty:
            return ""
        
        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' - '.join(str(c) for c in col if str(c) != 'nan').strip(' - ') 
                          for col in df.columns.values]
        
        df.columns = [str(col) for col in df.columns]
        df = df.fillna('')
        
        # Convert to markdown
        try:
            return df.to_markdown(index=False)
        except ImportError:
            # Manual fallback if tabulate not installed
            headers = list(df.columns)
            header_row = "| " + " | ".join(str(h) for h in headers) + " |"
            separator = "| " + " | ".join(["---"] * len(headers)) + " |"
            data_rows = []
            for _, row in df.iterrows():
                row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
                data_rows.append(row_str)
            return "\n".join([header_row, separator] + data_rows)
            
    except Exception as e:
        logger.debug(f"Table conversion failed: {e}")
        return ""


def extract_text_from_html(html: str) -> str:
    """Extract plain text from HTML, stripping all tags."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_section_recursive(element: Optional[ET.Element]) -> str:
    """
    Recursively extract content from a section and all its nested subsections.
    Handles <text> elements and <component><section> structures.
    """
    if element is None:
        return ""
    
    parts = []
    
    # 1. Extract text from the current section's <text> element
    text_elem = element.find('hl7:text', NS)
    if text_elem is not None:
        html_content = ET.tostring(text_elem, encoding='unicode', method='html')
        
        # Remove images/multimedia
        html_content = re.sub(r'<renderMultiMedia[^>]*>.*?</renderMultiMedia>', '', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<renderMultiMedia[^>]*/>', '', html_content)
        html_content = re.sub(r'<img[^>]*/>', '', html_content)
        
        # Extract tables and text
        table_pattern = r'(<table[^>]*>.*?</table>)'
        segments = re.split(table_pattern, html_content, flags=re.DOTALL | re.IGNORECASE)
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            if segment.lower().startswith('<table'):
                md_table = html_table_to_markdown(segment)
                if md_table:
                    parts.append(md_table)
            else:
                text = extract_text_from_html(segment)
                if text:
                    parts.append(text)

    # 2. Recursively process nested components/sections
    # Note: Structure is usually <component><section>...</section></component>
    for component in element.findall('hl7:component', NS):
        sub_section = component.find('hl7:section', NS)
        if sub_section is not None:
            # Add subsection title if present
            title_node = sub_section.find('hl7:title', NS)
            if title_node is not None:
                # Use itertext() to handle nested tags like <content> inside title
                title_text = " ".join("".join(title_node.itertext()).split())
                if title_text:
                    parts.append(f"\n### {title_text}")
            
            # Recurse
            sub_content = extract_section_recursive(sub_section)
            if sub_content:
                parts.append(sub_content)
    
    return "\n\n".join(parts)


# ============================================================================
# SPL PARSING
# ============================================================================
def get_text(element: Optional[ET.Element]) -> str:
    """Extract all text from element."""
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def parse_spl_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse SPL XML and extract all 8 key sections with table support."""
    try:
        parser = ET.XMLParser(recover=True)
        tree = ET.parse(str(xml_path), parser=parser)
        root = tree.getroot()
        
        # Extract identifiers
        set_id_elem = root.find('.//hl7:setId', NS)
        set_id = set_id_elem.get('root', '') if set_id_elem is not None else ''
        
        title_elem = root.find('.//hl7:title', NS)
        title = get_text(title_elem)
        
        drug_name = title
        name_elem = root.find('.//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:name', NS)
        if name_elem is not None and name_elem.text:
            drug_name = name_elem.text.strip()
        
        manufacturer = ""
        org_name = root.find('.//hl7:representedOrganization/hl7:name', NS)
        if org_name is not None:
            manufacturer = org_name.text or ""
        
        # Extract ALL 8 sections
        sections = {}
        for section in root.findall('.//hl7:section', NS):
            code_elem = section.find('hl7:code', NS)
            if code_elem is not None:
                code = code_elem.get('code', '')
                if code in SECTION_CODES:
                    # RECURSIVE EXTRACTION: Pass the whole section element
                    content = extract_section_recursive(section)
                    
                    field_name = SECTION_CODES[code]
                    # Merge warnings and warnings_precautions into 'warnings'
                    if field_name == 'warnings_precautions':
                        field_name = 'warnings'
                        if 'warnings' in sections:
                            content = sections['warnings'] + "\n\n" + content
                    
                    # If field already exists (e.g. multiple sections with same code), append
                    if field_name in sections:
                        sections[field_name] += "\n\n" + content
                    else:
                        sections[field_name] = content[:SECTION_CHAR_LIMIT]
            
            # MEMORY CLEANUP: Clear section after processing
            section.clear()
        
        # CONSTRUCT HIGHLIGHTS (Preamble + Boxed Warning + 48780-1)
        # 1. Preamble (Title)
        hl_parts = [title]
        
        # 2. Boxed Warning (34066-1)
        if 'boxed_warning' in sections:
            hl_parts.append(sections['boxed_warning'])
            
        # 3. Existing Highlights (48780-1)
        if 'highlights' in sections:
            hl_parts.append(sections['highlights'])
            
        final_highlights = "\n\n".join(hl_parts)

        return {
            "set_id": set_id,
            "drug_name": drug_name,
            "title": title,
            "manufacturer": manufacturer,
            "highlights": final_highlights[:SECTION_CHAR_LIMIT],
            "indications": sections.get('indications', ''),
            "dosage": sections.get('dosage', ''),
            "contraindications": sections.get('contraindications', ''),
            "warnings": sections.get('warnings', ''),
            "adverse_reactions": sections.get('adverse_reactions', ''),
            "clinical_pharmacology": sections.get('clinical_pharmacology', ''),
            "clinical_studies": sections.get('clinical_studies', ''),
            "source": "dailymed",
        }
        
    except Exception as e:
        logger.debug(f"Error parsing {xml_path}: {e}")
        return None


# ============================================================================
# EMBEDDING TEXT CREATION
# ============================================================================
def create_embedding_text(drug: Dict[str, Any]) -> str:
    """Create text for Cloud Inference embeddings - use highlights + indications + dosage."""
    parts = [
        drug.get("drug_name", ""),
        drug.get("highlights", "")[:800],
        drug.get("indications", "")[:500],
        drug.get("dosage", "")[:500],
    ]
    return ". ".join(p for p in parts if p)[:2000]


# ============================================================================
# BATCH CREATION WITH CLOUD INFERENCE
# ============================================================================
def create_points_batch(drugs: List[Dict[str, Any]]) -> tuple[List[PointStruct], List[str]]:
    """
    Create PointStruct objects using Cloud Inference for BOTH vectors.
    
    Uses named vectors:
    - "" (default): Dense vector via mixedbread-ai/mxbai-embed-large-v1
    - "sparse": Sparse vector via prithvida/splade_pp_en_v1
    """
    points = []
    set_ids = []
    
    for drug in drugs:
        set_id = drug.get("set_id")
        if not set_id:
            continue
        
        emb_text = create_embedding_text(drug)
        if not emb_text.strip() or len(emb_text) < 20:
            continue
        
        # Deterministic point ID (enables upsert/replace)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"dailymed_{set_id}"))
        
        # Build payload with ALL 8 sections
        payload = {
            "set_id": set_id,
            "drug_name": drug.get("drug_name", "")[:200],
            "title": drug.get("title", "")[:300],
            "manufacturer": drug.get("manufacturer", ""),
            "highlights": drug.get("highlights", "")[:SECTION_CHAR_LIMIT],
            "indications": drug.get("indications", "")[:SECTION_CHAR_LIMIT],
            "dosage": drug.get("dosage", "")[:SECTION_CHAR_LIMIT],
            "contraindications": drug.get("contraindications", "")[:SECTION_CHAR_LIMIT],
            "warnings": drug.get("warnings", "")[:SECTION_CHAR_LIMIT],
            "adverse_reactions": drug.get("adverse_reactions", "")[:SECTION_CHAR_LIMIT],
            "clinical_pharmacology": drug.get("clinical_pharmacology", "")[:SECTION_CHAR_LIMIT],
            "clinical_studies": drug.get("clinical_studies", "")[:SECTION_CHAR_LIMIT],
            "source": "dailymed",
            "article_type": "drug_label",
        }
        
        # Create point with BOTH dense and sparse vectors via Cloud Inference
        point = PointStruct(
            id=point_id,
            vector={
                # Default (unnamed) dense vector - use empty string key
                "": Document(text=emb_text, model=DENSE_MODEL),
                # Named sparse vector - use "sparse" key matching collection config
                "sparse": Document(text=emb_text, model=SPARSE_MODEL),
            },
            payload=payload
        )
        
        points.append(point)
        set_ids.append(set_id)
    
    return points, set_ids


# ============================================================================
# UPSERT WITH RETRY
# ============================================================================
def upsert_batch(client: QdrantClient, points: List[PointStruct], set_ids: List[str]) -> bool:
    """Upsert batch with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=False)
            with open(CHECKPOINT_FILE, 'a') as f:
                for sid in set_ids:
                    f.write(f"{sid}\n")
            counters.increment_success(len(points))
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} after {sleep_time}s: {str(e)[:100]}")
                time.sleep(sleep_time)
            else:
                logger.error(f"Batch failed: {str(e)[:200]}")
                counters.increment_errors(len(points))
                return False
    return False


# ============================================================================
# MAIN INGESTION
# ============================================================================
def run_ingestion(xml_dir: Path):
    """Run DailyMed Cloud Ingestion."""
    logger.info("=" * 70)
    logger.info("🔬 DailyMed Cloud Ingestion (Dense + Sparse via Cloud Inference)")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    # Connect to Qdrant with Cloud Inference enabled
    client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY, 
        timeout=120,  # Reduced from 600s for faster failure detection
        cloud_inference=True  # CRITICAL: Enable Cloud Inference
    )
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to Qdrant: {COLLECTION_NAME}")
        logger.info(f"   Current points: {info.points_count:,}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Section limit: {SECTION_CHAR_LIMIT:,} chars")
    logger.info(f"   Dense model: {DENSE_MODEL}")
    logger.info(f"   Sparse model: {SPARSE_MODEL}")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Sections: highlights, indications, dosage, contraindications, warnings, adverse_reactions, clinical_pharmacology, clinical_studies")
    
    # Get checkpoint
    ingested = set()
    if CHECKPOINT_FILE.exists():
        ingested = set(CHECKPOINT_FILE.read_text().strip().split('\n'))
    logger.info(f"   Already ingested: {len(ingested):,}")
    
    # Find XML files
    xml_files = list(xml_dir.glob("*.xml"))
    logger.info(f"\n📤 Found {len(xml_files):,} XML files")
    
    # Process
    current_batch = []
    batch_num = 0
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        pending = []
        
        for xml_file in tqdm(xml_files, desc="Processing"):
            # Log file being processed for debugging stalls
            logger.debug(f"Parsing: {xml_file.name}")
            
            # Timeout wrapper to skip problematic files
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Parsing {xml_file.name} timed out")
            
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(FILE_TIMEOUT_SECONDS)
                drug = parse_spl_xml(xml_file)
                signal.alarm(0)  # Cancel alarm
            except TimeoutError as e:
                logger.warning(f"⏱️ Skipping slow file: {xml_file.name}")
                signal.alarm(0)
                continue
            except Exception as e:
                logger.debug(f"Error parsing {xml_file.name}: {e}")
                signal.alarm(0)
                continue
            
            if not drug:
                continue
            
            set_id = drug.get("set_id")
            if not set_id or set_id in ingested:
                continue
            
            current_batch.append(drug)
            
            if len(current_batch) >= BATCH_SIZE:
                points, set_ids = create_points_batch(current_batch)
                if points:
                    future = executor.submit(upsert_batch, client, points, set_ids)
                    pending.append(future)
                current_batch = []
                batch_num += 1
                pending = [f for f in pending if not f.done()]
                
                if batch_num % 20 == 0:
                    logger.info(f"Progress: {counters.success:,} ingested, {counters.errors:,} errors, {counters.get_rate():.1f}/sec")
        
        # Final batch
        if current_batch:
            points, set_ids = create_points_batch(current_batch)
            if points:
                future = executor.submit(upsert_batch, client, points, set_ids)
                pending.append(future)
        
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error: {e}")
    
    # Stats
    info = client.get_collection(COLLECTION_NAME)
    elapsed = time.time() - counters.start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ DailyMed Cloud Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"📊 Results:")
    logger.info(f"   Ingested: {counters.success:,}")
    logger.info(f"   Errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"   Rate: {counters.get_rate():.1f} articles/sec")
    logger.info(f"   Collection total: {info.points_count:,}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DailyMed Cloud Ingestion")
    parser.add_argument("--xml-dir", type=Path, default=Path("/data/dailymed/xml"))
    args = parser.parse_args()
    
    if not args.xml_dir.exists():
        logger.error(f"XML directory not found: {args.xml_dir}")
        sys.exit(1)
    
    run_ingestion(args.xml_dir)


if __name__ == "__main__":
    main()
