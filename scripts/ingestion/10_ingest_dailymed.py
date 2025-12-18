#!/usr/bin/env python3
"""
DailyMed Ingestion Script - Extract and ingest drug labels to Qdrant.

Parses HL7 SPL XML files and ingests to Qdrant using Cloud Inference.
"""

import os
import sys
import json
import logging
import uuid
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
        logging.FileHandler('/data/dailymed_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://cf6c28ca-8a2a-43fa-9424-1f2af9e9a5f3.us-east-1-1.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pmc_medical_rag_fulltext"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

BATCH_SIZE = 50
PARALLEL_WORKERS = 4
MAX_RETRIES = 3
CHECKPOINT_FILE = Path("/data/dailymed_checkpoint.txt")

# HL7 SPL namespace
NS = {'hl7': 'urn:hl7-org:v3'}

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


def get_text(element: Optional[ET.Element]) -> str:
    """Extract all text from an element recursively."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def parse_spl_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse an HL7 SPL XML file and extract drug information."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract set ID
        set_id_elem = root.find('.//hl7:setId', NS)
        set_id = set_id_elem.get('root', '') if set_id_elem is not None else ''
        
        # Extract title
        title_elem = root.find('.//hl7:title', NS)
        title = get_text(title_elem)
        
        # Extract drug name from manufactured product
        drug_name = title  # Default to title
        name_elem = root.find('.//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:name', NS)
        if name_elem is not None and name_elem.text:
            drug_name = name_elem.text.strip()
        
        # Extract active ingredients
        active_ingredients = []
        for ingredient in root.findall('.//hl7:ingredient[@classCode="ACTIB"]', NS):
            ing_name = ingredient.find('.//hl7:ingredientSubstance/hl7:name', NS)
            if ing_name is not None and ing_name.text:
                active_ingredients.append(ing_name.text.strip())
        
        # Extract sections by code
        sections = {}
        section_codes = {
            '34067-9': 'indications',      # INDICATIONS & USAGE
            '34070-3': 'contraindications', # CONTRAINDICATIONS
            '34068-7': 'dosage',           # DOSAGE & ADMINISTRATION
            '34084-4': 'adverse_reactions', # ADVERSE REACTIONS
            '34073-7': 'interactions',     # DRUG INTERACTIONS
            '34071-1': 'warnings',         # WARNINGS
            '43685-7': 'warnings_precautions', # WARNINGS AND PRECAUTIONS
            '34089-3': 'description',      # DESCRIPTION
            '43679-0': 'mechanism',        # MECHANISM OF ACTION
            '34090-1': 'clinical_pharmacology', # CLINICAL PHARMACOLOGY
        }
        
        for section in root.findall('.//hl7:section', NS):
            code_elem = section.find('hl7:code', NS)
            if code_elem is not None:
                code = code_elem.get('code', '')
                if code in section_codes:
                    text_elem = section.find('hl7:text', NS)
                    sections[section_codes[code]] = get_text(text_elem)[:5000]
        
        # Extract manufacturer
        manufacturer = ""
        org_name = root.find('.//hl7:representedOrganization/hl7:name', NS)
        if org_name is not None:
            manufacturer = org_name.text or ""
        
        # Build drug record
        drug = {
            "set_id": set_id,
            "drug_name": drug_name,
            "title": title,
            "active_ingredients": active_ingredients,
            "manufacturer": manufacturer,
            "indications": sections.get('indications', ''),
            "contraindications": sections.get('contraindications', ''),
            "dosage": sections.get('dosage', ''),
            "adverse_reactions": sections.get('adverse_reactions', ''),
            "interactions": sections.get('interactions', ''),
            "warnings": sections.get('warnings', '') or sections.get('warnings_precautions', ''),
            "description": sections.get('description', ''),
            "mechanism": sections.get('mechanism', ''),
            "source": "dailymed",
            "source_file": str(xml_path),
        }
        
        return drug
        
    except Exception as e:
        logger.debug(f"Error parsing {xml_path}: {e}")
        return None


def create_embedding_text(drug: Dict[str, Any]) -> str:
    """Create text for embedding."""
    parts = [
        drug.get("drug_name", ""),
        f"Active ingredients: {', '.join(drug.get('active_ingredients', []))}",
        drug.get("indications", ""),
        drug.get("contraindications", ""),
        drug.get("warnings", ""),
        drug.get("dosage", "")[:500],
    ]
    combined = ". ".join(p for p in parts if p)
    return combined[:2000]


def create_points_batch(drugs: List[Dict[str, Any]]) -> tuple[List[PointStruct], List[str]]:
    """Create PointStruct objects for a batch of drugs."""
    points = []
    set_ids = []
    
    for drug in drugs:
        set_id = drug.get("set_id")
        if not set_id:
            continue
        
        embedding_text = create_embedding_text(drug)
        if not embedding_text.strip() or len(embedding_text) < 20:
            continue
        
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"dailymed_{set_id}"))
        
        payload = {
            "set_id": set_id,
            "drug_name": drug.get("drug_name", "")[:200],
            "title": drug.get("title", "")[:300],
            "active_ingredients": drug.get("active_ingredients", [])[:10],
            "manufacturer": drug.get("manufacturer", ""),
            "indications": drug.get("indications", "")[:1000],
            "contraindications": drug.get("contraindications", "")[:500],
            "warnings": drug.get("warnings", "")[:500],
            "adverse_reactions": drug.get("adverse_reactions", "")[:500],
            "dosage": drug.get("dosage", "")[:500],
            "source": "dailymed",
            "article_type": "drug_label",
        }
        
        point = PointStruct(
            id=point_id,
            vector=Document(text=embedding_text, model=EMBEDDING_MODEL),
            payload=payload
        )
        
        points.append(point)
        set_ids.append(set_id)
    
    return points, set_ids


def upsert_batch(client: QdrantClient, points: List[PointStruct], set_ids: List[str]) -> bool:
    """Upsert a batch with retry logic."""
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
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Batch failed: {str(e)[:100]}")
                counters.increment_errors(len(points))
                return False
    return False


def run_ingestion(xml_dir: Path):
    """Run DailyMed ingestion."""
    logger.info("=" * 70)
    logger.info("🔬 DailyMed Ingestion with Cloud Inference")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=600, cloud_inference=True)
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to Qdrant: {COLLECTION_NAME}")
        logger.info(f"   Current points: {info.points_count:,}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    # Get checkpoint
    ingested = set()
    if CHECKPOINT_FILE.exists():
        ingested = set(CHECKPOINT_FILE.read_text().strip().split('\n'))
    logger.info(f"   Already ingested: {len(ingested):,}")
    
    # Find XML files
    xml_files = list(xml_dir.glob("*.xml"))
    logger.info(f"\n📤 Found {len(xml_files):,} XML files")
    
    # Process files
    current_batch = []
    batch_num = 0
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        pending = []
        
        for xml_file in tqdm(xml_files, desc="Processing"):
            # Parse XML
            drug = parse_spl_xml(xml_file)
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
                
                if batch_num % 10 == 0:
                    logger.info(f"Progress: {counters.success:,} ingested, {counters.errors:,} errors")
        
        # Final batch
        if current_batch:
            points, set_ids = create_points_batch(current_batch)
            if points:
                future = executor.submit(upsert_batch, client, points, set_ids)
                pending.append(future)
        
        # Wait for pending
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error: {e}")
    
    # Final stats
    info = client.get_collection(COLLECTION_NAME)
    elapsed = time.time() - counters.start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ DailyMed Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"📊 Results:")
    logger.info(f"   Ingested: {counters.success:,}")
    logger.info(f"   Errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed:.1f} seconds")
    logger.info(f"   Collection total: {info.points_count:,}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DailyMed Ingestion")
    parser.add_argument("--xml-dir", type=Path, default=Path("/data/dailymed/xml"))
    args = parser.parse_args()
    
    if not args.xml_dir.exists():
        logger.error(f"XML directory not found: {args.xml_dir}")
        sys.exit(1)
    
    run_ingestion(args.xml_dir)


if __name__ == "__main__":
    main()

