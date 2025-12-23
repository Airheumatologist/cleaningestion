#!/usr/bin/env python3
"""
Ingest DailyMed Drug Labels to Qdrant.

Parses HL7 SPL XML files and ingests to Qdrant using Cloud Inference.

Usage:
    python 07_ingest_dailymed.py --xml-dir /data/dailymed/xml

Expected Duration: ~5 minutes for 50K drug labels
"""

import os
import sys
import json
import logging
import uuid
import time
import argparse
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dailymed_ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pmc_medical_rag_fulltext")
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "4"))
MAX_RETRIES = 3
CHECKPOINT_FILE = Path("dailymed_checkpoint.txt")

# HL7 SPL namespace
NS = {'hl7': 'urn:hl7-org:v3'}

# Section codes
SECTION_CODES = {
    '34067-9': 'indications',
    '34070-3': 'contraindications',
    '34068-7': 'dosage',
    '34084-4': 'adverse_reactions',
    '34073-7': 'interactions',
    '34071-1': 'warnings',
    '43685-7': 'warnings_precautions',
    '34089-3': 'description',
    '43679-0': 'mechanism',
}

# =============================================================================


class Counters:
    """Thread-safe counters."""
    
    def __init__(self):
        self.success = 0
        self.errors = 0
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


def get_text(element: Optional[ET.Element]) -> str:
    """Extract all text from element."""
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()


def parse_spl_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse HL7 SPL XML file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Set ID
        set_id_elem = root.find('.//hl7:setId', NS)
        set_id = set_id_elem.get('root', '') if set_id_elem is not None else ''
        if not set_id:
            set_id = xml_path.stem
        
        # Title
        title_elem = root.find('.//hl7:title', NS)
        title = get_text(title_elem)
        
        # Drug name
        drug_name = title
        name_elem = root.find('.//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:name', NS)
        if name_elem is not None and name_elem.text:
            drug_name = name_elem.text.strip()
        
        # Active ingredients
        active_ingredients = []
        for ingredient in root.findall('.//hl7:ingredient[@classCode="ACTIB"]', NS):
            ing_name = ingredient.find('.//hl7:ingredientSubstance/hl7:name', NS)
            if ing_name is not None and ing_name.text:
                active_ingredients.append(ing_name.text.strip())
        
        # Sections
        sections = {}
        for section in root.findall('.//hl7:section', NS):
            code_elem = section.find('hl7:code', NS)
            if code_elem is not None:
                code = code_elem.get('code', '')
                if code in SECTION_CODES:
                    text_elem = section.find('hl7:text', NS)
                    sections[SECTION_CODES[code]] = get_text(text_elem)[:5000]
        
        # Manufacturer
        manufacturer = ""
        org_name = root.find('.//hl7:representedOrganization/hl7:name', NS)
        if org_name is not None:
            manufacturer = org_name.text or ""
        
        return {
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
        }
        
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


def create_points_batch(drugs: List[Dict[str, Any]]) -> tuple:
    """Create PointStruct objects."""
    points = []
    ids = []
    
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
            "indications": drug.get("indications", "")[:5000],        # Increased from 1000
            "contraindications": drug.get("contraindications", "")[:2000],  # Increased from 500
            "warnings": drug.get("warnings", "")[:3000],            # Increased from 500
            "adverse_reactions": drug.get("adverse_reactions", "")[:3000], # Increased from 500
            "dosage": drug.get("dosage", "")[:5000],               # Increased from 500
            "source": "dailymed",
            "article_type": "drug_label",
        }
        
        point = PointStruct(
            id=point_id,
            vector=Document(text=embedding_text, model=EMBEDDING_MODEL),
            payload=payload
        )
        
        points.append(point)
        ids.append(set_id)
    
    return points, ids


def upsert_batch(client: QdrantClient, points: List[PointStruct], ids: List[str], counters: Counters) -> bool:
    """Upsert with retry."""
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points, wait=False)
            with open(CHECKPOINT_FILE, 'a') as f:
                for sid in ids:
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
    logger.info("💊 DailyMed Ingestion with Cloud Inference")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    # Connect
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,
        cloud_inference=True
    )
    
    try:
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected: {COLLECTION_NAME}")
        logger.info(f"   Current points: {info.points_count:,}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    # Checkpoint
    counters = Counters()
    ingested = set()
    if CHECKPOINT_FILE.exists():
        ingested = set(CHECKPOINT_FILE.read_text().strip().split('\n'))
    logger.info(f"   Already ingested: {len(ingested):,}")
    
    # Find XMLs
    xml_files = list(xml_dir.glob("*.xml"))
    logger.info(f"\n📤 Found {len(xml_files):,} XML files")
    
    # Process
    current_batch = []
    batch_num = 0
    
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        pending = []
        
        for xml_file in tqdm(xml_files, desc="Processing"):
            drug = parse_spl_xml(xml_file)
            if not drug:
                continue
            
            set_id = drug.get("set_id")
            if not set_id or set_id in ingested:
                continue
            
            current_batch.append(drug)
            
            if len(current_batch) >= BATCH_SIZE:
                points, ids = create_points_batch(current_batch)
                if points:
                    future = executor.submit(upsert_batch, client, points, ids, counters)
                    pending.append(future)
                current_batch = []
                batch_num += 1
                pending = [f for f in pending if not f.done()]
                
                if batch_num % 10 == 0:
                    logger.info(f"Progress: {counters.success:,} ingested, {counters.errors:,} errors")
        
        # Final batch
        if current_batch:
            points, ids = create_points_batch(current_batch)
            if points:
                future = executor.submit(upsert_batch, client, points, ids, counters)
                pending.append(future)
        
        # Wait
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error: {e}")
    
    # Stats
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
    parser = argparse.ArgumentParser(description="Ingest DailyMed to Qdrant")
    parser.add_argument("--xml-dir", type=Path, default=Path("/data/dailymed/xml"))
    args = parser.parse_args()
    
    if not args.xml_dir.exists():
        logger.error(f"Directory not found: {args.xml_dir}")
        sys.exit(1)
    
    run_ingestion(args.xml_dir)


if __name__ == "__main__":
    main()

