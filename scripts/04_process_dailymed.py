#!/usr/bin/env python3
"""
Process DailyMed SPL XML Files.

Extracts drug information from Structured Product Labeling (SPL) XML files
and converts to JSONL format.

Extracts:
- Drug name and active ingredients
- Indications, contraindications, warnings
- Dosage information
- Adverse reactions
- Drug interactions

Usage:
    python 04_process_dailymed.py --spl-dir /data/dailymed/xml --output /data/dailymed_drugs.jsonl

Expected Duration: 10-30 minutes
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import xml.etree.ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HL7 SPL namespace
NS = {'hl7': 'urn:hl7-org:v3'}

# Section codes for drug label sections
SECTION_CODES = {
    '34067-9': 'indications',           # INDICATIONS & USAGE
    '34070-3': 'contraindications',     # CONTRAINDICATIONS
    '34068-7': 'dosage',                # DOSAGE & ADMINISTRATION
    '34084-4': 'adverse_reactions',     # ADVERSE REACTIONS
    '34073-7': 'interactions',          # DRUG INTERACTIONS
    '34071-1': 'warnings',              # WARNINGS
    '43685-7': 'warnings_precautions',  # WARNINGS AND PRECAUTIONS
    '34089-3': 'description',           # DESCRIPTION
    '43679-0': 'mechanism',             # MECHANISM OF ACTION
    '34090-1': 'clinical_pharmacology', # CLINICAL PHARMACOLOGY
    '42229-5': 'spl_unclassified',      # SPL UNCLASSIFIED SECTION
}


def get_text(element: Optional[ET.Element]) -> str:
    """Extract all text from an element recursively."""
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()


def parse_spl_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse an HL7 SPL XML file and extract drug information."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract set ID
        set_id_elem = root.find('.//hl7:setId', NS)
        set_id = set_id_elem.get('root', '') if set_id_elem is not None else ''
        
        if not set_id:
            # Fallback to filename
            set_id = xml_path.stem
        
        # Extract title
        title_elem = root.find('.//hl7:title', NS)
        title = get_text(title_elem)
        
        # Extract drug name from manufactured product
        drug_name = title  # Default to title
        name_elem = root.find('.//hl7:manufacturedProduct/hl7:manufacturedProduct/hl7:name', NS)
        if name_elem is not None and name_elem.text:
            drug_name = name_elem.text.strip()
        
        # If still no drug name, try other paths
        if not drug_name or drug_name == title:
            for path in [
                './/hl7:name',
                './/hl7:manufacturedProduct//hl7:name',
            ]:
                elem = root.find(path, NS)
                if elem is not None and elem.text:
                    drug_name = elem.text.strip()
                    break
        
        # Extract active ingredients
        active_ingredients = []
        for ingredient in root.findall('.//hl7:ingredient[@classCode="ACTIB"]', NS):
            ing_name = ingredient.find('.//hl7:ingredientSubstance/hl7:name', NS)
            if ing_name is not None and ing_name.text:
                active_ingredients.append(ing_name.text.strip())
        
        # Also try alternative paths
        if not active_ingredients:
            for ingredient in root.findall('.//hl7:ingredient', NS):
                ing_name = ingredient.find('.//hl7:name', NS)
                if ing_name is not None and ing_name.text:
                    active_ingredients.append(ing_name.text.strip())
        
        # Extract sections by code
        sections = {}
        for section in root.findall('.//hl7:section', NS):
            code_elem = section.find('hl7:code', NS)
            if code_elem is not None:
                code = code_elem.get('code', '')
                if code in SECTION_CODES:
                    text_elem = section.find('hl7:text', NS)
                    section_text = get_text(text_elem)
                    if section_text:
                        sections[SECTION_CODES[code]] = section_text[:5000]
        
        # Extract manufacturer
        manufacturer = ""
        org_name = root.find('.//hl7:representedOrganization/hl7:name', NS)
        if org_name is not None and org_name.text:
            manufacturer = org_name.text.strip()
        
        # Build drug record
        drug = {
            "set_id": set_id,
            "drug_name": drug_name,
            "title": title,
            "active_ingredients": active_ingredients[:10],
            "manufacturer": manufacturer,
            "indications": sections.get('indications', ''),
            "contraindications": sections.get('contraindications', ''),
            "dosage": sections.get('dosage', ''),
            "adverse_reactions": sections.get('adverse_reactions', ''),
            "interactions": sections.get('interactions', ''),
            "warnings": sections.get('warnings', '') or sections.get('warnings_precautions', ''),
            "description": sections.get('description', ''),
            "mechanism": sections.get('mechanism', ''),
            "clinical_pharmacology": sections.get('clinical_pharmacology', ''),
            "source": "dailymed",
            "article_type": "drug_label",
            "source_file": str(xml_path),
            "extracted_at": datetime.utcnow().isoformat(),
        }
        
        return drug
        
    except ET.ParseError as e:
        logger.debug(f"XML parse error in {xml_path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Error parsing {xml_path}: {e}")
        return None


def process_dailymed(spl_dir: Path, output_file: Path):
    """Process all SPL XML files in directory."""
    
    logger.info("=" * 70)
    logger.info("💊 DailyMed SPL Processing")
    logger.info("=" * 70)
    
    if not spl_dir.exists():
        logger.error(f"Directory not found: {spl_dir}")
        sys.exit(1)
    
    # Find XML files
    xml_files = list(spl_dir.glob("*.xml"))
    logger.info(f"Found {len(xml_files):,} XML files")
    
    if not xml_files:
        logger.error("No XML files found!")
        sys.exit(1)
    
    # Process files
    processed = 0
    errors = 0
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for xml_file in tqdm(xml_files, desc="Processing SPL files"):
            drug = parse_spl_xml(xml_file)
            if drug and drug.get('drug_name'):
                f.write(json.dumps(drug) + '\n')
                processed += 1
            else:
                errors += 1
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ DailyMed Processing Complete!")
    logger.info("=" * 70)
    logger.info(f"   Processed: {processed:,}")
    logger.info(f"   Errors: {errors:,}")
    logger.info(f"   Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process DailyMed SPL files")
    parser.add_argument("--spl-dir", type=Path, required=True, help="Directory with SPL XML files")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    
    args = parser.parse_args()
    process_dailymed(args.spl_dir, args.output)


if __name__ == "__main__":
    main()

