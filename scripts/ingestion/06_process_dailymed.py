#!/usr/bin/env python3
"""
Process DailyMed SPL Files.

Extracts drug information from Structured Product Labeling (SPL) XML files
and converts to JSONL format matching the article schema.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_text(element: Optional[ET.Element], default: str = "") -> str:
    """Extract text from XML element."""
    if element is None:
        return default
    return ' '.join(element.itertext()).strip()


def extract_drug_name(root: ET.Element) -> str:
    """Extract drug name."""
    # SPL structure: /document/component/structuredBody/component/section[title='DESCRIPTION']/text
    # Or: /document/setId
    
    # Try multiple paths
    name_elem = root.find(".//description/name")
    if name_elem is not None:
        return extract_text(name_elem)
    
    name_elem = root.find(".//productName")
    if name_elem is not None:
        return extract_text(name_elem)
    
    # Fallback to set ID
    set_id_elem = root.find(".//setId")
    if set_id_elem is not None:
        return set_id_elem.get("root", "") or extract_text(set_id_elem)
    
    return "Unknown"


def extract_active_ingredients(root: ET.Element) -> List[str]:
    """Extract active ingredients."""
    ingredients = []
    
    # SPL structure for ingredients
    for ingredient in root.findall(".//ingredient"):
        name = extract_text(ingredient.find("ingredientName"))
        if name:
            ingredients.append(name)
    
    # Alternative path
    for ingredient in root.findall(".//activeIngredient"):
        name = extract_text(ingredient)
        if name:
            ingredients.append(name)
    
    return ingredients


def extract_indications(root: ET.Element) -> str:
    """Extract indications."""
    # Look for indications section
    indications_section = root.find(".//section[title='INDICATIONS AND USAGE']")
    if indications_section is None:
        indications_section = root.find(".//section[title='INDICATIONS']")
    
    if indications_section is not None:
        text = extract_text(indications_section.find("text"))
        return text
    
    return ""


def extract_contraindications(root: ET.Element) -> str:
    """Extract contraindications."""
    contraindications_section = root.find(".//section[title='CONTRAINDICATIONS']")
    
    if contraindications_section is not None:
        text = extract_text(contraindications_section.find("text"))
        return text
    
    return ""


def extract_dosage(root: ET.Element) -> str:
    """Extract dosage information."""
    dosage_section = root.find(".//section[title='DOSAGE AND ADMINISTRATION']")
    if dosage_section is None:
        dosage_section = root.find(".//section[title='DOSAGE']")
    
    if dosage_section is not None:
        text = extract_text(dosage_section.find("text"))
        return text
    
    return ""


def extract_adverse_effects(root: ET.Element) -> str:
    """Extract adverse effects."""
    adverse_section = root.find(".//section[title='ADVERSE REACTIONS']")
    if adverse_section is None:
        adverse_section = root.find(".//section[title='ADVERSE EFFECTS']")
    
    if adverse_section is not None:
        text = extract_text(adverse_section.find("text"))
        return text
    
    return ""


def extract_interactions(root: ET.Element) -> str:
    """Extract drug interactions."""
    interactions_section = root.find(".//section[title='DRUG INTERACTIONS']")
    if interactions_section is None:
        interactions_section = root.find(".//section[title='INTERACTIONS']")
    
    if interactions_section is not None:
        text = extract_text(interactions_section.find("text"))
        return text
    
    return ""


def parse_spl_file(spl_file: Path) -> Optional[Dict]:
    """Parse a single SPL XML file."""
    try:
        tree = ET.parse(spl_file)
        root = tree.getroot()
        
        # Extract set ID
        set_id_elem = root.find(".//setId")
        set_id = set_id_elem.get("root", "") if set_id_elem is not None else spl_file.stem
        
        drug_name = extract_drug_name(root)
        active_ingredients = extract_active_ingredients(root)
        indications = extract_indications(root)
        contraindications = extract_contraindications(root)
        dosage = extract_dosage(root)
        adverse_effects = extract_adverse_effects(root)
        interactions = extract_interactions(root)
        
        return {
            "set_id": set_id,
            "drug_name": drug_name,
            "active_ingredients": active_ingredients,
            "indications": indications[:5000] if indications else "",  # Limit size
            "contraindications": contraindications[:5000] if contraindications else "",
            "dosage": dosage[:5000] if dosage else "",
            "adverse_effects": adverse_effects[:5000] if adverse_effects else "",
            "interactions": interactions[:5000] if interactions else "",
            "source": "dailymed",
            "extracted_at": datetime.utcnow().isoformat(),
            "source_file": str(spl_file)
        }
    
    except Exception as e:
        logger.debug(f"Error parsing {spl_file}: {e}")
        return None


def process_spl_files(spl_dir: Path, output_file: Path):
    """Process all SPL XML files in directory."""
    logger.info(f"Processing SPL files from {spl_dir}...")
    
    if not spl_dir.exists():
        logger.error(f"Directory not found: {spl_dir}")
        return
    
    spl_files = list(spl_dir.glob("*.xml"))
    logger.info(f"Found {len(spl_files)} SPL files")
    
    processed = 0
    with open(output_file, 'w') as f_out:
        for spl_file in tqdm(spl_files, desc="Processing SPL files"):
            drug_data = parse_spl_file(spl_file)
            if drug_data:
                f_out.write(json.dumps(drug_data) + '\n')
                processed += 1
    
    logger.info(f"✅ Processed {processed} drug records")


def main():
    """Main processing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process DailyMed SPL files")
    parser.add_argument("--spl-dir", type=Path, required=True, help="Directory with SPL XML files")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("💊 Processing DailyMed SPL Files")
    print("=" * 70)
    
    process_spl_files(args.spl_dir, args.output)
    
    print(f"\n✅ Processing complete! Output: {args.output}")


if __name__ == "__main__":
    main()

