#!/usr/bin/env python3
"""
Extract PMC Articles from XML to JSONL.

Extracts articles from PMC XML files with:
- Full text content (from <body> element)
- Structured metadata for reranking
- Tables in multiple formats (markdown, row-by-row)
- Author affiliations and institutions
- Evidence grade classification

Usage:
    python 02_extract_pmc.py --xml-dir /data/pmc_fulltext/xml --output /data/pmc_articles.jsonl

Expected Duration: 3-4 hours for 5M XML files
"""

import os
import sys
import json
import logging
import argparse
import gzip
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime




try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip3 install tqdm --quiet")
    from tqdm import tqdm

from scripts.ingestion_utils import parse_pmc_xml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MIN_YEAR = 2015
MAX_YEAR = 2025
MAX_FULL_TEXT_CHARS = 50000
MAX_ABSTRACT_CHARS = 2000


def process_file_wrapper(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Wrapper around parse_pmc_xml to add dataset-specific metadata."""
    article = parse_pmc_xml(xml_path)
    if not article:
        return None

    year = article.get("year")
    if year is None or year < MIN_YEAR or year > MAX_YEAR:
        return None

    # Calculate derived fields
    authors = article.get("authors", [])
    article["first_author"] = authors[0] if authors else None
    article["author_count"] = len(authors)
    
    section_titles = article.get("section_titles", [])
    has_methods = any('method' in t.lower() for t in section_titles)
    has_results = any('result' in t.lower() for t in section_titles)
    has_discussion = any('discuss' in t.lower() or 'conclu' in t.lower() for t in section_titles)
    
    article.update({
        "has_methods": has_methods,
        "has_results": has_results,
        "has_discussion": has_discussion,
        "table_count": len(article.get("tables", [])),
        "figure_count": 0, 
        "is_open_access": True,
        "has_full_text": bool(article.get("full_text")),
        "source": "pmc",
        "source_file": str(xml_path),
        "extracted_at": datetime.utcnow().isoformat(),
        "institutions": article.get("affiliations", [])[:5]
    })
    
    return article


def main():
    parser = argparse.ArgumentParser(description="Extract PMC articles from XML to JSONL")
    parser.add_argument("--xml-dir", type=Path, required=True, help="Directory with XML files")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete each source file after successful extraction (streaming mode)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("📄 PMC Article Extraction")
    logger.info("=" * 70)
    
    if not args.xml_dir.exists():
        logger.error(f"Directory not found: {args.xml_dir}")
        sys.exit(1)
    
    # Find XML files
    logger.info(f"Scanning {args.xml_dir} for XML files...")
    xml_files = list(args.xml_dir.glob("*.xml"))
    xml_files.extend(args.xml_dir.glob("*.xml.gz"))
    
    # Also check subdirectories
    if not xml_files:
        xml_files = list(args.xml_dir.rglob("*.xml"))
        xml_files.extend(args.xml_dir.rglob("*.xml.gz"))
    
    total = len(xml_files)
    logger.info(f"Found {total:,} XML files")
    
    if total == 0:
        logger.error("No XML files found!")
        sys.exit(1)
    
    # Process files
    extracted = 0
    errors = 0
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w') as f:
        for xml_file in tqdm(xml_files, desc="Extracting"):
            try:
                article = process_file_wrapper(xml_file)
                if article:
                    f.write(json.dumps(article) + '\n')
                    extracted += 1
                    if args.delete_source:
                        try:
                            xml_file.unlink(missing_ok=True)
                        except Exception as e:
                            logger.debug(f"Failed to delete {xml_file}: {e}")
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                logger.debug(f"Error: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Extraction Complete!")
    logger.info("=" * 70)
    logger.info(f"   Extracted: {extracted:,}")
    logger.info(f"   Skipped: {errors:,}")
    logger.info(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
