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
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import xml.etree.ElementTree as ET
from collections import Counter

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

# Configuration
MIN_YEAR = 2015
MAX_YEAR = 2025
MAX_FULL_TEXT_CHARS = 50000
MAX_ABSTRACT_CHARS = 2000

# Evidence hierarchy for article types
EVIDENCE_HIERARCHY = {
    'meta-analysis': ('A', 1),
    'systematic-review': ('A', 1),
    'systematic review': ('A', 1),
    'practice-guideline': ('A', 1),
    'guideline': ('A', 1),
    'randomized-controlled-trial': ('A', 2),
    'randomized controlled trial': ('A', 2),
    'clinical-trial': ('A', 2),
    'clinical trial': ('A', 2),
    'cohort-study': ('B', 3),
    'cohort study': ('B', 3),
    'review': ('B', 3),
    'case-control': ('B', 4),
    'cross-sectional': ('B', 4),
    'case-report': ('C', 5),
    'case report': ('C', 5),
    'case-series': ('C', 5),
    'editorial': ('D', 6),
    'letter': ('D', 6),
    'comment': ('D', 6),
}


def extract_text(element: Optional[ET.Element], default: str = "") -> str:
    """Extract all text from an element recursively."""
    if element is None:
        return default
    return ' '.join(element.itertext()).strip()


def extract_table_markdown(table_wrap: ET.Element) -> str:
    """Extract table as markdown format."""
    caption = ""
    caption_elem = table_wrap.find('.//caption')
    if caption_elem is not None:
        caption = extract_text(caption_elem)
    
    table = table_wrap.find('.//table')
    if table is None:
        return ""
    
    rows = []
    
    # Extract headers
    header_row = table.find('.//thead/tr') or table.find('.//tr')
    if header_row is not None:
        headers = []
        for cell in header_row.findall('.//th') + header_row.findall('.//td'):
            cell_text = extract_text(cell).replace('|', '\\|')
            headers.append(cell_text)
        if headers:
            rows.append("| " + " | ".join(headers) + " |")
            rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Extract data rows
    for row in table.findall('.//tbody/tr') + table.findall('.//tr'):
        if row == header_row:
            continue
        cells = []
        for cell in row.findall('.//td'):
            cell_text = extract_text(cell).replace('|', '\\|')
            cells.append(cell_text)
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    
    markdown = "\n".join(rows)
    if caption:
        markdown = f"**{caption}**\n\n{markdown}"
    
    return markdown


def extract_table_row_by_row(table_wrap: ET.Element) -> str:
    """Extract table in row-by-row format for structured queries."""
    caption_elem = table_wrap.find('.//caption')
    caption = extract_text(caption_elem) if caption_elem is not None else ""
    
    table = table_wrap.find('.//table')
    if table is None:
        return ""
    
    # Get headers
    header_row = table.find('.//thead/tr') or table.find('.//tr')
    headers = []
    if header_row is not None:
        for cell in header_row.findall('.//th') + header_row.findall('.//td'):
            headers.append(extract_text(cell))
    
    # Extract rows
    row_texts = []
    row_num = 0
    for row in table.findall('.//tbody/tr') + table.findall('.//tr'):
        if row == header_row:
            continue
        row_num += 1
        
        cells = []
        for j, cell in enumerate(row.findall('.//td')):
            header = headers[j] if j < len(headers) else f"Column{j+1}"
            value = extract_text(cell)
            cells.append(f"{header}: {value}")
        
        if cells:
            row_texts.append(f"Row {row_num}: {', '.join(cells)}")
    
    result = f"Table: {caption}. " if caption else "Table: "
    result += ". ".join(row_texts)
    return result


def extract_tables(root: ET.Element) -> List[Dict[str, Any]]:
    """Extract all tables from article with multiple formats."""
    tables = []
    
    for i, table_wrap in enumerate(root.findall('.//table-wrap')[:5], 1):
        caption_elem = table_wrap.find('.//caption')
        caption = extract_text(caption_elem) if caption_elem is not None else ""
        
        tables.append({
            "id": f"table-{i}",
            "caption": caption,
            "markdown": extract_table_markdown(table_wrap),
            "row_by_row": extract_table_row_by_row(table_wrap),
        })
    
    return tables


def classify_evidence_grade(article_type: str, pub_types: List[str]) -> tuple:
    """Classify evidence grade based on article type."""
    article_type_lower = article_type.lower().replace('_', '-').replace(' ', '-')
    pub_types_lower = [pt.lower() for pt in pub_types]
    
    # Check article type first
    for pattern, (grade, level) in EVIDENCE_HIERARCHY.items():
        if pattern in article_type_lower:
            return grade, level
        for pt in pub_types_lower:
            if pattern in pt:
                return grade, level
    
    # Default
    return 'B', 3


def extract_affiliations(root: ET.Element) -> tuple:
    """Extract author affiliations and determine country."""
    affiliations = []
    countries = []
    
    for aff in root.findall('.//aff'):
        aff_text = extract_text(aff)
        if aff_text:
            affiliations.append(aff_text)
            
            # Try to extract country (last comma-separated part)
            parts = aff_text.split(',')
            if parts:
                potential_country = parts[-1].strip()
                if len(potential_country) < 50:
                    countries.append(potential_country)
    
    # Determine primary country (most common)
    country = None
    if countries:
        country_counts = Counter(countries)
        country = country_counts.most_common(1)[0][0]
    
    return affiliations[:10], country


def parse_xml_file(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a single PMC XML file and extract all data."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get PMCID from filename
        pmcid = xml_path.stem
        
        # Extract year and filter
        year = None
        for pub_date in root.findall('.//pub-date'):
            year_elem = pub_date.find('year')
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                    break
                except ValueError:
                    continue
        
        if year is None or year < MIN_YEAR or year > MAX_YEAR:
            return None
        
        # Extract title
        title = extract_text(root.find('.//article-title'))
        if not title:
            return None
        
        # Extract abstract
        abstract_parts = []
        for abstract in root.findall('.//abstract'):
            for p in abstract.findall('.//p'):
                text = extract_text(p)
                if text:
                    abstract_parts.append(text)
        abstract = ' '.join(abstract_parts)
        
        # Extract full text from body (ONLY available in OA articles)
        full_text = ""
        body = root.find('.//body')
        if body is not None:
            sections = []
            for sec in body.findall('.//sec'):
                sec_title = extract_text(sec.find('title'))
                sec_parts = []
                for p in sec.findall('p'):
                    text = extract_text(p)
                    if text and len(text) > 20:
                        sec_parts.append(text)
                if sec_parts:
                    sec_text = '\n'.join(sec_parts)
                    if sec_title:
                        sections.append(f"\n## {sec_title}\n{sec_text}")
                    else:
                        sections.append(sec_text)
            
            # Also get direct paragraphs
            for p in body.findall('p'):
                text = extract_text(p)
                if text and len(text) > 20:
                    sections.append(text)
            
            full_text = '\n'.join(sections)
            if len(full_text) > MAX_FULL_TEXT_CHARS:
                full_text = full_text[:MAX_FULL_TEXT_CHARS]
        
        if not abstract and not full_text:
            return None
        
        # Extract tables
        tables = extract_tables(root)
        
        # Extract journal
        journal = extract_text(root.find('.//journal-title')) or "Unknown"
        
        # Extract article type and publication types
        article_type = ""
        article_elem = root.find('.//article')
        if article_elem is not None:
            article_type = article_elem.get('article-type', '')
        
        pub_types = []
        for subj in root.findall('.//article-categories//subject'):
            pt = extract_text(subj)
            if pt:
                pub_types.append(pt)
        
        # Classify evidence
        evidence_grade, evidence_level = classify_evidence_grade(article_type, pub_types)
        
        # Extract authors
        authors = []
        for contrib in root.findall('.//contrib[@contrib-type="author"]'):
            name = contrib.find('name')
            if name is not None:
                surname = extract_text(name.find('surname'))
                given = extract_text(name.find('given-names'))
                if surname:
                    authors.append(f"{surname}, {given}" if given else surname)
        
        # Extract keywords
        keywords = []
        for kwd in root.findall('.//kwd'):
            text = extract_text(kwd)
            if text:
                keywords.append(text)
        
        # Extract MeSH terms
        mesh_terms = []
        for kwd_group in root.findall('.//kwd-group[@kwd-group-type="MESH"]'):
            for kwd in kwd_group.findall('kwd'):
                text = extract_text(kwd)
                if text:
                    mesh_terms.append(text)
        
        # Extract affiliations and country
        affiliations, country = extract_affiliations(root)
        
        # Extract identifiers
        pmid = None
        doi = None
        for article_id in root.findall('.//article-id'):
            id_type = article_id.get('pub-id-type', '')
            if id_type == 'pmid':
                pmid = article_id.text
            elif id_type == 'doi':
                doi = article_id.text
        
        # Section structure analysis
        section_titles = []
        has_methods = False
        has_results = False
        has_discussion = False
        
        for sec in root.findall('.//sec'):
            title_elem = sec.find('title')
            if title_elem is not None:
                sec_title = extract_text(title_elem)
                section_titles.append(sec_title)
                sec_lower = sec_title.lower()
                if 'method' in sec_lower:
                    has_methods = True
                if 'result' in sec_lower:
                    has_results = True
                if 'discussion' in sec_lower or 'conclusion' in sec_lower:
                    has_discussion = True
        
        # Counts
        figure_count = len(root.findall('.//fig'))
        table_count = len(root.findall('.//table-wrap'))
        
        # Volume, issue, pages
        volume = extract_text(root.find('.//volume'))
        issue = extract_text(root.find('.//issue'))
        fpage = root.findtext('.//fpage') or ""
        lpage = root.findtext('.//lpage') or ""
        pages = f"{fpage}-{lpage}" if fpage and lpage and lpage != fpage else fpage
        
        # Build article record
        return {
            # Identifiers
            "pmcid": pmcid,
            "pmid": pmid,
            "doi": doi,
            
            # Content
            "title": title,
            "abstract": abstract[:MAX_ABSTRACT_CHARS] if abstract else "",
            "full_text": full_text,
            "tables": tables,
            
            # Publication info
            "year": year,
            "journal": journal,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            
            # Classification
            "article_type": article_type,
            "publication_type_list": pub_types[:5],
            "evidence_grade": evidence_grade,
            "evidence_level": evidence_level,
            
            # Keywords and MeSH
            "keywords": keywords[:20],
            "mesh_terms": mesh_terms[:20],
            
            # Authors
            "authors": authors[:20],
            "first_author": authors[0] if authors else None,
            "author_count": len(authors),
            
            # Geography
            "country": country,
            "institutions": affiliations[:5],
            
            # Structure
            "section_titles": section_titles[:10],
            "has_methods": has_methods,
            "has_results": has_results,
            "has_discussion": has_discussion,
            "figure_count": figure_count,
            "table_count": table_count,
            
            # Access
            "is_open_access": True,
            "has_full_text": bool(full_text),
            
            # Metadata
            "source": "pmc",
            "source_file": str(xml_path),
            "extracted_at": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        logger.debug(f"Error parsing {xml_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract PMC articles from XML to JSONL")
    parser.add_argument("--xml-dir", type=Path, required=True, help="Directory with XML files")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    
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
    
    # Also check subdirectories
    if not xml_files:
        xml_files = list(args.xml_dir.rglob("*.xml"))
    
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
                article = parse_xml_file(xml_file)
                if article:
                    f.write(json.dumps(article) + '\n')
                    extracted += 1
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

