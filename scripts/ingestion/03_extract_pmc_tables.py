#!/usr/bin/env python3
"""
Enhanced PMC Extraction with Table-Aware Parsing.

Extracts articles from PMC XML files with multiple table representation formats:
1. Markdown format (primary for embedding)
2. Row-by-row format (for structured queries)
3. Column-wise format (for aggregations)
4. Table summary (for context)

Processes both existing full-text XML files and newly downloaded abstract data.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import xml.etree.ElementTree as ET
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


def extract_text(element: Optional[ET.Element], default: str = "") -> str:
    """Extract text from XML element."""
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
    """Extract table in row-by-row format."""
    caption = ""
    caption_elem = table_wrap.find('.//caption')
    if caption_elem is not None:
        caption = extract_text(caption_elem)
    
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
    for i, row in enumerate(table.findall('.//tbody/tr') + table.findall('.//tr'), 1):
        if row == header_row:
            continue
        
        cells = []
        for j, cell in enumerate(row.findall('.//td')):
            header = headers[j] if j < len(headers) else f"Column{j+1}"
            value = extract_text(cell)
            cells.append(f"{header}: {value}")
        
        if cells:
            row_texts.append(f"Row {i}: {', '.join(cells)}")
    
    result = f"Table: {caption}. " if caption else "Table: "
    result += ". ".join(row_texts)
    return result


def extract_table_column_wise(table_wrap: ET.Element) -> str:
    """Extract table in column-wise format."""
    table = table_wrap.find('.//table')
    if table is None:
        return ""
    
    # Get headers
    header_row = table.find('.//thead/tr') or table.find('.//tr')
    headers = []
    if header_row is not None:
        for cell in header_row.findall('.//th') + header_row.findall('.//td'):
            headers.append(extract_text(cell))
    
    # Collect column data
    columns_data = {h: [] for h in headers}
    
    for row in table.findall('.//tbody/tr') + table.findall('.//tr'):
        if row == header_row:
            continue
        
        cells = row.findall('.//td')
        for j, cell in enumerate(cells):
            if j < len(headers):
                value = extract_text(cell)
                columns_data[headers[j]].append(value)
    
    # Format as column-wise text
    column_texts = []
    for header, values in columns_data.items():
        if values:
            column_texts.append(f"{header}: {', '.join(values)}")
    
    return ". ".join(column_texts)


def extract_tables(root: ET.Element) -> List[Dict[str, Any]]:
    """Extract all tables from article with multiple formats."""
    tables = []
    
    for i, table_wrap in enumerate(root.findall('.//table-wrap'), 1):
        table_id = f"table-{i}"
        
        # Extract caption
        caption = ""
        caption_elem = table_wrap.find('.//caption')
        if caption_elem is not None:
            caption = extract_text(caption_elem)
        
        # Extract in multiple formats
        markdown = extract_table_markdown(table_wrap)
        row_by_row = extract_table_row_by_row(table_wrap)
        column_wise = extract_table_column_wise(table_wrap)
        
        # Get surrounding context (previous and next paragraphs)
        context_parts = []
        parent = table_wrap.getparent()
        if parent is not None:
            # Get previous sibling paragraphs
            prev_elem = table_wrap.getprevious()
            if prev_elem is not None and prev_elem.tag == 'p':
                context_parts.append(extract_text(prev_elem))
            
            # Get next sibling paragraphs
            next_elem = table_wrap.getnext()
            if next_elem is not None and next_elem.tag == 'p':
                context_parts.append(extract_text(next_elem))
        
        context = " ".join(context_parts)
        
        tables.append({
            "id": table_id,
            "caption": caption,
            "markdown": markdown,
            "row_by_row": row_by_row,
            "column_wise": column_wise,
            "context": context
        })
    
    return tables


def extract_abstract(root: ET.Element) -> str:
    """Extract abstract text."""
    parts = []
    for abstract in root.findall('.//abstract'):
        for p in abstract.findall('.//p'):
            text = extract_text(p)
            if text:
                parts.append(text)
    return ' '.join(parts)


def extract_body_text(root: ET.Element) -> str:
    """Extract full body text."""
    parts = []
    body = root.find('.//body')
    if body is None:
        return ""
    
    for sec in body.findall('.//sec'):
        title = extract_text(sec.find('title'))
        if title:
            parts.append(f"\n## {title}\n")
        for p in sec.findall('p'):
            text = extract_text(p)
            if text and len(text) > 20:
                parts.append(text)
    
    for p in body.findall('p'):
        text = extract_text(p)
        if text and len(text) > 20:
            parts.append(text)
    
    full_text = '\n'.join(parts)
    return full_text[:MAX_FULL_TEXT_CHARS] if len(full_text) > MAX_FULL_TEXT_CHARS else full_text


def extract_authors(root: ET.Element) -> List[str]:
    """Extract authors."""
    authors = []
    for contrib in root.findall('.//contrib[@contrib-type="author"]'):
        name = contrib.find('name')
        if name is not None:
            surname = extract_text(name.find('surname'))
            given = extract_text(name.find('given-names'))
            if surname:
                authors.append(f"{surname}, {given}" if given else surname)
    return authors[:20]


def extract_keywords(root: ET.Element) -> List[str]:
    """Extract keywords."""
    keywords = []
    for kwd_group in root.findall('.//kwd-group'):
        for kwd in kwd_group.findall('kwd'):
            text = extract_text(kwd)
            if text:
                keywords.append(text)
    return keywords[:20]


def extract_mesh_terms(root: ET.Element) -> tuple[List[str], List[str], List[str]]:
    """Extract MeSH terms (all, major, and minor)."""
    mesh_major = []
    mesh_minor = []
    all_mesh = []
    
    # Extract from kwd-group with MESH type
    for kwd_group in root.findall('.//kwd-group[@kwd-group-type="MESH"]'):
        for kwd in kwd_group.findall('kwd'):
            text = extract_text(kwd)
            if text:
                all_mesh.append(text)
                # Check if marked as major
                if kwd.get('content-type') == 'major' or 'major' in str(kwd.getparent()):
                    mesh_major.append(text)
                else:
                    mesh_minor.append(text)
    
    # Also try to extract from subject elements
    for subject in root.findall('.//subject[@subject-type="heading"]'):
        text = extract_text(subject)
        if text and text not in all_mesh:
            all_mesh.append(text)
            if subject.get('content-type') == 'major':
                mesh_major.append(text)
            else:
                mesh_minor.append(text)
    
    return all_mesh[:20], mesh_major[:10], mesh_minor[:10]


def extract_publication_types(root: ET.Element) -> tuple[str, List[str]]:
    """Extract publication types."""
    pub_types = []
    pub_type_major = None
    
    # From article-categories
    for subj_group in root.findall('.//article-categories/subj-group[@subj-group-type="article-type"]'):
        for subject in subj_group.findall('subject'):
            pub_type = extract_text(subject)
            if pub_type:
                pub_types.append(pub_type)
                if not pub_type_major:
                    pub_type_major = pub_type
    
    # From publication-type (if available)
    for pub_type_elem in root.findall('.//publication-type'):
        pub_type = extract_text(pub_type_elem)
        if pub_type and pub_type not in pub_types:
            pub_types.append(pub_type)
    
    return pub_type_major or "research", pub_types


def extract_volume_issue_pages(root: ET.Element) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract volume, issue, and page numbers."""
    volume = None
    issue = None
    pages = None
    
    # Volume
    volume_elem = root.find('.//volume')
    if volume_elem is not None:
        volume = extract_text(volume_elem)
    
    # Issue
    issue_elem = root.find('.//issue')
    if issue_elem is not None:
        issue = extract_text(issue_elem)
    
    # Pages
    fpage = root.findtext('.//fpage')
    lpage = root.findtext('.//lpage')
    if fpage:
        if lpage and lpage != fpage:
            pages = f"{fpage}-{lpage}"
        else:
            pages = fpage
    
    return volume, issue, pages


def extract_publication_date(root: ET.Element) -> tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    """Extract full publication date."""
    for pub_date in root.findall('.//pub-date'):
        year_elem = pub_date.find('year')
        month_elem = pub_date.find('month')
        day_elem = pub_date.find('day')
        
        year = None
        month = None
        day = None
        date_str = None
        
        if year_elem is not None and year_elem.text:
            try:
                year = int(year_elem.text)
            except ValueError:
                pass
        
        if month_elem is not None and month_elem.text:
            try:
                month = int(month_elem.text)
            except ValueError:
                # Handle month names
                month_map = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                month = month_map.get(month_elem.text.lower()[:3])
        
        if day_elem is not None and day_elem.text:
            try:
                day = int(day_elem.text)
            except ValueError:
                pass
        
        if year:
            if month and day:
                date_str = f"{year}-{month:02d}-{day:02d}"
            elif month:
                date_str = f"{year}-{month:02d}-01"
            else:
                date_str = f"{year}-01-01"
        
        if year:  # Return first valid date found
            return year, month, day, date_str
    
    return None, None, None, None


def extract_affiliations(root: ET.Element) -> tuple[List[str], Optional[str]]:
    """Extract author affiliations and determine country."""
    affiliations = []
    countries = []
    
    # Extract affiliations
    for aff in root.findall('.//aff'):
        aff_text = extract_text(aff)
        if aff_text:
            affiliations.append(aff_text)
            
            # Try to extract country (simple heuristic - last part often country)
            # Could be improved with country name lookup
            parts = aff_text.split(',')
            if parts:
                potential_country = parts[-1].strip()
                if len(potential_country) < 50:  # Reasonable country name length
                    countries.append(potential_country)
    
    # Determine primary country (most common)
    country = None
    if countries:
        from collections import Counter
        country_counts = Counter(countries)
        country = country_counts.most_common(1)[0][0]
    
    return affiliations[:10], country


def classify_evidence_grade(article_type: str, pub_types: List[str]) -> tuple[Optional[str], Optional[int]]:
    """Classify evidence grade based on article type."""
    article_type_lower = article_type.lower()
    pub_types_lower = [pt.lower() for pt in pub_types]
    
    # Check for highest evidence
    if any(term in article_type_lower or any(term in pt for pt in pub_types_lower) 
           for term in ['systematic review', 'meta-analysis', 'meta analysis']):
        return "A", 1
    
    if any(term in article_type_lower or any(term in pt for pt in pub_types_lower)
           for term in ['guideline', 'practice guideline', 'consensus']):
        return "A", 1
    
    # Strong evidence
    if any(term in article_type_lower or any(term in pt for pt in pub_types_lower)
           for term in ['randomized controlled trial', 'rct', 'clinical trial']):
        return "A", 2
    
    # Good evidence
    if any(term in article_type_lower or any(term in pt for pt in pub_types_lower)
           for term in ['review', 'cohort study', 'observational study']):
        return "B", 3
    
    # Standard evidence
    if any(term in article_type_lower or any(term in pt for pt in pub_types_lower)
           for term in ['case-control', 'case control', 'cross-sectional']):
        return "B", 4
    
    # Lower evidence
    if any(term in article_type_lower or any(term in pt for pt in pub_types_lower)
           for term in ['case report', 'case series', 'expert opinion']):
        return "C", 5
    
    # Default
    return "B", 3


def extract_section_structure(root: ET.Element) -> tuple[List[str], bool, bool, bool]:
    """Extract section structure information."""
    section_titles = []
    has_methods = False
    has_results = False
    has_discussion = False
    
    for sec in root.findall('.//sec'):
        title_elem = sec.find('title')
        if title_elem is not None:
            title = extract_text(title_elem).lower()
            section_titles.append(extract_text(title_elem))
            
            if 'method' in title:
                has_methods = True
            if 'result' in title:
                has_results = True
            if 'discussion' in title or 'conclusion' in title:
                has_discussion = True
    
    return section_titles[:20], has_methods, has_results, has_discussion


def extract_figure_table_counts(root: ET.Element) -> tuple[int, int]:
    """Extract count of figures and tables."""
    figure_count = len(root.findall('.//fig'))
    table_count = len(root.findall('.//table-wrap'))
    return figure_count, table_count


def extract_year(root: ET.Element) -> Optional[int]:
    """Extract publication year."""
    for pub_date in root.findall('.//pub-date'):
        year_elem = pub_date.find('year')
        if year_elem is not None and year_elem.text:
            try:
                return int(year_elem.text)
            except ValueError:
                continue
    return None


def extract_doi(root: ET.Element) -> Optional[str]:
    """Extract DOI."""
    for article_id in root.findall('.//article-id'):
        if article_id.get('pub-id-type') == 'doi':
            return article_id.text
    return None


def extract_pmid(root: ET.Element) -> Optional[str]:
    """Extract PMID."""
    for article_id in root.findall('.//article-id'):
        if article_id.get('pub-id-type') == 'pmid':
            return article_id.text
    return None


def parse_xml_file(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse a single PMC XML file with table extraction."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        pmcid = xml_path.stem
        year = extract_year(root)
        if year is None or year < MIN_YEAR or year > MAX_YEAR:
            return None
        
        title = extract_text(root.find('.//article-title'))
        if not title:
            return None
        
        abstract = extract_abstract(root)
        full_text = extract_body_text(root)
        tables = extract_tables(root)
        
        if not abstract and not full_text:
            return None
        
        journal = extract_text(root.find('.//journal-title')) or "Unknown"
        
        # Extract enhanced metadata
        article_type, publication_types = extract_publication_types(root)
        mesh_terms, mesh_major, mesh_minor = extract_mesh_terms(root)
        volume, issue, pages = extract_volume_issue_pages(root)
        pub_year, month, day, publication_date = extract_publication_date(root)
        affiliations, country = extract_affiliations(root)
        section_titles, has_methods, has_results, has_discussion = extract_section_structure(root)
        figure_count, table_count = extract_figure_table_counts(root)
        evidence_grade, evidence_level = classify_evidence_grade(article_type, publication_types)
        
        authors_list = extract_authors(root)
        
        return {
            "pmcid": pmcid,
            "pmid": extract_pmid(root),
            "doi": extract_doi(root),
            "title": title,
            "abstract": abstract[:MAX_ABSTRACT_CHARS] if abstract else "",
            "full_text": full_text,
            "tables": tables,
            
            # Publication Details
            "year": pub_year or year,
            "publication_date": publication_date,
            "month": month,
            "day": day,
            "journal": journal,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "language": "en",  # Default, could extract from XML if available
            
            # Article Classification
            "article_type": article_type,
            "publication_type_list": publication_types,
            "publication_type_major": publication_types[0] if publication_types else article_type,
            "evidence_grade": evidence_grade,
            "evidence_level": evidence_level,
            
            # Medical Indexing
            "keywords": extract_keywords(root),
            "mesh_terms": mesh_terms,
            "mesh_major": mesh_major,
            "mesh_minor": mesh_minor,
            
            # Authors and Affiliations
            "authors": authors_list,
            "first_author": authors_list[0] if authors_list else None,
            "author_count": len(authors_list),
            "author_affiliations": affiliations,
            "country": country,
            "institutions": affiliations[:5],  # Top 5 institutions
            
            # Article Structure
            "section_titles": section_titles,
            "has_methods": has_methods,
            "has_results": has_results,
            "has_discussion": has_discussion,
            "figure_count": figure_count,
            "table_count": table_count,
            
            # Access
            "is_open_access": True,  # PMC OA articles are all OA
            "has_full_text": bool(full_text),
            
            # Technical
            "source_file": str(xml_path),
            "extracted_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.debug(f"Error parsing {xml_path}: {e}")
        return None


def process_existing_xml_files(xml_dir: Path, output_file: Path):
    """Process existing XML files from EC2 or local directory."""
    logger.info(f"Processing XML files from {xml_dir}...")
    
    if not xml_dir.exists():
        logger.error(f"Directory not found: {xml_dir}")
        return
    
    xml_files = list(xml_dir.glob("*.xml"))
    logger.info(f"Found {len(xml_files)} XML files")
    
    processed = 0
    with open(output_file, 'w') as f_out:
        for xml_file in tqdm(xml_files, desc="Extracting"):
            article = parse_xml_file(xml_file)
            if article:
                f_out.write(json.dumps(article) + '\n')
                processed += 1
    
    logger.info(f"✅ Processed {processed} articles")


def process_abstract_jsonl(abstract_file: Path, output_file: Path):
    """Process abstract JSONL file (from PubMed API)."""
    logger.info(f"Processing abstracts from {abstract_file}...")
    
    if not abstract_file.exists():
        logger.warning(f"Abstract file not found: {abstract_file}")
        return
    
    processed = 0
    with open(output_file, 'a') as f_out:  # Append mode
        with open(abstract_file, 'r') as f_in:
            for line in tqdm(f_in, desc="Processing abstracts"):
                if not line.strip():
                    continue
                
                try:
                    article = json.loads(line)
                    
                    # Convert to standard format
                    output_article = {
                        "pmcid": article.get("pmcid"),
                        "pmid": article.get("pmid"),
                        "doi": article.get("doi"),
                        "title": article.get("title", ""),
                        "abstract": article.get("abstract", "")[:MAX_ABSTRACT_CHARS],
                        "full_text": "",  # No full text for abstracts
                        "tables": [],  # No tables in abstracts
                        "year": article.get("year"),
                        "journal": article.get("journal", ""),
                        "article_type": "research",  # Default
                        "keywords": [],
                        "authors": article.get("authors", []),
                        "first_author": article.get("authors", [""])[0] if article.get("authors") else None,
                        "country": None,
                        "evidence_grade": None,
                        "has_full_text": False,
                        "source_file": article.get("source", "pubmed_api"),
                        "extracted_at": article.get("extracted_at", datetime.utcnow().isoformat())
                    }
                    
                    f_out.write(json.dumps(output_article) + '\n')
                    processed += 1
                
                except Exception as e:
                    logger.debug(f"Error processing abstract: {e}")
                    continue
    
    logger.info(f"✅ Processed {processed} abstracts")


def main():
    """Main extraction function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract PMC articles with table support")
    parser.add_argument("--xml-dir", type=Path, help="Directory with XML files")
    parser.add_argument("--abstract-file", type=Path, help="JSONL file with abstracts")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--from-ec2", action="store_true", help="Process files from EC2")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📄 Extracting PMC Articles with Table Support")
    print("=" * 70)
    
    # Process XML files
    if args.xml_dir or args.from_ec2:
        if args.from_ec2:
            # Download files from EC2 first (simplified - would need full implementation)
            logger.info("EC2 processing not fully implemented - use --xml-dir for local files")
        else:
            process_existing_xml_files(args.xml_dir, args.output)
    
    # Process abstract files
    if args.abstract_file:
        process_abstract_jsonl(args.abstract_file, args.output)
    
    print(f"\n✅ Extraction complete! Output: {args.output}")


if __name__ == "__main__":
    main()

