# PRD: PMC & DailyMed Data Ingestion Pipeline

## Production Implementation Guide

**Version:** 1.0  
**Date:** December 2024  
**Status:** Production Ready  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites](#3-prerequisites)
4. [AWS EC2 Setup](#4-aws-ec2-setup)
5. [PMC Data Download & Extraction](#5-pmc-data-download--extraction)
6. [DailyMed Data Download & Extraction](#6-dailymed-data-download--extraction)
7. [Qdrant Cloud Setup](#7-qdrant-cloud-setup)
8. [Data Ingestion Pipeline](#8-data-ingestion-pipeline)
9. [Metadata Schema for Reranking](#9-metadata-schema-for-reranking)
10. [Monthly Update Process](#10-monthly-update-process)
11. [Monitoring & Troubleshooting](#11-monitoring--troubleshooting)
12. [Lessons Learned & Optimizations](#12-lessons-learned--optimizations)
13. [Cost Estimates](#13-cost-estimates)
14. [Appendix: Complete Scripts](#14-appendix-complete-scripts)

---

## 1. Executive Summary

### Purpose
Build a production-grade medical literature RAG (Retrieval-Augmented Generation) pipeline by ingesting:
- **1.2M+ PubMed Central (PMC)** open-access full-text articles
- **51K+ DailyMed** FDA drug labels

### Key Outcomes
| Metric | Value |
|--------|-------|
| Total documents ingested | 1,290,211 |
| PMC articles | 1,239,285 |
| DailyMed drug records | 51,121 |
| Vector dimensions | 1024 (dense) |
| Embedding model | mixedbread-ai/mxbai-embed-large-v1 |
| Quantization | Binary |
| Total ingestion time | ~75 minutes |

### Technology Stack
- **Compute:** AWS EC2 (r6i.4xlarge)
- **Vector Database:** Qdrant Cloud
- **Embeddings:** Qdrant Cloud Inference
- **Data Sources:** PMC S3, DailyMed FTP

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Data Ingestion Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐ │
│  │ PMC S3       │────▶│ AWS EC2      │────▶│ Qdrant Cloud                 │ │
│  │ Open Access  │     │ r6i.4xlarge  │     │ (Cloud Inference Enabled)    │ │
│  └──────────────┘     │              │     │                              │ │
│                       │ - Download   │     │ - Dense vectors (1024-d)     │ │
│  ┌──────────────┐     │ - Extract    │     │ - Binary quantization        │ │
│  │ DailyMed     │────▶│ - Transform  │     │ - Payload indexes            │ │
│  │ SPL Archives │     │ - Ingest     │     │ - 4 shards                   │ │
│  └──────────────┘     └──────────────┘     └──────────────────────────────┘ │
│                                                                              │
│  Data Flow:                                                                  │
│  1. Download from S3/FTP ──▶ 2. Extract XML ──▶ 3. Parse to JSONL           │
│  4. Stream to Qdrant ──▶ 5. Cloud Inference embeds ──▶ 6. Store + Index     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Prerequisites

### 3.1 AWS Account Requirements
- AWS account with EC2 access
- IAM role with S3 read permissions for `pmc-oa-opendata` bucket
- SSH key pair (.pem file) for EC2 access
- VPC with internet access

### 3.2 Qdrant Cloud Account
- Qdrant Cloud account (https://cloud.qdrant.io)
- Cluster provisioned in `us-east-1` (same region as EC2)
- **Cloud Inference enabled** for `mixedbread-ai/mxbai-embed-large-v1`
- API key generated

### 3.3 Local Development Environment
```bash
# Required tools
- Python 3.10+
- AWS CLI configured
- SSH client
- Git
```

### 3.4 Environment Variables
Create a `.env` file:
```bash
# Qdrant Configuration
QDRANT_URL=https://your-cluster-id.us-east-1-1.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# EC2 Configuration
EC2_HOST=your-ec2-public-ip
EC2_USER=ec2-user
EC2_SSH_KEY_PATH=/path/to/your-key.pem

# Optional: AWS Configuration
AWS_REGION=us-east-1
```

---

## 4. AWS EC2 Setup

### 4.1 Instance Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Instance Type** | `r6i.4xlarge` | Memory-optimized for XML parsing |
| **vCPUs** | 16 | Parallel processing |
| **RAM** | 128 GB | Large XML files in memory |
| **Storage** | 500 GB gp3 | PMC data ~200GB, DailyMed ~50GB |
| **Region** | us-east-1 | Same region as Qdrant Cloud |
| **AMI** | Amazon Linux 2023 | Latest, stable |

### 4.2 Launch EC2 Instance

```bash
# Via AWS CLI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type r6i.4xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx \
  --block-device-mappings '[{
    "DeviceName": "/dev/xvda",
    "Ebs": {
      "VolumeSize": 500,
      "VolumeType": "gp3",
      "Iops": 3000,
      "Throughput": 125
    }
  }]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pmc-ingestion}]'
```

### 4.3 Initial EC2 Configuration

```bash
# SSH into EC2
ssh -i /path/to/your-key.pem ec2-user@<EC2_PUBLIC_IP>

# Update system
sudo yum update -y

# Install Python and dependencies
sudo yum install -y python3 python3-pip git

# Install required Python packages
pip3 install --user \
  qdrant-client \
  tqdm \
  lxml \
  requests \
  python-dotenv

# Create data directories
sudo mkdir -p /data/pmc_fulltext/xml
sudo mkdir -p /data/dailymed/xml
sudo chown -R ec2-user:ec2-user /data

# Configure AWS CLI for S3 access (no credentials needed for public bucket)
aws configure set default.region us-east-1
```

### 4.4 Security Group Configuration

| Type | Port | Source | Purpose |
|------|------|--------|---------|
| SSH | 22 | Your IP | Admin access |
| HTTPS | 443 | 0.0.0.0/0 | Qdrant Cloud API |

---

## 5. PMC Data Download & Extraction

### 5.1 Data Source

PMC Open Access articles are available via AWS S3 (no authentication required):

```
s3://pmc-oa-opendata/oa_comm/xml/all/
```

**IMPORTANT - Understanding PMC Full Text Access:**

| Subset | S3 Path | License | Full Text? |
|--------|---------|---------|------------|
| **oa_comm** | `s3://pmc-oa-opendata/oa_comm/xml/all/` | Commercial-friendly (CC BY, CC0) | ✅ **Full text available** |
| **oa_noncomm** | `s3://pmc-oa-opendata/oa_noncomm/xml/all/` | Non-commercial only | ✅ Full text available |
| **oa_other** | `s3://pmc-oa-opendata/oa_other/xml/all/` | Other OA licenses | ✅ Full text available |
| Non-OA articles | Not in S3 | Restricted | ❌ Abstract only |

**We use `oa_comm`** because:
1. Contains **full-text XML** with complete article body, tables, figures
2. **Commercial-friendly license** - can be used in production applications
3. Full-text is embedded in the `<body>` element of JATS XML

**Statistics:**
- Total files: ~5 million XML files
- Total size: ~200 GB compressed
- Content: **Full-text articles in JATS XML format** (not just abstracts)

**Optional: Download ALL Open Access subsets for maximum coverage:**
```bash
# Commercial-friendly (recommended for production)
aws s3 sync s3://pmc-oa-opendata/oa_comm/xml/all/ /data/pmc_fulltext/xml/ --no-sign-request

# Non-commercial (if license allows)
aws s3 sync s3://pmc-oa-opendata/oa_noncomm/xml/all/ /data/pmc_fulltext/xml/ --no-sign-request

# Other OA licenses
aws s3 sync s3://pmc-oa-opendata/oa_other/xml/all/ /data/pmc_fulltext/xml/ --no-sign-request
```

### 5.2 Download Script

Create `/data/download_pmc.py`:

```python
#!/usr/bin/env python3
"""
Download PMC Open Access articles from AWS S3.

This script uses AWS CLI to sync the PMC S3 bucket to local storage.
No AWS credentials required - bucket is publicly accessible.
"""

import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
S3_BUCKET = "s3://pmc-oa-opendata/oa_comm/xml/all/"
LOCAL_DIR = "/data/pmc_fulltext/xml"

def download_pmc():
    """Download PMC articles using AWS CLI sync."""
    
    logger.info(f"Starting PMC download from {S3_BUCKET}")
    logger.info(f"Destination: {LOCAL_DIR}")
    
    # Create destination directory
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    
    # AWS CLI sync command
    # --no-sign-request: No credentials needed for public bucket
    # --only-show-errors: Reduce output verbosity
    cmd = [
        "aws", "s3", "sync",
        S3_BUCKET,
        LOCAL_DIR,
        "--no-sign-request",
        "--only-show-errors"
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode == 0:
        logger.info("✅ PMC download complete!")
    else:
        logger.error(f"❌ Download failed with code {process.returncode}")
    
    return process.returncode

if __name__ == "__main__":
    download_pmc()
```

### 5.3 Run Download

```bash
# Start download in background with logging
nohup python3 /data/download_pmc.py > /data/pmc_download.log 2>&1 &

# Monitor progress
tail -f /data/pmc_download.log

# Check file count (run periodically)
find /data/pmc_fulltext/xml -name "*.xml" | wc -l
```

**Expected Duration:** 4-8 hours (depending on network speed)

### 5.4 PMC Extraction Script

Create `/data/extract_pmc.py`:

```python
#!/usr/bin/env python3
"""
Extract PMC articles from XML to JSONL format.

Extracts:
- Full text content
- Structured metadata for reranking
- Tables in multiple formats
- Author affiliations and institutions
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/pmc_fulltext/extract_pmc.log')
    ]
)
logger = logging.getLogger(__name__)

# Namespaces for JATS XML
NAMESPACES = {
    'xlink': 'http://www.w3.org/1999/xlink',
    'mml': 'http://www.w3.org/1998/Math/MathML'
}

# Evidence hierarchy for article types
EVIDENCE_HIERARCHY = {
    'meta-analysis': 'A',
    'systematic-review': 'A',
    'practice-guideline': 'A',
    'guideline': 'A',
    'randomized-controlled-trial': 'B',
    'rct': 'B',
    'clinical-trial': 'B',
    'cohort-study': 'C',
    'case-control': 'C',
    'cross-sectional': 'C',
    'case-report': 'D',
    'case-series': 'D',
    'review': 'C',
    'editorial': 'E',
    'letter': 'E',
    'comment': 'E',
}


def get_text(element: Optional[ET.Element]) -> str:
    """Extract all text from an element recursively."""
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()


def extract_article(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Extract article data from PMC XML."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find article element
        article = root.find('.//article') or root
        front = article.find('.//front')
        body = article.find('.//body')
        back = article.find('.//back')
        
        if front is None:
            return None
        
        article_meta = front.find('.//article-meta')
        journal_meta = front.find('.//journal-meta')
        
        if article_meta is None:
            return None
        
        # === IDENTIFIERS ===
        pmcid = ""
        pmid = ""
        doi = ""
        
        for article_id in article_meta.findall('.//article-id'):
            id_type = article_id.get('pub-id-type', '')
            if id_type == 'pmc':
                pmcid = f"PMC{article_id.text}" if article_id.text else ""
            elif id_type == 'pmid':
                pmid = article_id.text or ""
            elif id_type == 'doi':
                doi = article_id.text or ""
        
        if not pmcid:
            # Try to extract from filename
            pmcid = xml_path.stem
        
        # === TITLE ===
        title_group = article_meta.find('.//title-group')
        title = get_text(title_group.find('.//article-title')) if title_group else ""
        
        # === ABSTRACT ===
        abstract_elem = article_meta.find('.//abstract')
        abstract = get_text(abstract_elem) if abstract_elem else ""
        
        # === FULL TEXT EXTRACTION ===
        # IMPORTANT: Full text is ONLY available in PMC Open Access articles
        # The <body> element contains the complete article text
        # Non-OA articles will NOT have a <body> element (abstract only)
        full_text = ""
        if body is not None:
            sections = []
            for sec in body.findall('.//sec'):
                sec_title = get_text(sec.find('title'))
                sec_text = get_text(sec)
                if sec_text:
                    sections.append(f"{sec_title}\n{sec_text}" if sec_title else sec_text)
            full_text = "\n\n".join(sections)
            # Note: full_text can be 10,000-100,000+ characters for full articles
        
        # === PUBLICATION DATE ===
        year = None
        month = None
        day = None
        pub_date = article_meta.find('.//pub-date[@pub-type="epub"]') or \
                   article_meta.find('.//pub-date[@pub-type="ppub"]') or \
                   article_meta.find('.//pub-date')
        
        if pub_date is not None:
            year_elem = pub_date.find('year')
            month_elem = pub_date.find('month')
            day_elem = pub_date.find('day')
            year = int(year_elem.text) if year_elem is not None and year_elem.text else None
            month = int(month_elem.text) if month_elem is not None and month_elem.text else None
            day = int(day_elem.text) if day_elem is not None and day_elem.text else None
        
        # === JOURNAL ===
        journal = ""
        if journal_meta is not None:
            journal_title = journal_meta.find('.//journal-title')
            journal = journal_title.text if journal_title is not None else ""
        
        # === ARTICLE TYPE ===
        article_type = article.get('article-type', '')
        
        # === PUBLICATION TYPES ===
        pub_types = []
        for pt in article_meta.findall('.//article-categories//subject'):
            pub_types.append(pt.text or "")
        
        # === EVIDENCE GRADE ===
        evidence_grade = 'D'  # Default
        article_type_lower = article_type.lower().replace(' ', '-')
        for pattern, grade in EVIDENCE_HIERARCHY.items():
            if pattern in article_type_lower:
                evidence_grade = grade
                break
        
        # === KEYWORDS ===
        keywords = []
        for kwd in article_meta.findall('.//kwd'):
            if kwd.text:
                keywords.append(kwd.text.strip())
        
        # === MESH TERMS ===
        mesh_terms = []
        # MeSH terms may be in different locations
        for mesh in article_meta.findall('.//kwd-group[@kwd-group-type="mesh"]//kwd'):
            if mesh.text:
                mesh_terms.append(mesh.text.strip())
        
        # === AUTHORS ===
        authors = []
        author_affiliations = []
        institutions = set()
        country = None
        
        for contrib in article_meta.findall('.//contrib[@contrib-type="author"]'):
            name_elem = contrib.find('.//name')
            if name_elem is not None:
                surname = name_elem.find('surname')
                given = name_elem.find('given-names')
                author_name = f"{surname.text if surname is not None else ''}, {given.text if given is not None else ''}".strip(', ')
                authors.append(author_name)
                
                # Get affiliations
                for aff_ref in contrib.findall('.//xref[@ref-type="aff"]'):
                    aff_id = aff_ref.get('rid', '')
                    aff_elem = article_meta.find(f'.//aff[@id="{aff_id}"]')
                    if aff_elem is not None:
                        aff_text = get_text(aff_elem)
                        author_affiliations.append({
                            'author': author_name,
                            'affiliation': aff_text
                        })
                        
                        # Extract institution and country
                        inst = aff_elem.find('.//institution')
                        if inst is not None and inst.text:
                            institutions.add(inst.text)
                        
                        country_elem = aff_elem.find('.//country')
                        if country_elem is not None and country_elem.text:
                            country = country_elem.text
        
        first_author = authors[0] if authors else None
        
        # === SECTION ANALYSIS ===
        section_titles = []
        has_methods = False
        has_results = False
        has_discussion = False
        
        if body is not None:
            for sec in body.findall('.//sec'):
                title_elem = sec.find('title')
                if title_elem is not None and title_elem.text:
                    sec_title = title_elem.text.lower()
                    section_titles.append(title_elem.text)
                    if 'method' in sec_title:
                        has_methods = True
                    if 'result' in sec_title:
                        has_results = True
                    if 'discussion' in sec_title:
                        has_discussion = True
        
        # === FIGURES AND TABLES ===
        figure_count = len(article.findall('.//fig'))
        table_count = len(article.findall('.//table-wrap'))
        
        # === TABLES EXTRACTION ===
        tables = []
        for i, table_wrap in enumerate(article.findall('.//table-wrap')[:5]):  # Limit to 5 tables
            caption = get_text(table_wrap.find('.//caption'))
            table_elem = table_wrap.find('.//table')
            
            if table_elem is not None:
                # Extract as row-by-row text
                rows = []
                for row in table_elem.findall('.//tr'):
                    cells = [get_text(cell) for cell in row.findall('.//*[self::th or self::td]')]
                    rows.append(" | ".join(cells))
                
                tables.append({
                    'id': f'table-{i+1}',
                    'caption': caption,
                    'row_by_row': "\n".join(rows)
                })
        
        # === VOLUME/ISSUE/PAGES ===
        volume = get_text(article_meta.find('.//volume'))
        issue = get_text(article_meta.find('.//issue'))
        fpage = get_text(article_meta.find('.//fpage'))
        lpage = get_text(article_meta.find('.//lpage'))
        pages = f"{fpage}-{lpage}" if fpage and lpage else fpage
        
        # === BUILD ARTICLE RECORD ===
        article_data = {
            # Identifiers
            'pmcid': pmcid,
            'pmid': pmid,
            'doi': doi,
            
            # Content
            'title': title,
            'abstract': abstract,
            'full_text': full_text,
            'tables': tables,
            
            # Publication info
            'year': year,
            'publication_date': f"{year}-{month:02d}-{day:02d}" if year and month and day else None,
            'month': month,
            'day': day,
            'journal': journal,
            'volume': volume,
            'issue': issue,
            'pages': pages,
            'language': 'en',
            
            # Article classification
            'article_type': article_type,
            'publication_type_list': pub_types[:5],
            'publication_type_major': pub_types[0] if pub_types else None,
            'evidence_grade': evidence_grade,
            'evidence_level': ord(evidence_grade) - ord('A') + 1 if evidence_grade else 5,
            
            # Keywords and MeSH
            'keywords': keywords[:20],
            'mesh_terms': mesh_terms[:20],
            'mesh_major': mesh_terms[:5],
            'mesh_minor': mesh_terms[5:15],
            
            # Authors
            'authors': authors[:10],
            'first_author': first_author,
            'author_count': len(authors),
            'author_affiliations': author_affiliations[:10],
            
            # Geography
            'country': country,
            'institutions': list(institutions)[:5],
            
            # Structure analysis
            'section_titles': section_titles[:10],
            'has_methods': has_methods,
            'has_results': has_results,
            'has_discussion': has_discussion,
            'figure_count': figure_count,
            'table_count': table_count,
            
            # Metadata
            'is_open_access': True,
            'has_full_text': bool(full_text),
            'source_file': str(xml_path),
            'extracted_at': datetime.now().isoformat(),
        }
        
        return article_data
        
    except Exception as e:
        logger.debug(f"Error extracting {xml_path}: {e}")
        return None


def process_batch(xml_files: List[Path]) -> List[Dict[str, Any]]:
    """Process a batch of XML files."""
    results = []
    for xml_file in xml_files:
        article = extract_article(xml_file)
        if article:
            results.append(article)
    return results


def main():
    """Main extraction function."""
    xml_dir = Path("/data/pmc_fulltext/xml")
    output_file = Path("/data/pmc_fulltext/pmc_articles.jsonl")
    
    logger.info("=" * 70)
    logger.info("PMC Article Extraction")
    logger.info("=" * 70)
    
    # Find all XML files
    logger.info(f"Scanning {xml_dir} for XML files...")
    xml_files = list(xml_dir.rglob("*.xml"))
    total_files = len(xml_files)
    logger.info(f"Found {total_files:,} XML files")
    
    # Process files
    extracted = 0
    errors = 0
    
    with open(output_file, 'w') as f:
        with tqdm(total=total_files, desc="Extracting") as pbar:
            # Process in batches for efficiency
            batch_size = 100
            for i in range(0, total_files, batch_size):
                batch = xml_files[i:i+batch_size]
                
                for xml_file in batch:
                    try:
                        article = extract_article(xml_file)
                        if article:
                            f.write(json.dumps(article) + "\n")
                            extracted += 1
                        else:
                            errors += 1
                    except Exception as e:
                        errors += 1
                    
                    pbar.update(1)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Extraction Complete!")
    logger.info("=" * 70)
    logger.info(f"   Extracted: {extracted:,}")
    logger.info(f"   Errors: {errors:,}")
    logger.info(f"   Output: {output_file}")


if __name__ == "__main__":
    main()
```

### 5.5 Run Extraction

```bash
# Run extraction in background
nohup python3 /data/extract_pmc.py > /data/pmc_extraction.log 2>&1 &

# Monitor progress
tail -f /data/pmc_fulltext/extract_pmc.log

# Check output file size
ls -lh /data/pmc_fulltext/pmc_articles.jsonl
```

**Expected Duration:** 3-4 hours for 5M XML files

---

## 6. DailyMed Data Download & Extraction

### 6.1 Data Source

DailyMed provides FDA-approved drug labels in SPL (Structured Product Labeling) XML format:

**URL:** https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm

**Files to download:**
- `dm_spl_release_human_rx_part1.zip` through `dm_spl_release_human_rx_part5.zip`
- Each ZIP contains nested ZIPs with individual drug labels

### 6.2 Download Script

Create `/data/download_dailymed.py`:

```python
#!/usr/bin/env python3
"""
Download DailyMed SPL drug labels.

IMPORTANT: DailyMed ZIPs contain NESTED ZIPs, not XML directly.
Each nested ZIP contains one drug's SPL XML file.
"""

import os
import io
import zipfile
import logging
import requests
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/dailymed/download.log')
    ]
)
logger = logging.getLogger(__name__)

# DailyMed download URLs
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/downloads"
DAILYMED_ZIPS = [
    "dm_spl_release_human_rx_part1.zip",
    "dm_spl_release_human_rx_part2.zip",
    "dm_spl_release_human_rx_part3.zip",
    "dm_spl_release_human_rx_part4.zip",
    "dm_spl_release_human_rx_part5.zip",
]

OUTPUT_DIR = Path("/data/dailymed/xml")


def download_and_extract():
    """Download and extract DailyMed ZIP files."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_extracted = 0
    
    for zip_name in DAILYMED_ZIPS:
        url = f"{DAILYMED_BASE_URL}/{zip_name}"
        zip_path = OUTPUT_DIR / zip_name
        
        logger.info(f"Downloading {zip_name}...")
        
        # Download ZIP
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=zip_name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Extract nested ZIPs
        logger.info(f"Extracting nested ZIPs from {zip_name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as outer_zip:
            nested_zips = [n for n in outer_zip.namelist() if n.endswith('.zip')]
            
            for nested_name in tqdm(nested_zips, desc=f"Extracting {zip_name}"):
                try:
                    with outer_zip.open(nested_name) as nested_data:
                        nested_bytes = nested_data.read()
                        
                        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner_zip:
                            for xml_name in inner_zip.namelist():
                                if xml_name.endswith('.xml'):
                                    xml_content = inner_zip.read(xml_name)
                                    
                                    # Use unique filename
                                    base_name = Path(nested_name).stem
                                    out_path = OUTPUT_DIR / f"{base_name}.xml"
                                    
                                    with open(out_path, 'wb') as f:
                                        f.write(xml_content)
                                    
                                    total_extracted += 1
                
                except Exception as e:
                    logger.debug(f"Error extracting {nested_name}: {e}")
        
        # Remove downloaded ZIP to save space
        zip_path.unlink()
        logger.info(f"Completed {zip_name}, total extracted: {total_extracted:,}")
    
    logger.info(f"✅ Total XML files extracted: {total_extracted:,}")


if __name__ == "__main__":
    download_and_extract()
```

### 6.3 DailyMed Extraction Script

Create `/data/process_dailymed.py`:

```python
#!/usr/bin/env python3
"""
Process DailyMed SPL XML files to JSONL.

Extracts:
- Drug name and active ingredients
- Indications, contraindications, warnings
- Dosage information
- Adverse reactions
- Drug interactions
"""

import os
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# HL7 SPL namespace
NS = {'hl7': 'urn:hl7-org:v3'}

# Section codes for drug label sections
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


def get_text(element: Optional[ET.Element]) -> str:
    """Extract all text from element."""
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()


def parse_spl_xml(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Parse SPL XML and extract drug information."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Set ID
        set_id_elem = root.find('.//hl7:setId', NS)
        set_id = set_id_elem.get('root', '') if set_id_elem is not None else ''
        
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
        
        # Extract sections
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
            'set_id': set_id,
            'drug_name': drug_name,
            'title': title,
            'active_ingredients': active_ingredients,
            'manufacturer': manufacturer,
            'indications': sections.get('indications', ''),
            'contraindications': sections.get('contraindications', ''),
            'dosage': sections.get('dosage', ''),
            'adverse_reactions': sections.get('adverse_reactions', ''),
            'interactions': sections.get('interactions', ''),
            'warnings': sections.get('warnings', '') or sections.get('warnings_precautions', ''),
            'description': sections.get('description', ''),
            'mechanism': sections.get('mechanism', ''),
            'source': 'dailymed',
            'source_file': str(xml_path),
            'extracted_at': datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.debug(f"Error parsing {xml_path}: {e}")
        return None


def main():
    xml_dir = Path("/data/dailymed/xml")
    output_file = Path("/data/dailymed/dailymed_drugs.jsonl")
    
    xml_files = list(xml_dir.glob("*.xml"))
    logger.info(f"Found {len(xml_files):,} XML files")
    
    extracted = 0
    
    with open(output_file, 'w') as f:
        for xml_file in tqdm(xml_files, desc="Processing"):
            drug = parse_spl_xml(xml_file)
            if drug and drug.get('drug_name'):
                f.write(json.dumps(drug) + "\n")
                extracted += 1
    
    logger.info(f"✅ Extracted {extracted:,} drug records to {output_file}")


if __name__ == "__main__":
    main()
```

### 6.4 Run DailyMed Pipeline

```bash
# Download and extract
python3 /data/download_dailymed.py

# Process to JSONL
python3 /data/process_dailymed.py
```

**Expected Duration:** 30-60 minutes

---

## 7. Qdrant Cloud Setup

### 7.1 Create Qdrant Cloud Cluster

1. Go to https://cloud.qdrant.io
2. Create new cluster:
   - **Region:** `us-east-1` (same as EC2)
   - **Configuration:** Production (for 1M+ vectors)
   - **RAM:** 8GB minimum recommended

3. Enable **Cloud Inference**:
   - Go to Cluster Settings → Inference
   - Enable `mixedbread-ai/mxbai-embed-large-v1`

4. Generate API Key:
   - Go to Cluster Settings → API Keys
   - Create key with write permissions
   - Save securely

### 7.2 Create Collection

Create `/data/setup_qdrant.py`:

```python
#!/usr/bin/env python3
"""
Setup Qdrant collection with optimal configuration.

Configuration:
- Dense vectors: 1024-d (mxbai-embed-large-v1)
- Sparse vectors: SPLADE support (optional)
- Binary quantization: Enabled
- Sharding: 4 shards for parallel ingestion
- Payload indexes: year, source, article_type, journal
"""

import os
from qdrant_client import QdrantClient
from qdrant_client import models

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "pmc_medical_rag_fulltext"

# Vector configuration
VECTOR_SIZE = 1024  # mxbai-embed-large-v1
SHARD_NUMBER = 4    # For parallel ingestion


def setup_collection():
    """Create and configure collection."""
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    
    # Delete existing collection if present
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    # Create collection with optimized settings
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
            on_disk=True,  # Store vectors on disk for large collections
        ),
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # IDF weighting for sparse vectors
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=200,
        ),
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=True,  # Keep quantized vectors in RAM for speed
            )
        ),
        shard_number=SHARD_NUMBER,
        replication_factor=1,
    )
    
    print(f"✅ Created collection: {COLLECTION_NAME}")
    
    # Create payload indexes for filtering
    indexes = [
        ("year", models.PayloadSchemaType.INTEGER),
        ("source", models.PayloadSchemaType.KEYWORD),
        ("article_type", models.PayloadSchemaType.KEYWORD),
        ("journal", models.PayloadSchemaType.KEYWORD),
        ("country", models.PayloadSchemaType.KEYWORD),
        ("evidence_grade", models.PayloadSchemaType.KEYWORD),
    ]
    
    for field_name, field_type in indexes:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_type,
            )
            print(f"✅ Created index: {field_name}")
        except Exception as e:
            print(f"⚠️ Index {field_name} may already exist: {e}")
    
    # Verify collection
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n📊 Collection Info:")
    print(f"   Points: {info.points_count:,}")
    print(f"   Status: {info.status}")


if __name__ == "__main__":
    setup_collection()
```

### 7.3 Run Setup

```bash
export QDRANT_URL="https://your-cluster.us-east-1-1.aws.cloud.qdrant.io:6333"
export QDRANT_API_KEY="your-api-key"

python3 /data/setup_qdrant.py
```

---

## 8. Data Ingestion Pipeline

### 8.0 Full Text Embedding Strategy

**CRITICAL: Understanding Full Text Handling**

Due to Qdrant Cloud Inference limits, full text is handled in TWO ways:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Full Text Handling Strategy                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PMC Full-Text Article                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Title (100 chars) + Abstract (500 chars) + Full Text (50,000 chars) │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                    ┌───────────────┴───────────────┐                        │
│                    ▼                               ▼                         │
│  ┌─────────────────────────────┐   ┌─────────────────────────────────────┐  │
│  │   EMBEDDING TEXT (2000 ch)  │   │   PAYLOAD STORAGE (10,000 chars)    │  │
│  │   ─────────────────────────  │   │   ──────────────────────────────────│  │
│  │   Title + Abstract +         │   │   full_text field stores first      │  │
│  │   First ~1400 chars of       │   │   10K chars for RAG context         │  │
│  │   full text                  │   │   generation after retrieval        │  │
│  │                              │   │                                     │  │
│  │   → Used for SEARCH          │   │   → Used for ANSWER GENERATION      │  │
│  └─────────────────────────────┘   └─────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why 2000 char limit for embeddings?**
- Qdrant Cloud Inference has a ~2000 character limit per document
- Exceeding this causes `500 Internal Server Error`
- The limit applies ONLY to the text sent for embedding, not payload storage

**How full text is embedded:**
```python
def create_embedding_text(article):
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    full_text = article.get("full_text", "")
    
    # Combine title + abstract first
    combined = f"{title}. {abstract}"
    
    # Add as much full text as fits within 2000 char limit
    if full_text and len(combined) < 1500:
        remaining_space = 2000 - len(combined) - 10
        combined = f"{combined}\n\n{full_text[:remaining_space]}"
    
    return combined[:2000]  # Strict limit for Cloud Inference
```

**How full text is stored for RAG:**
```python
payload = {
    "title": article.get("title", "")[:300],
    "abstract": article.get("abstract", "")[:1000],
    "full_text": article.get("full_text", "")[:10000],  # Store 10K chars!
    # ... other fields
}
```

**RAG Pipeline Usage:**
1. **Search Phase**: Uses embedding (title + abstract + intro of full text)
2. **Rerank Phase**: Uses `title` + `abstract` from payload
3. **Answer Generation**: Uses `full_text` from payload (10K chars available)

### 8.1 Ingestion Script

Create `/data/ingest_to_qdrant.py`:

```python
#!/usr/bin/env python3
"""
Fast ingestion script using Qdrant Cloud Inference.

Features:
- Cloud Inference for embeddings (no local GPU needed)
- Streaming ingestion for memory efficiency
- Checkpoint/resume support
- Parallel upserts
- Retry logic for robustness

Optimized settings:
- Batch size: 50 (optimal for Cloud Inference)
- Parallel workers: 4 (matches shard count)
- Text limit: 2000 chars (Cloud Inference limit)
"""

import os
import sys
import json
import logging
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "pmc_medical_rag_fulltext"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# Optimized for Cloud Inference
BATCH_SIZE = 50           # Cloud Inference optimal batch size
PARALLEL_WORKERS = 4      # Match shard count
MAX_RETRIES = 3           # Retry failed batches
MAX_TEXT_LENGTH = 2000    # Cloud Inference text limit

CHECKPOINT_FILE = Path("/data/ingest_checkpoint.txt")

# ============================================================================


class Counters:
    """Thread-safe counters."""
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


def get_checkpoint() -> set:
    """Load checkpoint of already ingested IDs."""
    if CHECKPOINT_FILE.exists():
        return set(CHECKPOINT_FILE.read_text().strip().split('\n'))
    return set()


def save_checkpoint(ids: List[str]):
    """Append IDs to checkpoint file."""
    with open(CHECKPOINT_FILE, 'a') as f:
        for id_val in ids:
            f.write(f"{id_val}\n")


def create_embedding_text(article: Dict[str, Any]) -> str:
    """
    Create text for embedding from article data.
    
    IMPORTANT: Cloud Inference has ~2000 char limit!
    We include as much content as possible within this limit:
    - Title (typically 50-200 chars)
    - Abstract (typically 200-500 chars)
    - Beginning of full text (remaining space up to 2000 chars)
    
    Full text is ALSO stored separately in payload (10K chars) for RAG context.
    """
    title = article.get("title", "") or ""
    abstract = article.get("abstract", "") or ""
    full_text = article.get("full_text", "") or ""
    
    # For DailyMed drugs
    if article.get("source") == "dailymed":
        drug_name = article.get("drug_name", "") or ""
        ingredients = ", ".join(article.get("active_ingredients", []))
        indications = article.get("indications", "") or ""
        return f"{drug_name}. {ingredients}. {indications}"[:MAX_TEXT_LENGTH]
    
    # For PMC articles - include full text in embedding!
    combined = f"{title}. {abstract}"
    
    # Add as much full text as fits within the 2000 char limit
    if full_text and len(combined) < MAX_TEXT_LENGTH - 100:
        remaining_space = MAX_TEXT_LENGTH - len(combined) - 10
        combined = f"{combined}\n\n{full_text[:remaining_space]}"
    
    return combined[:MAX_TEXT_LENGTH]


def create_payload(article: Dict[str, Any]) -> Dict[str, Any]:
    """Create payload with metadata for reranking."""
    
    # Common fields
    payload = {
        "source": article.get("source", "pmc"),
        "article_type": article.get("article_type", ""),
    }
    
    # DailyMed specific
    if article.get("source") == "dailymed":
        payload.update({
            "set_id": article.get("set_id"),
            "drug_name": article.get("drug_name", "")[:200],
            "title": article.get("title", "")[:300],
            "active_ingredients": article.get("active_ingredients", [])[:10],
            "manufacturer": article.get("manufacturer", ""),
            "indications": article.get("indications", "")[:1000],
            "contraindications": article.get("contraindications", "")[:500],
            "warnings": article.get("warnings", "")[:500],
            "adverse_reactions": article.get("adverse_reactions", "")[:500],
            "dosage": article.get("dosage", "")[:500],
        })
    else:
        # PMC article specific
        payload.update({
            # Identifiers
            "pmcid": article.get("pmcid"),
            "pmid": article.get("pmid"),
            "doi": article.get("doi"),
            
            # Content (truncated for payload size)
            "title": article.get("title", "")[:300],
            "abstract": article.get("abstract", "")[:1000],
            "full_text": article.get("full_text", "")[:10000],  # Include full text!
            
            # Publication info
            "year": article.get("year"),
            "journal": article.get("journal", ""),
            "publication_type": article.get("publication_type_list", [])[:5],
            
            # Evidence signals for reranking
            "evidence_grade": article.get("evidence_grade"),
            "country": article.get("country"),
            "institutions": article.get("institutions", [])[:5],
            
            # Subject classification
            "keywords": article.get("keywords", [])[:10],
            "mesh_terms": article.get("mesh_terms", [])[:15],
            
            # Authorship
            "authors": article.get("authors", [])[:5],
            "first_author": article.get("first_author"),
            "author_count": article.get("author_count", 0),
            
            # Structure signals
            "has_full_text": article.get("has_full_text", False),
            "has_methods": article.get("has_methods", False),
            "has_results": article.get("has_results", False),
            "table_count": article.get("table_count", 0),
            "figure_count": article.get("figure_count", 0),
        })
    
    return payload


def create_points_batch(articles: List[Dict[str, Any]]) -> tuple:
    """Create PointStruct objects for a batch."""
    points = []
    ids = []
    
    for article in articles:
        # Get unique ID
        doc_id = article.get("pmcid") or article.get("set_id") or article.get("pmid")
        if not doc_id:
            continue
        
        embedding_text = create_embedding_text(article)
        if not embedding_text.strip() or len(embedding_text) < 20:
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
    
    return points, ids


def upsert_batch(client: QdrantClient, points: List[PointStruct], ids: List[str], counters: Counters) -> bool:
    """Upsert batch with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=False
            )
            save_checkpoint(ids)
            counters.increment_success(len(points))
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Batch failed after {MAX_RETRIES} retries: {str(e)[:100]}")
                counters.increment_errors(len(points))
                return False
    return False


def run_ingestion(jsonl_files: List[Path]):
    """Run ingestion for multiple JSONL files."""
    
    logger.info("=" * 70)
    logger.info("🚀 Qdrant Ingestion with Cloud Inference")
    logger.info("=" * 70)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY not set!")
        sys.exit(1)
    
    # Initialize client with Cloud Inference
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=600,
        cloud_inference=True  # Enable Cloud Inference
    )
    
    # Verify connection
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"✅ Connected to Qdrant")
    logger.info(f"   Collection: {COLLECTION_NAME}")
    logger.info(f"   Current points: {info.points_count:,}")
    
    counters = Counters()
    ingested = get_checkpoint()
    logger.info(f"   Already ingested: {len(ingested):,}")
    
    # Count total articles
    total = 0
    for jsonl_file in jsonl_files:
        total += sum(1 for _ in open(jsonl_file))
    logger.info(f"   Total articles to process: {total:,}")
    
    # Process each file
    for jsonl_file in jsonl_files:
        logger.info(f"\n📤 Processing {jsonl_file.name}...")
        
        current_batch = []
        
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            pending = []
            
            with open(jsonl_file) as f:
                for line in tqdm(f, desc=f"Ingesting {jsonl_file.name}"):
                    if not line.strip():
                        continue
                    
                    try:
                        article = json.loads(line)
                        doc_id = article.get("pmcid") or article.get("set_id")
                        
                        if doc_id and str(doc_id) not in ingested:
                            current_batch.append(article)
                        
                        if len(current_batch) >= BATCH_SIZE:
                            points, ids = create_points_batch(current_batch)
                            if points:
                                future = executor.submit(upsert_batch, client, points, ids, counters)
                                pending.append(future)
                            current_batch = []
                            pending = [f for f in pending if not f.done()]
                    
                    except json.JSONDecodeError:
                        continue
                
                # Final batch
                if current_batch:
                    points, ids = create_points_batch(current_batch)
                    if points:
                        future = executor.submit(upsert_batch, client, points, ids, counters)
                        pending.append(future)
            
            # Wait for pending
            for future in as_completed(pending):
                future.result()
    
    # Final stats
    elapsed = time.time() - counters.start_time
    info = client.get_collection(COLLECTION_NAME)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Ingestion Complete!")
    logger.info("=" * 70)
    logger.info(f"   Ingested: {counters.success:,}")
    logger.info(f"   Errors: {counters.errors:,}")
    logger.info(f"   Time: {elapsed/60:.1f} minutes")
    logger.info(f"   Rate: {counters.get_rate():.1f} articles/sec")
    logger.info(f"   Collection total: {info.points_count:,}")


def main():
    # Files to ingest
    jsonl_files = [
        Path("/data/pmc_fulltext/pmc_articles.jsonl"),
        Path("/data/dailymed/dailymed_drugs.jsonl"),
    ]
    
    # Filter to existing files
    existing = [f for f in jsonl_files if f.exists()]
    
    if not existing:
        logger.error("No JSONL files found!")
        sys.exit(1)
    
    run_ingestion(existing)


if __name__ == "__main__":
    main()
```

### 8.2 Run Ingestion

```bash
export QDRANT_URL="https://your-cluster.us-east-1-1.aws.cloud.qdrant.io:6333"
export QDRANT_API_KEY="your-api-key"

# Clear checkpoint for fresh start
rm -f /data/ingest_checkpoint.txt

# Run ingestion
nohup python3 /data/ingest_to_qdrant.py > /data/ingestion_run.log 2>&1 &

# Monitor progress
tail -f /data/ingestion.log
```

**Expected Duration:** ~75 minutes for 1.29M documents

### 8.3 Verify Ingestion

```python
from qdrant_client import QdrantClient

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
info = client.get_collection("pmc_medical_rag_fulltext")

print(f"Total points: {info.points_count:,}")
print(f"Status: {info.status}")
```

---

## 9. Metadata Schema for Reranking

### 9.1 PMC Article Payload Schema

```json
{
  // Identifiers
  "pmcid": "PMC12345678",
  "pmid": "12345678",
  "doi": "10.1000/example",
  
  // Content
  "title": "Article title (max 300 chars)",
  "abstract": "Abstract text (max 1000 chars)",
  "full_text": "Full article text (max 10000 chars)",
  
  // Publication Info
  "year": 2024,
  "journal": "Nature Medicine",
  "article_type": "research-article",
  "publication_type": ["Research Article", "Clinical Trial"],
  
  // Evidence Signals (Critical for Reranking)
  "evidence_grade": "A",  // A=Meta-analysis/Guidelines, B=RCT, C=Cohort, D=Case, E=Opinion
  "country": "USA",
  "institutions": ["Harvard Medical School", "NIH"],
  
  // Subject Classification
  "keywords": ["diabetes", "treatment", "insulin"],
  "mesh_terms": ["Diabetes Mellitus", "Drug Therapy"],
  
  // Authorship
  "authors": ["Smith, John", "Doe, Jane"],
  "first_author": "Smith, John",
  "author_count": 5,
  
  // Structure Signals
  "source": "pmc",
  "has_full_text": true,
  "has_methods": true,
  "has_results": true,
  "table_count": 3,
  "figure_count": 5
}
```

### 9.2 DailyMed Drug Payload Schema

```json
{
  "set_id": "uuid-here",
  "drug_name": "Metformin Hydrochloride",
  "title": "FDA Label Title",
  "active_ingredients": ["METFORMIN HYDROCHLORIDE"],
  "manufacturer": "Generic Pharma Inc",
  
  // Clinical Information
  "indications": "Treatment of type 2 diabetes...",
  "contraindications": "Renal impairment, metabolic acidosis...",
  "warnings": "Lactic acidosis risk...",
  "adverse_reactions": "Nausea, diarrhea...",
  "dosage": "500mg twice daily...",
  
  // Classification
  "source": "dailymed",
  "article_type": "drug_label"
}
```

### 9.3 Reranking Priority

```
Evidence Hierarchy for Reranking:

1. Grade A (Highest Priority)
   - Meta-analyses
   - Systematic reviews
   - Clinical practice guidelines

2. Grade B
   - Randomized controlled trials (RCTs)
   - Clinical trials

3. Grade C
   - Cohort studies
   - Case-control studies
   - Observational studies

4. Grade D
   - Case reports
   - Case series

5. Grade E (Lowest Priority)
   - Editorials
   - Letters
   - Comments
```

---

## 10. Monthly Update Process

### 10.1 Update Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Monthly Update Workflow                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Day 1 of Month:                                            │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────┐ │
│  │ Spin up    │───▶│ Download   │───▶│ Process &          │ │
│  │ EC2        │    │ Incremental│    │ Ingest to Qdrant   │ │
│  └────────────┘    └────────────┘    └────────────────────┘ │
│                                              │               │
│                                              ▼               │
│                                       ┌────────────┐        │
│                                       │ Terminate  │        │
│                                       │ EC2        │        │
│                                       └────────────┘        │
│                                                              │
│  Sources:                                                    │
│  - PMC: s3://pmc-oa-opendata/oa_comm/xml/incr/YYYY-MM-DD/   │
│  - DailyMed: Monthly update ZIP files                        │
│                                                              │
│  Estimated:                                                  │
│  - New articles: 6,000-15,000/month                         │
│  - Processing time: 1-2 hours                                │
│  - EC2 cost: ~$2/month                                       │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Monthly Update Script

Create `/data/monthly_update.py`:

```python
#!/usr/bin/env python3
"""
Monthly incremental update script.

Downloads and ingests new articles from PMC and DailyMed
that were published in the last 30 days.
"""

import os
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PMC_INCR_BASE = "s3://pmc-oa-opendata/oa_comm/xml/incr/"
LOCAL_INCR_DIR = Path("/data/pmc_incremental")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def download_pmc_incremental(days_back: int = 30):
    """Download PMC incremental updates for the past N days."""
    
    LOCAL_INCR_DIR.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now()
    
    for i in range(days_back):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        s3_path = f"{PMC_INCR_BASE}{date_str}/"
        local_path = LOCAL_INCR_DIR / date_str
        
        logger.info(f"Syncing {date_str}...")
        
        cmd = [
            "aws", "s3", "sync",
            s3_path,
            str(local_path),
            "--no-sign-request",
            "--quiet"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.debug(f"No updates for {date_str}")
    
    # Count downloaded files
    xml_files = list(LOCAL_INCR_DIR.rglob("*.xml"))
    logger.info(f"Downloaded {len(xml_files):,} incremental XML files")
    
    return xml_files


def process_and_ingest(xml_files: list):
    """Process new XML files and ingest to Qdrant."""
    
    # Import extraction and ingestion functions
    from extract_pmc import extract_article
    from ingest_to_qdrant import create_points_batch, upsert_batch
    
    from qdrant_client import QdrantClient
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        cloud_inference=True
    )
    
    # Get existing PMCIDs to avoid duplicates
    # (In production, query Qdrant for existing IDs)
    
    articles = []
    for xml_file in xml_files:
        article = extract_article(xml_file)
        if article:
            articles.append(article)
    
    logger.info(f"Extracted {len(articles):,} new articles")
    
    # Ingest in batches
    batch_size = 50
    ingested = 0
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        points, ids = create_points_batch(batch)
        
        if points:
            client.upsert(
                collection_name="pmc_medical_rag_fulltext",
                points=points
            )
            ingested += len(points)
    
    logger.info(f"✅ Ingested {ingested:,} new articles")
    
    return ingested


def main():
    logger.info("=" * 60)
    logger.info("Monthly RAG Update")
    logger.info("=" * 60)
    
    # Download incremental updates
    xml_files = download_pmc_incremental(days_back=30)
    
    if xml_files:
        # Process and ingest
        count = process_and_ingest(xml_files)
        logger.info(f"Update complete: {count:,} new articles added")
    else:
        logger.info("No new articles to process")


if __name__ == "__main__":
    main()
```

### 10.3 Monthly Update Procedure

```bash
# 1. Start EC2 instance (if stopped)
aws ec2 start-instances --instance-ids i-xxxxxxxx

# 2. Wait for instance to be running
aws ec2 wait instance-running --instance-ids i-xxxxxxxx

# 3. SSH and run update
ssh -i /path/to/key.pem ec2-user@<EC2_IP> << 'EOF'
  export QDRANT_URL="https://your-cluster.qdrant.io:6333"
  export QDRANT_API_KEY="your-api-key"
  
  python3 /data/monthly_update.py
EOF

# 4. Stop EC2 instance to save costs
aws ec2 stop-instances --instance-ids i-xxxxxxxx
```

### 10.4 Automation Options

**Option A: AWS EventBridge + Lambda**
- Trigger monthly on 1st of month
- Lambda starts EC2, runs update, stops EC2
- Cost: ~$5/month

**Option B: GitHub Actions (Free)**
```yaml
name: Monthly RAG Update
on:
  schedule:
    - cron: '0 3 1 * *'  # 1st of month at 3 AM

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - name: Start EC2
        run: aws ec2 start-instances --instance-ids ${{ secrets.EC2_ID }}
      
      - name: Run Update
        run: |
          ssh ec2-user@${{ secrets.EC2_IP }} "python3 /data/monthly_update.py"
      
      - name: Stop EC2
        run: aws ec2 stop-instances --instance-ids ${{ secrets.EC2_ID }}
```

---

## 11. Monitoring & Troubleshooting

### 11.1 Health Checks

```python
# Check Qdrant collection status
from qdrant_client import QdrantClient

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
info = client.get_collection("pmc_medical_rag_fulltext")

print(f"Points: {info.points_count:,}")
print(f"Status: {info.status}")  # Should be 'green'
print(f"Segments: {info.segments_count}")
```

### 11.2 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `500 Internal Server Error` | Batch size too large | Reduce batch size to 50 |
| `cloud_inference not found` | Model not enabled | Enable Cloud Inference in Qdrant dashboard |
| `Index required for year` | Missing payload index | Create index with `create_payload_index()` |
| `Timeout errors` | Network/large payload | Increase timeout, reduce text length |
| Slow ingestion | Too few workers | Increase parallel workers (max = shard count × 2) |

### 11.3 Log Locations

```
EC2 Instance:
├── /data/pmc_download.log      # S3 download progress
├── /data/pmc_fulltext/extract_pmc.log  # XML extraction
├── /data/dailymed/download.log # DailyMed download
├── /data/ingestion.log         # Qdrant ingestion
└── /data/ingest_checkpoint.txt # Resume checkpoint
```

---

## 12. Lessons Learned & Optimizations

### 12.1 Critical Optimizations

1. **Batch Size for Cloud Inference**
   - ❌ 5000 per batch → 500 errors (overwhelms inference)
   - ✅ 50 per batch → 0 errors, 238 articles/sec

2. **Text Length Limits for Embedding**
   - Cloud Inference has ~2000 char limit per document
   - Truncate `embedding_text` to 2000 chars max
   - Include: title + abstract + beginning of full text (fits in 2000 chars)

3. **Full Text Download - PMC Open Access Only**
   - ❌ Non-OA articles: Only abstracts available via PubMed E-utilities
   - ✅ OA articles: Full text in `<body>` element of JATS XML from S3
   - S3 bucket `oa_comm` = Commercial-friendly Open Access with full text
   - Full text articles typically 10,000-100,000+ characters

4. **Full Text Storage Strategy**
   - Embedding text: Limited to 2000 chars (title + abstract + intro)
   - Payload storage: Store 10K chars of full_text for RAG context
   - RAG uses payload full_text for answer generation (not embedding text)

5. **DailyMed Nested ZIPs**
   - DailyMed ZIPs contain nested ZIPs, not XML directly
   - Must extract nested ZIPs to get XML files

6. **Payload Indexes Required**
   - Qdrant requires indexes for filtered fields
   - Create indexes for: `year`, `source`, `article_type`, `journal`

### 12.2 Performance Benchmarks

| Operation | Duration | Throughput |
|-----------|----------|------------|
| PMC S3 Download | 4-8 hours | ~25 GB/hour |
| PMC Extraction | 3-4 hours | ~300 files/sec |
| DailyMed Download | 30 mins | - |
| Qdrant Ingestion | 75 mins | 238 articles/sec |

### 12.3 Cost Optimization

- **Don't keep EC2 running 24/7** - only needed for initial load and monthly updates
- **Use `wait=False`** in upserts - don't wait for indexing
- **Checkpoint frequently** - enables resume without re-processing
- **Stream processing** - don't load all data into memory

---

## 13. Cost Estimates

### 13.1 Initial Setup (One-Time)

| Resource | Cost |
|----------|------|
| EC2 r6i.4xlarge (8 hours) | ~$10 |
| EBS 500GB gp3 (1 day) | ~$1 |
| Data transfer | ~$0 (S3 same region) |
| **Total Initial** | **~$11** |

### 13.2 Monthly Ongoing

| Resource | Cost |
|----------|------|
| Qdrant Cloud (8GB, 1.3M vectors) | ~$50-100/month |
| EC2 monthly update (2 hours) | ~$2 |
| EBS storage (if retained) | ~$40/month |
| **Total Monthly** | **~$90-150** |

### 13.3 Cost Optimization Tips

1. **Stop EC2 when not in use** - saves ~$200/month
2. **Use Qdrant Cloud Free tier** for testing
3. **Delete EBS after initial load** - data is in Qdrant
4. **Use spot instances** for batch processing

---

## 14. Appendix: Complete Scripts

All scripts are available in the repository:

```
/data/
├── download_pmc.py          # PMC S3 download
├── extract_pmc.py           # PMC XML extraction
├── download_dailymed.py     # DailyMed download
├── process_dailymed.py      # DailyMed extraction
├── setup_qdrant.py          # Collection setup
├── ingest_to_qdrant.py      # Main ingestion script
└── monthly_update.py        # Monthly incremental update
```

### Quick Start Commands

```bash
# 1. Initial Setup (on EC2)
python3 /data/download_pmc.py           # 4-8 hours
python3 /data/extract_pmc.py            # 3-4 hours
python3 /data/download_dailymed.py      # 30 mins
python3 /data/process_dailymed.py       # 10 mins

# 2. Qdrant Setup
export QDRANT_URL="https://your-cluster.qdrant.io:6333"
export QDRANT_API_KEY="your-key"
python3 /data/setup_qdrant.py

# 3. Ingestion
python3 /data/ingest_to_qdrant.py       # 75 mins

# 4. Monthly Update
python3 /data/monthly_update.py         # 1-2 hours
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2024 | Initial production release |

---

**End of Document**

