# Product Requirements Document (PRD): Medical AI RAG Chatbot

**Version:** 5.0 (Consolidated)  
**Date:** December 2024  
**Purpose:** Production-grade RAG API for medical questions grounded in PMC Open Access literature

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Functional Requirements](#3-functional-requirements)
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [Technology Stack Comparison](#5-technology-stack-comparison)
6. [Data Sources & Licensing](#6-data-sources--licensing)
7. [Data Pipeline](#7-data-pipeline)
8. [Embedding Strategy Options](#8-embedding-strategy-options)
9. [Implementation Scripts](#9-implementation-scripts)
10. [RAG Pipeline Implementation](#10-rag-pipeline-implementation)
11. [Production Deployment](#11-production-deployment)
12. [Cost Analysis](#12-cost-analysis)
13. [Acceptance Criteria & Checklist](#13-acceptance-criteria--checklist)
14. [Timeline & Milestones](#14-timeline--milestones)
15. [Lessons Learned & Best Practices](#15-lessons-learned--best-practices)
16. [Next Steps for Production](#16-next-steps-for-production)

---

## 1. Executive Summary

### 1.1 Overview

Build a **production-ready medical RAG (Retrieval-Augmented Generation) chatbot** that answers medical questions grounded in **1.4M-2.2M peer-reviewed PMC articles** (2020-2025). The system delivers accurate, cited responses from the PubMed Central Open Access corpus.

### 1.2 Key Objectives

| Objective | Target |
|-----------|--------|
| **Target Users** | 100,000 monthly active users (MAU) |
| **Response Latency** | p99 < 2 seconds |
| **Cost per Query** | < $0.01 (embeddings + inference) |
| **Citation Quality** | ≥95% of answers contain valid citations |
| **Data Freshness** | P90 content ≤ 48 hours behind PMC publication |

### 1.3 🚀 Game-Changer: Qdrant Cloud Inference

| Approach | Speed | Time for 1.4M articles | Cost |
|----------|-------|------------------------|------|
| CPU (FastEmbed) | 5 vectors/sec | **~37 hours** | ~$50 EC2 |
| GPU (sentence-transformers) | 50 vectors/sec | ~4 hours | ~$10 EC2 |
| **Qdrant Cloud Inference** | **237 vectors/sec** | **~1.6 hours** | **~$2** ✅ |

> **Key Insight:** Qdrant Cloud Inference handles embedding generation internally - no separate GPU/CPU embedding step needed!

---

## 2. High-Level Architecture

### 2.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEDICAL RAG CHATBOT ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌───────────────────┐
                         │   User Query      │
                         │   "What are the   │
                         │   treatments for  │
                         │   Type 2 Diabetes?"│
                         └─────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     FastAPI Backend         │
                    │  (Local Dev / Railway /     │
                    │   Docker Container)         │
                    └──────────────┬──────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
    ┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
    │ Query Embedding│     │ Vector Search │     │  LLM Answer   │
    │               │     │               │     │  Synthesis    │
    │ Option A:     │     │  Qdrant Cloud │     │               │
    │ ClinicalBERT  │     │  + Binary     │     │  Groq API     │
    │ 768-d         │     │  Quantization │     │  Llama-3.1-8B │
    │               │     │               │     │  or Mixtral   │
    │ Option B:     │     │  Hybrid:      │     │               │
    │ Qdrant Cloud  │     │  Dense+SPLADE │     │  Cost: ~$0.001│
    │ Inference     │     │  Top-20       │     │  per query    │
    │ MiniLM 384-d  │     │  Results      │     │               │
    └───────────────┘     └───────────────┘     └───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │    Cited Response           │
                    │  + PMCID/PMID Links         │
                    │  + Evidence Grade           │
                    │  + Journal Attribution      │
                    └─────────────────────────────┘
```

### 2.2 Query Flow (Detailed)

```
User question 
  → FastAPI/Streamlit (local dev first, then Docker/Railway)
  → Query embedding (Option A: ClinicalBERT 768-d | Option B: Qdrant Cloud Inference 384-d)
  → Qdrant Cloud (binary-quantized vectors)
  → Hybrid search (Dense vectors + optional SPLADE sparse)
  → Top-20 results retrieved
  → Groq LLM (answer synthesis with Llama-3.1-8B or Mixtral-8x7B)
  → Cited answer (PMCID + passage ID + PubMed link)
```

### 2.3 Data Ingestion Pipeline

```
AWS S3 (PMC-OA public bucket: s3://pmc-oa-opendata)
  → EC2 c6i.4xlarge or r6i.4xlarge (Download & Extract: 4-6 hrs)
  → Parse XML → JSONL (2-3 hrs)
  → Generate embeddings:
      Option A: ClinicalBERT 768-d (GPU/CPU: 10-14 hrs)
      Option B: Qdrant Cloud Inference (1.5-2 hrs) ⚡
  → Binary quantization (Qdrant automatic)
  → Qdrant Cloud Upsert (2-3 hrs)
  → LOCAL TESTING (validation queries)
  → READY FOR PRODUCTION ✅

Daily Sync (Cron):
  → AWS S3 Sync (new articles only)
  → Incremental embedding + quantization
  → Qdrant incremental upsert
```

---

## 3. Functional Requirements

| ID     | Requirement                                                                    | Acceptance Criteria                                          |
|--------|--------------------------------------------------------------------------------|--------------------------------------------------------------|
| FR-1   | Ingest CC-BY/CC0/CC-BY-SA/CC-BY-ND articles from PMC-OA (2020-2025)           | 1.4M-2.2M articles ingested, metadata validated             |
| FR-2   | Extract title + abstract + full-text into structured JSON                      | >99% successful extraction, 0 malformed records             |
| FR-3   | Generate embeddings (ClinicalBERT 768-d OR MiniLM 384-d)                       | All articles embedded, vectors normalized                   |
| FR-4   | Apply binary quantization (compression ratio maintained)                        | Qdrant collection created with binary quantization enabled  |
| FR-5   | Store sparse SPLADE vectors for lexical search (optional)                      | Hybrid search capability, RRF fusion α = 0.5               |
| FR-6   | Hybrid search with reranking on top-20 results                                 | Rerank latency p99 < 500ms, rerank top-5 with Groq         |
| FR-7   | Return answer + citations (PMCID + passage index + journal)                    | ≥ 95% of answers contain ≥ 1 citation with PubMed link     |
| FR-8   | Support ≥ 100 concurrent requests                                              | Qdrant QPS > 300, latency p99 < 2s per query               |
| FR-9   | Cost per query ≤ $0.01 (Groq LLM cost-optimized)                              | Monthly inference cost ≤ $120 for 1.2M queries             |
| FR-10  | Synchronize with PMC-OA dataset daily (incremental updates)                    | Vector count increases daily, last sync timestamp ≤ 24h    |
| FR-11  | Enrich metadata: article type, journal, publication year                       | >90% of articles have complete metadata in payload         |
| FR-12  | Boost evidence hierarchy (Systematic reviews > RCTs > Observational)           | Ranking algorithm prioritizes high-evidence-quality        |

---

## 4. Non-Functional Requirements

| ID    | Requirement              | Metric                                              |
|-------|--------------------------|-----------------------------------------------------|
| NFR-1 | Availability             | ≥ 99.5% monthly uptime (Qdrant Cloud SLA)          |
| NFR-2 | Security / HIPAA         | Zero patient data in corpus; all data publicly available |
| NFR-3 | Licensing                | All articles have CC-BY/CC0 commercial-use licenses |
| NFR-4 | Latency (p99)            | < 2 seconds from query to response (US-East)       |
| NFR-5 | Data Freshness           | P90 content ≤ 48 hours behind PMC publication      |
| NFR-6 | Scalability              | Auto-scale to 400+ concurrent users                |
| NFR-7 | Memory Efficiency        | Binary quantization reduces memory footprint       |
| NFR-8 | Embedding Quality        | Cosine similarity > 0.85 for clinically similar docs |
| NFR-9 | Local Dev Validation     | All components tested locally before deployment    |

---

## 5. Technology Stack Comparison

### 5.1 Data Ingestion Stack

| Component         | Technology                              | Purpose                              |
|-------------------|----------------------------------------|--------------------------------------|
| **Data Source**   | AWS S3: `s3://pmc-oa-opendata/`        | Public PMC-OA corpus (2020-2025)    |
| **Compute**       | EC2 c6i.4xlarge or r6i.4xlarge         | High-bandwidth, large RAM           |
| **Embedding**     | See Section 8 (Two Options)            | Vector generation                   |
| **Quantization**  | Qdrant Binary Quantization (1-bit)     | 96% memory reduction                |
| **Vector DB**     | Qdrant Cloud (Managed)                 | Production vector search            |
| **Orchestration** | Python scripts (AWS Lambda or cron)    | Daily incremental sync              |

### 5.2 Query Processing Stack

| Component          | Technology                              | Purpose                              |
|--------------------|----------------------------------------|--------------------------------------|
| **Dev Environment**| FastAPI + Streamlit (local Python)     | Local testing before deployment     |
| **Embedding**      | Same as ingestion for consistency       | Query vectorization                 |
| **Search**         | Qdrant (hybrid dense + sparse)         | Vector retrieval (top-20)           |
| **LLM**            | Groq API (Llama-3.1-8B or Mixtral)     | Cost-efficient answer synthesis     |
| **Deployment**     | Docker + Railway (post-testing)        | Containerized inference             |

### 5.3 Component Comparison

| Component | Option A (ClinicalBERT) | Option B (Cloud Inference) |
|-----------|-------------------------|----------------------------|
| **Embedding Model** | ClinicalBERT 768-d | MiniLM L6 v2 384-d |
| **Embedding Generation** | FastEmbed (CPU/GPU) | Qdrant Cloud Inference |
| **Embedding Speed** | 5-50 vectors/sec | 237 vectors/sec |
| **Medical Specificity** | High (trained on MIMIC + PubMed) | Medium (general purpose) |
| **Memory per 1M vectors** | ~5.7GB (raw), ~180MB (quantized) | ~2.5GB (raw), ~120MB (quantized) |
| **Setup Complexity** | Higher (requires GPU or long CPU time) | Lower (cloud handles everything) |
| **Cost** | ~$10-50 initial | ~$2 initial |

---

## 6. Data Sources & Licensing

### 6.1 Source Details

- **Source:** `s3://pmc-oa-opendata/oa_comm/xml/all/` (AWS open data registry, free access)
- **Format:** XML files organized by year
- **Volume:** ~250 GB compressed (~600-700 GB uncompressed)
- **Update Frequency:** Daily (new articles published to PMC-OA)

### 6.2 License Categories

| License | Commercial Use | Modification | Articles (Est.) |
|---------|---------------|--------------|-----------------|
| CC-BY | ✅ Yes | ✅ Yes | ~1.5M |
| CC0 | ✅ Yes | ✅ Yes | ~200K |
| CC-BY-SA | ✅ Yes | ✅ Yes (share-alike) | ~300K |
| CC-BY-ND | ✅ Yes | ❌ No | ~200K |

### 6.3 Data Model (Qdrant Payload)

```json
{
  "pmcid": "PMC8123456",
  "pmid": "12345678",
  "title": "Novel Treatment for Type 2 Diabetes",
  "abstract": "Background: This study examines...",
  "full_text": "Complete article body text (up to 50K chars)...",
  "metadata": {
    "year": 2023,
    "journal": "The Lancet",
    "article_type": "Randomized Controlled Trial",
    "evidence_grade": "A",
    "keywords": ["diabetes", "treatment", "insulin"],
    "country": "USA",
    "first_author": "Smith, J.",
    "doi": "10.1234/example"
  }
}
```

---

## 7. Data Pipeline

### 7.1 Download PMC Data (4-6 hours)

```bash
#!/bin/bash
set -e

# Configuration
REGION="us-east-1"
LOCAL_PATH="/mnt/data/pmc"
S3_BUCKET="s3://pmc-oa-opendata/oa_comm/xml/all/"

# Create directory
mkdir -p $LOCAL_PATH
cd $LOCAL_PATH

# Download all PMC-OA commercial-use articles
echo "Starting PMC-OA download..."
aws s3 sync \
  $S3_BUCKET \
  $LOCAL_PATH/ \
  --region $REGION \
  --no-sign-request \
  --cli-max-bandwidth 1GB/s 2>&1 | tee $LOCAL_PATH/sync.log

echo "✅ Download complete!"
du -sh $LOCAL_PATH/
```

**Output:** ~250GB, 2.6M XML files

### 7.2 Extract Articles (2-3 hours)

**Key fields to extract:**

```python
{
    "pmcid": "PMC12345678",
    "title": "Article title...",
    "abstract": "Full abstract text...",
    "full_text": "Complete body text (up to 50K chars)...",
    "year": 2024,
    "journal": "Nature Medicine",
    "keywords": ["cancer", "immunotherapy"],
    "authors": ["Smith, J.", "Jones, M."],
    "article_type": "research"
}
```

**Filters Applied:**
- Year: 2020-2025
- Has abstract: ≥50 characters
- Has title: ≥10 characters

**Output:** ~1.37M-2.2M articles in JSONL format

---

## 8. Embedding Strategy Options

### Option A: ClinicalBERT 768-d (High Medical Specificity)

**Best for:** Maximum clinical relevance, when embedding time is not a constraint

| Parameter | Value |
|-----------|-------|
| **Model** | `emilyalsentzer/Bio_ClinicalBERT` |
| **Dimensions** | 768 |
| **Training Data** | MIMIC clinical notes + PubMed abstracts |
| **Speed** | 5-50 vectors/sec (CPU/GPU) |
| **Time for 2.2M articles** | 10-14 hours (GPU) |
| **Memory (quantized)** | ~180MB for 2.2M vectors |

**Initialization:**

```python
from fastembed import TextEmbedding

embedder = TextEmbedding(model_name="emilyalsentzer/Bio_ClinicalBERT")
# Output: 768-dimensional embeddings
```

---

### Option B: Qdrant Cloud Inference (Speed Optimized) ⚡

**Best for:** Fast initial ingestion, iterative development, cost-conscious deployment

| Parameter | Value |
|-----------|-------|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Dimensions** | 384 |
| **Speed** | 237 vectors/sec |
| **Time for 1.4M articles** | ~1.6 hours |
| **Cost** | ~$0.01/1M tokens |
| **Memory (quantized)** | ~120MB for 1.4M vectors |

**Setup in Qdrant Cloud Console:**

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io)
2. Select your cluster
3. Go to **Inference** tab
4. Click **Enable Inference**
5. Wait for cluster restart (~1-2 minutes)

**Initialization:**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Document

client = QdrantClient(
    url="https://xxx.cloud.qdrant.io:6333",
    api_key="your-api-key",
    cloud_inference=True  # CRITICAL!
)

# Create collection with 384 dimensions (MiniLM)
client.create_collection(
    collection_name="pmc_medical_rag",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Upload with automatic embedding!
points = [
    PointStruct(
        id=1,
        vector=Document(
            text="Article title. Article abstract...",
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        payload={"pmcid": "PMC123", "title": "..."}
    )
]
client.upsert(collection_name="pmc_medical_rag", points=points)
```

### Embedding Model Comparison Summary

| Model | Dims | Medical Specificity | Speed | Best Use Case |
|-------|------|---------------------|-------|---------------|
| **ClinicalBERT** | 768 | ⭐⭐⭐ High | ⭐ Slow | Maximum clinical relevance |
| **MiniLM L6 v2** | 384 | ⭐⭐ Medium | ⭐⭐⭐ Fast | Fast iteration, MVP |
| **Mixedbread Large** | 1024 | ⭐⭐ Medium | ⭐⭐ Medium | High accuracy, larger budget |

---

## 9. Implementation Scripts

### 9.1 XML to JSONL Extraction

**File:** `02_extract_pmc.py`

```python
#!/usr/bin/env python3.11
"""
Extract PMC-OA XML files to structured JSONL format.
Preserves: title, abstract, full-text chunks, metadata, keywords.
"""

import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def extract_pmc_article(xml_file):
    """Extract article data from PMC XML"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract identifiers
        pmcid = root.findtext('.//article-id[@pub-id-type="pmc"]')
        pmid = root.findtext('.//article-id[@pub-id-type="pmid"]')
        doi = root.findtext('.//article-id[@pub-id-type="doi"]')
        
        # Extract basic metadata
        title = root.findtext('.//article-title')
        if not title or len(title) < 10:
            return None
        
        # Extract abstract
        abstract_elem = root.find('.//abstract')
        if abstract_elem is not None:
            abstract = ' '.join([
                ''.join(p.itertext()).strip() 
                for p in abstract_elem.findall('.//p')
            ])
        else:
            abstract = ""
        
        if len(abstract) < 50:
            return None  # Skip articles without substantial abstracts
        
        # Extract full text sections
        full_text = ""
        for section in root.findall('.//sec'):
            section_title = section.findtext('.//title', '')
            section_content = ' '.join([
                ''.join(p.itertext()).strip()
                for p in section.findall('.//p')
            ])
            full_text += f"\n{section_title}\n{section_content}"
        
        # Extract keywords
        keywords = [
            kwd.text for kwd in root.findall('.//kwd')
            if kwd.text
        ]
        
        # Extract publication date
        pub_date = root.find('.//pub-date')
        year = pub_date.findtext('.//year') if pub_date is not None else None
        
        # Filter by year (2020-2025)
        if year and (int(year) < 2020 or int(year) > 2025):
            return None
        
        # Extract journal
        journal = root.findtext('.//journal-title', 'Unknown')
        
        # Extract article type
        article_type = root.get('article-type', 'research')
        
        # Extract authors
        authors = []
        for contrib in root.findall('.//contrib[@contrib-type="author"]'):
            name_elem = contrib.find('.//name')
            if name_elem is not None:
                surname = name_elem.findtext('.//surname', '')
                given_names = name_elem.findtext('.//given-names', '')
                if surname:
                    authors.append(f"{surname}, {given_names}".strip())
        
        return {
            'pmcid': pmcid,
            'pmid': pmid,
            'doi': doi,
            'title': title.strip(),
            'abstract': abstract.strip()[:2000],  # Limit for payload size
            'full_text': full_text.strip()[:50000],  # Up to 50K chars
            'keywords': keywords[:10],
            'authors': authors[:5],
            'year': int(year) if year else None,
            'journal': journal[:100],
            'article_type': article_type,
            'source_file': str(xml_file),
            'extracted_at': datetime.utcnow().isoformat()
        }
    except Exception as e:
        return None  # Silently skip malformed files

def process_all_pmc(base_path='/mnt/data/pmc', output_file='/mnt/data/pmc/articles.jsonl'):
    """Process all PMC XML files"""
    
    with open(output_file, 'w') as f_out:
        total_articles = 0
        errors = 0
        
        xml_files = list(Path(base_path).glob('**/*.xml'))
        print(f"\n📖 Processing {len(xml_files)} XML files")
        
        for xml_file in tqdm(xml_files, desc="Extracting"):
            article = extract_pmc_article(xml_file)
            if article:
                f_out.write(json.dumps(article) + '\n')
                total_articles += 1
            else:
                errors += 1
        
        print(f"\n✅ Extraction complete!")
        print(f"   Total articles: {total_articles}")
        print(f"   Skipped/errors: {errors}")
        print(f"   Output: {output_file}")

if __name__ == '__main__':
    process_all_pmc()
```

### 9.2 Qdrant Collection Setup

**File:** `04_setup_qdrant.py`

```python
#!/usr/bin/env python3.11
"""
Create Qdrant collection with binary quantization enabled.
Supports both ClinicalBERT (768-d) and MiniLM (384-d).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    QuantizationConfig,
    BinaryQuantization,
    HnswConfigDiff,
    Distance
)
import os

def setup_qdrant_collection(
    qdrant_url: str = None,
    qdrant_api_key: str = None,
    collection_name: str = "pmc_medical_rag",
    vector_size: int = 384,  # 384 for MiniLM, 768 for ClinicalBERT
    use_cloud_inference: bool = True
):
    """
    Create Qdrant collection with binary quantization.
    
    Args:
        qdrant_url: URL to Qdrant instance
        qdrant_api_key: API key for Qdrant Cloud
        collection_name: Name of collection
        vector_size: 384 (MiniLM) or 768 (ClinicalBERT)
        use_cloud_inference: Enable Qdrant Cloud Inference
    """
    
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        cloud_inference=use_cloud_inference
    )
    
    # Check if collection exists
    try:
        collections = client.get_collections()
        if collection_name in [c.name for c in collections.collections]:
            print(f"⚠️  Collection '{collection_name}' already exists. Deleting...")
            client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print(f"Note: {e}")
    
    # Create collection with binary quantization
    print(f"Creating collection '{collection_name}'...")
    print(f"Vector dimension: {vector_size}")
    print(f"Cloud Inference: {'Enabled' if use_cloud_inference else 'Disabled'}")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
            quantization_config=QuantizationConfig(
                binary=BinaryQuantization(
                    always_ram=False,
                )
            )
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=200,
            ef_search=200,
            full_scan_threshold=10000,
        ),
        on_disk_payload=True,
    )
    
    print(f"\n✅ Collection created with binary quantization!")
    print(f"   - Vector dimension: {vector_size}")
    print(f"   - Quantization: Binary (1-bit)")
    print(f"   - Distance metric: Cosine")

if __name__ == '__main__':
    # Default: MiniLM with Cloud Inference
    setup_qdrant_collection(vector_size=384, use_cloud_inference=True)
    
    # Alternative: ClinicalBERT without Cloud Inference
    # setup_qdrant_collection(vector_size=768, use_cloud_inference=False)
```

### 9.3 Cloud Inference Ingestion Script

**File:** `qdrant_inference_embed.py`

```python
#!/usr/bin/env python3.11
"""
Upload articles to Qdrant with Cloud Inference (automatic embedding).
Speed: ~237 articles/second!
"""

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Document
import json
from tqdm import tqdm
import os

def upload_with_cloud_inference(
    articles_file: str = '/mnt/data/pmc/articles.jsonl',
    collection_name: str = "pmc_medical_rag",
    batch_size: int = 100
):
    """Upload articles with automatic Cloud Inference embedding."""
    
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        cloud_inference=True
    )
    
    points = []
    point_id = 0
    total_upserted = 0
    
    print(f"Reading articles from {articles_file}")
    print(f"Using Qdrant Cloud Inference (MiniLM 384-d)")
    
    with open(articles_file, 'r') as f:
        for line in tqdm(f, desc="Uploading with Cloud Inference"):
            article = json.loads(line)
            
            # Prepare text for embedding (title + abstract)
            text_to_embed = f"{article['title']}. {article['abstract']}"
            
            # Create point with Document (triggers Cloud Inference)
            point = PointStruct(
                id=point_id,
                vector=Document(
                    text=text_to_embed,
                    model="sentence-transformers/all-MiniLM-L6-v2"
                ),
                payload={
                    'pmcid': article.get('pmcid'),
                    'pmid': article.get('pmid'),
                    'title': article['title'][:200],
                    'abstract': article['abstract'][:500],
                    'full_text': article.get('full_text', '')[:5000],
                    'keywords': article.get('keywords', [])[:5],
                    'year': article.get('year'),
                    'journal': article.get('journal', 'Unknown')[:100],
                    'article_type': article.get('article_type', 'research'),
                    'doi': article.get('doi'),
                }
            )
            
            points.append(point)
            point_id += 1
            
            # Upsert in batches
            if len(points) >= batch_size:
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=False
                )
                total_upserted += len(points)
                points = []
    
    # Upsert remaining
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
        total_upserted += len(points)
    
    # Verify
    collection_info = client.get_collection(collection_name)
    print(f"\n✅ Upload complete!")
    print(f"   Total vectors: {collection_info.points_count}")
    print(f"   Embedding: Cloud Inference (MiniLM 384-d)")

if __name__ == '__main__':
    upload_with_cloud_inference()
```

---

## 10. RAG Pipeline Implementation

### 10.1 Complete RAG Pipeline with Groq LLM

**File:** `rag_pipeline.py`

```python
#!/usr/bin/env python3.11
"""
Production-ready Medical RAG Pipeline with Groq LLM integration.
Supports both ClinicalBERT and Cloud Inference embedding modes.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Document
from typing import List, Dict, Optional
import os
from groq import Groq

class MedicalRAGPipeline:
    def __init__(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = "pmc_medical_rag",
        groq_api_key: str = None,
        use_cloud_inference: bool = True,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize Medical RAG Pipeline.
        
        Args:
            qdrant_url: Qdrant Cloud URL
            qdrant_api_key: Qdrant API key
            collection_name: Collection name in Qdrant
            groq_api_key: Groq API key for LLM
            use_cloud_inference: Whether to use Qdrant Cloud Inference
            embedding_model: Model name for Cloud Inference
        """
        
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            cloud_inference=use_cloud_inference
        )
        
        self.collection_name = collection_name
        self.use_cloud_inference = use_cloud_inference
        self.embedding_model = embedding_model
        
        # Initialize Groq client
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.llm_model = "llama-3.1-8b-instant"  # Fast, cost-effective
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Retrieve top-k relevant documents from Qdrant.
        Uses Cloud Inference for query embedding if enabled.
        """
        
        if self.use_cloud_inference:
            # Use Cloud Inference for query embedding
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=Document(
                    text=query,
                    model=self.embedding_model
                ),
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True
            )
        else:
            # For ClinicalBERT: generate embedding locally
            from fastembed import TextEmbedding
            embedder = TextEmbedding(model_name="emilyalsentzer/Bio_ClinicalBERT")
            query_embedding = list(embedder.embed([query]))[0].tolist()
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        
        # Format results
        context = []
        for hit in results:
            payload = hit.payload if hasattr(hit, 'payload') else hit
            context.append({
                'pmcid': payload.get('pmcid', 'N/A'),
                'pmid': payload.get('pmid', 'N/A'),
                'title': payload.get('title', 'Unknown'),
                'abstract': payload.get('abstract', ''),
                'journal': payload.get('journal', 'Unknown'),
                'year': payload.get('year'),
                'score': hit.score if hasattr(hit, 'score') else 0,
            })
        
        return context
    
    def format_citations(self, context: List[Dict]) -> str:
        """Format retrieved documents as citations."""
        citations = []
        for i, doc in enumerate(context, 1):
            pmid = doc['pmid']
            pmcid = doc['pmcid']
            title = doc['title']
            year = doc.get('year', 'N/A')
            
            if pmid and pmid != 'N/A':
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                citations.append(
                    f"[{i}] {title} ({year}) - {doc['journal']}\n"
                    f"    PMID: {pmid} | Link: {pubmed_url}"
                )
            else:
                pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                citations.append(
                    f"[{i}] {title} ({year}) - {doc['journal']}\n"
                    f"    PMCID: {pmcid} | Link: {pmc_url}"
                )
        
        return "\n".join(citations)
    
    def synthesize_answer(
        self,
        query: str,
        context: List[Dict]
    ) -> str:
        """Synthesize answer using Groq LLM."""
        
        # Format context for LLM
        context_text = "\n\n".join([
            f"[{i+1}] {doc['title']} ({doc['year']}) - {doc['journal']}\n"
            f"Abstract: {doc['abstract']}"
            for i, doc in enumerate(context)
        ])
        
        system_prompt = """You are a medical AI assistant specialized in answering clinical questions 
based on peer-reviewed research from PubMed Central. 

RULES:
1. Ground all answers in the provided research context
2. Cite each claim with [1], [2], etc. corresponding to the references
3. Be concise but comprehensive
4. Do NOT make up information not in the context
5. Include caveats about clinical applicability
6. For treatment advice, always recommend consulting healthcare professionals

Format your answer with:
- Brief answer (2-3 sentences)
- Key findings with citations
- Clinical implications
- Limitations of the evidence"""
        
        user_prompt = f"""Question: {query}

Research Context:
{context_text}

Please answer the question based ONLY on the provided research. Use inline citations [1], [2], etc."""
        
        # Call Groq API
        message = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=self.llm_model,
            temperature=0.3,  # Low for medical accuracy
            max_tokens=1024,
        )
        
        return message.choices[0].message.content
    
    def answer_query(self, query: str) -> Dict:
        """Full RAG pipeline: retrieve → synthesize → cite."""
        
        # Step 1: Retrieve context
        print(f"🔍 Searching for: {query}")
        context = self.retrieve_context(query, top_k=5)
        
        if not context:
            return {
                'query': query,
                'answer': "Sorry, I could not find relevant medical literature to answer this question.",
                'citations': [],
                'status': 'no_results'
            }
        
        # Step 2: Format citations
        citations_text = self.format_citations(context)
        
        # Step 3: Synthesize answer with Groq
        print(f"📝 Synthesizing answer with Groq {self.llm_model}...")
        llm_answer = self.synthesize_answer(query, context)
        
        # Step 4: Format final response
        answer = f"""
{llm_answer}

---
**Sources:**
{citations_text}

**Disclaimer:** This answer is based on PubMed Central articles (2020-2025). 
Always consult qualified healthcare professionals for medical decisions.
"""
        
        return {
            'query': query,
            'answer': answer.strip(),
            'context': context,
            'status': 'success'
        }


# Testing
if __name__ == '__main__':
    print("🚀 Medical RAG Pipeline Test")
    print("=" * 80)
    
    rag = MedicalRAGPipeline(use_cloud_inference=True)
    
    test_queries = [
        "What are the latest treatments for Type 2 Diabetes?",
        "How does mRNA vaccine technology work?",
        "What is the role of gut microbiome in obesity?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Q: {query}\n")
        result = rag.answer_query(query)
        print(result['answer'])
        print(f"{'='*80}\n")
```

### 10.2 FastAPI Endpoint

**File:** `api.py`

```python
#!/usr/bin/env python3.11
"""
FastAPI REST API for Medical RAG Chatbot.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from rag_pipeline import MedicalRAGPipeline
import os

app = FastAPI(
    title="Medical AI RAG API",
    description="Answers medical questions grounded in PMC literature",
    version="5.0"
)

# Initialize RAG pipeline
rag = MedicalRAGPipeline(
    use_cloud_inference=True,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class Source(BaseModel):
    pmcid: str
    pmid: Optional[str]
    title: str
    journal: str
    year: Optional[int]
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    status: str

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a medical question with citations."""
    try:
        result = rag.answer_query(request.question)
        
        sources = [
            Source(
                pmcid=ctx.get('pmcid', 'N/A'),
                pmid=ctx.get('pmid'),
                title=ctx.get('title', 'Unknown'),
                journal=ctx.get('journal', 'Unknown'),
                year=ctx.get('year'),
                score=ctx.get('score', 0)
            )
            for ctx in result.get('context', [])
        ]
        
        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            status=result['status']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "5.0"}

# Run with: uvicorn api:app --reload
```

---

## 11. Production Deployment

### 11.1 Environment Variables

```bash
# .env file
QDRANT_URL=https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
GROQ_API_KEY=your-groq-api-key
```

### 11.2 Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```text
# requirements.txt
qdrant-client>=1.7.0
groq>=0.4.0
fastapi>=0.109.0
uvicorn>=0.27.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

### 11.3 Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

---

## 12. Cost Analysis

### 12.1 Initial Setup (One-time)

| Component | Option A (ClinicalBERT) | Option B (Cloud Inference) |
|-----------|------------------------|---------------------------|
| EC2 Download + Extract (6-8 hrs) | $8 | $8 |
| EC2 GPU Embedding (12 hrs) | $10 | — |
| Qdrant Cloud Inference | — | $2 |
| EBS Storage (500GB, 2 days) | $2.50 | $2.50 |
| **Subtotal** | **~$20** | **~$12** |

### 12.2 Monthly Operations

| Component | Cost | Notes |
|-----------|------|-------|
| Qdrant Cloud (8GB+ RAM) | $50-100 | Recommended for production |
| Groq API (100K queries) | $50 | ~$0.0005 per query |
| Data Transfer | $0 | Within AWS free tier |
| **Monthly Total** | **$100-150** | |

### 12.3 Per-Query Cost

| Component | Cost |
|-----------|------|
| Groq inference | ~$0.0005 |
| Vector search | ~$0.001 |
| Cloud Inference embedding | ~$0.00001 |
| **Total per query** | **~$0.0015** ✅ |

---

## 13. Acceptance Criteria & Checklist

### 13.1 Data Ingestion

- [ ] **FR-1**: 1.4M-2.2M articles downloaded from PMC-OA (2020-2025)
- [ ] **FR-2**: All articles extracted to JSONL with title, abstract, metadata
- [ ] **FR-3**: Embeddings generated for all articles (768-d OR 384-d)
- [ ] **FR-4**: Binary quantization enabled in Qdrant
- [ ] **FR-10**: Qdrant collection contains all vectors

### 13.2 Query Performance

- [ ] **FR-6**: Search latency p99 < 500ms
- [ ] **FR-7**: ≥95% of test answers contain valid citations
- [ ] **FR-8**: Support ≥100 concurrent queries
- [ ] **FR-9**: Cost per query ≤ $0.01

### 13.3 Quality Validation

- [ ] **NFR-8**: Cosine similarity > 0.85 for clinically similar documents
- [ ] **NFR-9**: Pipeline tested locally with ≥5 medical queries
- [ ] Groq integration validated (latency <2s, cost <$0.001/query)
- [ ] Citations match retrieved documents accurately

---

## 14. Timeline & Milestones

### Fresh Build Timeline (Optimized with Cloud Inference)

| Step | Duration | Notes |
|------|----------|-------|
| Setup Qdrant Cloud + Enable Inference | 10 min | Enable Inference in console |
| Launch EC2, download PMC data | 4-6 hrs | Any instance with 500GB storage |
| Extract articles to JSONL | 2-3 hrs | Filter by year, validate |
| **Upload with Cloud Inference** | **~1.5 hrs** | 237 articles/sec ⚡ |
| Test and verify | 30 min | |
| Deploy API | 1 hr | Docker/Railway |
| **Total** | **~8-10 hours** | Same day completion! |

### Alternative Timeline (ClinicalBERT)

| Step | Duration | Notes |
|------|----------|-------|
| Setup infrastructure | 30 min | EC2 + Qdrant Cloud |
| Download PMC data | 4-6 hrs | |
| Extract articles | 2-3 hrs | |
| Generate embeddings (GPU) | 10-14 hrs | g4dn.2xlarge |
| Upsert to Qdrant | 2-3 hrs | |
| Test and verify | 1 hr | |
| **Total** | **~21-28 hours** | 1-2 days |

---

## 15. Lessons Learned & Best Practices

### ✅ Do This

1. **Use Qdrant Cloud Inference** - 14x faster than CPU, cheaper than GPU
2. **Enable Inference in Qdrant Cloud Console** before starting
3. **Use MiniLM 384-d model** for fast MVP; upgrade to ClinicalBERT later if needed
4. **Upgrade to paid Qdrant tier** (8GB+) to avoid rate limits and 502 errors
5. **Use r6i.4xlarge (128GB RAM)** for data extraction if handling large datasets
6. **Extract full text** for better reranking and context later
7. **Test locally first** before deploying to production

### ❌ Avoid This

1. **Don't use CPU embedding** (FastEmbed) for large datasets - takes 37+ hours
2. **Don't wait for GPU quota** - Cloud Inference is faster anyway
3. **Don't use free Qdrant tier** for large ingestion (triggers 502 errors)
4. **Don't use 768-d models with Cloud Inference** (only 384/1024 available)
5. **Don't skip local validation** before production deployment

---

## 16. Next Steps for Production

### Immediate (Week 1)
- [x] Complete data ingestion with Cloud Inference
- [ ] Deploy FastAPI backend (Railway/Render/AWS)
- [ ] Test with 100+ medical queries

### Short-term (Weeks 2-4)
- [ ] Add Cohere reranking for improved result quality
- [ ] Implement Redis caching for repeat queries
- [ ] Build frontend UI (Next.js/React)
- [ ] Set up daily sync for new PMC articles

### Long-term (Months 2-3)
- [ ] Evaluate ClinicalBERT for higher medical specificity
- [ ] Implement hybrid search (dense + SPLADE sparse)
- [ ] Add evidence hierarchy boosting
- [ ] Scale to handle 100K MAU

---

## Appendix: Quick Reference Commands

```bash
# 1. Create Qdrant collection (384 dims for MiniLM)
python3 -c "
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
client = QdrantClient(url='...', api_key='...', cloud_inference=True)
client.create_collection('pmc_medical_rag', 
    vectors_config=VectorParams(size=384, distance=Distance.COSINE))
"

# 2. Run Cloud Inference ingestion
nohup python3.11 qdrant_inference_embed.py > inference.log 2>&1 &

# 3. Monitor progress
watch "python3.11 -c \"from qdrant_client import QdrantClient; \
  client = QdrantClient(url='...', api_key='...'); \
  print(client.get_collection('pmc_medical_rag').points_count)\""

# 4. Start API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 5. Test query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are treatments for Type 2 Diabetes?"}'
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v5.0 | Dec 2024 | Consolidated PRD combining v3.1 and v4.0 |
| v4.0 | Dec 2024 | Qdrant Cloud Inference discovery, MiniLM 384-d |
| v3.1 | Dec 2024 | ClinicalBERT 768-d + Local Dev + Groq LLM |
| v3.0 | Dec 2024 | EC2 + Qdrant Cloud + Binary Quantization |
| v2.3 | Dec 2024 | Original Railway-based version |

---

*This document consolidates PRD-Medical-AI-v3.1.md (ClinicalBERT approach) and PRD-Medical-AI-v4.0.md (Qdrant Cloud Inference approach) into a comprehensive production guide.*
