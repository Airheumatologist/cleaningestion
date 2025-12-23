# Medical RAG Pipeline

Production-ready Retrieval-Augmented Generation (RAG) pipeline for medical literature, featuring 1.2M+ PubMed Central articles and 51K+ DailyMed drug labels with advanced preprocessing, reranking, and synthesis.

## ✨ Recent Updates (December 2024)

### 🖥️ Next.js Frontend with Real-Time Streaming
- **Elixir AI Chat Interface** - Modern React-based frontend with glassmorphism design
- **Real-time SSE Streaming** - See pipeline progress (Query Analysis → Retrieval → Reranking → PDF Check → Synthesis)
- **Embedded PDF Viewer** - Split-screen PDF viewing with clickable references
- **AMA-Style Citations** - Formatted with DOI links, journal names, and authors

### 🔬 Unified Deep Research Mode
- Removed "fast search" option for consistent, comprehensive responses
- All queries now use the full ELIXIR System Prompt with multi-step generation
- Improved context building with top 5 full-text articles + 100 abstracts

### 📊 Optimized Reranking & Retrieval
- **Lower Pre-Filter Threshold** (0.10) for better recall
- **Post-Cohere Relevance Filter** (0.30) to eliminate low-quality results
- **Parallel PMC + DailyMed Retrieval** for reduced latency
- **DailyMed Bypass** - Drug labels always included when drug names mentioned

### 💊 DailyMed Ingestion Improvements
- Memory-efficient `lxml` iterative parsing for large XML files
- Per-file timeouts to skip problematic files
- Optimized Qdrant upsert with reduced batch sizes
- Complete 51K+ drug label corpus

### 📝 Citation & Context Enhancements
- Full-text context for top 5 high-value articles (12K chars each)
- All reranked sources displayed (no filtering of cited articles)
- NaN-safe JSON serialization for API stability
- Fixed in-text citation numbering

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Medical RAG Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ 1. QUERY     │───▶│ 2. RETRIEVAL │───▶│ 3. RERANKING │───▶│ 4. SYNTHESIS │  │
│  │ PREPROCESSING│    │              │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │           │
│         ▼                   ▼                   ▼                   ▼           │
│  • Medical Entity    • Qdrant Vector    • Cohere Rerank    • OpenRouter LLM    │
│    Expansion (MeSH)    Search (1024-d)   • Evidence Boost    • ELIXIR Prompt   │
│  • Query Decompose   • Metadata Filter  • Entity Matching   • Full-text Cite  │
│  • Filter Extract    • Hybrid Scoring   • Paper Aggregate   • PDF Check       │
│                                                                                  │
│  Frontend Layer:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Next.js Frontend (Elixir AI) - Real-time SSE streaming, PDF viewer         ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
│  Data Layer:                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Qdrant Cloud (1.29M documents, Binary Quantization, 4 Shards)               ││
│  │ • PMC Open Access: 1.24M full-text articles                                 ││
│  │ • DailyMed: 51K FDA drug labels                                             ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Pipeline Components

| Component | Technology | Description |
|-----------|------------|-------------|
| **Vector Database** | Qdrant Cloud | 1.29M documents, Cloud Inference, Binary Quantization |
| **Embeddings** | mixedbread-ai/mxbai-embed-large-v1 | 1024-dimensional dense vectors |
| **Reranking** | Cohere rerank-v4.0-pro | Neural reranking with YAML formatting |
| **LLM** | OpenRouter (Nvidia Nemotron) | Response synthesis with citations |
| **Entity Expansion** | MeSH (Medical Subject Headings) | Medical acronym expansion |

---

## 🔄 RAG Pipeline Stages

### Stage 1: Query Preprocessing

**Location:** `src/query_preprocessor.py`, `src/medical_entity_expander.py`

```python
# Example: "What are the 2023 ACR/EULAR APS classification criteria?"
# 
# Output:
# - Rewritten: "ACR EULAR Antiphospholipid Syndrome APS classification criteria"
# - Keyword: "APS antiphospholipid syndrome classification criteria ACR EULAR"
# - Filters: {"year": "2023-2025", "field_of_study": "Medicine"}
```

**Features:**
- **Medical Entity Expansion** using MeSH dictionary (10K+ acronyms)
  - `APS` → `Antiphospholipid Syndrome`
  - `COPD` → `Chronic Obstructive Pulmonary Disease`
  - `HFpEF` → `Heart Failure with Preserved Ejection Fraction`
  - MeSH cache included: `data/mesh/mesh_cache.json` (350K+ entries)
  
- **LLM Query Decomposition** extracts:
  - Year range (`"latest"` → 2022-2025)
  - Journal filters (`"from NEJM"` → filter by venue)
  - Rewritten query for semantic search
  - Keyword query for hybrid matching

```python
from src.query_preprocessor import QueryPreprocessor

preprocessor = QueryPreprocessor()
result = preprocessor.decompose_query("latest treatments for heart failure")

# Result:
# rewritten_query: "treatments for heart failure HFpEF HFrEF"
# keyword_query: "heart failure treatment therapy HFpEF HFrEF"
# search_filters: {"year": "2022-2025", "field_of_study": "Medicine"}
```

---

### Stage 2: Retrieval

**Location:** `src/retriever_qdrant.py`

**Features:**
- **Vector Search** with Qdrant Cloud Inference (no local GPU needed)
- **Metadata Filtering** on indexed fields:
  - `year` (integer range)
  - `source` (pmc, dailymed)
  - `article_type` (systematic_review, clinical_trial, etc.)
  - `journal` (keyword match)
  - `country` (keyword match)
  - `evidence_grade` (A, B, C, D)
  
- **Hybrid Scoring** combines:
  - Dense vector similarity (70% weight)
  - Keyword/BM25 matching (30% weight)

```python
from src.retriever_qdrant import QdrantRetriever

retriever = QdrantRetriever(n_retrieval=100)
passages = retriever.retrieve_passages(
    query="COPD management guidelines",
    year="2022-2025",
    article_type="guideline,systematic_review"
)
# Returns: 100 relevant passages with metadata
```

---

### Stage 3: Reranking & Post-Processing

**Location:** `src/reranker.py`

**Reranking Pipeline:**

1. **Cohere Neural Reranking** (rerank-v4.0-pro)
   - Documents formatted as YAML for optimal performance
   - Returns semantic relevance scores (0.0 - 1.0)

2. **Evidence Hierarchy Boosting**
   ```python
   EVIDENCE_HIERARCHY = {
       "systematic_review": 1.25,    # Highest evidence
       "meta_analysis": 1.25,
       "guideline": 1.30,
       "clinical_trial": 1.20,
       "randomized_controlled_trial": 1.20,
       "review_article": 1.10,
       "cohort_study": 1.05,
       "research_article": 1.00,
       "case_report": 0.90,
       "letter": 0.85,
       "editorial": 0.85,
   }
   ```

3. **Entity Matching Boost**
   - Papers containing query medical entities get score boost
   - Prevents irrelevant results (e.g., IgG4-RD when querying APS)

4. **Recency Boost**
   - 2024-2025: 1.05x multiplier
   - 2022-2023: 1.02x multiplier

5. **High-Impact Journal Boost** (1.08x)
   - NEJM, Lancet, JAMA, BMJ, Nature Medicine, Annals of Internal Medicine

6. **Paper Aggregation**
   - Multiple passages from same paper → single paper entry
   - Uses max score across passages

```python
from src.reranker import PaperFinderWithReranker

reranker = PaperFinderWithReranker(context_threshold=0.3)
reranked = reranker.rerank("COPD management", passages)
papers_df = reranker.aggregate_into_dataframe(reranked)
```

---

### Stage 4: LLM Synthesis

**Location:** `src/rag_pipeline.py`, `src/prompts.py`

**Features:**
- **ELIXIR System Prompt** for medical education responses
- **Priority Journal Handling** (NEJM, Lancet, JAMA get full text)
- **Context Building:**
  - Top 5 articles: Full text (12K chars each)
  - Next 100 articles: Abstract only (800 chars each)
- **Structured Output:**
  - Markdown formatting with headers
  - Tables for staging/classification systems
  - Inline citations [1], [2], [3]...
  - Evidence-based recommendations

```python
from src.rag_pipeline import MedicalRAGPipeline

pipeline = MedicalRAGPipeline()
result = pipeline.answer("management of COPD")

# Returns:
# {
#   "answer": "## Overview\n\nCOPD management...[1][2]\n\n## Treatment...",
#   "sources": [{"pmcid": "PMC123", "title": "...", "pdf_url": "..."}],
#   "full_text_articles": [{"source_num": 1, "title": "..."}],
#   "status": "success"
# }
```

---

### Stage 5: PDF Availability Check

**Location:** `src/rag_pipeline.py`

- **Europe PMC API** integration
- Parallel PDF availability check (10 concurrent)
- Returns open-access PDF URLs when available
- Format: `https://europepmc.org/articles/PMC123?pdf=render`

---

## 📁 Project Structure

```
RAG-pipeline/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── env.example                      # Environment template
├── .gitignore
│
├── docs/
│   └── PRD-Pipeline.md             # Detailed PRD (2000+ lines)
│
├── data/                            # Data files
│   └── mesh/
│       └── mesh_cache.json         # MeSH medical abbreviation cache (350K+ entries)
│
├── frontend/                        # Next.js Frontend (Elixir AI)
│   ├── src/
│   │   └── app/
│   │       ├── page.tsx            # Main chat UI with streaming
│   │       ├── globals.css         # Global styles (glassmorphism)
│   │       └── layout.tsx          # App layout
│   ├── package.json
│   └── next.config.ts
│
├── scripts/                         # Data ingestion scripts
│   ├── 01_download_pmc.py          # PMC S3 download
│   ├── 02_extract_pmc.py           # PMC XML extraction
│   ├── 03_download_dailymed.py     # DailyMed download
│   ├── 04_process_dailymed.py      # DailyMed processing
│   ├── 05_setup_qdrant.py          # Qdrant collection setup
│   ├── 06_ingest_pmc.py            # PMC ingestion
│   ├── 07_ingest_dailymed.py       # DailyMed ingestion
│   ├── 08_monthly_update.py        # Monthly updates
│   └── ingestion/
│       └── 11_ingest_dailymed_cloud.py  # Optimized DailyMed ingestion
│
└── src/                             # RAG pipeline source
    ├── config.py                   # Configuration & env vars
    ├── query_preprocessor.py       # Query decomposition
    ├── medical_entity_expander.py  # MeSH acronym expansion
    ├── retriever_qdrant.py         # Qdrant vector search
    ├── reranker.py                 # Cohere reranking
    ├── rag_pipeline.py             # Main pipeline orchestrator
    ├── prompts.py                  # ELIXIR system prompt
    ├── specialty_journals.py       # High-impact journal list
    ├── api_server.py               # FastAPI server with SSE streaming
    └── splade_encoder.py           # SPLADE sparse encoder (optional)
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/Airheumatologist/RAG-pipeline.git
cd RAG-pipeline

# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your API keys
```

### 2. Required API Keys

```bash
# .env file
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 3. Run the Pipeline

```python
from src.rag_pipeline import MedicalRAGPipeline

# Initialize
pipeline = MedicalRAGPipeline()

# Query
result = pipeline.answer("What are the latest treatments for type 2 diabetes?")

# Access results
print(result["answer"])  # Markdown-formatted response
print(result["sources"]) # List of source articles with PDF URLs
```

### 4. Run API Server

```bash
python -m src.api_server
# Server runs at http://localhost:8000

# API Endpoints:
# POST /api/chat/stream - Streaming query with SSE progress updates
# GET /api/health - Health check
```

### 5. Run Frontend (Elixir AI)

```bash
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:3000
```

The frontend connects to the backend at `http://localhost:8000` and provides:
- Real-time streaming of pipeline progress
- Expandable references with clickable DOI links
- Split-screen PDF viewer for source articles
- Modern glassmorphism UI design

---

## 📋 Metadata Schema

### PMC Articles (Qdrant Payload)

```json
{
  "pmcid": "PMC12345678",
  "pmid": "12345678",
  "doi": "10.1000/example",
  "title": "Article Title (300 chars max)",
  "abstract": "Abstract (1000 chars max)",
  "full_text": "Full article text (10K chars max)",
  
  "year": 2024,
  "journal": "Nature Medicine",
  "article_type": "systematic_review",
  "publication_type": ["Systematic Review", "Meta-Analysis"],
  
  "evidence_grade": "A",
  "evidence_level": 1,
  "country": "USA",
  "institutions": ["Harvard Medical School", "NIH"],
  
  "keywords": ["diabetes", "treatment"],
  "mesh_terms": ["Diabetes Mellitus", "Drug Therapy"],
  
  "authors": ["Smith, John", "Doe, Jane"],
  "first_author": "Smith, John",
  "author_count": 5,
  
  "has_full_text": true,
  "has_methods": true,
  "has_results": true,
  "table_count": 3,
  "figure_count": 5,
  
  "source": "pmc"
}
```

### DailyMed Drugs (Qdrant Payload)

```json
{
  "set_id": "uuid",
  "drug_name": "Metformin Hydrochloride",
  "title": "FDA Label Title",
  "active_ingredients": ["METFORMIN HYDROCHLORIDE"],
  "manufacturer": "Generic Pharma Inc",
  
  "indications": "Treatment of type 2 diabetes...",
  "contraindications": "Renal impairment...",
  "warnings": "Lactic acidosis risk...",
  "adverse_reactions": "Nausea, diarrhea...",
  "dosage": "500mg twice daily...",
  
  "source": "dailymed",
  "article_type": "drug_label"
}
```

---

## ⚙️ Configuration Options

### Pipeline Settings (`src/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `TOP_K_RESULTS` | 5 | Documents to retrieve |
| `SCORE_THRESHOLD` | 0.3 | Minimum similarity score |
| `BULK_RETRIEVAL_LIMIT` | 200 | Max candidates before reranking |
| `RERANK_TOP_K` | 20 | Final articles after reranking |
| `FULL_TEXT_COUNT` | 10 | Articles with full text in output |
| `DENSE_WEIGHT` | 0.7 | Dense vector weight in hybrid search |
| `SPARSE_WEIGHT` | 0.3 | Sparse vector weight in hybrid search |

### Evidence Hierarchy for Reranking

| Grade | Article Types | Boost |
|-------|---------------|-------|
| **A** | Meta-analysis, Systematic Review, Guidelines | 1.25-1.30x |
| **B** | RCT, Clinical Trial | 1.20x |
| **C** | Review, Cohort Study | 1.05-1.10x |
| **D** | Case Report, Case Series | 0.90-0.92x |
| **E** | Editorial, Letter, Comment | 0.80-0.85x |

---

## 🔧 Data Ingestion (EC2)

See [docs/PRD-Pipeline.md](docs/PRD-Pipeline.md) for detailed instructions.

### Quick Commands

```bash
# On EC2 (r6i.4xlarge recommended)

# 1. Download PMC articles (4-8 hours)
python scripts/01_download_pmc.py

# 2. Extract to JSONL (3-4 hours)
python scripts/02_extract_pmc.py --xml-dir /data/pmc_fulltext/xml --output /data/pmc_articles.jsonl

# 3. Download DailyMed (30 mins)
python scripts/03_download_dailymed.py

# 4. Process DailyMed (10 mins)
python scripts/04_process_dailymed.py --spl-dir /data/dailymed/xml --output /data/dailymed_drugs.jsonl

# 5. Setup Qdrant collection
python scripts/05_setup_qdrant.py

# 6. Ingest PMC (~60 mins)
python scripts/06_ingest_pmc.py --articles-file /data/pmc_articles.jsonl

# 7. Ingest DailyMed (~5 mins)
python scripts/07_ingest_dailymed.py --xml-dir /data/dailymed/xml

# 8. Monthly updates
python scripts/08_monthly_update.py
```

---

## 🧪 Testing

```python
# Test preprocessing
from src.query_preprocessor import QueryPreprocessor
preprocessor = QueryPreprocessor()
result = preprocessor.decompose_query("COPD management guidelines 2024")
print(f"Rewritten: {result.rewritten_query}")
print(f"Filters: {result.search_filters}")

# Test retrieval
from src.retriever_qdrant import QdrantRetriever
retriever = QdrantRetriever(n_retrieval=10)
passages = retriever.retrieve_passages("COPD treatment")
print(f"Retrieved: {len(passages)} passages")

# Test reranking
from src.reranker import PaperFinderWithReranker
reranker = PaperFinderWithReranker()
reranked = reranker.rerank("COPD treatment", passages)
print(f"Top score: {reranked[0].get('boosted_score', 0):.3f}")

# Test full pipeline
from src.rag_pipeline import MedicalRAGPipeline
pipeline = MedicalRAGPipeline()
result = pipeline.answer("management of COPD")
print(f"Answer length: {len(result['answer'])} chars")
print(f"Sources: {len(result['sources'])} articles")
```

---

## 📈 Performance Benchmarks

| Operation | Duration | Throughput |
|-----------|----------|------------|
| Query preprocessing | ~500ms | - |
| Vector retrieval (100 docs) | ~200ms | - |
| Cohere reranking (100 docs) | ~1-2s | - |
| LLM synthesis | ~5-15s | - |
| PDF check (parallel) | ~2-5s | 10 concurrent |
| **Total E2E latency** | **~10-25s** | - |

### Ingestion Benchmarks

| Operation | Duration | Throughput |
|-----------|----------|------------|
| PMC S3 Download | 4-8 hours | ~25 GB/hour |
| PMC Extraction | 3-4 hours | ~300 files/sec |
| DailyMed Download | 30 mins | - |
| Qdrant Ingestion | 75 mins | 238 articles/sec |

---

## 🔒 Security Notes

- API keys stored in `.env` (gitignored)
- Qdrant API key required for vector operations
- Cohere API key required for reranking
- OpenRouter API key required for LLM synthesis
- No PHI (Protected Health Information) stored

---

## 📚 Documentation

- [PRD-Pipeline.md](docs/PRD-Pipeline.md) - Complete Production Requirements Document
  - Detailed architecture
  - Step-by-step implementation
  - Performance optimizations
  - Cost estimates
  - Lessons learned

---

## 📄 License

MIT License

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

**Maintained by:** Medical AI Team  
**Version:** 1.0.0  
**Last Updated:** December 2024
