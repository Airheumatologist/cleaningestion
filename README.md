# Elixir AI - Medical RAG Pipeline

**Internal Team Documentation** | Private Repository

A production-grade **Medical RAG (Retrieval-Augmented Generation) Pipeline** designed for clinical decision support. Elixir AI provides comprehensive, evidence-based medical answers grounded in peer-reviewed literature from PubMed Central (PMC), DailyMed drug labels, and authoritative medical sources.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ELIXIR AI ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────────────┐   │
│  │   Next.js    │      │   FastAPI    │      │    DeepInfra API         │   │
│  │   Frontend   │◄────►│   Backend    │◄────►│  • Embeddings            │   │
│  │  (Port 3000) │  SSE │  (Port 8000) │      │  • Reranking             │   │
│  └──────────────┘      └──────┬───────┘      │  • LLM Inference         │   │
│                               │              └──────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     RAG PIPELINE (src/rag_pipeline.py)               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │   Query     │→ │  Hybrid     │→ │   Reranker  │→ │    LLM      │ │    │
│  │  │Preprocessing│  │  Retrieval  │  │   (Qwen3)   │  │  Synthesis  │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              QDRANT VECTOR DATABASE (Self-Hosted on Hetzner)         │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │                   COLLECTION: rag_pipeline                   │   │    │
│  │  │  • Dense Vectors:    Qwen3-Embedding-0.6B (1024-d)          │   │    │
│  │  │  • Sparse Vectors:   BM25 (Hybrid Search)                   │   │    │
│  │  │  • Quantization:     Scalar (int8, 75% memory reduction)    │   │    │
│  │  │  • Payload:          Medical metadata + Full text           │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. User query → Next.js frontend
2. FastAPI backend receives query
3. Query preprocessing (LLM decomposition, typo correction)
4. Hybrid retrieval from Qdrant (dense + sparse vectors)
5. Reranking with evidence hierarchy (DeepInfra Qwen3-Reranker)
6. Context building (intelligent section selection)
7. LLM synthesis (DeepInfra Nemotron/DeepSeek)
8. Streaming response to frontend

---

## 🚀 Key Features

### 1. **Multi-Stage Retrieval Pipeline**
- **Query Preprocessing**: LLM-based query decomposition with typo correction, entity extraction, and intent detection
- **Hybrid Search**: Dense (semantic) + Sparse (BM25) vector search for optimal recall
- **Batch Query Optimization**: Reduces HTTP calls from 6+ to 1 for multi-query retrieval
- **Evidence Hierarchy Boosting**: Prioritizes guidelines, systematic reviews, and RCTs over case reports

### 2. **Advanced Reranking**
- **Strict Qwen3 Reranking**: Uses `Qwen/Qwen3-Reranker-0.6B` via DeepInfra
- **Medical Entity Matching**: Post-reranking filtering based on medical condition matching
- **Evidence Tier System**: 3.0x boost for guidelines, 1.5x for trials, 0.2x penalty for case reports

### 3. **Intelligent Context Building**
- **Smart Section Selection**: For DailyMed drug labels, selects relevant sections based on query intent
- **Citation Cleaning**: Strips internal citations to prevent LLM confusion
- **Priority Journal Boost**: +15% for high-impact journals (NEJM, Lancet, JAMA, etc.)

### 4. **Streaming Response**
- Real-time SSE (Server-Sent Events) streaming for progressive answer generation
- Step-by-step progress indicators (Query Analysis → Retrieval → Reranking → PDF Check → Synthesis)

---

## 📁 Project Structure

```
RAG-pipeline/
├── src/                                    # Core pipeline modules
│   ├── api_server.py                       # FastAPI REST API
│   ├── rag_pipeline.py                     # Main RAG orchestration
│   ├── retriever_qdrant.py                 # Qdrant hybrid retriever
│   ├── reranker.py                         # Reranking with evidence hierarchy
│   ├── query_preprocessor.py               # LLM query decomposition
│   ├── config.py                           # Central configuration
│   ├── prompts.py                          # LLM system prompts
│   ├── medical_qdrant_client.py            # Qdrant client wrapper
│   ├── medical_entity_expander.py          # MeSH acronym expansion
│   ├── bm25_sparse.py                      # BM25 sparse encoder
│   ├── splade_encoder.py                   # SPLADE sparse encoder (backup)
│   └── specialty_journals.py               # Journal priority lists
│
├── scripts/                                # Data ingestion pipeline
│   ├── 01_download_pmc.py                  # Download PMC OA articles
│   ├── 02_extract_pmc.py                   # Extract XML content
│   ├── 03_download_dailymed.py             # Download FDA drug labels
│   ├── 04_process_dailymed.py              # Process drug label XML
│   ├── 05_setup_qdrant.py                  # Initialize Qdrant collection
│   ├── 06_ingest_pmc.py                    # Ingest PMC to Qdrant
│   ├── 07_ingest_dailymed.py               # Ingest DailyMed to Qdrant
│   ├── 08_monthly_update.py                # Incremental updates
│   ├── 09_smoke_test.py                    # Validation tests
│   ├── 10_download_gov_abstracts.py        # Download ClinicalTrials.gov/FDA
│   ├── 11_download_author_manuscripts.py   # Download PMC Author Manuscripts
│   ├── 14_ingest_author_manuscripts.py     # Ingest Author Manuscripts
│   ├── 15_ingest_gov_abstracts.py          # Ingest Gov Abstracts
│   ├── 20_download_pubmed_baseline.py      # Download PubMed abstracts
│   ├── 21_ingest_pubmed_abstracts.py       # Ingest PubMed to Qdrant
│   ├── 22_add_fulltext_to_pmc.py           # Add full text to existing points
│   ├── config_ingestion.py                 # Ingestion config
│   ├── ingestion_utils.py                  # Core ingestion utilities
│   └── ingestion_utils_enhanced.py         # Enhanced chunking/validation
│
├── frontend/                               # Next.js React frontend
│   ├── src/app/
│   │   ├── page.tsx                        # Main chat interface
│   │   ├── layout.tsx                      # App layout
│   │   └── globals.css                     # Global styles
│   ├── package.json                        # Frontend dependencies
│   └── next.config.ts                      # Next.js configuration
│
├── deploy/                                 # Deployment configurations
│   └── hetzner_setup.md                    # Self-hosted Qdrant guide
│
├── requirements.txt                        # Python dependencies
├── .env                                    # Environment configuration
└── README.md                               # This file
```

---

## 🔧 Core Components Deep Dive

### 1. Query Preprocessing (`src/query_preprocessor.py`)

Uses LLM to decompose queries into structured components:

```python
class DecomposedQuery(BaseModel):
    earliest_search_year: str      # Publication year filter
    latest_search_year: str
    venues: str                     # Journal filters
    rewritten_query: str           # Optimized for semantic search
    rewritten_query_for_keyword_search: str  # For keyword matching
    drug_names: List[str]          # Extracted medications
    medical_conditions: List[str]  # Extracted conditions
    corrected_query: str           # Typo-corrected query
    corrected_medical_conditions: List[str]  # Corrected conditions
```

**Features:**
- Medical acronym expansion (MeSH-based)
- Drug name extraction with brand/generic mapping (e.g., "golimumab" → ["golimumab", "SIMPONI", "SIMPONI ARIA"])
- Typo detection and correction (e.g., "NUEROBROCELLOSIS" → "neurobrucellosis")
- Query variation generation for multi-query retrieval (treatment, diagnosis, guidelines angles)

### 2. Hybrid Retriever (`src/retriever_qdrant.py`)

Implements batched hybrid search with RRF (Reciprocal Rank Fusion):

```python
# Dense + Sparse vectors combined
dense_weight = 0.7    # Semantic similarity (Qwen3 embeddings)
sparse_weight = 0.3   # BM25 lexical matching

# Batch query reduces HTTP calls from 6+ to 1
batch_results = client.query_batch_points(
    collection_name=collection_name,
    requests=batch_requests  # All queries in one call
)
```

**Key Methods:**
- `batch_hybrid_search()` - Main retrieval method, batches all query variations
- `search_dailymed_by_drug()` - Parallel drug label lookup by drug names
- `_build_filter()` - Metadata filtering (year, venue, article_type)

### 3. Reranker with Evidence Hierarchy (`src/reranker.py`)

Multi-factor scoring system:

```python
# Evidence tier multipliers (defined in reranker.py)
TIER_1_BOOST = 3.00   # Guidelines, systematic reviews, meta-analyses
TIER_2_BOOST = 1.50   # RCTs, clinical trials, review articles  
TIER_3_BOOST = 1.00   # Standard research
TIER_4_PENALTY = 0.20 # Case reports, letters, editorials (suppressed)

# Additional multipliers
RECENCY_MULT = 1.10   # Recent papers (last 2 years, non-case-reports only)
JOURNAL_MULT = 1.15   # High-impact journals (NEJM, Lancet, etc.)
```

**Pipeline:**
1. DeepInfra reranking (`Qwen/Qwen3-Reranker-0.6B`) → raw relevance scores
2. Entity matching score (30% weight) → medical condition overlap
3. Combined score = 0.7 * rerank_score + 0.3 * entity_score
4. Evidence tier multiplier application
5. DOI/title-based deduplication
6. Paper-level aggregation (max score per paper)

### 4. Context Builder (`src/rag_pipeline.py`)

Intelligent context assembly in `_get_papers_for_context()`:

**For PMC Articles:**
- Abstract only (1200 char limit)
- Priority journal ordering (NEJM, Lancet, JAMA first)
- Article type badges

**For DailyMed Drug Labels:**
```python
# Always include
- Highlights of Prescribing Information (8000 chars)
- Clinical Studies section (15000 chars)

# Conditional (if query mentions mechanisms, PK/PD)
- Clinical Pharmacology section (10000 chars)
```

**Features:**
- Citation cleaning (`_clean_source_text()`) - removes `[1]`, `[2-5]` patterns
- Smart recency (preserves seminal older guidelines)

### 5. LLM Synthesis (`src/rag_pipeline.py`)

**System Prompt**: `ELIXIR_SYSTEM_PROMPT` (in `src/prompts.py`)
- Clinical decision support persona for physicians
- Extracts specific details: staging systems, medication protocols, trial results
- Markdown tables for comparisons
- Inline citations `[1]`, `[2]` strictly enforced
- No general disclaimers (peer-to-peer professional communication)

**Model:**
- Uses `openai/gpt-oss-20b` via DeepInfra
- Fallback logic disabled (Strict configuration)

---

## 🛠️ Setup & Development

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- Docker (for local Qdrant testing)
- DeepInfra API key (ask team lead)

### 1. Clone & Install

```bash
cd RAG-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp env.example .env
```

Edit `.env` with team credentials:

```env
# =============================================================================
# QDRANT (Self-Hosted on Private Server - Hetzner)
# =============================================================================
QDRANT_URL=http://65.109.112.253:6333
QDRANT_API_KEY=<ask-team-lead>
QDRANT_COLLECTION=rag_pipeline

# =============================================================================
# DEEPINFRA API (Team Account)
# =============================================================================
DEEPINFRA_API_KEY=<ask-team-lead>
DEEPINFRA_MODEL=openai/gpt-oss-20b
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai

# Embedding Model
EMBEDDING_PROVIDER=deepinfra
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B-batch

# Reranker Model  
RERANKER_PROVIDER=deepinfra
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B

# =============================================================================
# CHUNKING (Optimized for Qwen3 32K context)
# =============================================================================
CHUNK_SIZE_TOKENS=2048
CHUNK_OVERLAP_TOKENS=256

# =============================================================================
# SEARCH CONFIGURATION
# =============================================================================
USE_HYBRID_SEARCH=true
SPARSE_RETRIEVAL_MODE=bm25
RETRIEVAL_CHUNK_LIMIT=400
RERANK_TOP_CHUNKS=100
FINAL_TOP_ARTICLES=50
```

### 3. Local Development (without full dataset)

```bash
# Start local Qdrant for testing
docker run -d \
  --name qdrant-local \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.12.0

# Update .env for local testing
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=""  # No auth for local

# Run backend
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload

# Run frontend (new terminal)
cd frontend
npm install
npm run dev
```

Access at `http://localhost:3000`

### 4. Production Data Ingestion

Use the provided bash scripts for a streamlined workflow:

```bash
# 1. Complete Ingestion (Downloads + Ingests all datasets)
./scripts/run_complete_ingestion.sh

# OR 2. Background Ingestion (Run in background with logs)
./scripts/run_ingestion_background.sh
```

**Manual Breakdown (if running step-by-step):**

```bash
# Setup collection
python scripts/05_setup_qdrant.py --recreate

# 1. PMC Articles
python scripts/01_download_pmc.py --years 2020-2025
python scripts/02_extract_pmc.py
python scripts/06_ingest_pmc.py
python scripts/22_add_fulltext_to_pmc.py

# 2. DailyMed Drug Labels
python scripts/03_download_dailymed.py
python scripts/04_process_dailymed.py
python scripts/07_ingest_dailymed.py

# 3. PubMed Abstracts
python scripts/20_download_pubmed_baseline.py
python scripts/21_ingest_pubmed_abstracts.py

# 4. Author Manuscripts & Government Abstracts
python scripts/11_download_author_manuscripts.py
python scripts/14_ingest_author_manuscripts.py
python scripts/10_download_gov_abstracts.py
python scripts/15_ingest_gov_abstracts.py
```

---

## 🔌 API Endpoints

### Chat Endpoint (Streaming) - PRIMARY

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest treatments for rheumatoid arthritis?", "stream": true}'
```

**Response:** Server-Sent Events (SSE)

```
data: {"step": "query_expansion", "status": "running"}
data: {"step": "retrieval", "status": "running", "retrieved_count": 400}
data: {"step": "reranking", "status": "complete", "sources": [...]}
data: {"step": "generation", "status": "running", "token": "The latest treatments..."}
data: {"step": "complete", "status": "success", "answer": "...", "sources": [...]}
data: [DONE]
```

### Non-Streaming Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Management of IgG4-related disease"}'
```

### Query Decomposition (Debug)

```bash
curl -X POST http://localhost:8000/api/query/decompose \
  -H "Content-Type: application/json" \
  -d '{"query": "Systematic reviews on SGLT2 inhibitors from 2020"}'
```

---

## 📊 Configuration Reference

### Embedding Models (via DeepInfra)

| Model | Dimension | Context | Use Case |
|-------|-----------|---------|----------|
| Qwen/Qwen3-Embedding-0.6B-batch | 1024 | 32K | **Strict Default** |

### Reranker Models (via DeepInfra)

| Model | Max Tokens | Notes |
|-------|------------|-------|
| Qwen/Qwen3-Reranker-0.6B | 32K | **Strict Default** |

### LLM Models (via DeepInfra)

| Model | Use Case |
|-------|----------|
| openai/gpt-oss-20b | **Primary generation** (No fallback) |

---

## 🧪 Testing & Debugging

```bash
# Validate configuration
python src/config.py

# Test Qdrant connection
python src/medical_qdrant_client.py

# Test reranker
python src/reranker.py

# Test query preprocessing
python src/query_preprocessor.py

# Run smoke tests
python scripts/09_smoke_test.py
```

---

## 🚢 Deployment

### Production Server (Hetzner)

Qdrant is self-hosted on Hetzner AX52 (64GB RAM, AMD Ryzen 7000).

See `deploy/hetzner_setup.md` for:
- Server provisioning
- Docker setup
- Qdrant configuration
- Firewall rules

### Backend Deployment

```bash
# Production with multiple workers
uvicorn src.api_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --proxy-headers
```

### Frontend Deployment

```bash
cd frontend
npm run build
npm start  # Production mode
```

---

## ⚡ Performance Optimization

### 1. Chunking Strategy
- **Size**: 2048 tokens (optimal for Qwen3 with 32K context)
- **Overlap**: 256 tokens (12.5% for continuity)
- **Filtering**: Conservative profile for clinical backmatter

### 2. Vector Quantization
- **Type**: Scalar (int8)
- **Memory Reduction**: 75% (4x)
- **Accuracy Loss**: <1%
- **Rescore**: Enabled for query-time accuracy recovery

### 3. Batch Query Optimization
- Reduces HTTP calls from 6+ to 1 per user query
- Parallel DailyMed search in background thread
- Pre-computed sparse vectors for all query variations

### 4. Retrieval Limits
- **Initial Retrieval**: 400 chunks
- **Rerank Input**: 200 chunks (2 per article max)
- **Rerank Output**: 100 chunks
- **Final Articles**: 50 papers

---

## 🔒 Security Notes

- **Qdrant**: Protected by API key, firewall restricted to team IPs
- **DeepInfra**: API key stored in environment variables only
- **CORS**: Configured for specific origins in `src/api_server.py`
- **No PII**: Pipeline processes only public medical literature

---

## 🐛 Common Issues

### Qdrant Connection Timeout
```
# Increase timeout in src/config.py
QDRANT_TIMEOUT = 180  # seconds
```

### DeepInfra Rate Limiting
- Batch size is set to 64 for embeddings
- Retries enabled with exponential backoff

### Out of Memory During Ingestion
- Reduce `BATCH_SIZE` in `scripts/config_ingestion.py`
- Use `MAX_WORKERS=4` instead of 8

---

## 📞 Team Contacts

- **Infrastructure/Deployment**: See team wiki
- **Data Ingestion Issues**: Check `CHANGES_SUMMARY.md` for recent fixes
- **API Issues**: Review `src/api_server.py` logs

---

## 📝 Recent Changes

See `CHANGES_SUMMARY.md` for:
- Token range fixes in QualityValidator
- Author manuscripts logic bug fixes
- SemanticChunker tokenizer integration

---

**Internal Use Only** | Do not distribute outside the organization
