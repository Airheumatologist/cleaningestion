# Elixir AI - Medical RAG Pipeline

⚠️ **INTERNAL USE ONLY — PRIVATE REPOSITORY** ⚠️

This repository is **private and restricted to the internal team only**. Do not share, distribute, or expose to external parties. If you need access, contact the team lead.

---

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
│  │  │                   COLLECTION: rag_pipeline                    │   │    │
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
7. LLM synthesis (DeepInfra `openai/gpt-oss-20b`)
8. Streaming response to frontend

---

## 🧭 Production Stack Integration (Developer Guide)

This section is the source of truth for integrating product frontend services into the production stack.

### 1) Production network model

```
┌──────────────────────────┐
│ Browser (User)           │
└────────────┬─────────────┘
             │ HTTPS
             ▼
┌──────────────────────────┐
│ Azure App Service        │
│ Next.js (server runtime) │
│ - Route handlers/proxy   │
└────────────┬─────────────┘
             │ Private traffic only
             │ (WireGuard tunnel)
             ▼
┌──────────────────────────┐
│ Hetzner rag-api:8000     │
│ FastAPI + Gunicorn       │
└────────────┬─────────────┘
             │ Docker internal network
             ▼
┌──────────────────────────┐
│ Qdrant (qdrant:6333)     │
└──────────────────────────┘
```

Rules:
1. Do not call Hetzner API directly from browser code.
2. Do not expose bearer service token to client-side JavaScript.
3. Keep Qdrant non-public; only `rag-api` talks to Qdrant over Docker network.

### 2) API contract used by frontend services

```
Primary:
  POST /api/v1/chat
  POST /api/v1/chat/stream
  GET  /api/v1/health
  POST /api/v1/debug/decompose  (debug, protected)

Temporary legacy aliases:
  /api/chat         -> /api/v1/chat
  /api/chat/stream  -> /api/v1/chat/stream
  /health           -> /api/v1/health
```

Auth:
```
Authorization: Bearer <service_token>
```

Non-health auth behavior:
1. `401` missing/invalid token
2. `403` token disabled

### 3) Request path expected from frontend

```
Browser fetch("/api/chat/stream")
        │
        ▼
Next.js route handler (server)
  injects Authorization header
  forwards to Hetzner /api/v1/chat/stream
        │
        ▼
SSE stream returns through same path back to browser
```

### 4) Required environment variables (frontend runtime)

Set these in Azure App Service:

```env
RAG_API_BASE_URL=http://<private-hetzner-ip-or-dns>:8000
RAG_API_TOKEN=<service_token>
```

For local integration via SSH tunnel:

```env
RAG_API_BASE_URL=http://localhost:8001
RAG_API_TOKEN=<same-test-token>
```

### 5) Minimal Next.js server proxy examples

Use this server-proxy pattern for production. A pure rewrite-only approach cannot safely attach service auth.

`app/api/chat/route.ts` (sync):

```ts
export async function POST(req: Request) {
  const body = await req.text();
  const upstream = await fetch(`${process.env.RAG_API_BASE_URL}/api/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.RAG_API_TOKEN}`,
    },
    body,
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}
```

`app/api/chat/stream/route.ts` (SSE):

```ts
export async function POST(req: Request) {
  const body = await req.text();
  const upstream = await fetch(`${process.env.RAG_API_BASE_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${process.env.RAG_API_TOKEN}`,
    },
    body,
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
```

Your browser/client components should call only relative paths:
1. `/api/chat`
2. `/api/chat/stream`

### 6) Local test topology (Mac + tunnel)

```
Mac Browser (localhost:3000/3001)
        │
        ▼
Local Next.js server
        │ calls RAG_API_BASE_URL=http://localhost:8001
        ▼
SSH tunnel localhost:8001 -> Hetzner localhost:8000
        │
        ▼
Hetzner rag-api
```

Reference docs:
1. `deploy/hetzner/README.md`
2. `deploy/integration/README.md`

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
│   ├── service_auth.py                     # Bearer token validation + hash utils
│   └── specialty_journals.py               # Journal priority lists
│
├── scripts/                                # Data ingestion pipeline
│   ├── 01_download_pmc_unified.py          # Download PMC OA + Author Manuscripts from PMC Cloud Service (AWS S3)
│   ├── 03_download_dailymed.py             # Download FDA drug labels
│   ├── 05_setup_qdrant.py                  # Initialize Qdrant collection
│   ├── 06_ingest_pmc.py                    # Ingest unified PMC XML sources to Qdrant
│   ├── 07_ingest_dailymed.py               # Ingest DailyMed XML to Qdrant
│   ├── 08_weekly_update.py                 # Weekly incremental updates (PubMed, DailyMed, PMC)
│   ├── 20_download_pubmed_baseline.py      # Download PubMed abstracts (includes gov affiliation)
│   ├── 21_ingest_pubmed_abstracts.py       # Ingest PubMed to Qdrant
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
│   ├── hetzner/README.md                   # Co-located Hetzner stack (Qdrant + rag-api)
│   └── integration/README.md               # Service integration contract + examples
│
├── start_ingestion.sh                      # Interactive ingestion starter
├── start_remaining_ingestion.sh            # Resume partial ingestion
├── deploy_production.sh                    # Production deployment
├── requirements.txt                        # Python dependencies
├── .env                                    # Environment configuration
└── README.md                               # This file
```

---

## 🔧 Core Components Deep Dive

### 1. Query Preprocessing (`src/query_preprocessor.py`)

Uses LLM to decompose queries into structured components:

```text
                  ┌────────────────────┐
                  │    Raw Query:      │
                  │  "NUEROBROCELLOSIS │
                  │     treatmnts"     │
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │ LLM Decomposition  │
                  └─────────┬──────────┘
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│Corrected Query│   │   Entities    │   │  Query Vars   │
│Neurobrucellosis   │ - Medications │   │ - Diagnosis   │
│ treatments    │   │ - Conditions  │   │ - Treatment   │
└───────────────┘   └───────────────┘   └───────────────┘
```

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

```text
┌──────────────┐     ┌───────────────────────┐
│              │     │      Qdrant DB        │
│    Query     ├────►│ ┌────────┐ ┌────────┐ │
│  Variations  │     │ │ Dense  │ │ Sparse │ │
│              │     │ │(Cosine)│ │ (BM25) │ │
└──────────────┘     │ └────────┘ └────────┘ │
                     │       Fusion (RRF)    │
                     └───────────┬───────────┘
                                 ▼
                         400 Top Chunks
```

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

```text
┌───────────────┐    ┌────────────────────────┐
│  400 Chunks   ├───►│     Qwen3 Reranker     │
│  from Qdrant  │    │    (DeepInfra API)     │
└───────────────┘    └───────────┬────────────┘
                                 │ (Primary Score)
                   ┌─────────────▼────────────┐
                   │   Entity Match Filter    │
                   │    (30% Score Weight)    │
                   └─────────────┬────────────┘
                                 │
                   ┌─────────────▼────────────┐
                   │ Evidence Tier Multiplier │
                   │  Guidelines: x3.0 Boost  │
                   │  RCTs:       x1.5 Boost  │
                   │  Case Rep:  x0.2 Penalty │
                   └─────────────┬────────────┘
                                 ▼
                       Final Top 100 Chunks 
                        Passed to Context
```

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
- **LLM**: `openai/gpt-oss-20b` via DeepInfra
- **Reranker**: `Qwen/Qwen3-Reranker-0.6B` via DeepInfra

---

## 🛠️ Setup & Development

If you are integrating an existing product frontend into production, start with the
`Production Stack Integration (Developer Guide)` section above and use this section only for local pipeline development.

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
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=<ask-team-lead>
QDRANT_COLLECTION=rag_pipeline
COLLECTION_NAME=rag_pipeline

# =============================================================================
# DEEPINFRA API (Team Account)
# =============================================================================
DEEPINFRA_API_KEY=<ask-team-lead>
LLM_MODEL=openai/gpt-oss-20b
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai
DEEPINFRA_CHAT_TIMEOUT_SECONDS=300
DEEPINFRA_EMBED_TIMEOUT_SECONDS=120
DEEPINFRA_RERANK_TIMEOUT_SECONDS=60

# =============================================================================
# API SERVICE AUTH
# =============================================================================
API_AUTH_ENABLED=true
API_KEYS_FILE=/opt/RAG-pipeline/api_keys.json
API_KEYS_CACHE_TTL_SECONDS=30

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
MAX_CHUNKS_PER_ARTICLE_PRE_RERANK=3
RERANK_INPUT_CHUNK_LIMIT=200
RERANK_TOP_CHUNKS=100
FINAL_TOP_ARTICLES=50

# =============================================================================
# DATA DIRECTORIES
# =============================================================================
DATA_DIR=/data/ingestion
PMC_XML_DIR=/data/ingestion/pmc_xml
DAILYMED_XML_DIR=/data/ingestion/dailymed/xml
DAILYMED_STATE_DIR=/data/ingestion/dailymed/state
DAILYMED_CHECKPOINT_FILE=/data/ingestion/dailymed_ingested_ids.txt
DAILYMED_SET_ID_MANIFEST=/data/ingestion/dailymed/state/dailymed_last_update_set_ids.txt
PUBMED_BASELINE_DIR=/data/ingestion/pubmed_baseline
PUBMED_ABSTRACTS_FILE=/data/ingestion/pubmed_baseline/filtered/pubmed_abstracts.jsonl
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
API_AUTH_ENABLED=false

# Run backend
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload

# Run frontend (new terminal)
cd frontend
npm install
export API_REWRITE_TARGET=http://localhost:8000
npm run dev
```

Access at `http://localhost:3000`

### 4. Production Data Ingestion

Use the interactive script:

```bash
# Start ingestion interactively (resumes from checkpoints)
./start_ingestion.sh

# Fresh ingestion from scratch (clears checkpoints and recreates collection)
./start_ingestion.sh --fresh

# Show help
./start_ingestion.sh --help
```

The script will:
1. Auto-detect available data sources (PMC XML, PubMed JSONL, DailyMed XML)
2. Let you choose which dataset(s) to ingest
3. Run in sequence (recommended) or parallel
4. Handle checkpointing for resumable ingestion

**Or run manually (complete re-ingestion workflow):**

```bash
# Setup collection (fresh start)
python scripts/05_setup_qdrant.py --recreate

# 1. Unified PMC pipeline on PMC Cloud Service (AWS S3)
# all: full metadata inventory scan
# incremental: only metadata rows newer than last successful incremental run
python scripts/01_download_pmc_unified.py --datasets pmc_oa,author_manuscript --release-mode all

# Incremental refresh (recommended for regular update runs)
python scripts/01_download_pmc_unified.py --datasets pmc_oa,author_manuscript --release-mode incremental
python scripts/06_ingest_pmc.py --xml-dir /data/ingestion/pmc_xml

# 2. DailyMed Drug Labels
python scripts/03_download_dailymed.py
python scripts/07_ingest_dailymed.py --xml-dir /data/ingestion/dailymed/xml

# IMPORTANT: Generate drug lookup cache AFTER DailyMed ingestion completes
# This enables fast O(1) drug name → set_id lookups instead of slow BM25 fallback
python scripts/generate_drug_lookup.py

# 3. PubMed Abstracts (Unified Pipeline - includes Gov Affiliation)
# This unified pipeline downloads PubMed baseline and extracts BOTH:
# - High-value article types (reviews, trials, guidelines)
# - Government affiliations (NIH, CDC, FDA, VA, etc.)
python scripts/20_download_pubmed_baseline.py
python scripts/21_ingest_pubmed_abstracts.py

# NOTE: Government abstracts pipeline has been MERGED into PubMed pipeline above.
# Old scripts (10_download_gov_abstracts.py, 13_ingest_gov_abstracts.py) are deprecated.
# Use filter: is_gov_affiliated=true in queries to get government-authored articles.
```

**Monitor Progress:**
```bash
# Check logs
tail -f /data/ingestion/logs/*.log

# Check PIDs
cat /data/ingestion/logs/*.pid

# View checkpoints
ls -la /data/ingestion/*checkpoint* /data/ingestion/*_ingested_ids.txt
```

**Post-Ingestion Steps (Required for complete re-ingestion):**

After DailyMed ingestion completes, generate the drug lookup cache:

```bash
# Using the interactive script
./start_ingestion.sh
# Select option 6: Post-ingestion: Generate drug lookup

# Or run manually
python scripts/generate_drug_lookup.py
```

This creates `src/data/drug_setid_lookup.json` which enables:
- **Fast O(1) lookups**: Direct JSON lookup instead of BM25 vector search
- **Better performance**: Drug name queries resolve instantly
- **Brand/generic mapping**: Active ingredients are mapped to their drug labels

---

## 📊 Data Ingestion Details

### Storage Requirements (Estimated)

| Dataset | Compressed | Extracted XML | Qdrant Index |
| :--- | :--- | :--- | :--- |
| **PMC (oa_comm)** | ~98 GB | ~392 GB | ~200 GB |
| **DailyMed** | ~8 GB | ~35 GB | ~15 GB |
| **PubMed (Unified)** | ~10 GB | ~40 GB | ~20 GB |
| **Total** | **~120 GB** | **~470 GB** | **~250 GB** |

**Note:** The PubMed Unified pipeline includes government-affiliated articles (NIH, CDC, FDA, VA, etc.) 
that were previously in a separate pipeline. Use `is_gov_affiliated=true` filter to retrieve 
government-authored articles.

**Total Disk Required:** ~850 GB (Safe margin: 1 TB)

### Chunking Strategy

```text
┌────────────────────────────────────────────────────────┐
│                  Original Document                     │
├──────────────────┬──────────────────┬──────────────────┤
│   Chunk 1 (2k)   │   Chunk 2 (2k)   │   Chunk 3 (2k)   │
└────────┬─────────┴────────┬─────────┴────────┬─────────┘
         │      Overlap     │      Overlap     │
         ◄────256 tokens────►────256 tokens────►
```

**PMC Articles:**
- Abstract chunk (title + abstract)
- Section chunks with context: "Title: X\n\nSection: Y\n\nContent"
- Table chunks with full content (caption + row-by-row data)
- Token-aware chunking: 2048 tokens with 256 token overlap (configurable via CHUNK_SIZE_TOKENS)

**PubMed Abstracts (Unified Pipeline):**
- Single source: `pubmed_abstract` (replaces separate `pubmed_gov`)
- Content type: `abstract`
- New fields: `is_gov_affiliated` (bool), `gov_agencies` (list)
- Same chunking strategy as PMC articles
- Filter examples:
  - `is_gov_affiliated=true` - Only government-authored articles
  - `gov_agencies=["NIH","CDC"]` - Specific agencies

**DailyMed Drug Labels:**
- Overview chunk (name + ingredients + key sections)
- Section chunks: Indications, Dosage, Adverse Reactions, Drug Interactions, Warnings
- Table chunks parsed and embedded separately

### Post-Ingestion: Scalar Quantization

After all ingestion completes, enable scalar quantization for 75% memory reduction:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

client = QdrantClient(url="http://your-server:6333", api_key="...")

client.update_collection(
    collection_name="rag_pipeline",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    ),
)
```

---

## 🔌 API Endpoints

These endpoints are for trusted service-to-service calls.
Frontend browser code should call your app's internal Next.js API routes only.

### `POST /api/v1/chat/stream` (Primary SSE)

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer <service_token>" \
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

### `POST /api/v1/chat` (Non-streaming)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer <service_token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "Management of IgG4-related disease"}'
```

### `POST /api/v1/debug/decompose` (Debug, auth required)

```bash
curl -X POST http://localhost:8000/api/v1/debug/decompose \
  -H "Authorization: Bearer <service_token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "Systematic reviews on SGLT2 inhibitors from 2020"}'
```

### `GET /api/v1/health` + OpenAPI

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/openapi.json
```

Legacy aliases remain temporarily:
1. `/api/chat` -> `/api/v1/chat`
2. `/api/chat/stream` -> `/api/v1/chat/stream`
3. `/health` -> `/api/v1/health`

Auth behavior on non-health endpoints:
1. `401` missing or invalid bearer token
2. `403` disabled token

Production rule:
1. Never place `Authorization: Bearer ...` in client-side code.
2. Keep token injection in server-side route handlers only.

---

## 📊 Configuration Reference

### Embedding Models (via DeepInfra)

| Model | Dimension | Context | Use Case |
|-------|-----------|---------|----------|
| Qwen/Qwen3-Embedding-0.6B-batch | 1024 | 32K | **Default** |

### Reranker Models (via DeepInfra)

| Model | Max Tokens | Notes |
|-------|------------|-------|
| Qwen/Qwen3-Reranker-0.6B | 32K | **Default** |

### LLM Models (via DeepInfra)

| Model | Use Case |
|-------|----------|
| openai/gpt-oss-20b | **Primary generation** |

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
```

---

## 🚢 Deployment

### Production Server (Hetzner)

Qdrant is self-hosted on Hetzner AX52 (64GB RAM, AMD Ryzen 7000).

Use these production docs:
1. [deploy/hetzner/README.md](deploy/hetzner/README.md)
2. [deploy/integration/README.md](deploy/integration/README.md)

### Backend Deployment

```bash
gunicorn src.api_server:app \
  -k uvicorn.workers.UvicornWorker \
  -w 2 \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --graceful-timeout 90
```

### Frontend Deployment

```bash
cd frontend
npm run build
npm start  # Production mode
```

Production frontend integration checklist:
1. Browser calls only relative routes (for example `/api/chat/stream`).
2. Next.js server routes inject `Authorization: Bearer <service_token>`.
3. `RAG_API_BASE_URL` points to private Hetzner API endpoint.
4. `RAG_API_TOKEN` stays server-only (`NEXT_PUBLIC_*` must not include tokens).

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
- **Rerank Input**: up to 200 chunks (3 per article max; typically 150-200 with ~50-70 unique articles)
- **Rerank Output**: up to 100 chunks (can be lower after relevance filtering)
- **Passed to LLM Context**: up to 50 papers (article-level context blocks)

---

## 🔒 Security Notes

1. **Qdrant**: keep private on Docker internal network in production.
2. **Service auth**: non-health endpoints require bearer token validation from `API_KEYS_FILE`.
3. **Token storage**: store hashed tokens only in `api_keys.json`; rotate on schedule/incidents.
4. **Frontend security**: no service token in browser code, browser storage, or `NEXT_PUBLIC_*` envs.
5. **DeepInfra key**: keep in server environment only.
6. **CORS**: keep production origin list minimal.
7. **No PII**: pipeline processes public medical literature only.

---

## 🐛 Common Issues

### Qdrant Connection Timeout
```python
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
- **Data Ingestion Issues**: Check script logs in `/data/ingestion/logs/`
- **API Issues**: Review `src/api_server.py` logs

---

**Internal Use Only** | Do not distribute outside the organization
