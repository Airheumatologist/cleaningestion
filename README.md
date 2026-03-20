# Elixir AI - Medical RAG Pipeline

Internal repository for the Elixir medical retrieval-augmented generation (RAG) stack.

## Overview

This codebase runs a production-oriented medical QA pipeline with:

- FastAPI backend (`src/api_server.py`)
- Query decomposition + hybrid retrieval + reranking + answer synthesis (`src/rag_pipeline.py`)
- turbopuffer as the primary vector + full-text backend
- Runtime retrieval is turbopuffer-only
- Optional Next.js frontend (`frontend/`)

## Architecture (ASCII)

### Runtime request path

```text
┌───────────────────────────────┐
│ Client                        │
│ - Browser / service caller    │
└───────────────┬───────────────┘
                │ HTTP(S)
                ▼
┌───────────────────────────────┐
│ FastAPI                       │
│ src/api_server.py             │
│ /api/v1/chat, /chat/stream    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ MedicalRAGPipeline            │
│ src/rag_pipeline.py           │
│ 1) preprocess                 │
│ 2) retrieve                   │
│ 3) rerank                     │
│ 4) synthesize                 │
└───────────────┬───────────────┘
                │
                ▼
┌──────────────────────────────────────────────────┐
│ Retriever Factory                                │
│ src/retriever_factory.py                         │
│ RETRIEVAL_BACKEND=turbopuffer                    │
└───────────────┬──────────────────────────────────┘
                │
┌───────────────────────────────┐
│ turbopuffer retriever         │
│ primary runtime path          │
└───────────────────────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Reranker + LLM providers      │
│ DeepInfra reranker            │
│ Groq/DeepInfra generation     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ JSON or SSE response          │
└───────────────────────────────┘
```

### Ingestion pipeline path

```text
┌───────────────────────────────────────────────────────────┐
│ Source corpora                                            │
│ - PMC XML                                                 │
│ - DailyMed XML                                            │
│ - PubMed JSONL                                            │
└──────────────────────────────┬────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────┐
│ Ingestion scripts                                         │
│ scripts/06_ingest_pmc.py                                 │
│ scripts/07_ingest_dailymed.py                            │
│ scripts/21_ingest_pubmed_abstracts.py                    │
└──────────────────────────────┬────────────────────────────┘
                               │ build points + embeddings
                               ▼
┌───────────────────────────────────────────────────────────┐
│ Sink selector (compat shim)                              │
│ scripts/turbopuffer_ingestion_sink.py                    │
└───────────────┬─────────────────────────────┬─────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│ TurbopufferIngestionSink                                 │
│ - dense-only vectors                                      │
│ - native BM25 full-text fields                            │
│ - namespace routing: medical_pmc / medical_pubmed / ...   │
└───────────────┬───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│ turbopuffer namespaces                                     │
│ medical_pmc | medical_pubmed | medical_dailymed           │
└───────────────────────────────────────────────────────────┘
```

Migration tracker and validation logs:

- `ingestionplan.md`

## API Contract

Primary versioned endpoints:

- `POST /api/v1/chat`
- `POST /api/v1/chat/stream` (SSE)
- `GET /api/v1/health`
- `POST /api/v1/debug/decompose`

Legacy aliases still available:

- `POST /api/chat` -> `/api/v1/chat`
- `POST /api/chat/stream` -> `/api/v1/chat/stream`
- `GET /health` -> `/api/v1/health`
- `POST /api/query/decompose` -> `/api/v1/debug/decompose`

OpenAPI:

- `/api/v1/openapi.json`
- `/api/v1/docs`

### Auth

When `API_AUTH_ENABLED=true` (default), all non-health routes require:

```http
Authorization: Bearer <service_token>
```

Auth behavior:

- `401`: missing/invalid token
- `403`: token exists but disabled

Token file format (`api_keys.example.json`):

```json
{
  "tokens": [
    {
      "service_id": "azure-nextjs-app",
      "token_hash": "$2b$12$replace_with_bcrypt_hash",
      "enabled": true
    }
  ]
}
```

## Quickstart (Local API)

1. Create environment and install dependencies.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Copy env file and set required secrets.

```bash
cp env.example .env
```

Required keys:

- `DEEPINFRA_API_KEY`
- `GROQ_API_KEY`
- `TURBOPUFFER_API_KEY`

3. Set backend flags (recommended).

```bash
export RETRIEVAL_BACKEND=turbopuffer
export VECTOR_BACKEND=turbopuffer
export TURBOPUFFER_REGION=gcp-us-central1
export TURBOPUFFER_NAMESPACE_PMC=medical_pmc
export TURBOPUFFER_NAMESPACE_PUBMED=medical_pubmed
export TURBOPUFFER_NAMESPACE_DAILYMED=medical_dailymed
```

Optional local auth bypass:

```bash
export API_AUTH_ENABLED=false
```

4. Run API server.

```bash
PYTHONPATH=. python3 -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

5. Health check.

```bash
curl http://localhost:8000/api/v1/health
```

## API Usage Examples

Non-streaming:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query":"Latest guideline-supported RA treatment options","stream":false}'
```

Streaming (SSE):

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"query":"SGLT2 cardiovascular benefit evidence","stream":true}'
```

SSE emits `data: {...}` events and finishes with `data: [DONE]`.

## Frontend (Optional)

Next.js app lives in `frontend/`.

```bash
cd frontend
npm install
API_REWRITE_TARGET=http://localhost:8000 npm run dev
```

Notes:

- Browser calls should stay relative (`/api/chat/stream`).
- `frontend/src/app/api/chat/stream/route.ts` proxies to `API_REWRITE_TARGET` (or `NEXT_PUBLIC_API_URL`).
- That proxy forwards the incoming `Authorization` header.
- `frontend/src/app/page.tsx` currently includes a hardcoded dev bearer token for local testing; replace it for real environments.

## Ingestion Config (No-Ambiguity Reference)

### Source of truth and precedence

1. `scripts/config_ingestion.py` defines ingestion defaults and reads `.env` via `dotenv`.
2. Exported environment variables override `.env` values at runtime.
3. CLI flags (for example `--xml-dir` or `--input`) override path defaults for that run.

### Ingestion variables (`scripts/config_ingestion.py`)

| Variable | Default | Used for |
| --- | --- | --- |
| `VECTOR_BACKEND` | `turbopuffer` | Ingestion backend mode |
| `TURBOPUFFER_API_KEY` | empty | turbopuffer auth token |
| `TURBOPUFFER_NAMESPACE_PMC` | `medical_pmc` | PMC ingestion namespace |
| `TURBOPUFFER_NAMESPACE_PUBMED` | `medical_pubmed` | PubMed ingestion namespace |
| `TURBOPUFFER_NAMESPACE_DAILYMED` | `medical_dailymed` | DailyMed ingestion namespace |
| `TURBOPUFFER_WRITE_BATCH_SIZE` | `500` | Rows per write call |
| `TURBOPUFFER_MAX_CONCURRENT_WRITES` | `4` | Parallel write ceiling |
| `TURBOPUFFER_MAX_RETRIES` | `5` | Retry attempts for write failures |
| `INGEST_DRY_RUN` | `false` | Compute rows without writing |
| `BATCH_SIZE` | `100` | Ingestion batch size |
| `MAX_WORKERS` | `8` | Parallel workers |
| `MAX_RETRIES` | `3` | Retry attempts for transient failures |
| `EMBEDDING_PROVIDER` | `deepinfra` | Ingestion embedding provider |
| `INGESTION_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B-batch` | Ingestion embedding model |
| `CHUNK_SIZE_TOKENS` | `2048` | Chunk size |
| `CHUNK_OVERLAP_TOKENS` | `256` | Chunk overlap |

### Runtime retrieval variables (`src/config.py`)

| Variable | Default | Used for |
| --- | --- | --- |
| `RETRIEVAL_BACKEND` | `turbopuffer` | Runtime retriever choice |
| `TURBOPUFFER_API_KEY` | empty | turbopuffer auth token |
| `TURBOPUFFER_REGION` | `gcp-us-central1` | turbopuffer region for runtime queries |
| `TURBOPUFFER_NAMESPACE_PMC` | `medical_pmc` | PMC retrieval namespace |
| `TURBOPUFFER_NAMESPACE_PUBMED` | `medical_pubmed` | PubMed retrieval namespace |
| `TURBOPUFFER_NAMESPACE_DAILYMED` | `medical_dailymed` | DailyMed retrieval namespace |
| `TURBOPUFFER_TIMEOUT_SECONDS` | `30` | turbopuffer client timeout |
| `RETRIEVAL_RRF_K` | `60` | RRF fusion constant |
| `RETRIEVAL_DENSE_WEIGHT` | `0.7` | Dense component weight |
| `RETRIEVAL_SPARSE_WEIGHT` | `0.3` | BM25/lexical component weight |

### Recommended env profiles

turbopuffer primary (recommended):

```env
VECTOR_BACKEND=turbopuffer
TURBOPUFFER_API_KEY=<set-turbopuffer-api-key>
TURBOPUFFER_REGION=gcp-us-central1
TURBOPUFFER_NAMESPACE_PMC=medical_pmc
TURBOPUFFER_NAMESPACE_PUBMED=medical_pubmed
TURBOPUFFER_NAMESPACE_DAILYMED=medical_dailymed
TURBOPUFFER_WRITE_BATCH_SIZE=500
INGEST_DRY_RUN=false

RETRIEVAL_BACKEND=turbopuffer
TURBOPUFFER_TIMEOUT_SECONDS=30
```

## Ingestion (turbopuffer Primary)

Set backend before running ingestion scripts:

```bash
export VECTOR_BACKEND=turbopuffer
export TURBOPUFFER_API_KEY=<set-turbopuffer-api-key>
export TURBOPUFFER_REGION=gcp-us-central1
export TURBOPUFFER_NAMESPACE_PMC=medical_pmc
export TURBOPUFFER_NAMESPACE_PUBMED=medical_pubmed
export TURBOPUFFER_NAMESPACE_DAILYMED=medical_dailymed
```

Run source ingesters:

```bash
python3 scripts/06_ingest_pmc.py --xml-dir /data/ingestion/pmc_xml
python3 scripts/07_ingest_dailymed.py --xml-dir /data/ingestion/dailymed/xml
python3 scripts/21_ingest_pubmed_abstracts.py --input /data/ingestion/pubmed_baseline/filtered/pubmed_abstracts.jsonl
```

Direct PMC S3 ingest (no local XML staging):

```bash
# Precreate target namespace with first valid PMC write
python3 scripts/06_ingest_pmc_s3.py \
  --datasets pmc_oa,author_manuscript \
  --release-mode all \
  --max-files 50 \
  --workers 4 \
  --precreate-only

# Run direct S3 test ingestion
python3 scripts/06_ingest_pmc_s3.py \
  --datasets pmc_oa,author_manuscript \
  --release-mode all \
  --max-files 50 \
  --workers 4
```

## Backend Selection

Runtime retriever backend is selected by `RETRIEVAL_BACKEND`:

- `turbopuffer`: uses `src/retriever_turbopuffer.py`
- Any other value fails fast during retriever creation

## Validation Commands

Run targeted regression checks:

```bash
python3 -m pytest -q tests/test_turbopuffer_ingestion_sink.py
python3 -m pytest -q tests/test_pipeline_fanout_routing.py tests/test_pmc_ingest_author_manuscript.py tests/test_recency_selection.py
```

## Repo Layout

```text
src/                    API server and RAG runtime
scripts/                ingestion + migration utilities
schema/                 schema/index contracts (legacy + active)
tests/                  unit and migration gates
frontend/               optional Next.js UI
ingestionplan.md        migration execution tracker
```

## Security Notes

- Keep service tokens server-side only.
- Do not expose privileged API tokens in client-side code.
- Restrict CORS to known origins in production.
- Use HTTPS and private network routing for production API access.
