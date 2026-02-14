# PRD: Medical RAG Migration to Hetzner Self-Hosted Qdrant (v4)

**Version:** 4.0  
**Date:** February 14, 2026  
**Status:** Approved for implementation  
**Supersedes:** `PRD-Migration-Hetzner-Qdrant-v3.md`

## 1. Executive Summary

This PRD defines the production migration of the Medical RAG system to a self-hosted Hetzner Qdrant architecture while maximizing clinically relevant corpus coverage and improving retrieval quality.

Primary goals:
1. Run Qdrant fully self-hosted on Hetzner (no managed Qdrant DB dependency).
2. Run ingestion/update execution on Hetzner as scheduled jobs.
3. Keep production RAG application/API code outside Hetzner; it connects remotely to Qdrant endpoints.
4. Ingest maximal relevant baseline data (PMC OA full text + PubMed baseline/updatefiles + DailyMed baseline/incrementals).
5. Improve retrieval quality by moving from single-truncated article embeddings to token-aware multi-chunk embeddings.

Key principle:
1. Embeddings must be generated from token-aware chunks (`384` tokens, `64` overlap), not a single char-capped article text.
2. OA embedding text must exclude reference/publication back-matter with conservative, structure-aware rules.

Non-goal:
1. Do not introduce dual-index/hierarchical chunking in v4.
2. Do not deploy full production RAG serving stack on Hetzner in v4.
3. Keep rollout to a single production-ready collection path.

## 2. What Changed From v3

1. Replaced char-capped embedding logic with token-aware chunking.
2. Added conservative section-aware back-matter exclusion policy for OA embeddings.
3. Added no-section fallback chunking behavior for articles with weak/absent structure.
4. Updated PMC source strategy to cloud-service primary with temporary FTP fallback during 2026 transition.
5. Kept reranker default at `rerank-v4-fast`.
6. Locked architecture boundary: Hetzner runs Qdrant + scheduled updater jobs; production app remains external.

## 3. Baseline Data Scope

Baseline ingestion scope is defined to maximize clinically relevant retrieval coverage.

1. PMC OA full text corpus and related OA-compatible article sets, including license metadata.
2. PubMed baseline plus ordered daily updatefiles to cover abstract-only records not present as PMC OA full text.
3. DailyMed full baseline plus incremental updates.

Coverage objective:
1. Maximize relevant medical evidence recall while preserving legal/licensing metadata and minimizing non-clinical embedding noise.

## 4. Source Download Strategy (2026-safe)

### PMC
1. Primary source mode: cloud-service-compatible PMC endpoints/layout.
2. Fallback mode: legacy FTP only during transition window.
3. Transition sunset: retire FTP fallback after August 2026.

### PubMed
1. Ingest annual baseline once.
2. Apply daily updatefiles in order.
3. Handle revised/deleted records in ingestion state and upsert/delete operations.

### DailyMed
1. Ingest baseline bulk release once.
2. Ingest incrementals through update feeds/API or date-based listing.

## 5. Ingestion and Embedding Design

### 5.1 Chunking
1. Chunk size: `384` tokens.
2. Overlap: `64` tokens.
3. Split order:
1. Section-aware segmentation first.
2. Token-window fallback second.

### 5.2 OA filtering for embedding text only
Exclude only high-confidence publication back-matter sections:
1. `references`, `bibliography`.
2. acknowledgments.
3. funding statements.
4. conflict/disclosure statements.
5. author contributions.
6. supplementary/data-availability/ethics/publisher-note back-matter.

### 5.3 Conservative matcher policy
1. Use `sec-type` and normalized exact/near-exact section-title matching.
2. Do not use aggressive fuzzy keyword exclusion.

### 5.4 Clinical keep precedence
Always keep clinically relevant content even when labels are near ambiguous:
1. pathogenesis/mechanism.
2. diagnosis/clinical features.
3. treatment/management.
4. outcomes/results/discussion/conclusion.

### 5.5 No-section article handling
1. If no section tags are present, split by paragraph/sentence blocks.
2. If abstract only, chunk abstract by token windows.
3. If no usable text exists, skip record and log skip reason.

### 5.6 Embedding text limits
1. Remove fixed `2000-char` truncation for OA full-text embedding path.
2. Enforce model-token budget as the true input constraint.

### 5.7 Payload policy
1. Keep richer `full_text` payload behavior for synthesis/reranking context unchanged.
2. Embed cleaned/chunked text only.

### 5.8 Important changes to public APIs / interfaces / types
Environment/config additions:
1. `EMBED_FILTER_ENABLED=true`
2. `EMBED_FILTER_MODE=conservative`
3. `EMBED_FILTER_PROFILE=clinical_backmatter`
4. `CHUNK_SIZE_TOKENS=384`
5. `CHUNK_OVERLAP_TOKENS=64`
6. `QDRANT_COLLECTION=rag_pipeline`
7. `SPARSE_ENABLED=true`
8. `SPARSE_MODE=bm25`

Ingestion schema changes:
1. Point granularity changes from article-level to chunk-level for OA/full-text and other long-form ingestion.
2. New chunk metadata fields are required in payload.
3. No external API contract changes for frontend/public HTTP endpoints; retrieval aggregation changes are internal.

## 6. Qdrant Data Model and Collection Setup

### 6.1 Collection and hosting
1. Qdrant remains self-hosted on Hetzner.
2. Quantization remains binary, with 2-bit patch where server/client support exists.
3. Collection name for v4 baseline and serving is `rag_pipeline` (Docker-hosted self-managed Qdrant).
4. Full baseline runs may recreate `rag_pipeline` explicitly when starting from scratch.
5. Hetzner host responsibilities are limited to vector DB and scheduled ingestion/update execution; production RAG app/runtime is out of scope on this host.
6. Day-1 retrieval mode is hybrid dense + BM25-style sparse vectors (Qdrant sparse index with IDF modifier).

### 6.2 Point model (chunk-level)
Each embedded chunk is a separate point with deterministic ID.

Required payload fields:
1. `doc_id`
2. `source`
3. `chunk_id`
4. `chunk_index`
5. `chunk_token_count`
6. `section_title`
7. `section_type`
8. `is_backmatter_excluded=false` for stored chunks
9. Existing metadata retained where available:
1. `year`
2. `journal`
3. `article_type`
4. `evidence_grade`
5. `country`
6. other existing retrieval/rerank metadata

### 6.3 Retrieval behavior
1. Retrieve top chunks.
2. Aggregate chunk hits to document-level candidates before rerank/synthesis.
3. Preserve citation traceability from document to chunk.

## 7. Monthly Incremental Update Design

Execution model:
1. Monthly/daily update runs execute on Hetzner, but as isolated updater runtime (containerized job image or minimal runner), not as full production app deployment.
2. Production clients consume Qdrant remotely via secured endpoint and do not require ingestion code deployment.

### 7.1 PubMed incrementals
1. Maintain processed-file tracker.
2. Use idempotent deterministic IDs for upsert safety.
3. Apply revised/deleted records handling each run.

### 7.2 DailyMed incrementals
1. Use incremental feed/API polling with checkpointing.
2. Reuse deterministic point IDs for idempotent writes.

### 7.3 OA incrementals
1. Apply same section-filter and chunking logic as baseline.
2. Keep processing rules consistent between baseline and monthly updates.

### 7.4 Cron schedule
1. Monthly update job.
2. Daily backup job.
3. Frequent health check job.
4. Cron invokes updater runtime artifact (container image) with env/config, without requiring a full repository checkout on Hetzner.

## 8. Quality Gates and Testing

### Test Cases and Scenarios

#### 8.1 Section filter unit tests
1. `References` excluded.
2. `Acknowledgments` excluded.
3. `Funding` excluded.
4. `Conflict of interest` excluded.
5. `Pathogenesis` included.
6. `Mechanism` included.

#### 8.2 Chunker tests
1. Sectioned article produces valid `384/64` chunks.
2. No-section article fallback behaves correctly.
3. Long abstract is chunked without losing trailing clinically relevant content.

#### 8.3 Ingestion tests
1. Deterministic IDs are stable across reruns.
2. Checkpoint/resume works correctly.
3. Skip/retry counters and logs are correct.

#### 8.4 Retrieval QA
1. Compare fixed medical query set before/after chunked filtering migration.
2. Validate no drop on mechanism/pathogenesis-heavy queries.
3. Confirm precision improvements on reference-heavy article sets.

#### 8.5 Safety tests
1. Ensure all writes target only the self-hosted Docker Qdrant endpoint and collection `rag_pipeline`.
2. Ensure collection recreation is only performed in explicit full-baseline mode.
3. Ensure public production API/service components are not deployed on Hetzner host as part of v4 rollout.

### Acceptance criteria
1. Back-matter text is excluded from embedding chunks in validation sample.
2. Clinical sections remain represented in embeddings.
3. End-to-end ingestion and retrieval pass regression suite.
4. All ingestion writes are confirmed in `rag_pipeline` on self-hosted Docker Qdrant.

## 9. Rollout, Cutover, and Safety

1. Prepare `rag_pipeline` collection for chunked corpus on self-hosted Docker Qdrant.
2. Pilot ingest on sample subset.
3. Medium-scale ingest + QA.
4. Full baseline ingest.
5. Cut over only after acceptance criteria are met.
6. Keep rollback path to previous serving collection.

Safety rules:
1. Production reads/writes must point to self-hosted Docker Qdrant.
2. Use explicit run mode flags for full-baseline recreate vs incremental update.
3. Enforce network controls for Qdrant endpoint (API key + IP allowlist and/or private tunnel).

## 10. Cost, Capacity, and Risk

### 10.1 Capacity/cost impact
1. Chunk-level indexing increases vector count versus article-level embedding.
2. Storage and ingestion duration increase are expected.
3. Mitigation: single-size chunking only (`384/64`) in v4; no dual-index/hierarchical expansion in initial release.

### 10.2 Risk register
1. PMC source transition risk in 2026.
2. Mitigation: cloud-primary with temporary FTP fallback and defined sunset.
3. Retrieval drift risk from schema change.
4. Mitigation: fixed QA set, staged rollout, and reversible cutover.
5. Operational coupling risk if updater and DB share same host.
6. Mitigation: isolate updater runtime, checkpoint every pipeline stage, and maintain daily snapshots.

## 11. File-Level Implementation Map

1. `scripts/01_download_pmc.py`
1. Cloud-primary + fallback logic.
2. `scripts/02_extract_pmc.py`
1. Section classifier + chunk materialization inputs.
3. `scripts/06_ingest_pmc.py`
1. Token chunking pipeline and chunk payload schema.
4. `scripts/08_monthly_update.py`
1. Same filtering/chunking behavior for incrementals.
5. `src/retriever_qdrant.py`
1. Chunk retrieval and document aggregation.
6. `env.example`
1. New chunk/filter environment variables.
7. `deploy/hetzner/*`
1. Keep Qdrant infrastructure baseline and add updater-runtime runbook (cron invoking container image).
2. Keep production serving code deployment outside Hetzner.

## 12. Assumptions and Defaults

1. New-from-scratch embedding run is required.
2. Serving collection name is `rag_pipeline`.
3. Conservative filter mode is default.
4. Single-size chunking (`384/64`) is default.
5. Embed-only filtering is default.
6. PubMed and DailyMed are included in baseline for maximum relevant coverage.
7. Hetzner runs Qdrant and scheduled updater jobs only.
8. Production RAG service remains external and communicates with Qdrant securely.

---
