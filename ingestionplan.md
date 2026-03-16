# Ingestion Migration Plan Tracker (Phase 1)

## Objective
- Execute **Phase 1 cleanup only** for Qdrant -> LanceDB migration.
- Remove retired Qdrant/Hetzner artifacts that are safe in this phase.
- Track execution and verification in this file.

## Delete Scope (Phase 1 Exact Targets)
- `deploy/hetzner/` (entire directory)
- `deploy/hetzner_setup.md`
- `scripts/40_qdrant_truth_snapshot.py`
- `scripts/41_retrieval_benchmark.py`
- `scripts/42_qdrant_safe_tune.py`
- `scripts/43_latency_split.py`
- `scripts/44_compact_segments.py`
- `src/medical_qdrant_client.py`
- `tests/test_docs_alignment.py`

## Execution Checklist
- [x] Confirm Phase 1 target paths exist
- [x] Delete all Phase 1 target paths
- [x] Run stale-reference grep for deleted files/paths
- [x] Remove stale references found in docs/rules
- [x] Re-run stale-reference grep (clean)
- [x] Run unit tests (excluding files blocked by Python 3.14 `crypt` removal)
- [x] Run light API import smoke check
- [x] Keep Phase 2 runtime/ingestion core files untouched

## Audit Log
- `2026-03-15T23:41:13Z` Verified Phase 1 targets existed:
  - `ls -ld deploy/hetzner deploy/hetzner_setup.md scripts/40_qdrant_truth_snapshot.py scripts/41_retrieval_benchmark.py scripts/42_qdrant_safe_tune.py scripts/43_latency_split.py scripts/44_compact_segments.py src/medical_qdrant_client.py tests/test_docs_alignment.py`
- `2026-03-15T23:41:13Z` Deleted Phase 1 targets:
  - `rm -rf deploy/hetzner deploy/hetzner_setup.md scripts/40_qdrant_truth_snapshot.py scripts/41_retrieval_benchmark.py scripts/42_qdrant_safe_tune.py scripts/43_latency_split.py scripts/44_compact_segments.py src/medical_qdrant_client.py tests/test_docs_alignment.py`
- `2026-03-15T23:41:13Z` Confirmed deletions:
  - `ls -ld ...` returned `No such file or directory` for all targets
- `2026-03-15T23:41:13Z` Ran stale-reference grep:
  - `rg -n "deploy/hetzner|deploy/hetzner_setup\.md|40_qdrant_truth_snapshot\.py|41_retrieval_benchmark\.py|42_qdrant_safe_tune\.py|43_latency_split\.py|44_compact_segments\.py|medical_qdrant_client\.py|test_docs_alignment\.py" -S README.md .cursorrules src scripts tests start_ingestion.sh env.example requirements.txt`
  - Found stale references in `README.md` and `.cursorrules`
- `2026-03-15T23:41:13Z` Removed stale references from:
  - `README.md`
  - `.cursorrules`
- `2026-03-15T23:41:13Z` Re-ran stale-reference grep:
  - Same command as above, result: **no matches**
- `2026-03-15T23:41:13Z` Test runs:
  - `pytest -q tests` -> failed during collection (`src` import path + Python 3.14 `crypt` issue)
  - `PYTHONPATH=. pytest -q tests` -> collection blocked by Python 3.14 `crypt` removal
  - `PYTHONPATH=. pytest -q tests --ignore=tests/test_api_contract.py --ignore=tests/test_service_auth.py` -> **36 passed**
- `2026-03-15T23:41:13Z` API import smoke:
  - `python3 -c "import src.api_server"` -> failed on Python 3.14 (`ModuleNotFoundError: crypt`)
  - `python3.12 -c "import src.api_server; print('api_server import ok (py3.12)')"` -> **passed**

## Verification
- Deleted targets are absent.
- No stale references remain to deleted Phase 1 files/paths in:
  - `README.md`, `.cursorrules`, `src/`, `scripts/`, `tests/`, `start_ingestion.sh`, `env.example`, `requirements.txt`
- Unit tests:
  - `36 passed` with two service-auth tests ignored due Python 3.14 `crypt` deprecation/removal.
- API startup-path smoke import:
  - Passes on Python 3.12.

## Open Risks
- Local default interpreter is Python 3.14, where `crypt` is unavailable; this breaks `src/service_auth.py` imports and two test modules during collection.
- If CI/runtime uses Python 3.14+, service-auth implementation needs migration away from `crypt`.

---

# LanceDB Migration Tracker (Phase 1 + Phase 2 + Phase 3)

## Scope Completed
- Phase 1: LanceDB SDK/version hardening and canonical API lock.
- Phase 2: schema parity gate with source-specific constraints and CI failure on drift.
- Phase 3: index strategy implementation with reproducible profile manifest and planner-use validation.

## Artifacts Added/Updated
- `requirements.txt`
  - Added strict pin: `lancedb==0.29.2`
- `scripts/lancedb_compat_smoke.py`
  - Compatibility smoke script for index/search APIs.
- `tests/test_lancedb_compat_smoke.py`
  - CI gate for compatibility smoke.
- `lancedb_known_good_api_surface.md`
  - Version matrix + canonical sync API snippets.
- `scripts/lancedb_schema_parity.py`
  - Schema parity validator from real payload constructors in:
    - `scripts/21_ingest_pubmed_abstracts.py`
    - `scripts/06_ingest_pmc.py`
    - `scripts/07_ingest_dailymed.py`
- `schema/lancedb_schema_contract.json`
  - Locked schema contract snapshot used by CI parity gate.
- `lancedb_schema_parity_spec.md`
  - Field-by-field parity table across:
    - `pubmed_abstract`
    - `pmc_oa`
    - `pmc_author_manuscript`
    - `dailymed`
- `tests/test_lancedb_schema_parity.py`
  - CI gate test that fails on field/type/nullability/required drift.
- `schema/lancedb_index_profiles.json`
  - Reproducible index configs for:
    - default candidate: `ivf_rq`
    - challenger: `ivf_hnsw_sq`
  - Includes FTS/scalar filter index configuration and prefilter default.
- `scripts/lancedb_index_manager.py`
  - Index strategy manager with commands:
    - `build`
    - `status`
    - `validate`
- `tests/test_lancedb_index_manager.py`
  - CI gate test for Phase 3 build/status/validate flow (demo mode).
- `lancedb_migration_research_plan.md`
  - Added “Phase 1 Hardening Outputs (Implemented)”, “Phase 2 Schema Parity Outputs (Implemented)”, and “Phase 3 Index Strategy Outputs (Implemented)”.

## Commands Run and Results
- `python3 scripts/lancedb_compat_smoke.py --json`
  - Result: `status=ok`
  - Verified:
    - `create_index` for `IVF_RQ` and `IVF_HNSW_SQ`
    - `create_fts_index`
    - scalar indexes (`source`, `source_family`, `year`, `article_type`)
    - `wait_for_index`
    - hybrid search (`query_type="hybrid"`) + `RRFReranker`
    - planner contains both ANN and FTS operators
- `python3 -m unittest tests/test_lancedb_compat_smoke.py`
  - Result: `OK`
- `python3 scripts/lancedb_schema_parity.py --write-contract --json`
  - Result: wrote `schema/lancedb_schema_contract.json`
- `python3 scripts/lancedb_schema_parity.py --write-spec --json`
  - Result: wrote `lancedb_schema_parity_spec.md`
- `python3 scripts/lancedb_schema_parity.py --json`
  - Result: `status=ok`, `errors=[]`
- `python3 -m unittest tests/test_lancedb_schema_parity.py`
  - Result: `OK`
- `python3 scripts/lancedb_index_manager.py --demo --json build`
  - Result: `status=ok` (FTS + scalar indexes + `IVF_RQ` built)
- `python3 scripts/lancedb_index_manager.py --demo --json status`
  - Result: `status=ok` (index inventory includes `vector_idx`, `page_content_idx`, scalar indexes)
- `python3 scripts/lancedb_index_manager.py --demo --json validate`
  - Result: `status=ok` with planner assertions:
    - `planner_has_ann=true`
    - `planner_has_fts=true`
    - `planner_has_filter=true`
- `python3 -m unittest tests/test_lancedb_index_manager.py`
  - Result: `OK`

## Current Gates (Phase 1/2/3)
- Compatibility/API gate:
  - `python3 -m unittest tests/test_lancedb_compat_smoke.py`
- Schema parity gate:
  - `python3 -m unittest tests/test_lancedb_schema_parity.py`
- Index strategy gate:
  - `python3 -m unittest tests/test_lancedb_index_manager.py`

## Latest Audit Update (2026-03-16)
- Phase 3 implementation completed and verified.
- Added index strategy files:
  - `schema/lancedb_index_profiles.json`
  - `scripts/lancedb_index_manager.py`
  - `tests/test_lancedb_index_manager.py`
- Updated supporting docs:
  - `lancedb_known_good_api_surface.md`
  - `lancedb_migration_research_plan.md`
- Final gate re-run status:
  - `python3 -m unittest tests/test_lancedb_compat_smoke.py` -> `OK`
  - `python3 -m unittest tests/test_lancedb_schema_parity.py` -> `OK`
  - `python3 -m unittest tests/test_lancedb_index_manager.py` -> `OK`

## Phase 4-8 Implementation Update (2026-03-16)

### Phase 4: Retriever Parity Adapter
- [x] Added backend selector + rollback-safe factory:
  - `src/retriever_factory.py`
  - `src/config.py` (`RETRIEVAL_BACKEND`, `RETRIEVAL_BACKEND_ROLLBACK_ON_ERROR`, LanceDB URI/table settings)
- [x] Implemented LanceDB retriever preserving existing output contract:
  - `src/retriever_lancedb.py`
  - Hybrid RRF-compatible path (`dense + FTS + fused scoring`)
  - Preserved key fields (`score`, `raw_score`, `dense_score`, `sparse_score`, `stype`, source metadata)
  - Preserved filter semantics (`year`, `source_family`, `article_type`, venue, gov filters)
  - Preserved DailyMed retrieval path
- [x] Wired pipeline to backend-agnostic retriever creation:
  - `src/rag_pipeline.py`
- [x] Added contract coverage:
  - `tests/test_lancedb_retriever_contract.py`

### Phase 5: Benchmark Shootout and Selection
- [x] Added benchmark harness:
  - `scripts/lancedb_benchmark_shootout.py`
- [x] Harness supports:
  - `IVF_RQ` vs `IVF_HNSW_SQ`
  - filtered + unfiltered hybrid workloads
  - concurrency execution
  - `P50/P95/P99`, throughput, error rate, weighted profile selection
- [x] Added CI test:
  - `tests/test_lancedb_phase5_8_tools.py::test_phase5_benchmark_demo`

### Phase 6: Ingestion Pipeline Migration
- [x] Added backend-agnostic ingestion sink abstraction:
  - `scripts/lancedb_ingestion_sink.py`
  - `QdrantIngestionSink` + `LanceDBIngestionSink`
- [x] Migrated ingestion writes in:
  - `scripts/06_ingest_pmc.py`
  - `scripts/07_ingest_dailymed.py`
  - `scripts/21_ingest_pubmed_abstracts.py`
- [x] Added backend/dry-run/index-refresh controls:
  - `scripts/config_ingestion.py`
  - `VECTOR_BACKEND` (default `lancedb`)
  - `INGEST_DRY_RUN`
  - `LANCEDB_REINDEX_INTERVAL_BATCHES`
- [x] Added ingestion sink test:
  - `tests/test_lancedb_ingestion_sink.py`

### Phase 7: Validation, Canary, and Cutover
- [x] Added dual-read canary comparator:
  - `scripts/lancedb_dual_read_canary.py`
- [x] Includes:
  - sampled traffic (`--sample-rate`)
  - top-k overlap metric
  - source consistency metric
  - staged rollout plan `[1,5,25,50,100]`
  - explicit rollback-ready status output
- [x] Added CLI gate test:
  - `tests/test_lancedb_dual_read_canary.py`

### Phase 8: Decommission and Cleanup
- [x] Added runtime coupling audit:
  - `scripts/lancedb_decommission_audit.py`
- [x] Audit reports blocking Qdrant references outside allowlist and enforces emergency snapshot reminder.
- [x] Added CI test:
  - `tests/test_lancedb_phase5_8_tools.py::test_phase8_decommission_audit_outputs_contract`

## Remaining Operational Work (Post-Implementation)
- [ ] Run full benchmark on production-like corpus and lock selected dense profile.
- [ ] Execute full dual-read burn-in window and SLO signoff before 100% cutover.
- [ ] Perform stability-window cleanup and physically remove Qdrant runtime paths.

## Latest Verification Update (2026-03-16)
- Re-ran migration gates and new phase 4-8 contract tests in one pass:
  - `PYTHONPATH=. python3 -m unittest tests/test_lancedb_compat_smoke.py tests/test_lancedb_schema_parity.py tests/test_lancedb_index_manager.py tests/test_lancedb_retriever_contract.py tests/test_lancedb_ingestion_sink.py tests/test_lancedb_dual_read_canary.py tests/test_lancedb_phase5_8_tools.py tests/test_pipeline_fanout_routing.py tests/test_retriever_fanout.py`
  - Result: `OK` (`18` tests passed)
- Re-ran ingestion regression for modified PMC path:
  - `PYTHONPATH=. python3 -m unittest tests/test_pmc_ingest_author_manuscript.py`
  - Result: `OK`
- Net status:
  - Phase 1-8 migration implementation is now covered by passing compatibility, schema, index, retriever contract, ingestion sink, canary tooling, benchmark tooling, and routing/fanout tests.
