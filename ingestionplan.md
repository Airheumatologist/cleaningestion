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
