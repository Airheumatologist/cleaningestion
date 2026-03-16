# LanceDB Migration Phase 1-8 Implementation Review

## Executive Summary

This document contains a review of the Phase 1-8 implementation for the Qdrant → LanceDB migration. Overall, the implementation is solid and follows the planned architecture. However, several discrepancies have been identified that should be addressed before starting the ingestion process.

---

## Critical Discrepancies (Must Fix Before Ingestion)

### 1. Vector Fields Missing from Schema Contract

**Severity**: HIGH

**Issue**: The `vector`, `sparse_indices`, and `sparse_values` fields are not documented in `schema/lancedb_schema_contract.json` or `lancedb_schema_parity_spec.md`.

**Evidence**:
- The ingestion sink (`scripts/lancedb_ingestion_sink.py` lines 96-103) writes these fields:
  ```python
  row = {
      "point_id": str(getattr(point, "id", "")),
      "vector": self._extract_dense_vector(vector),
      "sparse_indices": sparse_indices,
      "sparse_values": sparse_values,
  }
  ```
- The index profiles (`schema/lancedb_index_profiles.json`) reference `vector_column: "vector"`
- However, these fields are absent from the schema contract

**Impact**: Schema parity CI gate does not validate the most critical fields for vector search.

**Recommendation**: Add the following to the schema contract:
```json
{
  "vector": {
    "nullable": false,
    "required": true,
    "types": ["list<float>"],
    "description": "Dense embedding vector (1024-dim for Qwen3-Embedding-0.6B)"
  },
  "sparse_indices": {
    "nullable": false,
    "required": true,
    "types": ["list<int>"],
    "description": "BM25 sparse vector indices"
  },
  "sparse_values": {
    "nullable": false,
    "required": true,
    "types": ["list<float>"],
    "description": "BM25 sparse vector values"
  },
  "point_id": {
    "nullable": false,
    "required": true,
    "types": ["str"],
    "description": "Original Qdrant point ID"
  }
}
```

---

### 2. [P0] PMC Full-Text Path Will Crash at Runtime

**Severity**: CRITICAL

**Verified**: ✅ CONFIRMED

**Location**: 
- `src/retriever_lancedb.py` line 305-316: `get_all_chunks_for_doc()`
- Called by `src/rag_pipeline.py` line 1194

**Issue**: The code uses `.search(query_type="fts").text(...)` which is not valid in LanceDB 0.29.2:
```python
rows = (
    self.table.search(query_type="fts")
    .text(doc_id)  # <-- NotImplementedError: text() is not valid for query_type="fts"
    .where(where_sql, prefilter=True)
    .limit(512)
    .to_list()
)
```

**Impact**: PMC full-text reconstruction will crash with `NotImplementedError`, breaking request handling for any PMC documents that need full-text retrieval.

**Evidence**: 
- `lancedb_known_good_api_surface.md` documents the correct hybrid search API:
  ```python
  results = (
      table.search(query_type="hybrid")
      .vector(query_embedding)
      .text(query_text)  # text() is only valid with hybrid
      ...
  )
  ```
- The FTS-only search should use `table.search(text_query, query_type="fts")` directly

**Recommendation**: 
- Option A: Use `self.table.search(doc_id, query_type="fts")` for FTS-only search
- Option B: Use a different retrieval pattern that doesn't require FTS text search (e.g., `.where()` with `LIKE` on `page_content`)

---

### 3. [P1] Auto-Reindex During Ingestion Is Broken

**Severity**: HIGH

**Verified**: ✅ CONFIRMED

**Location**: `scripts/lancedb_ingestion_sink.py` lines 128-139

**Issue**: Subprocess argument order is incorrect. The current code places `--json` after the subcommand:
```python
cmd = [
    "python3",
    "scripts/lancedb_index_manager.py",
    "--manifest", str(self.manifest_path),  # global arg
    "--uri", self.uri,                       # global arg
    "--table", self.table_name,              # global arg
    "build",                                  # subcommand
    "--json",                                 # global arg (WRONG POSITION)
]
```

But `lancedb_index_manager.py` (lines 224-235) uses argparse with subparsers:
```python
parser.add_argument("--json", action="store_true", ...)  # global arg
subparsers.add_parser("build", ...)  # subcommand
```

With subparsers, global arguments MUST come before the subcommand. The current ordering causes:
```
unrecognized arguments: --json
```

**Impact**: Periodic reindex never runs; new tables remain without the intended vector/FTS/scalar index profile. This means:
- No vector index on newly created tables
- No FTS index for hybrid search
- No scalar indexes for filtering
- Queries will fall back to slow brute-force scans

**Recommendation**: Move `--json` before the subcommand:
```python
cmd = [
    "python3",
    "scripts/lancedb_index_manager.py",
    "--manifest", str(self.manifest_path),
    "--uri", self.uri,
    "--table", self.table_name,
    "--json",  # <-- Move here, before subcommand
    "build",
]
```

---

### 4. [P1] Dense-Only Ranking Is Reversed (Worst Matches Ranked First)

**Severity**: HIGH

**Verified**: ✅ CONFIRMED

**Location**: `src/retriever_lancedb.py` lines 224-232

**Issue**: `_distance` is used directly as score and sorted descending:
```python
dense_rows = self._search_dense(query_embedding, where_sql, self.n_retrieval)
passages = [
    self._apply_retrieval_recency_boost(
        self._format_row_passage(row, score=float(row.get("_distance", 0.0)), stype="vector_search"),
        filter_kwargs=filter_kwargs,
    )
    for row in dense_rows
]
passages.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)  # DESCENDING
```

For cosine distance (used in the index profile), **lower is better**. The current logic:
- Distance 0.0 = identical vectors (best match)
- Distance 2.0 = completely opposite vectors (worst match)

Sorting descending puts distance 2.0 first!

**Impact**: 
- Dense-only paths return worst matches first
- Affects `retrieve_additional_papers()` (line 302) which calls `retrieve_passages(use_hybrid=False)`
- Affects any fallback to dense-only search

**Evidence**: 
- `schema/lancedb_index_profiles.json` specifies `"metric": "cosine"`
- For cosine similarity: `similarity = 1 - distance` (when distance is normalized)
- Correct scoring should be: `score = 1.0 - distance` or `score = -distance`

**Recommendation**: Convert distance to similarity score:
```python
# Option 1: Invert distance (for normalized cosine distance 0-2 range)
score = 1.0 - (float(row.get("_distance", 0.0)) / 2.0)

# Option 2: Negate (works for ranking, though scores are negative)
score = -float(row.get("_distance", 0.0))

# Option 3: Use 1 - distance (if distance is already 0-1 normalized)
score = 1.0 - float(row.get("_distance", 0.0))
```

---

## Minor Discrepancies (Should Fix)

### 5. [P2] Gov-Agency Filter Parity Gap vs Qdrant

**Severity**: MEDIUM

**Verified**: ✅ CONFIRMED

**Location**: 
- `src/retriever_lancedb.py` lines 48-86: `_build_where_sql()`
- `src/retriever_qdrant.py` lines 690-695

**Issue**: LanceDB filter builder handles `is_gov_affiliated` but not `gov_agency`:
```python
# LanceDB (line 80-84):
if kwargs.get("is_gov_affiliated") is not None:
    bool_value = kwargs["is_gov_affiliated"]
    ...
    clauses.append(f"is_gov_affiliated = {'true' if bool_value else 'false'}")
# Missing: gov_agency filter
```

Qdrant path supports it:
```python
# Qdrant (lines 690-695):
if "gov_agency" in kwargs and kwargs["gov_agency"]:
    agencies = [a.strip() for a in kwargs["gov_agency"].split(",")]
    if len(agencies) == 1:
        conditions.append(
            FieldCondition(key="gov_agencies", match=MatchValue(value=agencies[0]))
        )
```

**Impact**: Behavioral parity mismatch. Users cannot filter by specific government agencies (e.g., "FDA", "CDC") in LanceDB mode, only by the boolean `is_gov_affiliated` flag.

**Recommendation**: Add gov_agency filter to LanceDB:
```python
if kwargs.get("gov_agency"):
    agencies = [a.strip() for a in str(kwargs["gov_agency"]).split(",") if a.strip()]
    if agencies:
        # LanceDB doesn't have array containment operator like Qdrant
        # Use OR of LIKE conditions or require exact match
        agency_conditions = [f"gov_agencies LIKE '%{self._safe_sql(a)}%'" for a in agencies]
        clauses.append(f"({' OR '.join(agency_conditions)})")
```

---

### 6. Vector Dimension Configuration Inconsistency

**Severity**: MEDIUM

**Issue**: The schema contract doesn't specify vector dimensions, but the index profiles assume specific configurations.

**Evidence**:
- `src/config.py` line 123: `EMBEDDING_DIMENSION = 1024`
- `scripts/config_ingestion.py` lines 108-110: `EMBEDDING_DIMENSIONS = {"deepinfra": 1024}`
- `schema/lancedb_index_profiles.json` uses `num_sub_vectors: 64` which assumes 1024-dim vectors (1024/64 = 16)

**Impact**: If vector dimensions change, the index configuration may become invalid.

**Recommendation**: Add vector dimension validation to the schema parity gate.

---

### 7. License Field Missing from Schema Contract

**Severity**: MEDIUM

**Issue**: The `license` field is set in PMC ingestion but not in schema contract.

**Evidence**:
- `scripts/06_ingest_pmc.py` line 240: `"license": article.get("license") or content_flags.get("license", "unknown")`
- Field is not present in `schema/lancedb_schema_contract.json`

**Impact**: Schema drift - field exists in data but not in contract.

**Recommendation**: Add `license` field to schema contract for PMC sources.

---

### 8. Test Dependency on External Tool (ripgrep)

**Severity**: LOW

**Issue**: `tests/test_lancedb_phase5_8_tools.py::test_phase8_decommission_audit_outputs_contract` fails when `rg` (ripgrep) is not installed.

**Evidence**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'rg'
```

**Impact**: Test suite fails on environments without ripgrep.

**Recommendation**: 
- Option A: Use `grep -r` as fallback when `rg` is not available
- Option B: Skip test gracefully with informative message
- Option C: Document ripgrep as a test dependency

---

### 9. Sparse Vector Storage Format

**Severity**: LOW

**Issue**: Sparse vectors are stored as separate `sparse_indices` and `sparse_values` arrays, but LanceDB has native sparse vector support.

**Evidence**:
- Current implementation stores sparse vectors as two parallel arrays
- LanceDB 0.29.2 supports native sparse vector type

**Impact**: Minor storage inefficiency; search uses BM25 via FTS index instead of sparse vector index.

**Recommendation**: Consider migrating to native sparse vectors in future iteration if hybrid search performance needs improvement.

---

### 10. Text Field Mapping in Retriever

**Severity**: LOW

**Issue**: The retriever maps `page_content` to `text` field in output, but there's inconsistency in how this is handled.

**Evidence**:
- `src/retriever_lancedb.py` line 142: `"text": row.get("page_content") or row.get("abstract", "")`
- Some legacy code may expect a `text` field in the database

**Impact**: Potential confusion between `text` and `page_content` fields.

**Recommendation**: Document that `page_content` is the canonical field in LanceDB, and `text` is only in output contract for backward compatibility.

---

## Verification Checklist

### Schema Parity
- [x] pubmed_abstract fields documented (50 fields)
- [x] pmc_oa fields documented (37 fields)
- [x] pmc_author_manuscript fields documented (37 fields)
- [x] dailymed fields documented (53 fields)
- [ ] Vector fields (vector, sparse_indices, sparse_values) documented ❌
- [ ] License field documented ❌

### Vector Configuration
- [x] Dimension: 1024 (matches Qwen3-Embedding-0.6B)
- [x] Metric: cosine
- [x] Index type: IVF_RQ (default)
- [x] FTS index on page_content
- [x] Scalar indexes on source, source_family, year, article_type

### Retriever Contract
- [x] score field preserved
- [x] raw_score field preserved
- [x] dense_score field preserved (hybrid)
- [x] sparse_score field preserved (hybrid)
- [x] stype field preserved
- [x] All metadata fields mapped
- [x] DailyMed retrieval path preserved
- [ ] Dense-only ranking correct ❌ (REVERSED)
- [ ] PMC full-text retrieval works ❌ (WILL CRASH)
- [ ] Gov-agency filter parity ❌ (MISSING)

### Ingestion Pipeline
- [x] Backend-agnostic sink abstraction
- [x] Qdrant sink maintained for rollback
- [x] LanceDB sink implemented
- [x] Point ID preservation
- [x] Vector extraction from Qdrant format
- [ ] Auto-reindex works ❌ (ARG ORDER BROKEN)

---

## Recommendations Summary

### Before Ingestion Starts (Critical)

1. **[P0] Fix PMC full-text FTS API call** (CRITICAL)
   - Change `.search(query_type="fts").text(doc_id)` to `.search(doc_id, query_type="fts")`

2. **[P1] Fix auto-reindex argument order** (HIGH)
   - Move `--json` before `build` subcommand in subprocess call

3. **[P1] Fix dense-only ranking** (HIGH)
   - Convert `_distance` to similarity: `score = 1.0 - (distance / 2.0)`

4. **[P2] Add gov_agency filter** (MEDIUM)
   - Implement gov_agencies LIKE filter for parity with Qdrant

5. **Add vector fields to schema contract** (HIGH)
   - Update `scripts/lancedb_schema_parity.py` to include vector field validation
   - Regenerate `schema/lancedb_schema_contract.json`

### Post-Ingestion (Nice to Have)

6. Add license field to schema contract
7. Fix ripgrep dependency in tests
8. Document text vs page_content field mapping
9. Consider native sparse vector support

---

## Test Results

| Test Suite | Status | Notes |
|------------|--------|-------|
| test_lancedb_compat_smoke | ✅ PASS | API compatibility verified |
| test_lancedb_schema_parity | ✅ PASS | Schema validation passes (but missing vector fields) |
| test_lancedb_index_manager | ✅ PASS | Index build/validate works |
| test_lancedb_retriever_contract | ✅ PASS | Retriever output contract verified |
| test_lancedb_ingestion_sink | ✅ PASS | Sink writes correctly |
| test_lancedb_dual_read_canary | ✅ PASS | CLI works (skipped actual canary) |
| test_phase5_benchmark_demo | ✅ PASS | Benchmark harness works |
| test_phase8_decommission_audit | ❌ FAIL | Missing ripgrep dependency |

---

## Conclusion

The Phase 1-8 implementation has **critical runtime bugs** that MUST be fixed before ingestion:

1. **PMC full-text will crash** - FTS API usage is incorrect
2. **Auto-reindex is broken** - Argument order prevents index creation
3. **Dense ranking is inverted** - Worst matches returned first
4. **Gov-agency filter missing** - Behavioral parity gap

These are in addition to the **schema documentation gaps** identified earlier.

**DO NOT START INGESTION** until at minimum:
- [ ] P0: PMC full-text FTS API is fixed
- [ ] P1: Auto-reindex argument order is fixed
- [ ] P1: Dense-only ranking is fixed

The implementation is structurally sound but has implementation bugs that will cause production failures.
