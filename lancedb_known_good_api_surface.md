# LanceDB Known-Good API Surface (Phase 1)

This file locks the canonical LanceDB API style for migration code and docs.

## Version Matrix

| Component | Version | Notes |
|---|---:|---|
| Python | 3.14.0 | Verified local runtime |
| LanceDB SDK | 0.29.2 | Exact pin in `requirements.txt` |
| API style | Sync | Canonical style for migration scripts and tests |

## Compatibility Gate

Run:

```bash
python3 scripts/lancedb_compat_smoke.py --json
```

CI gate:

```bash
python3 -m unittest tests/test_lancedb_compat_smoke.py
```

Both must pass before Phase 2+ changes.

## Canonical, Verified Snippets

### 1) Create Dense Index (`IVF_RQ`)

```python
table.create_index(
    metric="cosine",
    vector_column_name="vector",
    index_type="IVF_RQ",
    num_partitions=2,
    num_sub_vectors=2,
    replace=True,
)
```

### 2) Create Dense Challenger (`IVF_HNSW_SQ`)

```python
table.create_index(
    metric="cosine",
    vector_column_name="vector",
    index_type="IVF_HNSW_SQ",
    num_partitions=1,
    m=8,
    ef_construction=80,
    replace=True,
)
```

### 3) Create FTS + Scalar Filter Indexes

```python
table.create_fts_index("page_content", replace=True)
table.create_scalar_index("source", replace=True)
table.create_scalar_index("source_family", replace=True)
table.create_scalar_index("year", replace=True)
table.create_scalar_index("article_type", replace=True)
```

### 4) Wait for Index Build Completion

```python
index_names = [index.name for index in table.list_indices()]
table.wait_for_index(index_names)
```

### 5) Hybrid Query with RRF (Sync API)

```python
from lancedb.rerankers import RRFReranker

results = (
    table.search(query_type="hybrid")
    .vector(query_embedding)
    .text(query_text)
    .where("year >= 2020", prefilter=True)
    .rerank(reranker=RRFReranker())
    .limit(20)
    .to_list()
)
```

Important: for this SDK version, do not pass the text query directly into `search(...)` when `query_type="hybrid"`. Use `.text(...)` instead.

## Phase 3 Index Strategy Commands

Profile manifest:

```bash
cat schema/lancedb_index_profiles.json
```

Build indexes on a real table:

```bash
python3 scripts/lancedb_index_manager.py --uri ./medical_data.lancedb --table medical_docs --profile ivf_rq build --json
```

Planner validation (index usage):

```bash
python3 scripts/lancedb_index_manager.py --uri ./medical_data.lancedb --table medical_docs --profile ivf_rq validate --json
```
