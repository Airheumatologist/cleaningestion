# LanceDB Migration Research Plan: Optimal Ingestion Configuration

## Executive Summary

This document provides a research-based plan for migrating from Qdrant self-hosting to LanceDB for medical document ingestion (PubMed, DailyMed, PMC). The goal is to achieve optimal dense and sparse retrieval performance while preserving ALL existing schema fields **exactly as they currently exist**.

**Key Finding**: LanceDB does NOT natively support sparse vectors as a separate column type. Instead, sparse retrieval should use LanceDB's **Full-Text Search (FTS)** index with BM25 ranking, which is optimized for keyword-based retrieval.

**CRITICAL NOTE**: All API examples in this document are based on LanceDB Python SDK documentation as of March 2026. The LanceDB Python API is evolving rapidly - **validate all syntax against your specific lancedb version (`pip show lancedb`) before implementation**.

### Phase 1 Hardening Outputs (Implemented)

- `requirements.txt` now pins `lancedb==0.29.2`.
- Compatibility smoke script added: `scripts/lancedb_compat_smoke.py`
- CI gate test added: `tests/test_lancedb_compat_smoke.py`
- Canonical, verified API reference added: `lancedb_known_good_api_surface.md`

### Phase 2 Schema Parity Outputs (Implemented)

- Canonical schema contract snapshot: `schema/lancedb_schema_contract.json`
- Schema parity validator: `scripts/lancedb_schema_parity.py`
- CI parity gate test: `tests/test_lancedb_schema_parity.py`
- Field-by-field parity matrix + source constraints: `lancedb_schema_parity_spec.md`

### Phase 3 Index Strategy Outputs (Implemented)

- Reproducible index profile manifest: `schema/lancedb_index_profiles.json`
- Index manager CLI: `scripts/lancedb_index_manager.py`
  - `build`: create FTS + scalar + dense profile indexes
  - `status`: report index inventory and dense profile state
  - `validate`: assert planner uses ANN + FTS + filter with prefilter mode
- CI index strategy gate test: `tests/test_lancedb_index_manager.py`

---

## 1. Current State Analysis

### 1.1 EXACT Schema Fields (Preserved Unchanged)

Based on exhaustive analysis of the ingestion scripts (`21_ingest_pubmed_abstracts.py`, `06_ingest_pmc.py`, `07_ingest_dailymed.py`), the following fields MUST be preserved with their **exact current types and structures**:

#### Core Identifiers (All Sources)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `doc_id` | string | Primary document identifier |
| `chunk_id` | string | Unique chunk identifier |
| `pmcid` | string | PubMed Central ID |
| `pmid` | string | PubMed ID |
| `doi` | string | Digital Object Identifier |
| `pii` | string | Publisher Item Identifier |
| `other_ids` | dict/map | Additional identifiers (preserved as struct) |
| `set_id` | string | DailyMed set identifier |

#### Content Fields (All Sources)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `page_content` | string | Full text content for retrieval (FTS indexed) |
| `title` | string | Document title |
| `abstract` | string | Abstract text |
| `full_text` | string | Full text content |
| `text_preview` | string | Preview text (first 500 chars) |

#### Section & Chunking Metadata (All Sources)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `chunk_index` | int | Chunk sequence number |
| `total_chunks` | int | Total chunks for document |
| `token_count` | int | Token count estimate |
| `section_type` | string | Type of section (abstract, methods, etc.) |
| `section_title` | string | Section heading |
| `section_id` | string | Unique section identifier |
| `parent_section_id` | string/null | Parent section for sub-chunks |
| `full_section_text` | string | Complete section text |
| `section_weight` | float | Retrieval priority weight |

#### Publication Metadata (PubMed/PMC)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `journal` | string | Journal name |
| `journal_full` | struct/dict | Full journal info (ISSN, volume, issue, publisher, nlm_unique_id) |
| `nlm_unique_id` | string | NLM catalog ID |
| `year` | int | Publication year |
| `publication_date` | struct/dict | Full date structure (year, month, day) |
| `country` | string | Publication country |

#### Classification & Evidence (PubMed/PMC)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `mesh_terms` | list[string] | Flat MeSH descriptors |
| `mesh_terms_full` | list[struct] | Full MeSH with qualifiers: `[{descriptor, qualifiers[]}]` |
| `keywords` | list[string] | Article keywords (flat) |
| `keywords_full` | list[struct] | Full keyword structures: `[{keyword, major_yn}]` |
| `article_type` | string | Type of article |
| `publication_type` | list[string] | Flat publication categories |
| `publication_types_full` | list[struct] | Full pub types: `[{type, ui}]` |
| `evidence_grade` | string | A/B/C/D grade |
| `evidence_level` | int/null | 1-4 evidence level |
| `evidence_term` | string/null | Matched evidence term |
| `evidence_source` | string/null | Source of evidence classification |

#### DailyMed-Specific Fields
| Field | Current Type | Notes |
|-------|--------------|-------|
| `drug_name` | string | Medication name |
| `manufacturer` | string | Drug manufacturer |
| `active_ingredients` | list[string] | Active components |
| `label_type_code` | string | SPL document type code |
| `label_type_display` | string | Human-readable label type |
| `table_caption` | string | Table caption text |
| `table_id` | string | Table identifier |
| `table_type` | string | Type of table section |
| `has_tables` | boolean | Whether section contains tables |

#### Source Tracking (All Sources)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `source` | string | Source system (pubmed_abstract, pmc_oa, pmc_author_manuscript, dailymed) |
| `source_family` | string | Broad category (pubmed, pmc, dailymed) |
| `content_type` | string | Content classification |
| `has_full_text` | boolean | Full text availability |
| `is_open_access` | boolean/null | OA status |
| `is_author_manuscript` | boolean | NIHMS manuscript flag |
| `nihms_id` | string/null | NIHMS identifier |
| `license` | string | License type |

#### Abstract Structure (PubMed)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `abstract_structured` | list[struct] | Structured abstract sections: `[{label, text}]` |
| `has_structured_abstract` | boolean | Whether abstract is structured |

#### Affiliation (PubMed)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `is_gov_affiliated` | boolean | Government affiliation flag |
| `gov_agencies` | list[string] | Government agencies |

#### Table Metadata (All Sources)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `is_table` | boolean | Whether chunk is a table |
| `table_count` | int | Number of tables in section |

#### Ingestion Metadata (All Sources)
| Field | Current Type | Notes |
|-------|--------------|-------|
| `ingestion_timestamp` | float | Unix timestamp of ingestion |

### 1.2 Current Vector Configuration (Qdrant)

```python
# Dense vector configuration
DENSE_VECTOR_SIZE = 1024  # Qwen/Qwen3-Embedding-0.6B-batch
DISTANCE_METRIC = "Cosine"

# Sparse vector configuration (BM25-style) - TO BE REPLACED WITH FTS
SPARSE_ENABLED = True
SPARSE_MODE = "bm25"
SPARSE_MAX_TERMS_DOC = 256
SPARSE_MAX_TERMS_QUERY = 64
```

---

## 2. LanceDB Architecture & Recommendations

### 2.1 Storage Format

**Recommendation**: Use LanceDB with **Lance Format 2.2**:

```python
# Write with format 2.2 for best compression
lance.write_dataset(
    table,
    "./medical_db.lance",
    data_storage_version="2.2"
)
```

### 2.2 Vector Column Strategy

| Qdrant Approach | LanceDB Equivalent | Implementation |
|-----------------|-------------------|----------------|
| Dense vector (1024-dim) | `FixedSizeList[float]` column | Store as `pa.list_(pa.float32(), 1024)` |
| Sparse vector (BM25) | Full-Text Search (FTS) index | Create FTS index on `page_content` |

### 2.3 EXACT Arrow Schema (No Transformations)

**IMPORTANT**: Use native Arrow nested types to preserve exact structures. No JSON serialization.

```python
import pyarrow as pa

# Define nested structs exactly as they exist in current payloads
MESH_TERM_FULL_STRUCT = pa.struct([
    pa.field("descriptor", pa.string()),
    pa.field("qualifiers", pa.list_(pa.string()))
])

KEYWORD_FULL_STRUCT = pa.struct([
    pa.field("keyword", pa.string()),
    pa.field("major_yn", pa.string())  # 'Y' or 'N'
])

PUBLICATION_TYPE_FULL_STRUCT = pa.struct([
    pa.field("type", pa.string()),
    pa.field("ui", pa.string())
])

STRUCTURED_ABSTRACT_STRUCT = pa.struct([
    pa.field("label", pa.string()),
    pa.field("text", pa.string())
])

JOURNAL_FULL_STRUCT = pa.struct([
    pa.field("title", pa.string()),
    pa.field("abbreviation", pa.string()),
    pa.field("publisher", pa.string()),
    pa.field("issn_print", pa.string()),
    pa.field("issn_electronic", pa.string()),
    pa.field("nlm_unique_id", pa.string()),
    pa.field("volume", pa.string()),
    pa.field("issue", pa.string()),
])

PUBLICATION_DATE_STRUCT = pa.struct([
    pa.field("year", pa.int32()),
    pa.field("month", pa.int32()),
    pa.field("day", pa.int32()),
])

OTHER_IDS_STRUCT = pa.map_(pa.string(), pa.string())

MEDICAL_SCHEMA = pa.schema([
    # --- Vector Column (Dense) ---
    pa.field("vector", pa.list_(pa.float32(), 1024)),
    
    # --- Core Identifiers ---
    pa.field("doc_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("pmcid", pa.string()),
    pa.field("pmid", pa.string()),
    pa.field("doi", pa.string()),
    pa.field("pii", pa.string()),
    pa.field("other_ids", OTHER_IDS_STRUCT),
    pa.field("set_id", pa.string()),
    
    # --- Content Fields ---
    pa.field("page_content", pa.string()),  # FTS indexed
    pa.field("title", pa.string()),
    pa.field("abstract", pa.string()),
    pa.field("full_text", pa.string()),
    pa.field("text_preview", pa.string()),
    
    # --- Section & Chunking Metadata ---
    pa.field("chunk_index", pa.int32()),
    pa.field("total_chunks", pa.int32()),
    pa.field("token_count", pa.int32()),
    pa.field("section_type", pa.string()),
    pa.field("section_title", pa.string()),
    pa.field("section_id", pa.string()),
    pa.field("parent_section_id", pa.string()),
    pa.field("full_section_text", pa.string()),
    pa.field("section_weight", pa.float32()),
    
    # --- Publication Metadata ---
    pa.field("journal", pa.string()),
    pa.field("journal_full", JOURNAL_FULL_STRUCT),
    pa.field("nlm_unique_id", pa.string()),
    pa.field("year", pa.int32()),
    pa.field("country", pa.string()),
    pa.field("publication_date", PUBLICATION_DATE_STRUCT),
    
    # --- Classification (Exact Structures) ---
    pa.field("mesh_terms", pa.list_(pa.string())),
    pa.field("mesh_terms_full", pa.list_(MESH_TERM_FULL_STRUCT)),
    pa.field("keywords", pa.list_(pa.string())),
    pa.field("keywords_full", pa.list_(KEYWORD_FULL_STRUCT)),
    pa.field("article_type", pa.string()),
    pa.field("publication_type", pa.list_(pa.string())),
    pa.field("publication_types_full", pa.list_(PUBLICATION_TYPE_FULL_STRUCT)),
    
    # --- Evidence ---
    pa.field("evidence_grade", pa.string()),
    pa.field("evidence_level", pa.int32()),
    pa.field("evidence_term", pa.string()),
    pa.field("evidence_source", pa.string()),
    
    # --- DailyMed Specific ---
    pa.field("drug_name", pa.string()),
    pa.field("manufacturer", pa.string()),
    pa.field("active_ingredients", pa.list_(pa.string())),
    pa.field("label_type_code", pa.string()),
    pa.field("label_type_display", pa.string()),
    pa.field("table_caption", pa.string()),
    pa.field("table_id", pa.string()),
    pa.field("table_type", pa.string()),
    pa.field("has_tables", pa.bool_()),
    
    # --- Source Tracking ---
    pa.field("source", pa.string()),
    pa.field("source_family", pa.string()),
    pa.field("content_type", pa.string()),
    pa.field("has_full_text", pa.bool_()),
    pa.field("is_open_access", pa.bool_()),
    pa.field("is_author_manuscript", pa.bool_()),
    pa.field("nihms_id", pa.string()),
    pa.field("license", pa.string()),
    
    # --- Abstract Structure ---
    pa.field("has_structured_abstract", pa.bool_()),
    pa.field("abstract_structured", pa.list_(STRUCTURED_ABSTRACT_STRUCT)),
    
    # --- Affiliation ---
    pa.field("is_gov_affiliated", pa.bool_()),
    pa.field("gov_agencies", pa.list_(pa.string())),
    
    # --- Table Metadata ---
    pa.field("is_table", pa.bool_()),
    pa.field("table_count", pa.int32()),
    
    # --- Ingestion Metadata ---
    pa.field("ingestion_timestamp", pa.float64()),
])
```

---

## 3. Index Configuration Recommendations

### 3.1 Dense Vector Index: `IVF_RQ` (Safer Default for Filter-Heavy Workloads)

**Why IVF_RQ as default**: For filter-heavy retriever behavior, `IVF_RQ` (RaBitQ quantization) provides:
- Strong compression (~32:1) with good recall
- Better filter pushdown performance than HNSW variants
- More predictable latency with filters

```python
import lancedb
from lancedb.index import IvfRq

# Recommended default for filter-heavy workloads
table.create_index(
    column="vector",
    config=IvfRq(
        distance_type="cosine",
        num_partitions=num_rows // 4096,  # Standard formula
    )
)
```

**Benchmark IVF_HNSW_SQ if recall is critical**:
```python
from lancedb.index import HnswSq

# Use only if IVF_RQ recall is insufficient
# Higher memory usage but better recall/latency trade-off
table.create_index(
    column="vector",
    config=HnswSq(
        distance_type="cosine",
        num_partitions=max(1, num_rows // 1_048_576),  # Fewer partitions for HNSW
        m=20,
        ef_construction=300,
    )
)
```

### 3.2 Alternative: `IVF_PQ` (If Dimension Allows)

For 1024-dim vectors with `num_sub_vectors=64` (1024/16):

```python
from lancedb.index import IvfPq

table.create_index(
    column="vector",
    config=IvfPq(
        distance_type="cosine",
        num_partitions=num_rows // 4096,
        num_sub_vectors=64,  # 1024 / 16
        num_bits=8,
    )
)
```

### 3.3 Full-Text Search (FTS) Index (Replaces BM25 Sparse Vectors)

```python
# Create FTS index on page_content for lexical retrieval
table.create_fts_index(
    "page_content",
    use_tantivy=True,
    with_position=True,
)
```

### 3.4 Scalar Indexes (For Filtering)

```python
# BTREE indexes for range filters
table.create_scalar_index("year", index_type="BTREE")
table.create_scalar_index("section_weight", index_type="BTREE")
table.create_scalar_index("ingestion_timestamp", index_type="BTREE")

# BITMAP indexes for low-cardinality categorical filters
table.create_scalar_index("source", index_type="BITMAP")
table.create_scalar_index("source_family", index_type="BITMAP")
table.create_scalar_index("evidence_grade", index_type="BITMAP")
table.create_scalar_index("is_open_access", index_type="BITMAP")
table.create_scalar_index("is_author_manuscript", index_type="BITMAP")
table.create_scalar_index("is_gov_affiliated", index_type="BITMAP")
table.create_scalar_index("is_table", index_type="BITMAP")
table.create_scalar_index("has_tables", index_type="BITMAP")
table.create_scalar_index("has_full_text", index_type="BITMAP")

# LABEL_LIST indexes for array containment queries
table.create_scalar_index("mesh_terms", index_type="LABEL_LIST")
table.create_scalar_index("keywords", index_type="LABEL_LIST")
table.create_scalar_index("publication_type", index_type="LABEL_LIST")
table.create_scalar_index("active_ingredients", index_type="LABEL_LIST")
table.create_scalar_index("gov_agencies", index_type="LABEL_LIST")
```

---

## 4. Query Configuration

### 4.1 Dense Vector Search with Filters

```python
# Vector search with pre-filtering (uses scalar indexes)
results = table.search(query_vector) \
    .where("source_family = 'pubmed' AND year >= 2020") \
    .limit(10) \
    .to_pandas()
```

### 4.2 Full-Text Search (Replaces Sparse Vector Search)

```python
# Basic FTS (BM25 ranking)
results = table.search("hypertension treatment", query_type="fts") \
    .limit(10) \
    .to_pandas()

# FTS with filters
results = table.search("systematic review diabetes", query_type="fts") \
    .where("source_family = 'pubmed'") \
    .limit(10) \
    .to_pandas()
```

### 4.3 Hybrid Search (Dense + FTS with RRF)

```python
# Native hybrid search with RRF reranking
results = table.search(query_type="hybrid") \
    .vector(query_vector) \
    .text("hypertension treatment guidelines") \
    .limit(10) \
    .to_pandas()

# With custom RRF parameters
from lancedb.rerankers import RRFReranker

reranker = RRFReranker(k=60)
results = table.search(query_type="hybrid") \
    .vector(query_vector) \
    .text("treatment guidelines") \
    .rerank(reranker) \
    .limit(10) \
    .to_pandas()
```

---

## 5. Ingestion Configuration

### 5.1 Batch Size Recommendations

| Operation | Recommended Batch Size | Notes |
|-----------|------------------------|-------|
| Initial data add | 10,000 - 50,000 rows | Balance memory vs. speed |
| Incremental updates | 1,000 - 5,000 rows | Smaller batches for low latency |
| Index creation | N/A (full table) | Run after all data ingested |
| Compaction | - | Run `optimize()` periodically |

### 5.2 Post-Ingestion Optimization

```python
# After large ingestions, optimize table
await table.optimize(
    cleanup_older_than=timedelta(days=7)
)

# Verify index exists and is being used
print(table.list_indices())
```

---

## 6. Migration Path

### Phase 1: Schema & Index Setup
1. Create new LanceDB database with EXACT schema (no field transformations)
2. Create all scalar indexes (BTREE, BITMAP, LABEL_LIST)
3. Set up FTS index on `page_content`

### Phase 2: Data Migration
1. Export data from Qdrant (vectors + payloads)
2. Transform to Arrow RecordBatch (preserving exact field structures)
3. Import into LanceDB in batches
4. Create dense vector index (`IVF_RQ` first, benchmark before trying `IVF_HNSW_SQ`)

### Phase 3: Query Migration
1. Update retriever to use LanceDB API
2. Replace sparse vector queries with FTS
3. Implement hybrid search with RRF reranking
4. Update filters to use LanceDB `where()` syntax

### Phase 4: Validation & Optimization
1. Run `optimize()` to compact data
2. Verify index usage with `explain_plan()`
3. Benchmark recall@k against Qdrant baseline
4. Tune `nprobes` based on query patterns

---

## 7. Configuration Summary

### Recommended `.env` Updates

```bash
# LanceDB Configuration
LANCEDB_URI="./medical_data.lancedb"
LANCEDB_TABLE_NAME="medical_chunks"
LANCEDB_STORAGE_VERSION="2.2"

# Vector Index Configuration
# RECOMMENDED: Start with IVF_RQ for filter-heavy workloads
VECTOR_INDEX_TYPE="IVF_RQ"
# Alternative: IVF_HNSW_SQ (benchmark if recall insufficient)
# VECTOR_INDEX_TYPE="IVF_HNSW_SQ"
VECTOR_DISTANCE_METRIC="cosine"

# HNSW params (only if using IVF_HNSW_SQ)
HNSW_M=20
HNSW_EF_CONSTRUCTION=300

# FTS Configuration
FTS_ENABLED=true
FTS_TOKENIZER="default"

# Scalar Index Configuration
SCALAR_INDEX_BTREE="year,section_weight,ingestion_timestamp"
SCALAR_INDEX_BITMAP="source,source_family,evidence_grade,is_open_access,is_author_manuscript,is_gov_affiliated,is_table,has_tables,has_full_text"
SCALAR_INDEX_LABELLIST="mesh_terms,keywords,publication_type,active_ingredients,gov_agencies"

# Ingestion Batch Configuration
LANCEDB_BATCH_SIZE=10000
LANCEDB_MAX_WORKERS=8
```

---

## 8. API Version Warning

**⚠️ CRITICAL**: LanceDB Python API is evolving. Validate all code against your installed version:

```bash
pip show lancedb
```

Key version-sensitive areas:
1. **Index configuration classes** (`IvfRq`, `IvfPq`, `HnswSq`) - API may vary
2. **FTS index creation** - `create_fts_index()` parameters vary by version
3. **Hybrid search** - `query_type="hybrid"` syntax is relatively new
4. **Scalar indexes** - `create_scalar_index()` API may differ

**Always test with a small subset before full migration**.

---

## 9. Key Differences from Qdrant

| Feature | Qdrant | LanceDB |
|---------|--------|---------|
| Sparse vectors | Native sparse vector type | FTS (BM25) index on text column |
| Distance metrics | Cosine, Euclid, Dot | l2, cosine, dot, hamming |
| Default index | HNSW | Manual (IVF_RQ/PQ/HNSW_SQ) |
| Storage | Proprietary | Open Lance format |
| Hybrid search | Manual merge | Native RRF reranking |
| Filtering | Pre/post filter | Predicate pushdown with scalar indexes |
| Nested fields | JSON objects | Native Arrow struct/list types |

---

## 10. References

1. [LanceDB Vector Index Documentation](https://docs.lancedb.com/indexing/vector-index)
2. [LanceDB Full-Text Search](https://docs.lancedb.com/search/full-text-search)
3. [LanceDB Hybrid Search](https://docs.lancedb.com/search/hybrid-search)
4. [Lance Format 2.2 Blog Post](https://lancedb.com/blog/lance-file-format-2-2-taming-complex-data/)
5. [AWS Architecture Blog: 1B+ Vectors with LanceDB](https://aws.amazon.com/blogs/architecture/a-scalable-elastic-database-and-search-solution-for-1b-vectors-built-on-lancedb-and-amazon-s3/)
