# LanceDB Optimal Ingestion Configuration Plan

## Objective
Migrate from Qdrant to LanceDB while preserving the exact schema used for PubMed, DailyMed, and PMC ingestion. The configuration is optimized for fast dense (IVF_PQ) and sparse (Full Text Search - FTS) retrieval at scale.

## 1. Schema Definition (Preserving Existing Qdrant Fields)
LanceDB uses PyArrow for schema definition. To ensure no data loss and exact compatibility with the existing system, the schema should be defined explicitly before creating the table.

```python
import pyarrow as pa

schema = pa.schema([
    # Core vector and text
    pa.field("vector", pa.list_(pa.float32(), 1024)), # Qwen3-Embedding-0.6B dimension
    pa.field("text", pa.string()), # The actual chunk text for FTS
    
    # Metadata Fields
    pa.field("year", pa.int32()),
    pa.field("source", pa.string()),
    pa.field("source_family", pa.string()),
    pa.field("article_type", pa.string()),
    pa.field("journal", pa.string()),
    pa.field("evidence_grade", pa.string()),
    pa.field("evidence_level", pa.int32()),
    pa.field("evidence_term", pa.string()),
    pa.field("evidence_source", pa.string()),
    pa.field("country", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("chunk_id", pa.string()),
    pa.field("section_type", pa.string()),
    pa.field("pmcid", pa.string()),
    pa.field("pmid", pa.string()),
    pa.field("set_id", pa.string()),
    pa.field("drug_name", pa.string()),
    
    # Booleans and Lists
    pa.field("is_gov_affiliated", pa.bool_()),
    pa.field("is_author_manuscript", pa.bool_()),
    pa.field("gov_agencies", pa.list_(pa.string())),
])
```

## 2. Optimal Indexing Configuration

Once data is ingested into the LanceDB table, creating the right indexes is critical for fast retrieval. Since the ingestion process adds large batches of data (e.g., weekly updates), indexes should ideally be built or updated *after* the bulk insertion.

### A. Dense Retrieval (Semantic Search)
LanceDB uses an IVF_PQ (Inverted File Product Quantization) index for fast approximate nearest neighbor (ANN) search on disk.

**Optimal `create_index` Parameters:**
- `metric="cosine"`: Matches the existing Qdrant setup for Qwen3 embeddings.
- `num_partitions`: Determines the number of IVF clusters.
  - **Guidance:** `Total Rows // 4096`
- `num_sub_vectors`: Controls the compression (PQ).
  - **Guidance:** `Dimension // 16`. For Qwen3 (1024-d), use `num_sub_vectors=64`. LanceDB optimizes for sub-vector dimensionalities of 8 or 16 for SIMD acceleration.
- `accelerator="cuda"`: If the ingestion/build server has a GPU, this dramatically speeds up the k-means clustering.

```python
table.create_index(
    metric="cosine",
    num_partitions=len(table) // 4096 or 1,
    num_sub_vectors=64, # 1024 / 16
    accelerator="cuda"  # Optional, but highly recommended for build speed
)
```

**Query Time:**
- `nprobes`: Number of partitions to search. Default is usually sufficient, but increasing it (e.g., to cover 5% of partitions) boosts recall at a slight cost to latency.

### B. Sparse Retrieval (Keyword / FTS / Tantivy)
LanceDB uses Tantivy (Rust) for Full Text Search (FTS). It creates an inverted index on the target text columns.

**Optimal `create_fts_index` Parameters:**
- `field_names=["text"]`: Instructs LanceDB to index the primary text column.
- `with_position=True`: Essential for exact phrase matching (e.g., "myocardial infarction").
- `replace=True`: To ensure updates rebuild the index cleanly.
- `lower_case=True`, `stem=True`, `remove_stop_words=True` to normalize clinical terms.

```python
table.create_fts_index(
    field_names=["text"],
    with_position=True,
    replace=True
)
```

**Note:** For BM25-like behavior in queries, use `table.search("query", query_type="hybrid")`, which requires both the IVF_PQ and FTS indexes to exist.

### C. Scalar Indexes (Metadata Filtering)
Qdrant automatically filters on keyword payloads. In LanceDB, explicit scalar indexes on frequently filtered fields will massively speed up pre-filtering during ANN search.

**Suggestions to Index:**
```python
# BTree indexes for fast exact match and range filtering
table.create_scalar_index("year")
table.create_scalar_index("source")
table.create_scalar_index("article_type")
table.create_scalar_index("pmcid")
table.create_scalar_index("set_id")
table.create_scalar_index("drug_name")
table.create_scalar_index("is_gov_affiliated")
```

## 3. Ingestion Strategy & Suggestions
- **Batching:** Insert data in batches (e.g., 50,000 rows). PyArrow tables or lists of dictionaries can be directly appended using `table.add(data)`.
- **Compaction:** LanceDB creates new fragments for every `add()` operation. Run `table.compact_files()` periodically (e.g., end of the weekly update) to merge small fragments and maintain optimal read speeds.
- **Index Rebuilding:** After a large `add()` or `compact_files()` operation, you must update the indexes for the new data to be searchable efficiently (or wait for automatic background indexing if supported in your version). Wait until the weekly batch finishes to call `table.create_index()` again.
