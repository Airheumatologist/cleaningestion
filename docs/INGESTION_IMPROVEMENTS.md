# Ingestion Improvements Summary

This document summarizes the improvements made to the PMC and DailyMed ingestion pipelines.

## Problem with Original Ingestion

### PMC (Original)
- ✅ Extracted sections and tables properly
- ❌ Tables only stored as count, not embedded
- ❌ All sections squashed into chunks without context
- ❌ No Cohere API support

### DailyMed (Original)
- ❌ Tables completely ignored
- ❌ One embedding per drug (all sections combined)
- ❌ Content truncated for embedding (2000 chars)
- ❌ No Cohere API support

## Improved Ingestion

### PMC Improved (`06_ingest_pmc.py`)

| Feature | Original | Improved |
|---------|----------|----------|
| **Abstract** | Chunks only | Dedicated abstract chunk with title context |
| **Sections** | Plain chunks | Context-rich: "Title: X\n\nSection: Y\n\nContent" |
| **Tables** | Count only | **Separate table chunks with full content** |
| **Long sections** | Single chunk | **Split into 5000-char parts** |
| **Cohere API** | ❌ No | ✅ Full support |
| **Upsert reliability** | `wait=False` | `wait=True` |

**Chunking Strategy:**
```
Article: PMC123456
├── Abstract chunk (title + abstract)
├── Introduction chunk(s)
├── Methods chunk(s)
├── Results chunk(s)
├── Discussion chunk(s)
├── Table 1 chunk (caption + row-by-row data)
├── Table 2 chunk (caption + row-by-row data)
└── ...
```

### DailyMed Improved (`07_ingest_dailymed.py`)

| Feature | Original | Improved |
|---------|----------|----------|
| **Chunks per drug** | 1 | **Multiple (overview + sections + tables)** |
| **Table extraction** | ❌ Ignored | ✅ **Parsed and embedded** |
| **Section search** | ❌ Combined | ✅ **Searchable separately** |
| **Cohere API** | ❌ Local only | ✅ Full support |
| **Upsert reliability** | `wait=False` | `wait=True` |

**Chunking Strategy:**
```
Drug: Ibuprofen
├── Overview chunk (name + ingredients + key sections)
├── Indications chunk
├── Dosage chunk
├── Adverse Reactions chunk
├── Drug Interactions chunk
├── Warnings chunk
├── Description chunk
├── Table 1: Dosage Schedule chunk
├── Table 2: Contraindications chunk
```

## Benefits of Improved Ingestion

1. **Better Search Relevance**
   - Tables are searchable (e.g., "dosage table for ibuprofen")
   - Sections can be filtered (e.g., only search "Methods" sections)

2. **More Context in Embeddings**
   - Each chunk includes article/drug title for context
   - Table captions preserved for semantic understanding

3. **No Data Loss**
   - Tables fully embedded, not just counted
   - Long sections split without truncation

4. **Reliability**
   - Cohere API support for production use
   - `wait=True` ensures data is persisted

## Usage

### PMC Improved
```bash
export PYTHONPATH=/opt/RAG-pipeline:$PYTHONPATH
export EMBEDDING_PROVIDER=cohere
export COHERE_API_KEY=your_key

python scripts/06_ingest_pmc.py \
    --xml-dir /data/ingestion/pmc_xml \
    --articles-file /data/ingestion/pmc_articles.jsonl
```

### DailyMed Improved
```bash
export PYTHONPATH=/opt/RAG-pipeline:$PYTHONPATH
export EMBEDDING_PROVIDER=cohere
export COHERE_API_KEY=your_key

python scripts/07_ingest_dailymed.py \
    --xml-dir /data/ingestion/dailymed/xml
```

## Complete Pipeline

Use `run_complete_ingestion.sh` for full pipeline:
```bash
screen -S ingestion -dm ./scripts/run_complete_ingestion.sh
```

This runs:
1. PMC Baseline Download
2. PMC Extraction
3. PMC Ingestion (improved)
4. DailyMed Download
5. DailyMed Processing
6. DailyMed Ingestion (improved)
7. PubMed Download
8. PubMed Ingestion

## Expected Point Counts

| Dataset | Articles/Drugs | Chunks per Item | Total Points |
|---------|----------------|-----------------|--------------|
| PMC Full | ~5M | ~10-15 | **~50-75M** |
| DailyMed | ~100K | ~8-12 | **~800K-1.2M** |
| PubMed | ~35M (filtered ~2M) | ~2-3 | **~4-6M** |
| **Total** | | | **~55-82M** |

## Storage Requirements

With 1536-dimensional float32 vectors:
- Per point: ~6KB (vector) + ~2KB (payload) = ~8KB
- 50M points: ~400GB
- 80M points: ~640GB

Plus original data:
- PMC XML: ~5TB
- DailyMed: ~5GB
- PubMed: ~35GB

**Total disk needed: ~6TB** (you have 1.5TB free - may need to delete XML after ingestion)

