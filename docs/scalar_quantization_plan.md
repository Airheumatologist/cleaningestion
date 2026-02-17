# Scalar Quantization Plan (Post-Ingestion)

> [!IMPORTANT]
> Apply AFTER all three ingestion pipelines (DailyMed, PMC, PubMed) have completed.

## Why Scalar Quantization?

| Metric | Without Quantization | With Scalar (int8) |
|--------|---------------------|--------------------|
| Vector storage per point | 1024 × 4 bytes = **4 KB** | 1024 × 1 byte = **1 KB** |
| Estimated total (7.5M pts) | **~30 GB** | **~7.5 GB** |
| Search speed | Baseline | **~2× faster** |
| Accuracy loss | — | **< 1%** (negligible) |

Server has ~128 GB RAM — not critical now, but scalar quantization is free performance.

## Steps

### 1. Verify All Ingestion is Complete

```bash
# Confirm no ingestion processes are running
ps aux | grep -E "(06_ingest_pmc|07_ingest_dailymed|21_ingest_pubmed)" | grep -v grep

# Verify final point count
cd /opt/RAG-pipeline && source .env && source venv/bin/activate && python3 -c "
from qdrant_client import QdrantClient
from scripts.config_ingestion import IngestionConfig
c = QdrantClient(url=IngestionConfig.QDRANT_URL, api_key=IngestionConfig.QDRANT_API_KEY or None)
info = c.get_collection(IngestionConfig.COLLECTION_NAME)
print(f'Total points: {info.points_count}')
print(f'Status: {info.status}')
"
```

### 2. Enable Scalar Quantization

```python
# Run this script on the server:
# /opt/RAG-pipeline/scripts/enable_quantization.py

from qdrant_client import QdrantClient
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType, QuantizationSearchParams
from scripts.config_ingestion import IngestionConfig

client = QdrantClient(
    url=IngestionConfig.QDRANT_URL,
    api_key=IngestionConfig.QDRANT_API_KEY or None,
    timeout=600,
)

# Enable scalar quantization (int8)
client.update_collection(
    collection_name=IngestionConfig.COLLECTION_NAME,
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,        # Clip outliers at 1st/99th percentile
            always_ram=True,      # Keep quantized vectors in RAM for speed
        ),
    ),
)

print("✅ Scalar quantization enabled. Qdrant will re-index in background.")
print("   Monitor with: curl localhost:6333/collections/rag_pipeline")
```

### 3. Monitor Re-indexing

Qdrant re-indexes asynchronously after enabling quantization:

```bash
# Check collection status (should go from "yellow" -> "green")
watch -n 10 'curl -s localhost:6333/collections/rag_pipeline | python3 -m json.tool | grep -E "status|points_count|indexed_vectors_count"'
```

Re-indexing ~7.5M vectors takes **~10-30 minutes** depending on server load.

### 4. Verify Search Still Works

```bash
cd /opt/RAG-pipeline && source .env && source venv/bin/activate && python3 -c "
from qdrant_client import QdrantClient
from scripts.config_ingestion import IngestionConfig

c = QdrantClient(url=IngestionConfig.QDRANT_URL, api_key=IngestionConfig.QDRANT_API_KEY or None)
info = c.get_collection(IngestionConfig.COLLECTION_NAME)
print(f'Status: {info.status}')
print(f'Points: {info.points_count}')
print(f'Quantization: {info.config.quantization_config}')
"
```

### 5. Optional: Tune Search Parameters

For best accuracy with quantization, use `rescore=True` (default) in search queries:

```python
# In retriever code, search params should include:
search_params=SearchParams(
    quantization=QuantizationSearchParams(
        rescore=True,      # Re-score top candidates with full vectors
        oversampling=1.5,  # Fetch 1.5x candidates before rescoring
    )
)
```

## Rollback

If needed, disable quantization:

```python
client.update_collection(
    collection_name="rag_pipeline",
    quantization_config=None,  # Removes quantization
)
```
