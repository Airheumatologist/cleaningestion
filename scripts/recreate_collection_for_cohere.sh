#!/bin/bash
# Recreate collection with correct dimensions for Cohere (1536)
# Run this on Hetzner server

set -e

echo "=========================================="
echo "  Recreate Collection for Cohere"
echo "  Vector size: 1536"
echo "=========================================="

cd /opt/RAG-pipeline
source venv/bin/activate

export PYTHONPATH=/opt/RAG-pipeline:$PYTHONPATH
export QDRANT_URL=http://localhost:6333
export QDRANT_API_KEY=QDRANT_API_KEY_REDACTED
export COHERE_API_KEY=COHERE_API_KEY_REDACTED
export QDRANT_COLLECTION=rag_pipeline
export COLLECTION_NAME=rag_pipeline
export EMBEDDING_PROVIDER=cohere
export EMBEDDING_MODEL=embed-v4.0

echo ""
echo "Step 1: Delete existing collection..."
curl -s -X DELETE -H "api-key: ${QDRANT_API_KEY}" \
  http://localhost:6333/collections/rag_pipeline
echo "Collection deleted (or didn't exist)"

echo ""
echo "Step 2: Create collection with 1536 dimensions (for Cohere)..."
python scripts/05_setup_qdrant.py --collection-name rag_pipeline
echo ""

echo "Step 3: Verify collection configuration..."
curl -s -H "api-key: ${QDRANT_API_KEY}" \
  http://localhost:6333/collections/rag_pipeline | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
print(f\"Vector size: {d['config']['params']['vectors']['size']}\")
print(f\"Distance: {d['config']['params']['vectors']['distance']}\")
print(f\"Sparse vectors: {'sparse' in d['config']['params'].get('sparse_vectors', {})}\")
"

echo ""
echo "Step 4: Clear ingestion checkpoint..."
rm -f /data/ingestion/pmc_ingested_ids.txt
echo "Checkpoint cleared"

echo ""
echo "Step 5: Re-ingest PMC data with Cohere embeddings..."
python scripts/06_ingest_pmc.py --xml-dir /data/ingestion/pmc_xml

echo ""
echo "Step 6: Ingest DailyMed data..."
# DailyMed script uses config for paths, defaults to /data/ingestion/dailymed/xml
python scripts/07_ingest_dailymed.py

echo ""
echo "Step 7: Ingest PubMed Abstracts..."
# PubMed script defaults to /data/pubmed_baseline/filtered/pubmed_abstracts.jsonl
if [ -f "/data/pubmed_baseline/filtered/pubmed_abstracts.jsonl" ]; then
    python scripts/21_ingest_pubmed_abstracts.py --input /data/pubmed_baseline/filtered/pubmed_abstracts.jsonl
else
    echo "PubMed filtered file not found at /data/pubmed_baseline/filtered/pubmed_abstracts.jsonl - skipping"
fi

echo ""
echo "=========================================="
echo "  Final Collection Stats"
echo "=========================================="
curl -s -H "api-key: ${QDRANT_API_KEY}" \
  http://localhost:6333/collections/rag_pipeline | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
print(f\"Points:  {d['points_count']}\")
print(f\"Vectors: {d['vectors_count']}\")
print(f\"Status:  {d['status']}\")
print(f\"Vector size: {d['config']['params']['vectors']['size']}\")
"

echo ""
echo "✅ Collection recreated and data ingested!"
