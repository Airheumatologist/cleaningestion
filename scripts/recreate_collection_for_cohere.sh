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

echo "Step 3: Resetting Data Directories (Full Fresh Start)..."
# Wipe data directories to ensure a clean slate as requested
rm -rf /data/ingestion/pmc_xml
rm -rf /data/ingestion/dailymed
rm -rf /data/ingestion/pubmed_baseline
rm -f /data/ingestion/*.jsonl
rm -f /data/ingestion/*_ingested_ids.txt
echo "Data directories cleared."

echo ""
echo "Step 4: Running Complete Ingestion Pipeline (Phase 1: Test with 2 files)..."
# This script handles downloading PMC, DailyMed, PubMed and ingesting them
# We limit to 2 files (~15GB extracted) to confirm end-to-end success before full download
export PMC_MAX_FILES=2
chmod +x scripts/run_complete_ingestion.sh
./scripts/run_complete_ingestion.sh

echo ""
echo "=========================================="
echo "  Full Reset & Ingestion Complete"
echo "=========================================="

echo ""
echo "=========================================="
echo "  Final Collection Stats"
echo "=========================================="
curl -s -H "api-key: ${QDRANT_API_KEY}" \
  http://localhost:6333/collections/rag_pipeline | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
print(f\"Points:  {d.get('points_count', 'N/A')}\")
print(f\"Vectors: {d.get('vectors_count', 'N/A')}\")
print(f\"Status:  {d.get('status', 'N/A')}\")
print(f\"Vector size: {d['config']['params']['vectors']['size']}\")
"

echo ""
echo "✅ Collection recreated and data ingested!"
