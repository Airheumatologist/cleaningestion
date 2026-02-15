#!/bin/bash
# Test ingestion pipeline with small batch (run on Hetzner server)
# Copy this to Hetzner and run: bash run_test_ingestion.sh

set -e  # Exit on error

echo "=========================================="
echo "  Test Data Ingestion Pipeline"
echo "  Server: $(hostname)"
echo "  Date: $(date)"
echo "=========================================="

# Setup
cd /opt/RAG-pipeline
source venv/bin/activate

# Ensure .env is up to date
export QDRANT_URL=http://localhost:6333
export QDRANT_GRPC_URL=localhost:6334
export QDRANT_API_KEY=QDRANT_API_KEY_REDACTED
export COHERE_API_KEY=COHERE_API_KEY_REDACTED
export QDRANT_COLLECTION=rag_pipeline
export COLLECTION_NAME=rag_pipeline
export EMBEDDING_PROVIDER=cohere
export EMBEDDING_MODEL=embed-v4.0
export DATA_DIR=/data/ingestion
export PMC_XML_DIR=/data/ingestion/pmc_xml
export DAILYMED_XML_DIR=/data/ingestion/dailymed/xml

echo ""
echo "Step 0: Check server resources..."
echo "------------------------------------------"
df -h | grep -E '(Filesystem|/dev/)'
echo ""
free -h
echo ""

echo "Step 1: Create data directories..."
echo "------------------------------------------"
mkdir -p /data/ingestion/pmc_xml
mkdir -p /data/ingestion/dailymed/xml
mkdir -p /data/backups
echo "✓ Data directories ready"
echo ""

echo "Step 22: Download PMC articles (test batch: 5 files)..."
echo "------------------------------------------"
python scripts/01_download_pmc.py --max-files 5
echo ""

echo "Step 23: Extract PMC XML → JSONL..."
echo "------------------------------------------"
python scripts/02_extract_pmc.py
echo ""

echo "Step 24: Ingest PMC into Qdrant..."
echo "------------------------------------------"
python scripts/06_ingest_pmc.py --xml-dir /data/ingestion/pmc_xml
echo ""

echo "=========================================="
echo "  Test Ingestion Complete!"
echo "=========================================="
echo ""
echo "Checking collection stats..."
curl -s -H "api-key: ${QDRANT_API_KEY}" \
  http://localhost:6333/collections/rag_pipeline | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
print(f\"Points:  {d['points_count']}\")
print(f\"Vectors: {d['vectors_count']}\")
print(f\"Status:  {d['status']}\")
"
echo ""
echo "Disk usage after ingestion:"
df -h | grep -E '(Filesystem|/dev/)'

echo ""
echo "=========================================="
echo "  Next Steps (after PMC test succeeds):"
echo "=========================================="
echo "1. Run full PMC download (remove --max-files):"
echo "   python scripts/01_download_pmc.py"
echo ""
echo "2. Download DailyMed (skipping for test):"
echo "   python scripts/03_download_dailymed.py"
echo "   python scripts/04_process_dailymed.py"
echo "   python scripts/07_ingest_dailymed.py --xml-dir /data/ingestion/dailymed/xml"
