#!/bin/bash
# Full PMC + DailyMed Ingestion Pipeline
# Run this on Hetzner server

set -e

echo "=========================================="
echo "  FULL DATA INGESTION PIPELINE"
echo "  Embedding: Cohere (1536 dims)"
echo "  Start: $(date)"
echo "=========================================="

cd /opt/RAG-pipeline
source venv/bin/activate

export PYTHONPATH=/opt/RAG-pipeline:$PYTHONPATH
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
export BATCH_SIZE=100
export EMBEDDING_BATCH_SIZE=96
export MAX_WORKERS=8

# Create log directory
mkdir -p /var/log/rag-ingestion
LOG_FILE="/var/log/rag-ingestion/full_ingestion_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo ""
echo "Step 1: Full PMC Download (this may take hours)..."
echo "Started at: $(date)"
python scripts/01_download_pmc.py 2>&1 | tee -a "$LOG_FILE"
echo "PMC Download completed at: $(date)"

echo ""
echo "Step 2: Extract PMC XML → JSONL..."
echo "Started at: $(date)"
python scripts/02_extract_pmc.py \
    --xml-dir /data/ingestion/pmc_xml \
    --output /data/ingestion/pmc_articles.jsonl 2>&1 | tee -a "$LOG_FILE"
echo "Extraction completed at: $(date)"

echo ""
echo "Step 3: Clear checkpoint for fresh ingestion..."
rm -f /data/ingestion/pmc_ingested_ids.txt
echo "Checkpoint cleared"

echo ""
echo "Step 4: Ingest PMC into Qdrant..."
echo "Started at: $(date)"
python scripts/06_ingest_pmc.py --xml-dir /data/ingestion/pmc_xml 2>&1 | tee -a "$LOG_FILE"
echo "PMC Ingestion completed at: $(date)"

echo ""
echo "Step 5: Download DailyMed..."
echo "Started at: $(date)"
python scripts/03_download_dailymed.py 2>&1 | tee -a "$LOG_FILE"
echo "DailyMed download completed at: $(date)"

echo ""
echo "Step 6: Process DailyMed..."
echo "Started at: $(date)"
python scripts/04_process_dailymed.py 2>&1 | tee -a "$LOG_FILE"
echo "DailyMed processing completed at: $(date)"

echo ""
echo "Step 7: Ingest DailyMed into Qdrant..."
echo "Started at: $(date)"
python scripts/07_ingest_dailymed.py --xml-dir /data/ingestion/dailymed/xml 2>&1 | tee -a "$LOG_FILE"
echo "DailyMed ingestion completed at: $(date)"

echo ""
echo "=========================================="
echo "  FINAL STATS"
echo "=========================================="
curl -s -H "api-key: ${QDRANT_API_KEY}" \
  http://localhost:6333/collections/rag_pipeline | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
print(f'Points:  {d[\"points_count\"]}')
print(f'Status:  {d[\"status\"]}')
print(f'Vector size: {d[\"config\"][\"params\"][\"vectors\"][\"size\"]}')
"

echo ""
echo "Disk usage:"
df -h | grep -E '(Filesystem|/dev/)'

echo ""
echo "=========================================="
echo "  FULL INGESTION COMPLETE!"
echo "  Log file: $LOG_FILE"
echo "  End: $(date)"
echo "=========================================="
