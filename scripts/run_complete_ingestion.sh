#!/bin/bash
# Complete Ingestion Pipeline: PMC + DailyMed (Improved) + PubMed
# Run this on Hetzner server

set -e

echo "=========================================="
echo "  COMPLETE DATA INGESTION PIPELINE"
echo "  Datasets: PMC + DailyMed + PubMed"
echo "  Embedding: Cohere (1536 dims)"
echo "  Start: $(date)"
echo "=========================================="

cd /opt/RAG-pipeline
source venv/bin/activate

export PYTHONPATH=/opt/RAG-pipeline:$PYTHONPATH
export QDRANT_URL=http://localhost:6333
export QDRANT_GRPC_URL=localhost:6334
export QDRANT_API_KEY=${QDRANT_API_KEY:-QDRANT_API_KEY_REDACTED}
export COHERE_API_KEY=${COHERE_API_KEY:-COHERE_API_KEY_REDACTED}
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
LOG_FILE="/var/log/rag-ingestion/complete_ingestion_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE")
exec 2>&1

log_step() {
    echo ""
    echo "=========================================="
    echo "  STEP $1: $2"
    echo "  Started at: $(date)"
    echo "=========================================="
}

# Function to check and report collection stats
check_stats() {
    echo ""
    echo "=== Collection Stats ==="
    curl -s -H "api-key: ${QDRANT_API_KEY}" \
      http://localhost:6333/collections/rag_pipeline | python3 -c "
import json, sys
d = json.load(sys.stdin)['result']
print(f'Points:  {d[\"points_count\"]:,}')
print(f'Status:  {d[\"status\"]}')
print(f'Vector:  {d[\"config\"][\"params\"][\"vectors\"][\"size\"]} dims')
"
}

# ============================================
# PHASE 1: PMC FULL DATASET
# ============================================

log_step "1A" "PMC Baseline Download"
echo "Downloading PMC baseline files (several TB)..."
# Use the modular download script
# If PMC_MAX_FILES is set (e.g. to 2), it will limit the download for testing/bootstrapping.
echo "Running download script (Max files: ${PMC_MAX_FILES:-"ALL"})..."

if [ -n "$PMC_MAX_FILES" ]; then
    python scripts/01_download_pmc.py --max-files "$PMC_MAX_FILES"
else
    python scripts/01_download_pmc.py
fi

log_step "1B" "PMC Extraction"
python scripts/02_extract_pmc.py \
    --xml-dir /data/ingestion/pmc_xml \
    --output /data/ingestion/pmc_articles.jsonl

log_step "1C" "PMC Ingestion"
rm -f /data/ingestion/pmc_ingested_ids.txt
python scripts/06_ingest_pmc.py --xml-dir /data/ingestion/pmc_xml

check_stats

# ============================================
# PHASE 2: DAILYMED (IMPROVED)
# ============================================

log_step "2A" "DailyMed Download"
python scripts/03_download_dailymed.py

log_step "2B" "DailyMed Processing"
python scripts/04_process_dailymed.py

log_step "2C" "DailyMed Ingestion (Improved - Multi-chunk)"
rm -f /data/ingestion/dailymed_ingested_ids.txt
python scripts/07_ingest_dailymed.py --xml-dir /data/ingestion/dailymed/xml

check_stats

# ============================================
# PHASE 3: PUBMED BASELINE
# ============================================

log_step "3A" "PubMed Baseline Download"
python scripts/20_download_pubmed_baseline.py \
    --output-dir /data/ingestion/pubmed_baseline \
    --min-year 2015

log_step "3B" "PubMed Ingestion"
rm -f /data/ingestion/pubmed_ingested_ids.txt
python scripts/21_ingest_pubmed_abstracts.py \
    --input /data/ingestion/pubmed_baseline/filtered/pubmed_abstracts.jsonl

check_stats

# ============================================
# FINAL SUMMARY
# ============================================

echo ""
echo "=========================================="
echo "  INGESTION COMPLETE!"
echo "=========================================="
echo ""

check_stats

echo ""
echo "=== Data Distribution ==="
curl -s -X POST -H "api-key: ${QDRANT_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"limit": 10000, "with_payload": true, "with_vector": false}' \
  http://localhost:6333/collections/rag_pipeline/points/scroll | python3 -c "
import json, sys
data = json.load(sys.stdin)
points = data.get('result', {}).get('points', [])

sources = {}
article_types = {}
for p in points:
    src = p.get('payload', {}).get('source', 'unknown')
    atype = p.get('payload', {}).get('article_type', 'unknown')
    sources[src] = sources.get(src, 0) + 1
    article_types[atype] = article_types.get(atype, 0) + 1

print('By Source:')
for src, count in sorted(sources.items(), key=lambda x: -x[1]):
    print(f'  {src}: {count:,}')

print('\nBy Article Type:')
for atype, count in sorted(article_types.items(), key=lambda x: -x[1])[:10]:
    print(f'  {atype}: {count:,}')
"

echo ""
echo "Disk usage:"
df -h | grep -E '(Filesystem|/dev/)'

echo ""
echo "Data directory sizes:"
du -sh /data/ingestion/* 2>/dev/null | sort -h

echo ""
echo "=========================================="
echo "  Log file: $LOG_FILE"
echo "  Completed at: $(date)"
echo "=========================================="
