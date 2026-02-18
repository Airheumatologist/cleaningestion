#!/bin/bash
# Start PubMed and DailyMed ingestion while PMC continues running

set -e

echo "=========================================="
echo "🚀 Starting PubMed & DailyMed Ingestion"
echo "=========================================="
echo ""

# Configuration
DATA_DIR="${DATA_DIR:-/data/ingestion}"
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"

# Activate virtual environment
source .venv/bin/activate || source venv/bin/activate

echo "🔧 Configuration:"
echo "   Embedding: ${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B-batch}"
echo "   Provider: ${EMBEDDING_PROVIDER:-deepinfra}"
echo ""

# Check if PMC is still running
if pgrep -f "06_ingest_pmc.py" > /dev/null; then
    echo "✅ PMC ingestion is still running (will continue with new code)"
    echo ""
fi

# ==========================================
# PUBMED INGESTION
# ==========================================
echo "📄 PubMed Abstracts Ingestion"
echo "----------------------------------------"

# Find PubMed file
PUBMED_FILE="${PUBMED_FILE:-$DATA_DIR/pubmed_baseline/pubmed_abstracts.jsonl}"

if [ ! -f "$PUBMED_FILE" ]; then
    # Try to find any pubmed jsonl file
    PUBMED_FILE=$(find $DATA_DIR -name "*pubmed*.jsonl" -type f 2>/dev/null | head -1)
fi

if [ -z "$PUBMED_FILE" ] || [ ! -f "$PUBMED_FILE" ]; then
    echo "⚠️  PubMed file not found at $PUBMED_FILE"
    read -p "Enter PubMed JSONL file path: " PUBMED_FILE
    if [ ! -f "$PUBMED_FILE" ]; then
        echo "❌ File not found, skipping PubMed"
        SKIP_PUBMED=1
    fi
else
    echo "✅ Found PubMed file: $PUBMED_FILE"
fi

if [ -z "$SKIP_PUBMED" ]; then
    # Check checkpoint
    CHECKPOINT_FILE="$DATA_DIR/pubmed_ingested_ids.txt"
    if [ -f "$CHECKPOINT_FILE" ]; then
        INGESTED_COUNT=$(wc -l < "$CHECKPOINT_FILE")
        echo "   Checkpoint: $INGESTED_COUNT articles already ingested"
    fi
    
    read -p "Start PubMed ingestion? [Y/n]: " confirm
    if [[ ! $confirm =~ [nN] ]]; then
        LOG_FILE="$LOG_DIR/pubmed_$(date +%Y%m%d_%H%M%S).log"
        echo "   Starting PubMed ingestion..."
        echo "   Log: $LOG_FILE"
        
        nohup python3 scripts/21_ingest_pubmed_abstracts.py \
            --input "$PUBMED_FILE" \
            > "$LOG_FILE" 2>&1 &
        
        PUBMED_PID=$!
        echo $PUBMED_PID > "$LOG_DIR/pubmed.pid"
        echo "   ✅ Started (PID: $PUBMED_PID)"
    else
        echo "   Skipped PubMed"
    fi
fi

echo ""

# ==========================================
# DAILYMED INGESTION
# ==========================================
echo "💊 DailyMed Drug Labels Ingestion"
echo "----------------------------------------"

DAILYMED_DIR="${DAILYMED_DIR:-$DATA_DIR/dailymed/xml}"

if [ ! -d "$DAILYMED_DIR" ]; then
    echo "⚠️  DailyMed directory not found at $DAILYMED_DIR"
    read -p "Enter DailyMed XML directory path: " DAILYMED_DIR
    if [ ! -d "$DAILYMED_DIR" ]; then
        echo "❌ Directory not found, skipping DailyMed"
        SKIP_DAILYMED=1
    fi
else
    XML_COUNT=$(find "$DAILYMED_DIR" -name "*.xml" 2>/dev/null | wc -l)
    echo "✅ Found DailyMed directory: $DAILYMED_DIR"
    echo "   XML files: $XML_COUNT"
fi

if [ -z "$SKIP_DAILYMED" ]; then
    # Check checkpoint
    CHECKPOINT_FILE="$DATA_DIR/dailymed_ingested_ids.txt"
    if [ -f "$CHECKPOINT_FILE" ]; then
        INGESTED_COUNT=$(wc -l < "$CHECKPOINT_FILE")
        echo "   Checkpoint: $INGESTED_COUNT labels already ingested"
    fi
    
    read -p "Start DailyMed ingestion? [Y/n]: " confirm
    if [[ ! $confirm =~ [nN] ]]; then
        LOG_FILE="$LOG_DIR/dailymed_$(date +%Y%m%d_%H%M%S).log"
        echo "   Starting DailyMed ingestion..."
        echo "   Log: $LOG_FILE"
        
        nohup python3 scripts/07_ingest_dailymed.py \
            --xml-dir "$DAILYMED_DIR" \
            > "$LOG_FILE" 2>&1 &
        
        DAILYMED_PID=$!
        echo $DAILYMED_PID > "$LOG_DIR/dailymed.pid"
        echo "   ✅ Started (PID: $DAILYMED_PID)"
    else
        echo "   Skipped DailyMed"
    fi
fi

echo ""
echo "=========================================="
echo "📊 Summary"
echo "=========================================="
echo ""

# Show all running processes
echo "Running ingestion processes:"
ps aux | grep "ingest_" | grep -v grep | awk '{print "  PID: " $2 " - " $11 " " $12}'

echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "Check checkpoints:"
echo "  wc -l $DATA_DIR/*_ingested_ids.txt $DATA_DIR/*_checkpoint.txt 2>/dev/null"
echo ""
echo "Stop all ingestion:"
echo "  kill \$(cat $LOG_DIR/*.pid 2>/dev/null)"
echo ""
