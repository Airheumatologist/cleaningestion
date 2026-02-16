#!/bin/bash
# Start ingestion scripts for PMC, PubMed, and DailyMed

set -e

echo "=========================================="
echo "📚 Starting Medical Data Ingestion"
echo "=========================================="

# Configuration
DATA_DIR="${DATA_DIR:-/data/ingestion}"
LOG_DIR="${DATA_DIR}/logs"
mkdir -p "$LOG_DIR"

# Activate virtual environment
source .venv/bin/activate || source venv/bin/activate

echo ""
echo "🔧 Configuration:"
echo "   Data directory: $DATA_DIR"
echo "   Log directory: $LOG_DIR"
echo "   Embedding: ${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B-batch}"
echo "   Provider: ${EMBEDDING_PROVIDER:-deepinfra}"
echo ""

# Function to run ingestion with logging
run_ingestion() {
    local name=$1
    local script=$2
    local args=$3
    local logfile="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "🚀 Starting $name ingestion..."
    echo "   Log: $logfile"
    
    nohup python3 "$script" $args > "$logfile" 2>&1 &
    echo $! > "$LOG_DIR/${name}.pid"
    echo "   PID: $(cat "$LOG_DIR/${name}.pid")"
}

# Ask user which ingestion to start
echo "Select ingestion to start:"
echo "   1) PMC (PubMed Central full-text articles)"
echo "   2) PubMed Abstracts"
echo "   3) DailyMed (drug labels)"
echo "   4) All (in sequence)"
echo "   5) All (parallel - not recommended for CPU)"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "📄 Starting PMC ingestion..."
        read -p "PMC XML directory [$DATA_DIR/pmc_xml]: " pmc_dir
        pmc_dir=${pmc_dir:-$DATA_DIR/pmc_xml}
        run_ingestion "pmc" "scripts/06_ingest_pmc.py" "--xml-dir $pmc_dir"
        ;;
    
    2)
        echo ""
        echo "📄 Starting PubMed Abstracts ingestion..."
        read -p "PubMed JSONL file path: " pubmed_file
        if [ -z "$pubmed_file" ]; then
            echo "❌ Error: PubMed file path required"
            exit 1
        fi
        run_ingestion "pubmed" "scripts/21_ingest_pubmed_abstracts.py" "--input $pubmed_file"
        ;;
    
    3)
        echo ""
        echo "💊 Starting DailyMed ingestion..."
        read -p "DailyMed XML directory [$DATA_DIR/dailymed/xml]: " dailymed_dir
        dailymed_dir=${dailymed_dir:-$DATA_DIR/dailymed/xml}
        run_ingestion "dailymed" "scripts/07_ingest_dailymed.py" "--xml-dir $dailymed_dir"
        ;;
    
    4)
        echo ""
        echo "🔄 Starting all ingestions in sequence..."
        echo "   (Recommended for CPU servers)"
        echo ""
        
        # PMC
        read -p "PMC XML directory [$DATA_DIR/pmc_xml]: " pmc_dir
        pmc_dir=${pmc_dir:-$DATA_DIR/pmc_xml}
        if [ -d "$pmc_dir" ]; then
            echo "📄 Ingesting PMC..."
            python3 scripts/06_ingest_pmc.py --xml-dir "$pmc_dir" 2>&1 | tee "$LOG_DIR/pmc_$(date +%Y%m%d_%H%M%S).log"
        else
            echo "⚠️  PMC directory not found, skipping..."
        fi
        
        # PubMed
        read -p "PubMed JSONL file path (or skip): " pubmed_file
        if [ -n "$pubmed_file" ] && [ -f "$pubmed_file" ]; then
            echo "📄 Ingesting PubMed Abstracts..."
            python3 scripts/21_ingest_pubmed_abstracts.py --input "$pubmed_file" 2>&1 | tee "$LOG_DIR/pubmed_$(date +%Y%m%d_%H%M%S).log"
        else
            echo "⚠️  PubMed file not provided or not found, skipping..."
        fi
        
        # DailyMed
        read -p "DailyMed XML directory [$DATA_DIR/dailymed/xml]: " dailymed_dir
        dailymed_dir=${dailymed_dir:-$DATA_DIR/dailymed/xml}
        if [ -d "$dailymed_dir" ]; then
            echo "💊 Ingesting DailyMed..."
            python3 scripts/07_ingest_dailymed.py --xml-dir "$dailymed_dir" 2>&1 | tee "$LOG_DIR/dailymed_$(date +%Y%m%d_%H%M%S).log"
        else
            echo "⚠️  DailyMed directory not found, skipping..."
        fi
        ;;
    
    5)
        echo ""
        echo "⚠️  WARNING: Parallel ingestion uses more resources"
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm != [yY] ]]; then
            exit 0
        fi
        
        read -p "PMC XML directory [$DATA_DIR/pmc_xml]: " pmc_dir
        pmc_dir=${pmc_dir:-$DATA_DIR/pmc_xml}
        run_ingestion "pmc" "scripts/06_ingest_pmc.py" "--xml-dir $pmc_dir"
        
        read -p "DailyMed XML directory [$DATA_DIR/dailymed/xml]: " dailymed_dir
        dailymed_dir=${dailymed_dir:-$DATA_DIR/dailymed/xml}
        run_ingestion "dailymed" "scripts/07_ingest_dailymed.py" "--xml-dir $dailymed_dir"
        
        echo ""
        echo "✅ Parallel ingestion started"
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Ingestion process started!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "   - Check logs: tail -f $LOG_DIR/*.log"
echo "   - Check PIDs: cat $LOG_DIR/*.pid"
echo "   - View checkpoints: ls -la $DATA_DIR/*checkpoint*"
echo ""
echo "To stop ingestion:"
echo "   kill \$(cat $LOG_DIR/*.pid)"
echo ""
