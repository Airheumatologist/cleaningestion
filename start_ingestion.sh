#!/bin/bash
# Start ingestion scripts for PMC, PubMed, and DailyMed
# Usage: ./start_ingestion.sh [--fresh]
#   --fresh: Recreate collection and clear all checkpoints before ingestion

set -e

# Parse arguments
FRESH_MODE=false
for arg in "$@"; do
    case $arg in
        --fresh)
            FRESH_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--fresh]"
            echo ""
            echo "Options:"
            echo "  --fresh    Recreate Qdrant collection and clear all checkpoints before ingestion"
            echo "  --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                  # Interactive ingestion (resumes from checkpoints)"
            echo "  $0 --fresh          # Fresh ingestion from scratch"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "📚 Medical Data Ingestion Launcher"
echo "=========================================="

# Configuration
DATA_DIR="${DATA_DIR:-/data/ingestion}"
LOG_DIR="$DATA_DIR/logs"
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

# =============================================================================
# FRESH INGESTION MODE
# =============================================================================
if [ "$FRESH_MODE" = true ]; then
    echo "⚠️  FRESH INGESTION MODE ENABLED"
    echo "   This will DELETE all existing data and start from scratch!"
    echo ""
    read -p "Are you sure you want to continue? [yes/no]: " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi
    
    echo ""
    echo "🧹 Step 1: Clearing checkpoint files..."
    
    # List of checkpoint files to clear
    CHECKPOINT_FILES=(
        "$DATA_DIR/pubmed_ingested_ids.txt"
        "$DATA_DIR/dailymed_ingested_ids.txt"
        "$DATA_DIR/pmc_ingested_ids.txt"
        "$DATA_DIR/author_manuscript_ingested_ids.txt"
    )
    
    for file in "${CHECKPOINT_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "   Removing: $(basename "$file")"
            rm -f "$file"
        fi
    done
    
    # Also clear any PID files
    if [ -d "$LOG_DIR" ]; then
        rm -f "$LOG_DIR"/*.pid
    fi
    
    echo "   ✅ Checkpoints cleared"
    echo ""
    
    echo "🗄️  Step 2: Recreating Qdrant collection..."
    python3 scripts/05_setup_qdrant.py --recreate
    echo "   ✅ Collection recreated"
    echo ""
    
    echo "=========================================="
    echo "✅ Fresh start prepared!"
    echo "=========================================="
    echo ""
fi

# =============================================================================
# INGESTION FUNCTIONS
# =============================================================================

# Function to run ingestion with logging (background)
run_ingestion_bg() {
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

# Function to run ingestion in foreground with tee
run_ingestion_fg() {
    local name=$1
    local script=$2
    local args=$3
    local logfile="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "🚀 Starting $name ingestion..."
    echo "   Log: $logfile"
    
    python3 "$script" $args 2>&1 | tee "$logfile"
}

# =============================================================================
# CHECK DATA AVAILABILITY
# =============================================================================

echo "📁 Checking data availability..."
echo ""

# Check PMC
PMC_AVAILABLE=false
PMC_DEFAULT_DIR="$DATA_DIR/pmc_xml"
if [ -d "$PMC_DEFAULT_DIR" ] && [ "$(find "$PMC_DEFAULT_DIR" -name "*.xml" -o -name "*.nxml" 2>/dev/null | wc -l)" -gt 0 ]; then
    PMC_XML_COUNT=$(find "$PMC_DEFAULT_DIR" -name "*.xml" -o -name "*.nxml" 2>/dev/null | wc -l)
    echo "   ✅ PMC: Found $PMC_XML_COUNT XML files in $PMC_DEFAULT_DIR"
    PMC_AVAILABLE=true
else
    echo "   ⚠️  PMC: No XML files found in $PMC_DEFAULT_DIR"
fi

# Check PubMed
PUBMED_AVAILABLE=false
PUBMED_DEFAULT_FILE="$DATA_DIR/pubmed_baseline/filtered/pubmed_abstracts.jsonl"
if [ -f "$PUBMED_DEFAULT_FILE" ]; then
    echo "   ✅ PubMed: Found $PUBMED_DEFAULT_FILE"
    PUBMED_AVAILABLE=true
else
    # Try to find any pubmed jsonl
    PUBMED_FOUND=$(find "$DATA_DIR" -name "*pubmed*.jsonl" -type f 2>/dev/null | head -1)
    if [ -n "$PUBMED_FOUND" ]; then
        echo "   ✅ PubMed: Found $PUBMED_FOUND"
        PUBMED_DEFAULT_FILE="$PUBMED_FOUND"
        PUBMED_AVAILABLE=true
    else
        echo "   ⚠️  PubMed: No JSONL files found"
        echo "      Note: PubMed baseline must be in JSONL format."
        echo "      If you have XML files, run: python scripts/20_download_pubmed_baseline.py --filter-only"
    fi
fi

# Check DailyMed
DAILYMED_AVAILABLE=false
DAILYMED_DEFAULT_DIR="$DATA_DIR/dailymed/xml"
if [ -d "$DAILYMED_DEFAULT_DIR" ] && [ "$(find "$DAILYMED_DEFAULT_DIR" -name "*.xml" 2>/dev/null | wc -l)" -gt 0 ]; then
    DAILYMED_XML_COUNT=$(find "$DAILYMED_DEFAULT_DIR" -name "*.xml" 2>/dev/null | wc -l)
    echo "   ✅ DailyMed: Found $DAILYMED_XML_COUNT XML files in $DAILYMED_DEFAULT_DIR"
    DAILYMED_AVAILABLE=true
else
    echo "   ⚠️  DailyMed: No XML files found in $DAILYMED_DEFAULT_DIR"
fi

echo ""

# =============================================================================
# USER SELECTION
# =============================================================================

echo "Select ingestion to start:"
echo "   1) PMC (PubMed Central full-text articles)"
echo "   2) PubMed Abstracts"
echo "   3) DailyMed (drug labels)"
echo "   4) All (in sequence - recommended)"
echo "   5) All (parallel - uses more resources)"
echo "   6) Post-ingestion: Generate drug lookup (run after DailyMed completes)"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "📄 Starting PMC ingestion..."
        read -p "PMC XML directory [$PMC_DEFAULT_DIR]: " pmc_dir
        pmc_dir=${pmc_dir:-$PMC_DEFAULT_DIR}
        
        if [ ! -d "$pmc_dir" ]; then
            echo "❌ Error: Directory not found: $pmc_dir"
            exit 1
        fi
        
        run_ingestion_bg "pmc" "scripts/06_ingest_pmc.py" "--xml-dir $pmc_dir"
        ;;
    
    2)
        echo ""
        echo "📄 Starting PubMed Abstracts ingestion..."
        
        if [ "$PUBMED_AVAILABLE" = true ]; then
            read -p "PubMed JSONL file [$PUBMED_DEFAULT_FILE]: " pubmed_file
            pubmed_file=${pubmed_file:-$PUBMED_DEFAULT_FILE}
        else
            read -p "PubMed JSONL file path: " pubmed_file
        fi
        
        if [ -z "$pubmed_file" ] || [ ! -f "$pubmed_file" ]; then
            echo "❌ Error: PubMed file not found: $pubmed_file"
            exit 1
        fi
        
        run_ingestion_bg "pubmed" "scripts/21_ingest_pubmed_abstracts.py" "--input $pubmed_file"
        ;;
    
    3)
        echo ""
        echo "💊 Starting DailyMed ingestion..."
        read -p "DailyMed XML directory [$DAILYMED_DEFAULT_DIR]: " dailymed_dir
        dailymed_dir=${dailymed_dir:-$DAILYMED_DEFAULT_DIR}
        
        if [ ! -d "$dailymed_dir" ]; then
            echo "❌ Error: Directory not found: $dailymed_dir"
            exit 1
        fi
        
        run_ingestion_fg "dailymed" "scripts/07_ingest_dailymed.py" "--xml-dir $dailymed_dir"
        
        # Generate drug lookup after DailyMed ingestion completes
        echo ""
        echo "📋 Generating drug lookup cache..."
        python3 scripts/generate_drug_lookup.py
        ;;
    
    4)
        echo ""
        echo "🔄 Starting all ingestions in sequence..."
        echo "   (Recommended for CPU/memory constrained systems)"
        echo ""
        
        # PMC
        if [ "$PMC_AVAILABLE" = true ]; then
            read -p "PMC XML directory [$PMC_DEFAULT_DIR]: " pmc_dir
            pmc_dir=${pmc_dir:-$PMC_DEFAULT_DIR}
            if [ -d "$pmc_dir" ]; then
                echo ""
                echo "📄 Ingesting PMC..."
                run_ingestion_fg "pmc" "scripts/06_ingest_pmc.py" "--xml-dir $pmc_dir"
            fi
        else
            echo "⚠️  Skipping PMC (data not found)"
        fi
        
        # PubMed
        if [ "$PUBMED_AVAILABLE" = true ]; then
            echo ""
            read -p "PubMed JSONL file [$PUBMED_DEFAULT_FILE]: " pubmed_file
            pubmed_file=${pubmed_file:-$PUBMED_DEFAULT_FILE}
            if [ -f "$pubmed_file" ]; then
                echo ""
                echo "📄 Ingesting PubMed Abstracts..."
                run_ingestion_fg "pubmed" "scripts/21_ingest_pubmed_abstracts.py" "--input $pubmed_file"
            fi
        else
            echo "⚠️  Skipping PubMed (data not found)"
        fi
        
        # DailyMed
        if [ "$DAILYMED_AVAILABLE" = true ]; then
            echo ""
            read -p "DailyMed XML directory [$DAILYMED_DEFAULT_DIR]: " dailymed_dir
            dailymed_dir=${dailymed_dir:-$DAILYMED_DEFAULT_DIR}
            if [ -d "$dailymed_dir" ]; then
                echo ""
                echo "💊 Ingesting DailyMed..."
                run_ingestion_fg "dailymed" "scripts/07_ingest_dailymed.py" "--xml-dir $dailymed_dir"
            fi
        else
            echo "⚠️  Skipping DailyMed (data not found)"
        fi
        
        # Generate drug lookup for fast O(1) drug name lookups
        echo ""
        echo "📋 Generating drug lookup cache..."
        python3 scripts/generate_drug_lookup.py || echo "⚠️  Drug lookup generation failed (non-critical)"
        
        echo ""
        echo "✅ All sequential ingestion complete!"
        exit 0
        ;;
    
    5)
        echo ""
        echo "⚠️  WARNING: Parallel ingestion uses more CPU/memory resources"
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm != [yY] ]]; then
            exit 0
        fi
        
        started_any=false
        
        # PMC
        if [ "$PMC_AVAILABLE" = true ]; then
            read -p "PMC XML directory [$PMC_DEFAULT_DIR]: " pmc_dir
            pmc_dir=${pmc_dir:-$PMC_DEFAULT_DIR}
            if [ -d "$pmc_dir" ]; then
                run_ingestion_bg "pmc" "scripts/06_ingest_pmc.py" "--xml-dir $pmc_dir"
                started_any=true
            fi
        fi
        
        # PubMed
        if [ "$PUBMED_AVAILABLE" = true ]; then
            read -p "PubMed JSONL file [$PUBMED_DEFAULT_FILE]: " pubmed_file
            pubmed_file=${pubmed_file:-$PUBMED_DEFAULT_FILE}
            if [ -f "$pubmed_file" ]; then
                run_ingestion_bg "pubmed" "scripts/21_ingest_pubmed_abstracts.py" "--input $pubmed_file"
                started_any=true
            fi
        fi
        
        # DailyMed
        if [ "$DAILYMED_AVAILABLE" = true ]; then
            read -p "DailyMed XML directory [$DAILYMED_DEFAULT_DIR]: " dailymed_dir
            dailymed_dir=${dailymed_dir:-$DAILYMED_DEFAULT_DIR}
            if [ -d "$dailymed_dir" ]; then
                run_ingestion_bg "dailymed" "scripts/07_ingest_dailymed.py" "--xml-dir $dailymed_dir"
                started_any=true
            fi
        fi
        
        if [ "$started_any" = false ]; then
            echo "❌ No data sources available to start"
            exit 1
        fi
        
        echo ""
        echo "✅ Parallel ingestion started"
        echo ""
        echo "⚠️  IMPORTANT: After DailyMed ingestion completes, run option 6 to generate drug lookup"
        ;;
    
    6)
        echo ""
        echo "📋 Running post-ingestion tasks..."
        echo ""
        echo "Generating drug lookup cache for fast O(1) drug name → set_id lookups..."
        python3 scripts/generate_drug_lookup.py
        echo ""
        echo "✅ Post-ingestion tasks complete!"
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

# Only show monitoring info for actual ingestion choices (not post-ingestion)
if [ "$choice" != "6" ]; then
    echo ""
    echo "=========================================="
    echo "✅ Ingestion process started!"
    echo "=========================================="
    echo ""
    echo "Monitor progress:"
    echo "   - Check logs: tail -f $LOG_DIR/*.log"
    echo "   - Check PIDs: cat $LOG_DIR/*.pid"
    echo "   - View checkpoints: ls -la $DATA_DIR/*checkpoint* $DATA_DIR/*_ingested_ids.txt"
    echo ""
    echo "To stop ingestion:"
    echo "   kill \$(cat $LOG_DIR/*.pid)"
    echo ""
    echo "📋 IMPORTANT: After DailyMed ingestion completes, run:"
    echo "   ./start_ingestion.sh  →  Select option 6) Post-ingestion: Generate drug lookup"
    echo ""
fi
