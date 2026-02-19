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
PUBMED_BASELINE_DIR="${PUBMED_BASELINE_DIR:-$DATA_DIR/pubmed_baseline}"
PUBMED_DEFAULT_FILE="${PUBMED_ABSTRACTS_FILE:-$PUBMED_BASELINE_DIR/filtered/pubmed_abstracts.jsonl}"
PMC_XML_DIR="${PMC_XML_DIR:-$DATA_DIR/pmc_xml}"
DAILYMED_STATE_DIR="${DAILYMED_STATE_DIR:-$DATA_DIR/dailymed/state}"
DAILYMED_SET_ID_MANIFEST="${DAILYMED_SET_ID_MANIFEST:-$DAILYMED_STATE_DIR/dailymed_last_update_set_ids.txt}"
PMC_INCREMENTAL_STATE_FILE="$PMC_XML_DIR/.pmc_s3_inventory_state.json"
PMC_INCREMENTAL_MARKERS_DIR="$PMC_XML_DIR/.pmc_s3_markers"
PUBMED_UPDATES_TRACKER="$DATA_DIR/processed_updates.json"
PUBMED_UPDATEFILES_DIR="$DATA_DIR/pubmed_updatefiles"
PUBMED_FILTER_PROGRESS_FILE="$PUBMED_BASELINE_DIR/progress.json"
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"

# Activate virtual environment (deployment standard: venv; keep .venv fallback)
source venv/bin/activate || source .venv/bin/activate

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
    
    # Active checkpoint files to clear
    CHECKPOINT_FILES=(
        "$DATA_DIR/pubmed_ingested_ids.txt"
        "$DATA_DIR/dailymed_ingested_ids.txt"
        "$DATA_DIR/pmc_ingested_ids.txt"
    )
    
    for file in "${CHECKPOINT_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "   Removing: $(basename "$file")"
            rm -f "$file"
        fi
    done

    # Legacy migration checkpoint (read-only in current ingest code).
    # Keep clearing it in fresh mode so old IDs cannot affect a true reset.
    LEGACY_AUTHOR_CHECKPOINT_FILE="$DATA_DIR/author_manuscript_ingested_ids.txt"
    if [ -f "$LEGACY_AUTHOR_CHECKPOINT_FILE" ]; then
        echo "   Removing legacy: $(basename "$LEGACY_AUTHOR_CHECKPOINT_FILE")"
        rm -f "$LEGACY_AUTHOR_CHECKPOINT_FILE"
    fi
    
    # Also clear any PID files
    if [ -d "$LOG_DIR" ]; then
        rm -f "$LOG_DIR"/*.pid
    fi

    echo "🧹 Step 2: Clearing incremental state files..."
    STATE_FILES=(
        "$PUBMED_UPDATES_TRACKER"
        "$PUBMED_FILTER_PROGRESS_FILE"
        "$DAILYMED_SET_ID_MANIFEST"
        "$PMC_INCREMENTAL_STATE_FILE"
    )

    for file in "${STATE_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "   Removing state: $(basename "$file")"
            rm -f "$file"
        fi
    done

    if [ -d "$PMC_INCREMENTAL_MARKERS_DIR" ]; then
        echo "   Removing state dir: $PMC_INCREMENTAL_MARKERS_DIR"
        rm -rf "$PMC_INCREMENTAL_MARKERS_DIR"
    fi

    if [ -d "$PUBMED_UPDATEFILES_DIR" ]; then
        echo "   Removing update cache dir: $PUBMED_UPDATEFILES_DIR"
        rm -rf "$PUBMED_UPDATEFILES_DIR"
    fi
    
    echo "   ✅ Checkpoints and incremental state cleared"
    echo ""
    
    echo "🗄️  Step 3: Recreating Qdrant collection..."
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
PMC_DEFAULT_DIR="$PMC_XML_DIR"
if [ -d "$PMC_DEFAULT_DIR" ] && [ "$(find "$PMC_DEFAULT_DIR" -name "*.xml" -o -name "*.nxml" 2>/dev/null | wc -l)" -gt 0 ]; then
    PMC_XML_COUNT=$(find "$PMC_DEFAULT_DIR" -name "*.xml" -o -name "*.nxml" 2>/dev/null | wc -l)
    echo "   ✅ PMC: Found $PMC_XML_COUNT XML files in $PMC_DEFAULT_DIR"
    PMC_AVAILABLE=true
else
    echo "   ⚠️  PMC: No XML files found in $PMC_DEFAULT_DIR"
fi

# Check PubMed
PUBMED_AVAILABLE=false
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
echo "   6) Post-ingestion: Generate drug lookup + finalize Qdrant indexing"
echo ""
read -p "Enter choice [1-6]: " choice
parallel_drug_lookup_hook_scheduled=false

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
        echo "⚙️  Finalizing Qdrant indexing..."
        python3 scripts/05_setup_qdrant.py --finalize
        
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
        dailymed_started=false
        dailymed_pid=""
        
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
                dailymed_started=true
                dailymed_pid=$(cat "$LOG_DIR/dailymed.pid")
            fi
        fi
        
        if [ "$started_any" = false ]; then
            echo "❌ No data sources available to start"
            exit 1
        fi

        # In parallel mode, auto-run drug lookup when DailyMed ingestion completes.
        # This prevents a missed manual option 6 step from leaving stale lookup data.
        if [ "$dailymed_started" = true ] && [ -n "$dailymed_pid" ]; then
            post_ingestion_log="$LOG_DIR/post_ingestion_$(date +%Y%m%d_%H%M%S).log"
            (
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for DailyMed PID $dailymed_pid to finish..."
                while kill -0 "$dailymed_pid" 2>/dev/null; do
                    sleep 30
                done

                echo "[$(date '+%Y-%m-%d %H:%M:%S')] DailyMed ingestion finished; generating drug lookup cache..."
                if python3 scripts/generate_drug_lookup.py; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Drug lookup cache generation complete."
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Drug lookup cache generation failed."
                fi
            ) >> "$post_ingestion_log" 2>&1 &

            echo $! > "$LOG_DIR/post_ingestion.pid"
            echo ""
            echo "📋 Post-hook scheduled: drug lookup will auto-run after DailyMed completes"
            echo "   Hook log: $post_ingestion_log"
            echo "   Hook PID: $(cat "$LOG_DIR/post_ingestion.pid")"
            parallel_drug_lookup_hook_scheduled=true
        fi
        
        echo ""
        echo "✅ Parallel ingestion started"
        echo ""
        echo "⚠️  IMPORTANT: After all ingestion pipelines complete, run option 6 to finalize Qdrant indexing"
        ;;
    
    6)
        echo ""
        echo "📋 Running post-ingestion tasks..."
        echo ""
        echo "Generating drug lookup cache for fast O(1) drug name → set_id lookups..."
        python3 scripts/generate_drug_lookup.py
        echo "Re-enabling Qdrant indexing (HNSW build)..."
        python3 scripts/05_setup_qdrant.py --finalize
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
    if [ "$choice" = "5" ]; then
        if [ "$parallel_drug_lookup_hook_scheduled" = true ]; then
            echo "📋 IMPORTANT: In parallel mode, drug lookup auto-runs after DailyMed completes."
            echo "   After all ingestion pipelines complete, run option 6 to finalize Qdrant indexing."
        else
            echo "📋 IMPORTANT: DailyMed was not started in this run."
            echo "   Run option 6 after ingestion completes to generate drug lookup and finalize indexing."
        fi
    else
        echo "📋 IMPORTANT: After all ingestion pipelines complete, run:"
        echo "   ./start_ingestion.sh  →  Select option 6) Post-ingestion tasks (drug lookup + indexing finalize)"
    fi
    echo ""
fi
