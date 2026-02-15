#!/bin/bash
# Run the PMC download script in the background using nohup

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure we are in the project root
cd "$PROJECT_ROOT"

# Log file path
LOG_FILE="download_pmc.log"

echo "Starting PMC download in background..."
# Check for virtual environment
if [ -f "venv/bin/python3" ]; then
    PYTHON_CMD="venv/bin/python3"
elif [ -f ".venv/bin/python3" ]; then
    PYTHON_CMD=".venv/bin/python3"
else
    PYTHON_CMD="python3"
    echo "Warning: No virtual environment found. Using system python3."
fi

# Run with nohup, unbuffered output
nohup $PYTHON_CMD -u scripts/01_download_pmc.py > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID $PID"
echo "To view logs, run: tail -f $LOG_FILE"
