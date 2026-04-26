#!/bin/zsh

set -u

ROOT_DIR="/Volumes/Vibing/cleaningestion"
DATA_DIR_DEFAULT="$ROOT_DIR/data/ingestion"
LOG_DIR_DEFAULT="$DATA_DIR_DEFAULT/logs"

export DATA_DIR="${DATA_DIR:-$DATA_DIR_DEFAULT}"
LOG_DIR="${LOG_DIR:-$LOG_DIR_DEFAULT}"
mkdir -p "$LOG_DIR"

SUPERVISOR_LOG="${SUPERVISOR_LOG:-$LOG_DIR/pubmed_ftp_stream_supervisor_$(date +%Y%m%d_%H%M%S).log}"
RESTART_DELAY="${RESTART_DELAY:-20}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"

attempt=0
while true; do
  attempt=$((attempt + 1))
  run_log="$LOG_DIR/pubmed_ftp_stream_restart_$(date +%Y%m%d_%H%M%S).log"
  start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '%s supervisor attempt=%d starting run_log=%s\n' "$start_ts" "$attempt" "$run_log" >> "$SUPERVISOR_LOG"

  (
    cd "$ROOT_DIR" || exit 1
    export PYTHONUNBUFFERED=1
    python3 scripts/baseline/ingest_pubmed_ftp_stream.py "$@"
  ) >> "$run_log" 2>&1
  exit_code=$?

  end_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '%s supervisor attempt=%d exit_code=%d run_log=%s\n' "$end_ts" "$attempt" "$exit_code" "$run_log" >> "$SUPERVISOR_LOG"

  if [ "$exit_code" -eq 0 ]; then
    printf '%s supervisor exiting after successful completion\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$SUPERVISOR_LOG"
    exit 0
  fi

  if [ "$MAX_RESTARTS" -gt 0 ] && [ "$attempt" -ge "$MAX_RESTARTS" ]; then
    printf '%s supervisor giving up after %d attempts\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$attempt" >> "$SUPERVISOR_LOG"
    exit "$exit_code"
  fi

  printf '%s supervisor sleeping %ss before restart\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$RESTART_DELAY" >> "$SUPERVISOR_LOG"
  sleep "$RESTART_DELAY"
done
