#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

{ source venv/bin/activate || source .venv/bin/activate; }
set -a
source .env
set +a

PMC_XML_DIR="${PMC_XML_DIR:-/data/ingestion/pmc_xml}"
COLLECTION_NAME="${COLLECTION_NAME:-${QDRANT_COLLECTION:-rag_pipeline}}"

log() {
  printf '%s %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

log "Starting semiannual PMC incremental refresh (datasets=pmc_oa,author_manuscript)"

python scripts/01_download_pmc_unified.py \
  --output-dir "${PMC_XML_DIR}" \
  --datasets pmc_oa,author_manuscript \
  --release-mode incremental

python scripts/06_ingest_pmc.py \
  --xml-dir "${PMC_XML_DIR}" \
  --delete-source

python scripts/05_setup_qdrant.py \
  --finalize \
  --collection-name "${COLLECTION_NAME}" \
  --indexing-threshold 10000

POINTS_COUNT="$(
  python -c "import os; from qdrant_client import QdrantClient; c = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY') or None, timeout=120, prefer_grpc=os.getenv('USE_GRPC', 'true').strip().lower() in {'1','true','yes','on'}); info = c.get_collection(os.getenv('COLLECTION_NAME') or os.getenv('QDRANT_COLLECTION') or 'rag_pipeline'); print(info.points_count)"
)"

log "PMC semiannual refresh complete (collection=${COLLECTION_NAME} points=${POINTS_COUNT})"
