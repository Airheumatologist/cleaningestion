#!/usr/bin/env bash
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-rag_pipeline}"
BACKUP_DIR="${BACKUP_DIR:-/data/backups}"
S3_BUCKET="${S3_BUCKET:-}"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-}"

if [[ -z "${QDRANT_API_KEY}" ]]; then
  echo "QDRANT_API_KEY is required"
  exit 1
fi

mkdir -p "${BACKUP_DIR}"

response="$(curl -fsS -X POST \
  -H "api-key: ${QDRANT_API_KEY}" \
  "${QDRANT_URL}/collections/${QDRANT_COLLECTION}/snapshots")"

snapshot_name="$(printf '%s' "${response}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["name"])')"
snapshot_path="/opt/qdrant/qdrant_storage/snapshots/${QDRANT_COLLECTION}/${snapshot_name}"

if [[ ! -f "${snapshot_path}" ]]; then
  echo "Snapshot not found at ${snapshot_path}"
  exit 1
fi

cp "${snapshot_path}" "${BACKUP_DIR}/${snapshot_name}"

echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") snapshot=${snapshot_name}"

if [[ -n "${S3_BUCKET}" && -n "${S3_ENDPOINT_URL}" ]]; then
  aws s3 cp "${BACKUP_DIR}/${snapshot_name}" "${S3_BUCKET}/${snapshot_name}" --endpoint-url "${S3_ENDPOINT_URL}"
fi

ls -1t "${BACKUP_DIR}"/*.snapshot 2>/dev/null | awk 'NR>7' | xargs -r rm -f
