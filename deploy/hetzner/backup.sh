#!/usr/bin/env bash
# Qdrant snapshot backup script.
#
# Called automatically after the weekly update via medical-rag-update.cron.
# Can also be run standalone for ad-hoc backups.
#
# Required env vars:
#   QDRANT_API_KEY   - Qdrant service API key
#   QDRANT_URL       - Qdrant base URL (default: http://localhost:6333)
#   QDRANT_COLLECTION - collection name   (default: rag_pipeline)
#
# Optional env vars:
#   BACKUP_DIR       - local backup directory  (default: /data/backups)
#   S3_BUCKET        - S3 destination URI, e.g. s3://my-bucket/qdrant-backups
#                      MUST include the s3:// scheme prefix.
#   S3_ENDPOINT_URL  - custom S3-compatible endpoint (e.g. for Hetzner Object Storage)
#
# Retention: the 7 most recent .snapshot files are kept (~7 weeks history).
# Adjust the awk 'NR>7' line below to change this.
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-rag_pipeline}"
BACKUP_DIR="${BACKUP_DIR:-/data/backups}"
# S3_BUCKET must be a full URI including scheme, e.g.: s3://my-bucket/qdrant-backups
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

# Host path to Qdrant storage is /opt/qdrant/qdrant_storage (mapped to
# /qdrant/storage inside the container per docker-compose.yml).
# Qdrant writes snapshots to {storage}/snapshots/{collection}/{name}.
snapshot_path="/opt/qdrant/qdrant_storage/snapshots/${QDRANT_COLLECTION}/${snapshot_name}"

if [[ ! -f "${snapshot_path}" ]]; then
  echo "Snapshot not found at ${snapshot_path}"
  exit 1
fi

cp "${snapshot_path}" "${BACKUP_DIR}/${snapshot_name}"

echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") snapshot=${snapshot_name} dest=${BACKUP_DIR}/${snapshot_name}"

if [[ -n "${S3_BUCKET}" && -n "${S3_ENDPOINT_URL}" ]]; then
  aws s3 cp "${BACKUP_DIR}/${snapshot_name}" "${S3_BUCKET}/${snapshot_name}" --endpoint-url "${S3_ENDPOINT_URL}"
  echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") uploaded to ${S3_BUCKET}/${snapshot_name}"
fi

# Retain the 7 most recent snapshots (~7 weeks). Older files are deleted.
# Change 'NR>7' to 'NR>4' if you prefer ~1 month of retention.
ls -1t "${BACKUP_DIR}"/*.snapshot 2>/dev/null | awk 'NR>7' | xargs -r rm -f
