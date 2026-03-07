#!/usr/bin/env bash
# Qdrant snapshot backup script.
#
# Called automatically after scheduled weekly and semiannual PMC updates via
# medical-rag-update.cron.
# Can also be run standalone for ad-hoc backups.
#
# Required env vars:
#   QDRANT_API_KEY   - Qdrant service API key
#
# Optional env vars:
#   QDRANT_URL       - Qdrant base URL; when unset, resolves from Docker container IP
#   QDRANT_CONTAINER_NAME - container name used for IP resolution (default: qdrant)
#   QDRANT_COLLECTION - collection name   (default: rag_pipeline)
#   BACKUP_DIR       - local backup directory  (default: /data/backups)
#   S3_BUCKET        - S3 destination URI, e.g. s3://my-bucket/qdrant-backups
#                      MUST include the s3:// scheme prefix.
#   S3_ENDPOINT_URL  - custom S3-compatible endpoint (e.g. for Hetzner Object Storage)
#
# Retention: the 7 most recent .snapshot files are kept (~7 weeks history).
# Adjust the awk 'NR>7' line below to change this.
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
QDRANT_CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-qdrant}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-rag_pipeline}"
BACKUP_DIR="${BACKUP_DIR:-/data/backups}"
# S3_BUCKET must be a full URI including scheme, e.g.: s3://my-bucket/qdrant-backups
S3_BUCKET="${S3_BUCKET:-}"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-}"

if [[ -z "${QDRANT_API_KEY}" ]]; then
  echo "QDRANT_API_KEY is required"
  exit 1
fi

resolve_qdrant_url() {
  if [[ -n "${QDRANT_URL}" ]]; then
    printf '%s\n' "${QDRANT_URL}"
    return 0
  fi

  local ip
  ip="$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${QDRANT_CONTAINER_NAME}" 2>/dev/null || true)"
  if [[ -z "${ip}" ]]; then
    return 1
  fi
  printf 'http://%s:6333\n' "${ip}"
}

TARGET_URL="$(resolve_qdrant_url || true)"
if [[ -z "${TARGET_URL}" ]]; then
  echo "Failed to resolve Qdrant URL. Set QDRANT_URL or ensure container '${QDRANT_CONTAINER_NAME}' is running."
  exit 1
fi

mkdir -p "${BACKUP_DIR}"

response="$(curl -fsS -X POST \
  -H "api-key: ${QDRANT_API_KEY}" \
  "${TARGET_URL}/collections/${QDRANT_COLLECTION}/snapshots")"

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
