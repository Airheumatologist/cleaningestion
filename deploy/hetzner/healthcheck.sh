#!/usr/bin/env bash
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

if ! curl -fsS -H "api-key: ${QDRANT_API_KEY}" "${QDRANT_URL}/healthz" >/dev/null; then
  msg="Qdrant health check failed on $(hostname) at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "${msg}"
  if [[ -n "${SLACK_WEBHOOK_URL}" ]]; then
    curl -fsS -X POST -H 'Content-type: application/json' \
      --data "{\"text\":\"${msg}\"}" "${SLACK_WEBHOOK_URL}" >/dev/null || true
  fi
  exit 1
fi
