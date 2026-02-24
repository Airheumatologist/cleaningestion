#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

if ! curl -fsS "${API_URL}/api/v1/health" >/dev/null; then
  msg="RAG API health check failed on $(hostname) at $(date -u +"%Y-%m-%dT%H:%M:%SZ") target=${API_URL}"
  echo "${msg}"
  if [[ -n "${SLACK_WEBHOOK_URL}" ]]; then
    curl -fsS -X POST -H "Content-type: application/json" \
      --data "{\"text\":\"${msg}\"}" "${SLACK_WEBHOOK_URL}" >/dev/null || true
  fi
  exit 1
fi
