#!/usr/bin/env bash
set -euo pipefail

QDRANT_URL="${QDRANT_URL:-}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
QDRANT_CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-qdrant}"

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
  msg="Qdrant health check failed: unable to resolve ${QDRANT_CONTAINER_NAME} container IP on $(hostname) at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "${msg}"
  if [[ -n "${SLACK_WEBHOOK_URL}" ]]; then
    curl -fsS -X POST -H 'Content-type: application/json' \
      --data "{\"text\":\"${msg}\"}" "${SLACK_WEBHOOK_URL}" >/dev/null || true
  fi
  exit 1
fi

if ! curl -fsS -H "api-key: ${QDRANT_API_KEY}" "${TARGET_URL}/healthz" >/dev/null; then
  msg="Qdrant health check failed on $(hostname) at $(date -u +"%Y-%m-%dT%H:%M:%SZ") target=${TARGET_URL}"
  echo "${msg}"
  if [[ -n "${SLACK_WEBHOOK_URL}" ]]; then
    curl -fsS -X POST -H 'Content-type: application/json' \
      --data "{\"text\":\"${msg}\"}" "${SLACK_WEBHOOK_URL}" >/dev/null || true
  fi
  exit 1
fi
