#!/bin/bash
# DailyMed strict 1k gate runner:
# 1) Download XML (HTTPS)
# 2) Ingest 1,000 deterministic labels into test namespace A + validate
# 3) Repeat for namespace B + validate
# 4) Delete both test namespaces
# 5) Delete test XMLs
# 6) Optionally redownload and start production ingestion

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

START_PRODUCTION=false
DEEPINFRA_KEY="${DAILYMED_DEEPINFRA_API_KEY:-${DEEPINFRA_DAILYMED_API_KEY:-}}"
TEST_LABEL_COUNT="${TEST_LABEL_COUNT:-1000}"
PRODUCTION_NAMESPACE="${PRODUCTION_NAMESPACE:-medical_database_dailymed}"
NS_A="${DAILYMED_TEST_NAMESPACE_A:-medical_database_dailymed_test_1000_a}"
NS_B="${DAILYMED_TEST_NAMESPACE_B:-medical_database_dailymed_test_1000_b}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deepinfra-key)
      DEEPINFRA_KEY="${2:-}"
      shift 2
      ;;
    --start-production)
      START_PRODUCTION=true
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage: scripts/run_dailymed_1k_test_gate.sh [--deepinfra-key <key>] [--start-production]

Environment overrides:
  DAILYMED_DEEPINFRA_API_KEY / DEEPINFRA_DAILYMED_API_KEY
  DATA_DIR
  DAILYMED_XML_DIR
  TEST_LABEL_COUNT (default: 1000)
  DAILYMED_TEST_NAMESPACE_A (default: medical_database_dailymed_test_1000_a)
  DAILYMED_TEST_NAMESPACE_B (default: medical_database_dailymed_test_1000_b)
  PRODUCTION_NAMESPACE (default: medical_database_dailymed)
  MAX_WORKERS_TEST (default: 2)
  DAILYMED_FILE_BATCH_SIZE_TEST (default: 100)
  EMBEDDING_CONCURRENCY_TEST (default: 1)
  EMBEDDING_MAX_REQUESTS_PER_SEC_TEST (default: 2)
  TURBOPUFFER_MAX_CONCURRENT_WRITES_TEST (default: 4)
  TURBOPUFFER_DISABLE_BACKPRESSURE_TEST (default: 1)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${DEEPINFRA_KEY}" ]]; then
  echo "Missing DeepInfra key. Use --deepinfra-key or DAILYMED_DEEPINFRA_API_KEY." >&2
  exit 2
fi

if [[ -z "${TURBOPUFFER_API_KEY:-}" ]]; then
  echo "Missing TURBOPUFFER_API_KEY in environment." >&2
  exit 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/ingestion}"
DAILYMED_XML_DIR="${DAILYMED_XML_DIR:-${DATA_DIR}/dailymed/xml}"
DAILYMED_STATE_DIR="${DAILYMED_STATE_DIR:-${DATA_DIR}/dailymed/state}"
LOG_DIR="${DATA_DIR}/logs/dailymed_test_gate_${RUN_TS}"
ARCHIVE_DIR="${DAILYMED_STATE_DIR}/test_gate_reports/${RUN_TS}"

mkdir -p "${LOG_DIR}" "${ARCHIVE_DIR}" "${DAILYMED_STATE_DIR}"

CHECKPOINT_A="${DAILYMED_STATE_DIR}/dailymed_test_1000_a_ingested_ids.txt"
CHECKPOINT_B="${DAILYMED_STATE_DIR}/dailymed_test_1000_b_ingested_ids.txt"
AUDIT_A="${DAILYMED_STATE_DIR}/dailymed_test_1000_a_audit_${RUN_TS}.json"
AUDIT_B="${DAILYMED_STATE_DIR}/dailymed_test_1000_b_audit_${RUN_TS}.json"
REPORT_A="${ARCHIVE_DIR}/dailymed_test_1000_a_report.json"
REPORT_B="${ARCHIVE_DIR}/dailymed_test_1000_b_report.json"
DELETE_REPORT_A="${ARCHIVE_DIR}/dailymed_test_1000_a_delete.json"
DELETE_REPORT_B="${ARCHIVE_DIR}/dailymed_test_1000_b_delete.json"

MAX_WORKERS_TEST="${MAX_WORKERS_TEST:-2}"
DAILYMED_FILE_BATCH_SIZE_TEST="${DAILYMED_FILE_BATCH_SIZE_TEST:-100}"
EMBEDDING_CONCURRENCY_TEST="${EMBEDDING_CONCURRENCY_TEST:-1}"
EMBEDDING_MAX_REQUESTS_PER_SEC_TEST="${EMBEDDING_MAX_REQUESTS_PER_SEC_TEST:-2}"
TURBOPUFFER_MAX_CONCURRENT_WRITES_TEST="${TURBOPUFFER_MAX_CONCURRENT_WRITES_TEST:-4}"
TURBOPUFFER_DISABLE_BACKPRESSURE_TEST="${TURBOPUFFER_DISABLE_BACKPRESSURE_TEST:-1}"

is_protected_pmc_namespace() {
  local ns="$1"
  local configured_pmc="${TURBOPUFFER_NAMESPACE_PMC:-medical_database_pmc}"
  if [[ -z "${ns}" ]]; then
    return 1
  fi
  if [[ "${ns}" == "medical_database_pmc" || "${ns}" == "medical_pmc" || "${ns}" == "${configured_pmc}" ]]; then
    return 0
  fi
  if [[ "${ns}" == "${configured_pmc}"_shard_* || "${ns}" == "medical_database_pmc_shard_"* ]]; then
    return 0
  fi
  return 1
}

assert_not_protected_namespace() {
  local ns="$1"
  local label="$2"
  if is_protected_pmc_namespace "${ns}"; then
    echo "Refusing to use ${label}='${ns}' because it targets protected PMC production namespace." >&2
    exit 2
  fi
}

assert_not_protected_namespace "${NS_A}" "DAILYMED_TEST_NAMESPACE_A"
assert_not_protected_namespace "${NS_B}" "DAILYMED_TEST_NAMESPACE_B"

run_logged() {
  local name="$1"
  shift
  local logfile="${LOG_DIR}/${name}.log"
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] Running ${name}"
  "$@" 2>&1 | tee "${logfile}"
}

run_ingest_test_namespace() {
  local namespace="$1"
  local checkpoint_file="$2"
  local audit_file="$3"
  local run_name="$4"

  rm -f "${checkpoint_file}" "${audit_file}"
  run_logged "${run_name}" env \
    DEEPINFRA_API_KEY="${DEEPINFRA_KEY}" \
    DEEPINFRA_API_KEYS="${DEEPINFRA_KEY}" \
    MAX_WORKERS="${MAX_WORKERS_TEST}" \
    DAILYMED_FILE_BATCH_SIZE="${DAILYMED_FILE_BATCH_SIZE_TEST}" \
    EMBEDDING_CONCURRENCY="${EMBEDDING_CONCURRENCY_TEST}" \
    EMBEDDING_MAX_REQUESTS_PER_SEC="${EMBEDDING_MAX_REQUESTS_PER_SEC_TEST}" \
    TURBOPUFFER_MAX_CONCURRENT_WRITES="${TURBOPUFFER_MAX_CONCURRENT_WRITES_TEST}" \
    TURBOPUFFER_DISABLE_BACKPRESSURE="${TURBOPUFFER_DISABLE_BACKPRESSURE_TEST}" \
    TURBOPUFFER_NAMESPACE_DAILYMED="${namespace}" \
    python3 scripts/07_ingest_dailymed.py \
      --xml-dir "${DAILYMED_XML_DIR}" \
      --max-files "${TEST_LABEL_COUNT}" \
      --namespace "${namespace}" \
      --checkpoint-file "${checkpoint_file}" \
      --audit-json "${audit_file}" \
      --file-batch-size "${DAILYMED_FILE_BATCH_SIZE_TEST}" \
      --max-workers "${MAX_WORKERS_TEST}" \
      --disable-backpressure
}

run_validate_namespace() {
  local namespace="$1"
  local checkpoint_file="$2"
  local audit_file="$3"
  local report_file="$4"
  local run_name="$5"

  run_logged "${run_name}" env \
    TURBOPUFFER_API_KEY="${TURBOPUFFER_API_KEY}" \
    python3 scripts/08_validate_dailymed_namespace.py \
      --namespace "${namespace}" \
      --expected-labels "${TEST_LABEL_COUNT}" \
      --audit-json "${audit_file}" \
      --report-json "${report_file}" \
      --checkpoint-file "${checkpoint_file}"
}

echo "=== DailyMed 1k Test Gate ==="
echo "Run timestamp: ${RUN_TS}"
echo "Data directory: ${DATA_DIR}"
echo "XML directory: ${DAILYMED_XML_DIR}"
echo "Test namespaces: ${NS_A}, ${NS_B}"

run_logged "download_test_xml" python3 scripts/03_download_dailymed.py --output-dir "${DAILYMED_XML_DIR}"

XML_COUNT="$(find "${DAILYMED_XML_DIR}" -name '*.xml' -type f | wc -l | tr -d ' ')"
if [[ "${XML_COUNT}" -lt "${TEST_LABEL_COUNT}" ]]; then
  echo "Not enough XML files for deterministic test set: found ${XML_COUNT}, expected >= ${TEST_LABEL_COUNT}" >&2
  exit 1
fi

run_ingest_test_namespace "${NS_A}" "${CHECKPOINT_A}" "${AUDIT_A}" "ingest_test_namespace_a"
run_validate_namespace "${NS_A}" "${CHECKPOINT_A}" "${AUDIT_A}" "${REPORT_A}" "validate_test_namespace_a"

run_ingest_test_namespace "${NS_B}" "${CHECKPOINT_B}" "${AUDIT_B}" "ingest_test_namespace_b"
run_validate_namespace "${NS_B}" "${CHECKPOINT_B}" "${AUDIT_B}" "${REPORT_B}" "validate_test_namespace_b"

run_logged "delete_test_namespace_a" env \
  TURBOPUFFER_API_KEY="${TURBOPUFFER_API_KEY}" \
  python3 scripts/09_delete_turbopuffer_namespace.py \
    --namespace "${NS_A}" \
    --confirm-delete \
    --report-json "${DELETE_REPORT_A}"

run_logged "delete_test_namespace_b" env \
  TURBOPUFFER_API_KEY="${TURBOPUFFER_API_KEY}" \
  python3 scripts/09_delete_turbopuffer_namespace.py \
    --namespace "${NS_B}" \
    --confirm-delete \
    --report-json "${DELETE_REPORT_B}"

cp -f "${AUDIT_A}" "${ARCHIVE_DIR}/dailymed_test_1000_a_audit.json"
cp -f "${AUDIT_B}" "${ARCHIVE_DIR}/dailymed_test_1000_b_audit.json"
rm -f "${CHECKPOINT_A}" "${CHECKPOINT_B}" "${AUDIT_A}" "${AUDIT_B}"

rm -rf "${DAILYMED_XML_DIR}"
mkdir -p "${DAILYMED_XML_DIR}"

if [[ "${START_PRODUCTION}" == "true" ]]; then
  assert_not_protected_namespace "${PRODUCTION_NAMESPACE}" "PRODUCTION_NAMESPACE"
  PROD_CHECKPOINT="${DAILYMED_CHECKPOINT_FILE:-${DATA_DIR}/dailymed_ingested_ids.txt}"
  run_logged "download_production_xml" python3 scripts/03_download_dailymed.py --output-dir "${DAILYMED_XML_DIR}"
  run_logged "ingest_production" env \
    DEEPINFRA_API_KEY="${DEEPINFRA_KEY}" \
    DEEPINFRA_API_KEYS="${DEEPINFRA_KEY}" \
    TURBOPUFFER_DISABLE_BACKPRESSURE="${TURBOPUFFER_DISABLE_BACKPRESSURE_TEST}" \
    TURBOPUFFER_MAX_CONCURRENT_WRITES="${TURBOPUFFER_MAX_CONCURRENT_WRITES_TEST}" \
    TURBOPUFFER_NAMESPACE_DAILYMED="${PRODUCTION_NAMESPACE}" \
    python3 scripts/07_ingest_dailymed.py \
      --xml-dir "${DAILYMED_XML_DIR}" \
      --namespace "${PRODUCTION_NAMESPACE}" \
      --checkpoint-file "${PROD_CHECKPOINT}" \
      --file-batch-size "${DAILYMED_FILE_BATCH_SIZE_TEST}" \
      --max-workers "${MAX_WORKERS_TEST}" \
      --disable-backpressure
else
  echo "Gate passed and test artifacts archived at: ${ARCHIVE_DIR}"
  echo "Test XMLs were deleted. Re-download is required before production ingestion."
  echo "Re-run with --start-production to automatically launch production ingestion."
fi
