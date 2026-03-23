#!/usr/bin/env python3
"""Validate DailyMed test namespace ingestion results from audit + turbopuffer state."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import turbopuffer as tpuf

sys.path.insert(0, str(Path(__file__).parent))
from config_ingestion import IngestionConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_REQUIRED_NONEMPTY_FIELDS = (
    "doc_id",
    "chunk_id",
    "set_id",
    "page_content",
    "title",
    "source",
    "source_family",
    "drug_name",
    "section_type",
)


def _as_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump(mode="python")
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(response, "to_dict"):
        try:
            dumped = response.to_dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {}


def _iter_candidate_sequences(data: Any) -> Iterable[Any]:
    if isinstance(data, dict):
        for value in data.values():
            yield value
            yield from _iter_candidate_sequences(value)
    elif isinstance(data, list):
        yield data
        for value in data:
            yield from _iter_candidate_sequences(value)


def _response_has_rows(response: Any) -> bool:
    # recall() responses expose aggregate counts even when no row list is returned.
    for attr_name in ("avg_ann_count", "avg_exhaustive_count"):
        attr_value = getattr(response, attr_name, None)
        if isinstance(attr_value, (int, float)) and attr_value > 0:
            return True

    ids = getattr(response, "ids", None)
    if ids is not None:
        try:
            return len(ids) > 0
        except Exception:
            pass

    rows = getattr(response, "rows", None)
    if isinstance(rows, list):
        return len(rows) > 0

    dumped = _as_dict(response)
    for key in ("ids", "rows", "results", "matches"):
        value = dumped.get(key)
        if isinstance(value, list) and value:
            return True
    for key in ("avg_ann_count", "avg_exhaustive_count"):
        value = dumped.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return True
    for value in _iter_candidate_sequences(dumped):
        if isinstance(value, list) and value:
            return True
    return False


def _normalize_checkpoint_line(value: str) -> str | None:
    line = value.strip()
    if not line:
        return None
    if line.startswith("dailymed:"):
        return line
    if ":" in line:
        return None
    return f"dailymed:{line}"


def _load_checkpoint_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    checkpoint_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            normalized = _normalize_checkpoint_line(raw_line)
            if normalized:
                checkpoint_ids.add(normalized)
    return checkpoint_ids


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return True


def _validate_required_fields(
    rows: List[Dict[str, Any]],
    required_fields: Iterable[str],
    max_failures: int = 25,
) -> List[str]:
    failures: List[str] = []
    required = tuple(required_fields)
    for idx, row in enumerate(rows):
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            failures.append(f"row[{idx}] missing payload dict")
            if len(failures) >= max_failures:
                break
            continue
        for field in required:
            if not _is_nonempty(payload.get(field)):
                failures.append(
                    f"row[{idx}] field '{field}' is empty"
                    f" (set_id={payload.get('set_id', '')}, chunk_id={payload.get('chunk_id', '')})"
                )
                if len(failures) >= max_failures:
                    break
        if len(failures) >= max_failures:
            break
    return failures


def _validate_source_fields(rows: List[Dict[str, Any]], max_failures: int = 25) -> List[str]:
    failures: List[str] = []
    expected = {
        "source": "dailymed",
        "source_family": "dailymed",
    }
    for idx, row in enumerate(rows):
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        for key, expected_value in expected.items():
            actual = str(payload.get(key, "")).strip()
            if actual != expected_value:
                failures.append(
                    f"row[{idx}] field '{key}' expected '{expected_value}' got '{actual}'"
                    f" (set_id={payload.get('set_id', '')})"
                )
                if len(failures) >= max_failures:
                    return failures
    return failures


def _validate_vector_fields(rows: List[Dict[str, Any]], expected_dim: int, max_failures: int = 25) -> List[str]:
    failures: List[str] = []
    for idx, row in enumerate(rows):
        is_list = bool(row.get("vector_is_dense_list"))
        vector_dim = row.get("vector_dim")
        if not is_list:
            failures.append(f"row[{idx}] vector is not a dense list")
        if not isinstance(vector_dim, int) or vector_dim <= 0:
            failures.append(f"row[{idx}] invalid vector_dim={vector_dim}")
        elif vector_dim != expected_dim:
            failures.append(f"row[{idx}] vector_dim={vector_dim} expected={expected_dim}")
        if len(failures) >= max_failures:
            break
    return failures


def _add_check(checks: List[Dict[str, Any]], name: str, passed: bool, details: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DailyMed namespace ingestion quality gates")
    parser.add_argument("--namespace", required=True, help="Target turbopuffer namespace to validate")
    parser.add_argument("--expected-labels", type=int, default=1000, help="Expected number of labels checkpointed")
    parser.add_argument("--audit-json", type=Path, required=True, help="Audit JSON from 07_ingest_dailymed.py")
    parser.add_argument("--report-json", type=Path, required=True, help="Path to write validation report JSON")
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
        help="Optional checkpoint file override (defaults to checkpoint path in audit summary)",
    )
    args = parser.parse_args()

    checks: List[Dict[str, Any]] = []
    failures: List[str] = []

    if not args.audit_json.exists():
        logger.error("Audit JSON not found: %s", args.audit_json)
        return 2

    with args.audit_json.open("r", encoding="utf-8") as handle:
        audit_data = json.load(handle)

    summary = audit_data.get("summary", {}) if isinstance(audit_data, dict) else {}
    rows = audit_data.get("rows", []) if isinstance(audit_data, dict) else []
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(rows, list):
        rows = []

    required_fields = summary.get("required_nonempty_fields")
    if not isinstance(required_fields, list) or not required_fields:
        required_fields = list(DEFAULT_REQUIRED_NONEMPTY_FIELDS)
    expected_vector_size = int(summary.get("expected_vector_size", IngestionConfig.get_vector_size()))
    labels_attempted = int(summary.get("labels_attempted", 0))
    labels_checkpointed = int(summary.get("labels_checkpointed", 0))

    _add_check(
        checks,
        "audit_rows_present",
        len(rows) > 0,
        f"rows_audited={len(rows)}",
    )
    if len(rows) == 0:
        failures.append("No audit rows found")

    namespace_in_audit = str(summary.get("namespace", "")).strip()
    namespace_matches = namespace_in_audit == args.namespace
    _add_check(
        checks,
        "namespace_matches_audit",
        namespace_matches,
        f"audit_namespace={namespace_in_audit} expected={args.namespace}",
    )
    if not namespace_matches:
        failures.append(
            f"Namespace mismatch: audit namespace '{namespace_in_audit}' != expected '{args.namespace}'"
        )

    field_failures = _validate_required_fields(rows, required_fields)
    _add_check(
        checks,
        "required_fields_nonempty",
        len(field_failures) == 0,
        f"missing_or_empty_fields={len(field_failures)}",
    )
    failures.extend(field_failures)

    source_failures = _validate_source_fields(rows)
    _add_check(
        checks,
        "source_fields_match",
        len(source_failures) == 0,
        f"source_field_mismatches={len(source_failures)}",
    )
    failures.extend(source_failures)

    vector_failures = _validate_vector_fields(rows, expected_vector_size)
    _add_check(
        checks,
        "vector_shape_valid",
        len(vector_failures) == 0,
        f"vector_validation_failures={len(vector_failures)} expected_dim={expected_vector_size}",
    )
    failures.extend(vector_failures)

    attempted_ok = labels_attempted == args.expected_labels
    checkpointed_ok = labels_checkpointed == args.expected_labels
    _add_check(
        checks,
        "label_counts_from_audit",
        attempted_ok and checkpointed_ok,
        (
            f"labels_attempted={labels_attempted} labels_checkpointed={labels_checkpointed} "
            f"expected={args.expected_labels}"
        ),
    )
    if not attempted_ok:
        failures.append(f"labels_attempted={labels_attempted} expected={args.expected_labels}")
    if not checkpointed_ok:
        failures.append(f"labels_checkpointed={labels_checkpointed} expected={args.expected_labels}")

    checkpoint_file = args.checkpoint_file
    if checkpoint_file is None:
        checkpoint_from_audit = str(summary.get("checkpoint_file", "")).strip()
        checkpoint_file = Path(checkpoint_from_audit) if checkpoint_from_audit else None
    if checkpoint_file is None:
        _add_check(checks, "checkpoint_file_available", False, "checkpoint path missing")
        failures.append("Checkpoint file path is missing")
        checkpoint_count = 0
    else:
        checkpoint_ids = _load_checkpoint_ids(checkpoint_file)
        checkpoint_count = len(checkpoint_ids)
        checkpoint_ok = checkpoint_count == args.expected_labels
        _add_check(
            checks,
            "checkpoint_count_matches_expected",
            checkpoint_ok,
            f"checkpoint_count={checkpoint_count} expected={args.expected_labels} file={checkpoint_file}",
        )
        if not checkpoint_ok:
            failures.append(
                f"Checkpoint count mismatch: checkpoint_count={checkpoint_count} expected={args.expected_labels}"
            )

    ns_exists = False
    schema_readable = False
    namespace_has_rows = False
    approx_row_count = 0
    unindexed_bytes = 0
    index_status = "unknown"
    try:
        if not IngestionConfig.TURBOPUFFER_API_KEY:
            raise RuntimeError("TURBOPUFFER_API_KEY is required")
        client = tpuf.Turbopuffer(
            api_key=IngestionConfig.TURBOPUFFER_API_KEY,
            region=IngestionConfig.TURBOPUFFER_REGION,
        )
        ns = client.namespace(args.namespace)
        ns_exists = bool(ns.exists())
        if ns_exists:
            _ = ns.schema()
            schema_readable = True
            metadata = ns.metadata()
            approx_row_count = int(getattr(metadata, "approx_row_count", 0) or 0)
            namespace_has_rows = approx_row_count > 0
            index = getattr(metadata, "index", None)
            index_status = str(getattr(index, "status", "unknown"))
            unindexed_bytes = int(getattr(index, "unindexed_bytes", 0) or 0)
    except Exception as exc:
        failures.append(f"Namespace validation failed: {exc}")

    _add_check(checks, "namespace_exists", ns_exists, f"namespace={args.namespace}")
    _add_check(checks, "namespace_schema_readable", schema_readable, f"namespace={args.namespace}")
    _add_check(
        checks,
        "namespace_rows_present",
        namespace_has_rows,
        (
            f"namespace={args.namespace} approx_row_count={approx_row_count} "
            f"index_status={index_status} unindexed_bytes={unindexed_bytes}"
        ),
    )

    if not ns_exists:
        failures.append(f"Namespace does not exist: {args.namespace}")
    if not schema_readable:
        failures.append(f"Namespace schema unreadable: {args.namespace}")
    if not namespace_has_rows:
        failures.append(f"Namespace has no rows: {args.namespace}")

    passed = len(failures) == 0
    report = {
        "passed": passed,
        "timestamp": int(time.time()),
        "namespace": args.namespace,
        "expected_labels": int(args.expected_labels),
        "audit_json": str(args.audit_json),
        "report_json": str(args.report_json),
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
        "stats": {
            "rows_audited": len(rows),
            "expected_vector_size": expected_vector_size,
            "labels_attempted": labels_attempted,
            "labels_checkpointed": labels_checkpointed,
            "checkpoint_count": checkpoint_count,
            "namespace_approx_row_count": approx_row_count,
            "namespace_index_status": index_status,
            "namespace_unindexed_bytes": unindexed_bytes,
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    if passed:
        logger.info("Validation PASSED for namespace=%s", args.namespace)
        return 0

    logger.error("Validation FAILED for namespace=%s", args.namespace)
    for failure in failures[:20]:
        logger.error(" - %s", failure)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
