#!/usr/bin/env python3
"""Delete a turbopuffer namespace and verify post-delete empty state."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import turbopuffer as tpuf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_ingestion import IngestionConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _protected_namespace_reason(namespace: str) -> str | None:
    target = namespace.strip()
    if not target:
        return "Namespace is empty"

    configured_pmc = IngestionConfig.TURBOPUFFER_NAMESPACE_PMC.strip() or "medical_database_pmc"
    hard_protected = {"medical_database_pmc", "medical_pmc", configured_pmc}
    if target in hard_protected:
        return (
            f"Namespace '{target}' is protected (PMC production namespace) "
            "and cannot be deleted."
        )

    if target.startswith(f"{configured_pmc}_shard_"):
        return (
            f"Namespace '{target}' matches protected PMC shard pattern "
            f"'{configured_pmc}_shard_*' and cannot be deleted."
        )

    if target.startswith("medical_database_pmc_shard_"):
        return (
            f"Namespace '{target}' matches protected PMC shard pattern "
            "'medical_database_pmc_shard_*' and cannot be deleted."
        )

    return None


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
    return {}


def _response_has_rows(response: Any) -> bool:
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
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete turbopuffer namespace with explicit confirmation")
    parser.add_argument("--namespace", required=True, help="Namespace to delete")
    parser.add_argument(
        "--confirm-delete",
        action="store_true",
        help="Required safety flag to perform deletion",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON deletion report",
    )
    args = parser.parse_args()

    if not args.confirm_delete:
        logger.error("Refusing to delete namespace without --confirm-delete")
        return 2

    if not IngestionConfig.TURBOPUFFER_API_KEY:
        logger.error("TURBOPUFFER_API_KEY is required")
        return 2

    report: Dict[str, Any] = {
        "namespace": args.namespace,
        "timestamp": int(time.time()),
        "deleted": False,
        "exists_before": False,
        "exists_after": False,
        "rows_after_delete": None,
        "status": "failed",
        "error": "",
    }

    protected_reason = _protected_namespace_reason(args.namespace)
    if protected_reason is not None:
        report["status"] = "blocked_protected_namespace"
        report["error"] = protected_reason
        logger.error(protected_reason)
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            with args.report_json.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=True, indent=2)
                handle.write("\n")
        return 2

    try:
        client = tpuf.Turbopuffer(
            api_key=IngestionConfig.TURBOPUFFER_API_KEY,
            region=IngestionConfig.TURBOPUFFER_REGION,
        )
        ns = client.namespace(args.namespace)
        exists_before = bool(ns.exists())
        report["exists_before"] = exists_before
        if exists_before:
            ns.delete_all()
            report["deleted"] = True
            logger.info("Issued delete_all for namespace=%s", args.namespace)
        else:
            logger.info("Namespace does not exist before deletion call: %s", args.namespace)

        time.sleep(1.0)
        exists_after = bool(ns.exists())
        report["exists_after"] = exists_after

        rows_after = 0
        if exists_after:
            recall_response = ns.recall(num=1, top_k=1)
            rows_after = 1 if _response_has_rows(recall_response) else 0
        report["rows_after_delete"] = rows_after

        if exists_after and rows_after > 0:
            report["status"] = "failed"
            report["error"] = "Namespace still contains rows after delete_all"
            logger.error(report["error"])
            return_code = 1
        else:
            report["status"] = "ok"
            logger.info(
                "Namespace deletion verified: namespace=%s exists_after=%s rows_after_delete=%s",
                args.namespace,
                exists_after,
                rows_after,
            )
            return_code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        logger.error("Namespace deletion failed: %s", exc)
        return_code = 1

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
