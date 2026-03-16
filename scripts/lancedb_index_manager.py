#!/usr/bin/env python3
"""Phase 3 LanceDB index manager: build/status/validate for dense+sparse profiles."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import lancedb

EXPECTED_LANCEDB_VERSION = "0.29.2"
DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "schema" / "lancedb_index_profiles.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"default_profile", "vector_column", "metric", "fts_columns", "scalar_filter_indexes", "profiles"}
    missing = required - set(data.keys())
    if missing:
        raise RuntimeError(f"Manifest missing required keys: {sorted(missing)}")
    return data


def _divisors(n: int) -> list[int]:
    vals: list[int] = []
    for i in range(1, n + 1):
        if n % i == 0:
            vals.append(i)
    return vals


def _pick_num_sub_vectors(target: int | None, dim: int) -> int | None:
    if dim <= 0:
        return None
    if target is None or target <= 0:
        return None
    candidates = [d for d in _divisors(dim) if d <= target]
    if not candidates:
        return None
    return max(candidates)


def _sample_rows() -> list[dict[str, Any]]:
    return [
        {
            "vector": [0.11, 0.22, 0.33, 0.44],
            "page_content": "Aspirin dose in adults with fever",
            "source": "pubmed_abstract",
            "source_family": "pubmed",
            "year": 2024,
            "article_type": "review",
            "doc_id": "pmid-1",
        },
        {
            "vector": [0.12, 0.18, 0.37, 0.43],
            "page_content": "Aspirin contraindications and renal cautions",
            "source": "dailymed",
            "source_family": "dailymed",
            "year": 2023,
            "article_type": "drug_label",
            "doc_id": "setid-2",
        },
        {
            "vector": [0.81, 0.74, 0.63, 0.59],
            "page_content": "Checkpoint inhibitor outcomes in metastatic melanoma",
            "source": "pmc_oa",
            "source_family": "pmc",
            "year": 2022,
            "article_type": "trial",
            "doc_id": "pmc-3",
        },
    ]


def _ensure_version() -> str:
    version = getattr(lancedb, "__version__", "unknown")
    if version != EXPECTED_LANCEDB_VERSION:
        raise RuntimeError(f"Expected lancedb=={EXPECTED_LANCEDB_VERSION}, found {version}")
    return version


def _connect_and_table(uri: str, table_name: str):
    db = lancedb.connect(uri)
    table = db.open_table(table_name)
    return db, table


def _wait_for_indices(table: Any) -> list[str]:
    names = [index.name for index in table.list_indices()]
    if names:
        table.wait_for_index(names, timeout=timedelta(seconds=300))
    return names


def _vector_dim(table: Any, vector_column: str) -> int:
    head = table.head(1).to_pylist()
    if not head:
        raise RuntimeError("Table has no rows; cannot infer vector dimension.")
    vector = head[0].get(vector_column)
    if not isinstance(vector, list) or not vector:
        raise RuntimeError(f"Vector column '{vector_column}' missing or empty.")
    return len(vector)


def _build_indexes(table: Any, manifest: dict[str, Any], profile_name: str) -> dict[str, Any]:
    if profile_name not in manifest["profiles"]:
        raise RuntimeError(f"Unknown profile '{profile_name}'. Options: {sorted(manifest['profiles'].keys())}")

    vector_column = manifest["vector_column"]
    metric = manifest["metric"]
    prefilter_default = bool(manifest.get("prefilter_default", True))
    dense_profile = dict(manifest["profiles"][profile_name])
    index_type = dense_profile.pop("index_type")

    for col in manifest["fts_columns"]:
        table.create_fts_index(col, replace=True)
    for col in manifest["scalar_filter_indexes"]:
        table.create_scalar_index(col, replace=True)

    row_count = table.count_rows()
    dim = _vector_dim(table, vector_column)

    requested_num_sub_vectors = dense_profile.get("num_sub_vectors")
    effective_num_sub_vectors = _pick_num_sub_vectors(requested_num_sub_vectors, dim)
    if "num_sub_vectors" in dense_profile:
        dense_profile["num_sub_vectors"] = effective_num_sub_vectors

    # For small tables, cap partitions so index build remains valid while preserving profile intent.
    requested_partitions = dense_profile.get("num_partitions")
    if isinstance(requested_partitions, int) and requested_partitions > 0:
        dense_profile["num_partitions"] = max(1, min(requested_partitions, max(1, row_count)))

    create_kwargs = {
        "metric": metric,
        "vector_column_name": vector_column,
        "index_type": index_type,
        "replace": True,
        **{k: v for k, v in dense_profile.items() if v is not None},
    }
    table.create_index(**create_kwargs)
    index_names = _wait_for_indices(table)

    return {
        "profile_name": profile_name,
        "index_type": index_type,
        "requested_num_sub_vectors": requested_num_sub_vectors,
        "effective_num_sub_vectors": effective_num_sub_vectors,
        "effective_params": create_kwargs,
        "index_names": index_names,
        "row_count": row_count,
        "vector_dim": dim,
        "prefilter_default": prefilter_default,
    }


def _validate_planner(table: Any, manifest: dict[str, Any], profile_name: str) -> dict[str, Any]:
    filter_expr = "source_family = 'pubmed'"
    plan = (
        table.search(query_type="hybrid")
        .vector([0.11, 0.22, 0.33, 0.44])
        .text("aspirin dose")
        .where(filter_expr, prefilter=True)
        .limit(5)
        .explain_plan()
    )
    has_ann = "ANNSubIndex" in plan
    has_fts = "MatchQuery" in plan
    has_filter = "FilterExec" in plan
    if not (has_ann and has_fts and has_filter):
        raise RuntimeError(
            "Planner validation failed. Expected ANNSubIndex + MatchQuery + FilterExec in explain plan."
        )

    return {
        "profile_name": profile_name,
        "filter_expr": filter_expr,
        "planner_has_ann": has_ann,
        "planner_has_fts": has_fts,
        "planner_has_filter": has_filter,
        "status": "ok",
    }


def _status(table: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    indices = table.list_indices()
    dense_index_types = [index.index_type for index in indices if "vector" in index.columns]
    return {
        "index_names": [index.name for index in indices],
        "index_types": [str(index.index_type) for index in indices],
        "dense_index_types": [str(t) for t in dense_index_types],
        "row_count": table.count_rows(),
        "manifest_default_profile": manifest["default_profile"],
        "status": "ok",
    }


def _run_with_table(args, op):
    if args.demo:
        with tempfile.TemporaryDirectory(prefix="lancedb-index-manager-") as tmp:
            db = lancedb.connect(tmp)
            table = db.create_table(args.table, data=_sample_rows(), mode="overwrite")
            return op(table)
    _db, table = _connect_and_table(args.uri, args.table)
    return op(table)


def _validate_with_optional_bootstrap(table: Any, manifest: dict[str, Any], profile: str, demo: bool) -> dict[str, Any]:
    if demo:
        _build_indexes(table, manifest, profile)
    return _validate_planner(table, manifest, profile)


def _status_with_optional_bootstrap(table: Any, manifest: dict[str, Any], profile: str, demo: bool) -> dict[str, Any]:
    if demo:
        _build_indexes(table, manifest, profile)
    return _status(table, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage LanceDB indexes for migration Phase 3.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to index profile manifest JSON.")
    parser.add_argument("--uri", default="./medical_data.lancedb", help="LanceDB URI.")
    parser.add_argument("--table", default="medical_docs", help="Table name.")
    parser.add_argument("--profile", default=None, help="Dense profile key from manifest.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--demo", action="store_true", help="Run against an ephemeral demo table.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Build FTS/scalar indexes and selected dense profile.")
    subparsers.add_parser("status", help="Show index status.")
    subparsers.add_parser("validate", help="Validate query planner index usage.")

    args = parser.parse_args()
    version = _ensure_version()
    manifest = _load_manifest(Path(args.manifest))
    profile = args.profile or manifest["default_profile"]

    if args.command == "build":
        result = _run_with_table(args, lambda table: _build_indexes(table, manifest, profile))
        result["status"] = "ok"
    elif args.command == "status":
        result = _run_with_table(
            args, lambda table: _status_with_optional_bootstrap(table, manifest, profile, args.demo)
        )
    else:
        result = _run_with_table(
            args, lambda table: _validate_with_optional_bootstrap(table, manifest, profile, args.demo)
        )

    result["lancedb_version"] = version
    result["manifest_path"] = str(Path(args.manifest).resolve())
    result["table"] = args.table
    result["profile"] = profile
    result["demo"] = bool(args.demo)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)


if __name__ == "__main__":
    main()
