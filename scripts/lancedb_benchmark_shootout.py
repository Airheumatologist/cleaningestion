#!/usr/bin/env python3
"""Phase 5 benchmark harness: IVF_RQ vs IVF_HNSW_SQ."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import lancedb
from lancedb.rerankers import RRFReranker

from lancedb_index_manager import _build_indexes, _load_manifest


def _sample_rows(total: int = 2000) -> list[dict[str, Any]]:
    random.seed(17)
    rows: list[dict[str, Any]] = []
    for idx in range(total):
        family = "pmc" if idx % 3 == 0 else ("pubmed" if idx % 3 == 1 else "dailymed")
        source = "pmc_oa" if family == "pmc" else ("pubmed_abstract" if family == "pubmed" else "dailymed")
        article_type = "trial" if family == "pmc" else ("review" if family == "pubmed" else "drug_label")
        y = 2018 + (idx % 8)
        base = 0.1 if family == "pmc" else 0.5 if family == "pubmed" else 0.9
        rows.append(
            {
                "point_id": f"pt-{idx}",
                "doc_id": f"doc-{idx}",
                "pmcid": f"PMC{idx}",
                "chunk_id": f"chunk-{idx}",
                "page_content": f"clinical aspirin query token {idx}",
                "source": source,
                "source_family": family,
                "year": y,
                "article_type": article_type,
                "vector": [base + random.random() * 0.05 for _ in range(16)],
            }
        )
    return rows


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((p / 100.0) * (len(values) - 1)))))
    return values[idx]


def _run_query(table: Any, query_vector: list[float], query_text: str, where_sql: str | None, top_k: int) -> tuple[float, int]:
    start = time.perf_counter()
    search = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(query_text)
        .rerank(reranker=RRFReranker())
        .limit(top_k)
    )
    if where_sql:
        search = search.where(where_sql, prefilter=True)
    rows = search.to_list()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, len(rows)


def _benchmark_profile(table: Any, profile: str, manifest: dict[str, Any], concurrency: int, queries: list[dict[str, Any]]) -> dict[str, Any]:
    _build_indexes(table, manifest, profile)

    latencies: list[float] = []
    rows_returned: list[int] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _run_query,
                table,
                q["vector"],
                q["text"],
                q.get("where"),
                q.get("top_k", 20),
            )
            for q in queries
        ]
        for future in as_completed(futures):
            try:
                latency_ms, count = future.result()
                latencies.append(latency_ms)
                rows_returned.append(count)
            except Exception:
                errors += 1

    successful = len(latencies)
    throughput_qps = successful / (sum(latencies) / 1000.0) if latencies else 0.0

    return {
        "profile": profile,
        "concurrency": concurrency,
        "queries": len(queries),
        "successful": successful,
        "errors": errors,
        "error_rate": errors / max(1, len(queries)),
        "rows_avg": statistics.mean(rows_returned) if rows_returned else 0.0,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "mean": statistics.mean(latencies) if latencies else 0.0,
        },
        "throughput_qps": throughput_qps,
    }


def _weighted_score(result: dict[str, Any]) -> float:
    # Lower is better: weighted latency + error penalty.
    p95 = result["latency_ms"]["p95"]
    p99 = result["latency_ms"]["p99"]
    err = result["error_rate"]
    qps = result["throughput_qps"]
    return (0.55 * p95) + (0.25 * p99) + (1000.0 * err) - (0.2 * qps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark IVF_RQ vs IVF_HNSW_SQ for LanceDB migration")
    parser.add_argument("--manifest", default="schema/lancedb_index_profiles.json")
    parser.add_argument("--uri", default="./medical_data.lancedb")
    parser.add_argument("--table", default="medical_docs")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.manifest))
    profiles = ["ivf_rq", "ivf_hnsw_sq"]

    if args.demo:
        with tempfile.TemporaryDirectory(prefix="lancedb-bench-") as tmp:
            db = lancedb.connect(tmp)
            table = db.create_table(args.table, data=_sample_rows(), mode="overwrite")
            result = _run_benchmark(table, manifest, profiles, args.queries, args.concurrency)
    else:
        db = lancedb.connect(args.uri)
        table = db.open_table(args.table)
        result = _run_benchmark(table, manifest, profiles, args.queries, args.concurrency)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)


def _run_benchmark(table: Any, manifest: dict[str, Any], profiles: list[str], query_count: int, concurrency: int) -> dict[str, Any]:
    workloads: list[dict[str, Any]] = []
    for idx in range(query_count):
        where = None
        if idx % 2 == 0:
            where = "source_family IN ('pmc','pubmed') AND year >= 2020"
        workloads.append(
            {
                "vector": [0.12 + ((idx % 5) * 0.01) for _ in range(16)],
                "text": "aspirin dose clinical trial",
                "where": where,
                "top_k": 20,
            }
        )

    results = [
        _benchmark_profile(table, profile, manifest, concurrency, workloads)
        for profile in profiles
    ]

    for item in results:
        item["weighted_score"] = _weighted_score(item)

    selected = sorted(results, key=lambda x: x["weighted_score"])[0]

    return {
        "status": "ok",
        "profiles": results,
        "selected_profile": selected["profile"],
        "selection_reason": "lowest_weighted_score",
    }


if __name__ == "__main__":
    main()
