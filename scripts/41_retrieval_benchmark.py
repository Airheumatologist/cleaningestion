#!/usr/bin/env python3
"""Benchmark dense and hybrid retrieval modes with a fixed query corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ModeMetrics:
    mode: str
    total_queries: int
    successes: int
    timeouts: int
    errors: int
    success_rate: float
    timeout_rate: float
    error_rate: float
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = (len(ordered) - 1) * percentile
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def _read_queries(path: Path) -> list[str]:
    queries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if candidate:
            queries.append(candidate)
    return queries


def _run_mode(retriever: Any, mode: str, query: str) -> int:
    if mode == "dense_broad":
        return len(retriever.retrieve_passages(query, use_hybrid=False))
    if mode == "dense_source_family_pmc":
        return len(retriever.retrieve_passages(query, use_hybrid=False, source_family="pmc"))
    if mode == "dense_source_family_pubmed":
        return len(retriever.retrieve_passages(query, use_hybrid=False, source_family="pubmed"))
    if mode == "hybrid_2query":
        queries = [query, f"{query} treatment"]
        sparse_vectors = retriever.build_sparse_query_vectors(queries)
        return len(retriever.batch_hybrid_search(queries=queries, sparse_vectors=sparse_vectors))
    raise ValueError(f"Unsupported benchmark mode: {mode}")


def benchmark_mode(retriever: Any, mode: str, queries: list[str]) -> tuple[ModeMetrics, list[dict[str, Any]]]:
    durations_ms: list[float] = []
    samples: list[dict[str, Any]] = []
    successes = 0
    timeouts = 0
    errors = 0

    for query in queries:
        started = time.perf_counter()
        status = "ok"
        result_count = 0
        error = ""

        try:
            result_count = _run_mode(retriever, mode, query)
            successes += 1
        except TimeoutError as exc:
            status = "timeout"
            error = str(exc)
            timeouts += 1
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error = str(exc)
            errors += 1

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if status == "ok":
            durations_ms.append(elapsed_ms)

        samples.append(
            {
                "mode": mode,
                "query": query,
                "status": status,
                "elapsed_ms": round(elapsed_ms, 3),
                "result_count": result_count,
                "error": error,
            }
        )

    total = len(queries)
    metrics = ModeMetrics(
        mode=mode,
        total_queries=total,
        successes=successes,
        timeouts=timeouts,
        errors=errors,
        success_rate=(successes / total) if total else 0.0,
        timeout_rate=(timeouts / total) if total else 0.0,
        error_rate=(errors / total) if total else 0.0,
        p50_ms=_percentile(durations_ms, 0.50),
        p95_ms=_percentile(durations_ms, 0.95),
        p99_ms=_percentile(durations_ms, 0.99),
    )
    return metrics, samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency across dense/hybrid modes")
    parser.add_argument("--queries-file", required=True, help="Path to newline-delimited query corpus")
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--collection-name", default="rag_pipeline")
    parser.add_argument("--n-retrieval", type=int, default=150)
    parser.add_argument("--min-queries", type=int, default=50)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    queries_file = Path(args.queries_file)
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    queries = _read_queries(queries_file)
    if not args.allow_small and len(queries) < args.min_queries:
        raise ValueError(
            f"Need at least {args.min_queries} queries (got {len(queries)}). "
            "Use --allow-small to bypass."
        )

    os.environ["QDRANT_URL"] = args.qdrant_url
    os.environ["COLLECTION_NAME"] = args.collection_name
    os.environ["QDRANT_COLLECTION"] = args.collection_name
    if args.qdrant_api_key:
        os.environ["QDRANT_API_KEY"] = args.qdrant_api_key

    from src.retriever_qdrant import QdrantRetriever

    retriever = QdrantRetriever(n_retrieval=args.n_retrieval)

    modes = [
        "dense_broad",
        "dense_source_family_pmc",
        "dense_source_family_pubmed",
        "hybrid_2query",
    ]

    all_samples: list[dict[str, Any]] = []
    mode_metrics: list[dict[str, Any]] = []

    for mode in modes:
        metrics, samples = benchmark_mode(retriever, mode=mode, queries=queries)
        mode_metrics.append(asdict(metrics))
        all_samples.extend(samples)

    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_count": len(queries),
        "collection_name": args.collection_name,
        "n_retrieval": args.n_retrieval,
        "modes": mode_metrics,
        "samples": all_samples,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()
