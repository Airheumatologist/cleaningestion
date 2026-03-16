#!/usr/bin/env python3
"""Phase 7 dual-read canary validator for Qdrant vs LanceDB retrieval."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from typing import Any

from src.retriever_lancedb import LanceDBRetriever
from src.retriever_qdrant import QdrantRetriever


def _default_queries() -> list[str]:
    return [
        "aspirin dose in older adults",
        "checkpoint inhibitor adverse effects",
        "rheumatoid arthritis biologics guideline",
        "metformin contraindications in renal disease",
        "hypertension treatment trial",
    ]


def _topk_ids(passages: list[dict[str, Any]], top_k: int) -> list[str]:
    ids: list[str] = []
    for p in passages[:top_k]:
        ids.append(str(p.get("chunk_id") or p.get("corpus_id") or p.get("pmcid") or ""))
    return [item for item in ids if item]


def _overlap_ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / float(len(set(a) | set(b)))


def _source_consistency(a: list[dict[str, Any]], b: list[dict[str, Any]], top_k: int) -> float:
    source_a = {str(p.get("source", "")) for p in a[:top_k]}
    source_b = {str(p.get("source", "")) for p in b[:top_k]}
    if not source_a and not source_b:
        return 1.0
    if not source_a or not source_b:
        return 0.0
    return len(source_a & source_b) / float(len(source_a | source_b))


def _load_queries(path: str | None) -> list[str]:
    if not path:
        return _default_queries()
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-read canary comparison for LanceDB cutover")
    parser.add_argument("--queries-file", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample-rate", type=float, default=0.25)
    parser.add_argument("--min-overlap", type=float, default=0.35)
    parser.add_argument("--min-source-consistency", type=float, default=0.70)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    all_queries = _load_queries(args.queries_file)
    sample_size = max(1, int(round(len(all_queries) * max(0.01, min(1.0, args.sample_rate)))))
    sampled_queries = random.sample(all_queries, min(sample_size, len(all_queries)))

    qdrant = QdrantRetriever(n_retrieval=max(50, args.top_k))
    lance = LanceDBRetriever(n_retrieval=max(50, args.top_k))

    overlaps: list[float] = []
    source_scores: list[float] = []
    per_query: list[dict[str, Any]] = []
    errors = 0

    for query in sampled_queries:
        try:
            q_results = qdrant.retrieve_passages(query, use_hybrid=True)
            l_results = lance.retrieve_passages(query, use_hybrid=True)
            q_ids = _topk_ids(q_results, args.top_k)
            l_ids = _topk_ids(l_results, args.top_k)
            overlap = _overlap_ratio(q_ids, l_ids)
            source_consistency = _source_consistency(q_results, l_results, args.top_k)
            overlaps.append(overlap)
            source_scores.append(source_consistency)
            per_query.append(
                {
                    "query": query,
                    "overlap": overlap,
                    "source_consistency": source_consistency,
                    "qdrant_count": len(q_results),
                    "lancedb_count": len(l_results),
                }
            )
        except Exception as exc:
            errors += 1
            per_query.append({"query": query, "error": str(exc)})

    avg_overlap = statistics.mean(overlaps) if overlaps else 0.0
    avg_source_consistency = statistics.mean(source_scores) if source_scores else 0.0

    canary_pass = (
        errors == 0
        and avg_overlap >= args.min_overlap
        and avg_source_consistency >= args.min_source_consistency
    )

    result = {
        "status": "ok" if canary_pass else "failed",
        "sampled_queries": len(sampled_queries),
        "errors": errors,
        "avg_topk_overlap": avg_overlap,
        "avg_source_consistency": avg_source_consistency,
        "thresholds": {
            "min_overlap": args.min_overlap,
            "min_source_consistency": args.min_source_consistency,
        },
        "rollout_plan": [1, 5, 25, 50, 100],
        "rollback_ready": True,
        "details": per_query,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)


if __name__ == "__main__":
    main()
