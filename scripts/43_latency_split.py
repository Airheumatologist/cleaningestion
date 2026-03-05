#!/usr/bin/env python3
"""Isolate embedding latency vs Qdrant HNSW search latency.

Run with --dummy-vector to bypass the embedding API entirely (random unit vector).
Compare the two modes to identify where the ~47s is actually spent.

Usage:
  # Qdrant-only (no embedding call, random unit vector):
  python scripts/43_latency_split.py --qdrant-url http://localhost:6333 --dummy-vector --n 5

  # Full stack (embedding API + Qdrant):
  python scripts/43_latency_split.py --qdrant-url http://localhost:6333 --n 5

  # Override embedding endpoint:
  python scripts/43_latency_split.py --qdrant-url http://localhost:6333 \
      --embedding-url https://api.deepinfra.com/v1/openai \
      --embedding-api-key sk-... --n 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] * (1 - (idx - lo)) + ordered[hi] * (idx - lo))


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "min": None, "max": None, "mean": None}
    return {
        "p50": round(_percentile(values, 0.50) or 0, 1),
        "p95": round(_percentile(values, 0.95) or 0, 1),
        "p99": round(_percentile(values, 0.99) or 0, 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "mean": round(sum(values) / len(values), 1),
    }


def _random_unit_vector(dim: int) -> list[float]:
    import random
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latency split: measure embedding ms and Qdrant search ms separately"
    )
    parser.add_argument("--qdrant-url", required=True, help="Qdrant HTTP URL (e.g. http://qdrant:6333)")
    parser.add_argument("--qdrant-api-key", default=None, help="Qdrant API key (default: env QDRANT_API_KEY)")
    parser.add_argument("--collection-name", default="rag_pipeline")
    parser.add_argument(
        "--embedding-url",
        default=None,
        help="OpenAI-compatible embedding base URL (default: env DEEPINFRA_BASE_URL)",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=None,
        help="Embedding API key (default: env DEEPINFRA_API_KEY or HF_INFERENCE_ENDPOINT_API_KEY)",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model name",
    )
    parser.add_argument("--dimension", type=int, default=1024, help="Vector dimension (default: 1024)")
    parser.add_argument("--hnsw-ef", type=int, default=32, help="HNSW ef search parameter (default: 32)")
    parser.add_argument("--limit", type=int, default=150, help="Number of results to retrieve (default: 150)")
    parser.add_argument(
        "--dummy-vector",
        action="store_true",
        help="Skip embedding API; use a random unit vector to isolate Qdrant latency",
    )
    parser.add_argument("--n", type=int, default=3, help="Number of repeated calls (default: 3)")
    parser.add_argument(
        "--query",
        default="statin therapy cardiovascular risk reduction",
        help="Query string for embedding (ignored with --dummy-vector)",
    )
    args = parser.parse_args()

    qdrant_api_key = args.qdrant_api_key or os.getenv("QDRANT_API_KEY")
    embedding_url = args.embedding_url or os.getenv(
        "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"
    )
    embedding_api_key = args.embedding_api_key or os.getenv("DEEPINFRA_API_KEY") or os.getenv(
        "HF_INFERENCE_ENDPOINT_API_KEY"
    )

    from qdrant_client import QdrantClient
    from qdrant_client.models import SearchParams, QuantizationSearchParams

    qdrant = QdrantClient(
        url=args.qdrant_url,
        api_key=qdrant_api_key,
        timeout=120,
        prefer_grpc=False,  # HTTP only for simplicity
    )

    openai_client = None
    if not args.dummy_vector:
        from openai import OpenAI
        openai_client = OpenAI(
            api_key=embedding_api_key or "dummy",
            base_url=embedding_url,
            timeout=120.0,
        )

    instruction = (
        "Instruct: Given a medical question, retrieve relevant clinical passages "
        "that answer the query\nQuery: "
    )
    full_query = instruction + args.query

    search_params = SearchParams(
        hnsw_ef=args.hnsw_ef,
        quantization=QuantizationSearchParams(rescore=True, oversampling=2.0),
    )

    embed_times: list[float] = []
    qdrant_times: list[float] = []
    calls: list[dict[str, Any]] = []

    for i in range(args.n):
        call: dict[str, Any] = {"call": i + 1}

        # --- Embedding phase ---
        if args.dummy_vector:
            vector = _random_unit_vector(args.dimension)
            call["embed_ms"] = 0.0
        else:
            t0 = time.perf_counter()
            resp = openai_client.embeddings.create(  # type: ignore[union-attr]
                model=args.embedding_model,
                input=[full_query],
                encoding_format="float",
            )
            embed_ms = (time.perf_counter() - t0) * 1000.0
            vector = resp.data[0].embedding
            call["embed_ms"] = round(embed_ms, 1)
            embed_times.append(embed_ms)

        # --- Qdrant search phase ---
        t1 = time.perf_counter()
        results = qdrant.query_points(
            collection_name=args.collection_name,
            query=vector,
            using="dense",
            limit=args.limit,
            with_payload=False,
            search_params=search_params,
        )
        qdrant_ms = (time.perf_counter() - t1) * 1000.0
        call["qdrant_ms"] = round(qdrant_ms, 1)
        call["result_count"] = len(results.points)
        qdrant_times.append(qdrant_ms)

        call["total_ms"] = round(call["embed_ms"] + qdrant_ms, 1)
        calls.append(call)

    total_times = [c["total_ms"] for c in calls]

    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "dummy_vector" if args.dummy_vector else "real_embedding",
        "n": args.n,
        "collection": args.collection_name,
        "hnsw_ef": args.hnsw_ef,
        "limit": args.limit,
        "embed_ms": _stats(embed_times) if not args.dummy_vector else "skipped",
        "qdrant_ms": _stats(qdrant_times),
        "total_ms": _stats(total_times),
        "calls": calls,
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
