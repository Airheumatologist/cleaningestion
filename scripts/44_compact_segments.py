#!/usr/bin/env python3
"""Force Qdrant segment compaction and monitor progress.

The collection currently has 33 segments, causing 33 separate HNSW graph
traversals per query. This script configures the Qdrant optimizer to merge
them down to a target number of segments.

Usage:
  # Dry-run: show current state and what would be applied
  python scripts/44_compact_segments.py --qdrant-url http://localhost:6333

  # Apply compaction (targets 5 segments):
  python scripts/44_compact_segments.py --qdrant-url http://localhost:6333 --apply

  # Apply and monitor until done:
  python scripts/44_compact_segments.py --qdrant-url http://localhost:6333 \
      --apply --monitor --poll-interval 30 --timeout 7200

Notes:
  - The Qdrant optimizer runs in the background; this script triggers it.
  - Compaction on 22M vectors typically takes 20-90 minutes.
  - Reads are NOT blocked during compaction; queries continue to work.
  - max_segment_size = ceil(points_count / target_segments * 1.1) [vector count]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _to_jsonable(value.dict())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _get_segment_count(info: Any) -> int:
    """Extract segment count from collection info."""
    segments_count = getattr(info, "segments_count", None)
    if segments_count is not None:
        return int(segments_count)
    # Fallback: check config
    config = getattr(info, "config", None)
    if config:
        optimizers = getattr(config, "optimizer_config", None)
        if optimizers:
            return getattr(optimizers, "segments_count", 0) or 0
    return -1


def _get_optimizer_status(info: Any) -> str:
    status = getattr(info, "optimizer_status", None)
    if status is None:
        return "unknown"
    ok = getattr(status, "ok", None)
    if ok is True:
        return "ok"
    if ok is False:
        error = getattr(status, "error", "")
        return f"error: {error}"
    return str(status)


def _snapshot(client: Any, collection_name: str) -> dict[str, Any]:
    info = client.get_collection(collection_name)
    return {
        "segments_count": _get_segment_count(info),
        "points_count": getattr(info, "points_count", None),
        "vectors_count": getattr(info, "vectors_count", None),
        "optimizer_status": _get_optimizer_status(info),
        "status": str(getattr(info, "status", "unknown")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force Qdrant segment compaction to reduce 33 segments to a target count"
    )
    parser.add_argument("--qdrant-url", required=True, help="Qdrant HTTP URL (e.g. http://qdrant:6333)")
    parser.add_argument("--qdrant-api-key", default=None, help="Qdrant API key (default: env QDRANT_API_KEY)")
    parser.add_argument("--collection-name", default="rag_pipeline")
    parser.add_argument(
        "--target-segments",
        type=int,
        default=5,
        help="Target segment count after compaction (default: 5)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the optimizer config change (default is dry-run)",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="After applying, poll until segment count reaches target (requires --apply)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status polls when monitoring (default: 30)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Max seconds to monitor before giving up (default: 7200)",
    )
    args = parser.parse_args()

    qdrant_api_key = args.qdrant_api_key or os.getenv("QDRANT_API_KEY")

    from qdrant_client import QdrantClient
    from qdrant_client.models import OptimizersConfigDiff

    client = QdrantClient(
        url=args.qdrant_url,
        api_key=qdrant_api_key,
        timeout=300,
        prefer_grpc=False,
    )

    before = _snapshot(client, args.collection_name)
    points_count = before["points_count"] or 0

    # max_segment_size: vector count per segment with 10% headroom so we land at target
    import math
    max_segment_size = math.ceil(points_count / args.target_segments * 1.1) if points_count else 5_000_000

    result: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_name": args.collection_name,
        "dry_run": not args.apply,
        "before": before,
        "optimizer_config_to_apply": {
            "max_segment_size": max_segment_size,
            "target_segments": args.target_segments,
            "rationale": f"{points_count} vectors / {args.target_segments} segments × 1.1 headroom",
        },
    }

    if not args.apply:
        result["action"] = "dry_run_only"
        print(json.dumps(result, indent=2))
        return

    # Apply the optimizer config
    client.update_collection(
        collection_name=args.collection_name,
        optimizers_config=OptimizersConfigDiff(
            max_segment_size=max_segment_size,
            deleted_threshold=0.2,
            vacuum_min_vector_number=100,
        ),
    )
    result["action"] = "applied"
    result["applied_at_utc"] = datetime.now(timezone.utc).isoformat()

    if not args.monitor:
        after = _snapshot(client, args.collection_name)
        result["after_immediate"] = after
        print(json.dumps(result, indent=2))
        return

    # Monitor until segment count drops to target or timeout
    polls: list[dict[str, Any]] = []
    deadline = time.monotonic() + args.timeout
    print(f"Monitoring compaction (target ≤{args.target_segments} segments, "
          f"poll every {args.poll_interval}s, timeout {args.timeout}s)...", flush=True)

    while time.monotonic() < deadline:
        time.sleep(args.poll_interval)
        snap = _snapshot(client, args.collection_name)
        snap["elapsed_s"] = round(time.monotonic() - (deadline - args.timeout), 1)
        polls.append(snap)
        seg = snap["segments_count"]
        status = snap["optimizer_status"]
        print(
            f"  [{snap['elapsed_s']}s] segments={seg}  optimizer={status}",
            flush=True,
        )
        if seg != -1 and seg <= args.target_segments:
            print(f"  Compaction complete: {seg} segments (target was ≤{args.target_segments})")
            break
    else:
        print(f"  Timeout after {args.timeout}s — compaction still in progress")

    result["monitor_polls"] = polls
    result["final"] = polls[-1] if polls else None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
