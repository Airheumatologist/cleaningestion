#!/usr/bin/env python3
"""Read-only snapshot of Qdrant collection truth for latency tuning gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient


CRITICAL_PAYLOAD_INDEXES = [
    "source",
    "source_family",
    "article_type",
    "year",
    "journal",
]


@dataclass
class SnapshotSummary:
    collection_name: str
    captured_at_utc: str
    points_count: int | None
    indexed_vectors_count: int | None
    segments_count: int | None
    status: str
    missing_critical_indexes: list[str]


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


def collect_snapshot(client: QdrantClient, collection_name: str) -> dict[str, Any]:
    info = client.get_collection(collection_name)
    payload_schema = _to_jsonable(getattr(info, "payload_schema", {})) or {}
    payload_fields = sorted(payload_schema.keys()) if isinstance(payload_schema, dict) else []

    summary = SnapshotSummary(
        collection_name=collection_name,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        points_count=getattr(info, "points_count", None),
        indexed_vectors_count=getattr(info, "indexed_vectors_count", None),
        segments_count=getattr(info, "segments_count", None),
        status=str(getattr(info, "status", "unknown")),
        missing_critical_indexes=[
            field for field in CRITICAL_PAYLOAD_INDEXES if field not in payload_fields
        ],
    )

    return {
        "summary": asdict(summary),
        "collection_info": _to_jsonable(info),
        "payload_schema_fields": payload_fields,
        "critical_payload_indexes": {
            "required": CRITICAL_PAYLOAD_INDEXES,
            "present": [f for f in CRITICAL_PAYLOAD_INDEXES if f in payload_fields],
            "missing": [f for f in CRITICAL_PAYLOAD_INDEXES if f not in payload_fields],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Qdrant collection truth snapshot")
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--collection-name", default="rag_pipeline")
    parser.add_argument("--output", default="")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--prefer-grpc", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key or None,
        timeout=args.timeout,
        prefer_grpc=args.prefer_grpc,
    )

    snapshot = collect_snapshot(client, args.collection_name)
    rendered = json.dumps(snapshot, indent=2, sort_keys=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")

    print(rendered)


if __name__ == "__main__":
    main()
