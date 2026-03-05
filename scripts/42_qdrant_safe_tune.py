#!/usr/bin/env python3
"""Safely repair missing payload indexes and ensure scalar quantization settings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient, models


REQUIRED_INDEXES = {
    "source": models.PayloadSchemaType.KEYWORD,
    "source_family": models.PayloadSchemaType.KEYWORD,
    "article_type": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "journal": models.PayloadSchemaType.KEYWORD,
}


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


def _extract_quantization_config(info: Any) -> Any:
    config = getattr(info, "config", None)
    if config is None:
        return None

    direct = getattr(config, "quantization_config", None)
    if direct is not None:
        return direct

    params = getattr(config, "params", None)
    if params is not None:
        nested = getattr(params, "quantization_config", None)
        if nested is not None:
            return nested

    return None


def _quantization_matches_target(quantization_config: Any) -> bool:
    if quantization_config is None:
        return False

    payload = _to_jsonable(quantization_config)
    scalar = payload.get("scalar") if isinstance(payload, dict) else None
    if not isinstance(scalar, dict):
        return False

    type_value = str(scalar.get("type", "")).lower()
    quantile_value = scalar.get("quantile")
    always_ram_value = scalar.get("always_ram")

    return (
        type_value in {"int8", "scalartype.int8"}
        and isinstance(quantile_value, (int, float))
        and abs(float(quantile_value) - 0.99) < 1e-6
        and bool(always_ram_value)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe Qdrant tuning (indexes + quantization)")
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--qdrant-api-key", default=None)
    parser.add_argument("--collection-name", default="rag_pipeline")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    args = parser.parse_args()

    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key or None,
        timeout=args.timeout,
        prefer_grpc=args.prefer_grpc,
    )

    info = client.get_collection(args.collection_name)
    payload_schema = _to_jsonable(getattr(info, "payload_schema", {})) or {}
    existing_indexes = set(payload_schema.keys()) if isinstance(payload_schema, dict) else set()

    missing_indexes = [field for field in REQUIRED_INDEXES if field not in existing_indexes]

    quantization_config = _extract_quantization_config(info)
    quantization_matches = _quantization_matches_target(quantization_config)

    applied_indexes: list[str] = []
    quantization_updated = False

    if args.apply:
        for field in missing_indexes:
            client.create_payload_index(
                collection_name=args.collection_name,
                field_name=field,
                field_schema=REQUIRED_INDEXES[field],
            )
            applied_indexes.append(field)

        if not quantization_matches:
            client.update_collection(
                collection_name=args.collection_name,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )
            quantization_updated = True

    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection_name": args.collection_name,
        "dry_run": not args.apply,
        "indexes": {
            "required": list(REQUIRED_INDEXES.keys()),
            "missing_before": missing_indexes,
            "applied": applied_indexes,
        },
        "quantization": {
            "matches_target_before": quantization_matches,
            "updated": quantization_updated,
            "target": {
                "type": "int8",
                "quantile": 0.99,
                "always_ram": True,
            },
            "current": _to_jsonable(quantization_config),
        },
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
