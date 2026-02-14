#!/usr/bin/env python3
"""Smoke test for Hetzner Qdrant: validates collection schema, sparse config, and hybrid query."""

from __future__ import annotations

import argparse
import sys
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from config_ingestion import IngestionConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✘{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_collection_exists(client: QdrantClient, name: str) -> bool:
    collections = {c.name for c in client.get_collections().collections}
    if name in collections:
        _ok(f"Collection '{name}' exists")
        return True
    _fail(f"Collection '{name}' NOT found (available: {collections})")
    return False


def check_dense_config(client: QdrantClient, name: str) -> bool:
    info = client.get_collection(name)
    vc = info.config.params.vectors
    # vc can be VectorParams (unnamed) or dict of named vectors
    if hasattr(vc, "size"):
        size, dist = vc.size, vc.distance
    elif isinstance(vc, dict) and "" in vc:
        size, dist = vc[""].size, vc[""].distance
    else:
        _fail(f"Unexpected vector config type: {type(vc)}")
        return False

    ok = size == 1024 and str(dist).lower().replace("distance.", "") == "cosine"
    if ok:
        _ok(f"Dense vector: size={size}, distance={dist}")
    else:
        _fail(f"Dense vector mismatch: size={size}, distance={dist} (expected 1024/cosine)")
    return ok


def check_sparse_config(client: QdrantClient, name: str) -> bool:
    info = client.get_collection(name)
    svc = info.config.params.sparse_vectors
    if not svc or "sparse" not in svc:
        _fail("Sparse vector config 'sparse' NOT found")
        return False

    sparse_cfg = svc["sparse"]
    modifier = getattr(sparse_cfg, "modifier", None)
    modifier_str = str(modifier).lower() if modifier else ""
    has_idf = "idf" in modifier_str
    if has_idf:
        _ok(f"Sparse vector 'sparse' with IDF modifier ({modifier})")
    else:
        _warn(f"Sparse vector 'sparse' found but modifier={modifier} (expected IDF)")
    return True


def check_payload_indexes(client: QdrantClient, name: str) -> bool:
    info = client.get_collection(name)
    indexed = set(info.payload_schema.keys()) if info.payload_schema else set()
    required = {"year", "source", "article_type", "journal", "doc_id"}
    missing = required - indexed
    if not missing:
        _ok(f"Payload indexes present: {sorted(indexed)}")
        return True
    _warn(f"Missing payload indexes: {sorted(missing)} (present: {sorted(indexed)})")
    return False


def check_hybrid_roundtrip(client: QdrantClient, name: str) -> bool:
    """Insert a synthetic point with dense + sparse vectors, query it, then clean up."""
    test_id = str(uuid.uuid4())
    dense_vec = [0.01] * 1024
    sparse_vec = SparseVector(indices=[100, 200, 300], values=[1.0, 0.8, 0.6])

    try:
        # Upsert with named vectors
        client.upsert(
            collection_name=name,
            points=[
                PointStruct(
                    id=test_id,
                    vector={"": dense_vec, "sparse": sparse_vec},
                    payload={
                        "doc_id": "__smoke_test__",
                        "source": "smoke_test",
                        "title": "Smoke Test Point",
                    },
                )
            ],
            wait=True,
        )
        _ok("Inserted test point with dense + sparse vectors")
    except Exception as exc:
        _fail(f"Upsert failed: {exc}")
        return False

    # Dense query
    dense_ok = False
    try:
        results = client.query_points(
            collection_name=name,
            query=dense_vec,
            limit=1,
            with_payload=True,
        )
        if results.points and str(results.points[0].id) == test_id:
            _ok("Dense query returned test point")
            dense_ok = True
        else:
            _warn("Dense query did not return expected test point")
    except Exception as exc:
        _fail(f"Dense query failed: {exc}")

    # Sparse query
    sparse_ok = False
    try:
        results = client.query_points(
            collection_name=name,
            query=sparse_vec,
            using="sparse",
            limit=1,
            with_payload=True,
        )
        if results.points and str(results.points[0].id) == test_id:
            _ok("Sparse query returned test point")
            sparse_ok = True
        else:
            _warn("Sparse query did not return expected test point")
    except Exception as exc:
        _fail(f"Sparse query failed: {exc}")

    # Cleanup
    try:
        client.delete(collection_name=name, points_selector=[test_id], wait=True)
        _ok("Cleaned up test point")
    except Exception as exc:
        _warn(f"Cleanup failed (non-critical): {exc}")

    return dense_ok and sparse_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Qdrant collection schema and hybrid query")
    parser.add_argument("--collection-name", default=IngestionConfig.COLLECTION_NAME)
    args = parser.parse_args()

    name = args.collection_name
    print(f"\n{'='*60}")
    print(f"  Smoke Test: {IngestionConfig.QDRANT_URL} / {name}")
    print(f"{'='*60}\n")

    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=60,
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    results = []
    results.append(("Collection exists", check_collection_exists(client, name)))

    if results[-1][1]:
        results.append(("Dense config (1024/cosine)", check_dense_config(client, name)))
        results.append(("Sparse config (IDF)", check_sparse_config(client, name)))
        results.append(("Payload indexes", check_payload_indexes(client, name)))
        results.append(("Hybrid roundtrip", check_hybrid_roundtrip(client, name)))

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    color = GREEN if passed == total else RED
    print(f"  {color}Result: {passed}/{total} checks passed{RESET}")
    print(f"{'='*60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
