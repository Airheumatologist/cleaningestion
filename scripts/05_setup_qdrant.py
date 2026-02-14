#!/usr/bin/env python3
"""Create the self-hosted Qdrant collection for the medical RAG pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

import requests
from qdrant_client import QdrantClient, models

from config_ingestion import IngestionConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _patch_2bit_quantization(collection_name: str) -> None:
    """Attempt 2-bit quantization patch using REST for Qdrant >=1.15."""
    payload = {
        "quantization_config": {
            "binary": {
                "type": "2bit",
                "always_ram": True,
            }
        }
    }
    headers = {"Content-Type": "application/json"}
    if IngestionConfig.QDRANT_API_KEY:
        headers["api-key"] = IngestionConfig.QDRANT_API_KEY

    response = requests.patch(
        f"{IngestionConfig.QDRANT_URL}/collections/{collection_name}",
        json=payload,
        headers=headers,
        timeout=30,
    )
    if response.status_code >= 300:
        logger.warning("2-bit quantization patch was not applied: %s", response.text[:300])
    else:
        logger.info("Applied 2-bit quantization patch")


def setup_collection(collection_name: str, keep_existing: bool, shard_number: int, use_2bit: bool) -> None:
    if not IngestionConfig.QDRANT_URL:
        logger.error("QDRANT_URL is required")
        sys.exit(1)

    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=120,
        prefer_grpc=IngestionConfig.USE_GRPC,
    )

    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        if keep_existing:
            logger.info("Collection already exists and --keep-existing set: %s", collection_name)
        else:
            logger.info("Deleting existing collection: %s", collection_name)
            client.delete_collection(collection_name=collection_name)

    if collection_name not in existing or not keep_existing:
        logger.info("Creating collection: %s", collection_name)
        sparse_vectors_config = None
        if IngestionConfig.SPARSE_ENABLED and IngestionConfig.SPARSE_MODE == "bm25":
            sparse_vectors_config = {
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            sparse_vectors_config=sparse_vectors_config,
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=128,
                on_disk=True,
            ),
            shard_number=shard_number,
            on_disk_payload=True,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )

    if use_2bit:
        _patch_2bit_quantization(collection_name)

    indexes = {
        "year": models.PayloadSchemaType.INTEGER,
        "source": models.PayloadSchemaType.KEYWORD,
        "article_type": models.PayloadSchemaType.KEYWORD,
        "journal": models.PayloadSchemaType.KEYWORD,
        "evidence_grade": models.PayloadSchemaType.KEYWORD,
        "country": models.PayloadSchemaType.KEYWORD,
    }

    for field_name, field_type in indexes.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            logger.info("Index ready: %s", field_name)
        except Exception as exc:
            logger.warning("Index %s: %s", field_name, exc)

    info = client.get_collection(collection_name)
    logger.info("Collection status=%s points=%s", info.status, info.points_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup self-hosted Qdrant collection")
    parser.add_argument("--collection-name", default=IngestionConfig.COLLECTION_NAME)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--no-2bit", action="store_true", help="Skip 2-bit patch and keep standard binary")
    args = parser.parse_args()

    setup_collection(
        collection_name=args.collection_name,
        keep_existing=args.keep_existing,
        shard_number=args.shards,
        use_2bit=not args.no_2bit,
    )


if __name__ == "__main__":
    main()
