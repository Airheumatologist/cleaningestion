#!/usr/bin/env python3
"""Create the self-hosted Qdrant collection for the medical RAG pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

import requests
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)

from config_ingestion import IngestionConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _get_quantization_config():
    """Get quantization configuration based on settings."""
    quant_type = IngestionConfig.QUANTIZATION_TYPE
    
    if quant_type == "scalar":
        logger.info("Using Scalar Quantization (int8) - 75%% memory reduction")
        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=IngestionConfig.SCALAR_QUANTILE,
                always_ram=IngestionConfig.QUANTIZATION_ALWAYS_RAM,
            )
        )
    elif quant_type == "binary":
        logger.info("Using Binary Quantization - 87.5%% memory reduction")
        return BinaryQuantization(
            binary=BinaryQuantizationConfig(
                always_ram=IngestionConfig.QUANTIZATION_ALWAYS_RAM
            )
        )
    else:
        logger.info("Quantization disabled")
        return None


def _patch_2bit_quantization(collection_name: str) -> None:
    """Attempt 2-bit quantization patch using REST for Qdrant >=1.15."""
    # Only apply 2-bit patch if using binary quantization
    if IngestionConfig.QUANTIZATION_TYPE != "binary":
        logger.info("Skipping 2-bit patch (not using binary quantization)")
        return
        
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

        # Get vector size based on embedding provider
        vector_size = IngestionConfig.get_vector_size()
        logger.info("Using vector size %d for embedding provider: %s", vector_size, IngestionConfig.EMBEDDING_PROVIDER)
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            sparse_vectors_config=sparse_vectors_config,
            quantization_config=_get_quantization_config(),
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
        "evidence_level": models.PayloadSchemaType.INTEGER,
        "evidence_term": models.PayloadSchemaType.KEYWORD,
        "evidence_source": models.PayloadSchemaType.KEYWORD,
        "country": models.PayloadSchemaType.KEYWORD,
        "doc_id": models.PayloadSchemaType.KEYWORD,
        "chunk_id": models.PayloadSchemaType.KEYWORD,
        "section_type": models.PayloadSchemaType.KEYWORD,
        # Merged PubMed pipeline - Government affiliation fields
        "is_gov_affiliated": models.PayloadSchemaType.KEYWORD,
        "gov_agencies": models.PayloadSchemaType.KEYWORD,
        # PMC/PubMed identifiers (frequently queried)
        "pmcid": models.PayloadSchemaType.KEYWORD,
        "pmid": models.PayloadSchemaType.KEYWORD,
        "is_author_manuscript": models.PayloadSchemaType.BOOL,
        # DailyMed fields (frequently queried)
        "set_id": models.PayloadSchemaType.KEYWORD,
        "drug_name": models.PayloadSchemaType.KEYWORD,
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
    recreate_group = parser.add_mutually_exclusive_group()
    recreate_group.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep an existing collection and skip recreation",
    )
    recreate_group.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection if it already exists (default behavior)",
    )
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--no-2bit", action="store_true", help="Skip 2-bit patch and keep standard binary")
    parser.add_argument(
        "--quantization",
        choices=["scalar", "binary", "none"],
        default=IngestionConfig.QUANTIZATION_TYPE,
        help="Quantization type (default: from env or scalar)",
    )
    args = parser.parse_args()
    
    # Override config with CLI arg if provided
    if args.quantization != IngestionConfig.QUANTIZATION_TYPE:
        logger.info(f"Overriding quantization type: {IngestionConfig.QUANTIZATION_TYPE} -> {args.quantization}")
        IngestionConfig.QUANTIZATION_TYPE = args.quantization

    setup_collection(
        collection_name=args.collection_name,
        keep_existing=args.keep_existing,
        shard_number=args.shards,
        use_2bit=not args.no_2bit,
    )


if __name__ == "__main__":
    main()
