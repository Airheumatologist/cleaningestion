#!/usr/bin/env python3
"""
Setup Qdrant Collection for Medical RAG Pipeline.

Creates a Qdrant collection with:
- Dense vectors: mixedbread-ai/mxbai-embed-large-v1 (1024-d)
- Sparse vectors: SPLADE support (optional)
- Binary quantization: Enabled for memory efficiency
- Cloud Inference: Enabled for automatic embedding
- Sharding: 4 shards for parallel ingestion
- Payload indexes: year, source, article_type, journal, country, evidence_grade

Usage:
    python 05_setup_qdrant.py [--collection-name pmc_medical_rag_fulltext]

Requirements:
    - QDRANT_URL and QDRANT_API_KEY environment variables
    - Cloud Inference enabled in Qdrant Cloud dashboard
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models
except ImportError:
    print("Installing qdrant-client...")
    os.system("pip3 install qdrant-client python-dotenv --quiet")
    from qdrant_client import QdrantClient
    from qdrant_client import models

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pmc_medical_rag_fulltext")

# Vector configuration
VECTOR_SIZE = 1024  # mxbai-embed-large-v1
SHARD_NUMBER = 4    # For parallel ingestion


def setup_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    vector_size: int = VECTOR_SIZE,
    replace_existing: bool = True,
    shard_number: int = SHARD_NUMBER,
    enable_sparse: bool = True
):
    """
    Create and configure Qdrant collection.
    
    Args:
        collection_name: Name of the collection
        vector_size: Dimension of dense vectors
        replace_existing: Delete existing collection if present
        shard_number: Number of shards for parallel processing
        enable_sparse: Enable sparse vector support
    """
    
    logger.info("=" * 70)
    logger.info("🔧 Qdrant Collection Setup")
    logger.info("=" * 70)
    
    # Validate configuration
    if not QDRANT_URL:
        logger.error("❌ QDRANT_URL environment variable not set!")
        sys.exit(1)
    
    if not QDRANT_API_KEY:
        logger.error("❌ QDRANT_API_KEY environment variable not set!")
        sys.exit(1)
    
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=120,
        cloud_inference=True
    )
    
    logger.info(f"✅ Connected to Qdrant: {QDRANT_URL}")
    
    # Check if collection exists
    try:
        collections = client.get_collections()
        existing = [c.name for c in collections.collections]
        
        if collection_name in existing:
            if replace_existing:
                logger.info(f"⚠️  Collection '{collection_name}' exists. Deleting...")
                client.delete_collection(collection_name=collection_name)
                logger.info("✅ Deleted existing collection")
            else:
                logger.info(f"✅ Collection '{collection_name}' already exists (keeping)")
                return client
    except Exception as e:
        logger.warning(f"Error checking collections: {e}")
    
    # Prepare sparse vector config
    sparse_config = None
    if enable_sparse:
        sparse_config = {
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    
    # Create collection
    logger.info(f"\n📦 Creating collection: {collection_name}")
    logger.info(f"   Dense vector size: {vector_size}")
    logger.info(f"   Sparse vectors: {'Enabled' if enable_sparse else 'Disabled'}")
    logger.info(f"   Binary quantization: Enabled")
    logger.info(f"   Cloud Inference: Enabled")
    logger.info(f"   Shards: {shard_number}")
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            sparse_vectors_config=sparse_config,
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=False,
                )
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=200,
            ),
            shard_number=shard_number,
            on_disk_payload=True,
        )
        
        logger.info("\n✅ Collection created successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error creating collection: {e}")
        raise
    
    # Create payload indexes for filtering
    logger.info("\n📊 Creating payload indexes...")
    
    indexes = [
        ("year", models.PayloadSchemaType.INTEGER),
        ("source", models.PayloadSchemaType.KEYWORD),
        ("article_type", models.PayloadSchemaType.KEYWORD),
        ("journal", models.PayloadSchemaType.KEYWORD),
        ("country", models.PayloadSchemaType.KEYWORD),
        ("evidence_grade", models.PayloadSchemaType.KEYWORD),
    ]
    
    for field_name, field_type in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            logger.info(f"   ✅ Created index: {field_name}")
        except Exception as e:
            logger.warning(f"   ⚠️  Index {field_name}: {e}")
    
    # Verify collection
    info = client.get_collection(collection_name)
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 Collection Info")
    logger.info("=" * 70)
    logger.info(f"   Name: {collection_name}")
    logger.info(f"   Points: {info.points_count:,}")
    logger.info(f"   Status: {info.status}")
    logger.info(f"   Vectors: {vector_size}-dimensional")
    logger.info(f"   Shards: {shard_number}")
    
    return client


def main():
    parser = argparse.ArgumentParser(description="Setup Qdrant collection")
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Collection name"
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=VECTOR_SIZE,
        help="Vector dimension"
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=SHARD_NUMBER,
        help="Number of shards"
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing collection if present"
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse vector support"
    )
    
    args = parser.parse_args()
    
    setup_collection(
        collection_name=args.collection_name,
        vector_size=args.vector_size,
        replace_existing=not args.keep_existing,
        shard_number=args.shards,
        enable_sparse=not args.no_sparse,
    )
    
    print("\n✅ Collection setup complete!")


if __name__ == "__main__":
    main()

