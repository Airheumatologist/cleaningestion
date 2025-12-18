#!/usr/bin/env python3
"""
Setup Qdrant Collection with Enhanced Configuration.

Creates collection with:
- Dense vectors: mixedbread-ai/mxbai-embed-large-v1 (1024-d)
- Sparse vectors: SPLADE support
- Binary quantization: Enabled
- Cloud Inference: Enabled for dense embeddings
- Sharding: Enabled for parallel ingestion
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from qdrant_client import models
from src.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_DIMENSION

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration for parallel ingestion
SHARD_NUMBER = 4  # Number of shards for parallel ingestion


def setup_collection(
    collection_name: str = COLLECTION_NAME,
    vector_size: int = EMBEDDING_DIMENSION,
    replace_existing: bool = True,
    shard_number: int = SHARD_NUMBER
):
    """Create Qdrant collection with all features."""
    
    logger.info("=" * 70)
    logger.info("🔧 Setting up Qdrant Collection")
    logger.info("=" * 70)
    
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=120,
        cloud_inference=True  # Enable Cloud Inference
    )
    
    logger.info(f"✅ Connected to Qdrant: {QDRANT_URL}")
    
    # Check if collection exists
    try:
        collections = client.get_collections()
        existing_collections = [c.name for c in collections.collections]
        
        if collection_name in existing_collections:
            if replace_existing:
                logger.info(f"⚠️  Collection '{collection_name}' exists. Deleting...")
                client.delete_collection(collection_name=collection_name)
                logger.info(f"✅ Deleted existing collection")
            else:
                logger.info(f"✅ Collection '{collection_name}' already exists (keeping)")
                return client
    except Exception as e:
        logger.warning(f"Error checking collections: {e}")
    
    # Create collection with dense and sparse vectors
    logger.info(f"\n📦 Creating collection: {collection_name}")
    logger.info(f"   Dense vector size: {vector_size}")
    logger.info(f"   Sparse vectors: Enabled (SPLADE)")
    logger.info(f"   Binary quantization: Enabled")
    logger.info(f"   Cloud Inference: Enabled")
    logger.info(f"   Shards: {shard_number}")
    
    try:
        # Create collection with correct API format
        # Based on Qdrant docs: https://qdrant.tech/documentation/guides/quantization/
        
        client.create_collection(
            collection_name=collection_name,
            # Dense vectors config
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,  # Store original vectors on disk
            ),
            # Sparse vectors config (separate parameter)
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,  # IDF modifier for better sparse search
                )
            },
            # Binary quantization config
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=False  # Keep quantized vectors on disk when not in use
                )
            ),
            # HNSW config for fast search
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=200,
            ),
            # Sharding for parallel ingestion
            shard_number=shard_number,
            # Store payloads on disk
            on_disk_payload=True,
        )
        
        logger.info(f"\n✅ Collection created successfully!")
        logger.info(f"   Collection name: {collection_name}")
        logger.info(f"   Vector dimension: {vector_size}")
        logger.info(f"   Quantization: Binary (1-bit)")
        logger.info(f"   Distance metric: Cosine")
        logger.info(f"   Sparse vectors: Enabled with IDF modifier")
        logger.info(f"   Shards: {shard_number}")
        
        # Verify collection
        collection_info = client.get_collection(collection_name)
        logger.info(f"\n📊 Collection Info:")
        logger.info(f"   Points count: {collection_info.points_count}")
        logger.info(f"   Status: {collection_info.status}")
        if hasattr(collection_info, 'vectors_count'):
            logger.info(f"   Vectors count: {collection_info.vectors_count}")
        if hasattr(collection_info, 'indexed_vectors_count'):
            logger.info(f"   Indexed vectors count: {collection_info.indexed_vectors_count}")
        
        return client
        
    except Exception as e:
        logger.error(f"❌ Error creating collection: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Qdrant collection")
    parser.add_argument("--collection-name", type=str, default=COLLECTION_NAME, help="Collection name")
    parser.add_argument("--vector-size", type=int, default=EMBEDDING_DIMENSION, help="Vector dimension")
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing collection if it exists")
    parser.add_argument("--shards", type=int, default=SHARD_NUMBER, help="Number of shards for parallel ingestion")
    
    args = parser.parse_args()
    
    try:
        setup_collection(
            collection_name=args.collection_name,
            vector_size=args.vector_size,
            replace_existing=not args.keep_existing,
            shard_number=args.shards
        )
        print("\n✅ Collection setup complete!")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
