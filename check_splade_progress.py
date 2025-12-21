#!/usr/bin/env python3
"""
Check SPLADE Cloud Inference Progress.

This script queries the Qdrant collection to show current SPLADE progress
and statistics about documents with and without sparse vectors.
"""

import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
except ImportError:
    os.system("pip3 install qdrant-client --quiet")
    from qdrant_client import QdrantClient

# Load environment
load_dotenv()

def get_collection_stats(client: QdrantClient, collection_name: str) -> Dict[str, Any]:
    """Get detailed collection statistics."""
    info = client.get_collection(collection_name)
    return {
        'total_points': info.points_count,
        'status': info.status,
        'vectors_config': info.config.params.vectors,
        'optimizers_status': info.optimizer_status,
        'payload_schema': getattr(info.config.params, 'payload_schema', None)
    }

def count_documents_by_source(client: QdrantClient, collection_name: str, sample_size: int = 10000) -> Dict[str, int]:
    """Sample documents by source type to estimate counts."""
    sources = {}

    offset = None
    sampled = 0
    while sampled < sample_size:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=min(1000, sample_size - sampled),
            with_payload=['source'],
            with_vectors=False
        )

        if not records:
            break

        for record in records:
            source = record.payload.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            sampled += 1

        offset = next_offset
        if offset is None:
            break

    return sources

def sample_sparse_vectors(client: QdrantClient, collection_name: str, sample_size: int = 1000) -> Dict[str, int]:
    """Sample documents to estimate sparse vector coverage."""
    with_sparse = 0
    without_sparse = 0
    total_sampled = 0

    offset = None
    while total_sampled < sample_size:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=min(100, sample_size - total_sampled),
            with_payload=False,
            with_vectors=['sparse']
        )

        if not records:
            break

        for record in records:
            total_sampled += 1
            if hasattr(record, 'vector') and record.vector and 'sparse' in record.vector:
                with_sparse += 1
            else:
                without_sparse += 1

        offset = next_offset
        if offset is None:
            break

    return {
        'sampled': total_sampled,
        'with_sparse': with_sparse,
        'without_sparse': without_sparse,
        'sparse_percentage': (with_sparse / total_sampled * 100) if total_sampled > 0 else 0
    }

def main():
    # Configuration
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("COLLECTION_NAME", "pmc_medical_rag_fulltext")

    if not qdrant_url or not qdrant_api_key:
        print("❌ QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        sys.exit(1)

    print("🔬 Checking SPLADE Cloud Inference Progress")
    print("=" * 60)

    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

        print("✅ Connected to Qdrant")
        print(f"   Collection: {collection_name}")

        # Get detailed stats
        stats = get_collection_stats(client, collection_name)
        print(f"   Total points: {stats['total_points']:,}")
        print(f"   Collection status: {stats['status']}")

        # Check optimizer status
        if stats['optimizers_status']:
            print(f"   Optimizer status: {stats['optimizers_status']}")

        # Check vector configuration
        if stats['vectors_config']:
            print("   Vector configurations:")
            try:
                if hasattr(stats['vectors_config'], 'keys'):
                    for vec_name in stats['vectors_config'].keys():
                        vec_config = getattr(stats['vectors_config'], vec_name)
                        print(f"     - {vec_name}: {type(vec_config).__name__}")
                else:
                    print(f"     - Vector config type: {type(stats['vectors_config']).__name__}")
            except Exception as e:
                print(f"     - Vector config: {str(e)}")

        # Count by source
        print("\n📊 Document Counts by Source:")
        sources = count_documents_by_source(client, collection_name)
        for source, count in sorted(sources.items()):
            print(f"   {source}: {count:,}")

        # Sample sparse vector coverage
        print("\n🔍 Sparse Vector Coverage (sampled):")
        sparse_stats = sample_sparse_vectors(client, collection_name, 5000)
        print(f"   Sampled: {sparse_stats['sampled']:,} documents")
        print(f"   With sparse vectors: {sparse_stats['with_sparse']:,}")
        print(f"   Without sparse vectors: {sparse_stats['without_sparse']:,}")
        print(f"   Sparse vector coverage: {sparse_stats['sparse_percentage']:.1f}%")

        # Estimated totals
        estimated_with_sparse = int(stats['total_points'] * (sparse_stats['sparse_percentage'] / 100))
        estimated_without_sparse = stats['total_points'] - estimated_with_sparse

        print("\n📈 Estimated Totals:")
        print(f"   Documents with SPLADE: ~{estimated_with_sparse:,}")
        print(f"   Documents without SPLADE: ~{estimated_without_sparse:,}")

        print("\n✅ Progress check complete!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
