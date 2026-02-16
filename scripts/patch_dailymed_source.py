#!/usr/bin/env python3
"""
Patch existing DailyMed points in Qdrant to add 'source': 'dailymed' field.

This script updates payloads of existing points without re-ingesting.
Uses set_id field to identify DailyMed points.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from qdrant_client import QdrantClient
    from config_ingestion import IngestionConfig
except ImportError as e:
    logger.error(f"Failed to import: {e}")
    sys.exit(1)


def update_dailymed_source(batch_size: int = 1000) -> int:
    """
    Update all DailyMed points to add source field.
    Returns number of points updated.
    """
    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=600,
    )
    
    collection_name = IngestionConfig.COLLECTION_NAME
    logger.info(f"Connecting to collection: {collection_name}")
    
    # Get collection info
    info = client.get_collection(collection_name)
    total_points = info.points_count
    logger.info(f"Total points in collection: {total_points}")
    
    updated_count = 0
    scroll_offset = None
    
    logger.info("Starting payload update for DailyMed points...")
    
    while True:
        try:
            # Scroll through points with set_id (DailyMed indicator)
            # We fetch points that have set_id but may or may not have source
            response = client.scroll(
                collection_name=collection_name,
                offset=scroll_offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
            )
            
            points = response[0]
            if not points:
                break
            
            scroll_offset = response[1]
            
            # Filter points that:
            # 1. Have set_id (DailyMed indicator)
            # 2. Don't have source field or source is not "dailymed"
            points_to_update = []
            for point in points:
                payload = point.payload or {}
                has_set_id = bool(payload.get("set_id"))
                has_drug_name = bool(payload.get("drug_name"))
                current_source = payload.get("source")
                
                # DailyMed points have set_id and drug_name
                if has_set_id and has_drug_name and current_source != "dailymed":
                    points_to_update.append(point.id)
            
            if points_to_update:
                # Batch update payloads
                client.set_payload(
                    collection_name=collection_name,
                    payload={"source": "dailymed"},
                    points=points_to_update,
                )
                updated_count += len(points_to_update)
                logger.info(f"Updated {updated_count} points so far...")
            
            # Continue scrolling
            if scroll_offset is None:
                break
                
        except Exception as e:
            logger.error(f"Error during update: {e}")
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)
            continue
    
    logger.info(f"=" * 50)
    logger.info(f"Update complete!")
    logger.info(f"Total points updated: {updated_count}")
    logger.info(f"=" * 50)
    
    return updated_count


def verify_update(sample_size: int = 10) -> None:
    """Verify that points now have the source field."""
    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY or None,
        timeout=60,
    )
    
    collection_name = IngestionConfig.COLLECTION_NAME
    
    # Try to find points with source=dailymed
    response = client.scroll(
        collection_name=collection_name,
        limit=sample_size,
        with_payload=True,
        with_vectors=False,
    )
    
    logger.info("\nSample points after update:")
    for point in response[0]:
        payload = point.payload or {}
        set_id = payload.get("set_id", "N/A")
        drug_name = payload.get("drug_name", "N/A")[:30]
        source = payload.get("source", "MISSING")
        logger.info(f"  ID: {point.id[:20]}... | set_id: {set_id[:20]}... | drug: {drug_name}... | source: {source}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patch DailyMed points with source field")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for updates")
    parser.add_argument("--verify-only", action="store_true", help="Just verify, don't update")
    args = parser.parse_args()
    
    if args.verify_only:
        verify_update()
    else:
        updated = update_dailymed_source(batch_size=args.batch_size)
        if updated > 0:
            logger.info("\nVerifying update...")
            verify_update()
