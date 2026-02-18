#!/usr/bin/env python3
"""
Generate drug_setid_lookup.json for fast DailyMed drug name lookups.

This script queries Qdrant for all DailyMed entries and builds a lookup table
mapping drug names (and generic/active ingredient names) to their set_ids.

Usage:
    python scripts/generate_drug_lookup.py
    python scripts/generate_drug_lookup.py --output src/data/drug_setid_lookup.json
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_dailymed_drugs(client: QdrantClient, collection_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetch all DailyMed entries from Qdrant and extract drug name mappings.
    
    Returns a dict mapping lowercase drug name -> {"set_ids": [...], "names": [...]}
    """
    lookup: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"set_ids": set(), "names": set()})
    seen_set_ids: Set[str] = set()
    
    # Filter for DailyMed source only
    scroll_filter = Filter(
        must=[
            FieldCondition(key="source", match=MatchValue(value="dailymed"))
        ]
    )
    
    logger.info("Fetching DailyMed entries from Qdrant...")
    
    offset = None
    total_processed = 0
    unique_drugs = 0
    
    while True:
        try:
            results, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=1000,
                offset=offset,
                with_payload=True
            )
            
            if not results:
                break
            
            for point in results:
                payload = point.payload if hasattr(point, 'payload') else {}
                
                set_id = payload.get("set_id", "")
                drug_name = payload.get("drug_name", "")
                active_ingredients = payload.get("active_ingredients", []) or []
                
                if not set_id or not drug_name:
                    continue
                
                seen_set_ids.add(set_id)
                
                # Add primary drug name
                drug_name_clean = drug_name.strip()
                drug_name_lower = drug_name_clean.lower()
                
                if drug_name_lower:
                    lookup[drug_name_lower]["set_ids"].add(set_id)
                    lookup[drug_name_lower]["names"].add(drug_name_clean)
                
                # Add active ingredients (generic names)
                for ingredient in active_ingredients:
                    if ingredient and isinstance(ingredient, str):
                        ingredient_clean = ingredient.strip()
                        ingredient_lower = ingredient_clean.lower()
                        
                        if ingredient_lower:
                            lookup[ingredient_lower]["set_ids"].add(set_id)
                            lookup[ingredient_lower]["names"].add(ingredient_clean)
                
                # Also extract common variations of the drug name
                # e.g., "SIMPONI ARIA" -> add "simponi" as well
                name_parts = drug_name_lower.split()
                if len(name_parts) > 1:
                    # Add first word as potential short name
                    short_name = name_parts[0]
                    if len(short_name) > 2:  # Avoid single letters
                        lookup[short_name]["set_ids"].add(set_id)
                        lookup[short_name]["names"].add(name_parts[0].upper())
            
            total_processed += len(results)
            
            if total_processed % 10000 == 0:
                logger.info(f"  Processed {total_processed} entries, {len(lookup)} unique names...")
            
            if next_offset is None:
                break
            offset = next_offset
            
        except Exception as e:
            logger.error(f"Error scrolling through Qdrant: {e}")
            break
    
    logger.info(f"✅ Fetched {total_processed} entries with {len(seen_set_ids)} unique set_ids")
    logger.info(f"✅ Built lookup with {len(lookup)} unique drug names")
    
    return lookup


def convert_to_serializable(lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """Convert the lookup dict to a JSON-serializable format."""
    result = {}
    
    for name_lower, data in lookup.items():
        set_ids = sorted(list(data["set_ids"]))
        names = sorted(list(data["names"]))
        
        if set_ids:  # Only include if we have set_ids
            result[name_lower] = {
                "set_ids": set_ids,
                "names": names
            }
    
    return result


def save_lookup(lookup: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """Save the lookup table to a JSON file."""
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created directory: {output_dir}")
    
    # Convert to serializable format
    serializable = convert_to_serializable(lookup)
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Saved lookup table to: {output_path}")
    logger.info(f"   Total entries: {len(serializable)}")
    
    # Log some examples
    examples = list(serializable.items())[:5]
    logger.info("   Sample entries:")
    for name, data in examples:
        set_ids = data["set_ids"]
        logger.info(f"     - '{name}' -> {len(set_ids)} set_id(s)")


def generate_stats(lookup: Dict[str, Dict[str, Any]]) -> None:
    """Generate and log statistics about the lookup table."""
    total_names = len(lookup)
    total_set_ids = set()
    multi_mapping_names = 0
    
    for data in lookup.values():
        set_ids = data["set_ids"]
        total_set_ids.update(set_ids)
        if len(set_ids) > 1:
            multi_mapping_names += 1
    
    logger.info("📊 Lookup Statistics:")
    logger.info(f"   - Total unique drug names: {total_names}")
    logger.info(f"   - Total unique set_ids: {len(total_set_ids)}")
    logger.info(f"   - Names mapping to multiple set_ids: {multi_mapping_names}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate drug_setid_lookup.json for fast DailyMed lookups"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/data/drug_setid_lookup.json",
        help="Output path for the lookup JSON file (default: src/data/drug_setid_lookup.json)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME})"
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=QDRANT_URL,
        help=f"Qdrant URL (default: {QDRANT_URL})"
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=QDRANT_API_KEY,
        help="Qdrant API key (optional)"
    )
    
    args = parser.parse_args()
    
    # Resolve output path relative to project root
    project_root = Path(__file__).parent.parent
    if not os.path.isabs(args.output):
        output_path = str(project_root / args.output)
    else:
        output_path = args.output
    
    logger.info("=" * 60)
    logger.info("Generating Drug Set ID Lookup Table")
    logger.info("=" * 60)
    logger.info(f"Qdrant URL: {args.qdrant_url}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Output: {output_path}")
    logger.info("")
    
    try:
        # Connect to Qdrant
        client = QdrantClient(
            url=args.qdrant_url,
            api_key=args.qdrant_api_key or None,
            timeout=300
        )
        
        # Test connection
        try:
            collection_info = client.get_collection(args.collection)
            logger.info(f"✅ Connected to Qdrant collection: {args.collection}")
            logger.info(f"   Total points: {collection_info.points_count}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            sys.exit(1)
        
        # Fetch and build lookup
        lookup = fetch_dailymed_drugs(client, args.collection)
        
        # Check if lookup is empty
        if not lookup:
            logger.warning("⚠️  No DailyMed entries found in Qdrant!")
            logger.warning("   The lookup file will be empty.")
            logger.warning("   Run this script again after DailyMed ingestion completes.")
            # Still create the file (empty dict) so the retriever doesn't complain
        
        # Generate stats
        generate_stats(lookup)
        
        # Save to file
        save_lookup(lookup, output_path)
        
        if lookup:
            logger.info("")
            logger.info("=" * 60)
            logger.info("✅ Drug lookup generation complete!")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. The lookup file is now available at:")
            logger.info(f"     {output_path}")
            logger.info("  2. The retriever will automatically use this file for O(1) drug lookups")
            logger.info("  3. No code changes required - restart the API server to load the new lookup")
        else:
            logger.info("")
            logger.info("=" * 60)
            logger.info("⚠️  Drug lookup generation complete (EMPTY)")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Complete DailyMed ingestion first")
            logger.info("  2. Re-run this script to generate the lookup table")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
