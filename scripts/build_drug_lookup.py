#!/usr/bin/env python3
"""
Build a drug name → set_id lookup table from DailyMed API.

This creates a JSON file mapping drug names (both brand and generic) to their
DailyMed set_ids for instant O(1) lookup at runtime.

Usage:
    python scripts/build_drug_lookup.py

Output:
    src/data/drug_setid_lookup.json
"""

import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep, time
from typing import Dict, List, Set

# Configuration
OUTPUT_FILE = "src/data/drug_setid_lookup.json"
DAILYMED_API_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
PAGE_SIZE = 100  # Max allowed by API
MAX_WORKERS = 5  # Concurrent requests
RATE_LIMIT_DELAY = 0.1  # Seconds between batches


def fetch_drug_names_page(page: int) -> List[Dict]:
    """Fetch a single page of drug names from DailyMed API."""
    url = f"{DAILYMED_API_BASE}/drugnames.json"
    params = {"page": page, "pagesize": PAGE_SIZE}
    
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", []), data["metadata"]["total_pages"]
        except Exception as e:
            if attempt < 2:
                sleep(1)
            else:
                print(f"  ⚠️ Failed to fetch page {page}: {e}")
                return [], 0
    return [], 0


def fetch_setids_for_drug(drug_name: str) -> List[str]:
    """Fetch set_ids for a specific drug name."""
    url = f"{DAILYMED_API_BASE}/spls.json"
    params = {"drug_name": drug_name, "pagesize": 10}  # Limit to 10 per drug
    
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [item["setid"] for item in data.get("data", [])]
    except Exception as e:
        return []


def build_lookup_from_api():
    """Build complete drug name → set_id lookup from DailyMed API."""
    print("🔄 Building drug name → set_id lookup from DailyMed API")
    print("=" * 60)
    
    start_time = time()
    lookup = {}  # {drug_name_lower: {"type": "B"/"G", "set_ids": [...]}}
    
    # Step 1: Fetch all drug names
    print("\n📥 Step 1: Fetching all drug names...")
    first_page, total_pages = fetch_drug_names_page(1)
    print(f"   Total pages: {total_pages}")
    
    all_drug_names = []
    for item in first_page:
        all_drug_names.append({
            "name": item["drug_name"],
            "type": item["name_type"]
        })
    
    # Fetch remaining pages
    for page in range(2, total_pages + 1):
        data, _ = fetch_drug_names_page(page)
        for item in data:
            all_drug_names.append({
                "name": item["drug_name"],
                "type": item["name_type"]
            })
        
        if page % 100 == 0:
            print(f"   Fetched page {page}/{total_pages} ({len(all_drug_names):,} names)")
        sleep(RATE_LIMIT_DELAY)
    
    print(f"   ✅ Total drug names: {len(all_drug_names):,}")
    
    # Step 2: Fetch set_ids for each drug (parallel)
    print("\n📥 Step 2: Fetching set_ids for each drug...")
    print(f"   Using {MAX_WORKERS} parallel workers")
    
    processed = 0
    errors = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {}
        for drug in all_drug_names:
            future = executor.submit(fetch_setids_for_drug, drug["name"])
            futures[future] = drug
        
        # Collect results
        for future in as_completed(futures):
            drug = futures[future]
            try:
                set_ids = future.result()
                name_lower = drug["name"].lower()
                
                if set_ids:
                    if name_lower in lookup:
                        # Merge set_ids
                        existing = set(lookup[name_lower]["set_ids"])
                        existing.update(set_ids)
                        lookup[name_lower]["set_ids"] = list(existing)
                    else:
                        lookup[name_lower] = {
                            "type": drug["type"],
                            "set_ids": set_ids
                        }
            except Exception as e:
                errors += 1
            
            processed += 1
            if processed % 1000 == 0:
                print(f"   Processed {processed:,}/{len(all_drug_names):,} "
                      f"({len(lookup):,} with set_ids)")
    
    print(f"   ✅ Processed {processed:,} drugs, {errors} errors")
    print(f"   ✅ Lookup entries: {len(lookup):,}")
    
    # Step 3: Save to file
    print(f"\n💾 Step 3: Saving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(lookup, f, indent=2)
    
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    elapsed = time() - start_time
    
    print(f"   ✅ Saved {file_size:.1f} MB")
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")
    print("=" * 60)
    
    return lookup


def build_lookup_quick():
    """
    Quick alternative: Build lookup from just common/important drugs.
    Much faster for testing (~1 minute).
    """
    print("🔄 Building quick drug lookup (most common drugs only)")
    print("=" * 60)
    
    # Common drugs that are frequently queried
    common_drugs = [
        # Immunosuppressants
        "voclosporin", "lupkynis", "cyclosporine", "tacrolimus", "mycophenolate",
        # Gout
        "colchicine", "allopurinol", "febuxostat", "probenecid",
        # Biologics (RA/autoimmune)
        "humira", "adalimumab", "enbrel", "etanercept", "remicade", "infliximab",
        "xeljanz", "tofacitinib", "rinvoq", "upadacitinib", "olumiant", "baricitinib",
        "orencia", "abatacept", "actemra", "tocilizumab", "rituxan", "rituximab",
        # Psoriasis
        "cosentyx", "secukinumab", "taltz", "ixekizumab", "stelara", "ustekinumab",
        "skyrizi", "risankizumab", "tremfya", "guselkumab", "otezla", "apremilast",
        # DMARDs
        "methotrexate", "sulfasalazine", "leflunomide", "hydroxychloroquine", "plaquenil",
        # Steroids
        "prednisone", "prednisolone", "methylprednisolone", "dexamethasone",
        # NSAIDs
        "ibuprofen", "naproxen", "celecoxib", "celebrex", "meloxicam",
        # Diabetes
        "metformin", "ozempic", "semaglutide", "januvia", "sitagliptin",
        "jardiance", "empagliflozin", "farxiga", "dapagliflozin",
        # Cardiovascular
        "lisinopril", "amlodipine", "losartan", "atorvastatin", "lipitor",
        # Other common
        "omeprazole", "levothyroxine", "gabapentin", "amoxicillin",
    ]
    
    lookup = {}
    
    print(f"   Fetching set_ids for {len(common_drugs)} common drugs...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_setids_for_drug, d): d for d in common_drugs}
        
        for future in as_completed(futures):
            drug_name = futures[future]
            set_ids = future.result()
            
            if set_ids:
                lookup[drug_name.lower()] = {
                    "type": "G",  # Assume generic for now
                    "set_ids": set_ids
                }
                print(f"   ✅ {drug_name}: {len(set_ids)} set_ids")
            else:
                print(f"   ⚠️ {drug_name}: no set_ids found")
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(lookup, f, indent=2)
    
    print(f"\n💾 Saved {len(lookup)} drugs to {OUTPUT_FILE}")
    return lookup


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        build_lookup_quick()
    else:
        print("Building full lookup (this may take 30-60 minutes)...")
        print("Use --quick for a faster version with common drugs only")
        print()
        build_lookup_from_api()
