import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
collection_name = os.getenv("QDRANT_COLLECTION", "rag_pipeline")

# 1. Count total points
total = client.get_collection(collection_name).points_count
print(f"Total points in collection: {total}")

# 2. Count author manuscripts
filter_author = models.Filter(
    must=[
        models.FieldCondition(
            key="is_author_manuscript",
            match=models.MatchValue(value=True),
        )
    ]
)
count_author = client.count(collection_name=collection_name, count_filter=filter_author)
print(f"Author Manuscripts: {count_author.count}")

# 3. Get a sample point from PMC OA to see if license info exists
filter_pmc_oa = models.Filter(
    must=[
        models.FieldCondition(
            key="source",
            match=models.MatchValue(value="pmc_oa"),
        )
    ]
)
sample, _ = client.scroll(collection_name=collection_name, scroll_filter=filter_pmc_oa, limit=1)
if sample:
    payload_keys = list(sample[0].payload.keys())
    print("\nPayload keys for a PMC OA article:")
    for key in sorted(payload_keys):
        print(f"  - {key}")
        
    print(f"\nis_open_access value: {sample[0].payload.get('is_open_access')}")
    print(f"license value (if any): {sample[0].payload.get('license')}")
    print(f"rights value (if any): {sample[0].payload.get('rights')}")
else:
    print("\nNo PMC OA records found.")

