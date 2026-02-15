# Full Reset & Phased Ingestion Strategy

To ensure a clean, reliable state for the RAG pipeline, we have implemented a **Phased Reset Strategy**. This allows us to verify the entire end-to-end flow (Download -> Extract -> Ingest -> Vector DB) with a small dataset before committing to the multi-terabyte full download.

## Phase 1: Clean Slate & Validation (Current)

**Goal:** Wipe old data, reset the vector DB, and ingest a **representative sample** (First 2 PMC baseline archives).

**Reasoning:**
- **Speed:** Validates the fix in < 1 hour instead of days.
- **Storage:** Uses ~30-50GB instead of ~800GB.
- **Verification:** Ensures `cohere` embeddings (1536d) and Qdrant ingestion work correctly.

**How to Run:**
The `recreate_collection_for_cohere.sh` script is hardcoded to Phase 1 settings.

```bash
# On the server:
./scripts/recreate_collection_for_cohere.sh
```

**What it does:**
1.  **Deletes** `rag_pipeline` collection in Qdrant.
2.  **Deletes** `/data/ingestion/pmc_xml` and other data dirs.
3.  **Downloads** only the first **2** `oa_comm` baseline files.
4.  **Ingests** PMC, DailyMed, and PubMed baseline.

---

## Phase 2: Full Scale Ingestion (Next Step)

**Goal:** Download and ingest the remaining PMC data (Total ~13 files, ~4TB uncompressed processed).

**Prerequisite:** valid success in Phase 1.

**How to Run:**
Run the complete ingestion script directly, **unsetting** the limit variable.

```bash
# On the server:
export PMC_MAX_FILES=""  # Clear the limit
./scripts/run_complete_ingestion.sh
```

*Note: The script is smart enough to skip the files already downloaded in Phase 1.*

## Storage Requirements (Estimated)

| Dataset | Compressed | Extracted XML | Qdrant Index |
| :--- | :--- | :--- | :--- |
| **PMC (oa_comm)** | ~98 GB | ~392 GB | ~200 GB |
| **DailyMed** | ~8 GB | ~35 GB | ~15 GB |
| **PubMed** | ~10 GB | ~40 GB | ~20 GB |
| **Total** | **~120 GB** | **~470 GB** | **~250 GB** |

**Total Disk Required:** ~850 GB (Safe margin: 1 TB).
*Your current 1.7TB drive is sufficient.*
