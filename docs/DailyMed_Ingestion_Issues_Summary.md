# DailyMed Ingestion Issues Summary

**Date:** 2026-02-16  
**Reporter:** vinay  
**Status:** In Progress - Fixes Applied Locally, Need Deployment

---

## Executive Summary

The DailyMed drug label ingestion pipeline had multiple critical bugs preventing successful data ingestion. This document summarizes the issues discovered and the fixes applied.

---

## Issues Discovered

### Issue 1: Missing DailyMed XML Files

**Symptom:**
```
WARNING - No XML files found in /data/ingestion/dailymed/xml
```

**Root Cause:**
- The download script (`03_download_dailymed.py`) was defaulting to `/data/dailymed/xml`
- The ingestion script (`07_ingest_dailymed.py`) expects `/data/ingestion/dailymed/xml` (per centralized config)
- ZIP files were downloaded to wrong location and never extracted

**Evidence:**
```bash
# Found ZIPs in wrong location
find /data -name "*dailymed*" -o -name "*spl*.zip"
/data/dailymed/xml/dm_spl_release_human_rx_part1.zip
/data/dailymed/xml/dm_spl_release_human_rx_part2.zip
...
/data/ingestion/dailymed/xml/  # Empty!
```

**Status:** ✅ Fixed by extracting ZIPs to correct location

---

### Issue 2: Undefined `CHECKPOINT_FILE` Variable

**Symptom:**
```
ERROR - Ingestion failed: name 'CHECKPOINT_FILE' is not defined
```

**Location:** `scripts/07_ingest_dailymed.py`, line 287

**Root Cause:**
The script uses `CHECKPOINT_FILE` in functions `load_checkpoint()` and `append_checkpoint()`, but the variable was never defined.

**Fix:**
Add the following line before the threading import (around line 283):
```python
# Checkpoint file for DailyMed ingestion
CHECKPOINT_FILE = IngestionConfig.DATA_DIR / "dailymed_ingested_ids.txt"
```

**Status:** ✅ Fixed locally, needs deployment to server

---

### Issue 3: Missing `BM25SparseEncoder` Import

**Symptom:**
```
ERROR - Ingestion failed: name 'BM25SparseEncoder' is not defined
```

**Location:** `scripts/07_ingest_dailymed.py`, lines 302 and 407

**Root Cause:**
The script references `BM25SparseEncoder` but never imports it.

**Fix:**
Add the following import after the ingestion_utils import (around line 25):
```python
# Import BM25SparseEncoder
import importlib.util
spec = importlib.util.find_spec("src.bm25_sparse")
if spec is not None:
    from src.bm25_sparse import BM25SparseEncoder
else:
    BM25SparseEncoder = None  # type: ignore
```

**Status:** ✅ Fixed locally, needs deployment to server

---

### Issue 4: Logger Not Defined Before Imports

**Symptom:**
```
NameError: name 'logger' is not defined
```

**Root Cause:**
If imports fail (e.g., missing `lxml`), the exception handler tries to use `logger` before it's defined.

**Fix:**
Move logger initialization before imports and wrap imports in try/except:
```python
# Initialize logger FIRST before any imports that might fail
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import lxml.etree as ET
    from qdrant_client import QdrantClient
    from qdrant_client.models import Document, PointStruct
    # ... other imports
except Exception as import_err:
    logger.error("Failed to import required modules: %s", import_err)
    logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)
```

**Status:** ✅ Fixed locally, needs deployment to server

---

## Files Modified

1. **`scripts/03_download_dailymed.py`**
   - Uses centralized config for default path
   - Better error handling

2. **`scripts/07_ingest_dailymed.py`**
   - Added `CHECKPOINT_FILE` definition
   - Added `BM25SparseEncoder` import
   - Moved logger initialization before imports

---

## Current Server Status

### Data Location
```
/data/dailymed/xml/                          # 5 ZIP files (wrong location)
    dm_spl_release_human_rx_part1.zip        # 15,121 nested ZIPs
    dm_spl_release_human_rx_part2.zip        # 9,923 nested ZIPs
    dm_spl_release_human_rx_part3.zip        # 9,699 nested ZIPs
    dm_spl_release_human_rx_part4.zip        # 8,036 nested ZIPs
    dm_spl_release_human_rx_part5.zip        # 8,324 nested ZIPs

/data/ingestion/dailymed/xml/                # 51,103 XML files (correct location)
    # Extracted and ready for ingestion
```

### Extraction Completed ✅
```
Total XML files extracted: 51,103
Location: /data/ingestion/dailymed/xml
```

---

## Next Steps for Handoff Agent

### Step 1: Deploy Fixes to Server

```bash
# SSH to server
ssh user@your-server-ip
cd /opt/RAG-pipeline

# Option A: Git pull (if changes pushed)
git pull origin main

# Option B: Apply fixes manually (if not pushed)
# See "Manual Fix Commands" section below
```

### Step 2: Run Ingestion

```bash
cd /opt/RAG-pipeline
source venv/bin/activate  # or .venv/bin/activate

# Verify XML files exist
ls /data/ingestion/dailymed/xml/*.xml | wc -l
# Should output: 51103

# Run ingestion
python scripts/07_ingest_dailymed.py --xml-dir /data/ingestion/dailymed/xml
```

### Step 3: Monitor Progress

```bash
# Watch logs in real-time
tail -f /data/ingestion/logs/dailymed_*.log

# Check checkpoint progress
wc -l /data/ingestion/dailymed_ingested_ids.txt

# Check Qdrant collection
curl -s http://localhost:6333/collections/rag_pipeline | jq '.result.points_count'
```

---

## Manual Fix Commands (If Git Not Available)

Apply these fixes directly on the server if git pull is not an option:

```bash
cd /opt/RAG-pipeline

# Fix 1: Add CHECKPOINT_FILE definition
sed -i 's/^import threading/# Checkpoint file for DailyMed ingestion\nCHECKPOINT_FILE = IngestionConfig.DATA_DIR \/ "dailymed_ingested_ids.txt"\n\nimport threading/' scripts/07_ingest_dailymed.py

# Fix 2: Add BM25SparseEncoder import (add after ingestion_utils import line)
# This requires manual editing - see full file below

# Verify fixes
grep -n "CHECKPOINT_FILE\|BM25SparseEncoder" scripts/07_ingest_dailymed.py
```

---

## Expected Timeline

- **Download/Extraction:** ✅ Already done (~2 hours)
- **Ingestion:** 2-4 hours (51,103 drug labels, ~200K+ chunks)
- **Total:** ~4-6 hours from now

---

## Verification Checklist

- [ ] Fixes deployed to server
- [ ] `CHECKPOINT_FILE` defined in script
- [ ] `BM25SparseEncoder` imported in script
- [ ] Ingestion running without errors
- [ ] Checkpoint file growing (`/data/ingestion/dailymed_ingested_ids.txt`)
- [ ] Qdrant points increasing
- [ ] Completion log shows all 51,103 files processed

---

## Related Files

- `scripts/03_download_dailymed.py` - Download DailyMed ZIPs
- `scripts/04_process_dailymed.py` - Process to JSONL (legacy)
- `scripts/07_ingest_dailymed.py` - **Main ingestion script (BROKEN - FIXED)**
- `scripts/config_ingestion.py` - Centralized config
- `scripts/ingestion_utils.py` - Shared utilities

---

## Contact

For questions, check the logs:
```bash
tail -100 /data/ingestion/logs/dailymed_20260216_*.log
```
