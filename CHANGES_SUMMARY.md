# Ingestion Fixes Applied

## Summary of Changes

### 1. Fixed QualityValidator Token Range ✅
**File:** `scripts/ingestion_utils_enhanced.py`

**Problem:** The quality validator was using token range 200-800 as optimal, which penalized valid 2048-token chunks.

**Fix:** Updated the token scoring logic:
- Changed optimal range from `200-800` to `1500-2048` 
- Changed penalty threshold from `> 2000` to `> 2500`
- Now properly boosts chunks that match the configured `CHUNK_SIZE_TOKENS` (2048)

### 2. Fixed Author Manuscripts Logic Bug ✅
**File:** `scripts/14_ingest_author_manuscripts.py`

**Problem:** The validation check had an inverted condition:
```python
if len(chunks) < len(valid_chunks):  # Always False!
```

**Fix:** Store original count before filtering:
```python
original_count = len(chunks)
# ... filter ...
if len(chunks) < original_count:
    logger.debug("Validated chunks: %d valid out of %d total", len(chunks), original_count)
```

### 3. Improved SemanticChunker Token Counting ✅
**File:** `scripts/ingestion_utils_enhanced.py`

**Problem:** `SemanticChunker` used word-based approximation (1 token ≈ 0.75 words) instead of actual tokenizer.

**Fix:** 
- Added `_load_tokenizer()` method to import tokenizer from `ingestion_utils.Chunker`
- Updated `_count_tokens()` to use actual tokenizer when available
- Falls back to word approximation if tokenizer unavailable

**Result:** More accurate token counts that match the embedding model's tokenizer.

### 4. Standardized Imports ✅
**Files:** `scripts/14_ingest_author_manuscripts.py`, `scripts/15_ingest_gov_abstracts.py`

**Problem:** Duplicate `generate_section_id` and `get_section_weight` functions defined locally.

**Fix:** 
- Removed local function definitions
- Added imports from `ingestion_utils`:
  ```python
  from ingestion_utils import generate_section_id, get_section_weight
  ```

### 5. Improved Help Text ✅
**Files:** `scripts/14_ingest_author_manuscripts.py`, `scripts/15_ingest_gov_abstracts.py`

**Problem:** Help text for `--batch-size` showed static defaults instead of actual configured values.

**Fix:** Updated help text to use f-strings with actual variable values:
```python
help=f"Batch size for upserts (default: {BATCH_SIZE})"
```

## Verification

All changes have been tested and verified:

```bash
# Test SemanticChunker with tokenizer
python3 -c "
import sys
sys.path.insert(0, 'scripts')
from ingestion_utils_enhanced import SemanticChunker
sc = SemanticChunker()
print(f'Tokenizer loaded: {sc.tokenizer is not None}')
print(f'Token count: {sc._count_tokens(\"test sentence\")}')
"

# Test QualityValidator
python3 -c "
import sys
sys.path.insert(0, 'scripts')
from ingestion_utils_enhanced import QualityValidator
score = QualityValidator.compute_quality_score('x ' * 2000, {})
print(f'Large chunk score: {score}')
"
```

## Files Modified

1. `scripts/ingestion_utils_enhanced.py` - Token range fix, tokenizer integration
2. `scripts/14_ingest_author_manuscripts.py` - Bug fix, import standardization
3. `scripts/15_ingest_gov_abstracts.py` - Import standardization, help text

## Backward Compatibility

All changes are backward compatible:
- Token range changes only affect quality scoring (not chunking logic)
- Tokenizer integration has graceful fallback to word approximation
- Import changes maintain same function signatures
