# PubMed + Gov Abstracts Pipeline Merge

## Summary

The separate **Gov Abstracts** pipeline has been **merged** into the **PubMed Pipeline**.

### Why This Works
- Both download from the same source: `ftp.ncbi.nlm.nih.gov/pubmed/baseline/`
- Same XML schema, same ingestion pattern
- Only difference was filter criteria
- Single pass = efficiency gain (~3-4 hours saved)

## Schema Changes

### Before (Separate Pipelines)

| Pipeline | Source Field | Filter Criteria |
|----------|--------------|-----------------|
| PubMed | `pubmed_abstract` | Publication types (reviews, trials, guidelines) |
| Gov | `pubmed_gov` | Affiliation (NIH, CDC, FDA, VA) |

### After (Merged Pipeline)

Unified source with new metadata fields:

```json
{
  "source": "pubmed_abstract",
  "content_type": "abstract",
  "article_type": "randomized_controlled_trial",
  "publication_types": ["Randomized Controlled Trial"],
  
  // NEW: Government affiliation fields (merged from gov pipeline)
  "is_gov_affiliated": true,
  "gov_agencies": ["NIH", "NCI", "CDC"],
  
  // Existing fields remain unchanged
  "pmid": "12345678",
  "title": "...",
  "abstract": "...",
  "year": 2024,
  "journal": "...",
  "mesh_terms": [...]
}
```

## Migration Steps

### 1. Update Qdrant Collection (New Indexes)

```bash
# Add new indexes for gov fields
python scripts/05_setup_qdrant.py
```

This creates indexes for:
- `is_gov_affiliated` (keyword) - for boolean filtering
- `gov_agencies` (keyword) - for agency-specific filtering

### 2. Re-download with Unified Pipeline

```bash
# Download PubMed baseline with gov affiliation extraction
python scripts/20_download_pubmed_baseline.py \
    --output-dir /data/ingestion/pubmed_baseline \
    --min-year 2015

# Output: /data/ingestion/pubmed_baseline/filtered/pubmed_abstracts.jsonl
# Now includes is_gov_affiliated and gov_agencies fields
```

### 3. Re-ingest with Updated Script

```bash
# Ingest with new payload fields
python scripts/21_ingest_pubmed_abstracts.py \
    --input /data/ingestion/pubmed_baseline/filtered/pubmed_abstracts.jsonl
```

### 4. Update Queries (If Filtering by Source)

#### Before (Old Gov Pipeline)
```python
# Old way - separate source
filter = FieldCondition(key="source", match=MatchValue(value="pubmed_gov"))
```

#### After (Merged Pipeline)
```python
# New way - use is_gov_affiliated field
filter = FieldCondition(key="is_gov_affiliated", match=MatchValue(value=True))

# Or filter by specific agency
filter = FieldCondition(key="gov_agencies", match=MatchValue(value="NIH"))
```

## Query Examples

### Retrieve Only Government Articles
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

filter = Filter(
    must=[
        FieldCondition(key="is_gov_affiliated", match=MatchValue(value=True))
    ]
)
```

### Retrieve Specific Agency Articles
```python
filter = Filter(
    must=[
        FieldCondition(key="gov_agencies", match=MatchValue(value="CDC"))
    ]
)
```

### Combine with Article Type
```python
filter = Filter(
    must=[
        FieldCondition(key="is_gov_affiliated", match=MatchValue(value=True)),
        FieldCondition(key="article_type", match=MatchValue(value="practice_guideline"))
    ]
)
```

## Detected Government Agencies

The following agencies are detected in author affiliations:

| Agency Code | Full Name |
|-------------|-----------|
| NIH | National Institutes of Health |
| NCI | National Cancer Institute |
| NIAID | National Institute of Allergy and Infectious Diseases |
| NHLBI | National Heart, Lung, and Blood Institute |
| NIMH | National Institute of Mental Health |
| NIDDK | National Institute of Diabetes and Digestive and Kidney Diseases |
| NIA | National Institute on Aging |
| NICHD | National Institute of Child Health and Human Development |
| NINDS | National Institute of Neurological Disorders and Stroke |
| NEI | National Eye Institute |
| NHGRI | National Human Genome Research Institute |
| NLM | National Library of Medicine |
| CDC | Centers for Disease Control and Prevention |
| FDA | Food and Drug Administration |
| VA | Veterans Affairs |
| AHRQ | Agency for Healthcare Research and Quality |
| USUHS | Uniformed Services University of the Health Sciences |

## Deprecation Notice

The following scripts have been **removed**:

| Script | Status | Replacement |
|--------|--------|-------------|
| `10_download_gov_abstracts.py` | ❌ Removed | Use `20_download_pubmed_baseline.py` |
| `13_ingest_gov_abstracts.py` | ❌ Removed | Use `21_ingest_pubmed_abstracts.py` |

These scripts have been deleted as the functionality is now fully integrated into the unified PubMed pipeline.

## Benefits of Merge

1. **Single Download**: No need to download ~31GB baseline twice
2. **Consistent Processing**: Same chunking, embedding, validation for all PubMed articles
3. **Richer Metadata**: Articles can have BOTH high-value type AND gov affiliation
4. **Simpler Queries**: One source field, filter by metadata
5. **Better Deduplication**: No risk of same PMID from different pipelines

## Rollback Plan

If you need to rollback:

1. Keep old `pubmed_gov` source documents (don't delete them yet)
2. Re-run old ingestion scripts if needed
3. Update queries to use `source="pubmed_gov"` again

## Support

For questions about the migration:
- Check existing documents with: `source="pubmed_abstract" AND is_gov_affiliated=true`
- Verify agency detection: `gov_agencies` field in payload
- Contact: See team wiki
