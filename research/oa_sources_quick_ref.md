# Quick Reference: OA Sources for Commercial Medical RAG

## 🟢 READY TO IMPLEMENT (Clear Commercial License)

### 1. OpenAlex
- **License**: CC0 (public domain)
- **Size**: 474M works, 60M full-text PDFs
- **API**: 100K credits/day free, bulk download available
- **Best For**: Massive scale medical OA papers
- **CLI**: `pip install openalex-official`
- **Filter**: `is_oa:true,has_content.pdf:true`

### 2. PLOS Journals
- **License**: CC BY 4.0 (all journals)
- **Size**: 250K+ articles
- **API**: https://api.plos.org/
- **Best For**: High-quality peer-reviewed medicine
- **Key Journals**: PLOS Medicine, PLOS ONE, PLOS Biology

### 3. ClinicalTrials.gov
- **License**: Public Domain (US Gov)
- **Size**: 270K+ trials, 30K+ with results
- **API**: https://clinicaltrials.gov/data-api/api
- **Best For**: Trial results, adverse events, protocols
- **Note**: No rate limits for reasonable use

---

## 🟡 IMPLEMENT WITH CAUTION (Mixed Licenses - Filter Required)

### 4. Europe PMC
- **License**: Mixed (CC BY + CC BY-NC)
- **Size**: 7.6M OA articles
- **Action Required**: Filter for CC BY only
- **API**: https://www.ebi.ac.uk/europepmc/webservices/rest/
- **Benefit**: European research coverage

### 5. medRxiv
- **License**: Author-selected (CC BY or CC BY-NC)
- **Size**: 100K+ health preprints
- **Action Required**: Filter for CC BY only
- **Warning**: Preprints = not peer-reviewed
- **UI Flag**: "⚠️ PREPRINT - Not peer-reviewed"

---

## 🔴 DO NOT USE (Non-Commercial Only)

| Source | License | Why Exclude |
|--------|---------|-------------|
| **Cochrane Library** | CC BY-NC | Non-commercial only |
| **Many SpringerNature OA** | CC BY-NC | Check individually |
| **Many Elsevier OA** | CC BY-NC | Check individually |

---

## 📋 License Filter Logic

```python
def is_commercial_safe(license_str):
    license = license_str.lower()
    
    # ✅ SAFE
    safe = ['cc0', 'cc-by 4.0', 'cc-by 3.0', 
            'public domain', 'open government']
    if any(s in license for s in safe):
        return True
    
    # ❌ UNSAFE
    if 'nc' in license or 'non-commercial' in license:
        return False
    
    # ⚠️ REVIEW MANUALLY
    return None
```

---

## 📊 Implementation Priority

| Priority | Source | Effort | Impact |
|----------|--------|--------|--------|
| P1 | OpenAlex | Low | Massive |
| P1 | PLOS | Low | High |
| P1 | ClinicalTrials.gov | Low | Medium |
| P2 | Europe PMC | Medium | High |
| P2 | medRxiv | Medium | Medium |
| P3 | DOAJ | High | Medium |

---

## 🔗 Key URLs

- **OpenAlex Docs**: https://docs.openalex.org/
- **OpenAlex CLI**: https://github.com/ourresearch/openalex-official
- **PLOS API**: https://api.plos.org/
- **ClinicalTrials API**: https://clinicaltrials.gov/data-api/api
- **Europe PMC**: https://europepmc.org/RestfulWebService
- **medRxiv**: https://www.medrxiv.org/

---

## 💡 Pro Tips

1. **Start with OpenAlex** - CC0 means zero legal concerns
2. **Always store license info** in your metadata
3. **Flag preprints** clearly in the UI
4. **For Europe PMC/medRxiv**: Filter at download time, not query time
5. **Document everything** for audit purposes

---

*Quick Reference for Elixir AI Medical RAG*
*Focus: Commercial Use License Compliance*
