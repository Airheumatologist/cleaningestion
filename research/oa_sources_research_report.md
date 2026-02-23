# Open Access Medical Research Sources - Expansion Research Report

## Executive Summary

This report analyzes potential sources for expanding the Elixir AI medical research assistant's knowledge base. The analysis focuses on sources that are **open access (OA)** and allow **commercial use**, which is critical for your commercial medical RAG pipeline.

### Current Sources (Baseline)
1. **PubMed Central (PMC)** - OA subset + Author Manuscripts (via AWS S3)
2. **DailyMed** - FDA drug labels
3. **PubMed** - Abstracts (filtered: reviews, trials, guidelines, government-affiliated)

---

## Recommended Additional Sources

### 1. EUROPE PMC (European PubMed Central)

**Overview:**
- Europe's equivalent of PMC with broader international coverage
- ~11.5 million full-text articles, ~7.6 million in OA subset
- Updated weekly with full-text XML

**License:**
- Creative Commons licenses (varies by article: CC BY, CC BY-NC, etc.)
- **CAUTION:** Not all articles are CC BY - must filter for commercial use

**Access:**
- RESTful API: https://www.ebi.ac.uk/europepmc/webservices/rest/
- FTP bulk download: https://europepmc.org/downloads/openaccess
- OAI-PMH service for harvesting

**Value for Medical RAG:**
- ✅ European research coverage (complements US-centric PMC)
- ✅ Many CC-BY articles available
- ✅ Structured XML format (similar to PMC)
- ⚠️ Must filter by license type for commercial use

**Implementation Notes:**
```
# Filter for CC-BY licensed content only
# Check license statement in each article's metadata
```

---

### 2. PLOS (Public Library of Science)

**Overview:**
- 12 peer-reviewed OA journals
- Strong focus on medical/life sciences
- ~250,000+ articles

**License:**
- **CC BY 4.0** (fully commercial use allowed)
- Authors retain copyright
- "Anyone can reuse your article in whole or part for any purpose, for free, even for commercial purposes"

**Access:**
- API: https://api.plos.org/
- Bulk download available
- XML and PDF formats

**Value for Medical RAG:**
- ✅ **Fully CC BY** - no license ambiguity
- ✅ High-quality peer-reviewed content
- ✅ Rich metadata and structured abstracts
- ✅ Medicine, Biology, Computational Biology, Genetics, Neglected Tropical Diseases

---

### 3. OpenAlex

**Overview:**
- Open catalog of 474M+ scholarly works
- 60M+ full-text PDFs available
- Merges data from PubMed, arXiv, Crossref, Unpaywall

**License:**
- **CC0** (public domain dedication)
- "All data is CC0—use it anywhere without attribution"
- Full open source code

**Access:**
- REST API: 100,000 credits/day (free)
- Bulk data snapshots (free to download entire dataset)
- Content download API for PDFs (100 credits per file)
- CLI tool: `openalex-official`

**Value for Medical RAG:**
- ✅ **CC0 license** - maximum freedom
- ✅ Massive scale (474M works)
- ✅ Open access status tracking
- ✅ Links to OA PDFs from any repository
- ✅ Topic classifications, citation data
- ✅ SPECTER2 embeddings available

**Implementation Notes:**
```python
# Example: Download medical OA papers
openalex download \
  --api-key YOUR_KEY \
  --filter "topics.id:T10325,is_oa:true" \
  --content pdf
```

---

### 4. Semantic Scholar (S2ORC + Academic Graph)

**Overview:**
- 214M+ papers, 2.49B citations
- S2ORC: Full-text corpus with machine-readable content
- SPECTER2 embeddings available

**License:**
- **ODC-By 1.0** (Open Data Commons Attribution License)
- Allows commercial use with attribution
- Citation required: "S2ORC: The Semantic Scholar Open Research Corpus"

**Access:**
- REST API (free, rate-limited)
- Bulk dataset downloads via API
- JSON format

**Value for Medical RAG:**
- ✅ ODC-By license allows commercial use
- ✅ Full-text available for many papers
- ✅ SPECTER2 embeddings for semantic search
- ✅ Citation graphs for importance ranking
- ✅ Abstracts available for nearly all papers

---

### 5. bioRxiv / medRxiv (Preprint Servers)

**Overview:**
- **medRxiv**: Dedicated health sciences preprint server
- **bioRxiv**: Biology preprints (relevant for biomedical research)
- Combined: 325,000+ preprints
- Now managed by independent nonprofit openRxiv

**License Options (Author-Selected):**
- CC0 (public domain)
- **CC BY** (commercial OK)
- CC BY-NC (non-commercial only - exclude)
- CC BY-ND (no derivatives)
- CC BY-NC-ND (exclude)
- No license (exclude)

**Access:**
- API available
- Bulk download via OAI-PMH
- XML, PDF, HTML formats

**Value for Medical RAG:**
- ✅ **Early access** to cutting-edge research
- ✅ Authors can choose CC BY
- ✅ Indexed by PubMed, Europe PMC, Google Scholar
- ⚠️ Preprints are NOT peer-reviewed - flag appropriately
- ⚠️ Must filter for CC BY license only

**Important Consideration:**
```
For medical research assistant:
- Flag preprints clearly in UI: "⚠️ PREPRINT - Not peer-reviewed"
- Useful for "latest research" queries but with appropriate caveats
- Filter for CC BY only (exclude NC variants)
```

---

### 6. Cochrane Library (Systematic Reviews)

**Overview:**
- Gold standard for systematic reviews
- 10,000+ Cochrane Reviews
- Free in 100+ low/middle-income countries
- 12-month embargo then free (green OA)

**License:**
- **CC BY-NC** (non-commercial) for most content
- Some national provisions may allow broader access
- Protocols (from Nov 2024) are **CC BY**

**Access:**
- Limited API access via Wiley
- Some content in PubMed Central
- National provisions for bulk access

**Value for Medical RAG:**
- ✅ Highest quality systematic reviews
- ⚠️ **NON-COMMERCIAL license** (CC BY-NC)
- ⚠️ **CANNOT use** for commercial medical assistant without license
- Consider: Negotiate commercial license with Cochrane/Wiley

**Verdict:** ❌ Not suitable for current commercial use without additional licensing

---

### 7. DOAJ (Directory of Open Access Journals)

**Overview:**
- 17,000+ OA journals
- 6M+ articles indexed
- Community-curated quality

**License:**
- DOAJ metadata: **CC0** (waived all rights)
- Article licenses vary by journal
- Can filter for CC BY journals

**Access:**
- REST API
- OAI-PMH harvest
- Public data dumps
- CSV exports

**Value for Medical RAG:**
- ✅ CC0 metadata
- ✅ Can identify CC BY medical journals
- ✅ Quality-assured journals only
- ✅ Search by license type

**Implementation:**
```
Use DOAJ to identify CC BY medical journals
Then crawl/harvest from those specific journals
```

---

### 8. ClinicalTrials.gov

**Overview:**
- 270,000+ registered studies
- 30,000+ with posted results
- US government database

**License:**
- **Public domain** (US government work)
- Free to use for any purpose

**Access:**
- REST API: https://clinicaltrials.gov/data-api/api
- Bulk download in XML format
- No rate limits for reasonable use

**Value for Medical RAG:**
- ✅ **Public domain** - no restrictions
- ✅ Study results (not just abstracts)
- ✅ Adverse event data
- ✅ Protocol information
- ✅ Links to publications

**Use Case:**
```
- "What are the results of trial NCT12345678?"
- "Show me trials for drug X in condition Y"
- Cross-reference with published literature
```

---

### 9. WHO Global Health Observatory

**Overview:**
- WHO's health statistics database
- 1,000+ health indicators
- 194 member states

**License:**
- **CC BY** (confirmed in metadata)
- Can be used for commercial purposes with attribution

**Access:**
- API available
- Bulk data downloads
- CSV, JSON formats

**Value for Medical RAG:**
- ✅ Global health statistics
- ✅ Disease burden data
- ✅ Health system indicators
- ✅ SDG health targets

---

### 10. NHS Evidence / NICE Guidelines

**Overview:**
- UK National Institute for Health and Care Excellence
- Clinical guidelines, technology assessments
- Publicly funded

**License:**
- **Open Government License (OGL)** - UK
- Allows commercial use with attribution
- "You are free to: copy, publish, distribute and transmit the Information; adapt the Information; exploit the Information commercially"

**Access:**
- API available
- Bulk downloads
- XML/PDF formats

**Value for Medical RAG:**
- ✅ High-quality clinical guidelines
- ✅ Cost-effectiveness data
- ✅ Technology appraisals
- ✅ UK/European perspective

---

### 11. PubMed Central Canada / International

**Overview:**
- Other national PMC instances
- Similar to US PMC but with regional focus

**License:**
- Same as US PMC - CC BY or compatible licenses

**Access:**
- APIs available for each national instance

**Value for Medical RAG:**
- ✅ Regional coverage
- ✅ Same quality standards as US PMC

---

## Sources to AVOID (Non-Commercial Only)

| Source | License | Reason |
|--------|---------|--------|
| Cochrane Reviews | CC BY-NC | Non-commercial only |
| Some SpringerNature OA | CC BY-NC | Check individual articles |
| Some Elsevier OA | CC BY-NC | Check individual articles |
| Wikipedia / WikiProject Med | CC BY-SA | Copyleft complications |
| Many Author Manuscripts | Varies | Often NC-only |

---

## Recommended Implementation Priority

### Phase 1: High Value, Clear Licensing ✅

1. **OpenAlex** (CC0)
   - Bulk download OA medical papers
   - Use `has_content.pdf:true` filter
   - Free API access

2. **PLOS Journals** (CC BY)
   - All 12 journals are CC BY
   - Focus on PLOS Medicine, PLOS ONE (health section)
   - API integration

3. **ClinicalTrials.gov** (Public Domain)
   - Government data, no restrictions
   - Valuable study results data
   - Free API

### Phase 2: Regional/Complementary

4. **Europe PMC** (CC BY subset)
   - Filter for CC BY only
   - European research coverage

5. **medRxiv** (CC BY subset)
   - Filter for CC BY only
   - Latest preprints (with caveats)

6. **DOAJ** (CC0 metadata)
   - Identify additional CC BY medical journals

### Phase 3: Specialized Content

7. **WHO GHO** (CC BY)
   - Health statistics

8. **NICE Guidelines** (OGL)
   - UK clinical guidelines

---

## Technical Implementation Considerations

### License Filtering Strategy

```python
# Pseudo-code for license checking
def is_commercial_safe(record):
    license = record.get('license', '').lower()
    
    # Green light licenses
    if any(l in license for l in ['cc0', 'cc-by 4.0', 'cc-by 3.0', 
                                   'public domain', 'open government']):
        return True
    
    # Red light (non-commercial)
    if 'nc' in license or 'non-commercial' in license:
        return False
    
    # Yellow - manual review needed
    return None
```

### Recommended Metadata Schema Extension

```json
{
  "source": "openalex|plos|europepmc|medrxiv|clinicaltrials",
  "license": "cc-by-4.0",
  "license_verified": true,
  "commercial_use_allowed": true,
  "is_preprint": false,
  "peer_reviewed": true,
  "evidence_tier": "systematic_review|rct|cohort|preprint"
}
```

### Storage Estimates

| Source | Estimated Articles | Avg Size per Article | Total Size |
|--------|-------------------|---------------------|------------|
| OpenAlex (medical OA) | ~10M | 50KB | ~500GB |
| PLOS | ~250K | 100KB | ~25GB |
| ClinicalTrials.gov | ~400K | 20KB | ~8GB |
| Europe PMC (CC BY) | ~2M | 100KB | ~200GB |

---

## Legal Checklist for Each Source

Before ingesting any new source:

- [ ] Verify license allows commercial use
- [ ] Document license type in metadata
- [ ] Ensure attribution requirements are met
- [ ] Check for any special terms of use
- [ ] Verify bulk download is permitted
- [ ] Document data source for audit trail

---

## Summary Table

| Source | License | Commercial OK | Volume | Quality | Priority |
|--------|---------|---------------|--------|---------|----------|
| **OpenAlex** | CC0 | ✅ Yes | 474M works | High | P1 |
| **PLOS** | CC BY | ✅ Yes | 250K | Very High | P1 |
| **ClinicalTrials.gov** | Public Domain | ✅ Yes | 270K | High | P1 |
| **Europe PMC** | Mixed | ⚠️ Filter needed | 7.6M OA | High | P2 |
| **medRxiv** | Mixed | ⚠️ Filter needed | 100K+ | Medium* | P2 |
| **DOAJ** | CC0 metadata | ✅ Yes | 6M+ | High | P2 |
| **WHO GHO** | CC BY | ✅ Yes | 1000+ indicators | High | P3 |
| **NICE** | OGL | ✅ Yes | 1000s | Very High | P3 |
| **Cochrane** | CC BY-NC | ❌ No | 10K+ | Very High | ❌ |

*Preprints are not peer-reviewed

---

## Next Steps Recommendations

1. **Start with OpenAlex bulk download** - CC0 license, massive scale, good API
2. **Integrate PLOS API** - Pure CC BY, high quality
3. **Add ClinicalTrials.gov** - Public domain, valuable trial data
4. **Implement license filtering** for any mixed-license sources (Europe PMC, medRxiv)
5. **Consider legal review** before adding any source with ambiguous licensing

---

*Report generated: 2026-02-23*
*For: Elixir AI Medical Research Assistant*
*Focus: Open Access + Commercial Use License Compatibility*
