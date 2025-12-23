"""
ScholarQA-Style Reranker with Paper Aggregation.

Implements ai2-scholarqa-lib's exact reranking methodology:
- Passage-level reranking using Cohere (substitute for CrossEncoder)
- Paper aggregation using max rerank_score
- DataFrame output with ScholarQA reference_string format

No custom criteria (evidence hierarchy, article count limits, etc.)
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import cohere
from anyascii import anyascii
import yaml

from .config import COHERE_API_KEY

# Try to import medical entity expander for entity matching
try:
    from .medical_entity_expander import MedicalEntityExpander
    ENTITY_EXPANDER_AVAILABLE = True
except ImportError:
    ENTITY_EXPANDER_AVAILABLE = False
    MedicalEntityExpander = None

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions (from ScholarQA)
# =============================================================================

def make_int(value) -> int:
    """Convert value to int, returning 0 if not possible."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def get_ref_author_str(authors) -> str:
    """
    Format authors for reference string (ScholarQA format).
    
    Returns: "LastName, LastName2 et al." or "LastName"
    """
    if not authors:
        return "Unknown"
    
    if isinstance(authors, str):
        # Simple string format
        names = [a.strip() for a in authors.split(",")]
        if len(names) >= 2:
            return f"{names[0].split()[-1]}, {names[1].split()[-1]} et al."
        return names[0].split()[-1] if names else "Unknown"
    
    if isinstance(authors, list):
        author_names = []
        for a in authors[:2]:  # First 2 authors
            if isinstance(a, dict):
                name = a.get("name", "Unknown")
            else:
                name = str(a)
            # Get last name
            author_names.append(name.split()[-1] if name else "Unknown")
        
        if len(author_names) >= 2:
            return f"{author_names[0]}, {author_names[1]} et al."
        elif author_names:
            return author_names[0]
    
    return "Unknown"


# =============================================================================
# Abstract Reranker Interface (matches ScholarQA)
# =============================================================================

# =============================================================================
# SIMPLIFIED 3-TIER EVIDENCE SYSTEM
# =============================================================================

# Tier multipliers - much stronger differentiation
TIER_1_BOOST = 1.80  # Guidelines, systematic reviews, meta-analyses
TIER_2_BOOST = 1.25  # RCTs, clinical trials, review articles
TIER_3_BOOST = 1.00  # Standard research
TIER_4_PENALTY = 0.50  # Case reports, letters, editorials

# Article types for each tier
TIER_1_TYPES = {
    "guideline", "practice_guideline", "practice guideline",
    "systematic_review", "systematic review", 
    "meta_analysis", "meta-analysis",
    "consensus", "consensus_statement"
}
TIER_2_TYPES = {
    "clinical_trial", "clinical trial",
    "randomized_controlled_trial", "rct",
    "review_article", "review article", "review",
    "drug_label", "drug label"
}
TIER_4_TYPES = {
    "case_report", "case report",
    "case_series", "case series",
    "letter", "editorial", "comment", "commentary",
    "news", "correspondence"
}

# Title keywords that indicate guidelines (for misclassified articles)
GUIDELINE_TITLE_KEYWORDS = [
    "guideline", "guidelines", 
    "consensus", 
    "recommendation", "recommendations",
    "position statement",
    "clinical practice",
    "acr/vasculitis",  # ACR guidelines
    "aha/acc",  # American Heart Association
]

# High-impact medical journals
HIGH_IMPACT_JOURNALS = {
    "new england journal of medicine", "nejm",
    "the lancet", "lancet",
    "jama", "journal of the american medical association",
    "bmj", "british medical journal",
    "nature medicine",
    "annals of internal medicine",
    "circulation",
    "blood",
    "jama internal medicine",
    "the lancet oncology", "lancet oncology",
    "journal of clinical oncology",
    "european heart journal",
    "gastroenterology",
    "hepatology",
    "diabetes care",
    "chest",
    "arthritis & rheumatology",
    "arthritis care & research",
    "annals of the rheumatic diseases",
    "dailymed",
}


def get_evidence_multiplier(article_type: str, title: str) -> float:
    """
    Get evidence tier multiplier based on article type and title.
    
    Uses 3-tier system with title-based guideline detection.
    """
    article_type_lower = article_type.lower().replace("-", "_").replace(" ", "_") if article_type else ""
    title_lower = title.lower() if title else ""
    
    # Title-based detection first (catches misclassified guidelines)
    if any(kw in title_lower for kw in GUIDELINE_TITLE_KEYWORDS):
        return TIER_1_BOOST
    
    # Type-based tier assignment
    if article_type_lower in TIER_1_TYPES or any(t in article_type_lower for t in ["guideline", "systematic", "meta"]):
        return TIER_1_BOOST
    
    if article_type_lower in TIER_2_TYPES or any(t in article_type_lower for t in ["trial", "review"]):
        return TIER_2_BOOST
    
    if article_type_lower in TIER_4_TYPES or any(t in article_type_lower for t in ["case", "letter", "editorial"]):
        return TIER_4_PENALTY
    
    return TIER_3_BOOST  # Default for standard research


def apply_evidence_boosts(
    documents: list,
    current_year: int = 2025
) -> list:
    """
    Apply SIMPLIFIED evidence hierarchy boosts to reranked documents.
    
    Uses 3-tier system:
    - Tier 1 (1.80x): Guidelines, systematic reviews, meta-analyses
    - Tier 2 (1.25x): RCTs, clinical trials, review articles
    - Tier 3 (1.00x): Standard research
    - Tier 4 (0.50x): Case reports, letters, editorials
    
    Plus recency and high-impact journal boosts.
    """
    for doc in documents:
        # Use combined_score (includes entity matching) as base
        base_score = doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.5)))
        
        # 1. Evidence tier multiplier
        tier_mult = get_evidence_multiplier(
            doc.get("article_type", ""),
            doc.get("title", "")
        )
        
        # 2. Recency boost (recent articles +5%)
        recency_mult = 1.0
        year = doc.get("year")
        if year:
            try:
                year_int = int(year)
                if year_int >= current_year - 2:  # 2023-2025
                    recency_mult = 1.05
            except (ValueError, TypeError):
                pass
        
        # 3. High-impact journal boost (+15%)
        journal_mult = 1.0
        journal = (doc.get("journal") or doc.get("venue") or "").lower()
        if any(j in journal for j in HIGH_IMPACT_JOURNALS):
            journal_mult = 1.15
        
        # Calculate final boosted score
        doc["boosted_score"] = base_score * tier_mult * recency_mult * journal_mult
        doc["evidence_tier"] = tier_mult
        doc["boost_multiplier"] = tier_mult * recency_mult * journal_mult
    
    return documents



class AbstractReranker:
    """Abstract base class for rerankers."""
    
    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        raise NotImplementedError


# =============================================================================
# Cohere Reranker (substitute for CrossEncoder)
# =============================================================================

class CohereReranker(AbstractReranker):
    """
    Cohere-based reranker as substitute for ScholarQA's CrossEncoder.
    
    Uses Cohere rerank-v4.0-pro (upgraded from v3.5) for better healthcare performance.
    Formats documents as YAML for optimal reranking performance.
    """
    
    def __init__(self, model: str = "rerank-v4.0-pro"):
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not set")
        
        self.client = cohere.ClientV2(api_key=COHERE_API_KEY)
        self.model = model
        logger.info(f"✅ Cohere Reranker initialized with {model}")
    
    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Get relevance scores for documents.
        
        Formats documents as YAML for optimal performance (Cohere best practice).
        
        Args:
            query: Search query
            documents: List of document strings (can be plain text or YAML)
            
        Returns:
            List of relevance scores
        """
        if not documents:
            return []
        
        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=len(documents),  # Return all scores (Cohere best practice)
            )
            
            # Create score array indexed by original position
            scores = [0.0] * len(documents)
            for result in response.results:
                scores[result.index] = result.relevance_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            # Return default scores
            return [0.5] * len(documents)
    
    def format_document_as_yaml(self, title: str, content: str, abstract: str = "") -> str:
        """
        Format document as YAML string for optimal Cohere reranking.
        
        Args:
            title: Document title
            content: Document content (abstract or full text)
            abstract: Optional abstract (if content is full text)
            
        Returns:
            YAML-formatted string
        """
        import yaml
        
        doc_dict = {
            "Title": title,
        }
        
        if abstract:
            doc_dict["Abstract"] = abstract
            doc_dict["Content"] = content
        else:
            doc_dict["Content"] = content
        
        return yaml.dump(doc_dict, sort_keys=False, allow_unicode=True)


# =============================================================================
# Paper Finder with Reranker (matches ScholarQA's PaperFinderWithReranker)
# =============================================================================

class PaperFinderWithReranker:
    """
    ScholarQA-style paper finder with reranking.
    
    Flow:
    1. Rerank passages using Cohere
    2. Aggregate passages to paper level (max score)
    3. Format into DataFrame with reference strings
    """
    
    def __init__(
        self,
        reranker: AbstractReranker = None,
        n_rerank: int = -1,
        context_threshold: float = 0.0
    ):
        """
        Initialize paper finder.
        
        Args:
            reranker: Reranker engine (defaults to Cohere)
            n_rerank: Max passages to keep after rerank (-1 = all)
            context_threshold: Min score threshold
        """
        self.reranker_engine = reranker or CohereReranker()
        self.n_rerank = n_rerank
        self.context_threshold = context_threshold
        
        # Initialize medical entity expander for entity matching
        self.entity_expander = None
        if ENTITY_EXPANDER_AVAILABLE:
            try:
                self.entity_expander = MedicalEntityExpander()
                logger.info("✅ Medical entity expander initialized for reranking")
            except Exception as e:
                logger.warning(f"Medical entity expander not available for reranking: {e}")
    
    def rerank(
        self,
        query: str,
        retrieved_ctxs: List[Dict[str, Any]],
        pre_filter_threshold: float = 0.15,  # Lowered: hybrid RRF scores are normalized to 0.3-1.0
        cohere_relevance_threshold: float = 0.1  # Cohere minimum relevance (filters truly irrelevant)
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages using Cohere + evidence hierarchy boosts.
        
        Threshold Strategy (based on Cohere best practices):
        1. pre_filter_threshold (0.15): Filters by retrieval score BEFORE Cohere
           - Reduces API costs by removing obviously low-quality candidates
           - Set lower than post-rerank threshold to preserve recall
        
        2. cohere_relevance_threshold (0.1): Filters by Cohere rerank score AFTER reranking
           - Cohere scores are 0-1 normalized; <0.1 indicates very low relevance
           - This removes documents Cohere considers irrelevant regardless of retrieval score
        
        3. context_threshold (from __init__): Filters AFTER aggregation by boosted_score
           - Applied in aggregate_snippets_to_papers() using final combined scores
           - Controls the quality floor for papers included in context
        
        Pipeline:
        1. Pre-filter by retrieval score (reduces Cohere API cost)
        2. Cohere reranking for semantic relevance  
        3. Post-filter by Cohere relevance score (removes irrelevant docs)
        4. Calculate entity matching scores
        5. Apply evidence hierarchy boosts (article type, recency, journal)
        6. Sort by boosted_score
        """
        if not retrieved_ctxs:
            return []
        
        original_count = len(retrieved_ctxs)
        logger.info(f"Reranking {original_count} passages...")
        
        # Stage 1: Pre-filter by retrieval score (before Cohere API call)
        # This reduces API cost while preserving candidates for Cohere's judgment
        if pre_filter_threshold > 0:
            filtered_ctxs = [
                ctx for ctx in retrieved_ctxs 
                if ctx.get("score", 0.5) >= pre_filter_threshold
            ]
            if len(filtered_ctxs) < len(retrieved_ctxs):
                logger.info(f"   Pre-filter (retrieval score ≥{pre_filter_threshold}): {original_count} → {len(filtered_ctxs)} passages")
                retrieved_ctxs = filtered_ctxs
        
        # If no passages pass filter, return empty
        if not retrieved_ctxs:
            logger.warning("All passages filtered out by pre-filter, returning empty")
            return []

        
        # Format documents as YAML for optimal Cohere reranking (best practice)
        passages = []
        for doc in retrieved_ctxs:
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            text = doc.get("text", "")
            full_text = doc.get("full_text", "")
            
            # Use full text if available, otherwise use abstract/text
            content = full_text if full_text else (text if text else abstract)
            
            # Format as YAML for better performance
            if isinstance(self.reranker_engine, CohereReranker):
                yaml_doc = self.reranker_engine.format_document_as_yaml(
                    title=title,
                    content=content[:2000],  # Limit content length
                    abstract=abstract[:500] if abstract else ""
                )
                passages.append(yaml_doc)
            else:
                # Fallback to plain text
                passages.append((title + " " + content).strip()[:2000])
        
        # Stage 2: Get rerank scores from Cohere (0-1 normalized, query-dependent)
        rerank_scores = self.reranker_engine.get_scores(query, passages)
        
        # Attach rerank scores to contexts for filtering
        for i, score in enumerate(rerank_scores):
            retrieved_ctxs[i]["rerank_score"] = score
        
        # Stage 3: Post-filter by Cohere relevance score
        # Cohere best practice: scores <0.1 indicate very low relevance
        pre_cohere_count = len(retrieved_ctxs)
        if cohere_relevance_threshold > 0:
            retrieved_ctxs = [
                ctx for ctx in retrieved_ctxs
                if ctx.get("rerank_score", 0) >= cohere_relevance_threshold
            ]
            if len(retrieved_ctxs) < pre_cohere_count:
                logger.info(f"   Post-filter (Cohere score ≥{cohere_relevance_threshold}): {pre_cohere_count} → {len(retrieved_ctxs)} passages")
        
        if not retrieved_ctxs:
            logger.warning("All passages filtered out by Cohere relevance threshold, returning empty")
            return []
        
        # Stage 4: Calculate entity matching scores
        entity_scores = self._calculate_entity_scores(query, retrieved_ctxs)
        
        # Combine Cohere rerank scores with entity matching scores
        # Weight: 70% rerank score, 30% entity score
        combined_scores = []
        for i, entity_score in enumerate(entity_scores):
            rerank_score = retrieved_ctxs[i].get("rerank_score", 0)
            combined_score = 0.7 * rerank_score + 0.3 * entity_score
            combined_scores.append(combined_score)
            retrieved_ctxs[i]["entity_score"] = entity_score
            retrieved_ctxs[i]["combined_score"] = combined_score
        
        logger.info(f"Cohere top scores: {sorted(rerank_scores, reverse=True)[:5]}")
        logger.info(f"Entity scores: {[round(s, 3) for s in entity_scores[:5]]}")
        logger.info(f"Combined scores: {[round(s, 3) for s in combined_scores[:5]]}")
        
        # Apply evidence hierarchy boosts
        retrieved_ctxs = apply_evidence_boosts(retrieved_ctxs)
        
        # Log boost effects
        boosted = [(d.get("article_type", "?"), d.get("boost_multiplier", 1.0)) for d in retrieved_ctxs[:5]]
        logger.info(f"Evidence boosts applied: {boosted}")
        
        # Sort by boosted_score (includes evidence hierarchy and entity matching)
        # Prefer combined_score if available, otherwise use boosted_score
        sorted_ctxs = sorted(
            retrieved_ctxs,
            key=lambda x: x.get("boosted_score", x.get("combined_score", x.get("rerank_score", 0))),
            reverse=True
        )
        
        # Apply n_rerank limit if set
        if self.n_rerank > 0:
            sorted_ctxs = sorted_ctxs[:self.n_rerank]
        
        top_scores = [round(d.get("boosted_score", 0), 3) for d in sorted_ctxs[:5]]
        logger.info(f"Done reranking: {len(sorted_ctxs)} passages (top scores: {top_scores})")
        return sorted_ctxs
    
    def _calculate_entity_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate entity matching scores for documents.
        
        Extracts medical entities from query and scores documents based on entity overlap.
        
        Args:
            query: Search query
            documents: List of document dictionaries
            
        Returns:
            List of entity scores (0.0 to 1.0)
        """
        if not self.entity_expander or not documents:
            return [0.0] * len(documents)
        
        # Extract medical entities from query
        query_entities = self._extract_query_entities(query)
        
        if not query_entities:
            return [0.0] * len(documents)
        
        scores = []
        for doc in documents:
            title = str(doc.get("title", "")).lower()
            abstract = str(doc.get("abstract", "")).lower()
            text = str(doc.get("text", "")).lower()
            full_text = str(doc.get("full_text", "")).lower()
            
            # Combine text for searching
            doc_text = f"{title} {abstract} {text} {full_text}".lower()
            
            # Count entity matches
            matches = 0
            for entity in query_entities:
                entity_lower = entity.lower()
                # Check for entity in document (word boundary or exact match)
                if (entity_lower in doc_text or 
                    f" {entity_lower} " in doc_text or
                    entity_lower in title):
                    matches += 1
            
            # Score: proportion of query entities found in document
            score = matches / len(query_entities) if query_entities else 0.0
            scores.append(score)
        
        return scores
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract medical entities from query for entity-based scoring.
        
        Improved extraction using:
        1. Key medical terms (capitalized words, hyphenated terms)
        2. MeSH acronym expansion
        3. Disease/condition pattern matching
        """
        import re
        entities = []
        
        if not self.entity_expander:
            # Fallback: extract key terms without MeSH
            key_terms = re.findall(r'\b[A-Z][a-z]+(?:[-][a-z]+)*\b', query)
            key_terms += re.findall(r'\b[A-Z]{2,}\d*\b', query)  # Acronyms
            logger.info(f"Entity extraction (no MeSH): {key_terms}")
            return list(set(key_terms))
        
        # 1. Extract hyphenated medical terms (e.g., "IgG4-related")
        hyphenated = re.findall(r'\b\w+(?:-\w+)+\b', query)
        entities.extend(hyphenated)
        
        # 2. Extract acronyms (2+ uppercase letters, optionally with numbers)
        acronyms = re.findall(r'\b[A-Z]{2,}\d*\b', query)
        for acr in acronyms:
            entities.append(acr)
            # Try to expand acronym using MeSH
            expansions = self.entity_expander.expand_acronym(acr)
            if expansions:
                entities.append(expansions[0])
        
        # 3. Extract capitalized multi-word terms (e.g., "Antiphospholipid Syndrome")
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend([p for p in capitalized_phrases if len(p) > 3])
        
        # 4. Extract disease-related patterns more flexibly
        disease_patterns = [
            r'\b\w+[-]?related\s+disease\b',    # "IgG4-related disease"
            r'\b\w+\s+disease\b',               # "X disease"
            r'\b\w+\s+syndrome\b',              # "X syndrome"
            r'\b\w+\s+disorder\b',              # "X disorder"
            r'\b\w+itis\b',                     # "arthritis", etc.
            r'\b\w+osis\b',                     # "fibrosis", etc.
            r'\b\w+emia\b',                     # "anemia", etc.
            r'\b\w+pathy\b',                    # "neuropathy", etc.
        ]
        
        for pattern in disease_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        # 5. Also extract significant lowercase medical terms (length > 5)
        words = query.split()
        for word in words:
            clean = re.sub(r'[^\w-]', '', word)
            if len(clean) > 5 and clean.lower() not in ['management', 'treatment', 'therapy', 'diagnosis', 'clinical', 'features', 'symptoms', 'guidelines', 'recommendations']:
                entities.append(clean)
        
        # Deduplicate (case-insensitive) while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen and len(entity) > 2:
                seen.add(entity_lower)
                unique_entities.append(entity)
        
        logger.info(f"Extracted entities from query: {unique_entities}")
        return unique_entities

    
    def aggregate_snippets_to_papers(
        self,
        snippets_list: List[Dict[str, Any]],
        paper_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Aggregate passages to paper level with multi-level deduplication.
        
        Deduplication priority:
        1. DOI (most reliable - same content has same DOI across journals)
        2. Normalized title (catches sister journal publications)
        3. PMCID (fallback)
        
        Uses max rerank_score for each paper.
        """
        logger.info(f"Aggregating {len(snippets_list)} passages at paper level")
        
        paper_snippets = {}
        seen_dois = {}  # doi -> corpus_id mapping
        seen_titles = {}  # normalized_title -> corpus_id mapping
        
        def normalize_title(title: str) -> str:
            """Normalize title for comparison (lowercase, remove punctuation)."""
            import re
            if not title:
                return ""
            # Lowercase, remove punctuation, collapse whitespace
            normalized = re.sub(r'[^\w\s]', '', title.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            return normalized
        
        for snippet in snippets_list:
            # Get identifiers
            pmcid = snippet.get("pmcid") or snippet.get("corpus_id", "")
            doi = snippet.get("doi", "")
            title = snippet.get("title", "")
            normalized_title = normalize_title(title)
            
            if not pmcid and not doi and not normalized_title:
                continue
            
            # Determine unique key using multi-level deduplication
            # Priority: DOI > Title > PMCID
            corpus_id = pmcid
            is_duplicate = False
            duplicate_of = None
            
            # 1. Check if we've seen this DOI before
            if doi and doi in seen_dois:
                is_duplicate = True
                duplicate_of = seen_dois[doi]
                logger.debug(f"DOI duplicate: {doi} -> {duplicate_of}")
            
            # 2. Check if we've seen this title before (for sister journal publications)
            elif normalized_title and len(normalized_title) > 30 and normalized_title in seen_titles:
                is_duplicate = True
                duplicate_of = seen_titles[normalized_title]
                logger.debug(f"Title duplicate: '{title[:50]}...' -> {duplicate_of}")
            
            if is_duplicate and duplicate_of:
                # Merge with existing paper - use higher score
                if duplicate_of in paper_snippets:
                    current_score = paper_snippets[duplicate_of]["relevance_judgement"]
                    new_score = snippet.get("boosted_score", snippet.get("rerank_score", snippet.get("score", 0)))
                    paper_snippets[duplicate_of]["relevance_judgement"] = max(current_score, new_score)
                    # Add snippet to existing paper
                    paper_snippets[duplicate_of]["sentences"].append({
                        "text": snippet.get("text", ""),
                        "section_title": snippet.get("section_title", "abstract"),
                        "char_start_offset": snippet.get("char_offset", 0),
                    })
                continue
            
            # Not a duplicate - create new entry
            if corpus_id not in paper_snippets:
                paper_snippets[corpus_id] = {
                    "corpus_id": corpus_id,
                    "pmcid": corpus_id,
                    "pmid": snippet.get("pmid"),
                    "doi": doi,
                    "title": title,
                    "abstract": snippet.get("abstract", ""),
                    "venue": snippet.get("journal") or snippet.get("venue", ""),
                    "year": snippet.get("year"),
                    "authors": snippet.get("authors", []),
                    "article_type": snippet.get("article_type", "other"),
                    "sentences": [],
                    "relevance_judgement": -1,
                    "citation_count": snippet.get("citation_count", 0),
                }
                # Track this DOI and title for deduplication
                if doi:
                    seen_dois[doi] = corpus_id
                if normalized_title and len(normalized_title) > 30:
                    seen_titles[normalized_title] = corpus_id
            
            # Add sentence/snippet
            paper_snippets[corpus_id]["sentences"].append({
                "text": snippet.get("text", ""),
                "section_title": snippet.get("section_title", "abstract"),
                "char_start_offset": snippet.get("char_offset", 0),
            })
            
            # Update relevance using max boosted_score (evidence hierarchy aware)
            current_score = paper_snippets[corpus_id]["relevance_judgement"]
            new_score = snippet.get("boosted_score", snippet.get("rerank_score", snippet.get("score", 0)))
            paper_snippets[corpus_id]["relevance_judgement"] = max(current_score, new_score)
            
            # Update abstract if from abstract section
            if snippet.get("section_title") == "abstract" and not paper_snippets[corpus_id]["abstract"]:
                paper_snippets[corpus_id]["abstract"] = snippet.get("text", "")

        
        # Log deduplication stats
        total_snippets = len(snippets_list)
        unique_papers = len(paper_snippets)
        duplicates_removed = total_snippets - sum(len(p["sentences"]) for p in paper_snippets.values())
        if duplicates_removed > 0:
            logger.info(f"   Deduplication: {total_snippets} passages → {unique_papers} unique papers ({duplicates_removed} duplicates removed)")
        
        # Sort by relevance

        sorted_papers = sorted(
            paper_snippets.values(),
            key=lambda x: x["relevance_judgement"],
            reverse=True
        )
        
        # Apply context threshold
        sorted_papers = [
            p for p in sorted_papers
            if p["relevance_judgement"] >= self.context_threshold
        ]
        
        logger.info(f"Aggregated to {len(sorted_papers)} papers")
        logger.info(f"Scores: {[p['relevance_judgement'] for p in sorted_papers[:10]]}")
        
        return sorted_papers
    
    def format_retrieval_response(
        self,
        agg_candidates: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Format into DataFrame with ScholarQA reference strings.
        
        Creates:
        - reference_string: "[corpus_id | author | year | Citations: N]"
        - relevance_judgment_input_expanded: Formatted paper content
        """
        if not agg_candidates:
            return pd.DataFrame()
        
        def format_sections_to_markdown(sentences: List[Dict]) -> str:
            """Format paper sections to markdown."""
            if not sentences:
                return ""
            
            df = pd.DataFrame(sentences)
            if df.empty:
                return ""
            
            # Sort by offset
            if "char_start_offset" in df.columns:
                df = df.sort_values(by="char_start_offset")
            
            # Group by section
            grouped = df.groupby("section_title", sort=False)["text"].apply("\n...\n".join)
            
            # Exclude abstract (already in prepend)
            grouped = grouped[~grouped.index.isin(["abstract", "title"])]
            
            return "\n\n".join(f"## {title}\n{text}" for title, text in grouped.items())
        
        df = pd.DataFrame(agg_candidates)
        
        if df.empty:
            return df
        
        # Format authors
        df["authors"] = df["authors"].fillna("").apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        # Create prepend text (title, venue, authors, abstract)
        prepend_text = df.apply(
            lambda row: (
                f"# Title: {row['title']}\n"
                f"# Venue: {row['venue']}\n"
                f"# Authors: {', '.join([a.get('name', a) if isinstance(a, dict) else str(a) for a in row['authors']])}\n"
                f"## Abstract\n{row['abstract']}\n"
            ),
            axis=1
        )
        
        # Format sections
        section_text = df["sentences"].apply(format_sections_to_markdown)
        
        # Create expanded input
        df["relevance_judgment_input_expanded"] = prepend_text + section_text
        
        # Create reference string (ScholarQA format with PMCID)
        df["reference_string"] = df.apply(
            lambda row: anyascii(
                f"[{row['pmcid']} | {get_ref_author_str(row['authors'])} | "
                f"{make_int(row['year'])} | Citations: {make_int(row.get('citation_count', 0))}]"
            ),
            axis=1
        )
        
        logger.info(f"Formatted {len(df)} papers into DataFrame")
        return df
    
    def aggregate_into_dataframe(
        self,
        snippets_list: List[Dict[str, Any]],
        paper_metadata: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Full pipeline: aggregate then format (ScholarQA interface).
        """
        aggregated = self.aggregate_snippets_to_papers(snippets_list, paper_metadata)
        return self.format_retrieval_response(aggregated)


# For backward compatibility
ArticleReranker = PaperFinderWithReranker


if __name__ == "__main__":
    print("🧪 Testing ScholarQA-style Reranker")
    print("=" * 60)
    
    # Test reranker
    reranker = CohereReranker()
    
    sample_passages = [
        {"pmcid": "PMC001", "title": "COPD Treatment", "text": "COPD management...", "year": 2023, "journal": "Lancet", "authors": [{"name": "Smith J"}]},
        {"pmcid": "PMC002", "title": "Diabetes Care", "text": "Type 2 diabetes...", "year": 2024, "journal": "NEJM", "authors": [{"name": "Doe A"}, {"name": "Lee B"}]},
    ]
    
    finder = PaperFinderWithReranker()
    reranked = finder.rerank("COPD management", sample_passages)
    print(f"Reranked {len(reranked)} passages")
    
    df = finder.aggregate_into_dataframe(reranked)
    print(f"DataFrame: {len(df)} rows")
    print(f"Reference strings: {df['reference_string'].tolist()}")
