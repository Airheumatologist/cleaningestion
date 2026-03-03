"""
ScholarQA-Style Reranker with Paper Aggregation.

Implements ai2-scholarqa-lib's exact reranking methodology:
- Passage-level reranking using DeepInfra Qwen3-Reranker
- Paper aggregation using max rerank_score
- DataFrame output with ScholarQA reference_string format
"""

import logging
import re

from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from anyascii import anyascii
import requests
from .config import (
    RERANKER_MODEL,
    DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL,
    DEEPINFRA_RETRY_COUNT, DEEPINFRA_RETRY_DELAY,
    DEEPINFRA_RERANK_TIMEOUT_SECONDS,
    RERANKER_V2_ENABLED,
    TIER_1_BOOST as CFG_TIER_1_BOOST,
    TIER_2_BOOST as CFG_TIER_2_BOOST,
    TIER_3_BOOST as CFG_TIER_3_BOOST,
    TIER_4_PENALTY as CFG_TIER_4_PENALTY,
    RERANKER_SCORE_WEIGHT,
    ENTITY_SCORE_WEIGHT,
)
from .retry_utils import retry_with_exponential_backoff
from .specialty_journals import detect_guideline_society_signal, get_journal_tier

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


# =============================================================================
# Abstract Reranker Interface (matches ScholarQA)
# =============================================================================

# =============================================================================
# EVIDENCE HIERARCHY SYSTEM
# =============================================================================

# Tier multipliers - configurable via config.py / env vars.
# Legacy (v1) values kept for rollback when RERANKER_V2_ENABLED=False:
#   TIER_1=3.00  TIER_2=1.50  TIER_3=1.00  TIER_4=0.20
# v2 defaults (env-overridable): 2.00 / 1.25 / 1.00 / 0.40
_TIER_1_LEGACY, _TIER_2_LEGACY, _TIER_3_LEGACY, _TIER_4_LEGACY = 3.00, 1.50, 1.00, 0.20
TIER_1_BOOST   = CFG_TIER_1_BOOST   if RERANKER_V2_ENABLED else _TIER_1_LEGACY
TIER_2_BOOST   = CFG_TIER_2_BOOST   if RERANKER_V2_ENABLED else _TIER_2_LEGACY
TIER_3_BOOST   = CFG_TIER_3_BOOST   if RERANKER_V2_ENABLED else _TIER_3_LEGACY
TIER_4_PENALTY = CFG_TIER_4_PENALTY if RERANKER_V2_ENABLED else _TIER_4_LEGACY

# =============================================================================
# ENTITY EXTRACTION CONSTANTS
# =============================================================================

# Generic medical terms that add noise to entity matching (not specific disease terms)
ENTITY_STOPWORDS = {
    'management', 'treatment', 'therapy', 'diagnosis', 'clinical',
    'features', 'symptoms', 'guidelines', 'recommendations',
    'patients', 'patient', 'outcomes', 'approach', 'advances',
    'review', 'overview', 'practice', 'evidence', 'recent',
    'current', 'update', 'updates', 'prevention', 'screening',
    'assessment', 'evaluation', 'classification', 'pathogenesis',
    'epidemiology', 'prognosis', 'mechanism', 'pathophysiology',
    'presentation', 'complications', 'associated', 'chronic',
    'acute', 'primary', 'secondary', 'emerging', 'novel', 'standard',
    'latest', 'options', 'strategies', 'interventions',
}

# Suffixes that indicate a word is likely a medical term even if lowercase
MEDICAL_SUFFIXES = ('itis', 'osis', 'emia', 'pathy', 'oma', 'ectomy', 'plasty', 'scopy')

# Article types for each tier
TIER_1_TYPES = {
    "guideline", "practice_guideline", "practice guideline",
    "systematic_review", "systematic review", 
    "meta_analysis", "meta-analysis", "meta analysis",
    "consensus", "consensus_statement", "consensus statement"
}
TIER_2_TYPES = {
    "clinical_trial", "clinical trial",
    "clinical_trial_phase_i", "clinical trial, phase i",
    "clinical_trial_phase_ii", "clinical trial, phase ii", 
    "clinical_trial_phase_iii", "clinical trial, phase iii",
    "clinical_trial_phase_iv", "clinical trial, phase iv",
    "controlled_clinical_trial", "controlled clinical trial",
    "multicenter_study", "multicenter study",
    "randomized_controlled_trial", "rct", "randomized controlled trial",
    "review_article", "review article", "review",
    "drug_label", "drug label"
}
TIER_4_TYPES = {
    "case_report", "case report",
    "case_series", "case series",
    "letter", "editorial", "comment", "commentary",
    "news", "correspondence"
}

# Title keywords that indicate guidelines (catches misclassified articles)
GUIDELINE_TITLE_KEYWORDS = [
    # Standard terms
    "guideline", "guidelines", 
    "consensus", "consensus statement",
    "recommendation", "recommendations",
    "position statement",
    "scientific statement",
    "clinical practice",
    # Evidence-based patterns
    "evidence-based", "evidence based",
    "management guidelines", "treatment guidelines",
    "international consensus",
    "standards of care",
    # Specific patterns
    "acr/vasculitis", "aha/acc",
]

def _normalize_match_text(value: Any) -> str:
    """Normalize free text for token-boundary matching."""
    return re.sub(r"[^a-z0-9]+", " ", anyascii(str(value or "")).lower()).strip()


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    text_tokens = text.split()
    phrase_tokens = phrase.split()
    if len(phrase_tokens) > len(text_tokens):
        return False
    # Exact sliding-window match (original behaviour, unchanged)
    for idx in range(len(text_tokens) - len(phrase_tokens) + 1):
        if text_tokens[idx:idx + len(phrase_tokens)] == phrase_tokens:
            return True
    # Fuzzy fallback (v2): all phrase tokens present within a gap-tolerant window
    # e.g., "non small cell lung" matches "non-small-cell lung" after normalization
    if RERANKER_V2_ENABLED:
        window = len(phrase_tokens) + 2     # allow up to 2 extra tokens between phrase words
        phrase_set = set(phrase_tokens)
        for idx in range(max(0, len(text_tokens) - window + 1)):
            if phrase_set.issubset(set(text_tokens[idx:idx + window])):
                return True
    return False


def _normalize_publication_types(publication_types: Any) -> list[str]:
    if publication_types is None:
        return []
    if isinstance(publication_types, str):
        normalized = _normalize_match_text(publication_types)
        return [normalized] if normalized else []
    if isinstance(publication_types, dict):
        candidate = (
            publication_types.get("type")
            or publication_types.get("name")
            or publication_types.get("value")
        )
        normalized = _normalize_match_text(candidate)
        return [normalized] if normalized else []
    if isinstance(publication_types, (list, tuple, set)):
        normalized_values = []
        seen = set()
        for item in publication_types:
            if isinstance(item, dict):
                candidate = item.get("type") or item.get("name") or item.get("value")
            else:
                candidate = item
            normalized = _normalize_match_text(candidate)
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_values.append(normalized)
        return normalized_values
    normalized = _normalize_match_text(publication_types)
    return [normalized] if normalized else []


def get_evidence_multiplier(
    article_type: str,
    title: str,
    publication_types: list = None,
    journal: Optional[str] = None,
    evidence_term: Optional[str] = None,
    evidence_source: Optional[str] = None,
) -> float:
    """
    Get evidence tier multiplier based on article type, title, and publication types.
    
    Priority:
    1. Publication types list (highest confidence)
    2. Society-guideline detection (context gated)
    3. Title keywords (catches misclassified guidelines)
    4. Article type field
    """
    article_type_lower = (article_type or "").lower().replace("-", "_").replace(" ", "_")
    title_normalized = _normalize_match_text(title)
    pub_types_lower = _normalize_publication_types(publication_types)

    # 1. Publication types list (from PubMed - most reliable)
    TIER1_PUBTYPE_PATTERNS = ["systematic review", "meta analysis", "guideline", "practice guideline", "consensus"]
    TIER2_PUBTYPE_PATTERNS = ["randomized controlled trial", "clinical trial", "review"]
    TIER4_PUBTYPE_PATTERNS = ["case report", "letter", "editorial", "comment", "news"]

    for pt in pub_types_lower:
        if any(_contains_normalized_phrase(pt, p) for p in TIER1_PUBTYPE_PATTERNS):
            return TIER_1_BOOST

    # 2. Society signal with guideline context (precision-first)
    society_signal = detect_guideline_society_signal(
        title=title,
        journal=journal,
        evidence_term=evidence_term,
        evidence_source=evidence_source,
        publication_types=pub_types_lower,
    )
    if society_signal["is_match"]:
        return TIER_1_BOOST

    # 3. Title-based detection (generic terms only)
    if any(_contains_normalized_phrase(title_normalized, _normalize_match_text(kw)) for kw in GUIDELINE_TITLE_KEYWORDS):
        return TIER_1_BOOST

    for pt in pub_types_lower:
        if any(_contains_normalized_phrase(pt, p) for p in TIER4_PUBTYPE_PATTERNS):
            return TIER_4_PENALTY
    for pt in pub_types_lower:
        if any(_contains_normalized_phrase(pt, p) for p in TIER2_PUBTYPE_PATTERNS):
            return TIER_2_BOOST

    # 4. Article type-based tier assignment
    if article_type_lower in TIER_1_TYPES or any(t in article_type_lower for t in ["guideline", "systematic", "meta"]):
        return TIER_1_BOOST
    
    if article_type_lower in TIER_2_TYPES or any(t in article_type_lower for t in ["trial", "review"]):
        return TIER_2_BOOST
    
    if article_type_lower in TIER_4_TYPES or any(t in article_type_lower for t in ["case", "letter", "editorial"]):
        return TIER_4_PENALTY
    
    return TIER_3_BOOST  # Default for standard research


def apply_evidence_boosts(
    documents: list,
    current_year: int = None
) -> list:
    """
    Apply evidence hierarchy boosts to reranked documents.

    Tier System (v2 defaults, configurable via env):
    - Tier 1 (2.00x): Guidelines, systematic reviews, meta-analyses
    - Tier 2 (1.25x): RCTs, clinical trials, review articles
    - Tier 3 (1.00x): Standard research
    - Tier 4 (0.40x): Case reports, letters, editorials
    
    Smart Recency: Only applies to non-case-reports to preserve seminal older papers.
    Journal Boost:
    - Specialty journals: +20%
    - General/high-impact journals: +15%
    """
    if current_year is None:
        current_year = datetime.now().year
    for doc in documents:
        # Use combined_score (includes entity matching) as base
        base_score = doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.5)))
        
        # 1. Evidence tier multiplier (now with publication_types support)
        tier_mult = get_evidence_multiplier(
            doc.get("article_type", ""),
            doc.get("title", ""),
            doc.get("publication_type", []),  # From payload
            journal=doc.get("journal") or doc.get("venue"),
            evidence_term=doc.get("evidence_term"),
            evidence_source=doc.get("evidence_source"),
        )

        society_signal = detect_guideline_society_signal(
            title=doc.get("title"),
            journal=doc.get("journal") or doc.get("venue"),
            evidence_term=doc.get("evidence_term"),
            evidence_source=doc.get("evidence_source"),
            publication_types=doc.get("publication_type", []),
        )
        
        # 2. Smart recency boost: ONLY apply to non-case-reports
        # This prevents fresh case reports from outranking seminal older guidelines
        recency_mult = 1.0
        is_case_report = (tier_mult == TIER_4_PENALTY)
        if not is_case_report:
            year = doc.get("year")
            if year:
                try:
                    if RERANKER_V2_ENABLED:
                        # Graduated recency boost — no penalty for old seminal papers
                        age = current_year - int(year)
                        if age <= 1:
                            recency_mult = 1.50   # very recent: +50%
                        elif age <= 3:
                            recency_mult = 1.35   # recent: +35%
                        elif age <= 5:
                            recency_mult = 1.18   # somewhat recent: +18%
                        elif age <= 7:
                            recency_mult = 1.06   # slightly recent: +6%
                        # age > 7: neutral (1.0) — no penalty for older papers
                    else:
                        # Legacy binary boost
                        if int(year) >= current_year - 2:
                            recency_mult = 1.10   # +10% for recent high-quality articles
                except (ValueError, TypeError):
                    pass
        
        # 3. Journal-tier boost (specialty > general/high-impact)
        journal_mult = 1.0
        journal_tier = get_journal_tier(
            journal_name=doc.get("journal") or doc.get("venue"),
            nlm_id=doc.get("nlm_unique_id"),
        )
        if journal_tier == "specialty":
            journal_mult = 1.20
        elif journal_tier == "general":
            journal_mult = 1.15
        
        # Calculate final boosted score
        doc["boosted_score"] = base_score * tier_mult * recency_mult * journal_mult
        doc["evidence_tier"] = tier_mult
        doc["boost_multiplier"] = tier_mult * recency_mult * journal_mult
        doc["journal_tier"] = journal_tier
        doc["journal_boost"] = journal_mult
        doc["is_case_report"] = is_case_report  # Track for debugging
        doc["guideline_society_match"] = society_signal["is_match"]
        doc["matched_guideline_societies"] = society_signal["matched_societies"]
    
    return documents



class AbstractReranker:
    """Abstract base class for rerankers."""
    
    def get_scores(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[float]:
        raise NotImplementedError


# =============================================================================
# Self-Hosted Cross-Encoder Reranker (default)
# =============================================================================




# =============================================================================
# DeepInfra Reranker (OpenAI-compatible / Custom API)
# =============================================================================

class DeepInfraReranker(AbstractReranker):
    """
    Reranker using DeepInfra's inference endpoint for Qwen/Qwen3-Reranker models.
    
    API Documentation: https://deepinfra.com/Qwen/Qwen3-Reranker-0.6B/api
    """
    
    def __init__(self, model: str = None):
        self.model = model or "Qwen/Qwen3-Reranker-0.6B"
        if not DEEPINFRA_API_KEY:
            raise ValueError("DEEPINFRA_API_KEY not set")
        self.api_key = DEEPINFRA_API_KEY
        # DeepInfra inference endpoint for Qwen3-Reranker models
        self.api_url = f"https://api.deepinfra.com/v1/inference/{self.model}"
        self.retry_count = DEEPINFRA_RETRY_COUNT
        self.retry_delay = DEEPINFRA_RETRY_DELAY
        logger.info(f"✅ DeepInfra Reranker initialized with {self.model}")

    def get_scores(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[float]:
        """
        Get relevance scores for documents using DeepInfra inference API.
        
        DeepInfra Qwen3-Reranker API expects:
        - Request: {"queries": [...], "documents": [...]}
        - Response: {"scores": [...], "input_tokens": N}
        """
        if not documents:
            return []

        n_docs = len(documents)
        
        try:
            # DeepInfra Qwen3-Reranker API format
            payload = {
                "queries": [query],
                "documents": documents[:top_n] if top_n else documents
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            def _post_rerank() -> requests.Response:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=DEEPINFRA_RERANK_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return response

            response = retry_with_exponential_backoff(
                _post_rerank,
                max_attempts=self.retry_count + 1,
                base_delay=float(self.retry_delay),
                operation_name="DeepInfra Qwen3-Reranker API call",
                logger=logger,
            )
            data = response.json()
            
            # DeepInfra returns scores directly in "scores" array
            scores = data.get("scores", [])

            if not scores:
                logger.critical("RERANKER DEGRADED: Empty scores from API — all passages will receive uniform 0.5 scores")
                return [0.5] * n_docs

            # Health check: detect all-identical scores (indicates API malfunction)
            unique_scores = set(round(s, 6) for s in scores)
            if len(unique_scores) <= 1:
                logger.critical(
                    "RERANKER HEALTH CHECK FAILED: All %d scores identical (%.4f). "
                    "API may be returning dummy values — reranking is effectively disabled.",
                    len(scores), scores[0]
                )

            # Log token usage and score distribution for monitoring
            input_tokens = data.get("input_tokens", 0)
            score_min, score_max = min(scores), max(scores)
            score_mean = sum(scores) / len(scores)
            logger.info(
                "   Reranker API: %d input tokens, %d docs | scores: min=%.3f, max=%.3f, mean=%.3f",
                input_tokens, len(documents), score_min, score_max, score_mean
            )

            return scores

        except Exception as e:
            logger.critical(
                "RERANKER DEGRADED: DeepInfra API call failed: %s — "
                "all %d passages will receive uniform 0.5 scores, reranking is disabled for this query",
                e, n_docs
            )
            return [0.5] * n_docs

# =============================================================================
# Paper Finder with Reranker (matches ScholarQA's PaperFinderWithReranker)
# =============================================================================

class PaperFinderWithReranker:
    """
    ScholarQA-style paper finder with reranking.
    
    Flow:
    1. Rerank passages using DeepInfra Qwen3-Reranker
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
            reranker: Reranker engine (defaults based on RERANKER_PROVIDER config)
            n_rerank: Max passages to keep after rerank (-1 = all)
            context_threshold: Min score threshold
        """
        if reranker is not None:
            self.reranker_engine = reranker
        else:
            self.reranker_engine = DeepInfraReranker(model=RERANKER_MODEL)
        
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
        pre_filter_threshold: float = 0.0,  # Disabled by default: retrieval score scales vary by retrieval mode
        relevance_threshold: float = 0.1,   # Minimum relevance score (filters truly irrelevant)
        medical_conditions: Optional[List[str]] = None,  # LLM-extracted conditions for entity scoring
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages using DeepInfra Qwen3-Reranker + evidence hierarchy boosts.
        
        Threshold Strategy:
        1. pre_filter_threshold (0.0 default): Optional filter by retrieval score BEFORE reranking
           - Can reduce API costs when retrieval score scale is stable
           - Disabled by default to avoid score-scale coupling across retrieval modes
        
        2. relevance_threshold (0.1): Filters by reranker score AFTER reranking
           - Scores are 0-1 normalized; <0.1 indicates very low relevance
           - This removes documents considered irrelevant regardless of retrieval score
        
        3. context_threshold (from __init__): Filters AFTER aggregation by boosted_score
           - Applied in aggregate_snippets_to_papers() using final combined scores
           - Controls the quality floor for papers included in context
        
        Pipeline:
        1. Pre-filter by retrieval score (reduces API cost)
        2. Run Qwen3 reranking + entity matching concurrently
        3. Post-filter by relevance score (removes irrelevant docs)
        4. Combine reranker and entity scores
        5. Apply evidence hierarchy boosts (article type, recency, journal)
        6. Sort by boosted_score
        """
        if not retrieved_ctxs:
            return []
        
        original_count = len(retrieved_ctxs)
        logger.info(f"Reranking {original_count} passages...")
        
        # Stage 1: Pre-filter by retrieval score (before DeepInfra API call)
        # This reduces API cost while preserving candidates for reranker's judgment
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

        
        # Format documents as plain text for optimal DeepInfra reranking
        # using the same chunk elements produced by ingestion.
        passages = []
        for doc in retrieved_ctxs:
            passages.append(self._build_rerank_text(doc))
        
        # Call reranker API directly (I/O-bound — no threading needed here).
        # Entity scoring was removed: it was CPU-bound MeSH synonym expansion that
        # competed with this HTTP call and added latency. Its 15% weight contribution
        # is already handled by Qdrant's BM25 hybrid retrieval upstream.
        top_n_for_rerank = len(passages)
        if self.n_rerank > 0:
            top_n_for_rerank = min(self.n_rerank, len(passages))
        rerank_scores = self.reranker_engine.get_scores(
            query,
            passages,
            top_n=top_n_for_rerank,
        )

        # Attach rerank scores to contexts for filtering
        for i, score in enumerate(rerank_scores):
            retrieved_ctxs[i]["rerank_score"] = score
            retrieved_ctxs[i]["entity_score"] = 0.0  # removed; Qdrant BM25 covers this
        
        # Stage 3: Post-filter by reranker relevance score
        # Best practice: scores <0.1 indicate very low relevance
        pre_rerank_count = len(retrieved_ctxs)
        if relevance_threshold > 0:
            retrieved_ctxs = [
                ctx for ctx in retrieved_ctxs
                if ctx.get("rerank_score", 0) >= relevance_threshold
            ]
            if len(retrieved_ctxs) < pre_rerank_count:
                logger.info(f"   Post-filter (reranker score ≥{relevance_threshold}): {pre_rerank_count} → {len(retrieved_ctxs)} passages")
        
        if not retrieved_ctxs:
            logger.warning("All passages filtered out by reranker relevance threshold, returning empty")
            return []
        
        # Use reranker score directly as combined_score (entity scoring removed)
        combined_scores = []
        for doc in retrieved_ctxs:
            rerank_score = doc.get("rerank_score", 0)
            doc["combined_score"] = rerank_score
            combined_scores.append(rerank_score)

        logger.info(f"Reranker top scores: {sorted(rerank_scores, reverse=True)[:5]}")
        logger.info(f"Combined scores: {[round(s, 3) for s in combined_scores[:5]]}")
        
        # Apply evidence hierarchy boosts
        retrieved_ctxs = apply_evidence_boosts(retrieved_ctxs)
        
        # Log boost effects
        boosted = [
            (
                d.get("article_type", "?"),
                d.get("journal_tier", "none"),
                d.get("journal_boost", 1.0),
                d.get("boost_multiplier", 1.0),
            )
            for d in retrieved_ctxs[:5]
        ]
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

    def _build_rerank_text(self, doc: Dict[str, Any]) -> str:
        """Build rerank input from ingestion-aligned text elements."""
        title = str(doc.get("title", "")).strip()
        section_title = str(doc.get("section_title", "")).strip()
        content = self._get_document_content(doc)

        if not content:
            content = str(doc.get("abstract", "") or doc.get("full_text", "")).strip()

        if title and content and content.lower().startswith(title.lower()):
            text = content
        elif title and content:
            text = f"{title}\n\n{content}"
        elif section_title and content and not content.lower().startswith(section_title.lower()):
            text = f"{section_title}\n\n{content}"
        else:
            text = content or title

        return text[:4000]

    def _get_document_content(self, doc: Dict[str, Any]) -> str:
        """
        Return the best available chunk/document content using ingestion field order.
        """
        for key in ("page_content", "text", "full_section_text", "abstract", "full_text"):
            value = doc.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
        return ""
    
    def _build_entity_variants(self, entity: str) -> set:
        """
        Return a flat set of all normalized forms and MeSH synonyms for an entity.

        Pre-computing variants once per query entity keeps matching O(variants)
        per document instead of O(MeSH_vocab) per document.
        """
        def _norm(s: str) -> str:
            return re.sub(r'[\s\-]+', ' ', s).lower().strip()

        variants: set = {entity.lower(), _norm(entity)}

        if self.entity_expander:
            # MeSH synonyms for the entity treated as a full term
            for syn in self.entity_expander.get_synonyms(entity):
                variants.add(syn.lower())
                variants.add(_norm(syn))
            # Treat entity as acronym → expand to full terms
            for exp in self.entity_expander.expand_acronym(entity):
                variants.add(exp.lower())
                variants.add(_norm(exp))
            # Also look up synonyms of each expansion (catches abbreviation ↔ full-term)
            for syn in self.entity_expander.get_synonyms(entity):
                if re.match(r'^[A-Z0-9\-]{2,8}$', syn):
                    variants.add(syn.lower())

        # Remove empty strings that may have crept in
        variants.discard('')
        return variants

    def _calculate_entity_scores(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate entity matching scores for documents.

        Extracts medical entities from query and scores documents based on entity overlap.
        Uses fuzzy matching: normalizes hyphens/whitespace and expands MeSH synonyms so
        "IgG4-related disease" matches "IgG4 related disease", "RA" matches
        "Rheumatoid Arthritis", and "T2DM" matches "Type 2 Diabetes Mellitus".

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

        # Pre-compute variant sets once per entity (avoids scanning MeSH per document)
        entity_variant_sets = [self._build_entity_variants(e) for e in query_entities]

        scores = []
        for doc in documents:
            title = str(doc.get("title", "")).lower()
            abstract = str(doc.get("abstract", "")).lower()
            text = self._get_document_content(doc).lower()
            full_text = str(doc.get("full_text", "")).lower()

            # Combine text for searching; normalize hyphens/extra spaces once per doc
            raw_doc_text = f"{title} {abstract} {text} {full_text}"
            doc_text = re.sub(r'[\s\-]+', ' ', raw_doc_text).lower()
            # Keep original too so literal forms also match
            doc_text_orig = raw_doc_text.lower()

            # Count entity matches — any variant form found counts as a match
            matches = 0
            for variants in entity_variant_sets:
                if any(v in doc_text or v in doc_text_orig for v in variants):
                    matches += 1

            # Score: proportion of query entities found in document
            score = matches / len(query_entities) if query_entities else 0.0
            scores.append(score)

        return scores
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract medical entities from query for entity-based scoring.

        Extraction steps:
        1. Hyphenated medical terms (e.g., "IgG4-related disease")
        2. Acronyms (2+ uppercase letters) + MeSH expansion (keeps short ones like "RA", "SLE")
        3. Capitalized multi-word terms (e.g., "Antiphospholipid Syndrome")
        4. Disease-pattern suffixes (arthritis, fibrosis, anemia, neuropathy, …)
        5. Remaining words — only if they look medical (capitalized, contain digits,
           or end with a known medical suffix), filtered through ENTITY_STOPWORDS
        """
        entities = []

        if not self.entity_expander:
            # Fallback: extract key terms without MeSH
            key_terms = re.findall(r'\b[A-Z][a-z]+(?:[-][a-z]+)*\b', query)
            key_terms += re.findall(r'\b[A-Z]{2,}\d*\b', query)  # Acronyms
            logger.info(f"Entity extraction (no MeSH): {key_terms}")
            return list(set(key_terms))

        # 1. Hyphenated medical terms (e.g., "IgG4-related", "non-small-cell")
        hyphenated = re.findall(r'\b\w+(?:-\w+)+\b', query)
        entities.extend(hyphenated)

        # 2. Acronyms (2+ uppercase letters, optionally followed by digits)
        #    Keep short ones like "RA", "SLE" — the dedup filter used to drop len==2,
        #    fixed below (>= 2 instead of > 2).
        acronyms = re.findall(r'\b[A-Z]{2,}\d*\b', query)
        for acr in acronyms:
            entities.append(acr)
            # Expand acronym via MeSH (e.g., "RA" → "Rheumatoid Arthritis")
            expansions = self.entity_expander.expand_acronym(acr)
            if expansions:
                entities.append(expansions[0])

        # 3. Capitalized multi-word terms (e.g., "Antiphospholipid Syndrome")
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend([p for p in capitalized_phrases if len(p) > 3])

        # 4. Disease-pattern suffixes
        disease_patterns = [
            r'\b\w+[-]?related\s+disease\b',    # "IgG4-related disease"
            r'\b\w+\s+disease\b',               # "X disease"
            r'\b\w+\s+syndrome\b',              # "X syndrome"
            r'\b\w+\s+disorder\b',              # "X disorder"
            r'\b\w+itis\b',                     # arthritis, colitis, …
            r'\b\w+osis\b',                     # fibrosis, cirrhosis, …
            r'\b\w+emia\b',                     # anemia, leukemia, …
            r'\b\w+pathy\b',                    # neuropathy, nephropathy, …
        ]
        for pattern in disease_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)

        # 5. Remaining words — only keep if they look medical (selective catch-all)
        words = query.split()
        for word in words:
            clean = re.sub(r'[^\w-]', '', word)
            if (len(clean) > 5
                    and clean.lower() not in ENTITY_STOPWORDS
                    and (clean[0].isupper()
                         or any(c.isdigit() for c in clean)
                         or clean.lower().endswith(MEDICAL_SUFFIXES))):
                entities.append(clean)

        # Deduplicate (case-insensitive) while preserving order.
        # Fix: use >= 2 (not > 2) so 2-char acronyms like "RA" are kept.
        seen: set = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen and len(entity) >= 2:
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

        def normalized_evidence_level(value: Any) -> Optional[int]:
            """Normalize evidence level to int in range 1..4, else None."""
            try:
                if value is None:
                    return None
                level = int(value)
                return level if 1 <= level <= 4 else None
            except (TypeError, ValueError):
                return None
        
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
                        "text": snippet.get("text") or snippet.get("page_content", ""),
                        "section_title": snippet.get("section_title", "abstract"),
                        "char_start_offset": snippet.get("char_offset", 0),
                    })

                    existing_level = normalized_evidence_level(paper_snippets[duplicate_of].get("evidence_level"))
                    new_level = normalized_evidence_level(snippet.get("evidence_level"))
                    if new_level is not None and (existing_level is None or new_level < existing_level):
                        paper_snippets[duplicate_of]["evidence_grade"] = snippet.get("evidence_grade")
                        paper_snippets[duplicate_of]["evidence_level"] = new_level
                        paper_snippets[duplicate_of]["evidence_term"] = snippet.get("evidence_term")
                        paper_snippets[duplicate_of]["evidence_source"] = snippet.get("evidence_source")
                    else:
                        if not paper_snippets[duplicate_of].get("evidence_term") and snippet.get("evidence_term"):
                            paper_snippets[duplicate_of]["evidence_term"] = snippet.get("evidence_term")
                        if not paper_snippets[duplicate_of].get("evidence_source") and snippet.get("evidence_source"):
                            paper_snippets[duplicate_of]["evidence_source"] = snippet.get("evidence_source")
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
                    "evidence_grade": snippet.get("evidence_grade"),
                    "evidence_level": normalized_evidence_level(snippet.get("evidence_level")),
                    "evidence_term": snippet.get("evidence_term"),
                    "evidence_source": snippet.get("evidence_source"),
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
                "text": snippet.get("text") or snippet.get("page_content", ""),
                "section_title": snippet.get("section_title", "abstract"),
                "char_start_offset": snippet.get("char_offset", 0),
            })
            
            # Update relevance using max boosted_score (evidence hierarchy aware)
            current_score = paper_snippets[corpus_id]["relevance_judgement"]
            new_score = snippet.get("boosted_score", snippet.get("rerank_score", snippet.get("score", 0)))
            paper_snippets[corpus_id]["relevance_judgement"] = max(current_score, new_score)
            
            # Update abstract if from abstract section
            if snippet.get("section_title") == "abstract" and not paper_snippets[corpus_id]["abstract"]:
                paper_snippets[corpus_id]["abstract"] = snippet.get("text") or snippet.get("page_content", "")

        
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
                f"[{row['pmcid']} | Unknown | "
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
