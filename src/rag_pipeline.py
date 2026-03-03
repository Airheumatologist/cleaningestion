"""
Elixir Medical RAG Pipeline.

Optimized pipeline for fast, comprehensive medical responses:
1. Query preprocessing with decomposition
2. Qdrant hybrid retrieval (dense + BM25 sparse)
3. DeepInfra Qwen3-Reranker with paper aggregation and evidence hierarchy
4. Direct LLM synthesis with ELIXIR system prompt (Groq/DeepInfra GPT-OSS-20B)

Designed for speed and quality clinical decision support.
"""

import logging
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Generator, Optional
from dataclasses import asdict
from groq import Groq
from openai import OpenAI

from .config import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_BASE_URL,
    GROQ_API_KEY,
    LLM_PROVIDER,
    LLM_CHAT_TIMEOUT_SECONDS,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_REASONING_EFFORT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    BULK_RETRIEVAL_LIMIT,
    MAX_ABSTRACTS,
    MAX_DAILYMED_PER_DRUG,
    RETRIEVAL_CHUNK_LIMIT,
    MAX_CHUNKS_PER_ARTICLE_PRE_RERANK,
    RERANK_INPUT_CHUNK_LIMIT,
    RERANK_TOP_CHUNKS,
    FINAL_TOP_ARTICLES,
    FINAL_RECENCY_POLICY_MODE,
    FINAL_RECENCY_WINDOW_YEARS,
    FINAL_RECENCY_BACKFILL_MAX_EVIDENCE_LEVEL,
    FINAL_RECENCY_EXCLUDE_UNKNOWN_NON_DAILYMED,
    PMC_FULLTEXT_RECENT_ONLY,
    ENTITY_FILTER_ENABLED,
    PRE_RERANK_RECENT_WINDOW_YEARS,
    PRE_RERANK_RECENT_QUOTA_RATIO,
)
from .query_preprocessor import QueryPreprocessor, LLMProcessedQuery
from .retriever_qdrant import QdrantRetriever
from .reranker import PaperFinderWithReranker
from .prompts import ELIXIR_SYSTEM_PROMPT
from .query_cache import QueryCache
try:
    from scripts.ingestion_utils import get_evidence_hierarchy_levels
except Exception:
    def get_evidence_hierarchy_levels() -> Dict[str, Any]:
        return {
            "levels": [
                {"grade": "A", "level": 1, "label": "Highest evidence", "terms": []},
                {"grade": "B", "level": 2, "label": "High evidence", "terms": []},
                {"grade": "C", "level": 3, "label": "Moderate evidence", "terms": []},
                {"grade": "D", "level": 4, "label": "Lower evidence", "terms": []},
            ]
        }

logger = logging.getLogger(__name__)


class MedicalRAGPipeline:
    """
    Elixir Medical RAG Pipeline.
    
    Optimized flow:
    1. Preprocess query (decompose, extract filters)
    2. Retrieve passages from Qdrant
    3. Rerank passages with DeepInfra Qwen3-Reranker
    4. Aggregate to paper level
    5. Direct LLM synthesis with ELIXIR prompt (Groq/DeepInfra)
    """
    
    def __init__(
        self,
        model: str = LLM_MODEL,
        n_retrieval: int = RETRIEVAL_CHUNK_LIMIT,
        n_rerank: int = RERANK_TOP_CHUNKS,
        context_threshold: float = 0.3  # Post-aggregation threshold on boosted_score
        # Note: This is the FINAL threshold after Qwen3 reranking + entity matching + evidence boosts
        # With evidence tier multipliers (0.2x-3.0x), 0.3 ensures good recall while filtering noise
        # Reranker relevance filtering (0.1 minimum) is done earlier in PaperFinderWithReranker.rerank()
    ):

        """Initialize pipeline components."""
        logger.info("🚀 Initializing Elixir Medical RAG Pipeline...")

        if not DEEPINFRA_API_KEY:
            raise ValueError("DEEPINFRA_API_KEY not set")

        self.model = model
        self.max_chunks_per_article_pre_rerank = MAX_CHUNKS_PER_ARTICLE_PRE_RERANK
        self.rerank_input_chunk_limit = RERANK_INPUT_CHUNK_LIMIT
        self.pre_rerank_recent_window_years = PRE_RERANK_RECENT_WINDOW_YEARS
        self.pre_rerank_recent_quota_ratio = PRE_RERANK_RECENT_QUOTA_RATIO
        self.final_top_articles = FINAL_TOP_ARTICLES
        self.final_recency_policy_mode = FINAL_RECENCY_POLICY_MODE
        self.final_recency_window_years = max(1, FINAL_RECENCY_WINDOW_YEARS)
        self.final_recency_backfill_max_evidence_level = FINAL_RECENCY_BACKFILL_MAX_EVIDENCE_LEVEL
        self.final_recency_exclude_unknown_non_dailymed = FINAL_RECENCY_EXCLUDE_UNKNOWN_NON_DAILYMED
        self.pmc_fulltext_recent_only = PMC_FULLTEXT_RECENT_ONLY
        self._last_recency_stats = {
            "recent_kept_non_dailymed": 0,
            "dailymed_kept": 0,
            "older_backfilled": 0,
            "unknown_non_dailymed_excluded": 0,
            "recent_cutoff_year": datetime.now().year - self.final_recency_window_years + 1,
        }
        self._last_context_stats = {
            "pmc_recent_fulltext_used": 0,
        }
        self.llm_provider = LLM_PROVIDER
        if self.llm_provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set")
            self.llm_client = Groq(
                api_key=GROQ_API_KEY,
                timeout=LLM_CHAT_TIMEOUT_SECONDS,
            )
        elif self.llm_provider == "deepinfra":
            self.llm_client = OpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_BASE_URL,
                timeout=LLM_CHAT_TIMEOUT_SECONDS,
            )
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.llm_provider}")
        
        # Components
        self.preprocessor = QueryPreprocessor(model=model)
        self.retriever = QdrantRetriever(n_retrieval=n_retrieval)

        # Reuse the entity expander already initialized inside QueryPreprocessor
        self.entity_expander = self.preprocessor._entity_expander

        # Initialize reranker (DeepInfra Qwen only)
        self.paper_finder = PaperFinderWithReranker(
            n_rerank=n_rerank,
            context_threshold=context_threshold,
            entity_expander=self.entity_expander
        )
        logger.info("✅ Reranker initialized (DeepInfra Qwen3-Reranker-0.6B)")
        self.evidence_hierarchy = get_evidence_hierarchy_levels()
        
        # Initialize Query Cache
        self.cache = QueryCache()
        prompt_hash = hashlib.sha256(ELIXIR_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]
        reranker_model = getattr(getattr(self.paper_finder, "reranker_engine", None), "model", "unknown")
        self._cache_key_context = {
            "pipeline": "elixir",
            "pipeline_cache_version": 3,
            "llm_model": self.model,
            "reranker_model": reranker_model,
            "embedding_model": getattr(self.retriever, "embedding_model", "unknown"),
            "collection_name": getattr(self.retriever, "collection_name", "unknown"),
            "entity_filter_enabled": ENTITY_FILTER_ENABLED,
            "prompt_hash": prompt_hash,
            "n_retrieval": n_retrieval,
            "n_rerank": n_rerank,
            "final_top_articles": self.final_top_articles,
        }
        
        logger.info("✅ Pipeline initialized (Elixir direct synthesis, provider=%s)", self.llm_provider)

    def _cache_get(self, query: str) -> Dict[str, Any] | None:
        """Read query cache using pipeline context so stale entries are isolated."""
        if not self.cache.enabled:
            return None
        return self.cache.get(query, **self._cache_key_context)

    def _cache_set(self, query: str, response: Dict[str, Any]):
        """Write query cache using pipeline context."""
        if not self.cache.enabled:
            return
        self.cache.set(query, response, **self._cache_key_context)

    def _build_retrieval_stats(
        self,
        passages_retrieved: int,
        papers_after_aggregation: int,
        abstracts_used: int,
    ) -> Dict[str, Any]:
        recency_stats = self._last_recency_stats or {}
        context_stats = self._last_context_stats or {}
        return {
            "passages_retrieved": passages_retrieved,
            "papers_after_aggregation": papers_after_aggregation,
            "abstracts_used": abstracts_used,
            "recent_articles_kept": recency_stats.get("recent_kept_non_dailymed", 0),
            "older_high_evidence_backfilled": recency_stats.get("older_backfilled", 0),
            "unknown_year_non_dailymed_excluded": recency_stats.get("unknown_non_dailymed_excluded", 0),
            "pmc_recent_fulltext_used": context_stats.get("pmc_recent_fulltext_used", 0),
        }

    def _reset_run_stats(self) -> None:
        self._last_recency_stats = {
            "recent_kept_non_dailymed": 0,
            "dailymed_kept": 0,
            "older_backfilled": 0,
            "unknown_non_dailymed_excluded": 0,
            "recent_cutoff_year": self._recent_cutoff_year(),
        }
        self._last_context_stats = {
            "pmc_recent_fulltext_used": 0,
            "recent_cutoff_year": self._recent_cutoff_year(),
        }
    
    # =========================================================================
    # Step 1: Query Preprocessing
    # =========================================================================
    
    def preprocess_query(self, query: str) -> LLMProcessedQuery:
        """Decompose query and extract filters."""
        logger.info("📝 Step 1: Query Preprocessing")
        result = self.preprocessor.decompose_query(query)
        logger.info(f"   Rewritten: {result.rewritten_query}")
        logger.info(f"   Filters: {result.search_filters}")
        return result
    
    # =========================================================================
    # Step 2: Retrieval
    # =========================================================================
    
    def retrieve_passages(
        self,
        processed_query: LLMProcessedQuery
    ) -> List[Dict[str, Any]]:
        """
        Retrieve passages from Qdrant using multi-query expansion.
        
        Optimized with BATCH query API:
        - Batch encodes ALL query variations with BM25 in ONE call
        - Sends ALL queries (dense + sparse) to Qdrant in ONE HTTP request
        - DailyMed search runs in parallel thread
        """
        from concurrent.futures import ThreadPoolExecutor
        from qdrant_client.models import SparseVector
        
        logger.info("🔍 Step 2: Passage Retrieval (Batch Hybrid Search)")
        
        dailymed_results = []
        
        # Get queries and drug names
        queries_to_run = processed_query.expanded_queries or [processed_query.rewritten_query]
        # If expanded queries are already present, trust them as the complete set.
        # Only append keyword query for fallback paths that do not generate expansions.
        if not processed_query.expanded_queries and processed_query.keyword_query and processed_query.keyword_query not in queries_to_run:
            queries_to_run.append(processed_query.keyword_query)
            
        drug_names = []
        if processed_query.decomposed and processed_query.decomposed.drug_names:
            drug_names = processed_query.decomposed.drug_names
        else:
            # Fallback to manual extraction if decomposition failed
            drug_names = self._extract_drug_names(processed_query.original_query, processed_query.rewritten_query)

        logger.info(f"   PMC+PubMed Queries: {len(queries_to_run)} | DailyMed Drugs: {len(drug_names)}")
        
        # ========================================================================
        # OPTIMIZATION 1: Batch encode ALL query variations with configured sparse mode
        # ========================================================================
        sparse_vectors = []
        try:
            sparse_vectors = self.retriever.build_sparse_query_vectors(queries_to_run)
            logger.info(f"   ⚡ Built {len(sparse_vectors)} sparse query vectors")
        except Exception as e:
            logger.warning(f"   Sparse query vector build failed: {e}")
            sparse_vectors = [SparseVector(indices=[], values=[]) for _ in queries_to_run]
        
        # ========================================================================
        # OPTIMIZATION 2: Run DailyMed search in parallel with batch PMC query
        # ========================================================================
        def run_dailymed_search(drugs: List[str]) -> List[Dict[str, Any]]:
            try:
                if not drugs: return []
                return self.retriever.search_dailymed_by_drug(drugs)
            except Exception as e:
                logger.warning(f"DailyMed search failed: {e}")
                return []
        
        # Start DailyMed search in background thread
        with ThreadPoolExecutor(max_workers=2) as executor:
            dm_future = executor.submit(run_dailymed_search, drug_names)
            
            # ========================================================================
            # OPTIMIZATION 3: Single batch HTTP call for ALL PMC queries (6 → 1)
            # ========================================================================
            all_passages = self.retriever.batch_hybrid_search(
                queries=queries_to_run,
                sparse_vectors=sparse_vectors,
                **processed_query.search_filters
            )
            
            # Collect DailyMed results
            try:
                dailymed_results = dm_future.result()
                if dailymed_results:
                    logger.info(f"   ✅ DailyMed search found {len(dailymed_results)} results")
            except Exception as e:
                logger.warning(f"DailyMed task failed: {e}")
        
        logger.info(f"   Retrieved {len(all_passages)} unique passages (PMC+PubMed)")
        return all_passages, dailymed_results

    
    # Words that should NOT be extracted as drug names
    NON_DRUG_WORDS = {
        "guideline", "guidelines", "treatment", "treatments", "management",
        "screening", "diagnosis", "therapy", "therapies", "recommendation",
        "recommendations", "update", "review", "reviews", "criteria",
        "college", "rheumatology", "american", "european", "eular", "acr",
        "lupus", "nephritis", "arthritis", "disease", "syndrome", "disorder"
    }
    
    def _extract_drug_names(self, original_query: str, rewritten_query: str) -> List[str]:
        """
        Extract drug names from query using simple pattern matching.
        
        Looks for capitalized words that look like drug names (brand names)
        and known drug name patterns.
        """
        import re
        
        # Combine queries for better coverage
        text = f"{original_query} {rewritten_query}".lower()
        
        drug_names = set()
        
        # Common brand name drugs that are often asked about
        common_drugs = [
            "xeljanz", "tofacitinib", "humira", "adalimumab", "enbrel", "etanercept",
            "remicade", "infliximab", "rituxan", "rituximab", "orencia", "abatacept",
            "actemra", "tocilizumab", "simponi", "golimumab", "cimzia", "certolizumab",
            "rinvoq", "upadacitinib", "olumiant", "baricitinib", "kevzara", "sarilumab",
            "methotrexate", "mtx", "plaquenil", "hydroxychloroquine", "sulfasalazine",
            "leflunomide", "arava", "azathioprine", "imuran", "cyclosporine",
            "prednisone", "prednisolone", "methylprednisolone", "dexamethasone",
            "celebrex", "celecoxib", "meloxicam", "naproxen", "ibuprofen",
            "taltz", "ixekizumab", "cosentyx", "secukinumab", "stelara", "ustekinumab",
            "dupixent", "dupilumab", "otezla", "apremilast", "tremfya", "guselkumab",
        ]
        
        for drug in common_drugs:
            if re.search(r'\b' + re.escape(drug) + r'\b', text):
                drug_names.add(drug)
        
        # Extract capitalized words that might be brand names
        # Filter out NON_DRUG_WORDS to prevent false positives
        capitalized_words = re.findall(r'\b[A-Z][a-z]{3,}\b', original_query)
        for word in capitalized_words:
            word_lower = word.lower()
            if (len(word) >= 4 and 
                word_lower not in self.NON_DRUG_WORDS and
                word_lower not in ["what", "when", "where", "which", "this", "that", "with", "from", "have"]):
                drug_names.add(word_lower)
        
        return list(drug_names)[:5]  # Limit to 5 drugs


    
    # =========================================================================
    # Step 3: Reranking & Aggregation
    # =========================================================================

    @staticmethod
    def _to_int_year(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            match = re.search(r"\b(19|20)\d{2}\b", str(value))
            if not match:
                return None
            try:
                return int(match.group(0))
            except (TypeError, ValueError):
                return None

    @staticmethod
    def _to_int_evidence_level(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            level = int(value)
            if 1 <= level <= 4:
                return level
            return None
        except (TypeError, ValueError):
            return None

    def _recent_cutoff_year(self, current_year: Optional[int] = None) -> int:
        year_now = current_year if current_year is not None else datetime.now().year
        return year_now - self.final_recency_window_years + 1

    @staticmethod
    def _is_dailymed_row(row: Any) -> bool:
        pmcid = str(row.get("pmcid", "") or row.get("corpus_id", "")).lower()
        article_type = str(row.get("article_type", "")).lower()
        source = str(row.get("source", "")).lower()
        venue = str(row.get("venue", "") or row.get("journal", "")).lower()
        return (
            pmcid.startswith("dailymed_")
            or article_type == "drug_label"
            or source == "dailymed"
            or "dailymed" in venue
            or "fda drug label" in venue
        )

    @staticmethod
    def _attach_sort_score(papers_df):
        if papers_df.empty:
            return papers_df

        if "relevance_judgement" in papers_df.columns and "relevance_score" in papers_df.columns:
            papers_df["_sort_score"] = papers_df["relevance_judgement"].fillna(papers_df["relevance_score"]).fillna(0.0)
        elif "relevance_judgement" in papers_df.columns:
            papers_df["_sort_score"] = papers_df["relevance_judgement"].fillna(0.0)
        elif "relevance_score" in papers_df.columns:
            papers_df["_sort_score"] = papers_df["relevance_score"].fillna(0.0)
        else:
            papers_df["_sort_score"] = 0.0
        return papers_df

    def _apply_final_recency_policy(self, papers_df):
        import pandas as pd

        cutoff_year = self._recent_cutoff_year()
        stats = {
            "recent_kept_non_dailymed": 0,
            "dailymed_kept": 0,
            "older_backfilled": 0,
            "unknown_non_dailymed_excluded": 0,
            "recent_cutoff_year": cutoff_year,
        }

        if papers_df.empty:
            return papers_df, stats

        policy_mode = (self.final_recency_policy_mode or "hybrid").lower()
        if policy_mode not in {"hybrid", "strict"}:
            logger.info("   Final recency policy disabled (mode=%s)", policy_mode)
            return papers_df, stats

        ranked_df = self._attach_sort_score(papers_df.copy()).sort_values(by="_sort_score", ascending=False).reset_index(drop=True)

        selected_rows: List[Dict[str, Any]] = []
        older_candidates: List[Dict[str, Any]] = []

        for _, row in ranked_df.iterrows():
            row_dict = row.to_dict()

            if self._is_dailymed_row(row):
                selected_rows.append(row_dict)
                stats["dailymed_kept"] += 1
                continue

            year_value = self._to_int_year(row.get("year"))
            if year_value is None:
                if self.final_recency_exclude_unknown_non_dailymed:
                    stats["unknown_non_dailymed_excluded"] += 1
                    continue
                selected_rows.append(row_dict)
                stats["recent_kept_non_dailymed"] += 1
                continue

            if year_value >= cutoff_year:
                selected_rows.append(row_dict)
                stats["recent_kept_non_dailymed"] += 1
                continue

            older_candidates.append(row_dict)

        if policy_mode == "hybrid" and len(selected_rows) < self.final_top_articles:
            for row_dict in older_candidates:
                evidence_level = self._to_int_evidence_level(row_dict.get("evidence_level"))
                if evidence_level is None or evidence_level > self.final_recency_backfill_max_evidence_level:
                    continue
                selected_rows.append(row_dict)
                stats["older_backfilled"] += 1
                if len(selected_rows) >= self.final_top_articles:
                    break

        if selected_rows:
            selected_df = pd.DataFrame(selected_rows)
        else:
            selected_df = ranked_df.iloc[0:0].copy()

        selected_df = selected_df.drop(columns=["_sort_score"], errors="ignore").reset_index(drop=True)
        logger.info(
            "   Final recency gate (mode=%s, cutoff=%s): recent_kept_non_dailymed=%s | dailymed_kept=%s | older_backfilled=%s | unknown_non_dailymed_excluded=%s",
            policy_mode,
            cutoff_year,
            stats["recent_kept_non_dailymed"],
            stats["dailymed_kept"],
            stats["older_backfilled"],
            stats["unknown_non_dailymed_excluded"],
        )
        return selected_df, stats

    @staticmethod
    def _chunk_identity_for_preselection(passage: Dict[str, Any], idx: int) -> str:
        chunk_id = passage.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        article_id = (
            passage.get("pmcid")
            or passage.get("corpus_id")
            or passage.get("doc_id")
            or passage.get("pmid")
            or "unknown"
        )
        chunk_index = passage.get("chunk_index")
        return f"{article_id}:{chunk_index}:{idx}"

    def _article_identity_for_preselection(self, passage: Dict[str, Any], idx: int) -> str:
        article_id = (
            passage.get("pmcid")
            or passage.get("corpus_id")
            or passage.get("doc_id")
            or passage.get("pmid")
        )
        if article_id:
            return str(article_id)
        return f"unknown:{self._chunk_identity_for_preselection(passage, idx)}"

    def _is_recent_eligible_for_quota(self, passage: Dict[str, Any], current_year: int) -> bool:
        year = self._to_int_year(passage.get("year"))
        if year is None:
            return False
        age = max(0, current_year - year)
        if age > self.pre_rerank_recent_window_years:
            return False
        return not self.retriever._is_low_evidence_type(
            passage.get("article_type"),
            passage.get("publication_type"),
        )

    def _select_chunks_for_rerank(self, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Quota-based preselection:
        1) reserve recent high-signal first chunks per article
        2) fill remaining first-pass slots by score
        3) fill remaining chunk slots globally by score (max chunks/article respected)
        """
        if not passages:
            return []

        limit = self.rerank_input_chunk_limit
        max_per_article = self.max_chunks_per_article_pre_rerank
        recent_quota = max(0, min(limit, int(limit * self.pre_rerank_recent_quota_ratio)))
        current_year = datetime.now().year

        sorted_with_ids = sorted(
            enumerate(passages),
            key=lambda pair: float(pair[1].get("score", 0.0)),
            reverse=True,
        )

        # First chunk candidate per article (highest score only).
        first_chunk_candidates: List[tuple[int, Dict[str, Any], str, str]] = []
        seen_articles: set[str] = set()
        for idx, passage in sorted_with_ids:
            article_id = self._article_identity_for_preselection(passage, idx)
            if article_id in seen_articles:
                continue
            seen_articles.add(article_id)
            chunk_key = self._chunk_identity_for_preselection(passage, idx)
            first_chunk_candidates.append((idx, passage, article_id, chunk_key))

        recent_candidates = [
            item for item in first_chunk_candidates
            if self._is_recent_eligible_for_quota(item[1], current_year)
        ]
        recent_available = len(recent_candidates)

        selected: List[Dict[str, Any]] = []
        selected_chunk_keys: set[str] = set()
        per_article_counts: Dict[str, int] = {}
        selected_meta: List[tuple[str, Dict[str, Any]]] = []

        def _try_add(candidate: tuple[int, Dict[str, Any], str, str]) -> bool:
            _, passage, article_id, chunk_key = candidate
            if chunk_key in selected_chunk_keys:
                return False
            if per_article_counts.get(article_id, 0) >= max_per_article:
                return False
            selected.append(passage)
            selected_chunk_keys.add(chunk_key)
            per_article_counts[article_id] = per_article_counts.get(article_id, 0) + 1
            selected_meta.append((article_id, passage))
            return True

        # First pass A: enforce recent quota on first chunks.
        for candidate in recent_candidates:
            if len(selected) >= recent_quota:
                break
            _try_add(candidate)

        # First pass B: fill first-chunk slots by score regardless of age.
        for candidate in first_chunk_candidates:
            if len(selected) >= limit:
                break
            _try_add(candidate)

        # Second pass: fill remaining slots globally by score.
        for idx, passage in sorted_with_ids:
            if len(selected) >= limit:
                break
            article_id = self._article_identity_for_preselection(passage, idx)
            chunk_key = self._chunk_identity_for_preselection(passage, idx)
            _try_add((idx, passage, article_id, chunk_key))

        recent_selected = sum(
            1 for _, passage in selected_meta
            if self._is_recent_eligible_for_quota(passage, current_year)
        )

        logger.info(
            "   Chunk preselection: %s passages → %s rerank candidates | recent quota target=%s achieved=%s available=%s | max chunks/article=%s",
            len(passages),
            len(selected),
            recent_quota,
            recent_selected,
            recent_available,
            max_per_article,
        )
        return selected
    
    def rerank_and_aggregate(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        dailymed_results: List[Dict[str, Any]] = None,
        medical_conditions: List[str] = None,
        corrected_conditions: List[str] = None
    ):
        """Rerank passages and aggregate to paper level.
        
        Args:
            query: User query
            passages: Main passages (will be reranked)
            dailymed_results: DailyMed drug labels (bypass reranking, merged directly)
            medical_conditions: LLM-extracted conditions for strict filtering
            corrected_conditions: Typo-corrected conditions for matching
        """
        import pandas as pd
        
        logger.info("📊 Step 3: Reranking & Aggregation")
        self._last_recency_stats = {
            "recent_kept_non_dailymed": 0,
            "dailymed_kept": 0,
            "older_backfilled": 0,
            "unknown_non_dailymed_excluded": 0,
            "recent_cutoff_year": self._recent_cutoff_year(),
        }

        if self.paper_finder is None:
            # Skip reranking and create basic aggregation
            logger.info("   Skipping reranking (reranker not available)")
            reranked = self._select_chunks_for_rerank(passages)

            # Create a basic DataFrame-like structure (abstracts only - no full text)
            papers_df = pd.DataFrame([{
                'pmcid': p.get('pmcid', ''),
                'title': p.get('title', ''),
                'authors': p.get('authors', []),
                'venue': p.get('venue', ''),
                'journal': p.get('journal', ''),
                'year': p.get('year'),
                'doi': p.get('doi', ''),
                'article_type': p.get('article_type', ''),
                'evidence_grade': p.get('evidence_grade'),
                'evidence_level': p.get('evidence_level'),
                'evidence_term': p.get('evidence_term'),
                'evidence_source': p.get('evidence_source'),
                'relevance_score': p.get('relevance_judgement', 0),
                'abstract': p.get('abstract', ''),
                'corpus_id': p.get('corpus_id', '')
            } for p in reranked])
        else:
            rerank_candidates = self._select_chunks_for_rerank(passages)
            logger.info(f"   Reranking {len(rerank_candidates)} chunk candidates")
            reranked = self.paper_finder.rerank(
                query,
                rerank_candidates,
                medical_conditions=medical_conditions,
            )

            # Aggregate to paper level and format as DataFrame
            papers_df = self.paper_finder.aggregate_into_dataframe(reranked)

        logger.info(f"   Aggregated to {len(papers_df)} papers")
        
        # Deduplicate DailyMed entries (keep max per drug)
        papers_df = self._deduplicate_dailymed(papers_df)
        
        # Optional post-retrieval filtering: filter out papers that don't contain key medical entities
        # Pass both LLM-extracted conditions AND typo-corrected conditions for matching
        if ENTITY_FILTER_ENABLED:
            papers_df = self._filter_by_entities(query, papers_df, medical_conditions, corrected_conditions)
        
        # MERGE DailyMed results directly (bypass reranking and thresholds)
        if dailymed_results:
            logger.info(f"   Merging {len(dailymed_results)} DailyMed results (bypassed reranking)")
            existing_pmcids = set(papers_df['pmcid'].tolist()) if not papers_df.empty else set()
            
            # Create rows for DailyMed - give them high relevance score to appear at top
            dailymed_rows = []
            for dm in dailymed_results:
                pmcid = dm.get('pmcid', '')
                if pmcid and pmcid not in existing_pmcids:
                    dailymed_rows.append({
                        'pmcid': pmcid,
                        'title': dm.get('title', ''),
                        'authors': dm.get('authors', []),
                        'venue': dm.get('venue', '') or dm.get('journal', 'DailyMed'),
                        'journal': dm.get('journal', 'DailyMed'),
                        'year': dm.get('year'),
                        'doi': dm.get('doi', ''),
                        'article_type': dm.get('article_type', 'drug_label'),
                        'evidence_grade': dm.get('evidence_grade'),
                        'evidence_level': dm.get('evidence_level'),
                        'evidence_term': dm.get('evidence_term'),
                        'evidence_source': dm.get('evidence_source'),
                        'relevance_score': 0.95,  # High score to appear near top
                        'abstract': dm.get('abstract', ''),
                        'corpus_id': pmcid,
                        'source': 'dailymed',
                        'set_id': dm.get('set_id', ''),
                        'dailymed_sections': dm.get('dailymed_sections', {}),
                        # Include all 8 DailyMed sections for intelligent selection
                        'highlights': dm.get('highlights', ''),
                        'indications': dm.get('indications', ''),
                        'dosage': dm.get('dosage', ''),
                        'contraindications': dm.get('contraindications', ''),
                        'warnings': dm.get('warnings', ''),
                        'adverse_reactions': dm.get('adverse_reactions', ''),
                        'clinical_studies': dm.get('clinical_studies', ''),
                    })
                    existing_pmcids.add(pmcid)
            
            if dailymed_rows:
                dailymed_df = pd.DataFrame(dailymed_rows)
                papers_df = pd.concat([dailymed_df, papers_df], ignore_index=True)
                logger.info(f"   Total papers after DailyMed merge: {len(papers_df)}")

        papers_df, recency_stats = self._apply_final_recency_policy(papers_df)
        self._last_recency_stats = recency_stats

        # Keep only top-N final articles after all merges and filtering
        if not papers_df.empty:
            papers_df = self._attach_sort_score(papers_df)

            papers_df = (
                papers_df.sort_values(by="_sort_score", ascending=False)
                .head(self.final_top_articles)
                .drop(columns=["_sort_score"], errors="ignore")
                .reset_index(drop=True)
            )
            logger.info(f"   Final article cap applied: {len(papers_df)} (top {self.final_top_articles})")

        return papers_df, reranked

    
    def _deduplicate_dailymed(self, papers_df) -> Any:
        """
        Deduplicate DailyMed entries to keep only top entries per drug.
        
        Multiple manufacturers may have the same drug info, so we keep only
        the top MAX_DAILYMED_PER_DRUG entries per normalized drug name.
        
        Args:
            papers_df: DataFrame of papers
            
        Returns:
            Deduplicated DataFrame
        """
        if papers_df.empty:
            return papers_df
        
        import pandas as pd
        import re
        
        logger.info("🔄 Deduplicating DailyMed entries...")
        
        # Track DailyMed entries by normalized drug name
        dailymed_by_drug = {}  # drug_name -> list of (index, relevance_score)
        non_dailymed_indices = []
        
        for idx, row in papers_df.iterrows():
            pmcid = str(row.get('pmcid', '') or row.get('corpus_id', ''))
            article_type = str(row.get('article_type', ''))
            venue = str(row.get('venue', '') or row.get('journal', ''))
            
            is_dailymed = (
                pmcid.startswith('dailymed_') or 
                article_type == 'drug_label' or
                'dailymed' in venue.lower() or
                'fda drug label' in venue.lower()
            )
            
            if is_dailymed:
                # Extract and normalize drug name from title
                title = str(row.get('title', ''))
                # Remove trailing form descriptors (Tablets, Injection, etc.)
                # Use \s+ (one or more whitespace) to ensure form words only match
                # when preceded by drug name, not at the start of the title
                normalized_name = re.sub(
                    r'\s+(?:tablets?|capsules?|injection|solution|oral|intravenous|iv|im|powder|suspension|syrup|cream|ointment|gel|patch|spray)\b.*$',
                    '',
                    title,
                    flags=re.IGNORECASE
                ).strip().lower()
                
                # Further normalize: remove manufacturer info in parentheses
                normalized_name = re.sub(r'\s*\([^)]*\)\s*$', '', normalized_name).strip()
                
                # If normalized name is empty, use pmcid as unique key to avoid
                # incorrectly grouping different drugs with missing/invalid titles
                if not normalized_name:
                    normalized_name = f"__unique_{pmcid}"
                
                relevance = row.get('relevance_judgement', row.get('relevance_score', 0))
                
                if normalized_name not in dailymed_by_drug:
                    dailymed_by_drug[normalized_name] = []
                dailymed_by_drug[normalized_name].append((idx, relevance))
            else:
                non_dailymed_indices.append(idx)
        
        # Keep only top MAX_DAILYMED_PER_DRUG per drug
        kept_dailymed_indices = []
        removed_count = 0
        
        for drug_name, entries in dailymed_by_drug.items():
            # Sort by relevance score (descending)
            sorted_entries = sorted(entries, key=lambda x: x[1], reverse=True)
            
            # Keep top entries
            kept = sorted_entries[:MAX_DAILYMED_PER_DRUG]
            removed = sorted_entries[MAX_DAILYMED_PER_DRUG:]
            
            kept_dailymed_indices.extend([idx for idx, _ in kept])
            removed_count += len(removed)
            
            if len(sorted_entries) > MAX_DAILYMED_PER_DRUG:
                logger.info(f"   Drug '{drug_name}': kept {len(kept)}/{len(sorted_entries)} DailyMed entries")
        
        if removed_count > 0:
            logger.info(f"   Removed {removed_count} duplicate DailyMed entries (keeping max {MAX_DAILYMED_PER_DRUG} per drug)")
            
            # Combine non-DailyMed + kept DailyMed indices, then sort to preserve original ranking
            all_kept_indices = sorted(non_dailymed_indices + kept_dailymed_indices)
            papers_df = papers_df.loc[all_kept_indices].reset_index(drop=True)
        else:
            logger.info("   No duplicate DailyMed entries found")
        
        return papers_df
    
    def _filter_by_entities(self, query: str, papers_df, medical_conditions: List[str] = None, corrected_conditions: List[str] = None) -> Any:
        """
        Filter papers based on medical entity matching.
        
        When LLM-extracted medical_conditions are provided, uses flexible matching:
        - Splits compound conditions into individual terms
        - Normalizes case variations (e.g., IgG4 vs IGG4)
        - Matches if paper contains ANY key term from the condition
        - Uses BOTH original and typo-corrected conditions for matching
        
        Falls back to regex-based entity extraction if no conditions provided.
        
        Args:
            query: Original query
            papers_df: DataFrame of papers
            medical_conditions: LLM-extracted conditions for matching (original spelling)
            corrected_conditions: Typo-corrected conditions for matching
            
        Returns:
            Filtered DataFrame
        """
        if papers_df.empty:
            return papers_df
        
        import pandas as pd
        import re
        
        logger.info("🔍 Post-retrieval filtering: Checking entity matches...")
        
        # Combine original and corrected conditions (corrected takes priority for matching)
        raw_entities = []
        if medical_conditions:
            raw_entities.extend(medical_conditions)
            logger.info(f"   Using LLM-extracted conditions: {medical_conditions}")
        if corrected_conditions:
            raw_entities.extend(corrected_conditions)
            logger.info(f"   Also using typo-corrected conditions: {corrected_conditions}")
        
        # Fall back to regex extraction if no LLM conditions
        if not raw_entities and self.entity_expander:
            raw_entities = self._extract_query_entities(query)
            logger.info(f"   Using regex-extracted entities: {raw_entities}")
        
        if not raw_entities:
            logger.info("   No medical entities found in query, skipping filter")
            return papers_df
        
        # ========================================================================
        # Extract KEY TERMS from conditions for flexible matching
        # e.g., "Immunoglobulin G IGG4 disease" -> ["igg4", "immunoglobulin"]
        # ========================================================================
        key_terms = set()
        for entity in raw_entities:
            # Add the full entity (normalized)
            entity_lower = entity.lower()
            
            # Split by common separators and extract meaningful terms
            words = re.split(r'[\s\-_,]+', entity_lower)
            for word in words:
                # Skip common filler words
                if word in {'of', 'the', 'and', 'or', 'in', 'a', 'an', 'disease', 'syndrome', 'disorder', 'related'}:
                    continue
                # Keep meaningful medical terms (3+ chars)
                if len(word) >= 3:
                    key_terms.add(word)
            
            # Also add the full entity for exact phrase matching
            key_terms.add(entity_lower)
        
        # Add common variations (e.g., igg4 matches IgG4, IGG4)
        # These will match case-insensitively in the paper text
        logger.info(f"   Key terms for matching: {sorted(key_terms)[:10]}...")
        
        # Check each paper for entity matches
        filtered_indices = []
        removed_count = 0
        
        for idx, row in papers_df.iterrows():
            title = str(row.get('title', '')).lower()
            abstract = str(row.get('abstract', '')).lower()
            
            # Combine title and abstract for searching
            paper_text = f"{title} {abstract}"
            
            # Check if paper contains at least one key term
            has_entity = False
            for term in key_terms:
                if term in paper_text:
                    has_entity = True
                    break
            
            if has_entity:
                filtered_indices.append(idx)
            else:
                removed_count += 1
                logger.debug(f"   Removed: {row.get('title', 'Untitled')[:60]}... (no entity match)")
        
        if removed_count > 0:
            logger.info(f"   Filtered out {removed_count} papers without entity matches")
            papers_df = papers_df.loc[filtered_indices].reset_index(drop=True)
        else:
            logger.info("   All papers contain query entities")
        
        return papers_df
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract medical entities (disease names, acronyms) from query.
        
        Args:
            query: Original query string
            
        Returns:
            List of medical entity terms
        """
        entities = []
        
        if self.entity_expander is None:
            return entities
        
        # Extract acronyms and their expansions
        words = query.split()
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if it's a known acronym
            if self.entity_expander._is_likely_acronym(clean_word):
                expansions = self.entity_expander.expand_acronym(clean_word)
                if expansions:
                    # Add both acronym and full term
                    entities.append(clean_word.upper())
                    entities.append(expansions[0])
        
        # Also look for common medical condition patterns
        # (e.g., "Antiphospholipid Syndrome", "Classification Criteria")
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Syndrome\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Disease\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Disorder\b',
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen:
                seen.add(entity_lower)
                unique_entities.append(entity)
        
        return unique_entities
    
    # =========================================================================
    # Step 4: Direct LLM Synthesis
    # =========================================================================
    
    def _get_papers_for_context(self, papers_df, query: str = ""):
        """Build context and get used papers from reranked articles.
        
        For DailyMed articles, intelligently selects sections:
        - Always: highlights + clinical_studies
        """
        from .specialty_journals import PRIORITY_JOURNALS, PRIORITY_JOURNALS_NLM
        import math

        HIGH_VALUE_TYPES = {'review_article', 'clinical_trial', 'systematic_review', 'meta_analysis', 'guideline'}
        MAX_ABSTRACT_CHARS = 1200
        MAX_PMC_FULL_TEXT_CHARS = 30000
        
        priority_journal_papers = []
        other_papers = []

        for idx, row in papers_df.head(MAX_ABSTRACTS).iterrows():
            journal = row.get('journal', '') or row.get('venue', '')
            corpus_id = str(row.get('corpus_id', '')).strip()
            nlm_unique_id = str(row.get('nlm_unique_id', '')).strip() if row.get('nlm_unique_id') else ''
            normalized_journal = re.sub(r'[^a-z0-9]+', ' ', str(journal).lower()).strip()
            
            # Priority check: NLM Unique ID (most reliable), then corpus_id, then journal name
            is_priority_journal = (
                nlm_unique_id in PRIORITY_JOURNALS_NLM
                or corpus_id in PRIORITY_JOURNALS
                or journal in PRIORITY_JOURNALS
                or normalized_journal in PRIORITY_JOURNALS
            )
            article_type = row.get('article_type', 'other')
            is_high_value = article_type in HIGH_VALUE_TYPES

            if is_priority_journal or is_high_value:
                priority_journal_papers.append((idx, row))
            else:
                other_papers.append((idx, row))

        all_papers = priority_journal_papers + other_papers
        used_papers = []
        context_parts = []
        dailymed_section_count = 0
        pmc_full_text_included = 0
        recent_cutoff_year = self._recent_cutoff_year()
        self._last_context_stats = {
            "pmc_recent_fulltext_used": 0,
            "recent_cutoff_year": recent_cutoff_year,
        }

        def _safe_str(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float) and math.isnan(value):
                return ""
            return str(value).strip()

        for i, (idx, row) in enumerate(all_papers[:MAX_ABSTRACTS]):
            article_type = row.get('article_type', 'other')
            source_num = len(context_parts) + 1
            
            if hasattr(row, 'to_dict'):
                paper_dict = row.to_dict()
            else:
                paper_dict = dict(row)
            # Preserve canonical citation index so UI/reference ordering is stable.
            paper_dict["citation_index"] = source_num
            used_papers.append(paper_dict)

            paper_info = f"**[{source_num}]** {row.get('title', 'Untitled')}"
            if row.get('venue') or row.get('journal'):
                paper_info += f" | *{row.get('venue') or row.get('journal')}*"
            if row.get('year'):
                paper_info += f" ({row.get('year')})"
            paper_info += f" [{article_type}]"

            # DailyMed: Intelligent section selection
            if article_type == "drug_label" or row.get("source") == "dailymed":
                sections = []
                section_map = row.get("dailymed_sections", {})
                if not isinstance(section_map, dict):
                    section_map = {}

                def _safe_text(value: Any) -> str:
                    import math
                    if value is None:
                        return ""
                    if isinstance(value, float) and math.isnan(value):
                        return ""
                    return str(value).strip()

                def _get_dm_section(key: str) -> str:
                    direct_val = _safe_text(row.get(key, ""))
                    if direct_val:
                        return direct_val
                    return _safe_text(section_map.get(key, ""))
                
                # Always include highlights (summary + boxed warning)
                highlights = _get_dm_section("highlights")
                if highlights:
                    sections.append(f"### Highlights of Prescribing Information\n{highlights[:8000]}")
                
                # Always include clinical studies (efficacy data, trial results)
                clinical_studies = _get_dm_section("clinical_studies")
                if clinical_studies:
                    sections.append(f"### Clinical Studies\n{clinical_studies[:15000]}")
                
                if sections:
                    text_content = "\n\n".join(sections)
                    dailymed_section_count += 1
                else:
                    # Fallback to abstract if no sections available
                    text_content = (row.get('abstract', '') or row.get('text', ''))[:6000]
                
                text_content = self._clean_source_text(text_content)
                paper_info += f"\n{text_content}"
            else:
                pmcid = _safe_str(row.get('pmcid') or row.get('corpus_id')).upper()
                doc_id = _safe_str(row.get('doc_id') or row.get('corpus_id') or row.get('pmcid'))
                is_pmc_article = pmcid.startswith('PMC')
                year_value = self._to_int_year(row.get("year"))
                is_recent_for_fulltext = year_value is not None and year_value >= recent_cutoff_year

                text_content = ""
                should_load_pmc_fulltext = (
                    is_pmc_article
                    and pmc_full_text_included < 2
                    and doc_id
                    and (
                        not self.pmc_fulltext_recent_only
                        or is_recent_for_fulltext
                    )
                )
                if should_load_pmc_fulltext:
                    chunks = self.retriever.get_all_chunks_for_doc(doc_id)
                    reconstructed_text = self._reconstruct_full_text_from_chunks(chunks)
                    if reconstructed_text:
                        text_content = reconstructed_text[:MAX_PMC_FULL_TEXT_CHARS]
                        pmc_full_text_included += 1

                if not text_content:
                    # For remaining PMC + PubMed, keep abstract-only context.
                    text_content = (row.get('abstract', '') or row.get('text', ''))[:MAX_ABSTRACT_CHARS]

                text_content = self._clean_source_text(text_content)
                paper_info += f"\n{text_content}"
            
            context_parts.append(paper_info)
            
        logger.info(
            "Context built: %d papers (%d DailyMed with section selection, %d PMC full-text sources, cutoff=%d recent_only=%s).",
            len(context_parts),
            dailymed_section_count,
            pmc_full_text_included,
            recent_cutoff_year,
            self.pmc_fulltext_recent_only,
        )
        self._last_context_stats = {
            "pmc_recent_fulltext_used": pmc_full_text_included,
            "recent_cutoff_year": recent_cutoff_year,
        }
        return context_parts, used_papers

    def _reconstruct_full_text_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Reconstruct document text from ordered chunks while deduplicating section text.
        """
        if not chunks:
            return ""

        ordered_sections = []
        seen_sections = set()

        for chunk in chunks:
            section_text = str(
                chunk.get("full_section_text")
                or chunk.get("page_content")
                or chunk.get("text")
                or ""
            ).strip()
            if not section_text:
                continue

            normalized = re.sub(r"\s+", " ", section_text).strip()
            if not normalized or normalized in seen_sections:
                continue

            seen_sections.add(normalized)
            ordered_sections.append(section_text)

        return "\n\n".join(ordered_sections)

    def _clean_source_text(self, text: str) -> str:
        """
        Strip internal citations from source text to prevent LLM confusion.
        Target patterns: [1], [1, 2], [1-5], [1,2,3], etc.
        """
        if not text:
            return ""
        
        # Pattern for [1], [1, 2], [1-5], [1,2,3] etc.
        # Targets brackets containing numbers, commas, spaces, and hyphens/dashes.
        # Includes leading space to avoid "statement [1]." -> "statement ."
        cleaned = re.sub(r'\s*\[[\d\s,\-\–\.]+\]', '', text)
        
        # Also handle potential superscript numbers if they were converted to text like ^1
        cleaned = re.sub(r'\^[\d,]+', '', cleaned)
        
        # Basic cleanup of double spaces resulting from removal
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        
        return cleaned.strip()

    def run_generation(
        self,
        query: str,
        papers_df,
        stream: bool = False
    ):
        """Generate answer directly using LLM with ELIXIR system prompt."""
        logger.info("🧠 Step 4: Direct LLM Synthesis")

        if papers_df.empty:
            return ("No relevant evidence found for your query.", [], [])

        context_parts, used_papers = self._get_papers_for_context(papers_df, query)
        abstract_count = len(context_parts)

        logger.info(f"   Context: {abstract_count} abstracts = {len(context_parts)} total articles")
        context = "\n\n---\n\n".join(context_parts)

        # Use ELIXIR_SYSTEM_PROMPT for comprehensive deep research responses
        system_prompt = ELIXIR_SYSTEM_PROMPT
        logger.info(f"   Mode: Deep Research | Prompt: {len(system_prompt)} chars")

        # Use comprehensive deep research prompt
        user_prompt = f"""
# [QUERY]
{query}

# [CONTEXT]
(Source Literature: {abstract_count} articles)
{context}

# [INSTRUCTIONS]
Analyze and synthesize the medical literature above to create a detailed, clinically-focused clinical review or practice guideline section.
1. **Extract specific clinical details**: Classification systems, staging criteria, detailed medication protocols (dosing, administration, duration), trial results (outcomes, p-values), and guideline recommendations.
2. **Provide comparative analyses**: Efficacy comparisons with data points and safety profiles.
3. **Structure**: Use clear hierarchical headings, markdown tables for comparisons/staging, and evidence-based recommendations.
4. **Citation**: Use inline citations **[1]**, **[2]**, etc. strictly.
   - DailyMed drug labels use the same citation numbering namespace as journal articles.
5. **Depth**: High technical detail for physician decision-making. No word count limit, but maintain density.
"""

        try:
            if stream:
                # Return a generator for tokens
                return self._stream_generation(query, user_prompt, used_papers)
            
            response = self._create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"   Generated response ({len(answer)} chars)")
            return answer, used_papers

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return (f"Error generating response: {str(e)}", [])

    def _stream_generation(self, query: str, user_prompt: str, used_papers: list) -> Generator[Dict[str, Any], None, None]:
        """Generator for token-by-token streaming."""
        try:
            # Use ELIXIR_SYSTEM_PROMPT for deep research
            system_prompt = ELIXIR_SYSTEM_PROMPT
            
            response = self._create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            
            full_answer = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer += token
                    yield {"step": "generation", "status": "running", "token": token}
            
            yield {"step": "generation", "status": "complete", "answer": full_answer, "used_papers": used_papers}
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield {"step": "error", "message": str(e)}
    
    # =========================================================================
    # Fallback Generation (when no sources found)
    # =========================================================================
    
    def _run_fallback_generation(self, query: str, stream: bool = False):
        """
        Generate response when no sources are found using the same LLM model.
        
        Args:
            query: Original user query
            stream: Whether to stream the response
            
        Returns:
            If stream=False: tuple of (answer, [])
            If stream=True: Generator yielding response events
        """
        logger.info(f"📭 No sources found, using {LLM_MODEL} for response")
        
        fallback_system_prompt = """You are a medical AI assistant providing information based on your training knowledge.

IMPORTANT: You are responding WITHOUT access to specific literature sources. Your response is based on general medical knowledge.

Guidelines:
- Provide accurate, evidence-based medical information
- Use appropriate technical terminology for healthcare professionals  
- Structure your response clearly with headings where appropriate
- Be direct and clinically focused
- Do NOT use citation numbers like [1], [2] etc. since there are no sources

At the END of your response, add this disclaimer:
---
*Note: This response is based on the model's training knowledge as no relevant literature sources were found for this specific query. Please verify with current clinical guidelines and authoritative sources.*"""

        user_prompt = f"""Medical Query: {query}

Please provide a comprehensive clinical response based on your medical knowledge."""

        try:
            if stream:
                return self._stream_fallback_generation(fallback_system_prompt, user_prompt)
            
            response = self._create_chat_completion(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": fallback_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"   Fallback response generated ({len(answer)} chars)")
            return answer, []
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return (f"Unable to generate response: {str(e)}", [])
    
    def _stream_fallback_generation(self, system_prompt: str, user_prompt: str) -> Generator[Dict[str, Any], None, None]:
        """Stream fallback generation tokens."""
        try:
            response = self._create_chat_completion(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            
            full_answer = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer += token
                    yield {"step": "generation", "status": "running", "token": token}
            
            yield {"step": "generation", "status": "complete", "answer": full_answer, "used_papers": []}
            
        except Exception as e:
            logger.error(f"Fallback streaming failed: {e}")
            yield {"step": "error", "message": str(e)}

    def _create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ):
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
        }
        if stream:
            request_kwargs["stream"] = True
        if self.llm_provider == "groq":
            request_kwargs["max_completion_tokens"] = LLM_MAX_COMPLETION_TOKENS
            if LLM_REASONING_EFFORT:
                request_kwargs["reasoning_effort"] = LLM_REASONING_EFFORT
        return self.llm_client.chat.completions.create(**request_kwargs)
    
    # =========================================================================
    # PDF Availability Check via Europe PMC
    # =========================================================================
    
    def _check_pdf_availability(self, papers: list) -> list:
        """
        Check PDF availability for papers via Europe PMC API.
        Uses one batched Europe PMC query built from DOI/PMCID identifiers.
        
        Europe PMC API returns:
        - isOpenAccess: "Y" or "N" (string)
        - inEPMC: "Y" or "N" (string) 
        - hasPDF: "Y" or "N" (string)
        - fullTextUrlList.fullTextUrl: Array with documentStyle, availabilityCode, url
        
        PDF URL format: https://europepmc.org/articles/{PMCID}?pdf=render
        
        Args:
            papers: List of paper dicts with doi, pmcid, pmid fields
            
        Returns:
            List of source dicts with pdf_url field added where available
        """
        import requests

        def _normalize_doi(doi: Any) -> str:
            raw = str(doi or "").strip()
            if not raw:
                return ""
            raw = re.sub(r"^https?://(dx\.)?doi\.org/", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"^doi:\s*", "", raw, flags=re.IGNORECASE)
            return raw.strip().lower()

        def _normalize_pmcid(pmcid: Any) -> str:
            raw = str(pmcid or "").strip().upper()
            if not raw:
                return ""
            raw = raw.replace("PMCID:", "").strip()
            raw = re.sub(r"\s+", "", raw)
            if raw.startswith("PMC"):
                return raw
            if raw.isdigit():
                return f"PMC{raw}"
            return raw

        def _extract_pdf_url(article: Dict[str, Any]) -> str:
            url_list = article.get("fullTextUrlList", {}).get("fullTextUrl", [])
            if isinstance(url_list, dict):
                url_list = [url_list]

            for url_info in url_list:
                if not isinstance(url_info, dict):
                    continue
                doc_style = str(url_info.get("documentStyle", "")).lower()
                avail_code = str(url_info.get("availabilityCode", "")).upper()
                availability = str(url_info.get("availability", "")).lower()

                if doc_style == "pdf" and (avail_code == "OA" or "open access" in availability):
                    pdf_url = url_info.get("url")
                    if pdf_url:
                        return pdf_url

            is_open_access = article.get("isOpenAccess") == "Y"
            in_epmc = article.get("inEPMC") == "Y"
            has_pdf = article.get("hasPDF") == "Y"
            if has_pdf and (is_open_access or in_epmc):
                article_pmcid = _normalize_pmcid(article.get("pmcid") or article.get("id"))
                if article_pmcid:
                    return f"https://europepmc.org/articles/{article_pmcid}?pdf=render"

            return ""

        logger.info("📄 Step 5: Checking PDF availability via Europe PMC")
        
        if not papers:
            return []
        
        normalized_papers = []
        for i, paper in enumerate(papers):
            if hasattr(paper, 'to_dict'):
                p = paper.to_dict()
            else:
                p = dict(paper)
            if p.get("citation_index") in (None, ""):
                p["citation_index"] = i + 1
            normalized_papers.append(p)

        sources: List[Dict[str, Any]] = []
        doi_to_indices: Dict[str, List[int]] = {}
        pmcid_to_indices: Dict[str, List[int]] = {}
        query_terms: List[str] = []
        seen_dois = set()
        seen_pmcids = set()

        for i, paper in enumerate(normalized_papers):
            source = self._map_paper_to_source(paper)
            source["_order_index"] = i
            sources.append(source)

            doi_key = _normalize_doi(source.get("doi"))
            if doi_key:
                doi_to_indices.setdefault(doi_key, []).append(i)
                if doi_key not in seen_dois:
                    query_terms.append(f'DOI:"{doi_key}"')
                    seen_dois.add(doi_key)

            pmcid_key = _normalize_pmcid(source.get("pmcid"))
            if pmcid_key:
                pmcid_to_indices.setdefault(pmcid_key, []).append(i)
                if pmcid_key not in seen_pmcids:
                    query_terms.append(f"PMCID:{pmcid_key}")
                    seen_pmcids.add(pmcid_key)

        if query_terms:
            batch_query = "(" + " OR ".join(query_terms) + ")"
            api_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                "query": batch_query,
                "format": "json",
                "resultType": "core",
                "pageSize": 1000,
            }

            try:
                response = requests.get(api_url, params=params, timeout=20)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("resultList", {}).get("result", [])

                    for article in results:
                        pdf_url = _extract_pdf_url(article)
                        if not pdf_url:
                            continue

                        matched_indices = set()

                        doi_key = _normalize_doi(article.get("doi"))
                        if doi_key:
                            matched_indices.update(doi_to_indices.get(doi_key, []))

                        pmcid_key = _normalize_pmcid(article.get("pmcid"))
                        if pmcid_key:
                            matched_indices.update(pmcid_to_indices.get(pmcid_key, []))

                        # Some records expose PMCID in the generic "id" field.
                        pmcid_id_key = _normalize_pmcid(article.get("id"))
                        if pmcid_id_key:
                            matched_indices.update(pmcid_to_indices.get(pmcid_id_key, []))

                        for idx in matched_indices:
                            if not sources[idx].get("pdf_url"):
                                sources[idx]["pdf_url"] = pdf_url
                else:
                    logger.debug(f"Batch PDF check failed with status {response.status_code}")
            except requests.exceptions.Timeout:
                logger.debug("Batch PDF check timed out")
            except Exception as e:
                logger.debug(f"Batch PDF check failed: {e}")

        # Preserve stable citation order for UI/inline citation mapping.
        def _citation_rank(source: Dict[str, Any]) -> int:
            val = source.get("citation_index")
            try:
                if val is None or val == "":
                    return 10**9
                return int(val)
            except (TypeError, ValueError):
                return 10**9

        sources.sort(key=lambda s: (_citation_rank(s), s.get("_order_index", 10**9)))
        for source in sources:
            source.pop("_order_index", None)
        
        # Count and log PDFs found
        pdf_count = sum(1 for s in sources if s.get("pdf_url"))
        logger.info(f"   Found {pdf_count}/{len(sources)} articles with open access PDFs")
        
        return sources

    def _map_paper_to_source(self, paper: Any) -> Dict[str, Any]:
        """
        Helper to map a paper record (Dict or Series) to a standardized source object.
        Used both in initial streaming and final PDF verification.
        """
        # Convert Series/Namespace to dict if needed
        if hasattr(paper, 'to_dict'):
            p = paper.to_dict()
        else:
            p = dict(paper)

        pmcid = p.get("pmcid") or p.get("corpus_id", "")
        source_type = p.get("source", "")
        
        # Detect DailyMed articles
        is_dailymed = (
            source_type == "dailymed" or 
            p.get("article_type") == "drug_label" or
            str(pmcid).startswith("dailymed_")
        )
        
        # Extract set_id for DailyMed articles
        set_id = p.get("set_id", "")
        if is_dailymed and not set_id and str(pmcid).startswith("dailymed_"):
            set_id = pmcid.replace("dailymed_", "")

        # Helper to sanitize NaN values (pandas can return float('nan') which breaks JSON)
        def sanitize(val, default=""):
            import math
            if val is None:
                return default
            if isinstance(val, float) and math.isnan(val):
                return default
            return val

        return {
            "pmcid": pmcid,
            "pmid": sanitize(p.get("pmid"), ""),
            "title": sanitize(p.get("title"), "Untitled"),
            "authors": p.get("authors", []),
            "journal": sanitize(p.get("venue") or p.get("journal"), ""),
            "year": sanitize(p.get("year")),
            "doi": sanitize(p.get("doi"), ""),
            "article_type": sanitize(p.get("article_type"), ""),
            "evidence_grade": sanitize(p.get("evidence_grade"), None),
            "evidence_level": sanitize(p.get("evidence_level"), None),
            "evidence_term": sanitize(p.get("evidence_term"), None),
            "evidence_source": sanitize(p.get("evidence_source"), None),
            "relevance_score": sanitize(p.get("relevance_judgement", p.get("relevance_score", 0)), 0),
            "pdf_url": sanitize(p.get("pdf_url")), # Preserve if already present
            "source": source_type,
            "set_id": set_id,
            "dailymed_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else None,
            "citation_index": sanitize(p.get("citation_index"), None),
        }
    
    # =========================================================================
    # Full Pipeline
    # =========================================================================
    
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Run complete Elixir RAG pipeline.
        
        Returns structured response with answer and sources.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 Elixir Pipeline: {query}")
        logger.info(f"{'='*60}")
        
        # Step 0: Check Cache
        cached_result = self._cache_get(query)
        if cached_result:
            # Add a marker that this was a cache hit
            cached_result["status"] = "cache_hit"
            return cached_result

        self._reset_run_stats()

        # Step 1: Preprocess
        processed_query = self.preprocess_query(query)
        
        # Step 2: Retrieve
        passages, dailymed_results = self.retrieve_passages(processed_query)
        
        # Apply hybrid scoring (dense + keyword matching)
        if passages:
            passages = self.retriever.apply_hybrid_scoring(
                query=processed_query.rewritten_query,
                passages=passages,
                dense_weight=0.7,
                sparse_weight=0.3
            )
        
        if not passages and not dailymed_results:
            # No passages retrieved - use fallback LLM generation
            logger.info("📭 No passages retrieved, using fallback generation")
            answer, _ = self._run_fallback_generation(query, stream=False)
            result = {
                "query": query,
                "report_title": "Clinical Response (No Sources)",
                "answer": answer,
                "sections": [],
                "sources": [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "status": "fallback"
            }
            # Cache the fallback result too
            self._cache_set(query, result)
            return result
        
        # Step 3: Rerank & Aggregate
        # Pass LLM-extracted medical conditions AND typo-corrected conditions for strict filtering
        medical_conditions = []
        corrected_conditions = []
        if processed_query.decomposed:
            if processed_query.decomposed.medical_conditions:
                medical_conditions = processed_query.decomposed.medical_conditions
            if processed_query.decomposed.corrected_medical_conditions:
                corrected_conditions = processed_query.decomposed.corrected_medical_conditions
        papers_df, reranked_passages = self.rerank_and_aggregate(query, passages, dailymed_results, medical_conditions, corrected_conditions)
        
        if papers_df.empty:
            # No papers after filtering - use fallback LLM generation
            logger.info("📭 No papers after filtering, using fallback generation")
            answer, _ = self._run_fallback_generation(query, stream=False)
            result = {
                "query": query,
                "report_title": "Clinical Response (No Sources)",
                "answer": answer,
                "sections": [],
                "sources": [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "status": "fallback"
            }
            # Cache fallback
            self._cache_set(query, result)
            return result
        
        # Step 4: Direct LLM Synthesis
        synth_result = self.run_generation(query, papers_df)
        
        # Handle tuple return (answer, used_papers) or error string
        if isinstance(synth_result, tuple):
            if len(synth_result) == 2:
                answer, used_papers = synth_result
            else:
                answer = synth_result[0]
                used_papers = []
        else:
            # Error case
            answer = synth_result
            used_papers = []
        
        # Step 5: Check PDF availability via Europe PMC for all used papers
        sources_with_pdf = self._check_pdf_availability(used_papers)
        
        final_result = {
            "query": query,
            "report_title": "Clinical Response",
            "answer": answer,
            "sections": [],  # Direct synthesis - no sections
            "sources": sources_with_pdf,
            "evidence_hierarchy": self.evidence_hierarchy,
            "retrieval_stats": self._build_retrieval_stats(
                passages_retrieved=len(passages),
                papers_after_aggregation=len(papers_df),
                abstracts_used=len(used_papers),
            ),
            "status": "success"
        }
        
        # Step 6: Store in Cache
        self._cache_set(query, final_result)
        
        return final_result
    
    def answer_streaming(
        self,
        query: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Streaming version for real-time UI updates with parallel PDF checks."""
        from concurrent.futures import ThreadPoolExecutor

        cached_result = self._cache_get(query)
        if cached_result:
            cached_result["status"] = "cache_hit"
            yield {
                "step": "complete",
                "status": "cache_hit",
                "report_title": cached_result.get("report_title", "Clinical Response"),
                "answer": cached_result.get("answer", ""),
                "sources": cached_result.get("sources", []),
                "evidence_hierarchy": cached_result.get("evidence_hierarchy", self.evidence_hierarchy),
                "abstracts_used": cached_result.get("retrieval_stats", {}).get("abstracts_used", 0),
                "cache_hit": True,
            }
            return

        self._reset_run_stats()

        # Step 1: Preprocess
        yield {"step": "query_expansion", "status": "running", "message": "Analyzing query..."}
        processed_query = self.preprocess_query(query)
        yield {
            "step": "query_expansion",
            "status": "complete",
            "data": {"rewritten": processed_query.rewritten_query}
        }

        # Step 2: Retrieval
        yield {"step": "retrieval", "status": "running", "message": "Searching literature..."}
        passages, dailymed_results = self.retrieve_passages(processed_query)
        yield {
            "step": "retrieval",
            "status": "complete",
            "data": {"count": len(passages) + len(dailymed_results)}
        }

        if not passages and not dailymed_results:
            # No passages retrieved - use fallback LLM generation with streaming
            logger.info("📭 No passages retrieved, using fallback streaming generation")
            yield {"step": "generation", "status": "running", "message": "Generating response from medical knowledge..."}
            fallback_gen = self._run_fallback_generation(query, stream=True)
            final_answer = ""
            for event in fallback_gen:
                if event["step"] == "generation" and event["status"] == "running":
                    yield event
                elif event["step"] == "generation" and event["status"] == "complete":
                    final_answer = event.get("answer", "")
            yield {
                "step": "complete",
                "status": "fallback",
                "report_title": "Clinical Response (No Sources)",
                "answer": final_answer,
                "sources": [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "abstracts_used": 0
            }
            self._cache_set(query, {
                "query": query,
                "report_title": "Clinical Response (No Sources)",
                "answer": final_answer,
                "sections": [],
                "sources": [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "status": "fallback",
                "retrieval_stats": self._build_retrieval_stats(
                    passages_retrieved=0,
                    papers_after_aggregation=0,
                    abstracts_used=0,
                ),
            })
            return

        # Step 3: Reranking
        yield {"step": "reranking", "status": "running", "message": "Ranking papers..."}
        # Pass LLM-extracted medical conditions AND typo-corrected conditions for strict filtering
        medical_conditions = []
        corrected_conditions = []
        if processed_query.decomposed:
            if processed_query.decomposed.medical_conditions:
                medical_conditions = processed_query.decomposed.medical_conditions
            if processed_query.decomposed.corrected_medical_conditions:
                corrected_conditions = processed_query.decomposed.corrected_medical_conditions
        papers_df, _ = self.rerank_and_aggregate(query, passages, dailymed_results, medical_conditions, corrected_conditions)
        
        # [NEW] Reorder papers so reference ranking matches final citation order from the start
        # This prevents the UI from "jumping" and ensures references are correctly numbered
        _, used_papers = self._get_papers_for_context(papers_df, query)

        # Prepare initial sources (no PDF URLs yet)
        initial_sources = [self._map_paper_to_source(p) for p in used_papers]

        yield {
            "step": "reranking",
            "status": "complete",
            "data": {"papers": len(papers_df)},
            "sources": initial_sources,  # Send sources early and in corrected order!
            "evidence_hierarchy": self.evidence_hierarchy,
        }

        if papers_df.empty:
            # No papers after filtering - use fallback LLM generation with streaming
            logger.info("📭 No papers after filtering, using fallback streaming generation")
            yield {"step": "generation", "status": "running", "message": "Generating response from medical knowledge..."}
            fallback_gen = self._run_fallback_generation(query, stream=True)
            final_answer = ""
            for event in fallback_gen:
                if event["step"] == "generation" and event["status"] == "running":
                    yield event
                elif event["step"] == "generation" and event["status"] == "complete":
                    final_answer = event.get("answer", "")
            yield {
                "step": "complete",
                "status": "fallback",
                "report_title": "Clinical Response (No Sources)",
                "answer": final_answer,
                "sources": initial_sources if initial_sources else [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "abstracts_used": 0
            }
            self._cache_set(query, {
                "query": query,
                "report_title": "Clinical Response (No Sources)",
                "answer": final_answer,
                "sections": [],
                "sources": initial_sources if initial_sources else [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "status": "fallback",
                "retrieval_stats": self._build_retrieval_stats(
                    passages_retrieved=len(passages),
                    papers_after_aggregation=0,
                    abstracts_used=0,
                ),
            })
            return

        # used_papers already computed above; reuse for generation

        # START PARALLEL TASKS: Generation + PDF Check
        yield {"step": "pdf_check", "status": "running", "message": "Checking PDF availability..."}
        yield {"step": "generation", "status": "running", "message": "Synthesizing response..."}
        
        pdf_results = []
        pdf_yielded = False
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start PDF check in background
            pdf_future = executor.submit(self._check_pdf_availability, used_papers)
            
            # Start streaming generation
            answer = ""
            generation_gen = self.run_generation(query, papers_df, stream=True)
            for event in generation_gen:
                if event["step"] == "generation" and event["status"] == "running":
                    if "token" in event:
                        yield event
                    
                    # [NEW] Check if PDF future is done during token synthesis
                    # Yield PDF links AS SOON AS AVAILABLE instead of waiting for end
                    if not pdf_yielded and pdf_future.done():
                        try:
                            pdf_results = pdf_future.result()
                            pdf_count = sum(1 for s in pdf_results if s.get("pdf_url"))
                            yield {
                                "step": "pdf_check", 
                                "status": "complete", 
                                "data": {"pdf_count": pdf_count},
                                "sources": pdf_results, # UPDATE SOURCES WITH PDFs NOW
                                "evidence_hierarchy": self.evidence_hierarchy,
                            }
                            pdf_yielded = True
                            logger.info(f"   ✅ PDF check yielded early ({pdf_count} PDFs found)")
                        except Exception as e:
                            logger.error(f"Error yielding PDF results early: {e}")

                elif event["step"] == "generation" and event["status"] == "complete":
                    answer = event["answer"]
            
            # Final safety check if not already yielded
            if not pdf_results:
                pdf_results = pdf_future.result()

        if not pdf_yielded:
            yield {
                "step": "pdf_check",
                "status": "complete",
                "data": {"pdf_count": sum(1 for s in pdf_results if s.get("pdf_url"))},
                "sources": pdf_results,
                "evidence_hierarchy": self.evidence_hierarchy,
            }
        yield {"step": "generation", "status": "complete"}
        
        # NO FILTERING: Show all sources that were provided as context
        # Use initial_sources as base to ensure we never return empty list if PDFs check fails
        final_sources = pdf_results if pdf_results else initial_sources
        
        logger.info(f"🔍 Streaming Complete: Returning {len(final_sources)} sources")

        # Final event
        final_result = {
            "step": "complete",
            "status": "success",
            "report_title": "Clinical Response",
            "answer": answer,
            "sources": final_sources,
            "evidence_hierarchy": self.evidence_hierarchy,
            "abstracts_used": len(used_papers),
            "original_sources_count": len(pdf_results),
            "retrieval_stats": self._build_retrieval_stats(
                passages_retrieved=len(passages),
                papers_after_aggregation=len(papers_df),
                abstracts_used=len(used_papers),
            ),
        }
        yield final_result
        self._cache_set(query, {
            "query": query,
            "report_title": final_result["report_title"],
            "answer": final_result["answer"],
            "sections": [],
            "sources": final_result["sources"],
            "evidence_hierarchy": final_result["evidence_hierarchy"],
            "status": "success",
            "retrieval_stats": self._build_retrieval_stats(
                passages_retrieved=len(passages),
                papers_after_aggregation=len(papers_df),
                abstracts_used=len(used_papers),
            ),
        })
    
    # Backward compatibility
    def answer_scholarqa_style(self, query: str) -> Dict[str, Any]:
        """Alias for answer()."""
        return self.answer(query)


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing Elixir Medical RAG Pipeline")
    print("=" * 70)
    
    pipeline = MedicalRAGPipeline()
    result = pipeline.answer("management of copd")
    
    print(f"\n📋 Report: {result['report_title']}")
    print(f"📚 Sources: {len(result['sources'])}")
    print(f"\n{result['answer'][:2000]}...")
