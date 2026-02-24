"""
Qdrant Retriever Adapter for ScholarQA-style Pipeline.

Implements retrieval from Qdrant Vector Database with:
- Semantic vector search using mxbai-embed-large-v1 embeddings (Cloud Inference)
- Metadata filtering (year, venue, article_type, country)
- Passage-level results with section information
- Bulk search with multiple query embeddings
"""

import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText, Range, SparseVector, SearchParams, QuantizationSearchParams, QueryRequest
from qdrant_client.http.models import Document

from .config import (
    QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, SCORE_THRESHOLD, 
    EMBEDDING_MODEL, EMBEDDING_PROVIDER, QDRANT_CLOUD_INFERENCE, QDRANT_TIMEOUT, 
    QDRANT_RETRY_COUNT, QDRANT_RETRY_DELAY, USE_HYBRID_SEARCH,
    SPARSE_RETRIEVAL_MODE, SPARSE_MAX_TERMS_QUERY, SPARSE_MIN_TOKEN_LEN,
    SPARSE_REMOVE_STOPWORDS, DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL,
    DEEPINFRA_EMBED_TIMEOUT_SECONDS,
    QUANTIZATION_RESCORE, QUANTIZATION_OVERSAMPLING
)
from .bm25_sparse import BM25SparseEncoder
from .retry_utils import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base (matches ScholarQA pattern)
# =============================================================================

class AbstractRetriever(ABC):
    """Abstract base class for retrievers (mirrors ScholarQA's interface)."""
    
    @abstractmethod
    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Retrieve relevant passages for a query."""
        pass
    
    @abstractmethod
    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Retrieve additional papers via keyword/fallback search."""
        pass


# =============================================================================
# Qdrant Retriever
# =============================================================================

class QdrantRetriever(AbstractRetriever):
    """
    Qdrant-based retriever for medical articles.
    
    Features:
    - Vector search with mxbai-embed-large-v1 (1024-d) via Cloud Inference
    - Metadata filtering (year range, venue, article type, country)
    - Bulk search with multiple query embeddings
    - Integration with ScholarQA pipeline patterns
    """
    
    def __init__(
        self,
        n_retrieval: int = 150,
        n_keyword_search: int = 30,
        score_threshold: float = SCORE_THRESHOLD
    ):
        """
        Initialize Qdrant retriever with Cloud Inference and local fallback.

        Args:
            n_retrieval: Number of passages to retrieve per query
            n_keyword_search: Number of additional papers from keyword search
            score_threshold: Minimum similarity score
        """
        # Disable cloud_inference unless explicitly using qdrant_cloud_inference provider
        use_cloud = QDRANT_CLOUD_INFERENCE and EMBEDDING_PROVIDER == "qdrant_cloud_inference"
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
            cloud_inference=use_cloud,
        )
        self.retry_count = QDRANT_RETRY_COUNT
        self.retry_delay = QDRANT_RETRY_DELAY
        self.collection_name = COLLECTION_NAME
        self.embedding_model = EMBEDDING_MODEL
        self.embedding_provider = EMBEDDING_PROVIDER
        self.n_retrieval = n_retrieval
        self.n_keyword_search = n_keyword_search
        self.score_threshold = score_threshold

        # Initialize embedding backend based on provider
        self.local_encoder = None
        self.openai_client = None

        if self.embedding_provider == "deepinfra":
            from openai import OpenAI
            if not DEEPINFRA_API_KEY:
                raise ValueError("DEEPINFRA_API_KEY not set in config")
            self.openai_client = OpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_BASE_URL,
                timeout=DEEPINFRA_EMBED_TIMEOUT_SECONDS,
            )
            logger.info("✅ DeepInfra embedding initialized (model: %s)", EMBEDDING_MODEL)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}. Only 'deepinfra' is supported.")

        # Initialize sparse query encoder for hybrid search.
        self.bm25_sparse_encoder = None
        self.sparse_mode = SPARSE_RETRIEVAL_MODE
        if USE_HYBRID_SEARCH:
            self.bm25_sparse_encoder = BM25SparseEncoder(
                max_terms_doc=max(SPARSE_MAX_TERMS_QUERY, 64),
                max_terms_query=SPARSE_MAX_TERMS_QUERY,
                min_token_len=SPARSE_MIN_TOKEN_LEN,
                remove_stopwords=SPARSE_REMOVE_STOPWORDS,
            )
            logger.info("✅ BM25 sparse encoder initialized for hybrid search")
        else:
            logger.info("ℹ️ Hybrid search disabled in config")
        
        # Load drug name → set_id lookup for DailyMed
        self.drug_setid_lookup = {}
        try:
            import os
            lookup_path = os.path.join(os.path.dirname(__file__), "data", "drug_setid_lookup.json")
            if os.path.exists(lookup_path):
                import json
                with open(lookup_path, "r") as f:
                    self.drug_setid_lookup = json.load(f)
                logger.info(f"✅ Loaded drug lookup with {len(self.drug_setid_lookup)} drugs")
            else:
                logger.info(f"ℹ️ Drug lookup file not found, will use lazy cache + BM25 fallback")
        except Exception as e:
            logger.info(f"ℹ️ Drug lookup not loaded, will use lazy cache + BM25 fallback: {e}")
        
        # Lazy in-memory cache for drug lookups (built on-demand)
        self._drug_lookup_cache: Dict[str, str] = {}
        
        logger.info(
            "Initialized QdrantRetriever: %s (Embedding: %s, Sparse Mode: %s)",
            COLLECTION_NAME,
            self.embedding_provider,
            self.sparse_mode if USE_HYBRID_SEARCH else "disabled",
        )

    def _rank_key_from_point(self, point) -> str:
        """
        Key used for fusion/ranking.
        Prefer chunk-level identity so chunked corpora are not collapsed to one paper hit.
        """
        payload = point.payload or {}
        return str(
            payload.get("chunk_id")
            or payload.get("point_id")
            or point.id
            or payload.get("pmcid")
            or payload.get("doc_id")
        )

    def _doc_id_from_payload(self, payload: Dict[str, Any]) -> str:
        """Stable document ID for paper-level aggregation."""
        return str(payload.get("pmcid") or payload.get("doc_id") or payload.get("pmid") or "")

    def _extract_evidence_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract normalized evidence metadata from Qdrant payload."""
        return {
            "evidence_grade": payload.get("evidence_grade"),
            "evidence_level": payload.get("evidence_level"),
            "evidence_term": payload.get("evidence_term"),
            "evidence_source": payload.get("evidence_source"),
        }

    def _normalize_publication_type_list(self, value: Any) -> List[str]:
        """Normalize publication types from mixed payload formats."""
        if value is None:
            return []

        def _normalize_item(item: Any) -> str:
            if isinstance(item, dict):
                item = item.get("type") or item.get("name") or item.get("value")
            if item is None:
                return ""
            return str(item).strip()

        if isinstance(value, str):
            cleaned = _normalize_item(value)
            return [cleaned] if cleaned else []

        if isinstance(value, (list, tuple, set)):
            normalized: List[str] = []
            seen = set()
            for item in value:
                cleaned = _normalize_item(item)
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    normalized.append(cleaned)
            return normalized

        cleaned = _normalize_item(value)
        return [cleaned] if cleaned else []

    def _embed_query(self, query: str, use_instruction: bool = True):
        """
        Embed a single query using the configured embedding provider.
        
        For DeepInfra API, manually prepends instruction for better retrieval performance
        (1-5% improvement according to Qwen3 documentation).
        
        Args:
            query: The search query string
            use_instruction: Whether to prepend instruction (default True for queries)
        
        Returns a vector (list of floats) or Qdrant Document object,
        suitable for passing to client.query_points(query=...).
        Returns None if no embedding method is available.
        """
        
        if self.embedding_provider == "deepinfra" and self.openai_client is not None:
            try:
                # For queries, add instruction to improve retrieval (1-5% improvement)
                if use_instruction:
                    instruction = "Given a medical question, retrieve relevant clinical passages that answer the query"
                    query_with_instruct = f"Instruct: {instruction}\nQuery: {query}"
                else:
                    query_with_instruct = query

                response = retry_with_exponential_backoff(
                    lambda: self.openai_client.embeddings.create(
                        model=self.embedding_model,
                        input=[query_with_instruct],
                        encoding_format="float"
                    ),
                    max_attempts=self.retry_count + 1,
                    base_delay=float(self.retry_delay),
                    operation_name="DeepInfra embeddings.create",
                    logger=logger,
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"DeepInfra query embedding failed: {e}")
                return None

        if self.embedding_provider == "qdrant_cloud_inference":
            return Document(text=query, model=self.embedding_model)

        if self.local_encoder is not None:
            try:
                vec = self.local_encoder.encode(query, normalize_embeddings=True)
                return vec.tolist()
            except Exception as e:
                logger.error(f"Local query embedding failed: {e}")
                return None

        return None

    def _build_sparse_query_vector(self, query: str) -> Optional[SparseVector]:
        """Build sparse query vector using BM25."""
        if self.bm25_sparse_encoder is not None:
            return self.bm25_sparse_encoder.encode_query(query)
        return None

    def build_sparse_query_vectors(self, queries: List[str]) -> List[SparseVector]:
        """Batch-build sparse vectors for queries using BM25."""
        if self.bm25_sparse_encoder is not None:
            return self.bm25_sparse_encoder.encode_queries(queries)
        return [SparseVector(indices=[], values=[]) for _ in queries]
    
    def _build_filter(self, **kwargs) -> Optional[Filter]:
        """
        Build Qdrant filter from search parameters.
        
        Supported filters:
        - year: "2020-2025" or "2020-" or "-2025"
        - venue: "NEJM,Lancet" (comma-separated)
        - article_type: "systematic-review,rct"
        
        Note: DailyMed articles are EXCLUDED from semantic search to prevent
        retrieving drugs used for similar conditions. DailyMed should only
        appear when user explicitly asks about a specific drug.
        """
        conditions = []
        
        # EXCLUDE DailyMed from semantic search - prevents false drug matches
        # DailyMed should only be retrieved via exact keyword search
        conditions.append(
            Filter(
                must_not=[
                    FieldCondition(key="source", match=MatchValue(value="dailymed")),
                    FieldCondition(key="article_type", match=MatchValue(value="drug_label")),
                ]
            )
        )
        
        # Year filter (standard - no special DailyMed handling needed now)
        if "year" in kwargs and kwargs["year"]:
            year_str = kwargs["year"]
            parts = year_str.split("-")
            
            range_params = {}
            if len(parts) >= 1 and parts[0]:
                range_params["gte"] = int(parts[0])
            if len(parts) >= 2 and parts[1]:
                range_params["lte"] = int(parts[1])
            
            if range_params:
                conditions.append(FieldCondition(key="year", range=Range(**range_params)))
        
        # Venue filter (OR logic across venues)
        if "venue" in kwargs and kwargs["venue"]:
            venues = [v.strip() for v in kwargs["venue"].split(",")]
            venue_conditions = []
            for venue in venues:
                venue_conditions.append(
                    FieldCondition(key="journal", match=MatchValue(value=venue))
                )
            if len(venue_conditions) == 1:
                conditions.append(venue_conditions[0])
            # For multiple venues, we'd need nested Filter with should
            # For simplicity, we'll filter in post-processing
        
        # Article type filter
        if "article_type" in kwargs and kwargs["article_type"]:
            types = [t.strip() for t in kwargs["article_type"].split(",")]
            if len(types) == 1:
                conditions.append(
                    FieldCondition(key="article_type", match=MatchValue(value=types[0]))
                )
        
        # Government affiliation filter (merged from gov pipeline)
        if "is_gov_affiliated" in kwargs and kwargs["is_gov_affiliated"] is not None:
            gov_value = kwargs["is_gov_affiliated"]
            # Handle both boolean and string values
            if isinstance(gov_value, str):
                gov_value = gov_value.lower() in ("true", "1", "yes")
            conditions.append(
                FieldCondition(key="is_gov_affiliated", match=MatchValue(value=gov_value))
            )
        
        # Government agency filter
        if "gov_agency" in kwargs and kwargs["gov_agency"]:
            agencies = [a.strip() for a in kwargs["gov_agency"].split(",")]
            if len(agencies) == 1:
                conditions.append(
                    FieldCondition(key="gov_agencies", match=MatchValue(value=agencies[0]))
                )
        
        if conditions:
            return Filter(must=conditions)
        return None

    
    def retrieve_passages(
        self,
        query: str,
        use_hybrid: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        precomputed_sparse_vector: Optional[SparseVector] = None,
        **filter_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant passages using vector search with optional hybrid search (dense + sparse).

        Args:
            query: Search query (embedded via Cloud Inference or locally)
            use_hybrid: Whether to use hybrid search (dense + BM25 sparse)
            dense_weight: Weight for dense vector scores (default: 0.7)
            sparse_weight: Weight for sparse vector scores (default: 0.3)
            precomputed_sparse_vector: Optional pre-computed BM25 sparse vector (for batch optimization)
            **filter_kwargs: Filters (year, venue, article_type, country)

        Returns:
            List of passage dictionaries with metadata
        """
        # Build filter
        search_filter = self._build_filter(**filter_kwargs)

        # Try hybrid search if sparse query encoder is available
        if use_hybrid and self.bm25_sparse_encoder is not None:
            return self._hybrid_search(
                query=query,
                search_filter=search_filter,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                precomputed_sparse_vector=precomputed_sparse_vector,
                filter_kwargs=filter_kwargs,
            )

        # Dense-only search using configured embedding provider
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            logger.error("No embedding method available")
            return []

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="dense",
                limit=self.n_retrieval,
                score_threshold=self.score_threshold,
                query_filter=search_filter,
                with_payload=True,
                search_params=SearchParams(
                    quantization=QuantizationSearchParams(
                        rescore=QUANTIZATION_RESCORE,
                        oversampling=QUANTIZATION_OVERSAMPLING
                    )
                )
            )
            logger.info(f"✅ Dense search successful, retrieved {len(results.points)} passages")
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

        passages = []
        for point in results.points:
            # Detect DailyMed articles by source or article_type
            source = point.payload.get("source", "")
            article_type = point.payload.get("article_type", "")
            is_dailymed = source == "dailymed" or article_type == "drug_label"
            
            if is_dailymed:
                # DailyMed schema mapping
                passage = self._transform_dailymed_payload(point.payload, point.score)
            else:
                doc_id = self._doc_id_from_payload(point.payload)
                # Standard PMC schema - abstracts or chunks
                passage = {
                    "corpus_id": doc_id,
                    "pmcid": point.payload.get("pmcid", doc_id),
                    "pmid": point.payload.get("pmid"),
                    "doi": point.payload.get("doi"),
                    "title": point.payload.get("title", ""),
                    # Use page_content (chunk) if available, else abstract
                    "text": point.payload.get("page_content") or point.payload.get("abstract", ""),
                    "abstract": point.payload.get("abstract", ""),
                    "full_text": point.payload.get("full_text", ""),
                    "has_full_text": point.payload.get("has_full_text", False),
                    "section_title": point.payload.get("section_title", "abstract"),
                    "section_type": point.payload.get("section_type", "body"),
                    "chunk_id": point.payload.get("chunk_id"),
                    "chunk_index": point.payload.get("chunk_index"),
                    "journal": point.payload.get("journal", ""),
                    "venue": point.payload.get("journal", ""),
                    "nlm_unique_id": point.payload.get("nlm_unique_id"),
                    "year": point.payload.get("year"),
                    "authors": [],
                    "article_type": point.payload.get("article_type", ""),
                    "publication_type": self._normalize_publication_type_list(
                        point.payload.get("publication_type")
                    ),
                    "score": point.score,
                    "stype": "vector_search",
                    **self._extract_evidence_metadata(point.payload),
                    # Government affiliation (merged from gov pipeline)
                    "is_gov_affiliated": point.payload.get("is_gov_affiliated", False),
                    "gov_agencies": point.payload.get("gov_agencies", []),
                }

            passages.append(passage)

        logger.info(f"Retrieved {len(passages)} passages for query: {query[:50]}...")
        return passages
    
    def _hybrid_search(
        self,
        query: str,
        search_filter: Optional[Filter],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        precomputed_sparse_vector: Optional[SparseVector] = None,
        filter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both dense and sparse vectors.
        Uses RRF (Reciprocal Rank Fusion) to combine results.
        
        Args:
            query: Search query
            search_filter: Qdrant filter
            dense_weight: Weight for dense vector scores
            sparse_weight: Weight for sparse vector scores
            precomputed_sparse_vector: Optional pre-computed BM25 sparse vector (for batch optimization)
            
        Returns:
            List of passages with combined scores
        """
        # Use pre-computed sparse vector if provided, otherwise generate.
        sparse_vector = precomputed_sparse_vector or self._build_sparse_query_vector(query)
        if sparse_vector is None:
            logger.warning("Sparse encoding unavailable, falling back to dense-only")
            return self.retrieve_passages(query, use_hybrid=False, **(filter_kwargs or {}))
        if not sparse_vector.indices:
            logger.info("Sparse vector empty, falling back to dense-only")
            return self.retrieve_passages(query, use_hybrid=False, **(filter_kwargs or {}))
        
        # Perform dense search
        dense_results = None
        query_embedding = self._embed_query(query)
        if query_embedding is not None:
            try:
                dense_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    using="dense",
                    limit=self.n_retrieval * 2,  # Get more for fusion
                    score_threshold=self.score_threshold,
                    query_filter=search_filter,
                    with_payload=True,
                    search_params=SearchParams(
                        quantization=QuantizationSearchParams(
                            rescore=True,
                            oversampling=2.0
                        )
                    )
                )
            except Exception as e:
                logger.warning(f"Dense search failed: {e}")
        
        # Perform sparse search
        sparse_results = None
        try:
            sparse_results = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_vector,
                using="sparse",
                limit=self.n_retrieval * 2,  # Get more for fusion
                query_filter=search_filter,
                with_payload=True
            )
        except Exception as e:
            logger.warning(f"Sparse search failed: {e}")
        
        # Combine results using RRF (Reciprocal Rank Fusion)
        if not dense_results and not sparse_results:
            logger.warning("Both searches failed, returning empty results")
            return []
        
        # Build score maps keyed by chunk/point identity
        dense_scores = {}
        sparse_scores = {}
        
        if dense_results:
            for rank, point in enumerate(dense_results.points, 1):
                key = self._rank_key_from_point(point)
                # RRF: 1 / (k + rank), where k=60 is typical
                rrf_score = 1.0 / (60.0 + rank)
                dense_scores[key] = rrf_score * dense_weight
        
        if sparse_results:
            for rank, point in enumerate(sparse_results.points, 1):
                key = self._rank_key_from_point(point)
                rrf_score = 1.0 / (60.0 + rank)
                sparse_scores[key] = rrf_score * sparse_weight
        
        # Combine scores
        combined_scores = {}
        all_keys = set(dense_scores.keys()) | set(sparse_scores.keys())

        for key in all_keys:
            combined_scores[key] = (
                dense_scores.get(key, 0) + sparse_scores.get(key, 0)
            )
        
        # Get all points from both results
        all_points = {}
        if dense_results:
            for point in dense_results.points:
                key = self._rank_key_from_point(point)
                all_points[key] = point
        if sparse_results:
            for point in sparse_results.points:
                key = self._rank_key_from_point(point)
                if key not in all_points:
                    all_points[key] = point

        # Sort by combined score
        sorted_keys = sorted(
            all_keys,
            key=lambda p: combined_scores[p],
            reverse=True
        )[:self.n_retrieval]
        
        # Format results
        passages = []
        for key in sorted_keys:
            point = all_points[key]
            
            # Detect DailyMed articles
            source = point.payload.get("source", "")
            article_type = point.payload.get("article_type", "")
            is_dailymed = source == "dailymed" or article_type == "drug_label"
            
            if is_dailymed:
                passage = self._transform_dailymed_payload(point.payload, combined_scores[key])
                passage["dense_score"] = dense_scores.get(key, 0)
                passage["sparse_score"] = sparse_scores.get(key, 0)
                passage["stype"] = "hybrid_search"
            else:
                doc_id = self._doc_id_from_payload(point.payload)
                passage = {
                    "corpus_id": doc_id,
                    "pmcid": point.payload.get("pmcid", doc_id),
                    "pmid": point.payload.get("pmid"),
                    "doi": point.payload.get("doi"),
                    "title": point.payload.get("title", ""),
                    "text": point.payload.get("page_content") or point.payload.get("abstract", ""),
                    "abstract": point.payload.get("abstract", ""),
                    "full_text": point.payload.get("full_text", ""),
                    "has_full_text": point.payload.get("has_full_text", False),
                    "section_title": point.payload.get("section_title", "abstract"),
                    "section_type": point.payload.get("section_type", "body"),
                    "chunk_id": point.payload.get("chunk_id"),
                    "chunk_index": point.payload.get("chunk_index"),
                    "journal": point.payload.get("journal", ""),
                    "venue": point.payload.get("journal", ""),
                    "nlm_unique_id": point.payload.get("nlm_unique_id"),
                    "year": point.payload.get("year"),
                    "authors": [],
                    "article_type": point.payload.get("article_type", ""),
                    "publication_type": self._normalize_publication_type_list(
                        point.payload.get("publication_type")
                    ),
                    "score": combined_scores[key],
                    "dense_score": dense_scores.get(key, 0),
                    "sparse_score": sparse_scores.get(key, 0),
                    "stype": "hybrid_search",
                    **self._extract_evidence_metadata(point.payload),
                }
            
            passages.append(passage)
        
        logger.info(f"✅ Hybrid search successful, retrieved {len(passages)} passages")
        return passages
    
    def batch_hybrid_search(
        self,
        queries: List[str],
        sparse_vectors: Optional[List[SparseVector]] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        **filter_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform batched hybrid search for multiple queries in a SINGLE HTTP call.
        
        This consolidates 6+ HTTP calls into 1 batch request, significantly reducing
        network latency. Uses Qdrant's query_batch_points API with Cloud Inference.
        
        Args:
            queries: List of query strings (will use Cloud Inference for dense embedding)
            sparse_vectors: List of pre-computed BM25 sparse vectors (one per query)
            dense_weight: Weight for dense vector scores (default: 0.7)
            sparse_weight: Weight for sparse vector scores (default: 0.3)
            **filter_kwargs: Filters (year, venue, article_type)
            
        Returns:
            Combined list of unique passages from all queries with RRF fusion scores
        """
        if not queries:
            return []

        if sparse_vectors is None or len(sparse_vectors) != len(queries):
            sparse_vectors = self.build_sparse_query_vectors(queries)

        use_sparse = any(vec.indices for vec in sparse_vectors)
        per_query_requests = 2 if use_sparse else 1
        logger.info(
            "⚡ Batch search: %s queries × %s request(s) = %s in one HTTP call",
            len(queries),
            per_query_requests,
            len(queries) * per_query_requests,
        )
        
        # Build filter
        search_filter = self._build_filter(**filter_kwargs)
        
        # Build batch requests: interleave dense and sparse for each query
        batch_requests = []
        
        # Search params for binary quantization rescore (production-ready)
        search_params = SearchParams(
            quantization=QuantizationSearchParams(
                rescore=QUANTIZATION_RESCORE,
                oversampling=QUANTIZATION_OVERSAMPLING
            )
        )
        
        for i, query_text in enumerate(queries):
            # Dense query using configured embedding provider
            dense_query = self._embed_query(query_text)
            if dense_query is None:
                logger.warning("Skipping query without dense encoder available: %s", query_text[:80])
                continue

            dense_request = QueryRequest(
                query=dense_query,
                using="dense",
                filter=search_filter,
                limit=self.n_retrieval * 2,  # Get more for fusion
                score_threshold=self.score_threshold,
                with_payload=True,
                params=search_params  # Enable quantization rescore
            )
            batch_requests.append(dense_request)
            
            if use_sparse:
                # Sparse query using pre-computed BM25 sparse vector.
                sparse_request = QueryRequest(
                    query=sparse_vectors[i],
                    using="sparse",
                    filter=search_filter,
                    limit=self.n_retrieval * 2,
                    with_payload=True
                )
                batch_requests.append(sparse_request)

        if not batch_requests:
            return []
        
        # Execute single batch HTTP call
        try:
            batch_results = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=batch_requests
            )
            logger.info(f"⚡ Batch query returned {len(batch_results)} result sets")
        except Exception as e:
            logger.error(f"Batch hybrid search failed: {e}")
            # Fallback to sequential queries
            logger.info("⚠️ Falling back to sequential hybrid search")
            all_passages = []
            for idx, query in enumerate(queries):
                if use_sparse:
                    passages = self._hybrid_search(
                        query=query,
                        search_filter=search_filter,
                        dense_weight=dense_weight,
                        sparse_weight=sparse_weight,
                        precomputed_sparse_vector=sparse_vectors[idx],
                        filter_kwargs=filter_kwargs,
                    )
                else:
                    passages = self.retrieve_passages(query, use_hybrid=False, **filter_kwargs)
                all_passages.extend(passages)
            return all_passages
        
        # Process results: pair up dense/sparse for each query and apply RRF
        all_passages = []
        all_points = {}  # rank_key -> point (for payload access)
        combined_scores = {}  # rank_key -> RRF score
        
        for i in range(len(queries)):
            dense_idx = i * per_query_requests
            sparse_idx = dense_idx + 1
            
            dense_results = batch_results[dense_idx].points if dense_idx < len(batch_results) else []
            sparse_results = []
            if use_sparse and sparse_idx < len(batch_results):
                sparse_results = batch_results[sparse_idx].points
            
            # Build RRF scores for this query pair
            for rank, point in enumerate(dense_results, 1):
                key = self._rank_key_from_point(point)
                rrf_score = (1.0 / (60.0 + rank)) * dense_weight
                combined_scores[key] = combined_scores.get(key, 0) + rrf_score
                if key not in all_points:
                    all_points[key] = point

            for rank, point in enumerate(sparse_results, 1):
                key = self._rank_key_from_point(point)
                rrf_score = (1.0 / (60.0 + rank)) * sparse_weight
                combined_scores[key] = combined_scores.get(key, 0) + rrf_score
                if key not in all_points:
                    all_points[key] = point
        
        # Sort by score and limit total results
        sorted_keys = sorted(
            combined_scores.keys(),
            key=lambda p: combined_scores[p],
            reverse=True
        )[:self.n_retrieval]

        # Format passages
        for key in sorted_keys:
            point = all_points[key]
            source = point.payload.get("source", "")
            article_type = point.payload.get("article_type", "")
            is_dailymed = source == "dailymed" or article_type == "drug_label"

            if is_dailymed:
                passage = self._transform_dailymed_payload(point.payload, combined_scores[key])
                passage["stype"] = "batch_hybrid_search"
            else:
                doc_id = self._doc_id_from_payload(point.payload)
                passage = {
                    "corpus_id": doc_id,
                    "pmcid": point.payload.get("pmcid", doc_id),
                    "pmid": point.payload.get("pmid"),
                    "doi": point.payload.get("doi"),
                    "title": point.payload.get("title", ""),
                    "text": point.payload.get("page_content") or point.payload.get("abstract", ""),
                    "abstract": point.payload.get("abstract", ""),
                    "full_text": point.payload.get("full_text", ""),
                    "has_full_text": point.payload.get("has_full_text", False),
                    "section_title": point.payload.get("section_title", "abstract"),
                    "section_type": point.payload.get("section_type", "body"),
                    "chunk_id": point.payload.get("chunk_id"),
                    "chunk_index": point.payload.get("chunk_index"),
                    "journal": point.payload.get("journal", ""),
                    "venue": point.payload.get("journal", ""),
                    "nlm_unique_id": point.payload.get("nlm_unique_id"),
                    "year": point.payload.get("year"),
                    "authors": [],
                    "article_type": point.payload.get("article_type", ""),
                    "publication_type": self._normalize_publication_type_list(
                        point.payload.get("publication_type")
                    ),
                    "score": combined_scores[key],
                    "stype": "batch_hybrid_search",
                    **self._extract_evidence_metadata(point.payload),
                }
            all_passages.append(passage)
        
        logger.info(f"✅ Batch hybrid search complete: {len(all_passages)} unique passages")
        return all_passages

    def retrieve_additional_papers(
        self,
        query: str,
        **filter_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve additional papers (keyword-style fallback).

        Uses vector search with broader parameters to supplement main retrieval.
        """
        try:
            search_filter = self._build_filter(**filter_kwargs)
            query_embedding = self._embed_query(query, use_instruction=False)
            if query_embedding is None:
                logger.error("No embedding method available for additional search")
                return []

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using="dense",
                limit=self.n_keyword_search,
                score_threshold=max(self.score_threshold - 0.1, 0.2),  # Lower threshold
                query_filter=search_filter,
                with_payload=True,
                search_params=SearchParams(
                    quantization=QuantizationSearchParams(
                        rescore=QUANTIZATION_RESCORE,
                        oversampling=QUANTIZATION_OVERSAMPLING
                    )
                )
            )

            papers = []
            for point in results.points:
                # Detect DailyMed articles
                source = point.payload.get("source", "")
                article_type = point.payload.get("article_type", "")
                is_dailymed = source == "dailymed" or article_type == "drug_label"
                
                if is_dailymed:
                    paper = self._transform_dailymed_payload(point.payload, point.score)
                    paper["stype"] = "keyword_search"
                else:
                    doc_id = self._doc_id_from_payload(point.payload)
                    paper = {
                        "corpus_id": doc_id,
                        "pmcid": point.payload.get("pmcid", doc_id),
                        "pmid": point.payload.get("pmid"),
                        "doi": point.payload.get("doi"),
                        "title": point.payload.get("title", ""),
                        "text": point.payload.get("page_content") or point.payload.get("abstract", ""),
                        "abstract": point.payload.get("abstract", ""),
                        "full_text": point.payload.get("full_text", ""),
                        "has_full_text": point.payload.get("has_full_text", False),
                        "section_title": point.payload.get("section_title", "abstract"),
                        "section_type": point.payload.get("section_type", "body"),
                        "chunk_id": point.payload.get("chunk_id"),
                        "chunk_index": point.payload.get("chunk_index"),
                        "journal": point.payload.get("journal", ""),
                        "venue": point.payload.get("journal", ""),
                        "nlm_unique_id": point.payload.get("nlm_unique_id"),
                        "year": point.payload.get("year"),
                        "authors": [],
                        "article_type": point.payload.get("article_type", ""),
                        "publication_type": self._normalize_publication_type_list(
                            point.payload.get("publication_type")
                        ),
                        "score": point.score,
                        "stype": "keyword_search",
                        **self._extract_evidence_metadata(point.payload),
                    }
                papers.append(paper)

            return papers

        except Exception as e:
            logger.error(f"Additional papers retrieval error: {e}")
            return []
    
    def _drug_name_matches(self, search_term: str, label_name: str, additional_text: str = "") -> bool:
        """
        Check if a DailyMed drug label matches the searched drug name.
        
        Uses case-insensitive substring matching and handles common variations:
        - Brand names: "Humira" matches "HUMIRA INJECTION"
        - Generic names: "adalimumab" matches "Adalimumab Injection Solution"
        - Partial matches: "tofacitinib" matches "Xeljanz (tofacitinib)"
        - Generic in highlights: "voclosporin" matches LUPKYNIS via highlights text
        
        Args:
            search_term: The drug name to search for
            label_name: The drug label/title from DailyMed
            additional_text: Additional text to search (e.g., highlights section)
        
        Returns True if the search term appears in the label name or additional text.
        """
        if not search_term or not label_name:
            return False
        
        search_lower = search_term.lower().strip()
        label_lower = label_name.lower().strip()
        
        # Direct substring match in label name
        if search_lower in label_lower:
            return True
        
        # Check if any word in the label matches the search term
        label_words = label_lower.replace('-', ' ').replace('(', ' ').replace(')', ' ').split()
        for word in label_words:
            if word == search_lower:
                return True
            # Handle suffix variations (e.g., "adalimumab" vs "adalimumab-atto")
            if word.startswith(search_lower) and len(word) - len(search_lower) <= 5:
                return True
        
        # Check additional text (e.g., highlights section for generic name)
        # This catches cases like searching for "voclosporin" when label is "LUPKYNIS"
        # but highlights contains "LUPKYNIS (voclosporin) capsules"
        if additional_text:
            additional_lower = additional_text.lower()
            # Look for the drug name in the first 500 chars of highlights (prescribing info header)
            if search_lower in additional_lower[:500]:
                return True
        
        return False

    def search_dailymed_by_drug(self, drug_names: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search DailyMed drug labels using optimized lookup with lazy cache + BM25.
        
        Strategy:
        1. Check lazy in-memory cache (O(1) instant for repeated queries)
        2. Check pre-built lookup table (O(1) if JSON file exists)
        3. Fallback to BM25 sparse vector search (better than MatchText)
        4. Cache results in memory for future queries
        5. Deduplicate by set_id - handles brand/generic overlap
        """
        if not drug_names:
            return []
        
        all_results = []
        seen_set_ids = set()  # Dedup by set_id (handles brand/generic overlap)
        cache_hits = 0
        lookup_hits = 0
        bm25_hits = 0
        
        for drug_name in drug_names[:5]:  # Allow up to 5 drugs
            drug_lower = drug_name.lower().strip()
            
            # Step 1: Check lazy cache first (fastest, in-memory)
            if drug_lower in self._drug_lookup_cache:
                set_id = self._drug_lookup_cache[drug_lower]
                if set_id in seen_set_ids:
                    logger.info(f"   💊 '{drug_name}' → already have set_id (cached)")
                    continue
                
                result = self._fetch_dailymed_by_set_id(set_id)
                if result:
                    seen_set_ids.add(set_id)
                    all_results.append(result)
                    cache_hits += 1
                    continue
            
            # Step 2: Try pre-built lookup table (if JSON file exists)
            if drug_lower in self.drug_setid_lookup:
                lookup_data = self.drug_setid_lookup[drug_lower]
                set_ids = lookup_data.get("set_ids", [])
                
                if set_ids:
                    first_set_id = set_ids[0]
                    
                    if first_set_id in seen_set_ids:
                        logger.info(f"   💊 '{drug_name}' → already have set_id (brand/generic overlap)")
                        continue
                    
                    # Add to lazy cache for future queries
                    self._drug_lookup_cache[drug_lower] = first_set_id
                    
                    result = self._fetch_dailymed_by_set_id(first_set_id)
                    if result:
                        seen_set_ids.add(first_set_id)
                        all_results.append(result)
                        lookup_hits += 1
                    continue
            
            # Step 3: Fallback to BM25 sparse vector search
            logger.info(f"   💊 Cache miss for '{drug_name}', using BM25 sparse search")
            bm25_results = self._bm25_search_dailymed(drug_name, limit=3)
            
            for result in bm25_results:
                set_id = result.get("set_id")
                if set_id and set_id not in seen_set_ids:
                    seen_set_ids.add(set_id)
                    all_results.append(result)
                    # Cache this result for future queries
                    self._drug_lookup_cache[drug_lower] = set_id
                    bm25_hits += 1
                    break  # Only take first (best BM25 match)
        
        if all_results:
            logger.info(f"   ✅ Found {len(all_results)} DailyMed articles "
                       f"(cache: {cache_hits}, lookup: {lookup_hits}, BM25: {bm25_hits})")
        
        return all_results
    
    def _fetch_dailymed_by_set_id(self, set_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and aggregate a full DailyMed drug label from Qdrant by set_id.

        DailyMed ingestion stores section chunks per point, so this method scrolls
        all points for the set_id and reconstructs section-level fields.
        """
        try:
            filter_by_id = Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value="dailymed")),
                    FieldCondition(key="set_id", match=MatchValue(value=set_id)),
                ]
            )

            all_payloads: List[Dict[str, Any]] = []
            next_offset = None

            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_by_id,
                    offset=next_offset,
                    limit=256,
                    with_payload=True,
                )

                if not results:
                    break

                for point in results:
                    if point.payload:
                        all_payloads.append(point.payload)

                if next_offset is None:
                    break

            if all_payloads:
                aggregated_payload = self._aggregate_dailymed_payloads(all_payloads)
                passage = self._transform_dailymed_payload(aggregated_payload, 1.0)
                passage["stype"] = "dailymed_lookup"
                passage["retrieved_chunks"] = len(all_payloads)
                return passage
                
        except Exception as e:
            logger.warning(f"   Qdrant fetch by set_id failed: {e}")
        
        return None
    
    def _bm25_search_dailymed(self, drug_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback: BM25 sparse vector search for drug name matching.
        
        Uses the BM25 sparse encoder to perform better keyword matching than
        simple MatchText, handling term frequency and partial matches.
        """
        if self.bm25_sparse_encoder is None:
            logger.warning("   BM25 encoder not available, cannot search DailyMed")
            return []
        
        try:
            # Build sparse query vector using BM25
            sparse_query = self.bm25_sparse_encoder.encode_query(drug_name)
            
            # Skip if query resulted in empty sparse vector
            if not sparse_query.indices:
                logger.info(f"   Empty BM25 vector for '{drug_name}', using fallback")
                return self._matchtext_search_dailymed(drug_name, limit)
            
            # Filter to DailyMed source only
            source_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="dailymed"))]
            )
            
            # Query using sparse vector
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=sparse_query,
                using="sparse",
                query_filter=source_filter,
                limit=max(limit * 8, 40),
                with_payload=True
            )
            
            passages = []
            seen_set_ids = set()
            for point in search_results.points:
                payload = point.payload or {}
                set_id = payload.get("set_id")
                if not set_id or set_id in seen_set_ids:
                    continue

                full_label = self._fetch_dailymed_by_set_id(set_id)
                if not full_label:
                    continue

                full_label["score"] = point.score
                full_label["stype"] = "dailymed_bm25"
                passages.append(full_label)
                seen_set_ids.add(set_id)

                if len(passages) >= limit:
                    break
            
            if passages:
                logger.info(f"   BM25 found {len(passages)} results for '{drug_name}' (top score: {passages[0]['score']:.3f})")
            
            return passages
            
        except Exception as e:
            logger.warning(f"   BM25 search failed for '{drug_name}': {e}, trying MatchText fallback")
            return self._matchtext_search_dailymed(drug_name, limit)
    
    def _matchtext_search_dailymed(self, drug_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Last-resort fallback: MatchText search when BM25 fails.
        """
        try:
            keyword_filter = Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value="dailymed")),
                ],
                should=[
                    FieldCondition(key="drug_name", match=MatchText(text=drug_name)),
                    FieldCondition(key="text", match=MatchText(text=drug_name)),
                    FieldCondition(key="page_content", match=MatchText(text=drug_name)),
                    FieldCondition(key="section_title", match=MatchText(text=drug_name)),
                ]
            )
            
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=keyword_filter,
                limit=max(limit * 10, 50),
                with_payload=True
            )
            
            passages = []
            seen_set_ids = set()
            for point in results:
                payload = point.payload or {}
                set_id = payload.get("set_id")
                if not set_id or set_id in seen_set_ids:
                    continue

                full_label = self._fetch_dailymed_by_set_id(set_id)
                if not full_label:
                    continue

                full_label["score"] = 0.95
                full_label["stype"] = "dailymed_matchtext"
                passages.append(full_label)
                seen_set_ids.add(set_id)

                if len(passages) >= limit:
                    break
            
            if passages:
                logger.info(f"   MatchText fallback found {len(passages)} results for '{drug_name}'")
            return passages
            
        except Exception as e:
            logger.warning(f"   MatchText fallback failed for '{drug_name}': {e}")
            return []

    
    def search_bulk(
        self,
        queries: List[str],
        top_k_per_query: int = 50,
        total_limit: int = 200,
        **filter_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Bulk search with multiple queries.
        
        Args:
            queries: List of query strings
            top_k_per_query: Results per query
            total_limit: Maximum total results
            **filter_kwargs: Filters to apply
            
        Returns:
            Deduplicated list of passages
        """
        seen_ids = set()
        all_passages = []
        
        for query in queries:
            if len(all_passages) >= total_limit:
                break
            
            passages = self.retrieve_passages(
                query,
                **filter_kwargs
            )
            
            for passage in passages:
                pmcid = passage.get("pmcid")
                if pmcid and pmcid not in seen_ids:
                    seen_ids.add(pmcid)
                    all_passages.append(passage)
                    
                    if len(all_passages) >= total_limit:
                        break
        
        logger.info(f"Bulk search returned {len(all_passages)} unique passages")
        return all_passages
    
    def apply_hybrid_scoring(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Apply hybrid scoring (dense + sparse/lexical) to retrieved passages.
        
        Option B: BM25-style keyword matching on retrieved results.
        Combines dense embedding scores with keyword matching scores.
        
        Args:
            query: Search query
            passages: List of retrieved passages with 'score' field
            dense_weight: Weight for dense embedding scores (default: 0.7)
            sparse_weight: Weight for keyword matching scores (default: 0.3)
            
        Returns:
            List of passages with combined scores
        """
        if not passages:
            return passages
        
        logger.info(f"Applying hybrid scoring to {len(passages)} passages")
        
        # Calculate keyword matching scores (BM25-style)
        keyword_scores = self._calculate_keyword_scores(query, passages)
        
        # Normalize scores to 0-1 range
        max_dense = max((p.get("score", 0) for p in passages), default=1.0)
        max_keyword = max(keyword_scores, default=1.0) if keyword_scores else 1.0
        
        # Combine scores
        for i, passage in enumerate(passages):
            dense_score = passage.get("score", 0) / max_dense if max_dense > 0 else 0
            keyword_score = keyword_scores[i] / max_keyword if max_keyword > 0 else 0
            
            # Combined score
            combined_score = (dense_weight * dense_score) + (sparse_weight * keyword_score)
            
            passage["dense_score"] = dense_score
            passage["keyword_score"] = keyword_score
            passage["hybrid_score"] = combined_score
            # Update main score to hybrid score for sorting
            passage["score"] = combined_score
        
        # Sort by hybrid score
        passages.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        logger.info(f"Hybrid scoring complete (top scores: {[round(p.get('hybrid_score', 0), 3) for p in passages[:5]]})")
        return passages
    
    def _calculate_keyword_scores(self, query: str, passages: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate BM25-style keyword matching scores.
        
        Args:
            query: Search query
            passages: List of passages
            
        Returns:
            List of keyword scores
        """
        import re
        from collections import Counter
        
        # Tokenize query
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        scores = []
        for passage in passages:
            # Combine title, abstract, and text (abstracts only - no full text)
            title = str(passage.get("title", "")).lower()
            abstract = str(passage.get("abstract", "")).lower()
            text = str(passage.get("text", "")).lower()
            
            doc_text = f"{title} {abstract} {text}".lower()
            
            # Tokenize document
            doc_terms = Counter(re.findall(r'\b\w+\b', doc_text))
            
            # Calculate simple TF-IDF-like score
            score = 0.0
            for term in query_terms:
                # Term frequency in document
                tf = doc_terms.get(term, 0)
                
                # Boost title matches
                if term in title:
                    tf += 2
                
                # Boost abstract matches
                if term in abstract:
                    tf += 1
                
                # Simple scoring: log(1 + tf)
                if tf > 0:
                    score += (1 + tf) * (1 + len(term))  # Longer terms get more weight
            
            scores.append(score)
        
        return scores
    
    def _aggregate_dailymed_payloads(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate chunk-level DailyMed payloads into one label-level payload.
        """
        if not payloads:
            return {}

        section_keys = [
            "highlights",
            "boxed_warning",
            "indications",
            "dosage",
            "contraindications",
            "warnings",
            "adverse_reactions",
            "interactions",
            "use_in_specific_populations",
            "clinical_pharmacology",
            "clinical_studies",
        ]

        section_buckets: Dict[str, List[Any]] = {k: [] for k in section_keys}
        set_id = ""
        drug_name = ""
        manufacturer = ""
        title = ""
        evidence_meta = {}

        for payload in payloads:
            if not set_id:
                set_id = str(payload.get("set_id", "")).strip()
            if not drug_name:
                drug_name = str(payload.get("drug_name") or payload.get("title") or "").strip()
            if not manufacturer:
                manufacturer = str(payload.get("manufacturer", "")).strip()
            if not title:
                title = str(payload.get("title", "")).strip()

            for key in ("evidence_grade", "evidence_level", "evidence_term", "evidence_source"):
                if key not in evidence_meta and payload.get(key) is not None:
                    evidence_meta[key] = payload.get(key)

            section_type = str(payload.get("section_type", "")).strip().lower()
            table_type = str(payload.get("table_type", "")).strip().lower()
            section_key = table_type if section_type == "table" and table_type else section_type
            chunk_text = str(payload.get("text") or payload.get("page_content") or "").strip()
            chunk_id = str(payload.get("chunk_id") or "")
            try:
                chunk_index = int(payload.get("chunk_index", 0))
            except Exception:
                chunk_index = 0

            if section_key in section_buckets and chunk_text:
                section_buckets[section_key].append((chunk_index, chunk_id, chunk_text))

            # Support old/wide payload format too.
            for wide_key in section_keys:
                wide_text = str(payload.get(wide_key, "")).strip()
                if wide_text:
                    section_buckets[wide_key].append((0, f"wide:{wide_key}", wide_text))

        section_texts: Dict[str, str] = {}
        for key, entries in section_buckets.items():
            if not entries:
                continue

            entries.sort(key=lambda item: (item[0], item[1]))
            seen_text = set()
            ordered_parts = []
            for _, _, text in entries:
                norm = text.strip()
                if not norm or norm in seen_text:
                    continue
                seen_text.add(norm)
                ordered_parts.append(norm)

            if ordered_parts:
                section_texts[key] = "\n\n".join(ordered_parts)

        # Boxed warning should be included in highlights block when present.
        boxed_warning = section_texts.get("boxed_warning", "")
        highlights = section_texts.get("highlights", "")
        if boxed_warning:
            if highlights:
                section_texts["highlights"] = f"{boxed_warning}\n\n{highlights}"
            else:
                section_texts["highlights"] = boxed_warning

        merged = {
            "set_id": set_id,
            "drug_name": drug_name or title,
            "title": title or drug_name,
            "manufacturer": manufacturer,
            "source": "dailymed",
            "article_type": "drug_label",
            "dailymed_sections": section_texts,
            **evidence_meta,
        }
        merged.update(section_texts)
        return merged

    def _transform_dailymed_payload(self, payload: Dict[str, Any], score: float) -> Dict[str, Any]:
        """
        Transform DailyMed payload (chunk-level or label-level) into pipeline schema.
        """
        set_id = payload.get("set_id", "")
        drug_name = payload.get("drug_name", payload.get("title", "Unknown Drug"))
        manufacturer = payload.get("manufacturer", "")

        # Build a section map from wide fields or chunk-derived section data.
        section_map = {}
        existing_sections = payload.get("dailymed_sections", {})
        if isinstance(existing_sections, dict):
            for key, value in existing_sections.items():
                text_val = str(value).strip()
                if text_val:
                    section_map[key] = text_val

        for key in (
            "highlights",
            "boxed_warning",
            "indications",
            "dosage",
            "contraindications",
            "warnings",
            "adverse_reactions",
            "interactions",
            "use_in_specific_populations",
            "clinical_pharmacology",
            "clinical_studies",
        ):
            text_val = str(payload.get(key, "")).strip()
            if text_val:
                section_map[key] = text_val

        section_type = str(payload.get("section_type", "")).strip().lower()
        table_type = str(payload.get("table_type", "")).strip().lower()
        section_key = table_type if section_type == "table" and table_type else section_type
        chunk_text = str(payload.get("text") or payload.get("page_content") or "").strip()
        if section_key and chunk_text and section_key not in section_map:
            section_map[section_key] = chunk_text

        if section_map.get("boxed_warning"):
            if section_map.get("highlights"):
                section_map["highlights"] = f"{section_map['boxed_warning']}\n\n{section_map['highlights']}"
            else:
                section_map["highlights"] = section_map["boxed_warning"]

        highlights = section_map.get("highlights", "")
        indications = section_map.get("indications", "")
        dosage = section_map.get("dosage", "")
        contraindications = section_map.get("contraindications", "")
        warnings = section_map.get("warnings", "")
        adverse_reactions = section_map.get("adverse_reactions", "")
        clinical_pharmacology = section_map.get("clinical_pharmacology", "")
        clinical_studies = section_map.get("clinical_studies", "")

        # Build comprehensive content from available sections.
        content_parts = []
        
        if drug_name:
            content_parts.append(f"# {drug_name}")
        if manufacturer:
            content_parts.append(f"Manufacturer: {manufacturer}")
        
        if highlights:
            content_parts.append(f"\n## Highlights of Prescribing Information\n{highlights}")
        
        if indications:
            content_parts.append(f"\n## Indications and Usage\n{indications}")
        
        if dosage:
            content_parts.append(f"\n## Dosage and Administration\n{dosage}")
        
        if contraindications:
            content_parts.append(f"\n## Contraindications\n{contraindications}")
        
        if warnings:
            content_parts.append(f"\n## Warnings and Precautions\n{warnings}")
        
        if adverse_reactions:
            content_parts.append(f"\n## Adverse Reactions\n{adverse_reactions}")
        
        if clinical_pharmacology:
            content_parts.append(f"\n## Clinical Pharmacology\n{clinical_pharmacology}")
        
        if clinical_studies:
            content_parts.append(f"\n## Clinical Studies\n{clinical_studies}")

        extra_sections = [
            ("interactions", "Drug Interactions"),
            ("use_in_specific_populations", "Use in Specific Populations"),
        ]
        for key, title in extra_sections:
            text_val = section_map.get(key, "")
            if text_val:
                content_parts.append(f"\n## {title}\n{text_val}")
        
        final_content = "\n".join(content_parts)
        
        return {
            "corpus_id": f"dailymed_{set_id}" if set_id else "",
            "pmcid": f"dailymed_{set_id}" if set_id else "",
            "pmid": None,
            "title": drug_name or payload.get("title", ""),
            "text": final_content[:30000] if final_content else "",  # Increased for 8 sections
            "abstract": final_content,
            "section_title": "drug_label",
            "journal": "DailyMed",
            "venue": "FDA Drug Label",
            "year": datetime.now().year,
            "authors": [{"name": manufacturer}] if manufacturer else [],
            "article_type": "drug_label",
            "publication_type": self._normalize_publication_type_list(payload.get("publication_type")),
            "score": score,
            "stype": "vector_search",
            "evidence_grade": payload.get("evidence_grade"),
            "evidence_level": payload.get("evidence_level"),
            "evidence_term": payload.get("evidence_term"),
            "evidence_source": payload.get("evidence_source"),
            "source": "dailymed",
            "set_id": set_id,
            "drug_name": drug_name,
            "dailymed_sections": section_map,
            "highlights": highlights,
            "indications": indications,
            "dosage": dosage,
            "contraindications": contraindications,
            "warnings": warnings,
            "adverse_reactions": adverse_reactions,
            "clinical_pharmacology": clinical_pharmacology,
            "clinical_studies": clinical_studies,
        }

    



if __name__ == "__main__":
    # Test retriever
    print("🧪 Testing Qdrant Retriever")
    print("=" * 60)
    
    retriever = QdrantRetriever(n_retrieval=10)
    
    # Test basic search
    query = "What are the treatments for type 2 diabetes?"
    print(f"\n📝 Query: {query}")
    
    results = retriever.retrieve_passages(query)
    print(f"✅ Retrieved {len(results)} passages")
    
    for i, r in enumerate(results[:3], 1):
        print(f"\n[{i}] {r['title'][:60]}...")
        print(f"    Score: {r['score']:.3f} | Year: {r.get('year')} | Journal: {r.get('journal')}")
    
    # Test with year filter
    print("\n\n📅 Testing with year filter (2022-2025):")
    filtered_results = retriever.retrieve_passages(query, year="2022-2025")
    print(f"✅ Retrieved {len(filtered_results)} passages with year filter")
