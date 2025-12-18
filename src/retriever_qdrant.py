"""
Qdrant Retriever Adapter for ScholarQA-style Pipeline.

Implements retrieval from Qdrant Vector Database with:
- Semantic vector search using mxbai-embed-large-v1 embeddings (Cloud Inference)
- Metadata filtering (year, venue, article_type, country)
- Passage-level results with section information
- Bulk search with multiple query embeddings
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, SparseVector, IsNullCondition
from qdrant_client.http.models import Document

from .config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, SCORE_THRESHOLD, EMBEDDING_MODEL, QDRANT_CLOUD_INFERENCE

logger = logging.getLogger(__name__)

# Local embedding fallback
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available, local embeddings disabled")

# SPLADE encoder for sparse vectors
try:
    from .splade_encoder import get_splade_encoder
    SPLADE_AVAILABLE = True
except ImportError:
    SPLADE_AVAILABLE = False
    logger.warning("SPLADE encoder not available, hybrid search disabled")


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
        n_retrieval: int = 100,
        n_keyword_search: int = 20,
        score_threshold: float = SCORE_THRESHOLD
    ):
        """
        Initialize Qdrant retriever with Cloud Inference and local fallback.

        Args:
            n_retrieval: Number of passages to retrieve per query
            n_keyword_search: Number of additional papers from keyword search
            score_threshold: Minimum similarity score
        """
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=120,
            cloud_inference=QDRANT_CLOUD_INFERENCE,
        )
        self.collection_name = COLLECTION_NAME
        self.embedding_model = EMBEDDING_MODEL
        self.n_retrieval = n_retrieval
        self.n_keyword_search = n_keyword_search
        self.score_threshold = score_threshold

        # Initialize local embedding model as fallback
        self.local_encoder = None
        if LOCAL_EMBEDDINGS_AVAILABLE and not QDRANT_CLOUD_INFERENCE:
            try:
                logger.info("Loading local embedding model...")
                self.local_encoder = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
                logger.info("✅ Local embedding model loaded")
            except Exception as e:
                logger.warning(f"Failed to load local embedding model: {e}")

        # Initialize SPLADE encoder if available and hybrid search is enabled
        self.splade_encoder = None
        # SPLADE disabled - not using sparse vectors in current collection
        # if SPLADE_AVAILABLE:
        #     try:
        #         self.splade_encoder = get_splade_encoder()
        #         logger.info("✅ SPLADE encoder initialized for hybrid search")
        #     except Exception as e:
        #         logger.warning(f"SPLADE encoder not available: {e}")
        
        logger.info(f"Initialized QdrantRetriever: {COLLECTION_NAME} (Cloud Inference: {QDRANT_CLOUD_INFERENCE}, Local Fallback: {self.local_encoder is not None}, SPLADE: {self.splade_encoder is not None})")
    
    def _build_filter(self, **kwargs) -> Optional[Filter]:
        """
        Build Qdrant filter from search parameters.
        
        Supported filters:
        - year: "2020-2025" or "2020-" or "-2025"
        - venue: "NEJM,Lancet" (comma-separated)
        - field_of_study: "Medicine"
        - article_type: "systematic-review,rct"
        
        Note: DailyMed articles (source="dailymed") are always included regardless of 
        year filter, as they represent current FDA-approved drug information.
        """
        conditions = []
        year_filter = None
        
        # Year filter - but we'll make it OR with DailyMed source
        if "year" in kwargs and kwargs["year"]:
            year_str = kwargs["year"]
            parts = year_str.split("-")
            
            range_params = {}
            if len(parts) >= 1 and parts[0]:
                range_params["gte"] = int(parts[0])
            if len(parts) >= 2 and parts[1]:
                range_params["lte"] = int(parts[1])
            
            if range_params:
                # Create year filter but make it conditional (OR with DailyMed)
                year_filter = FieldCondition(key="year", range=Range(**range_params))
        
        # Venue filter (OR logic across venues)
        if "venue" in kwargs and kwargs["venue"]:
            venues = [v.strip() for v in kwargs["venue"].split(",")]
            # Use should (OR) for multiple venues
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
        
        # Build filter with special handling for year + DailyMed
        if year_filter:
            # Include articles matching year range OR DailyMed articles (which have no year)
            # Using should: (year in range) OR (source = "dailymed") OR (article_type = "drug_label")
            year_or_dailymed = Filter(
                should=[
                    Filter(must=[year_filter]),  # Articles with year in range
                    Filter(must=[FieldCondition(key="source", match=MatchValue(value="dailymed"))]),  # DailyMed articles
                    Filter(must=[FieldCondition(key="article_type", match=MatchValue(value="drug_label"))]),  # Drug labels
                ]
            )
            conditions.append(year_or_dailymed)
        
        if conditions:
            return Filter(must=conditions)
        return None
    
    def retrieve_passages(
        self,
        query: str,
        use_hybrid: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        **filter_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant passages using vector search with optional hybrid search (dense + sparse).

        Args:
            query: Search query (embedded via Cloud Inference or locally)
            use_hybrid: Whether to use hybrid search (dense + SPLADE sparse)
            dense_weight: Weight for dense vector scores (default: 0.7)
            sparse_weight: Weight for sparse vector scores (default: 0.3)
            **filter_kwargs: Filters (year, venue, article_type, country)

        Returns:
            List of passage dictionaries with metadata
        """
        # Build filter
        search_filter = self._build_filter(**filter_kwargs)

        # Try hybrid search if SPLADE is available
        if use_hybrid and self.splade_encoder:
            return self._hybrid_search(
                query=query,
                search_filter=search_filter,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )

        # Fall back to dense-only search
        # Try cloud inference first if enabled
        use_cloud_inference = QDRANT_CLOUD_INFERENCE
        results = None

        if use_cloud_inference:
            try:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=Document(
                        text=query,
                        model=self.embedding_model,
                    ),
                    limit=self.n_retrieval,
                    score_threshold=self.score_threshold,
                    query_filter=search_filter,
                    with_payload=True
                )
                logger.info(f"✅ Cloud inference successful, retrieved {len(results.points)} passages")
            except Exception as e:
                logger.warning(f"Cloud inference failed: {e}, falling back to local embeddings")
                use_cloud_inference = False  # Fall back to local embeddings for this query only

        # Use local embeddings if cloud inference is disabled or failed
        if not use_cloud_inference:
            if self.local_encoder is None:
                logger.error("No embedding method available (cloud inference disabled and local encoder not loaded)")
                return []

            try:
                # Generate embedding locally
                query_embedding = self.local_encoder.encode(query, normalize_embeddings=True)

                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding.tolist(),
                    limit=self.n_retrieval,
                    score_threshold=self.score_threshold,
                    query_filter=search_filter,
                    with_payload=True
                )
                logger.info(f"✅ Local embeddings successful, retrieved {len(results.points)} passages")
            except Exception as e:
                logger.error(f"Local embedding search failed: {e}")
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
                # Standard PMC schema
                passage = {
                    "corpus_id": point.payload.get("pmcid", ""),
                    "pmcid": point.payload.get("pmcid", ""),
                    "pmid": point.payload.get("pmid"),
                    "title": point.payload.get("title", ""),
                    "text": point.payload.get("abstract", ""),
                    "abstract": point.payload.get("abstract", ""),
                    "full_text": point.payload.get("full_text", ""),
                    "section_title": "abstract",  # Primary section
                    "journal": point.payload.get("journal", ""),
                    "venue": point.payload.get("journal", ""),
                    "year": point.payload.get("year"),
                    "authors": self._parse_authors(point.payload.get("authors")),
                    "article_type": point.payload.get("article_type", ""),
                    "score": point.score,
                    "stype": "vector_search"
                }

                # Add full text as additional passage if available
                if passage["full_text"] and len(passage["full_text"]) > 500:
                    passage["text"] = passage["full_text"][:2000]  # Truncate for processing
                    passage["section_title"] = "full_text"

            passages.append(passage)

        logger.info(f"Retrieved {len(passages)} passages for query: {query[:50]}...")
        return passages
    
    def _hybrid_search(
        self,
        query: str,
        search_filter: Optional[Filter],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using both dense and sparse vectors.
        Uses RRF (Reciprocal Rank Fusion) to combine results.
        
        Args:
            query: Search query
            search_filter: Qdrant filter
            dense_weight: Weight for dense vector scores
            sparse_weight: Weight for sparse vector scores
            
        Returns:
            List of passages with combined scores
        """
        # Generate SPLADE sparse vector for query
        sparse_vecs = self.splade_encoder.encode([query])
        if not sparse_vecs or not sparse_vecs[0]:
            logger.warning("SPLADE encoding failed, falling back to dense-only")
            return self.retrieve_passages(query, use_hybrid=False, **{})
        
        sparse_dict = sparse_vecs[0]
        sparse_vector = SparseVector(
            indices=list(sparse_dict.keys()),
            values=list(sparse_dict.values())
        )
        
        # Perform dense search
        dense_results = None
        try:
            if QDRANT_CLOUD_INFERENCE:
                dense_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=Document(
                        text=query,
                        model=self.embedding_model,
                    ),
                    limit=self.n_retrieval * 2,  # Get more for fusion
                    query_filter=search_filter,
                    with_payload=True
                )
            else:
                if self.local_encoder:
                    query_embedding = self.local_encoder.encode(query, normalize_embeddings=True)
                    dense_results = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_embedding.tolist(),
                        limit=self.n_retrieval * 2,
                        query_filter=search_filter,
                        with_payload=True
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
        
        # Build score maps
        dense_scores = {}
        sparse_scores = {}
        
        if dense_results:
            for rank, point in enumerate(dense_results.points, 1):
                pmcid = point.payload.get("pmcid") or point.id
                # RRF: 1 / (k + rank), where k=60 is typical
                rrf_score = 1.0 / (60.0 + rank)
                dense_scores[pmcid] = rrf_score * dense_weight
        
        if sparse_results:
            for rank, point in enumerate(sparse_results.points, 1):
                pmcid = point.payload.get("pmcid") or point.id
                rrf_score = 1.0 / (60.0 + rank)
                sparse_scores[pmcid] = rrf_score * sparse_weight
        
        # Combine scores
        combined_scores = {}
        all_pmcids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        for pmcid in all_pmcids:
            combined_scores[pmcid] = (
                dense_scores.get(pmcid, 0) + sparse_scores.get(pmcid, 0)
            )
        
        # Get all points from both results
        all_points = {}
        if dense_results:
            for point in dense_results.points:
                pmcid = point.payload.get("pmcid") or point.id
                all_points[pmcid] = point
        if sparse_results:
            for point in sparse_results.points:
                pmcid = point.payload.get("pmcid") or point.id
                if pmcid not in all_points:
                    all_points[pmcid] = point
        
        # Sort by combined score
        sorted_pmcids = sorted(
            all_pmcids,
            key=lambda p: combined_scores[p],
            reverse=True
        )[:self.n_retrieval]
        
        # Format results
        passages = []
        for pmcid in sorted_pmcids:
            point = all_points[pmcid]
            
            # Detect DailyMed articles
            source = point.payload.get("source", "")
            article_type = point.payload.get("article_type", "")
            is_dailymed = source == "dailymed" or article_type == "drug_label"
            
            if is_dailymed:
                passage = self._transform_dailymed_payload(point.payload, combined_scores[pmcid])
                passage["dense_score"] = dense_scores.get(pmcid, 0)
                passage["sparse_score"] = sparse_scores.get(pmcid, 0)
                passage["stype"] = "hybrid_search"
            else:
                passage = {
                    "corpus_id": point.payload.get("pmcid", ""),
                    "pmcid": point.payload.get("pmcid", ""),
                    "pmid": point.payload.get("pmid"),
                    "title": point.payload.get("title", ""),
                    "text": point.payload.get("abstract", ""),
                    "abstract": point.payload.get("abstract", ""),
                    "full_text": point.payload.get("full_text", ""),
                    "section_title": "abstract",
                    "journal": point.payload.get("journal", ""),
                    "venue": point.payload.get("journal", ""),
                    "year": point.payload.get("year"),
                    "authors": self._parse_authors(point.payload.get("authors")),
                    "article_type": point.payload.get("article_type", ""),
                    "score": combined_scores[pmcid],
                    "dense_score": dense_scores.get(pmcid, 0),
                    "sparse_score": sparse_scores.get(pmcid, 0),
                    "stype": "hybrid_search"
                }
                
                if passage["full_text"] and len(passage["full_text"]) > 500:
                    passage["text"] = passage["full_text"][:2000]
                    passage["section_title"] = "full_text"
            
            passages.append(passage)
        
        logger.info(f"✅ Hybrid search successful, retrieved {len(passages)} passages")
        return passages
    
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

            # Use same embedding logic as main retrieval
            use_cloud_inference = QDRANT_CLOUD_INFERENCE
            results = None

            if use_cloud_inference:
                try:
                    results = self.client.query_points(
                        collection_name=self.collection_name,
                        query=Document(
                            text=query,
                            model=self.embedding_model,
                        ),
                        limit=self.n_keyword_search,
                        score_threshold=max(self.score_threshold - 0.1, 0.2),  # Lower threshold
                        query_filter=search_filter,
                        with_payload=True
                    )
                except Exception as e:
                    logger.warning(f"Cloud inference failed for additional search: {e}, using local embeddings")
                    use_cloud_inference = False  # Fall back to local embeddings for this query only

            # Use local embeddings if cloud inference is disabled or failed
            if not use_cloud_inference:
                if self.local_encoder is None:
                    logger.error("No embedding method available for additional search")
                    return []
                query_embedding = self.local_encoder.encode(query, normalize_embeddings=True)
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding.tolist(),
                    limit=self.n_keyword_search,
                    score_threshold=max(self.score_threshold - 0.1, 0.2),
                    query_filter=search_filter,
                    with_payload=True
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
                    paper = {
                        "corpus_id": point.payload.get("pmcid", ""),
                        "pmcid": point.payload.get("pmcid", ""),
                        "pmid": point.payload.get("pmid"),
                        "title": point.payload.get("title", ""),
                        "text": point.payload.get("abstract", ""),
                        "abstract": point.payload.get("abstract", ""),
                        "full_text": point.payload.get("full_text", ""),
                        "section_title": "abstract",
                        "journal": point.payload.get("journal", ""),
                        "venue": point.payload.get("journal", ""),
                        "year": point.payload.get("year"),
                        "authors": self._parse_authors(point.payload.get("authors")),
                        "article_type": point.payload.get("article_type", ""),
                        "score": point.score,
                        "stype": "keyword_search"
                    }
                papers.append(paper)

            return papers

        except Exception as e:
            logger.error(f"Additional papers retrieval error: {e}")
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
            # Combine title, abstract, and text
            title = str(passage.get("title", "")).lower()
            abstract = str(passage.get("abstract", "")).lower()
            text = str(passage.get("text", "")).lower()
            full_text = str(passage.get("full_text", "")).lower()
            
            doc_text = f"{title} {abstract} {text} {full_text}".lower()
            
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
    
    def _transform_dailymed_payload(self, payload: Dict[str, Any], score: float) -> Dict[str, Any]:
        """
        Transform DailyMed payload to match PMC schema expected by pipeline.
        
        DailyMed fields: set_id, drug_name, title, indications, dosage, 
                         contraindications, warnings, adverse_reactions, manufacturer
        
        PMC fields needed: pmcid, abstract, full_text, journal, venue, year, authors
        """
        set_id = payload.get("set_id", "")
        drug_name = payload.get("drug_name", payload.get("title", "Unknown Drug"))
        
        # Extract DailyMed content sections
        indications = payload.get("indications", "")
        dosage = payload.get("dosage", "")
        contraindications = payload.get("contraindications", "")
        warnings = payload.get("warnings", "")
        adverse_reactions = payload.get("adverse_reactions", "")
        manufacturer = payload.get("manufacturer", "")
        active_ingredients = payload.get("active_ingredients", [])
        
        # Build abstract from key sections (for reranking)
        abstract_parts = []
        if drug_name:
            abstract_parts.append(f"Drug: {drug_name}")
        if active_ingredients:
            if isinstance(active_ingredients, list):
                abstract_parts.append(f"Active Ingredients: {', '.join(active_ingredients)}")
            else:
                abstract_parts.append(f"Active Ingredients: {active_ingredients}")
        if indications:
            abstract_parts.append(f"Indications: {indications[:800]}")
        elif dosage:
            abstract_parts.append(f"Dosage: {dosage[:800]}")
        
        abstract = "\n".join(abstract_parts)
        
        # Build full text from all available sections
        full_text_parts = []
        if drug_name:
            full_text_parts.append(f"# {drug_name}")
        if manufacturer:
            full_text_parts.append(f"Manufacturer: {manufacturer}")
        if active_ingredients:
            if isinstance(active_ingredients, list):
                full_text_parts.append(f"\n## Active Ingredients\n{', '.join(active_ingredients)}")
            else:
                full_text_parts.append(f"\n## Active Ingredients\n{active_ingredients}")
        if indications:
            full_text_parts.append(f"\n## Indications and Usage\n{indications}")
        if dosage:
            full_text_parts.append(f"\n## Dosage and Administration\n{dosage}")
        if contraindications:
            full_text_parts.append(f"\n## Contraindications\n{contraindications}")
        if warnings:
            full_text_parts.append(f"\n## Warnings and Precautions\n{warnings}")
        if adverse_reactions:
            full_text_parts.append(f"\n## Adverse Reactions\n{adverse_reactions}")
        
        full_text = "\n".join(full_text_parts)
        
        return {
            "corpus_id": f"dailymed_{set_id}" if set_id else "",
            "pmcid": f"dailymed_{set_id}" if set_id else "",
            "pmid": None,
            "title": drug_name or payload.get("title", ""),
            "text": abstract[:2000] if abstract else "",
            "abstract": abstract,
            "full_text": full_text,
            "section_title": "drug_label",
            "journal": "DailyMed",
            "venue": "FDA Drug Label",
            "year": 2024,  # DailyMed labels are current
            "authors": [{"name": manufacturer}] if manufacturer else [],
            "article_type": "drug_label",
            "score": score,
            "stype": "vector_search",
            "source": "dailymed",
            # Preserve original DailyMed fields for reference
            "set_id": set_id,
            "drug_name": drug_name,
            "indications": indications,
            "dosage": dosage,
            "contraindications": contraindications,
            "warnings": warnings,
            "adverse_reactions": adverse_reactions,
        }
    
    def _parse_authors(self, authors_data) -> List[Dict[str, str]]:
        """Parse authors from payload into ScholarQA format."""
        if not authors_data:
            return []
        
        if isinstance(authors_data, str):
            # Simple string format: "Author A, Author B"
            names = [a.strip() for a in authors_data.split(",")]
            return [{"name": name} for name in names if name]
        
        if isinstance(authors_data, list):
            return [{"name": a} if isinstance(a, str) else a for a in authors_data]
        
        return []


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
