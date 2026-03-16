"""LanceDB retriever adapter with Qdrant-compatible output contract."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import lancedb
from lancedb.rerankers import RRFReranker
from qdrant_client.models import SparseVector

from .config import (
    LANCEDB_TABLE,
    LANCEDB_URI,
    RETRIEVAL_PREFILTER,
)
from .retriever_qdrant import QdrantRetriever

logger = logging.getLogger(__name__)


class LanceDBRetriever(QdrantRetriever):
    """LanceDB retriever that preserves the existing retriever contract."""

    def __init__(
        self,
        n_retrieval: int = 150,
        n_keyword_search: int = 30,
        score_threshold: float = 0.25,
        uri: str = LANCEDB_URI,
        table_name: str = LANCEDB_TABLE,
    ):
        super().__init__(
            n_retrieval=n_retrieval,
            n_keyword_search=n_keyword_search,
            score_threshold=score_threshold,
        )
        self.lancedb_uri = uri
        self.lancedb_table_name = table_name
        self._db = lancedb.connect(uri)
        self.table = self._db.open_table(table_name)
        logger.info("Initialized LanceDBRetriever uri=%s table=%s", uri, table_name)

    @staticmethod
    def _safe_sql(value: str) -> str:
        return value.replace("'", "''")

    def _build_where_sql(self, include_dailymed: bool = False, **kwargs: Any) -> Optional[str]:
        clauses: List[str] = []

        if not include_dailymed:
            clauses.append("source != 'dailymed'")
            clauses.append("article_type != 'drug_label'")

        if kwargs.get("year"):
            year_str = str(kwargs["year"])
            parts = year_str.split("-")
            if len(parts) >= 1 and parts[0].strip():
                clauses.append(f"year >= {int(parts[0])}")
            if len(parts) >= 2 and parts[1].strip():
                clauses.append(f"year <= {int(parts[1])}")

        if kwargs.get("venue"):
            venues = [v.strip() for v in str(kwargs["venue"]).split(",") if v.strip()]
            if venues:
                venue_values = ", ".join(f"'{self._safe_sql(v)}'" for v in venues)
                clauses.append(f"journal IN ({venue_values})")

        if kwargs.get("article_type"):
            article_types = [t.strip() for t in str(kwargs["article_type"]).split(",") if t.strip()]
            if article_types:
                values = ", ".join(f"'{self._safe_sql(v)}'" for v in article_types)
                clauses.append(f"article_type IN ({values})")

        if kwargs.get("source_family"):
            source_family = str(kwargs["source_family"]).strip().lower()
            if source_family:
                clauses.append(f"source_family = '{self._safe_sql(source_family)}'")

        if kwargs.get("is_gov_affiliated") is not None:
            bool_value = kwargs["is_gov_affiliated"]
            if isinstance(bool_value, str):
                bool_value = bool_value.lower() in {"1", "true", "yes", "on"}
            clauses.append(f"is_gov_affiliated = {'true' if bool_value else 'false'}")

        if kwargs.get("gov_agency"):
            agencies = [a.strip() for a in str(kwargs["gov_agency"]).split(",") if a.strip()]
            if agencies:
                agency_clauses = [
                    f"array_contains(gov_agencies, '{self._safe_sql(agency)}')"
                    for agency in agencies
                ]
                clauses.append("(" + " OR ".join(agency_clauses) + ")")

        return " AND ".join(clauses) if clauses else None

    @staticmethod
    def _rank_key_from_row(row: Dict[str, Any]) -> str:
        return str(
            row.get("chunk_id")
            or row.get("point_id")
            or row.get("id")
            or row.get("pmcid")
            or row.get("doc_id")
        )

    @staticmethod
    def _doc_id_from_row(row: Dict[str, Any]) -> str:
        return str(row.get("pmcid") or row.get("doc_id") or row.get("pmid") or "")

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        # LanceDB vector search returns distance where lower is better.
        # Convert to a monotonic similarity so higher remains better.
        clamped = max(0.0, float(distance))
        return 1.0 / (1.0 + clamped)

    def _search_dense(self, query_embedding: Any, where_sql: Optional[str], limit: int) -> List[Dict[str, Any]]:
        query = self.table.search(query_embedding, query_type="vector")
        if where_sql:
            query = query.where(where_sql, prefilter=RETRIEVAL_PREFILTER)
        return query.limit(limit).to_list()

    def _search_fts(self, text_query: str, where_sql: Optional[str], limit: int) -> List[Dict[str, Any]]:
        query = self.table.search(text_query, query_type="fts")
        if where_sql:
            query = query.where(where_sql, prefilter=RETRIEVAL_PREFILTER)
        return query.limit(limit).to_list()

    def _format_row_passage(
        self,
        row: Dict[str, Any],
        score: float,
        stype: str,
        dense_score: Optional[float] = None,
        sparse_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        source = row.get("source", "")
        article_type = row.get("article_type", "")
        is_dailymed = source == "dailymed" or article_type == "drug_label"

        if is_dailymed:
            passage = self._transform_dailymed_payload(row, score)
            passage["stype"] = stype
            if dense_score is not None:
                passage["dense_score"] = dense_score
            if sparse_score is not None:
                passage["sparse_score"] = sparse_score
            return passage

        doc_id = self._doc_id_from_row(row)
        passage: Dict[str, Any] = {
            "corpus_id": doc_id,
            "pmcid": row.get("pmcid", doc_id),
            "pmid": row.get("pmid"),
            "doi": row.get("doi"),
            "title": row.get("title", ""),
            "text": row.get("page_content") or row.get("abstract", ""),
            "abstract": row.get("abstract", ""),
            "full_text": "",
            "has_full_text": row.get("has_full_text", False),
            "section_title": row.get("section_title", "abstract"),
            "section_type": row.get("section_type", "body"),
            "chunk_id": row.get("chunk_id"),
            "chunk_index": row.get("chunk_index"),
            "journal": row.get("journal", ""),
            "venue": row.get("journal", ""),
            "nlm_unique_id": row.get("nlm_unique_id"),
            "year": row.get("year"),
            "authors": [],
            "article_type": row.get("article_type", ""),
            "publication_type": self._normalize_publication_type_list(row.get("publication_type")),
            "score": score,
            "stype": stype,
            "is_gov_affiliated": row.get("is_gov_affiliated", False),
            "gov_agencies": row.get("gov_agencies", []),
            "source": row.get("source", ""),
            "source_family": row.get("source_family", ""),
            **self._extract_evidence_metadata(row),
        }
        if dense_score is not None:
            passage["dense_score"] = dense_score
        if sparse_score is not None:
            passage["sparse_score"] = sparse_score
        return passage

    def retrieve_passages(
        self,
        query: str,
        use_hybrid: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        precomputed_sparse_vector: Optional[SparseVector] = None,
        **filter_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        del precomputed_sparse_vector

        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return []

        where_sql = self._build_where_sql(**filter_kwargs)

        if use_hybrid:
            dense_rows = self._search_dense(query_embedding, where_sql, self.n_retrieval * 2)
            sparse_rows = self._search_fts(query, where_sql, self.n_retrieval * 2)
            dense_scores: Dict[str, float] = {}
            sparse_scores: Dict[str, float] = {}
            rows_by_key: Dict[str, Dict[str, Any]] = {}

            for rank, row in enumerate(dense_rows, 1):
                key = self._rank_key_from_row(row)
                dense_scores[key] = (1.0 / (60.0 + rank)) * dense_weight
                rows_by_key.setdefault(key, row)

            for rank, row in enumerate(sparse_rows, 1):
                key = self._rank_key_from_row(row)
                sparse_scores[key] = (1.0 / (60.0 + rank)) * sparse_weight
                rows_by_key.setdefault(key, row)

            combined = {k: dense_scores.get(k, 0.0) + sparse_scores.get(k, 0.0) for k in set(dense_scores) | set(sparse_scores)}
            ranked_keys = sorted(combined.keys(), key=lambda key: combined[key], reverse=True)[: self.n_retrieval]

            passages: List[Dict[str, Any]] = []
            for key in ranked_keys:
                row = rows_by_key[key]
                passage = self._format_row_passage(
                    row,
                    score=combined[key],
                    stype="hybrid_search",
                    dense_score=dense_scores.get(key, 0.0),
                    sparse_score=sparse_scores.get(key, 0.0),
                )
                passage["raw_score"] = combined[key]
                passages.append(self._apply_retrieval_recency_boost(passage, filter_kwargs=filter_kwargs))

            passages.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
            return passages

        dense_rows = self._search_dense(query_embedding, where_sql, self.n_retrieval)
        passages: List[Dict[str, Any]] = []
        for row in dense_rows:
            distance = float(row.get("_distance", 0.0) or 0.0)
            similarity = self._distance_to_similarity(distance)
            passages.append(
                self._apply_retrieval_recency_boost(
                    self._format_row_passage(row, score=similarity, stype="vector_search"),
                    filter_kwargs=filter_kwargs,
                )
            )
        passages.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
        return passages

    def build_sparse_query_vectors(self, queries: List[str]) -> List[SparseVector]:
        return [SparseVector(indices=[], values=[]) for _ in queries]

    def batch_hybrid_search(
        self,
        queries: List[str],
        sparse_vectors: Optional[List[SparseVector]] = None,
        **filter_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        del sparse_vectors
        all_rows: Dict[str, Dict[str, Any]] = {}
        combined_scores: Dict[str, float] = {}

        for query in queries:
            passages = self.retrieve_passages(
                query,
                use_hybrid=True,
                dense_weight=0.7,
                sparse_weight=0.3,
                **filter_kwargs,
            )
            for rank, passage in enumerate(passages, 1):
                key = str(passage.get("chunk_id") or passage.get("corpus_id"))
                combined_scores[key] = combined_scores.get(key, 0.0) + (1.0 / (60.0 + rank))
                all_rows[key] = passage

        ranked = sorted(combined_scores.keys(), key=lambda key: combined_scores[key], reverse=True)[: self.n_retrieval]
        merged: List[Dict[str, Any]] = []
        for key in ranked:
            passage = dict(all_rows[key])
            passage["raw_score"] = combined_scores[key]
            passage["score"] = combined_scores[key] * float(passage.get("retrieval_recency_mult", 1.0))
            passage["stype"] = "batch_hybrid_search"
            merged.append(passage)

        merged.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
        return merged

    def batch_dense_source_fanout_search(
        self,
        queries: List[str],
        min_results: int = 60,
        fallback_broad: bool = False,
        **filter_kwargs: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        family_passages: List[Dict[str, Any]] = []
        for family in ("pmc", "pubmed"):
            family_filters = dict(filter_kwargs)
            family_filters["source_family"] = family
            family_passages.extend(
                self.batch_hybrid_search(queries=queries, **family_filters)
            )

        dedup: Dict[str, Dict[str, Any]] = {}
        for passage in family_passages:
            key = str(passage.get("chunk_id") or passage.get("corpus_id"))
            current = dedup.get(key)
            if current is None or float(passage.get("score", 0.0)) > float(current.get("score", 0.0)):
                dedup[key] = passage

        merged = sorted(dedup.values(), key=lambda p: float(p.get("score", 0.0)), reverse=True)
        if len(merged) < min_results and not fallback_broad:
            return None
        if len(merged) < min_results and fallback_broad:
            return self.batch_hybrid_search(queries=queries, **filter_kwargs)
        return merged[: self.n_retrieval]

    def retrieve_additional_papers(self, query: str, **filter_kwargs: Any) -> List[Dict[str, Any]]:
        return self.retrieve_passages(query, use_hybrid=False, **filter_kwargs)[: self.n_keyword_search]

    def get_all_chunks_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        normalized = str(doc_id or "").strip()
        if not normalized:
            return []

        safe_doc_id = self._safe_sql(normalized)
        where_sql = f"doc_id = '{safe_doc_id}' OR pmcid = '{safe_doc_id}'"
        try:
            rows = (
                self.table.search()
                .where(where_sql, prefilter=True)
                .limit(4096)
                .to_list()
            )
        except Exception as exc:
            logger.warning("Failed to fetch chunks for doc_id=%s: %s", normalized, exc)
            return []
        rows.sort(key=self._chunk_sort_key)
        return rows

    def search_dailymed_by_drug(self, drug_names: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        hits: Dict[str, Dict[str, Any]] = {}

        for drug_name in drug_names:
            try:
                rows = self._search_fts(drug_name, None, max(limit * 8, 40))
            except Exception as exc:
                logger.warning("DailyMed FTS failed for %s: %s", drug_name, exc)
                rows = []

            if not rows:
                # Fallback path: broad source-filtered scan with lexical post-filter.
                try:
                    rows = self.table.head(max(limit * 20, 100)).to_pylist()
                except Exception as exc:
                    logger.warning("DailyMed fallback scan failed for %s: %s", drug_name, exc)
                    rows = []

            for row in rows:
                if str(row.get("source", "")).lower() != "dailymed":
                    continue
                haystack = " ".join(
                    [
                        str(row.get("drug_name", "")),
                        str(row.get("title", "")),
                        str(row.get("page_content", "")),
                        str(row.get("doc_id", "")),
                    ]
                ).lower()
                if drug_name.lower() not in haystack:
                    continue
                set_id = str(row.get("set_id") or row.get("doc_id") or row.get("pmcid") or "")
                if not set_id:
                    continue
                row_with_fallbacks = dict(row)
                row_with_fallbacks.setdefault("set_id", set_id)
                if not row_with_fallbacks.get("drug_name"):
                    row_with_fallbacks["drug_name"] = row_with_fallbacks.get("title") or set_id
                passage = self._transform_dailymed_payload(
                    row_with_fallbacks,
                    float(row.get("_relevance_score", 0.0)),
                )
                passage["stype"] = "dailymed_fts"
                prev = hits.get(set_id)
                if prev is None or float(passage.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    hits[set_id] = passage

        return sorted(hits.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)[:limit]

    def apply_hybrid_scoring(self, passages: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        # Retain existing behavior exactly via parent implementation.
        return super().apply_hybrid_scoring(passages, query)

    def hybrid_search_with_rrf(self, query: str, **filter_kwargs: Any) -> List[Dict[str, Any]]:
        """Single-query hybrid retrieval using LanceDB's built-in hybrid + RRF pipeline."""
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return []

        where_sql = self._build_where_sql(**filter_kwargs)
        hybrid_query = self.table.search(query_type="hybrid").vector(query_embedding).text(query)
        if where_sql:
            hybrid_query = hybrid_query.where(where_sql, prefilter=RETRIEVAL_PREFILTER)

        rows = hybrid_query.rerank(reranker=RRFReranker()).limit(self.n_retrieval).to_list()
        passages = [
            self._apply_retrieval_recency_boost(
                self._format_row_passage(
                    row,
                    score=float(row.get("_relevance_score", 0.0)),
                    stype="hybrid_rrf",
                ),
                filter_kwargs=filter_kwargs,
            )
            for row in rows
        ]
        passages.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
        return passages
