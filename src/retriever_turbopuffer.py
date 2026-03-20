"""turbopuffer retriever adapter with pipeline-compatible output contract."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

try:
    import turbopuffer as tpuf
except ImportError:  # pragma: no cover - environment-dependent
    tpuf = None  # type: ignore

from .config import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_BASE_URL,
    DEEPINFRA_EMBED_TIMEOUT_SECONDS,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    HF_INFERENCE_EMBED_TIMEOUT_SECONDS,
    HF_INFERENCE_ENDPOINT_API_KEY,
    HF_INFERENCE_ENDPOINT_URL,
    RETRIEVAL_DENSE_WEIGHT,
    RETRIEVAL_RRF_K,
    RETRIEVAL_SPARSE_WEIGHT,
    SCORE_THRESHOLD,
    TURBOPUFFER_API_KEY,
    TURBOPUFFER_REGION,
    TURBOPUFFER_TIMEOUT_SECONDS,
    TURBOPUFFER_NAMESPACE_DAILYMED,
    TURBOPUFFER_NAMESPACE_PMC,
    TURBOPUFFER_NAMESPACE_PUBMED,
    USE_HYBRID_SEARCH,
)

logger = logging.getLogger(__name__)


class TurbopufferRetriever:
    """turbopuffer retriever preserving the existing pipeline contract."""

    LOW_EVIDENCE_TERMS = {
        "case_report",
        "case_series",
        "letter",
        "editorial",
        "comment",
        "commentary",
        "news",
        "correspondence",
    }

    def __init__(
        self,
        n_retrieval: int = 150,
        n_keyword_search: int = 30,
        score_threshold: float = SCORE_THRESHOLD,
    ):
        if tpuf is None:
            raise ValueError("turbopuffer package is not installed")
        if not TURBOPUFFER_API_KEY:
            raise ValueError("TURBOPUFFER_API_KEY not set in config")
        self.n_retrieval = n_retrieval
        self.n_keyword_search = n_keyword_search
        self.score_threshold = score_threshold
        self.embedding_model = EMBEDDING_MODEL
        self.embedding_provider = EMBEDDING_PROVIDER
        self.use_hybrid_search = USE_HYBRID_SEARCH
        self.rrf_k = RETRIEVAL_RRF_K

        self.tpuf = tpuf.Turbopuffer(
            api_key=TURBOPUFFER_API_KEY,
            region=TURBOPUFFER_REGION,
            timeout=TURBOPUFFER_TIMEOUT_SECONDS,
        )
        self.ns_pmc = self.tpuf.namespace(TURBOPUFFER_NAMESPACE_PMC)
        self.ns_pubmed = self.tpuf.namespace(TURBOPUFFER_NAMESPACE_PUBMED)
        self.ns_dailymed = self.tpuf.namespace(TURBOPUFFER_NAMESPACE_DAILYMED)

        self.openai_client = None
        if self.embedding_provider == "hf_inference_endpoint":
            from openai import OpenAI

            self.openai_client = OpenAI(
                api_key=HF_INFERENCE_ENDPOINT_API_KEY,
                base_url=f"{HF_INFERENCE_ENDPOINT_URL}/v1",
                timeout=HF_INFERENCE_EMBED_TIMEOUT_SECONDS,
            )
        elif self.embedding_provider == "deepinfra":
            from openai import OpenAI

            self.openai_client = OpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_BASE_URL,
                timeout=DEEPINFRA_EMBED_TIMEOUT_SECONDS,
            )

    def _is_low_evidence_type(self, article_type: Any, publication_type: Any) -> bool:
        article = str(article_type or "").strip().lower()
        if article in self.LOW_EVIDENCE_TERMS:
            return True
        publications: list[str] = []
        if isinstance(publication_type, list):
            publications = [str(v).strip().lower() for v in publication_type]
        elif publication_type:
            publications = [str(publication_type).strip().lower()]
        return any(p in self.LOW_EVIDENCE_TERMS for p in publications)

    def _embed_query(self, query: str) -> Optional[List[float]]:
        if not self.openai_client:
            return None
        try:
            response = self.openai_client.embeddings.create(model=self.embedding_model, input=query)
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    def _embed_queries(self, queries: List[str]) -> List[Optional[List[float]]]:
        return [self._embed_query(query) for query in queries]

    @staticmethod
    def _rank_key_from_row(row: Dict[str, Any]) -> str:
        return str(row.get("chunk_id") or row.get("id") or row.get("pmcid") or row.get("doc_id") or "")

    @staticmethod
    def _doc_id_from_row(row: Dict[str, Any]) -> str:
        return str(row.get("pmcid") or row.get("doc_id") or row.get("pmid") or "")

    @staticmethod
    def _normalize_publication_type_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if value:
            return [str(value)]
        return []

    def _rrf_scores(self, rows: List[Dict[str, Any]], weight: float) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for rank, row in enumerate(rows, 1):
            key = str(row.get("_rank_key", ""))
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + ((1.0 / (float(self.rrf_k) + rank)) * weight)
        return scores

    @staticmethod
    def _token_overlap_score(query: str, text: str) -> float:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        text_tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
        if not query_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    def _query_namespace_dense(self, ns: Any, query_embedding: Optional[List[float]], limit: int) -> List[Dict[str, Any]]:
        if query_embedding is None:
            return []
        try:
            result = ns.query(rank_by=["vector", "ANN", query_embedding], top_k=limit, include_attributes=True)
            rows = getattr(result, "rows", None) or getattr(result, "results", None) or []
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Dense query failed: %s", exc)
            return []

    def _query_namespace_fts(self, ns: Any, query_text: str, limit: int) -> List[Dict[str, Any]]:
        try:
            result = ns.query(rank_by=["page_content", "BM25", query_text], top_k=limit, include_attributes=True)
            rows = getattr(result, "rows", None) or getattr(result, "results", None) or []
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("FTS query failed: %s", exc)
            return []

    def _query_namespace_fts_with_title_fallback(self, ns: Any, query_text: str, limit: int) -> List[Dict[str, Any]]:
        rows = self._query_namespace_fts(ns, query_text, limit)
        if rows:
            return rows
        try:
            result = ns.query(rank_by=["title", "BM25", query_text], top_k=limit, include_attributes=True)
            fallback_rows = getattr(result, "rows", None) or getattr(result, "results", None) or []
            return [dict(r) for r in fallback_rows]
        except Exception as exc:
            logger.warning("Title BM25 fallback failed: %s", exc)
            return []

    def _row_to_passage(self, row: Dict[str, Any], score: float, stype: str) -> Dict[str, Any]:
        source = row.get("source", "")
        article_type = row.get("article_type", "")
        if source == "dailymed" or article_type == "drug_label":
            return self._transform_dailymed_payload(row, score)
        doc_id = self._doc_id_from_row(row)
        return {
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
            "evidence_grade": row.get("evidence_grade"),
            "evidence_level": row.get("evidence_level"),
            "evidence_term": row.get("evidence_term"),
            "evidence_source": row.get("evidence_source"),
        }

    def build_sparse_query_vectors(self, queries: List[str]) -> List[dict[str, Any]]:
        _ = queries
        # Native BM25 in turbopuffer; rag_pipeline no longer needs sparse vectors.
        return []

    def batch_dense_source_fanout_search(
        self,
        queries: List[str],
        min_results: int = 60,
        fallback_broad: bool = False,
        **filter_kwargs: Any,
    ) -> Optional[List[Dict[str, Any]]]:
        _ = filter_kwargs, fallback_broad
        all_passages = self.batch_hybrid_search(queries=queries)
        if len(all_passages) < max(1, min_results):
            return None
        return all_passages

    def batch_hybrid_search(
        self,
        queries: List[str],
        sparse_vectors: Optional[List[Any]] = None,
        dense_weight: float = RETRIEVAL_DENSE_WEIGHT,
        sparse_weight: float = RETRIEVAL_SPARSE_WEIGHT,
        **filter_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        _ = sparse_vectors, filter_kwargs
        if not queries:
            return []

        dense_embeddings = self._embed_queries(queries)
        all_rows: Dict[str, Dict[str, Any]] = {}
        score_map: Dict[str, float] = {}

        namespaces = [self.ns_pmc, self.ns_pubmed]
        for i, query in enumerate(queries):
            dense_rows: List[Dict[str, Any]] = []
            fts_rows: List[Dict[str, Any]] = []
            for ns in namespaces:
                dense_rows.extend(self._query_namespace_dense(ns, dense_embeddings[i], self.n_retrieval))
                fts_rows.extend(self._query_namespace_fts_with_title_fallback(ns, query, self.n_retrieval))

            for row in dense_rows:
                key = self._rank_key_from_row(row)
                if not key:
                    continue
                row["_rank_key"] = key
                all_rows.setdefault(key, row)
            for row in fts_rows:
                key = self._rank_key_from_row(row)
                if not key:
                    continue
                row["_rank_key"] = key
                all_rows.setdefault(key, row)

            for key, score in self._rrf_scores(dense_rows, dense_weight).items():
                score_map[key] = score_map.get(key, 0.0) + score
            for key, score in self._rrf_scores(fts_rows, sparse_weight).items():
                score_map[key] = score_map.get(key, 0.0) + score

        ranked_keys = sorted(score_map.keys(), key=lambda k: score_map[k], reverse=True)[: self.n_retrieval]
        return [self._row_to_passage(all_rows[k], score_map[k], "batch_hybrid_search") for k in ranked_keys]

    def get_all_chunks_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        if not doc_id:
            return []
        try:
            rows = self._query_namespace_fts(self.ns_pmc, doc_id, 256)
            return [r for r in rows if str(r.get("doc_id", "")) == str(doc_id)]
        except Exception:
            return []

    def _aggregate_dailymed_payloads(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not payloads:
            return {}
        base = dict(payloads[0])
        base["dailymed_sections"] = {}
        for row in payloads:
            section_title = str(row.get("section_title") or "")
            content = row.get("page_content") or row.get("text") or ""
            if section_title and content:
                base["dailymed_sections"][section_title] = content
        return base

    def _transform_dailymed_payload(self, payload: Dict[str, Any], score: float) -> Dict[str, Any]:
        set_id = payload.get("set_id") or payload.get("doc_id") or ""
        return {
            "corpus_id": f"dailymed_{set_id}",
            "pmcid": f"dailymed_{set_id}",
            "set_id": set_id,
            "title": payload.get("title") or payload.get("drug_name", ""),
            "text": payload.get("page_content") or payload.get("text", ""),
            "abstract": payload.get("abstract", ""),
            "source": "dailymed",
            "article_type": "drug_label",
            "dailymed_sections": payload.get("dailymed_sections", {}),
            "score": score,
            "stype": "dailymed_lookup",
        }

    def search_dailymed_by_drug(self, drug_names: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        if not drug_names:
            return []
        results: List[Dict[str, Any]] = []
        seen_set_ids: set[str] = set()
        for drug in drug_names[:5]:
            rows = self._query_namespace_fts(self.ns_dailymed, drug, 64)
            matched: List[Dict[str, Any]] = []
            for row in rows:
                label = str(row.get("drug_name") or row.get("title") or "")
                text = str(row.get("page_content") or "")
                if self._token_overlap_score(drug, f"{label} {text}") <= 0:
                    continue
                matched.append(row)
            if not matched:
                continue
            set_id = str(matched[0].get("set_id") or matched[0].get("doc_id") or "")
            if not set_id or set_id in seen_set_ids:
                continue
            seen_set_ids.add(set_id)
            aggregate = self._aggregate_dailymed_payloads(matched)
            results.append(self._transform_dailymed_payload(aggregate, 1.0))
            if len(results) >= limit:
                break
        return results

    def apply_hybrid_scoring(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        dense_weight: float = RETRIEVAL_DENSE_WEIGHT,
        sparse_weight: float = RETRIEVAL_SPARSE_WEIGHT,
    ) -> List[Dict[str, Any]]:
        if not passages:
            return passages
        max_dense = max((p.get("score", 0.0) for p in passages), default=1.0) or 1.0
        for passage in passages:
            dense_score = float(passage.get("score", 0.0)) / max_dense
            keyword_score = self._token_overlap_score(query, str(passage.get("text", "")))
            hybrid = (dense_weight * dense_score) + (sparse_weight * keyword_score)
            passage["dense_score"] = dense_score
            passage["keyword_score"] = keyword_score
            passage["hybrid_score"] = hybrid
            passage["score"] = hybrid
        passages.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return passages
