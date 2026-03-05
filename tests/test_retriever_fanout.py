import unittest

from src.retriever_qdrant import QdrantRetriever


class DummyPoint:
    def __init__(self, pid: str, score: float, payload: dict):
        self.id = pid
        self.score = score
        self.payload = payload


class DummyResult:
    def __init__(self, points):
        self.points = points


class RetrieverFanoutTests(unittest.TestCase):
    def _make_retriever(self) -> QdrantRetriever:
        retriever = QdrantRetriever.__new__(QdrantRetriever)
        retriever.n_retrieval = 10
        retriever.score_threshold = 0.25
        retriever.collection_name = "rag_pipeline"
        return retriever

    @staticmethod
    def _make_point(key: str, score: float, source_family: str, source: str) -> DummyPoint:
        return DummyPoint(
            pid=key,
            score=score,
            payload={
                "chunk_id": key,
                "doc_id": key,
                "pmcid": key,
                "title": f"title-{key}",
                "source": source,
                "source_family": source_family,
                "page_content": f"text-{key}",
            },
        )

    def test_build_filter_supports_source_family(self):
        retriever = self._make_retriever()
        qfilter = retriever._build_filter(source_family="pmc")
        payload = qfilter.model_dump()
        must = payload.get("must", [])

        source_family_conditions = [
            item for item in must
            if item.get("key") == "source_family"
            and item.get("match", {}).get("value") == "pmc"
        ]
        self.assertEqual(len(source_family_conditions), 1)

    def test_fanout_merges_and_deduplicates_deterministically(self):
        retriever = self._make_retriever()
        retriever._embed_queries = lambda queries: [[0.01] for _ in queries]

        pmc_points = [
            self._make_point("A", 0.9, "pmc", "pmc_oa"),
            self._make_point("B", 0.8, "pmc", "pmc_oa"),
        ]
        pubmed_points = [
            self._make_point("B", 0.95, "pubmed", "pubmed_abstract"),
            self._make_point("C", 0.7, "pubmed", "pubmed_abstract"),
        ]

        def fake_run_dense_batch(operation, query_contexts, search_filter):
            if operation.endswith(":pmc"):
                return [DummyResult(pmc_points)]
            if operation.endswith(":pubmed"):
                return [DummyResult(pubmed_points)]
            raise AssertionError(f"Unexpected operation: {operation}")

        retriever._run_dense_batch = fake_run_dense_batch

        passages = retriever.batch_dense_source_fanout_search(
            queries=["rheumatoid arthritis treatment"],
            min_results=1,
            fallback_broad=False,
        )

        self.assertIsNotNone(passages)
        corpus_ids = [p.get("corpus_id") for p in passages]
        self.assertEqual(len(set(corpus_ids)), len(corpus_ids))
        self.assertEqual(set(corpus_ids), {"A", "B", "C"})
        self.assertEqual(passages[0].get("corpus_id"), "B")

    def test_fanout_returns_none_when_branch_fails_and_no_broad_fallback(self):
        retriever = self._make_retriever()
        retriever._embed_queries = lambda queries: [[0.01] for _ in queries]

        def fake_run_dense_batch(operation, query_contexts, search_filter):
            if operation.endswith(":pmc"):
                return [DummyResult([self._make_point("A", 0.9, "pmc", "pmc_oa")])]
            raise TimeoutError("pubmed timeout")

        retriever._run_dense_batch = fake_run_dense_batch

        passages = retriever.batch_dense_source_fanout_search(
            queries=["my query"],
            min_results=1,
            fallback_broad=False,
        )
        self.assertIsNone(passages)

    def test_fanout_uses_broad_fallback_when_enabled(self):
        retriever = self._make_retriever()
        retriever._embed_queries = lambda queries: [[0.01] for _ in queries]

        def fake_run_dense_batch(operation, query_contexts, search_filter):
            if operation.endswith(":pmc"):
                return [DummyResult([self._make_point("A", 0.9, "pmc", "pmc_oa")])]
            raise TimeoutError("pubmed timeout")

        retriever._run_dense_batch = fake_run_dense_batch
        retriever._batch_dense_broad_search = lambda query_contexts, filter_kwargs: [
            {"corpus_id": "broad1", "score": 0.8},
            {"corpus_id": "broad2", "score": 0.7},
        ]

        passages = retriever.batch_dense_source_fanout_search(
            queries=["my query"],
            min_results=2,
            fallback_broad=True,
        )

        self.assertIsNotNone(passages)
        self.assertEqual(len(passages), 2)
        self.assertEqual(passages[0]["corpus_id"], "broad1")


if __name__ == "__main__":
    unittest.main()
