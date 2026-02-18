import unittest

from src.query_preprocessor import LLMProcessedQuery
from src.rag_pipeline import MedicalRAGPipeline
from src.retriever_qdrant import QdrantRetriever


class _NoOpRetriever:
    def apply_hybrid_scoring(self, query, passages, dense_weight, sparse_weight):
        return passages


class _FakePapers:
    def __init__(self, rows):
        self.rows = rows
        self.empty = len(rows) == 0

    def __len__(self):
        return len(self.rows)


class TestDailyMedFallbackBehavior(unittest.TestCase):
    def _base_pipeline(self):
        pipeline = MedicalRAGPipeline.__new__(MedicalRAGPipeline)
        pipeline.retriever = _NoOpRetriever()
        pipeline.evidence_hierarchy = {"levels": []}
        pipeline.preprocess_query = lambda q: LLMProcessedQuery(
            rewritten_query=q,
            keyword_query=q,
            search_filters={},
            original_query=q,
            decomposed=None,
            expanded_queries=[],
        )
        pipeline._check_pdf_availability = lambda papers: papers
        return pipeline

    def test_answer_does_not_fallback_when_only_dailymed_results_exist(self):
        pipeline = self._base_pipeline()

        fallback_called = {"value": False}

        def _fallback(*args, **kwargs):
            fallback_called["value"] = True
            return "fallback", []

        pipeline._run_fallback_generation = _fallback
        pipeline.retrieve_passages = lambda processed_query: ([], [{"pmcid": "DM-1"}])
        pipeline.rerank_and_aggregate = lambda *args, **kwargs: (_FakePapers([{"pmcid": "DM-1"}]), [])
        pipeline.run_generation = lambda query, papers_df: ("generated-answer", [])

        result = pipeline.answer("test query")

        self.assertFalse(fallback_called["value"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "generated-answer")

    def test_answer_falls_back_when_no_passages_and_no_dailymed(self):
        pipeline = self._base_pipeline()

        fallback_called = {"value": False}

        def _fallback(*args, **kwargs):
            fallback_called["value"] = True
            return "fallback-answer", []

        pipeline._run_fallback_generation = _fallback
        pipeline.retrieve_passages = lambda processed_query: ([], [])
        pipeline.rerank_and_aggregate = lambda *args, **kwargs: (_FakePapers([]), [])
        pipeline.run_generation = lambda query, papers_df: ("generated-answer", [])

        result = pipeline.answer("test query")

        self.assertTrue(fallback_called["value"])
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["answer"], "fallback-answer")

    def test_hybrid_fallback_preserves_filters(self):
        retriever = QdrantRetriever.__new__(QdrantRetriever)
        captured = {}

        class _EmptySparse:
            indices = []

        retriever._build_sparse_query_vector = lambda query: _EmptySparse()

        def _capture_retrieve(query, use_hybrid=False, **kwargs):
            captured["use_hybrid"] = use_hybrid
            captured["kwargs"] = kwargs
            return []

        retriever.retrieve_passages = _capture_retrieve

        result = retriever._hybrid_search(
            query="heart failure latest guidance",
            search_filter=None,
            filter_kwargs={"year": "2022-2025", "venue": "Lancet"},
        )

        self.assertEqual(result, [])
        self.assertFalse(captured["use_hybrid"])
        self.assertEqual(captured["kwargs"]["year"], "2022-2025")
        self.assertEqual(captured["kwargs"]["venue"], "Lancet")


if __name__ == "__main__":
    unittest.main()
