import unittest

from qdrant_client.models import SparseVector

import src.rag_pipeline as rag_pipeline_module


class DummyRetriever:
    def __init__(self, fanout_result):
        self.fanout_result = fanout_result
        self.calls = []

    def batch_dense_source_fanout_search(self, **kwargs):
        self.calls.append("fanout")
        return self.fanout_result

    def build_sparse_query_vectors(self, queries):
        self.calls.append("sparse")
        return [SparseVector(indices=[], values=[]) for _ in queries]

    def batch_hybrid_search(self, **kwargs):
        self.calls.append("hybrid")
        return [{"corpus_id": "hybrid-result", "score": 0.5}]

    def search_dailymed_by_drug(self, drugs):
        self.calls.append("dailymed")
        return []


class DummyProcessedQuery:
    def __init__(self):
        self.retrieval_queries = ["query primary", "query secondary"]
        self.primary_query = "query primary"
        self.keyword_query = "query secondary"
        self.original_query = "query primary"
        self.decomposed = None


class DummyDecomposed:
    def __init__(self, is_drug_query: bool, drug_names):
        self.is_drug_query = is_drug_query
        self.drug_names = drug_names


class PipelineFanoutRoutingTests(unittest.TestCase):
    def _make_pipeline(self, retriever):
        pipeline = rag_pipeline_module.MedicalRAGPipeline.__new__(rag_pipeline_module.MedicalRAGPipeline)
        pipeline.retriever = retriever
        pipeline._extract_drug_names = lambda original, rewritten: []
        return pipeline

    def test_uses_fanout_path_without_hybrid_when_fanout_succeeds(self):
        original_enabled = rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED
        try:
            rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED = True
            retriever = DummyRetriever(fanout_result=[{"corpus_id": "fanout-result", "score": 0.9}])
            pipeline = self._make_pipeline(retriever)

            passages, dailymed = pipeline.retrieve_passages(DummyProcessedQuery())

            self.assertEqual(passages[0]["corpus_id"], "fanout-result")
            self.assertEqual(dailymed, [])
            self.assertIn("fanout", retriever.calls)
            self.assertNotIn("hybrid", retriever.calls)
        finally:
            rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED = original_enabled

    def test_falls_back_to_hybrid_when_fanout_returns_none(self):
        original_enabled = rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED
        try:
            rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED = True
            retriever = DummyRetriever(fanout_result=None)
            pipeline = self._make_pipeline(retriever)

            passages, dailymed = pipeline.retrieve_passages(DummyProcessedQuery())

            self.assertEqual(passages[0]["corpus_id"], "hybrid-result")
            self.assertEqual(dailymed, [])
            self.assertIn("fanout", retriever.calls)
            self.assertIn("sparse", retriever.calls)
            self.assertIn("hybrid", retriever.calls)
        finally:
            rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED = original_enabled

    def test_skips_dailymed_when_decomposition_marks_non_drug_query(self):
        original_enabled = rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED
        try:
            rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED = False
            retriever = DummyRetriever(fanout_result=[{"corpus_id": "unused", "score": 0.9}])
            pipeline = self._make_pipeline(retriever)

            processed = DummyProcessedQuery()
            processed.decomposed = DummyDecomposed(is_drug_query=False, drug_names=["metformin"])

            passages, dailymed = pipeline.retrieve_passages(processed)

            self.assertEqual(passages[0]["corpus_id"], "hybrid-result")
            self.assertEqual(dailymed, [])
            self.assertNotIn("dailymed", retriever.calls)
        finally:
            rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED = original_enabled

    def test_calls_dailymed_when_decomposition_marks_drug_query(self):
        original_enabled = rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED
        try:
            rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED = False
            retriever = DummyRetriever(fanout_result=[{"corpus_id": "unused", "score": 0.9}])
            pipeline = self._make_pipeline(retriever)

            processed = DummyProcessedQuery()
            processed.decomposed = DummyDecomposed(is_drug_query=True, drug_names=["golimumab"])

            _, dailymed = pipeline.retrieve_passages(processed)

            self.assertEqual(dailymed, [])
            self.assertIn("dailymed", retriever.calls)
        finally:
            rag_pipeline_module.RETRIEVAL_SOURCE_FANOUT_ENABLED = original_enabled


if __name__ == "__main__":
    unittest.main()
