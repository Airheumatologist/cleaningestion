from __future__ import annotations

import unittest
from unittest import mock

import src.retriever_turbopuffer as retriever_turbopuffer


class RetrieverTurbopufferQueryPayloadTests(unittest.TestCase):
    def _make_retriever(self):
        retriever = retriever_turbopuffer.TurbopufferRetriever.__new__(retriever_turbopuffer.TurbopufferRetriever)
        retriever.n_retrieval = 11
        return retriever

    def test_dense_query_payload_uses_vector_kwargs(self):
        retriever = self._make_retriever()
        ns = mock.Mock()
        ns.query.return_value = mock.Mock(rows=[{"chunk_id": "dense-1", "score": 0.9}])

        rows = retriever._query_namespace_dense(ns, [0.1, 0.2, 0.3], 7)

        self.assertEqual(rows, [{"chunk_id": "dense-1", "score": 0.9}])
        ns.query.assert_called_once_with(rank_by=["vector", "ANN", [0.1, 0.2, 0.3]], top_k=7, include_attributes=True)

    def test_fts_query_payload_uses_text_kwargs(self):
        retriever = self._make_retriever()
        ns = mock.Mock()
        ns.query.return_value = mock.Mock(rows=[{"chunk_id": "fts-1", "score": 0.7}])

        rows = retriever._query_namespace_fts(ns, "metformin drug label", 64)

        self.assertEqual(rows, [{"chunk_id": "fts-1", "score": 0.7}])
        ns.query.assert_called_once_with(rank_by=["page_content", "BM25", "metformin drug label"], top_k=64, include_attributes=True)

    def test_title_fallback_uses_rank_by_on_second_query(self):
        retriever = self._make_retriever()
        ns = mock.Mock()
        ns.query.side_effect = [mock.Mock(rows=[]), mock.Mock(rows=[{"chunk_id": "title-1", "score": 0.6}])]

        rows = retriever._query_namespace_fts_with_title_fallback(ns, "metformin", 5)

        self.assertEqual(rows, [{"chunk_id": "title-1", "score": 0.6}])
        self.assertEqual(ns.query.call_count, 2)
        self.assertEqual(
            ns.query.call_args_list[0].kwargs["rank_by"],
            ["page_content", "BM25", "metformin"],
        )
        self.assertEqual(
            ns.query.call_args_list[1].kwargs["rank_by"],
            ["title", "BM25", "metformin"],
        )


if __name__ == "__main__":
    unittest.main()
