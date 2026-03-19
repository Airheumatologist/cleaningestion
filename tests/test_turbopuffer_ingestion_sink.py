from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from turbopuffer_ingestion_sink import TurbopufferIngestionSink


class _Point:
    def __init__(self, point_id: str, vector: list[float], payload: dict):
        self.id = point_id
        self.vector = {"dense": vector}
        self.payload = payload


class TurbopufferIngestionSinkTests(unittest.TestCase):
    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Namespace")
    def test_schema_sent_only_on_first_write(self, mock_namespace_cls):
        ns = mock.Mock()
        mock_namespace_cls.return_value = ns
        sink = TurbopufferIngestionSink(namespace="medical_pmc", dry_run=False)
        point = _Point("11111111-1111-1111-1111-111111111111", [0.1, 0.2], {"page_content": "hello"})

        sink.write_points([point])
        sink.write_points([point])

        self.assertEqual(ns.write.call_count, 2)
        first_call = ns.write.call_args_list[0].kwargs
        second_call = ns.write.call_args_list[1].kwargs
        self.assertIn("schema", first_call)
        self.assertNotIn("schema", second_call)

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Namespace")
    def test_rows_transposed_with_union_of_keys(self, mock_namespace_cls):
        ns = mock.Mock()
        mock_namespace_cls.return_value = ns
        sink = TurbopufferIngestionSink(namespace="medical_pubmed", dry_run=False)
        p1 = _Point("11111111-1111-1111-1111-111111111111", [0.1], {"a": "x"})
        p2 = _Point("22222222-2222-2222-2222-222222222222", [0.2], {"b": "y"})

        sink.write_points([p1, p2])
        kwargs = ns.write.call_args.kwargs
        cols = kwargs["upsert_columns"]
        self.assertIn("a", cols)
        self.assertIn("b", cols)
        self.assertEqual(cols["a"], ["x", None])
        self.assertEqual(cols["b"], [None, "y"])

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Namespace")
    def test_retry_on_retryable_status(self, mock_namespace_cls):
        ns = mock.Mock()
        ns.write.side_effect = [RuntimeError("429 too many requests"), None]
        mock_namespace_cls.return_value = ns
        sink = TurbopufferIngestionSink(namespace="medical_dailymed", dry_run=False)
        point = _Point("33333333-3333-3333-3333-333333333333", [0.3], {"title": "x"})

        sink.write_points([point])
        self.assertEqual(ns.write.call_count, 2)

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Namespace")
    def test_dry_run_skips_writes(self, mock_namespace_cls):
        ns = mock.Mock()
        mock_namespace_cls.return_value = ns
        sink = TurbopufferIngestionSink(namespace="medical_pmc", dry_run=True)
        point = _Point("44444444-4444-4444-4444-444444444444", [0.4], {"title": "x"})

        written = sink.write_points([point])
        self.assertEqual(written, 1)
        ns.write.assert_not_called()

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Namespace")
    def test_sparse_vector_ignored(self, mock_namespace_cls):
        ns = mock.Mock()
        mock_namespace_cls.return_value = ns
        sink = TurbopufferIngestionSink(namespace="medical_pmc", dry_run=False)
        point = mock.Mock()
        point.id = "55555555-5555-5555-5555-555555555555"
        point.vector = {"dense": [0.1], "sparse": {"indices": [1], "values": [0.5]}}
        point.payload = {"title": "sparse-test"}

        sink.write_points([point])
        cols = ns.write.call_args.kwargs["upsert_columns"]
        self.assertIn("vector", cols)
        self.assertNotIn("sparse_indices", cols)
        self.assertNotIn("sparse_values", cols)


if __name__ == "__main__":
    unittest.main()

