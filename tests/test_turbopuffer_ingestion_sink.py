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
    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_REGION", "test-region")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Turbopuffer", create=True)
    def test_schema_sent_only_on_first_write(self, mock_tpuf_cls):
        client = mock.Mock()
        ns = mock.Mock()
        client.namespace.return_value = ns
        mock_tpuf_cls.return_value = client
        sink = TurbopufferIngestionSink(namespace="medical_pmc", dry_run=False)
        point = _Point("11111111-1111-1111-1111-111111111111", [0.1, 0.2], {"page_content": "hello"})

        mock_tpuf_cls.assert_called_once_with(api_key="test-key", region="test-region")
        client.namespace.assert_called_once_with("medical_pmc")
        sink.write_points([point])
        sink.write_points([point])

        self.assertEqual(ns.write.call_count, 2)
        first_call = ns.write.call_args_list[0].kwargs
        second_call = ns.write.call_args_list[1].kwargs
        self.assertIn("schema", first_call)
        self.assertEqual(first_call["distance_metric"], "cosine_distance")
        self.assertNotIn("schema", second_call)
        self.assertEqual(second_call["distance_metric"], "cosine_distance")
        self.assertEqual(first_call["schema"]["id"], "uuid")
        self.assertEqual(first_call["schema"]["vector"], {"type": "[1024]f16", "ann": True})
        self.assertEqual(first_call["schema"]["page_content"]["type"], "string")
        self.assertFalse(first_call["schema"]["page_content"]["filterable"])
        self.assertIn("full_text_search", first_call["schema"]["page_content"])

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_REGION", "test-region")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Turbopuffer", create=True)
    def test_rows_transposed_with_union_of_keys(self, mock_namespace_cls):
        client = mock.Mock()
        ns = mock.Mock()
        client.namespace.return_value = ns
        mock_namespace_cls.return_value = client
        sink = TurbopufferIngestionSink(namespace="medical_pubmed", dry_run=False)
        p1 = _Point("11111111-1111-1111-1111-111111111111", [0.1], {"a": "x"})
        p2 = _Point("22222222-2222-2222-2222-222222222222", [0.2], {"b": "y"})

        sink.write_points([p1, p2])
        kwargs = ns.write.call_args.kwargs
        self.assertEqual(kwargs["distance_metric"], "cosine_distance")
        cols = kwargs["upsert_columns"]
        self.assertIn("a", cols)
        self.assertIn("b", cols)
        self.assertEqual(cols["a"], ["x", None])
        self.assertEqual(cols["b"], [None, "y"])

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_REGION", "test-region")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Turbopuffer", create=True)
    @mock.patch("turbopuffer_ingestion_sink.time.sleep")
    @mock.patch("turbopuffer_ingestion_sink.random.uniform", return_value=0.0)
    def test_retry_on_retryable_status(self, mock_uniform, mock_sleep, mock_tpuf_cls):
        client = mock.Mock()
        ns = mock.Mock()
        ns.write.side_effect = [RuntimeError("429 too many requests"), None]
        client.namespace.return_value = ns
        mock_tpuf_cls.return_value = client
        sink = TurbopufferIngestionSink(namespace="medical_dailymed", dry_run=False)
        point = _Point("33333333-3333-3333-3333-333333333333", [0.3], {"title": "x"})

        sink.write_points([point])
        self.assertEqual(ns.write.call_count, 2)
        self.assertEqual(ns.write.call_args_list[0].kwargs["distance_metric"], "cosine_distance")
        self.assertEqual(ns.write.call_args_list[1].kwargs["distance_metric"], "cosine_distance")
        mock_uniform.assert_called_once()
        mock_sleep.assert_called_once_with(1.0)

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_REGION", "test-region")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Turbopuffer", create=True)
    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_WRITE_BATCH_SIZE", 1)
    @mock.patch("turbopuffer_ingestion_sink.time.sleep")
    @mock.patch("turbopuffer_ingestion_sink.time.monotonic", side_effect=[0.0, 0.0, 0.0])
    def test_min_batch_interval_paces_writes(
        self,
        _mock_monotonic,
        mock_sleep,
        mock_tpuf_cls,
    ):
        client = mock.Mock()
        ns = mock.Mock()
        client.namespace.return_value = ns
        mock_tpuf_cls.return_value = client
        with mock.patch.dict("os.environ", {"TURBOPUFFER_MIN_BATCH_INTERVAL_SECONDS": "0.25"}, clear=False):
            sink = TurbopufferIngestionSink(namespace="medical_pmc", dry_run=False)
            p1 = _Point("77777777-7777-7777-7777-777777777777", [0.1], {"title": "x"})
            p2 = _Point("88888888-8888-8888-8888-888888888888", [0.2], {"title": "y"})
            sink.write_points([p1, p2])

        self.assertEqual(ns.write.call_count, 2)
        mock_sleep.assert_called_once_with(0.25)

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_REGION", "test-region")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Turbopuffer", create=True)
    def test_dry_run_skips_writes(self, mock_namespace_cls):
        client = mock.Mock()
        ns = mock.Mock()
        client.namespace.return_value = ns
        mock_namespace_cls.return_value = client
        sink = TurbopufferIngestionSink(namespace="medical_pmc", dry_run=True)
        point = _Point("44444444-4444-4444-4444-444444444444", [0.4], {"title": "x"})

        written = sink.write_points([point])
        self.assertEqual(written, 1)
        ns.write.assert_not_called()

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_REGION", "test-region")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Turbopuffer", create=True)
    def test_sparse_vector_ignored(self, mock_namespace_cls):
        client = mock.Mock()
        ns = mock.Mock()
        client.namespace.return_value = ns
        mock_namespace_cls.return_value = client
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

    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_API_KEY", "test-key")
    @mock.patch("turbopuffer_ingestion_sink.IngestionConfig.TURBOPUFFER_REGION", "test-region")
    @mock.patch("turbopuffer_ingestion_sink.tpuf.Turbopuffer", create=True)
    def test_non_retryable_status_fails_without_retry(self, mock_tpuf_cls):
        client = mock.Mock()
        ns = mock.Mock()
        ns.write.side_effect = RuntimeError("400 bad request")
        client.namespace.return_value = ns
        mock_tpuf_cls.return_value = client
        sink = TurbopufferIngestionSink(namespace="medical_pmc", dry_run=False)
        point = _Point("66666666-6666-6666-6666-666666666666", [0.6], {"title": "x"})

        with self.assertRaises(RuntimeError):
            sink.write_points([point])

        self.assertEqual(ns.write.call_count, 1)


if __name__ == "__main__":
    unittest.main()
