import importlib.util
import sys
import threading
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
PMC_S3_INGEST_PATH = REPO_ROOT / "scripts" / "06_ingest_pmc_s3.py"


def load_pmc_s3_module():
    scripts_dir = str(REPO_ROOT / "scripts")
    added_scripts_path = False
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
        added_scripts_path = True

    try:
        module_name = f"pmc_s3_ingest_test_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, str(PMC_S3_INGEST_PATH))
        module = importlib.util.module_from_spec(spec)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load PMC S3 ingest module spec")
        spec.loader.exec_module(module)
        return module
    finally:
        if added_scripts_path:
            try:
                sys.path.remove(scripts_dir)
            except ValueError:
                pass


class PmcS3IngestionReliabilityTests(unittest.TestCase):
    def _process_entry(
        self,
        module,
        downloader,
        metadata_key,
        etag,
        datasets,
        processed_ids,
        skipped_ids,
        inflight_ids,
        checkpoint_lock,
        *,
        failed_metadata_keys_file,
        skip_checkpoint_file=None,
        skip_checkpoint_lock=None,
        xml_fetch_timeout_seconds=10,
        xml_fetch_retries=4,
        xml_fetch_backoff_factor=0.25,
    ):
        return module._process_metadata_entry(
            downloader,
            metadata_key,
            etag,
            datasets,
            processed_ids,
            skipped_ids,
            inflight_ids,
            checkpoint_lock,
            xml_fetch_timeout_seconds=xml_fetch_timeout_seconds,
            xml_fetch_retries=xml_fetch_retries,
            xml_fetch_backoff_factor=xml_fetch_backoff_factor,
            failed_metadata_keys_file=failed_metadata_keys_file,
            failed_metadata_keys_lock=threading.Lock(),
            skip_checkpoint_file=skip_checkpoint_file,
            skip_checkpoint_lock=skip_checkpoint_lock or threading.Lock(),
        )

    def test_download_xml_bytes_uses_configured_session_and_timeout(self):
        module = load_pmc_s3_module()
        response = mock.Mock()
        response.content = b"<article/>"
        session = mock.Mock()
        session.get.return_value = response

        with mock.patch.object(module, "_get_xml_download_session", return_value=session) as session_mock:
            xml_bytes = module._download_xml_bytes(
                "https://example.org/article.xml",
                timeout_seconds=17,
                max_retries=5,
                backoff_factor=1.75,
            )

        self.assertEqual(xml_bytes, b"<article/>")
        session_mock.assert_called_once_with(max_retries=5, backoff_factor=1.75)
        session.get.assert_called_once_with("https://example.org/article.xml", timeout=17)
        response.raise_for_status.assert_called_once()

    def test_process_metadata_entry_records_early_metadata_skips(self):
        module = load_pmc_s3_module()
        with TemporaryDirectory() as tmp:
            failed_keys_file = Path(tmp) / "failed.tsv"
            skip_checkpoint_file = Path(tmp) / "skipped.tsv"
            downloader = mock.Mock()
            downloader._download_metadata_json.return_value = {
                "is_pmc_openaccess": True,
                "license": "CC-BY-NC 4.0",
                "xml_url": "s3://pmc-oa-opendata/articles/PMC1.xml",
            }
            downloader._normalize_s3_or_https_url.return_value = "https://example.org/PMC.xml"
            with mock.patch.object(module, "_download_xml_bytes") as download_mock:
                with mock.patch.object(module.logger, "log") as log_mock:
                    processed_ids: set[str] = set()
                    skipped_ids: set[str] = set()
                    result = self._process_entry(
                        module,
                        downloader,
                        "metadata/b.json",
                        "etag-1",
                        ["pmc_oa"],
                        processed_ids,
                        skipped_ids,
                        set(),
                        threading.Lock(),
                        failed_metadata_keys_file=failed_keys_file,
                        skip_checkpoint_file=skip_checkpoint_file,
                    )

            self.assertIsNone(result)
            download_mock.assert_not_called()
            self.assertEqual(len(processed_ids), 1)
            self.assertEqual(len(skipped_ids), 1)
            self.assertEqual(
                skip_checkpoint_file.read_text(encoding="utf-8"),
                "pmc_oa:metadata/b.json:etag-1\tmetadata filter\t"
                "metadata field license marks a non-commercial license (CC-BY-NC 4.0)\n",
            )
            self.assertEqual(log_mock.call_args.args[0], module.logging.INFO)
            self.assertEqual(log_mock.call_args.args[1], "Skipping metadata entry %s at %s stage: %s")
            self.assertEqual(log_mock.call_args.args[2], "metadata/b.json")
            self.assertEqual(log_mock.call_args.args[3], "metadata filter")
            self.assertEqual(
                log_mock.call_args.args[4],
                "metadata field license marks a non-commercial license (CC-BY-NC 4.0)",
            )

            source_filter_downloader = mock.Mock()
            source_filter_downloader._download_metadata_json.return_value = {
                "is_pmc_openaccess": False,
                "xml_url": "s3://pmc-oa-opendata/articles/PMC0.xml",
            }
            source_filter_downloader._normalize_s3_or_https_url.return_value = "https://example.org/PMC0.xml"
            with mock.patch.object(module, "_download_xml_bytes") as source_download_mock:
                with mock.patch.object(module.logger, "log") as source_log_mock:
                    source_result = self._process_entry(
                        module,
                        source_filter_downloader,
                        "metadata/a.json",
                        "etag-2",
                        ["pmc_oa"],
                        set(),
                        set(),
                        set(),
                        threading.Lock(),
                        failed_metadata_keys_file=failed_keys_file,
                    )

            self.assertIsNone(source_result)
            source_download_mock.assert_not_called()
            self.assertEqual(source_log_mock.call_args.args[0], module.logging.INFO)
            self.assertEqual(source_log_mock.call_args.args[3], "source filter")
            self.assertEqual(
                source_log_mock.call_args.args[4],
                "metadata does not match requested datasets=pmc_oa",
            )

    def test_process_metadata_entry_records_xml_download_failures(self):
        module = load_pmc_s3_module()
        downloader = mock.Mock()
        downloader._download_metadata_json.return_value = {
            "is_pmc_openaccess": True,
            "xml_url": "s3://pmc-oa-opendata/articles/PMC1.xml",
        }
        downloader._normalize_s3_or_https_url.return_value = "https://example.org/PMC1.xml"
        inflight_ids: set[str] = set()

        with TemporaryDirectory() as tmp:
            failed_keys_file = Path(tmp) / "failed.tsv"
            with (
                mock.patch.object(
                    module,
                    "_download_xml_bytes",
                    side_effect=requests.exceptions.Timeout("timed out"),
                ),
                mock.patch.object(module.logger, "log") as log_mock,
            ):
                result = self._process_entry(
                    module,
                    downloader,
                    "metadata/c.json",
                    "etag-2",
                    ["pmc_oa"],
                    set(),
                    set(),
                    inflight_ids,
                    threading.Lock(),
                    xml_fetch_timeout_seconds=10,
                    xml_fetch_retries=4,
                    xml_fetch_backoff_factor=0.25,
                    failed_metadata_keys_file=failed_keys_file,
                )
            file_contents = failed_keys_file.read_text(encoding="utf-8")

        self.assertIsNone(result)
        self.assertEqual(inflight_ids, set())
        self.assertEqual(
            file_contents,
            "metadata/c.json\txml download\ttimed out\n",
        )
        self.assertEqual(log_mock.call_args.args[0], module.logging.WARNING)
        self.assertEqual(log_mock.call_args.args[1], "Skipping metadata entry %s at %s stage: %s")
        self.assertEqual(log_mock.call_args.args[3], "xml download")

    def test_process_metadata_entry_records_parse_failures(self):
        module = load_pmc_s3_module()
        downloader = mock.Mock()
        downloader._download_metadata_json.return_value = {
            "is_manuscript": True,
            "xml_url": "https://example.org/PMC2.xml",
        }
        downloader._normalize_s3_or_https_url.return_value = "https://example.org/PMC2.xml"

        with TemporaryDirectory() as tmp:
            failed_keys_file = Path(tmp) / "failed.tsv"
            skip_checkpoint_file = Path(tmp) / "skipped.tsv"
            with (
                mock.patch.object(module, "_download_xml_bytes", return_value=b"<article/>") as download_mock,
                mock.patch.object(module, "parse_pmc_xml_bytes", return_value=None) as parse_mock,
            ):
                processed_ids: set[str] = set()
                skipped_ids: set[str] = set()
                result = self._process_entry(
                    module,
                    downloader,
                    "metadata/d.json",
                    "etag-3",
                    ["author_manuscript"],
                    processed_ids,
                    skipped_ids,
                    set(),
                    threading.Lock(),
                    failed_metadata_keys_file=failed_keys_file,
                    skip_checkpoint_file=skip_checkpoint_file,
                )
            file_contents = failed_keys_file.read_text(encoding="utf-8")
            skip_contents = skip_checkpoint_file.read_text(encoding="utf-8")

        self.assertIsNone(result)
        self.assertEqual(len(processed_ids), 1)
        self.assertEqual(len(skipped_ids), 1)
        self.assertEqual(
            file_contents,
            "metadata/d.json\tparse\tPMC XML parser returned no article\n",
        )
        self.assertEqual(
            skip_contents,
            "pmc_author_manuscript:metadata/d.json:etag-3\tparse\tPMC XML parser returned no article\n",
        )
        self.assertEqual(parse_mock.call_count, 1)
        self.assertEqual(download_mock.call_count, 1)

        with (
            mock.patch.object(module, "_download_xml_bytes") as second_download_mock,
            mock.patch.object(module, "parse_pmc_xml_bytes") as second_parse_mock,
        ):
            second_result = self._process_entry(
                module,
                downloader,
                "metadata/d.json",
                "etag-3",
                ["author_manuscript"],
                processed_ids,
                skipped_ids,
                set(),
                threading.Lock(),
                failed_metadata_keys_file=failed_keys_file,
                skip_checkpoint_file=skip_checkpoint_file,
            )

        self.assertIsNone(second_result)
        second_download_mock.assert_not_called()
        second_parse_mock.assert_not_called()

    def test_compute_namespace_shards_defaults_to_single_namespace(self):
        module = load_pmc_s3_module()
        shards = module._compute_namespace_shards(
            base_namespace="medical_pmc",
            shard_count=1,
            pattern="{base}_shard_{shard}",
        )
        self.assertEqual(shards, ["medical_pmc"])

    def test_write_points_routes_by_doc_id_across_shards(self):
        module = load_pmc_s3_module()

        class _Point:
            def __init__(self, doc_id: str, point_id: str):
                self.id = point_id
                self.payload = {"doc_id": doc_id}

        sink = mock.Mock()
        shard_0 = mock.Mock()
        shard_1 = mock.Mock()
        shard_0.write_points.side_effect = lambda pts: len(pts)
        shard_1.write_points.side_effect = lambda pts: len(pts)

        points = [
            _Point("doc-alpha", "p1"),
            _Point("doc-alpha", "p2"),
            _Point("doc-beta", "p3"),
        ]
        written = module._write_points(
            points=points,
            checkpoint_ids=["ck-a", "ck-b"],
            sink=sink,
            shard_sinks=[shard_0, shard_1],
        )

        self.assertEqual(written, 3)
        sink.write_points.assert_not_called()
        doc_alpha_shard = module._stable_shard_index("doc-alpha", 2)
        doc_beta_shard = module._stable_shard_index("doc-beta", 2)
        shard_calls = [
            shard_0.write_points.call_args_list[0].args[0] if shard_0.write_points.call_args_list else [],
            shard_1.write_points.call_args_list[0].args[0] if shard_1.write_points.call_args_list else [],
        ]
        self.assertEqual(len(shard_calls[doc_alpha_shard]), 2)
        self.assertEqual(len(shard_calls[doc_beta_shard]), 1)

    def test_write_points_uses_checkpoint_fallback_without_doc_id(self):
        module = load_pmc_s3_module()

        class _Point:
            def __init__(self, point_id: str):
                self.id = point_id
                self.payload = {}

        sink = mock.Mock()
        shard_0 = mock.Mock()
        shard_1 = mock.Mock()
        shard_0.write_points.side_effect = lambda pts: len(pts)
        shard_1.write_points.side_effect = lambda pts: len(pts)

        written = module._write_points(
            points=[_Point("p1")],
            checkpoint_ids=["ck-fallback"],
            sink=sink,
            shard_sinks=[shard_0, shard_1],
        )
        self.assertEqual(written, 1)
        fallback_shard = module._stable_shard_index("ck-fallback:0", 2)
        target = shard_0 if fallback_shard == 0 else shard_1
        other = shard_1 if fallback_shard == 0 else shard_0
        target.write_points.assert_called_once()
        other.write_points.assert_not_called()

    def test_write_points_single_namespace_passthrough(self):
        module = load_pmc_s3_module()
        point = mock.Mock()
        point.payload = {"doc_id": "doc-one"}
        sink = mock.Mock()
        sink.write_points.return_value = 1

        written = module._write_points(
            points=[point],
            checkpoint_ids=["ck-1"],
            sink=sink,
            shard_sinks=None,
        )
        self.assertEqual(written, 1)
        sink.write_points.assert_called_once_with([point])

    def test_get_xml_download_session_configures_transient_retry_policy(self):
        module = load_pmc_s3_module()
        session = module._get_xml_download_session(max_retries=3, backoff_factor=0.75)
        https_adapter = session.get_adapter("https://example.org/article.xml")

        self.assertEqual(https_adapter.max_retries.total, 3)
        self.assertEqual(https_adapter.max_retries.connect, 3)
        self.assertEqual(https_adapter.max_retries.read, 3)
        self.assertEqual(https_adapter.max_retries.status, 3)
        self.assertEqual(https_adapter.max_retries.backoff_factor, 0.75)
        self.assertEqual(https_adapter.max_retries.allowed_methods, frozenset({"GET"}))
        self.assertEqual(
            set(https_adapter.max_retries.status_forcelist),
            {408, 429, 500, 502, 503, 504},
        )


if __name__ == "__main__":
    unittest.main()
