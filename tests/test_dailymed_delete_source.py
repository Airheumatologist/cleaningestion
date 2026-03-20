import importlib.util
import threading
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
DAILYMED_INGEST_PATH = REPO_ROOT / "scripts" / "07_ingest_dailymed.py"


def load_dailymed_module():
    module_name = f"dailymed_ingest_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(DAILYMED_INGEST_PATH))
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load DailyMed ingest module spec")
    spec.loader.exec_module(module)
    return module


class DailyMedDeleteSourceTests(unittest.TestCase):
    def test_delete_source_removes_only_successfully_ingested_xml(self):
        module = load_dailymed_module()

        with TemporaryDirectory() as tmp:
            ok1 = Path(tmp) / "ok1.xml"
            ok2 = Path(tmp) / "ok2.xml"
            bad = Path(tmp) / "bad.xml"
            for path in (ok1, ok2, bad):
                path.write_text("<label/>", encoding="utf-8")

            def parse_side_effect(xml_path: Path):
                if xml_path.name == "bad.xml":
                    return None, "parse_error"
                return {"set_id": xml_path.stem}, "ok"

            def chunks_side_effect(drug, _chunker, validate_chunks=True):
                return [{"text": "content", "chunk_id": f"{drug['set_id']}_chunk", "set_id": drug["set_id"]}]

            processed_ids: set[str] = set()
            lock = threading.Lock()

            with (
                mock.patch.object(module, "_parse_spl_xml_with_status", side_effect=parse_side_effect),
                mock.patch.object(module, "create_chunks", side_effect=chunks_side_effect),
                mock.patch.object(module, "build_points", return_value=([object(), object()], ["a", "b"])),
                mock.patch.object(module, "append_checkpoint_file") as append_mock,
            ):
                sink = mock.Mock()
                sink.write_points.return_value = 2
                inserted, _skipped = module.process_batch(
                    client=mock.Mock(),
                    batch_files=[ok1, ok2, bad],
                    embedding_provider=mock.Mock(),
                    processed_ids=processed_ids,
                    processed_lock=lock,
                    chunker=mock.Mock(),
                    refresh=False,
                    delete_source=True,
                    sink=sink,
                )

            self.assertEqual(inserted, 2)
            sink.write_points.assert_called_once()
            self.assertFalse(ok1.exists())
            self.assertFalse(ok2.exists())
            self.assertTrue(bad.exists())

            checkpoint_ids = append_mock.call_args.args[1]
            self.assertIn("dailymed:ok1", checkpoint_ids)
            self.assertIn("dailymed:ok2", checkpoint_ids)
            self.assertNotIn("dailymed:bad", checkpoint_ids)
            self.assertIn("dailymed:ok1", processed_ids)
            self.assertIn("dailymed:ok2", processed_ids)

    def test_delete_source_does_not_remove_xml_when_upsert_fails(self):
        module = load_dailymed_module()

        with TemporaryDirectory() as tmp:
            ok1 = Path(tmp) / "ok1.xml"
            ok1.write_text("<label/>", encoding="utf-8")

            with (
                mock.patch.object(module, "_parse_spl_xml_with_status", return_value=({"set_id": "ok1"}, "ok")),
                mock.patch.object(module, "create_chunks", return_value=[{"text": "content", "chunk_id": "ok1_chunk", "set_id": "ok1"}]),
                mock.patch.object(module, "build_points", return_value=([object()], ["x"])),
                mock.patch.object(module, "append_checkpoint_file") as append_mock,
            ):
                sink = mock.Mock()
                sink.write_points.side_effect = RuntimeError("upsert failed")
                inserted, skipped = module.process_batch(
                    client=mock.Mock(),
                    batch_files=[ok1],
                    embedding_provider=mock.Mock(),
                    processed_ids=set(),
                    processed_lock=threading.Lock(),
                    chunker=mock.Mock(),
                    refresh=False,
                    delete_source=True,
                    sink=sink,
                )

            self.assertEqual(inserted, 0)
            self.assertEqual(skipped, 0)
            self.assertTrue(ok1.exists())
            append_mock.assert_not_called()

    def test_checkpoint_is_not_appended_when_sink_write_fails(self):
        module = load_dailymed_module()

        with TemporaryDirectory() as tmp:
            ok1 = Path(tmp) / "ok1.xml"
            ok1.write_text("<label/>", encoding="utf-8")
            sink = mock.Mock()
            sink.write_points.side_effect = RuntimeError("turbopuffer write failed")

            with (
                mock.patch.object(module, "_parse_spl_xml_with_status", return_value=({"set_id": "ok1"}, "ok")),
                mock.patch.object(module, "create_chunks", return_value=[{"text": "content", "chunk_id": "ok1_chunk", "set_id": "ok1"}]),
                mock.patch.object(module, "build_points", return_value=([object()], ["x"])),
                mock.patch.object(module, "append_checkpoint_file") as append_mock,
            ):
                inserted, skipped = module.process_batch(
                    client=mock.Mock(),
                    batch_files=[ok1],
                    embedding_provider=mock.Mock(),
                    processed_ids=set(),
                    processed_lock=threading.Lock(),
                    chunker=mock.Mock(),
                    refresh=False,
                    delete_source=False,
                    sink=sink,
                )

            self.assertEqual(inserted, 0)
            self.assertEqual(skipped, 0)
            sink.write_points.assert_called_once()
            append_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
