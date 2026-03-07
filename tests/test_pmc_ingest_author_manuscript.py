import importlib.util
import sys
import threading
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
PMC_INGEST_PATH = REPO_ROOT / "scripts" / "06_ingest_pmc.py"


def load_pmc_module():
    scripts_dir = str(REPO_ROOT / "scripts")
    added_scripts_path = False
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
        added_scripts_path = True

    try:
        module_name = f"pmc_ingest_test_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, str(PMC_INGEST_PATH))
        module = importlib.util.module_from_spec(spec)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load PMC ingest module spec")
        spec.loader.exec_module(module)
        return module
    finally:
        if added_scripts_path:
            try:
                sys.path.remove(scripts_dir)
            except ValueError:
                pass


class PmcAuthorManuscriptIngestionTests(unittest.TestCase):
    def test_process_batch_ingests_author_manuscript_inputs(self):
        module = load_pmc_module()

        with TemporaryDirectory() as tmp:
            xml_root = Path(tmp)
            xml_path = xml_root / "author_manuscript" / "PMC123.xml"
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            xml_path.write_text("<article/>", encoding="utf-8")

            parse_mock = mock.Mock(return_value={"pmcid": "PMC123"})

            with (
                mock.patch("ingestion_utils.parse_pmc_xml", parse_mock),
                mock.patch.object(module, "build_points", return_value=([object()], ["chunk"])),
                mock.patch.object(module, "upsert_with_retry"),
                mock.patch.object(module, "append_checkpoint") as append_mock,
            ):
                inserted, _skipped = module.process_batch(
                    client=mock.Mock(),
                    batch_files=[xml_path],
                    embedding_provider=mock.Mock(),
                    processed_ids=set(),
                    processed_lock=threading.Lock(),
                    xml_root=xml_root,
                    sparse_encoder=None,
                    delete_source=False,
                )

            self.assertEqual(inserted, 1)
            kwargs = parse_mock.call_args.kwargs
            self.assertFalse(kwargs["require_pmid"])
            self.assertFalse(kwargs["require_open_access"])
            self.assertFalse(kwargs["require_commercial_license"])
            checkpoint_ids = append_mock.call_args.args[0]
            self.assertEqual(checkpoint_ids, ["pmc_author_manuscript:PMC123"])


if __name__ == "__main__":
    unittest.main()
