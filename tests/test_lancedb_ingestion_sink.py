import tempfile
import unittest
from unittest import mock
import sys
from pathlib import Path
import subprocess

import lancedb

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from lancedb_ingestion_sink import LanceDBIngestionSink


class _Sparse:
    def __init__(self):
        self.indices = [1, 4]
        self.values = [0.8, 0.2]


class _Point:
    def __init__(self):
        self.id = "pt-1"
        self.vector = {"dense": [0.1, 0.2, 0.3], "sparse": _Sparse()}
        self.payload = {
            "doc_id": "doc-1",
            "pmcid": "PMC1",
            "chunk_id": "chunk-1",
            "page_content": "clinical text",
            "source": "pubmed_abstract",
            "source_family": "pubmed",
            "year": 2024,
            "article_type": "review",
        }


class LanceDBIngestionSinkTests(unittest.TestCase):
    def test_sink_writes_rows_to_lancedb(self):
        with tempfile.TemporaryDirectory(prefix="lancedb-sink-test-") as tmp:
            sink = LanceDBIngestionSink(uri=tmp, table_name="medical_docs", dry_run=False)
            written = sink.write_points([_Point()])
            self.assertEqual(written, 1)

            db = lancedb.connect(tmp)
            table = db.open_table("medical_docs")
            rows = table.head(5).to_pylist()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["doc_id"], "doc-1")
            self.assertEqual(rows[0]["point_id"], "pt-1")

    @mock.patch("lancedb_ingestion_sink.subprocess.run")
    def test_reindex_command_places_global_flags_before_subcommand(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"profile":"ivf_rq"}',
            stderr="",
        )
        with tempfile.TemporaryDirectory(prefix="lancedb-sink-reindex-test-") as tmp:
            sink = LanceDBIngestionSink(
                uri=tmp,
                table_name="medical_docs",
                dry_run=False,
                reindex_interval_batches=1,
            )
            sink.write_points([_Point()])

        self.assertTrue(mock_run.called)
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[:2], ["python3", "scripts/lancedb_index_manager.py"])
        self.assertIn("--json", cmd)
        self.assertIn("build", cmd)
        self.assertLess(cmd.index("--json"), cmd.index("build"))


if __name__ == "__main__":
    unittest.main()
