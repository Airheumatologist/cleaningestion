import json
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path


class LanceDBPhaseToolsTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_phase5_benchmark_demo(self):
        script = self.repo_root / "scripts" / "lancedb_benchmark_shootout.py"
        proc = subprocess.run(
            ["python3", str(script), "--demo", "--queries", "8", "--concurrency", "2", "--json"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertIn(payload["selected_profile"], {"ivf_rq", "ivf_hnsw_sq"})

    def test_phase8_decommission_audit_outputs_contract(self):
        script = self.repo_root / "scripts" / "lancedb_decommission_audit.py"
        proc = subprocess.run(
            ["python3", str(script), "--json"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertIn(payload["status"], {"ok", "blocked"})
        self.assertIn("blockers", payload)
        self.assertIn("next_steps", payload)

    def test_phase8_decommission_audit_fallback_without_rg(self):
        scripts_dir = self.repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import lancedb_decommission_audit as audit  # noqa: PLC0415

        mock_proc = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="src/example.py:1:qdrant reference\n",
            stderr="",
        )
        with mock.patch.object(audit.shutil, "which", return_value=None), mock.patch.object(
            audit.subprocess, "run", return_value=mock_proc
        ) as mock_run:
            matches = audit._collect_matches(self.repo_root, excludes=[])

        self.assertEqual(len(matches), 1)
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[:4], ["grep", "-R", "-n", "-E"])


if __name__ == "__main__":
    unittest.main()
