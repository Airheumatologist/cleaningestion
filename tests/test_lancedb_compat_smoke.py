import json
import subprocess
import sys
import unittest
from pathlib import Path


class LanceDbCompatSmokeTests(unittest.TestCase):
    def test_smoke_script_passes(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "lancedb_compat_smoke.py"

        result = subprocess.run(
            [sys.executable, str(script), "--json"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )

        payload = json.loads(result.stdout)
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("installed_version"), "0.29.2")
        self.assertTrue(payload.get("hybrid_rrf_ok"))
        self.assertTrue(payload.get("planner_has_ann_and_fts"))


if __name__ == "__main__":
    unittest.main()
