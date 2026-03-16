import json
import subprocess
import sys
import unittest
from pathlib import Path


class LanceDbIndexManagerTests(unittest.TestCase):
    def test_build_and_validate_demo(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "lancedb_index_manager.py"

        build_result = subprocess.run(
            [sys.executable, str(script), "--demo", "--json", "build"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        build_payload = json.loads(build_result.stdout)
        self.assertEqual(build_payload.get("status"), "ok")
        self.assertEqual(build_payload.get("profile_name"), "ivf_rq")
        self.assertIn("vector_idx", build_payload.get("index_names", []))

        status_result = subprocess.run(
            [sys.executable, str(script), "--demo", "--json", "status"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        status_payload = json.loads(status_result.stdout)
        self.assertEqual(status_payload.get("status"), "ok")
        self.assertIn("vector_idx", status_payload.get("index_names", []))

        validate_result = subprocess.run(
            [sys.executable, str(script), "--demo", "--json", "validate"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        validate_payload = json.loads(validate_result.stdout)
        self.assertEqual(validate_payload.get("status"), "ok")
        self.assertTrue(validate_payload.get("planner_has_ann"))
        self.assertTrue(validate_payload.get("planner_has_fts"))
        self.assertTrue(validate_payload.get("planner_has_filter"))


if __name__ == "__main__":
    unittest.main()
