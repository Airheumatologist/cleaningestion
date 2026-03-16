import subprocess
import unittest
from pathlib import Path


class DualReadCanaryScriptTests(unittest.TestCase):
    def test_cli_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "lancedb_dual_read_canary.py"
        proc = subprocess.run(
            ["python3", str(script), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Dual-read canary", proc.stdout)


if __name__ == "__main__":
    unittest.main()
