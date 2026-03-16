import json
import subprocess
import sys
import unittest
from pathlib import Path


class LanceDbSchemaParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.script = cls.repo_root / "scripts" / "lancedb_schema_parity.py"
        cls.scripts_dir = cls.repo_root / "scripts"
        if str(cls.scripts_dir) not in sys.path:
            sys.path.insert(0, str(cls.scripts_dir))
        import lancedb_schema_parity  # noqa: PLC0415

        cls.parity_module = lancedb_schema_parity

    def test_schema_parity_contract(self):
        result = subprocess.run(
            [sys.executable, str(self.script), "--json"],
            capture_output=True,
            text=True,
            check=True,
            cwd=self.repo_root,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("errors"), [])
        self.assertEqual(payload.get("vector_validation", {}).get("status"), "ok")
        self.assertIn("pubmed_abstract", payload.get("source_counts", {}))
        self.assertIn("dailymed", payload.get("source_counts", {}))

    def test_observed_contract_contains_vector_critical_fields(self):
        rows_by_source = self.parity_module.collect_rows_by_source()
        observed_contract = self.parity_module.build_contract(rows_by_source)
        required_fields = {"point_id", "vector", "sparse_indices", "sparse_values"}

        for source, source_info in observed_contract.get("sources", {}).items():
            fields = set(source_info.get("fields", {}).keys())
            self.assertTrue(required_fields.issubset(fields), f"missing vector fields for {source}")

    def test_vector_dimension_mismatch_detected(self):
        rows_by_source = {
            "pubmed_abstract": [{"vector": [0.1] * 256, "source": "pubmed_abstract"}],
        }
        manifest = {
            "vector_column": "vector",
            "profiles": {"ivf_rq": {"num_sub_vectors": 64}},
        }
        validation = self.parity_module.validate_vector_configuration(
            rows_by_source=rows_by_source,
            expected_vector_dim=1024,
            runtime_vector_dim=1024,
            manifest=manifest,
        )
        self.assertEqual(validation.get("status"), "failed")
        self.assertTrue(any("vector dimension mismatch" in err for err in validation.get("errors", [])))

    def test_profile_incompatibility_detected(self):
        rows_by_source = {
            "pubmed_abstract": [{"vector": [0.1] * 1000, "source": "pubmed_abstract"}],
        }
        manifest = {
            "vector_column": "vector",
            "profiles": {"ivf_rq": {"num_sub_vectors": 64}},
        }
        validation = self.parity_module.validate_vector_configuration(
            rows_by_source=rows_by_source,
            expected_vector_dim=1000,
            runtime_vector_dim=1000,
            manifest=manifest,
        )
        self.assertEqual(validation.get("status"), "failed")
        self.assertTrue(any("num_sub_vectors" in err for err in validation.get("errors", [])))


if __name__ == "__main__":
    unittest.main()
