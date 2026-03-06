import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class DocsAlignmentTests(unittest.TestCase):
    def test_env_example_documents_split_embedding_vars(self):
        content = (REPO_ROOT / "env.example").read_text(encoding="utf-8")
        self.assertIn("RUNTIME_EMBEDDING_MODEL=", content)
        self.assertIn("INGESTION_EMBEDDING_MODEL=", content)
        self.assertIn("EMBEDDING_MODEL=", content)

    def test_readme_documents_weekly_lookup_refresh_and_embedding_precedence(self):
        content = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("04_prepare_dailymed_updates.py", content)
        self.assertIn("Weekly update behavior (`scripts/08_weekly_update.py`):", content)
        self.assertIn("generate_drug_lookup.py", content)
        self.assertIn("Embedding precedence:", content)
        self.assertIn("indexing_threshold=10000", content)

    def test_cursorrules_documents_groq_canonical_policy(self):
        content = (REPO_ROOT / ".cursorrules").read_text(encoding="utf-8")
        self.assertIn("Groq", content)
        self.assertIn("canonical runtime LLM provider", content)
        self.assertIn("INGESTION_EMBEDDING_MODEL", content)


if __name__ == "__main__":
    unittest.main()
