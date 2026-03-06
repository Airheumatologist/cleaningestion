import json
import unittest

from src.query_preprocessor import QueryPreprocessor


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content


class _DummyChoice:
    def __init__(self, content: str):
        self.message = _DummyMessage(content)


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [_DummyChoice(content)]


class QueryPreprocessorTests(unittest.TestCase):
    def _make_preprocessor(self) -> QueryPreprocessor:
        # Bypass full __init__ to avoid external provider setup in unit tests.
        preprocessor = QueryPreprocessor.__new__(QueryPreprocessor)
        preprocessor.use_entity_expansion = False
        preprocessor._entity_expander = None
        preprocessor.llm_provider = "deepinfra"
        preprocessor.model = "test-model"
        preprocessor.retry_count = 0
        preprocessor.retry_delay = 0.0
        preprocessor.expansion_count = 2
        return preprocessor

    def test_decompose_uses_minimal_schema_and_two_retrieval_queries(self):
        preprocessor = self._make_preprocessor()
        payload = {
            "corrected_query": "",
            "primary_query": "inguinal lymphadenopathy fever differential diagnosis",
            "keyword_query": "inguinal lymphadenopathy cat scratch disease pyogenic diagnosis",
            "key_entities": ["inguinal lymphadenopathy", "fever"],
            "corrected_entities": [],
            "is_drug_query": False,
            "drug_names": [],
        }
        preprocessor._chat_completion_with_retry = lambda messages, operation_name: _DummyResponse(json.dumps(payload))

        result = preprocessor.decompose_query("What is the most likely diagnosis?")

        self.assertEqual(result.primary_query, payload["primary_query"])
        self.assertEqual(result.keyword_query, payload["keyword_query"])
        self.assertEqual(result.retrieval_queries, [payload["primary_query"], payload["keyword_query"]])
        self.assertFalse(result.decomposed.is_drug_query)

    def test_decompose_maps_legacy_payload_keys(self):
        preprocessor = self._make_preprocessor()
        payload = {
            "corrected_query": "Management of neurobrucellosis",
            "rewritten_query": "management neurobrucellosis diagnosis",
            "rewritten_query_for_keyword_search": "NUEROBROCELLOSIS management treatment",
            "medical_conditions": ["NUEROBROCELLOSIS"],
            "corrected_medical_conditions": ["neurobrucellosis"],
            "is_drug_query": False,
            "drug_names": [],
        }
        preprocessor._chat_completion_with_retry = lambda messages, operation_name: _DummyResponse(json.dumps(payload))

        result = preprocessor.decompose_query("Management of NUEROBROCELLOSIS")

        self.assertEqual(result.decomposed.primary_query, "management neurobrucellosis diagnosis")
        self.assertEqual(result.decomposed.key_entities, ["NUEROBROCELLOSIS"])
        self.assertEqual(result.decomposed.corrected_entities, ["neurobrucellosis"])
        self.assertIn("neurobrucellosis", result.keyword_query)

    def test_fallback_compacts_long_vignette_and_strips_options(self):
        preprocessor = self._make_preprocessor()
        preprocessor._chat_completion_with_retry = lambda messages, operation_name: (_ for _ in ()).throw(RuntimeError("llm down"))

        long_prefix = " ".join(["history"] * 130)
        query = (
            f"{long_prefix} A 68-year-old woman presents with fever chills painful groin swelling and leukocytosis. "
            "What is the most likely diagnosis? (A) Lymphogranuloma venereum (B) Cat scratch disease "
            "(C) Pyogenic lymphadenitis (D) Inguinal hernia (E) Necrotizing fasciitis"
        )

        result = preprocessor.decompose_query(query)

        self.assertLessEqual(len(result.primary_query.split()), 25)
        self.assertNotIn("(A)", result.primary_query)
        self.assertIn("cat scratch disease", result.keyword_query)
        self.assertFalse(result.decomposed.is_drug_query)


if __name__ == "__main__":
    unittest.main()
