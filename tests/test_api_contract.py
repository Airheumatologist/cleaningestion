import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.service_auth import hash_service_token


class DummyDecomposed:
    def model_dump(self):
        return {"field_of_study": "Medicine"}


class DummyProcessed:
    def __init__(self):
        self.original_query = "original"
        self.rewritten_query = "rewritten"
        self.keyword_query = "keyword"
        self.search_filters = {}
        self.decomposed = DummyDecomposed()


class DummyPipeline:
    def answer_scholarqa_style(self, query: str):
        return {
            "query": query,
            "report_title": "Test Report",
            "answer": "Test answer",
            "sections": [],
            "sources": [],
            "evidence_hierarchy": {},
            "full_text_articles": [],
            "retrieval_stats": {},
            "status": "success",
        }

    def answer_streaming(self, _: str):
        yield {"step": "generation", "status": "running", "token": "Hello"}
        yield {"step": "complete", "status": "success", "answer": "Hello"}

    def preprocess_query(self, _: str):
        return DummyProcessed()


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.keys_file = Path(cls.tmpdir.name) / "api_keys.json"

        cls.valid_token = "valid-token"
        cls.disabled_token = "disabled-token"
        cls.keys_file.write_text(
            json.dumps(
                {
                    "tokens": [
                        {
                            "service_id": "svc-enabled",
                            "token_hash": hash_service_token(cls.valid_token),
                            "enabled": True,
                        },
                        {
                            "service_id": "svc-disabled",
                            "token_hash": hash_service_token(cls.disabled_token),
                            "enabled": False,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        os.environ["SKIP_PIPELINE_INIT"] = "1"
        os.environ["API_AUTH_ENABLED"] = "true"
        os.environ["API_KEYS_FILE"] = str(cls.keys_file)
        os.environ["API_KEYS_CACHE_TTL_SECONDS"] = "1"

        for module_name in ("src.config", "src.api_server"):
            if module_name in sys.modules:
                del sys.modules[module_name]

        cls.api_server = importlib.import_module("src.api_server")
        cls.api_server.pipeline = DummyPipeline()
        cls.client = TestClient(cls.api_server.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.tmpdir.cleanup()

    def _auth_header(self, token: str):
        return {"Authorization": f"Bearer {token}"}

    def test_health_endpoints_are_public(self):
        self.assertEqual(self.client.get("/api/v1/health").status_code, 200)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_missing_token_returns_401(self):
        res = self.client.post("/api/v1/chat", json={"query": "hi"})
        self.assertEqual(res.status_code, 401)

    def test_invalid_token_returns_401(self):
        res = self.client.post(
            "/api/v1/chat",
            headers=self._auth_header("wrong-token"),
            json={"query": "hi"},
        )
        self.assertEqual(res.status_code, 401)

    def test_disabled_token_returns_403(self):
        res = self.client.post(
            "/api/v1/chat",
            headers=self._auth_header(self.disabled_token),
            json={"query": "hi"},
        )
        self.assertEqual(res.status_code, 403)

    def test_versioned_and_legacy_chat_routes(self):
        headers = self._auth_header(self.valid_token)
        v1 = self.client.post("/api/v1/chat", headers=headers, json={"query": "hi"})
        self.assertEqual(v1.status_code, 200)
        self.assertEqual(v1.json()["answer"], "Test answer")

        legacy = self.client.post("/api/chat", headers=headers, json={"query": "hi"})
        self.assertEqual(legacy.status_code, 200)
        self.assertEqual(legacy.json()["status"], "success")

    def test_debug_decompose_route_requires_auth(self):
        missing = self.client.post("/api/v1/debug/decompose", json={"query": "hi"})
        self.assertEqual(missing.status_code, 401)

        ok = self.client.post(
            "/api/v1/debug/decompose",
            headers=self._auth_header(self.valid_token),
            json={"query": "hi"},
        )
        self.assertEqual(ok.status_code, 200)
        self.assertIn("rewritten_query", ok.json())

    def test_stream_route_and_legacy_alias(self):
        headers = self._auth_header(self.valid_token)
        res = self.client.post(
            "/api/v1/chat/stream",
            headers=headers,
            json={"query": "stream", "stream": True},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("data: [DONE]", res.text)

        legacy = self.client.post(
            "/api/chat/stream",
            headers=headers,
            json={"query": "stream", "stream": True},
        )
        self.assertEqual(legacy.status_code, 200)
        self.assertIn("data: [DONE]", legacy.text)

    def test_openapi_is_versioned(self):
        res = self.client.get("/api/v1/openapi.json")
        self.assertEqual(res.status_code, 200)
        self.assertIn("/api/v1/chat", res.json()["paths"])


if __name__ == "__main__":
    unittest.main()
