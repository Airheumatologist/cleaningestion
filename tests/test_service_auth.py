import json
import tempfile
import unittest
from pathlib import Path

from src.service_auth import ServiceTokenStore, hash_service_token


class ServiceTokenStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.keys_file = Path(self.tmpdir.name) / "api_keys.json"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_keys(self, payload):
        self.keys_file.write_text(json.dumps(payload), encoding="utf-8")

    def test_valid_enabled_token(self):
        plain = "token-enabled"
        self._write_keys(
            {
                "tokens": [
                    {
                        "service_id": "svc-enabled",
                        "token_hash": hash_service_token(plain),
                        "enabled": True,
                    }
                ]
            }
        )
        store = ServiceTokenStore(str(self.keys_file), cache_ttl_seconds=1)
        self.assertEqual(
            store.validate_token(plain),
            (True, "svc-enabled", True),
        )

    def test_disabled_token_returns_forbidden_signal(self):
        plain = "token-disabled"
        self._write_keys(
            {
                "tokens": [
                    {
                        "service_id": "svc-disabled",
                        "token_hash": hash_service_token(plain),
                        "enabled": False,
                    }
                ]
            }
        )
        store = ServiceTokenStore(str(self.keys_file), cache_ttl_seconds=1)
        self.assertEqual(
            store.validate_token(plain),
            (False, "svc-disabled", False),
        )

    def test_invalid_token(self):
        self._write_keys(
            {
                "tokens": [
                    {
                        "service_id": "svc-a",
                        "token_hash": hash_service_token("known-token"),
                        "enabled": True,
                    }
                ]
            }
        )
        store = ServiceTokenStore(str(self.keys_file), cache_ttl_seconds=1)
        self.assertEqual(store.validate_token("wrong-token"), (False, None, None))


if __name__ == "__main__":
    unittest.main()
