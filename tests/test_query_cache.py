import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.query_cache import QueryCache


class QueryCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _cache(self, *, namespace: str = "default", key_version: int = 2, expiry_days: int = 30) -> QueryCache:
        return QueryCache(
            cache_dir=self.tmpdir.name,
            expiry_days=expiry_days,
            namespace=namespace,
            key_version=key_version,
            enabled=True,
        )

    def test_serializes_path_objects(self):
        cache = self._cache()
        cache.set("path query", {"path": Path("/tmp/example.txt")})
        cached = cache.get("path query")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["path"], "/tmp/example.txt")

    def test_namespace_isolates_entries(self):
        cache_a = self._cache(namespace="ns-a")
        cache_b = self._cache(namespace="ns-b")
        cache_a.set("same query", {"value": 1})
        self.assertIsNotNone(cache_a.get("same query"))
        self.assertIsNone(cache_b.get("same query"))

    def test_key_version_isolates_entries(self):
        cache_v2 = self._cache(key_version=2)
        cache_v3 = self._cache(key_version=3)
        cache_v2.set("same query", {"value": 2})
        self.assertIsNotNone(cache_v2.get("same query"))
        self.assertIsNone(cache_v3.get("same query"))

    def test_expired_entry_removed_on_read(self):
        cache = self._cache(expiry_days=0)
        cache.set("stale query", {"value": "old"})
        stale_timestamp = (datetime.now() - timedelta(days=2)).isoformat()
        with sqlite3.connect(cache.db_path) as conn:
            conn.execute("UPDATE cache SET timestamp = ?", (stale_timestamp,))

        self.assertIsNone(cache.get("stale query"))
        with sqlite3.connect(cache.db_path) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        self.assertEqual(remaining, 0)


if __name__ == "__main__":
    unittest.main()
