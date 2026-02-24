"""
Persistent query cache for Medical RAG Pipeline.
Uses SQLite to store and retrieve past queries and their responses.
"""

import json
import sqlite3
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict

from .config import (
    QUERY_CACHE_ENABLED,
    QUERY_CACHE_DIR,
    QUERY_CACHE_EXPIRY_DAYS,
    QUERY_CACHE_NAMESPACE,
    QUERY_CACHE_KEY_VERSION,
)

logger = logging.getLogger(__name__)

class QueryCache:
    """
    Persistent SQLite-based cache for RAG pipeline responses.
    """
    
    def __init__(
        self,
        cache_dir: str = QUERY_CACHE_DIR,
        expiry_days: int = QUERY_CACHE_EXPIRY_DAYS,
        namespace: str = QUERY_CACHE_NAMESPACE,
        key_version: int = QUERY_CACHE_KEY_VERSION,
        enabled: Optional[bool] = None,
    ):
        self.enabled = QUERY_CACHE_ENABLED if enabled is None else enabled
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / "query_cache.db"
        self.expiry_days = expiry_days
        self.namespace = namespace.strip() or "default"
        self.key_version = key_version
        
        if self.enabled:
            self._init_db()
            
    def _init_db(self):
        """Initialize the SQLite database and create tables if they don't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        query TEXT,
                        response TEXT,
                        timestamp DATETIME,
                        hit_count INTEGER DEFAULT 0
                    )
                """)
                # Create index on timestamp for efficient pruning
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
            logger.info(f"✅ Query cache initialized at {self.db_path}")
            self.prune_expired()
        except Exception as e:
            logger.error(f"❌ Failed to initialize query cache: {e}")
            self.enabled = False

    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection configured for concurrent API access."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _generate_key(self, query: str, **kwargs) -> str:
        """Generate a unique SHA-256 hash for the query and its parameters."""
        # Normalize query: strip whitespace and lowercase
        normalized_query = query.strip().lower()

        key_payload = {
            "namespace": self.namespace,
            "key_version": self.key_version,
            "query": normalized_query,
            "params": kwargs,
        }
        key_json = json.dumps(key_payload, sort_keys=True, default=str)
        return hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if it exists and is not expired."""
        if not self.enabled:
            return None
            
        key = self._generate_key(query, **kwargs)
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "SELECT response, timestamp FROM cache WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    response_json, timestamp_str = row
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    # Check if expired
                    if datetime.now() - timestamp > timedelta(days=self.expiry_days):
                        logger.info(f"⏳ Cache entry for '{query[:30]}...' expired")
                        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                        return None
                    
                    # Update hit count
                    conn.execute(
                        "UPDATE cache SET hit_count = hit_count + 1 WHERE key = ?", (key,)
                    )
                    
                    logger.info(f"🎯 Cache hit for query: '{query[:50]}...'")
                    return json.loads(response_json)
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            
        return None

    def set(self, query: str, response: Dict[str, Any], **kwargs):
        """Store a response in the cache."""
        if not self.enabled:
            return
            
        key = self._generate_key(query, **kwargs)
        try:
            # Re-use serialization logic to handle sets/nan/etc.
            # We'll use a local simplified version or import from api_server if safe
            serialized_response = self._serialize(response)
            
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, query, response, timestamp, hit_count)
                    VALUES (?, ?, ?, ?, COALESCE((SELECT hit_count FROM cache WHERE key = ?), 0))
                    """,
                    (key, query, serialized_response, datetime.now().isoformat(), key)
                )
            logger.debug(f"💾 Cached response for query: '{query[:50]}...'")
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    def delete(self, key: str):
        """Delete a specific entry from the cache."""
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")

    def prune_expired(self):
        """Remove all expired entries from the cache."""
        if not self.enabled:
            return
        try:
            expiry_date = (datetime.now() - timedelta(days=self.expiry_days)).isoformat()
            with self._connect() as conn:
                cursor = conn.execute("DELETE FROM cache WHERE timestamp < ?", (expiry_date,))
                if cursor.rowcount > 0:
                    logger.info(f"🧹 Pruned {cursor.rowcount} expired cache entries")
        except Exception as e:
            logger.error(f"Error pruning cache: {e}")

    def _serialize(self, obj: Any) -> str:
        """Helper to serialize object to JSON string, handling special types."""
        def default(item):
            import math
            try:
                import numpy as np
            except Exception:
                np = None

            if isinstance(item, datetime):
                return item.isoformat()
            if isinstance(item, Path):
                return str(item)
            if isinstance(item, (set, tuple)):
                return list(item)
            if hasattr(item, 'to_dict'):
                return item.to_dict()
            if isinstance(item, float):
                if math.isnan(item) or math.isinf(item):
                    return None
                return float(item)
            if np is not None and isinstance(item, np.floating):
                if math.isnan(item) or math.isinf(item):
                    return None
                return float(item)
            if np is not None and isinstance(item, np.integer):
                return int(item)
            if np is not None and isinstance(item, np.ndarray):
                return item.tolist()
            return str(item)
            
        return json.dumps(obj, default=default)
