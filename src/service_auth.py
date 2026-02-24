"""Service-to-service bearer token authentication helpers."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import bcrypt  # type: ignore
except ImportError:  # pragma: no cover - exercised in minimal local environments
    bcrypt = None
import crypt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceTokenRecord:
    """A single service credential record loaded from JSON."""

    service_id: str
    token_hash: str
    enabled: bool = True


class ServiceTokenStore:
    """Loads and validates service tokens backed by bcrypt hashes."""

    def __init__(self, keys_file: str, cache_ttl_seconds: int = 30):
        self.keys_file = Path(keys_file)
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))
        self._records: list[ServiceTokenRecord] = []
        self._last_load_epoch = 0.0
        self._last_mtime: float | None = None

    def _record_from_dict(self, entry: dict[str, Any]) -> ServiceTokenRecord | None:
        service_id = str(entry.get("service_id", "")).strip()
        token_hash = str(entry.get("token_hash", "")).strip()
        if not service_id or not token_hash:
            return None
        enabled = bool(entry.get("enabled", True))
        return ServiceTokenRecord(service_id=service_id, token_hash=token_hash, enabled=enabled)

    def _normalize_records(self, data: Any) -> Iterable[ServiceTokenRecord]:
        if isinstance(data, dict):
            if isinstance(data.get("tokens"), list):
                for item in data["tokens"]:
                    if isinstance(item, dict):
                        record = self._record_from_dict(item)
                        if record:
                            yield record
                return

            # Optional map format: {"service-a": {"token_hash": "...", "enabled": true}}
            for service_id, value in data.items():
                if not isinstance(value, dict):
                    continue
                merged = {"service_id": service_id, **value}
                record = self._record_from_dict(merged)
                if record:
                    yield record
            return

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    record = self._record_from_dict(item)
                    if record:
                        yield record

    def _reload_if_needed(self) -> None:
        now = time.time()
        if now - self._last_load_epoch < self.cache_ttl_seconds:
            return

        try:
            mtime = self.keys_file.stat().st_mtime
        except FileNotFoundError:
            logger.error("Service token file missing: %s", self.keys_file)
            self._records = []
            self._last_load_epoch = now
            self._last_mtime = None
            return
        except OSError as exc:
            logger.error("Failed to stat service token file %s: %s", self.keys_file, exc)
            self._records = []
            self._last_load_epoch = now
            return

        if self._last_mtime is not None and mtime == self._last_mtime:
            self._last_load_epoch = now
            return

        try:
            data = json.loads(self.keys_file.read_text(encoding="utf-8"))
            records = list(self._normalize_records(data))
            self._records = records
            self._last_mtime = mtime
            self._last_load_epoch = now
            logger.info("Loaded %d service token record(s) from %s", len(records), self.keys_file)
        except Exception as exc:
            logger.error("Failed to parse service token file %s: %s", self.keys_file, exc)
            self._records = []
            self._last_load_epoch = now

    def validate_token(self, plain_token: str) -> tuple[bool, str | None, bool | None]:
        """
        Validate a bearer token against configured bcrypt hashes.

        Returns:
            (is_valid, service_id, enabled_flag_if_matched)
        """
        self._reload_if_needed()
        token_bytes = plain_token.encode("utf-8")
        for record in self._records:
            try:
                if _check_bcrypt_token(token_bytes, record.token_hash):
                    if not record.enabled:
                        return False, record.service_id, False
                    return True, record.service_id, True
            except ValueError:
                logger.warning("Invalid bcrypt hash for service_id=%s", record.service_id)
                continue
        return False, None, None


def hash_service_token(token: str, rounds: int = 12) -> str:
    """Generate a bcrypt hash for a plaintext service token."""
    rounds = max(4, min(16, int(rounds)))
    if bcrypt is not None:
        return bcrypt.hashpw(token.encode("utf-8"), bcrypt.gensalt(rounds)).decode("utf-8")

    rounds_value = 1 << rounds
    if not hasattr(crypt, "METHOD_BLOWFISH"):
        raise RuntimeError("bcrypt hashing backend unavailable (install bcrypt package)")
    salt = crypt.mksalt(crypt.METHOD_BLOWFISH, rounds=rounds_value)
    return crypt.crypt(token, salt)


def _check_bcrypt_token(token_bytes: bytes, token_hash: str) -> bool:
    if bcrypt is not None:
        return bcrypt.checkpw(token_bytes, token_hash.encode("utf-8"))

    check = crypt.crypt(token_bytes.decode("utf-8"), token_hash)
    return bool(check) and check == token_hash
