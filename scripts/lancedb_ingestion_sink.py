#!/usr/bin/env python3
"""Shared ingestion sink abstraction for Qdrant/LanceDB migration."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import lancedb

from config_ingestion import IngestionConfig
from ingestion_utils import upsert_with_retry

logger = logging.getLogger(__name__)


def _extract_dense_vector(vector_data: Any) -> list[float]:
    if isinstance(vector_data, dict):
        dense = vector_data.get("dense")
        if isinstance(dense, list):
            return dense
        return []
    if isinstance(vector_data, list):
        return vector_data
    return []


def _extract_sparse(vector_data: Any) -> tuple[list[int], list[float]]:
    sparse = vector_data.get("sparse") if isinstance(vector_data, dict) else None
    if sparse is None:
        return [], []

    indices = getattr(sparse, "indices", []) or []
    values = getattr(sparse, "values", []) or []
    return list(indices), list(values)


def points_to_lancedb_rows(points: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in points:
        payload = dict(getattr(point, "payload", {}) or {})
        vector = getattr(point, "vector", {})
        sparse_indices, sparse_values = _extract_sparse(vector)
        row = {
            "point_id": str(getattr(point, "id", "")),
            "vector": _extract_dense_vector(vector),
            "sparse_indices": sparse_indices,
            "sparse_values": sparse_values,
        }
        row.update(payload)
        rows.append(row)
    return rows


def build_reindex_command(manifest_path: Path, uri: str, table_name: str) -> list[str]:
    return [
        "python3",
        "scripts/lancedb_index_manager.py",
        "--manifest",
        str(manifest_path),
        "--uri",
        uri,
        "--table",
        table_name,
        "--json",
        "build",
    ]


@dataclass
class IngestionSinkStats:
    rows_written: int = 0
    batches_written: int = 0


class BaseIngestionSink:
    def write_points(self, points: Iterable[Any]) -> int:
        raise NotImplementedError

    def close(self) -> None:
        return


class QdrantIngestionSink(BaseIngestionSink):
    def __init__(self, client: Any):
        self.client = client

    def write_points(self, points: Iterable[Any]) -> int:
        point_list = list(points)
        if not point_list:
            return 0
        upsert_with_retry(self.client, point_list)
        return len(point_list)


class LanceDBIngestionSink(BaseIngestionSink):
    def __init__(
        self,
        uri: str,
        table_name: str,
        dry_run: bool = False,
        reindex_interval_batches: int = 0,
        manifest_path: Path | None = None,
    ):
        self.uri = uri
        self.table_name = table_name
        self.dry_run = dry_run
        self.reindex_interval_batches = max(0, int(reindex_interval_batches))
        self.manifest_path = manifest_path or (
            Path(__file__).resolve().parent.parent / "schema" / "lancedb_index_profiles.json"
        )
        self._lock = threading.Lock()
        self._db = lancedb.connect(uri)
        self._table = None
        self.stats = IngestionSinkStats()

    def _to_rows(self, points: Iterable[Any]) -> list[dict[str, Any]]:
        return points_to_lancedb_rows(points)

    def _ensure_table(self, rows: list[dict[str, Any]]) -> bool:
        if self._table is not None:
            return False

        try:
            self._table = self._db.open_table(self.table_name)
            return False
        except Exception:
            pass

        if not rows:
            raise RuntimeError("Cannot create LanceDB table without rows")

        self._table = self._db.create_table(self.table_name, data=rows, mode="create")
        return True

    def _maybe_reindex(self) -> None:
        if self.reindex_interval_batches <= 0:
            return
        if self.stats.batches_written % self.reindex_interval_batches != 0:
            return

        cmd = build_reindex_command(
            manifest_path=self.manifest_path,
            uri=self.uri,
            table_name=self.table_name,
        )
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stderr_excerpt = stderr[-400:] if stderr else ""
            logger.warning(
                "LanceDB reindex failed rc=%s cmd=%s stderr=%s",
                proc.returncode,
                shlex.join(cmd),
                stderr_excerpt,
            )
            return

        try:
            payload = json.loads(proc.stdout or "{}")
            logger.info("LanceDB reindex completed profile=%s", payload.get("profile"))
        except json.JSONDecodeError:
            logger.info("LanceDB reindex completed")

    def write_points(self, points: Iterable[Any]) -> int:
        point_list = list(points)
        if not point_list:
            return 0

        rows = self._to_rows(point_list)
        if self.dry_run:
            return len(rows)

        with self._lock:
            created_with_rows = self._ensure_table(rows)
            if not created_with_rows:
                self._table.add(rows)
            self.stats.rows_written += len(rows)
            self.stats.batches_written += 1
            self._maybe_reindex()
        return len(rows)


def build_ingestion_sink(client: Any = None) -> BaseIngestionSink:
    backend = IngestionConfig.VECTOR_BACKEND.strip().lower()
    if backend == "lancedb":
        return LanceDBIngestionSink(
            uri=IngestionConfig.LANCEDB_URI,
            table_name=IngestionConfig.LANCEDB_TABLE,
            dry_run=IngestionConfig.INGEST_DRY_RUN,
            reindex_interval_batches=IngestionConfig.LANCEDB_REINDEX_INTERVAL_BATCHES,
        )

    if client is None:
        raise RuntimeError("Qdrant backend requires initialized client")
    return QdrantIngestionSink(client)
