#!/usr/bin/env python3
"""Shared ingestion sink abstraction for Qdrant/LanceDB migration."""

from __future__ import annotations

import json
import logging
import os
import random
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import lancedb
import pyarrow as pa

from config_ingestion import IngestionConfig
from ingestion_utils import upsert_with_retry

logger = logging.getLogger(__name__)

def _lancedb_arrow_schema() -> pa.Schema:
    vector_dim = IngestionConfig.get_vector_size()
    return pa.schema(
        [
        pa.field("point_id", pa.string()),
        pa.field("vector", pa.list_(pa.float64(), vector_dim)),
        pa.field("sparse_indices", pa.list_(pa.int64())),
        pa.field("sparse_values", pa.list_(pa.float64())),
        pa.field("doc_id", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("pmcid", pa.string()),
        pa.field("pmid", pa.string()),
        pa.field("doi", pa.string()),
        pa.field("page_content", pa.string()),
        pa.field("title", pa.string()),
        pa.field("abstract", pa.string()),
        pa.field("journal", pa.string()),
        pa.field("nlm_unique_id", pa.string()),
        pa.field("year", pa.int64()),
        pa.field("country", pa.string()),
        pa.field("keywords", pa.list_(pa.string())),
        pa.field("mesh_terms", pa.list_(pa.string())),
        pa.field("evidence_grade", pa.string()),
        pa.field("evidence_level", pa.int64()),
        pa.field("evidence_term", pa.string()),
        pa.field("evidence_source", pa.string()),
        pa.field("section_type", pa.string()),
        pa.field("section_title", pa.string()),
        pa.field("is_table", pa.bool_()),
        pa.field("table_caption", pa.string()),
        pa.field("full_section_text", pa.string()),
        pa.field("section_id", pa.string()),
        pa.field("section_weight", pa.float64()),
        pa.field("source", pa.string()),
        pa.field("source_family", pa.string()),
        pa.field("content_type", pa.string()),
        pa.field("is_author_manuscript", pa.bool_()),
        pa.field("nihms_id", pa.string()),
        pa.field("article_type", pa.string()),
        pa.field("publication_type", pa.list_(pa.string())),
        pa.field("has_full_text", pa.bool_()),
        pa.field("is_open_access", pa.bool_()),
        pa.field("license", pa.string()),
        pa.field("chunk_index", pa.int64()),
        pa.field("total_chunks", pa.int64()),
        pa.field("text_preview", pa.string()),
        ]
    )


def _connect_lancedb(uri: str):
    connect_kwargs: dict[str, Any] = {}
    if uri.startswith("db://"):
        if IngestionConfig.LANCEDB_API_KEY:
            connect_kwargs["api_key"] = IngestionConfig.LANCEDB_API_KEY
        if IngestionConfig.LANCEDB_REGION:
            connect_kwargs["region"] = IngestionConfig.LANCEDB_REGION
    return lancedb.connect(uri, **connect_kwargs)


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


def _normalize_lancedb_row_types(row: dict[str, Any]) -> dict[str, Any]:
    vector_dim = IngestionConfig.get_vector_size()
    string_fields = {
        "point_id",
        "doc_id",
        "chunk_id",
        "pmcid",
        "pmid",
        "doi",
        "page_content",
        "title",
        "abstract",
        "journal",
        "nlm_unique_id",
        "country",
        "evidence_grade",
        "evidence_term",
        "evidence_source",
        "section_type",
        "section_title",
        "table_caption",
        "full_section_text",
        "section_id",
        "source",
        "source_family",
        "content_type",
        "nihms_id",
        "article_type",
        "license",
        "text_preview",
    }
    int_fields = {"year", "evidence_level", "chunk_index", "total_chunks"}
    float_fields = {"section_weight"}
    bool_fields = {"is_table", "is_author_manuscript", "has_full_text", "is_open_access"}
    list_fields = {"keywords", "mesh_terms", "publication_type", "sparse_indices", "sparse_values"}

    for field in string_fields:
        value = row.get(field)
        row[field] = "" if value is None else str(value)

    for field in int_fields:
        value = row.get(field)
        if value is None or value == "":
            row[field] = 0
        else:
            try:
                row[field] = int(value)
            except (TypeError, ValueError):
                row[field] = 0

    for field in float_fields:
        value = row.get(field)
        if value is None or value == "":
            row[field] = 0.0
        else:
            try:
                row[field] = float(value)
            except (TypeError, ValueError):
                row[field] = 0.0

    for field in bool_fields:
        row[field] = bool(row.get(field, False))

    for field in list_fields:
        value = row.get(field)
        row[field] = value if isinstance(value, list) else []

    vector_value = row.get("vector")
    if isinstance(vector_value, list) and len(vector_value) == vector_dim:
        row["vector"] = [float(v) for v in vector_value]
    else:
        row["vector"] = [0.0] * vector_dim

    return row


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
        rows.append(_normalize_lancedb_row_types(row))
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
        self.write_batch_rows = max(1, int(os.getenv("LANCEDB_WRITE_BATCH_ROWS", "2000")))
        self.min_write_batch_rows = max(200, int(os.getenv("LANCEDB_MIN_WRITE_BATCH_ROWS", "500")))
        self._adaptive_batch_rows = self.write_batch_rows
        self._clean_write_streak = 0
        self._lock = threading.Lock()
        self._db = _connect_lancedb(uri)
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

        table = pa.Table.from_pylist(rows, schema=_lancedb_arrow_schema())
        max_retries = int(os.getenv("MAX_RETRIES", "5"))
        
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                self._table = self._db.create_table(
                    self.table_name,
                    data=table.to_batches(max_chunksize=self.write_batch_rows),
                    schema=table.schema,
                    mode="create",
                )
                if attempt > 1:
                    logger.info("LanceDB table %s created successfully on attempt %s", self.table_name, attempt)
                return True
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                # Only retry simple http errors or timeouts
                exc_str = str(exc).lower()
                if "already exists" in exc_str:
                    logger.warning("LanceDB table %s exists but open_table failed previously", self.table_name)
                    # Try to open it again
                    try:
                        self._table = self._db.open_table(self.table_name)
                        return False
                    except Exception as inner_exc:
                        last_exc = inner_exc
                else:
                    wait = 2.0 * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    logger.warning(
                        "LanceDB table creation retry %s/%s (wait=%.2fs) after error: %s",
                        attempt, max_retries, wait, str(exc)[:200]
                    )
                    time.sleep(wait)
                    
        raise RuntimeError(f"Failed to create LanceDB table after {max_retries} attempts: {last_exc}") from last_exc

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

    def _add_rows_with_retry(self, rows: list[dict[str, Any]]) -> None:
        table = pa.Table.from_pylist(rows, schema=_lancedb_arrow_schema())
        attempts = max(1, int(IngestionConfig.MAX_RETRIES))
        current_batch_rows = max(self.min_write_batch_rows, self._adaptive_batch_rows)
        for attempt in range(1, attempts + 1):
            try:
                self._table.add(table.to_batches(max_chunksize=current_batch_rows))
                self._clean_write_streak += 1
                if self._clean_write_streak >= 3:
                    self._adaptive_batch_rows = min(
                        self.write_batch_rows,
                        int(max(self._adaptive_batch_rows, current_batch_rows) * 1.25),
                    )
                    self._clean_write_streak = 0
                return
            except Exception as exc:
                self._clean_write_streak = 0
                if attempt >= attempts:
                    raise
                # Adaptive downshift on first failure to reduce payload pressure.
                current_batch_rows = max(self.min_write_batch_rows, current_batch_rows // 2)
                self._adaptive_batch_rows = current_batch_rows
                if attempt >= 2:
                    self._reconnect_table()

                base_wait = min(4.0, 1.0 * (2 ** (attempt - 1)))
                jitter = random.uniform(0.0, max(0.1, base_wait * 0.35))
                wait_seconds = base_wait + jitter
                logger.warning(
                    "LanceDB add retry %s/%s (batch_rows=%s wait=%.2fs) after error: %s",
                    attempt,
                    attempts,
                    current_batch_rows,
                    wait_seconds,
                    str(exc)[:300],
                )
                time.sleep(wait_seconds)

    def _reconnect_table(self) -> None:
        try:
            self._db = _connect_lancedb(self.uri)
            self._table = self._db.open_table(self.table_name)
            logger.info("Reconnected LanceDB table handle for %s", self.table_name)
        except Exception as exc:
            logger.warning("LanceDB reconnect attempt failed: %s", str(exc)[:200])

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
                self._add_rows_with_retry(rows)
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
