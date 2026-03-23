#!/usr/bin/env python3
"""Shared ingestion sink abstraction for turbopuffer ingestion."""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import types
from dataclasses import dataclass
from typing import Any, Iterable

try:
    import turbopuffer as tpuf
except ImportError:  # pragma: no cover - environment-dependent
    tpuf = types.SimpleNamespace(Namespace=None)  # type: ignore

from config_ingestion import IngestionConfig

logger = logging.getLogger(__name__)

@dataclass
class IngestionSinkStats:
    rows_written: int = 0
    batches_written: int = 0


class BaseIngestionSink:
    def write_points(self, points: Iterable[Any]) -> int:
        raise NotImplementedError

    def close(self) -> None:
        return


def _extract_dense_vector(vector_data: Any) -> list[float]:
    if isinstance(vector_data, dict):
        dense = vector_data.get("dense")
        if isinstance(dense, list):
            return [float(v) for v in dense]
        return []
    if isinstance(vector_data, list):
        return [float(v) for v in vector_data]
    return []


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
            return value
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _infer_schema_type(values: list[Any]) -> Any:
    non_null_values = [value for value in values if value is not None]
    if not non_null_values:
        return "string"

    if all(isinstance(value, bool) for value in non_null_values):
        return "bool"

    if all(isinstance(value, int) and not isinstance(value, bool) for value in non_null_values):
        return "int"

    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in non_null_values):
        return "float" if any(isinstance(value, float) for value in non_null_values) else "int"

    if all(isinstance(value, list) for value in non_null_values):
        if all(
            all(item is None or isinstance(item, str) for item in value)
            for value in non_null_values
        ):
            return "[]string"

    return "string"


class TurbopufferIngestionSink(BaseIngestionSink):
    def __init__(self, namespace: str, dry_run: bool = False, disable_backpressure: bool | None = None):
        if getattr(tpuf, "Turbopuffer", None) is None:
            raise RuntimeError("turbopuffer package is not installed. Install requirements first.")
        if not IngestionConfig.TURBOPUFFER_API_KEY:
            raise RuntimeError("TURBOPUFFER_API_KEY is required for turbopuffer ingestion")
        self.namespace = namespace
        self.dry_run = dry_run
        self.batch_size = max(1, int(IngestionConfig.TURBOPUFFER_WRITE_BATCH_SIZE))
        self.max_retries = max(1, int(IngestionConfig.TURBOPUFFER_MAX_RETRIES))
        self.max_concurrent_writes = max(1, int(IngestionConfig.TURBOPUFFER_MAX_CONCURRENT_WRITES))
        self.sdk_max_retries = max(0, int(IngestionConfig.TURBOPUFFER_SDK_MAX_RETRIES))
        self.disable_backpressure = (
            bool(disable_backpressure)
            if disable_backpressure is not None
            else bool(IngestionConfig.TURBOPUFFER_DISABLE_BACKPRESSURE)
        )
        self.metadata_poll_interval_seconds = max(
            0.0,
            float(IngestionConfig.TURBOPUFFER_METADATA_POLL_INTERVAL_SECONDS),
        )
        self.min_batch_interval_seconds = max(
            0.0,
            float(os.getenv("TURBOPUFFER_MIN_BATCH_INTERVAL_SECONDS", "0") or "0"),
        )
        self._write_semaphore = threading.BoundedSemaphore(self.max_concurrent_writes)
        self._schema_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._pace_lock = threading.Lock()
        self._metadata_lock = threading.Lock()
        self._schema_declared = False
        self._last_batch_write_at: float | None = None
        self._last_metadata_logged_at: float | None = None
        self.stats = IngestionSinkStats()
        client = tpuf.Turbopuffer(
            api_key=IngestionConfig.TURBOPUFFER_API_KEY,
            region=IngestionConfig.TURBOPUFFER_REGION,
            max_retries=self.sdk_max_retries,
        )
        self.ns = client.namespace(namespace)
        logger.info(
            "Initialized turbopuffer sink namespace=%s batch_size=%s concurrency=%s disable_backpressure=%s sdk_max_retries=%s",
            namespace,
            self.batch_size,
            self.max_concurrent_writes,
            self.disable_backpressure,
            self.sdk_max_retries,
        )

    def _pace_next_batch_if_needed(self) -> None:
        if self.min_batch_interval_seconds <= 0:
            return
        with self._pace_lock:
            if self._last_batch_write_at is None:
                return
            now = time.monotonic()
            elapsed = now - self._last_batch_write_at
            if elapsed < self.min_batch_interval_seconds:
                time.sleep(self.min_batch_interval_seconds - elapsed)

    def _mark_batch_write(self) -> None:
        with self._pace_lock:
            self._last_batch_write_at = time.monotonic()

    @staticmethod
    def _schema_for_columns(columns: dict[str, list[Any]]) -> dict[str, Any]:
        vector_dims = IngestionConfig.get_vector_size()
        int_fields = {"year", "evidence_level", "chunk_index", "total_chunks", "table_count", "token_count"}
        float_fields = {"section_weight", "ingestion_timestamp"}
        bool_fields = {
            "is_table",
            "is_author_manuscript",
            "has_full_text",
            "is_open_access",
            "has_tables",
            "is_gov_affiliated",
            "has_structured_abstract",
        }
        list_string_fields = {"keywords", "mesh_terms", "publication_type", "active_ingredients", "gov_agencies"}
        json_large_fields = {
            "abstract_structured",
            "mesh_terms_full",
            "keywords_full",
            "publication_types_full",
            "journal_full",
            "publication_date",
            "other_ids",
        }
        schema: dict[str, Any] = {}
        for key, values in columns.items():
            if key == "id":
                schema[key] = "uuid"
            elif key == "vector":
                schema[key] = {"type": f"[{vector_dims}]f16", "ann": True}
            elif key in int_fields:
                schema[key] = "int"
            elif key in float_fields:
                schema[key] = "float"
            elif key in bool_fields:
                schema[key] = "bool"
            elif key in list_string_fields:
                schema[key] = "[]string"
            elif key in json_large_fields:
                schema[key] = {"type": "string", "filterable": False}
            elif key in {
                "page_content",
                "title",
                "abstract",
                "full_section_text",
                "text_preview",
                "journal",
                "section_title",
                "section_id",
                "table_caption",
            }:
                schema[key] = {"type": "string", "filterable": False}
            else:
                inferred = _infer_schema_type(values)
                schema[key] = inferred
        if "page_content" in schema:
            schema["page_content"] = {
                "type": "string",
                "full_text_search": {
                    "tokenizer": "word_v3",
                    "language": "english",
                    "stemming": True,
                    "remove_stopwords": True,
                },
                "filterable": False,
            }
        if "title" in schema:
            schema["title"] = {"type": "string", "full_text_search": True, "filterable": False}
        return schema

    def _point_to_row(self, point: Any) -> dict[str, Any]:
        payload = dict(getattr(point, "payload", {}) or {})
        vector = getattr(point, "vector", {})
        point_id = str(getattr(point, "id", ""))
        row: dict[str, Any] = {"id": point_id, "vector": _extract_dense_vector(vector)}
        for key, value in payload.items():
            coerced = _coerce_scalar(value)
            if key == "section_weight" and coerced is not None:
                try:
                    coerced = float(coerced)
                except (TypeError, ValueError):
                    coerced = 0.0
            row[key] = coerced
        return row

    @staticmethod
    def _rows_to_columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
        all_keys: set[str] = set()
        for row in rows:
            all_keys.update(row.keys())
        columns: dict[str, list[Any]] = {}
        for key in sorted(all_keys):
            columns[key] = [row.get(key) for row in rows]
        return columns

    def _log_namespace_metadata_if_due(self) -> None:
        if self.metadata_poll_interval_seconds <= 0:
            return
        with self._metadata_lock:
            now = time.monotonic()
            if self._last_metadata_logged_at is not None:
                elapsed = now - self._last_metadata_logged_at
                if elapsed < self.metadata_poll_interval_seconds:
                    return
            self._last_metadata_logged_at = now
        try:
            metadata = self.ns.metadata()
            index = getattr(metadata, "index", None)
            index_status = getattr(index, "status", "unknown")
            unindexed_bytes = int(getattr(index, "unindexed_bytes", 0) or 0)
            approx_rows = int(getattr(metadata, "approx_row_count", 0) or 0)
            approx_bytes = int(getattr(metadata, "approx_logical_bytes", 0) or 0)
            logger.info(
                "turbopuffer metadata namespace=%s index_status=%s unindexed_bytes=%s approx_rows=%s approx_logical_bytes=%s",
                self.namespace,
                index_status,
                unindexed_bytes,
                approx_rows,
                approx_bytes,
            )
        except Exception as exc:
            logger.debug("Failed to fetch turbopuffer metadata for namespace=%s: %s", self.namespace, exc)

    def _write_columns_with_retry(self, columns: dict[str, list[Any]]) -> None:
        for attempt in range(1, self.max_retries + 1):
            try:
                with self._schema_lock:
                    include_schema = not self._schema_declared
                kwargs: dict[str, Any] = {
                    "upsert_columns": columns,
                    "distance_metric": "cosine_distance",
                }
                if include_schema:
                    kwargs["schema"] = self._schema_for_columns(columns)
                if self.disable_backpressure:
                    kwargs["disable_backpressure"] = True
                with self._write_semaphore:
                    self.ns.write(**kwargs)
                if include_schema:
                    with self._schema_lock:
                        self._schema_declared = True
                self._log_namespace_metadata_if_due()
                return
            except Exception as exc:
                exc_str = str(exc).lower()
                exc_class = getattr(exc, "__class__", None)
                exc_name = exc_class.__name__.lower() if exc_class else ""
                
                retryable = any(code in exc_str for code in ("429", "408", "409", "500", "502", "503", "504", "timeout", "timed out", "connection error"))
                if "timeout" in exc_name or "connection" in exc_name or "readerror" in exc_name or "ssl" in exc_name:
                    retryable = True
                if attempt >= self.max_retries or not retryable:
                    raise
                wait = min(10.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
                logger.warning(
                    "turbopuffer write retry %s/%s namespace=%s wait=%.2fs disable_backpressure=%s error=%s",
                    attempt,
                    self.max_retries,
                    self.namespace,
                    wait,
                    self.disable_backpressure,
                    str(exc)[:300],
                )
                time.sleep(wait)

    def write_points(self, points: Iterable[Any]) -> int:
        point_list = list(points)
        if not point_list:
            return 0
        rows = [self._point_to_row(point) for point in point_list]
        if self.dry_run:
            return len(rows)

        # Deduplicate rows by id to avoid turbopuffer 400 "duplicate document IDs" errors.
        # This is a last-resort guard; upstream callers should also deduplicate.
        seen_ids: set[str] = set()
        deduped_rows: list[dict[str, Any]] = []
        for row in rows:
            row_id = str(row.get("id", ""))
            if row_id and row_id in seen_ids:
                logger.warning("Dropping duplicate row id=%s before turbopuffer write", row_id)
                continue
            if row_id:
                seen_ids.add(row_id)
            deduped_rows.append(row)
        rows = deduped_rows

        written = 0
        batches_written = 0
        for i in range(0, len(rows), self.batch_size):
            self._pace_next_batch_if_needed()
            batch_rows = rows[i : i + self.batch_size]
            columns = self._rows_to_columns(batch_rows)
            self._write_columns_with_retry(columns)
            self._mark_batch_write()
            written += len(batch_rows)
            batches_written += 1
        with self._stats_lock:
            self.stats.batches_written += batches_written
            self.stats.rows_written += written
        return len(rows)


def build_ingestion_sink(
    client: Any = None,
    namespace_override: str | None = None,
    disable_backpressure: bool | None = None,
) -> BaseIngestionSink:
    _ = client
    namespace = namespace_override or IngestionConfig.TURBOPUFFER_NAMESPACE_PMC
    return TurbopufferIngestionSink(
        namespace=namespace,
        dry_run=IngestionConfig.INGEST_DRY_RUN,
        disable_backpressure=disable_backpressure,
    )
