#!/usr/bin/env python3
"""Shared ingestion sink abstraction for turbopuffer ingestion."""

from __future__ import annotations

import json
import logging
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
    def __init__(self, namespace: str, dry_run: bool = False):
        if getattr(tpuf, "Turbopuffer", None) is None:
            raise RuntimeError("turbopuffer package is not installed. Install requirements first.")
        if not IngestionConfig.TURBOPUFFER_API_KEY:
            raise RuntimeError("TURBOPUFFER_API_KEY is required for turbopuffer ingestion")
        self.namespace = namespace
        self.dry_run = dry_run
        self.batch_size = max(1, int(IngestionConfig.TURBOPUFFER_WRITE_BATCH_SIZE))
        self.max_retries = max(1, int(IngestionConfig.TURBOPUFFER_MAX_RETRIES))
        self._lock = threading.Lock()
        self._schema_declared = False
        self.stats = IngestionSinkStats()
        client = tpuf.Turbopuffer(
            api_key=IngestionConfig.TURBOPUFFER_API_KEY,
            region=IngestionConfig.TURBOPUFFER_REGION,
        )
        self.ns = client.namespace(namespace)

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

    def _write_columns_with_retry(self, columns: dict[str, list[Any]]) -> None:
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {"upsert_columns": columns, "distance_metric": "cosine_distance"}
                if not self._schema_declared:
                    kwargs["schema"] = self._schema_for_columns(columns)
                self.ns.write(**kwargs)
                self._schema_declared = True
                return
            except Exception as exc:
                exc_str = str(exc).lower()
                retryable = any(code in exc_str for code in ("429", "408", "409", "500", "502", "503", "504"))
                if attempt >= self.max_retries or not retryable:
                    raise
                wait = min(10.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
                logger.warning(
                    "turbopuffer write retry %s/%s namespace=%s wait=%.2fs error=%s",
                    attempt,
                    self.max_retries,
                    self.namespace,
                    wait,
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

        with self._lock:
            written = 0
            for i in range(0, len(rows), self.batch_size):
                batch_rows = rows[i : i + self.batch_size]
                columns = self._rows_to_columns(batch_rows)
                self._write_columns_with_retry(columns)
                written += len(batch_rows)
                self.stats.batches_written += 1
            self.stats.rows_written += written
        return len(rows)


def build_ingestion_sink(client: Any = None, namespace_override: str | None = None) -> BaseIngestionSink:
    _ = client
    namespace = namespace_override or IngestionConfig.TURBOPUFFER_NAMESPACE_PMC
    return TurbopufferIngestionSink(namespace=namespace, dry_run=IngestionConfig.INGEST_DRY_RUN)
