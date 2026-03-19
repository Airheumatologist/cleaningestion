#!/usr/bin/env python3
"""Compatibility shim: ingestion sink now uses turbopuffer only."""

from __future__ import annotations

from turbopuffer_ingestion_sink import BaseIngestionSink, IngestionSinkStats, TurbopufferIngestionSink, build_ingestion_sink

__all__ = [
    "BaseIngestionSink",
    "IngestionSinkStats",
    "TurbopufferIngestionSink",
    "build_ingestion_sink",
]

