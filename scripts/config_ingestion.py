#!/usr/bin/env python3
"""Centralized ingestion settings for self-hosted deployments."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class IngestionConfig:
    # Self-hosted Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_GRPC_URL = os.getenv("QDRANT_GRPC_URL", "localhost:6334")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", os.getenv("COLLECTION_NAME", "rag_pipeline"))
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "turbopuffer").strip().lower()
    TURBOPUFFER_API_KEY = os.getenv("TURBOPUFFER_API_KEY", "")
    TURBOPUFFER_REGION = os.getenv("TURBOPUFFER_REGION", "gcp-us-central1")
    TURBOPUFFER_NAMESPACE_PMC = os.getenv("TURBOPUFFER_NAMESPACE_PMC", "medical_database_pmc")
    TURBOPUFFER_NAMESPACE_PUBMED = os.getenv("TURBOPUFFER_NAMESPACE_PUBMED", "medical_pubmed")
    TURBOPUFFER_NAMESPACE_DAILYMED = os.getenv("TURBOPUFFER_NAMESPACE_DAILYMED", "medical_dailymed")
    TURBOPUFFER_WRITE_BATCH_SIZE = int(os.getenv("TURBOPUFFER_WRITE_BATCH_SIZE", "500"))
    TURBOPUFFER_MAX_CONCURRENT_WRITES = int(os.getenv("TURBOPUFFER_MAX_CONCURRENT_WRITES", "4"))
    TURBOPUFFER_MAX_RETRIES = int(os.getenv("TURBOPUFFER_MAX_RETRIES", "5"))
    TURBOPUFFER_SDK_MAX_RETRIES = int(os.getenv("TURBOPUFFER_SDK_MAX_RETRIES", "4"))
    TURBOPUFFER_TIMEOUT_SECONDS = int(os.getenv("TURBOPUFFER_TIMEOUT_SECONDS", "60"))
    TURBOPUFFER_DISABLE_BACKPRESSURE = _as_bool(
        os.getenv("TURBOPUFFER_DISABLE_BACKPRESSURE"),
        default=False,
    )
    TURBOPUFFER_METADATA_POLL_INTERVAL_SECONDS = _as_float(
        os.getenv("TURBOPUFFER_METADATA_POLL_INTERVAL_SECONDS"),
        default=0.0,
    )
    # Deprecated legacy knobs kept for compatibility with older utility scripts.
    SPARSE_ENABLED = _as_bool(os.getenv("SPARSE_ENABLED"), default=False)
    SPARSE_MODE = os.getenv("SPARSE_MODE", "disabled").strip().lower()
    SPARSE_MAX_TERMS_DOC = int(os.getenv("SPARSE_MAX_TERMS_DOC", "0"))
    SPARSE_MAX_TERMS_QUERY = int(os.getenv("SPARSE_MAX_TERMS_QUERY", "0"))
    SPARSE_MIN_TOKEN_LEN = int(os.getenv("SPARSE_MIN_TOKEN_LEN", "0"))
    SPARSE_REMOVE_STOPWORDS = _as_bool(os.getenv("SPARSE_REMOVE_STOPWORDS"), default=False)
    INGEST_DRY_RUN = _as_bool(os.getenv("INGEST_DRY_RUN"), default=False)

    # Embedding provider: DeepInfra only
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "deepinfra")
    EMBEDDING_MODEL = os.getenv(
        "INGESTION_EMBEDDING_MODEL",
        os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B-batch"),
    )

    # Data paths
    DATA_DIR = Path(os.getenv("DATA_DIR", "/data/ingestion"))
    PMC_XML_DIR = Path(os.getenv("PMC_XML_DIR", str(DATA_DIR / "pmc_xml")))
    DAILYMED_CHECKPOINT_FILE = Path(
        os.getenv("DAILYMED_CHECKPOINT_FILE", str(DATA_DIR / "dailymed_ingested_ids.txt"))
    )
    PUBMED_BASELINE_DIR = Path(os.getenv("PUBMED_BASELINE_DIR", str(DATA_DIR / "pubmed_baseline")))
    PUBMED_ABSTRACTS_FILE = Path(
        os.getenv(
            "PUBMED_ABSTRACTS_FILE",
            str(PUBMED_BASELINE_DIR / "filtered" / "pubmed_abstracts.jsonl"),
        )
    )

    # Tuning
    # Default 100 - can be reduced to 25 if "too many open files" errors with RocksDB occur
    # Each batch creates many chunks; smaller batches = fewer open files
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", os.getenv("PARALLEL_WORKERS", "8")))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    USE_GRPC = _as_bool(os.getenv("USE_GRPC"), default=True)
    WEEKLY_UPDATE_THROTTLE_SECONDS = _as_float(
        os.getenv("WEEKLY_UPDATE_THROTTLE_SECONDS"), default=0.5
    )
    WEEKLY_UPDATE_BATCH_SIZE = int(os.getenv("WEEKLY_UPDATE_BATCH_SIZE", "0"))

    # V4 Migration: Chunking & Filtering
    EMBED_FILTER_ENABLED = _as_bool(os.getenv("EMBED_FILTER_ENABLED"), default=True)
    EMBED_FILTER_MODE = os.getenv("EMBED_FILTER_MODE", "conservative")
    EMBED_FILTER_PROFILE = os.getenv("EMBED_FILTER_PROFILE", "clinical_backmatter")
    # Optimized chunk size: 2048 tokens for Qwen3-Embedding-0.6B (32k context window)
    # Research shows 1024-2048 is optimal for medical QA with 10-15% overlap
    # 2048 tokens provides rich context while fitting comfortably in 32k window
    CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "2048"))
    CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "256"))

    # Quantization configuration
    # Options: "scalar" (int8, recommended), "binary" (faster, less accurate), "none"
    QUANTIZATION_TYPE = os.getenv("QUANTIZATION_TYPE", "scalar").strip().lower()
    SCALAR_QUANTILE = float(os.getenv("SCALAR_QUANTILE", "0.99"))  # Clip outliers at 1st/99th percentile
    QUANTIZATION_ALWAYS_RAM = _as_bool(os.getenv("QUANTIZATION_ALWAYS_RAM"), default=True)

    # Vector dimensions by provider (DeepInfra only)
    EMBEDDING_DIMENSIONS = {
        "deepinfra": 1024,  # Qwen/Qwen3-Embedding-0.6B-batch
    }

    @classmethod
    def get_vector_size(cls) -> int:
        """Get the vector size for the current embedding provider."""
        provider = cls.EMBEDDING_PROVIDER.lower().strip()
        return cls.EMBEDDING_DIMENSIONS.get(provider, 1024)


def ensure_data_dirs() -> None:
    IngestionConfig.DATA_DIR.mkdir(parents=True, exist_ok=True)
    IngestionConfig.PMC_XML_DIR.mkdir(parents=True, exist_ok=True)
    IngestionConfig.DAILYMED_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    IngestionConfig.PUBMED_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    IngestionConfig.PUBMED_ABSTRACTS_FILE.parent.mkdir(parents=True, exist_ok=True)
