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
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "lancedb").strip().lower()
    LANCEDB_URI = os.getenv("LANCEDB_URI", "./medical_data.lancedb")
    LANCEDB_TABLE = os.getenv("LANCEDB_TABLE", "medical_docs")
    LANCEDB_REINDEX_INTERVAL_BATCHES = int(os.getenv("LANCEDB_REINDEX_INTERVAL_BATCHES", "50"))
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
    DAILYMED_XML_DIR = Path(os.getenv("DAILYMED_XML_DIR", str(DATA_DIR / "dailymed" / "xml")))
    DAILYMED_STATE_DIR = Path(os.getenv("DAILYMED_STATE_DIR", str(DATA_DIR / "dailymed" / "state")))
    DAILYMED_CHECKPOINT_FILE = Path(
        os.getenv("DAILYMED_CHECKPOINT_FILE", str(DATA_DIR / "dailymed_ingested_ids.txt"))
    )
    DAILYMED_SET_ID_MANIFEST = Path(
        os.getenv(
            "DAILYMED_SET_ID_MANIFEST",
            str(DAILYMED_STATE_DIR / "dailymed_last_update_set_ids.txt"),
        )
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

    # Sparse indexing (BM25-style lexical sparse vectors)
    SPARSE_ENABLED = _as_bool(os.getenv("SPARSE_ENABLED"), default=True)
    SPARSE_MODE = os.getenv("SPARSE_MODE", "bm25").strip().lower()
    SPARSE_MAX_TERMS_DOC = int(os.getenv("SPARSE_MAX_TERMS_DOC", "256"))
    SPARSE_MAX_TERMS_QUERY = int(os.getenv("SPARSE_MAX_TERMS_QUERY", "64"))
    SPARSE_MIN_TOKEN_LEN = int(os.getenv("SPARSE_MIN_TOKEN_LEN", "2"))
    SPARSE_REMOVE_STOPWORDS = _as_bool(os.getenv("SPARSE_REMOVE_STOPWORDS"), default=True)

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
    IngestionConfig.DAILYMED_XML_DIR.mkdir(parents=True, exist_ok=True)
    IngestionConfig.DAILYMED_STATE_DIR.mkdir(parents=True, exist_ok=True)
    IngestionConfig.DAILYMED_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    IngestionConfig.DAILYMED_SET_ID_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    IngestionConfig.PUBMED_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    IngestionConfig.PUBMED_ABSTRACTS_FILE.parent.mkdir(parents=True, exist_ok=True)
