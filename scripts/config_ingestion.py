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


class IngestionConfig:
    # Self-hosted Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_GRPC_URL = os.getenv("QDRANT_GRPC_URL", "localhost:6334")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", os.getenv("COLLECTION_NAME", "medical_rag"))

    # Embedding options: "cohere" (API, default) or "local" (needs GPU)
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cohere")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-v4.0")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    QDRANT_INFERENCE_URL = os.getenv("QDRANT_INFERENCE_URL", "")
    QDRANT_INFERENCE_KEY = os.getenv("QDRANT_INFERENCE_KEY", "")

    # Data paths
    DATA_DIR = Path(os.getenv("DATA_DIR", "/data/ingestion"))
    PMC_XML_DIR = Path(os.getenv("PMC_XML_DIR", str(DATA_DIR / "pmc_xml")))
    PMC_JSONL_FILE = Path(os.getenv("PMC_JSONL_FILE", str(DATA_DIR / "pmc_articles.jsonl")))
    DAILYMED_XML_DIR = Path(os.getenv("DAILYMED_XML_DIR", str(DATA_DIR / "dailymed" / "xml")))

    # Tuning
    # Reduced from 100 to 25 to prevent "too many open files" errors with RocksDB
    # Each batch creates many chunks; smaller batches = fewer open files
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "25"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", os.getenv("PARALLEL_WORKERS", "4")))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    USE_GRPC = _as_bool(os.getenv("USE_GRPC"), default=True)
    CLOUD_INFERENCE = _as_bool(os.getenv("QDRANT_CLOUD_INFERENCE"), default=False)

    # V4 Migration: Chunking & Filtering
    EMBED_FILTER_ENABLED = _as_bool(os.getenv("EMBED_FILTER_ENABLED"), default=True)
    EMBED_FILTER_MODE = os.getenv("EMBED_FILTER_MODE", "conservative")
    EMBED_FILTER_PROFILE = os.getenv("EMBED_FILTER_PROFILE", "clinical_backmatter")
    CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "384"))
    CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))

    # Sparse indexing (BM25-style lexical sparse vectors)
    SPARSE_ENABLED = _as_bool(os.getenv("SPARSE_ENABLED"), default=True)
    SPARSE_MODE = os.getenv("SPARSE_MODE", "bm25").strip().lower()
    SPARSE_MAX_TERMS_DOC = int(os.getenv("SPARSE_MAX_TERMS_DOC", "256"))
    SPARSE_MAX_TERMS_QUERY = int(os.getenv("SPARSE_MAX_TERMS_QUERY", "64"))
    SPARSE_MIN_TOKEN_LEN = int(os.getenv("SPARSE_MIN_TOKEN_LEN", "2"))
    SPARSE_REMOVE_STOPWORDS = _as_bool(os.getenv("SPARSE_REMOVE_STOPWORDS"), default=True)

    # Vector dimensions for different embedding providers
    # This ensures collection is created with the correct vector size
    EMBEDDING_DIMENSIONS = {
        "cohere": 1536,  # Cohere embed-v4.0 and earlier models
        "local": 1024,   # mixedbread-ai/mxbai-embed-large-v1
        "qdrant_cloud_inference": 1024,  # Default for cloud inference
    }

    @classmethod
    def get_vector_size(cls) -> int:
        """Get the vector size for the current embedding provider."""
        provider = cls.EMBEDDING_PROVIDER.lower().strip()
        # Handle Cohere model variants
        if provider == "cohere":
            # Cohere embed-v4.0 outputs 1536 dimensions
            return 1536
        return cls.EMBEDDING_DIMENSIONS.get(provider, 1024)


def ensure_data_dirs() -> None:
    IngestionConfig.DATA_DIR.mkdir(parents=True, exist_ok=True)
    IngestionConfig.PMC_XML_DIR.mkdir(parents=True, exist_ok=True)
    IngestionConfig.DAILYMED_XML_DIR.mkdir(parents=True, exist_ok=True)
