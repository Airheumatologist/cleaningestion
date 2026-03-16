"""Retriever factory with rollback-safe backend selection."""

from __future__ import annotations

import logging

from .config import RETRIEVAL_BACKEND, RETRIEVAL_BACKEND_ROLLBACK_ON_ERROR
from .retriever_lancedb import LanceDBRetriever
from .retriever_qdrant import QdrantRetriever

logger = logging.getLogger(__name__)


def create_retriever(n_retrieval: int):
    backend = RETRIEVAL_BACKEND.strip().lower()
    if backend == "lancedb":
        try:
            return LanceDBRetriever(n_retrieval=n_retrieval)
        except Exception as exc:
            if RETRIEVAL_BACKEND_ROLLBACK_ON_ERROR:
                logger.error("LanceDB retriever init failed, rolling back to Qdrant: %s", exc)
                return QdrantRetriever(n_retrieval=n_retrieval)
            raise

    return QdrantRetriever(n_retrieval=n_retrieval)
