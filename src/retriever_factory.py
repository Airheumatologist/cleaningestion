"""Retriever factory with rollback-safe backend selection."""

from __future__ import annotations

import logging

from .config import RETRIEVAL_BACKEND, RETRIEVAL_BACKEND_ROLLBACK_ON_ERROR

logger = logging.getLogger(__name__)


def create_retriever(n_retrieval: int):
    backend = RETRIEVAL_BACKEND.strip().lower()
    if backend == "turbopuffer":
        try:
            from .retriever_turbopuffer import TurbopufferRetriever

            return TurbopufferRetriever(n_retrieval=n_retrieval)
        except Exception as exc:
            if RETRIEVAL_BACKEND_ROLLBACK_ON_ERROR:
                logger.error("turbopuffer retriever init failed, rolling back to Qdrant: %s", exc)
                from .retriever_qdrant import QdrantRetriever

                return QdrantRetriever(n_retrieval=n_retrieval)
            raise

    if backend == "lancedb":
        try:
            from .retriever_lancedb import LanceDBRetriever

            return LanceDBRetriever(n_retrieval=n_retrieval)
        except Exception as exc:
            if RETRIEVAL_BACKEND_ROLLBACK_ON_ERROR:
                logger.error("LanceDB retriever init failed, rolling back to Qdrant: %s", exc)
                from .retriever_qdrant import QdrantRetriever

                return QdrantRetriever(n_retrieval=n_retrieval)
            raise

    from .retriever_qdrant import QdrantRetriever

    return QdrantRetriever(n_retrieval=n_retrieval)
