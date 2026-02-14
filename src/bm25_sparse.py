"""Lightweight BM25-style sparse encoder for Qdrant sparse vectors."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Iterable, List

from qdrant_client.models import SparseVector

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Compact English stopword set to keep sparse vectors focused.
DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "had", "has", "have", "he", "her", "hers", "him", "his", "i", "if", "in", "into",
    "is", "it", "its", "itself", "me", "my", "myself", "no", "not", "of", "on", "or",
    "our", "ours", "ourselves", "she", "so", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "to", "too", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why",
    "will", "with", "you", "your", "yours", "yourself", "yourselves",
}


class BM25SparseEncoder:
    """
    Sparse encoder that emits hashed lexical term vectors.

    This is BM25-style because we encode term frequencies and rely on Qdrant's
    sparse vector IDF modifier for query-time IDF weighting.
    """

    def __init__(
        self,
        max_terms_doc: int = 256,
        max_terms_query: int = 64,
        min_token_len: int = 2,
        remove_stopwords: bool = True,
        k1: float = 1.2,
    ) -> None:
        self.max_terms_doc = max_terms_doc
        self.max_terms_query = max_terms_query
        self.min_token_len = min_token_len
        self.remove_stopwords = remove_stopwords
        self.k1 = k1

    def _tokenize(self, text: str) -> Iterable[str]:
        for token in TOKEN_RE.findall((text or "").lower()):
            if len(token) < self.min_token_len:
                continue
            if self.remove_stopwords and token in DEFAULT_STOPWORDS:
                continue
            yield token

    @staticmethod
    def _token_id(token: str) -> int:
        # Deterministic uint32 token ID for compatibility across runs/hosts.
        return int.from_bytes(hashlib.sha1(token.encode("utf-8")).digest()[:4], "little")

    def _bm25_tf(self, tf: int) -> float:
        return (tf * (self.k1 + 1.0)) / (tf + self.k1)

    def _encode(self, text: str, max_terms: int) -> SparseVector:
        token_counts = Counter(self._token_id(tok) for tok in self._tokenize(text))
        if not token_counts:
            return SparseVector(indices=[], values=[])

        weighted = ((idx, self._bm25_tf(tf)) for idx, tf in token_counts.items())
        top_terms = sorted(weighted, key=lambda pair: pair[1], reverse=True)[:max_terms]
        top_terms.sort(key=lambda pair: pair[0])  # stable ordering for deterministic payloads
        return SparseVector(
            indices=[idx for idx, _ in top_terms],
            values=[float(val) for _, val in top_terms],
        )

    def encode_document(self, text: str) -> SparseVector:
        return self._encode(text, self.max_terms_doc)

    def encode_query(self, text: str) -> SparseVector:
        return self._encode(text, self.max_terms_query)

    def encode_queries(self, texts: List[str]) -> List[SparseVector]:
        return [self.encode_query(text) for text in texts]
