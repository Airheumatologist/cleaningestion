"""Shared utilities for ingestion scripts."""

from __future__ import annotations

from typing import Any, Dict, List, Set

class SectionFilter:
    EXCLUDED_TYPES = {
        "references", "bibliography", "acknowledgments", "funding",
        "conflict", "disclosure", "author_contributions", "supplementary"
    }
    
    EXCLUDED_TITLES = {
        "references", "bibliography", "literature cited", "acknowledgments",
        "acknowledgements", "funding", "financial support", "conflict of interest",
        "author contributions", "disclosure", "supplementary data"
    }

    @classmethod
    def should_exclude(cls, section: Dict[str, Any]) -> bool:
        # Check type
        sec_type = section.get("type", "").lower()
        if sec_type in cls.EXCLUDED_TYPES:
            return True
            
        # Check title
        title = section.get("title", "").lower()
        if any(ex in title for ex in cls.EXCLUDED_TITLES):
            return True
            
        return False


class Chunker:
    def __init__(self, chunk_size: int = 384, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
        except Exception:
            self.tokenizer = None
            
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
            
        # Simple word-based fallback if tokenizer fails
        if not self.tokenizer:
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunks.append({
                    "text": " ".join(chunk_words),
                    "token_count": len(chunk_words)
                })
            return chunks
            
        # Token-aware chunking
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_tokens)
            })
            
        return chunks
