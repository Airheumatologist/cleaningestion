"""
SPLADE Encoder for Sparse Vector Generation.

Uses naver/splade-cocondenser-ensembledistil model to generate sparse vectors
for hybrid search (dense + sparse).
"""

import logging
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Model configuration
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"


class SPLADEEncoder:
    """SPLADE encoder for generating sparse vectors with caching."""
    
    def __init__(self, model_name: str = SPLADE_MODEL, device: Optional[str] = None, cache_size: int = 256):
        """
        Initialize SPLADE encoder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_size: Max number of query vectors to cache (default: 256)
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # LRU cache for query sparse vectors (key: query text, value: sparse dict)
        from collections import OrderedDict
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Loading SPLADE model: {model_name} on {device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            logger.info(f"✅ SPLADE model loaded successfully (cache_size={cache_size})")
        except Exception as e:
            logger.error(f"❌ Error loading SPLADE model: {e}")
            raise
    
    def encode_cached(self, text: str, max_length: int = 512) -> Dict[int, float]:
        """
        Encode a single text with caching.
        
        Args:
            text: Text string to encode
            max_length: Maximum sequence length
            
        Returns:
            Sparse vector dict (token_id -> score)
        """
        cache_key = text[:512]  # Truncate for cache key consistency
        
        # Check cache
        if cache_key in self._cache:
            self._cache_hits += 1
            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        
        # Cache miss - compute
        self._cache_misses += 1
        vectors = self.encode([text], max_length=max_length)
        sparse_dict = vectors[0] if vectors else {}
        
        # Add to cache
        self._cache[cache_key] = sparse_dict
        
        # Evict oldest if over capacity
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        
        return sparse_dict
    
    def encode_batch_cached(self, texts: List[str], max_length: int = 512) -> List[Dict[int, float]]:
        """
        Encode multiple texts with caching - returns cached results and computes only missing ones.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            List of sparse vectors in same order as input
        """
        results = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = text[:512]
            if cache_key in self._cache:
                self._cache_hits += 1
                self._cache.move_to_end(cache_key)
                results[i] = self._cache[cache_key]
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode all uncached texts in a single batch
        if texts_to_encode:
            self._cache_misses += len(texts_to_encode)
            encoded_vectors = self.encode(texts_to_encode, max_length=max_length)
            
            # Store results and update cache
            for idx, vector, text in zip(indices_to_encode, encoded_vectors, texts_to_encode):
                results[idx] = vector
                cache_key = text[:512]
                self._cache[cache_key] = vector
                
                # Evict oldest if over capacity
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        
        return results
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "hit_rate_pct": round(hit_rate, 1)
        }
    
    def encode(
        self,
        texts: List[str],
        max_length: int = 512,
        return_tokens: bool = False
    ) -> List[Dict[int, float]]:
        """
        Encode texts into sparse vectors.
        
        Args:
            texts: List of text strings to encode
            max_length: Maximum sequence length
            return_tokens: Whether to return token information
            
        Returns:
            List of sparse vectors (dicts mapping token_id -> score)
        """
        if not texts:
            return []
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        # Get logits and apply ReLU + log(1 + x) transformation
        logits = outputs.logits
        relu_log = torch.log(1 + torch.relu(logits))
        
        # Aggregate (max pooling over tokens)
        # Shape: [batch_size, vocab_size]
        sparse_vecs = relu_log.max(dim=1)[0]
        
        # Handle single text case - ensure we have 2D tensor
        if len(sparse_vecs.shape) == 1:
            sparse_vecs = sparse_vecs.unsqueeze(0)
        
        # Convert to sparse format (dict of token_id -> score)
        sparse_vectors = []
        for vec in sparse_vecs:
            # Filter out zero values and convert to dict
            sparse_dict = {}
            # Use nonzero() to efficiently get non-zero indices
            non_zero_indices = torch.nonzero(vec, as_tuple=True)[0]
            for token_id in non_zero_indices:
                idx = int(token_id.item())
                score = float(vec[idx].item())
                if score > 0:
                    sparse_dict[idx] = score
            sparse_vectors.append(sparse_dict)
        
        return sparse_vectors
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512
    ) -> List[Dict[int, float]]:
        """
        Encode texts in batches (optimized for large batches).
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing (optimal: 32-64)
            max_length: Maximum sequence length
            
        Returns:
            List of sparse vectors
        """
        if not texts:
            return []
        
        all_vectors = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectors = self.encode(batch, max_length=max_length)
            all_vectors.extend(vectors)
        
        return all_vectors
    
    def encode_batch_parallel(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4
    ) -> List[Dict[int, float]]:
        """
        Encode texts in parallel batches (for very large datasets).
        
        Args:
            texts: List of text strings
            batch_size: Batch size per worker
            max_length: Maximum sequence length
            num_workers: Number of parallel workers
            
        Returns:
            List of sparse vectors
        """
        from concurrent.futures import ThreadPoolExecutor
        
        if not texts:
            return []
        
        # Split texts into chunks for each worker
        chunk_size = len(texts) // num_workers
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        all_vectors = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.encode_batch, chunk, batch_size, max_length)
                for chunk in chunks
            ]
            
            for future in futures:
                vectors = future.result()
                all_vectors.extend(vectors)
        
        return all_vectors


# Global encoder instance (lazy loading)
_encoder_instance: Optional[SPLADEEncoder] = None


def get_splade_encoder() -> SPLADEEncoder:
    """Get or create global SPLADE encoder instance."""
    global _encoder_instance
    
    if _encoder_instance is None:
        _encoder_instance = SPLADEEncoder()
    
    return _encoder_instance


if __name__ == "__main__":
    # Test encoder
    print("🧪 Testing SPLADE Encoder")
    print("=" * 60)
    
    encoder = SPLADEEncoder()
    
    test_texts = [
        "What are the treatments for type 2 diabetes?",
        "How does mRNA vaccine technology work?",
        "What is the role of gut microbiome in obesity?"
    ]
    
    print(f"\nEncoding {len(test_texts)} texts...")
    vectors = encoder.encode(test_texts)
    
    for i, (text, vec) in enumerate(zip(test_texts, vectors), 1):
        print(f"\n[{i}] {text[:50]}...")
        print(f"    Sparse vector size: {len(vec)} non-zero values")
        print(f"    Top 5 tokens: {sorted(vec.items(), key=lambda x: x[1], reverse=True)[:5]}")

