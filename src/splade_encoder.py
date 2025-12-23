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
    """SPLADE encoder for generating sparse vectors."""
    
    def __init__(self, model_name: str = SPLADE_MODEL, device: Optional[str] = None):
        """
        Initialize SPLADE encoder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        logger.info(f"Loading SPLADE model: {model_name} on {device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            logger.info("✅ SPLADE model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading SPLADE model: {e}")
            raise
    
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

