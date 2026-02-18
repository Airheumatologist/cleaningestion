"""
Configuration module for Medical RAG Pipeline.
Loads environment variables and provides central settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# =============================================================================
# Qdrant Cloud Configuration
# =============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", os.getenv("QDRANT_COLLECTION", "rag_pipeline"))
GRAPHRAG_COLLECTION_NAME = os.getenv("GRAPHRAG_COLLECTION_NAME", "pmc_medical_graphrag")
QDRANT_CLOUD_INFERENCE = _env_bool("QDRANT_CLOUD_INFERENCE", default=False)
QDRANT_TIMEOUT = 180  # Client timeout in seconds
QDRANT_RETRY_COUNT = 3  # Number of retries for transient failures
QDRANT_RETRY_DELAY = 2  # Base delay between retries (exponential backoff)

# =============================================================================
# LLM Configuration (DeepInfra)
# =============================================================================
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")

DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")

# LLM Generation Parameters
LLM_TEMPERATURE = 0.7  # Controls randomness (0=deterministic, 1=creative)
LLM_TOP_P = 0.9  # Nucleus sampling threshold

# LLM Configuration (Single Model)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

# =============================================================================
# Embedding Model Configuration
# =============================================================================
# Provider: "deepinfra" (default), "local", or "qdrant_cloud_inference"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "deepinfra").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B-batch")
EMBEDDING_DIMENSION = 1024  # Qwen/Qwen3-Embedding-0.6B output dimension
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# =============================================================================
# Chunking Configuration (CRITICAL: Must match .env values)
# =============================================================================
# Optimized for Qwen3-Embedding-0.6B (32k context window)
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "2048"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "256"))

# Quantization Configuration
# =============================================================================
QUANTIZATION_TYPE = os.getenv("QUANTIZATION_TYPE", "scalar").strip().lower()
SCALAR_QUANTILE = float(os.getenv("SCALAR_QUANTILE", "0.99"))
QUANTIZATION_ALWAYS_RAM = _env_bool("QUANTIZATION_ALWAYS_RAM", default=True)
# Search rescore improves accuracy with quantized vectors
QUANTIZATION_RESCORE = _env_bool("QUANTIZATION_RESCORE", default=True)
QUANTIZATION_OVERSAMPLING = float(os.getenv("QUANTIZATION_OVERSAMPLING", "1.5"))

# SPLADE Sparse Vector Configuration
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
USE_HYBRID_SEARCH = _env_bool("USE_HYBRID_SEARCH", default=True)
DENSE_WEIGHT = 0.7  # Weight for dense vector scores in hybrid search
SPARSE_WEIGHT = 0.3  # Weight for sparse vector scores in hybrid search
SPARSE_RETRIEVAL_MODE = os.getenv("SPARSE_RETRIEVAL_MODE", "bm25").strip().lower()
SPARSE_MAX_TERMS_QUERY = int(os.getenv("SPARSE_MAX_TERMS_QUERY", "64"))
SPARSE_MIN_TOKEN_LEN = int(os.getenv("SPARSE_MIN_TOKEN_LEN", "2"))
SPARSE_REMOVE_STOPWORDS = _env_bool("SPARSE_REMOVE_STOPWORDS", default=True)

# =============================================================================
# Search Configuration
# =============================================================================
TOP_K_RESULTS = 5  # Number of documents to retrieve
SCORE_THRESHOLD = 0.25  # Lowered from 0.3 to capture more relevant papers
# Lower threshold helps retrieve more candidates before aggressive filtering

# Chunk retrieval/reranking profile for chunk-level indexing
# Aligned with .env values for 2048-token chunks (larger chunks = fewer needed)
RETRIEVAL_CHUNK_LIMIT = int(os.getenv("RETRIEVAL_CHUNK_LIMIT", "400"))
MAX_CHUNKS_PER_ARTICLE_PRE_RERANK = int(os.getenv("MAX_CHUNKS_PER_ARTICLE_PRE_RERANK", "2"))
RERANK_INPUT_CHUNK_LIMIT = int(os.getenv("RERANK_INPUT_CHUNK_LIMIT", "200"))
RERANK_TOP_CHUNKS = int(os.getenv("RERANK_TOP_CHUNKS", "100"))
FINAL_TOP_ARTICLES = int(os.getenv("FINAL_TOP_ARTICLES", "50"))

# =============================================================================
# Reranker Configuration
# =============================================================================
# Provider: "deepinfra" (default) or "cross-encoder" (self-hosted)
RERANKER_PROVIDER = "deepinfra"  # Fixed: Only DeepInfra supported
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")

# =============================================================================
# Query Preprocessing Configuration
# =============================================================================
QUERY_EXPANSION_COUNT = 2  # Number of expanded query variations (3 total with base query)
# Bulk retrieval limits (adjusted for larger 2048-token chunks)
BULK_RETRIEVAL_LIMIT = 300  # Reduced from 600 - larger chunks need fewer candidates
BULK_RETRIEVAL_PER_QUERY = 100  # Reduced from 150 - more efficient with larger chunks
RERANK_TOP_K = FINAL_TOP_ARTICLES  # Final articles after paper-level aggregation
MAX_ABSTRACTS = FINAL_TOP_ARTICLES  # Context article cap
MAX_DAILYMED_PER_DRUG = 2  # Max DailyMed entries per drug (deduplicate by drug name)

# =============================================================================
# Validation
# =============================================================================
def validate_config():
    """Validate that all required environment variables are set."""
    errors = []

    if not QDRANT_URL:
        errors.append("QDRANT_URL not set in .env")
    if not QDRANT_API_KEY:
        errors.append("QDRANT_API_KEY not set in .env")
    if not DEEPINFRA_API_KEY:
        errors.append("DEEPINFRA_API_KEY not set")

    if errors:
        raise ValueError(
            "Missing configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return True


if __name__ == "__main__":
    # Test configuration loading
    print("🔧 Configuration Check")
    print("=" * 50)
    try:
        validate_config()
        print(f"✅ QDRANT_URL: {QDRANT_URL[:50]}...")
        print(f"✅ QDRANT_API_KEY: {QDRANT_API_KEY[:20]}...")
        print(f"✅ DEEPINFRA_API_KEY: {DEEPINFRA_API_KEY[:20]}...")
        print(f"✅ LLM Model: {LLM_MODEL}")
        print(f"✅ DeepInfra Base URL: {DEEPINFRA_BASE_URL}")
        print(f"✅ Embedding Model: {EMBEDDING_MODEL}")
        print(f"✅ Collection Name: {COLLECTION_NAME}")
        print("\n✅ All configuration validated!")
        
        # Show search configuration
        print("\n🎯 SEARCH CONFIGURATION:")
        print(f"   - BULK_RETRIEVAL_LIMIT: {BULK_RETRIEVAL_LIMIT}")
        print(f"   - RERANK_TOP_K: {RERANK_TOP_K}")
        print(f"   - SCORE_THRESHOLD: {SCORE_THRESHOLD}")
        
    except ValueError as e:
        print(f"❌ Configuration error:\n{e}")
