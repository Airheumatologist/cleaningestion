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
COLLECTION_NAME = os.getenv("COLLECTION_NAME", os.getenv("QDRANT_COLLECTION", "medical_rag"))
GRAPHRAG_COLLECTION_NAME = os.getenv("GRAPHRAG_COLLECTION_NAME", "pmc_medical_graphrag")
QDRANT_CLOUD_INFERENCE = _env_bool("QDRANT_CLOUD_INFERENCE", default=False)
QDRANT_TIMEOUT = 180  # Client timeout in seconds
QDRANT_RETRY_COUNT = 3  # Number of retries for transient failures
QDRANT_RETRY_DELAY = 2  # Base delay between retries (exponential backoff)

# =============================================================================
# LLM Configuration (DeepInfra)
# =============================================================================
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_MODEL = "nvidia/Nemotron-3-Nano-30B-A3B"  # NVIDIA Nemotron 3 Nano model via DeepInfra
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# LLM Generation Parameters
LLM_TEMPERATURE = 0.7  # Controls randomness (0=deterministic, 1=creative)
LLM_TOP_P = 0.9  # Nucleus sampling threshold

# Fallback LLM (for no-results scenarios - larger model with more knowledge)
FALLBACK_LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"
FALLBACK_LLM_ENABLED = True

# Legacy OpenAI (for backward compatibility)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-5-nano"  # GPT-5 Nano model

# =============================================================================
# Groq LLM Configuration (legacy)
# =============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and cost-effective

# =============================================================================
# Embedding Model Configuration
# =============================================================================
# Provider: "cohere" (API, default) or "local" (needs GPU) or "qdrant_cloud_inference"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cohere").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embed-v4.0")
EMBEDDING_DIMENSION = 1024  # Cohere embed-v4.0 supports 256/512/1024/1536; we use 1024

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
RETRIEVAL_CHUNK_LIMIT = int(os.getenv("RETRIEVAL_CHUNK_LIMIT", "800"))
MAX_CHUNKS_PER_ARTICLE_PRE_RERANK = int(os.getenv("MAX_CHUNKS_PER_ARTICLE_PRE_RERANK", "3"))
RERANK_INPUT_CHUNK_LIMIT = int(os.getenv("RERANK_INPUT_CHUNK_LIMIT", "450"))
RERANK_TOP_CHUNKS = int(os.getenv("RERANK_TOP_CHUNKS", "220"))
FINAL_TOP_ARTICLES = int(os.getenv("FINAL_TOP_ARTICLES", "100"))

# =============================================================================
# Reranker Configuration
# =============================================================================
# Provider: "cohere" (API, default) or "cross-encoder" (self-hosted, needs GPU)
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "cohere").strip().lower()
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "mixedbread-ai/mxbai-rerank-large-v2")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# =============================================================================
# Query Preprocessing Configuration
# =============================================================================
QUERY_EXPANSION_COUNT = 2  # Number of expanded query variations (3 total with base query)
BULK_RETRIEVAL_LIMIT = 600  # Increased from 300 for better coverage
BULK_RETRIEVAL_PER_QUERY = 150  # Increased from 75 - more candidates before reranking
RERANK_TOP_K = FINAL_TOP_ARTICLES  # Final articles after paper-level aggregation
MAX_ABSTRACTS = FINAL_TOP_ARTICLES  # Context article cap
MAX_DAILYMED_PER_DRUG = 2  # Max DailyMed entries per drug (deduplicate by drug name)


# =============================================================================
# Multi-Stage Generation Configuration - NEW FOR 150+ ARTICLES
# =============================================================================
USE_MULTI_STAGE_GENERATION = True  # Enable multi-batch LLM generation
BATCH_SIZE_FOR_GENERATION = 50  # Abstracts per generation batch
MAX_GENERATION_BATCHES = 3  # Maximum number of generation batches (50 * 3 = 150 abstracts)
PROGRESSIVE_SYNTHESIS = True  # Use progressive summarization across batches

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
    if RERANKER_PROVIDER == "cohere" and not COHERE_API_KEY:
        errors.append("COHERE_API_KEY not set in .env (required when RERANKER_PROVIDER=cohere)")

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
        print(f"✅ DeepInfra Model: {DEEPINFRA_MODEL}")
        print(f"✅ DeepInfra Base URL: {DEEPINFRA_BASE_URL}")
        print(f"✅ Embedding Model: {EMBEDDING_MODEL}")
        print(f"✅ Collection Name: {COLLECTION_NAME}")
        print("\n✅ All configuration validated!")
        
        # Show optimized settings for 150+ articles
        print("\n🎯 OPTIMIZED FOR 150+ ARTICLES:")
        print(f"   - BULK_RETRIEVAL_LIMIT: {BULK_RETRIEVAL_LIMIT} (increased)")
        print(f"   - RERANK_TOP_K: {RERANK_TOP_K} (increased)")
        print(f"   - SCORE_THRESHOLD: {SCORE_THRESHOLD} (lowered)")
        print(f"   - MAX_GENERATION_BATCHES: {MAX_GENERATION_BATCHES}")
        print(f"   - BATCH_SIZE_FOR_GENERATION: {BATCH_SIZE_FOR_GENERATION}")
        
    except ValueError as e:
        print(f"❌ Configuration error:\n{e}")
