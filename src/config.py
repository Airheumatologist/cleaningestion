"""
Configuration module for Medical RAG Pipeline.
Loads environment variables and provides central settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# =============================================================================
# Qdrant Cloud Configuration
# =============================================================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "pmc_medical_rag_fulltext"
GRAPHRAG_COLLECTION_NAME = "pmc_medical_graphrag"  # New Graph RAG enhanced collection
QDRANT_CLOUD_INFERENCE = True  # Use Qdrant cloud inference for embeddings
QDRANT_TIMEOUT = 180  # Client timeout in seconds
QDRANT_RETRY_COUNT = 3  # Number of retries for transient failures
QDRANT_RETRY_DELAY = 2  # Base delay between retries (exponential backoff)

# =============================================================================
# LLM Configuration (DeepInfra)
# =============================================================================
DEEPINFRA_API_KEY = "Huum9P1B1P9UHu92sMY2AWF99GaVthTw"
DEEPINFRA_MODEL = "nvidia/Nemotron-3-Nano-30B-A3B"  # NVIDIA Nemotron 3 Nano model via DeepInfra
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# LLM Generation Parameters
LLM_TEMPERATURE = 0.7  # Controls randomness (0=deterministic, 1=creative)
LLM_TOP_P = 0.9  # Nucleus sampling threshold

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
# Using mixedbread mxbai-embed-large-v1 with Qdrant Cloud Inference (1024-d)
# Embeddings are generated server-side in Qdrant Cloud
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_DIMENSION = 1024

# SPLADE Sparse Vector Configuration
SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
USE_HYBRID_SEARCH = True  # Enabled - 100% of documents have sparse vectors
DENSE_WEIGHT = 0.7  # Weight for dense vector scores in hybrid search
SPARSE_WEIGHT = 0.3  # Weight for sparse vector scores in hybrid search

# =============================================================================
# Search Configuration - OPTIMIZED FOR 150+ ARTICLES
# =============================================================================
TOP_K_RESULTS = 5  # Number of documents to retrieve
SCORE_THRESHOLD = 0.25  # Lowered from 0.3 to capture more relevant papers
# Lower threshold helps retrieve more candidates before aggressive filtering

# =============================================================================
# Cohere Rerank Configuration
# =============================================================================
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# =============================================================================
# Query Preprocessing Configuration - SCALED UP FOR 150+ ARTICLES
# =============================================================================
QUERY_EXPANSION_COUNT = 2  # Number of expanded query variations (3 total with base query)
BULK_RETRIEVAL_LIMIT = 600  # Increased from 300 for better coverage
BULK_RETRIEVAL_PER_QUERY = 150  # Increased from 75 - more candidates before reranking
RERANK_TOP_K = 100  # Increased from 30 to 100 - Final articles after reranking (keep more before aggregation)
MAX_ABSTRACTS = 100  # Maximum abstracts to use in context - reduced from 150 to save tokens
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
    if not COHERE_API_KEY:
        errors.append("COHERE_API_KEY not set in .env")

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
