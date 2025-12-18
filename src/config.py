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

# =============================================================================
# LLM Configuration (OpenRouter)
# =============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"  # Nvidia Nemotron model via OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
USE_HYBRID_SEARCH = False  # Disabled - using dense vectors only (SPLADE not ingested)
DENSE_WEIGHT = 0.7  # Weight for dense vector scores in hybrid search
SPARSE_WEIGHT = 0.3  # Weight for sparse vector scores in hybrid search

# =============================================================================
# Search Configuration
# =============================================================================
TOP_K_RESULTS = 5  # Number of documents to retrieve
SCORE_THRESHOLD = 0.3  # Minimum similarity score

# =============================================================================
# Cohere Rerank Configuration
# =============================================================================
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# =============================================================================
# Query Preprocessing Configuration
# =============================================================================
QUERY_EXPANSION_COUNT = 4  # Number of expanded query variations
BULK_RETRIEVAL_LIMIT = 200  # Max candidates before reranking
BULK_RETRIEVAL_PER_QUERY = 50  # Candidates per expanded query
RERANK_TOP_K = 20  # Final articles after reranking
FULL_TEXT_COUNT = 10  # Articles with full text in output
ABSTRACT_ONLY_COUNT = 10  # Articles with abstract only in output

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
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY not set in .env")
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
        print(f"✅ OPENROUTER_API_KEY: {OPENROUTER_API_KEY[:20]}...")
        print(f"✅ OpenRouter Model: {OPENROUTER_MODEL}")
        print(f"✅ OpenRouter Base URL: {OPENROUTER_BASE_URL}")
        print(f"✅ Embedding Model: {EMBEDDING_MODEL}")
        print(f"✅ Collection Name: {COLLECTION_NAME}")
        print("\n✅ All configuration validated!")
    except ValueError as e:
        print(f"❌ Configuration error:\n{e}")
