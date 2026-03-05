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


def _env_csv(name: str, default: str = "") -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# =============================================================================
# API / CORS Configuration
# =============================================================================
CORS_ALLOWED_ORIGINS = _env_csv(
    "CORS_ALLOWED_ORIGINS",
    default="http://localhost:3000,http://127.0.0.1:3000",
)

# Service-to-service auth
API_AUTH_ENABLED = _env_bool("API_AUTH_ENABLED", default=True)
API_KEYS_FILE = os.getenv("API_KEYS_FILE", "/opt/RAG-pipeline/api_keys.json")
API_KEYS_CACHE_TTL_SECONDS = _env_int("API_KEYS_CACHE_TTL_SECONDS", 30)

# Runtime graceful shutdown controls
API_SHUTDOWN_GRACE_SECONDS = _env_int("API_SHUTDOWN_GRACE_SECONDS", 85)
API_INFLIGHT_DRAIN_POLL_SECONDS = _env_float("API_INFLIGHT_DRAIN_POLL_SECONDS", 0.2)
API_MAX_INFLIGHT_REQUESTS = _env_int("API_MAX_INFLIGHT_REQUESTS", 0)
API_INFLIGHT_ACQUIRE_TIMEOUT_MS = _env_int("API_INFLIGHT_ACQUIRE_TIMEOUT_MS", 250)

# =============================================================================
# Qdrant Cloud Configuration
# =============================================================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", os.getenv("QDRANT_COLLECTION", "rag_pipeline"))
GRAPHRAG_COLLECTION_NAME = os.getenv("GRAPHRAG_COLLECTION_NAME", "pmc_medical_graphrag")
QDRANT_CLOUD_INFERENCE = _env_bool("QDRANT_CLOUD_INFERENCE", default=False)
QDRANT_PREFER_GRPC = _env_bool("QDRANT_PREFER_GRPC", default=True)
QDRANT_GRPC_PORT = _env_int("QDRANT_GRPC_PORT", 6334)
QDRANT_HNSW_EF = _env_int("QDRANT_HNSW_EF", 64)
QDRANT_MAX_INFLIGHT_SEARCHES = _env_int("QDRANT_MAX_INFLIGHT_SEARCHES", 0)
QDRANT_SEARCH_TIMEOUT_SECONDS = _env_int("QDRANT_SEARCH_TIMEOUT_SECONDS", 10)
QDRANT_QUERY_BACKEND = os.getenv("QDRANT_QUERY_BACKEND", "batch").strip().lower()
QDRANT_INDEXED_ONLY = _env_bool("QDRANT_INDEXED_ONLY", default=False)
QDRANT_TIMEOUT = 180  # Client timeout in seconds
QDRANT_RETRY_COUNT = 3  # Number of retries for transient failures
QDRANT_RETRY_DELAY = 2  # Base delay between retries (exponential backoff)

# =============================================================================
# API Provider Configuration
# =============================================================================
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
DEEPINFRA_RETRY_COUNT = int(os.getenv("DEEPINFRA_RETRY_COUNT", "3"))  # Number of retries for transient DeepInfra failures
DEEPINFRA_RETRY_DELAY = float(os.getenv("DEEPINFRA_RETRY_DELAY", "1.0"))  # Base delay in seconds (exponential backoff)
DEEPINFRA_CHAT_TIMEOUT_SECONDS = _env_float("DEEPINFRA_CHAT_TIMEOUT_SECONDS", 300.0)
DEEPINFRA_EMBED_TIMEOUT_SECONDS = _env_float("DEEPINFRA_EMBED_TIMEOUT_SECONDS", 120.0)
DEEPINFRA_RERANK_TIMEOUT_SECONDS = _env_float("DEEPINFRA_RERANK_TIMEOUT_SECONDS", 60.0)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com")
GROQ_RETRY_COUNT = int(os.getenv("GROQ_RETRY_COUNT", "3"))
GROQ_RETRY_DELAY = float(os.getenv("GROQ_RETRY_DELAY", "1.0"))
GROQ_CHAT_TIMEOUT_SECONDS = _env_float("GROQ_CHAT_TIMEOUT_SECONDS", 300.0)

# LLM Generation Parameters
LLM_TEMPERATURE = 0.7  # Controls randomness (0=deterministic, 1=creative)
LLM_TOP_P = 0.9  # Nucleus sampling threshold

# LLM Configuration (Generation)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_MAX_COMPLETION_TOKENS = _env_int("LLM_MAX_COMPLETION_TOKENS", 8192)
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "medium").strip().lower()
LLM_RETRY_COUNT = GROQ_RETRY_COUNT if LLM_PROVIDER == "groq" else DEEPINFRA_RETRY_COUNT
LLM_RETRY_DELAY = GROQ_RETRY_DELAY if LLM_PROVIDER == "groq" else DEEPINFRA_RETRY_DELAY
LLM_CHAT_TIMEOUT_SECONDS = GROQ_CHAT_TIMEOUT_SECONDS if LLM_PROVIDER == "groq" else DEEPINFRA_CHAT_TIMEOUT_SECONDS

# =============================================================================
# Embedding Model Configuration
# =============================================================================
# Provider: "hf_inference_endpoint" (default) | "deepinfra" | "qdrant_cloud_inference"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "deepinfra").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DIMENSION = 1024  # Qwen/Qwen3-Embedding-0.6B output dimension
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# HF Self-Hosted Inference Endpoint (TEI) — used when EMBEDDING_PROVIDER=hf_inference_endpoint
# Endpoint exposes an OpenAI-compatible /v1/embeddings route.
HF_INFERENCE_ENDPOINT_URL = os.getenv(
    "HF_INFERENCE_ENDPOINT_URL",
    "https://jp5vxer2scm0dmve.us-east-1.aws.endpoints.huggingface.cloud",
)
HF_INFERENCE_ENDPOINT_API_KEY = os.getenv("HF_INFERENCE_ENDPOINT_API_KEY")
HF_INFERENCE_EMBED_TIMEOUT_SECONDS = _env_float("HF_INFERENCE_EMBED_TIMEOUT_SECONDS", 60.0)

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
QUANTIZATION_RESCORE = _env_bool("QUANTIZATION_RESCORE", default=False)
QUANTIZATION_OVERSAMPLING = float(os.getenv("QUANTIZATION_OVERSAMPLING", "1.0"))

# Hybrid Search Configuration
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
ENTITY_FILTER_ENABLED = _env_bool("ENTITY_FILTER_ENABLED", default=False)

# Chunk retrieval/reranking profile for chunk-level indexing
# Aligned with .env values for 2048-token chunks (larger chunks = fewer needed)
RETRIEVAL_CHUNK_LIMIT = int(os.getenv("RETRIEVAL_CHUNK_LIMIT", "150"))
MAX_CHUNKS_PER_ARTICLE_PRE_RERANK = int(os.getenv("MAX_CHUNKS_PER_ARTICLE_PRE_RERANK", "2"))
RERANK_INPUT_CHUNK_LIMIT = int(os.getenv("RERANK_INPUT_CHUNK_LIMIT", "90"))
RERANK_TOP_CHUNKS = int(os.getenv("RERANK_TOP_CHUNKS", "75"))
FINAL_TOP_ARTICLES = int(os.getenv("FINAL_TOP_ARTICLES", "50"))
FINAL_RECENCY_POLICY_MODE = os.getenv("FINAL_RECENCY_POLICY_MODE", "hybrid").strip().lower()
FINAL_RECENCY_WINDOW_YEARS = int(os.getenv("FINAL_RECENCY_WINDOW_YEARS", "5"))
FINAL_RECENCY_BACKFILL_MAX_EVIDENCE_LEVEL = int(os.getenv("FINAL_RECENCY_BACKFILL_MAX_EVIDENCE_LEVEL", "2"))
FINAL_RECENCY_EXCLUDE_UNKNOWN_NON_DAILYMED = _env_bool("FINAL_RECENCY_EXCLUDE_UNKNOWN_NON_DAILYMED", default=True)
PMC_FULLTEXT_RECENT_ONLY = _env_bool("PMC_FULLTEXT_RECENT_ONLY", default=True)
RETRIEVAL_RECENCY_BOOST_ENABLED = _env_bool("RETRIEVAL_RECENCY_BOOST_ENABLED", default=True)
RETRIEVAL_RECENCY_APPLY_WITH_YEAR_FILTER = _env_bool("RETRIEVAL_RECENCY_APPLY_WITH_YEAR_FILTER", default=True)
RETRIEVAL_RECENCY_Y1_MULT = float(os.getenv("RETRIEVAL_RECENCY_Y1_MULT", "1.20"))
RETRIEVAL_RECENCY_Y3_MULT = float(os.getenv("RETRIEVAL_RECENCY_Y3_MULT", "1.14"))
RETRIEVAL_RECENCY_Y5_MULT = float(os.getenv("RETRIEVAL_RECENCY_Y5_MULT", "1.08"))
RETRIEVAL_RECENCY_Y7_MULT = float(os.getenv("RETRIEVAL_RECENCY_Y7_MULT", "1.03"))
PRE_RERANK_RECENT_WINDOW_YEARS = int(os.getenv("PRE_RERANK_RECENT_WINDOW_YEARS", "7"))
PRE_RERANK_RECENT_QUOTA_RATIO = float(os.getenv("PRE_RERANK_RECENT_QUOTA_RATIO", "0.35"))
RETRIEVAL_SOURCE_FANOUT_ENABLED = _env_bool("RETRIEVAL_SOURCE_FANOUT_ENABLED", default=False)
RETRIEVAL_SOURCE_FANOUT_MODE = os.getenv("RETRIEVAL_SOURCE_FANOUT_MODE", "parallel").strip().lower()
RETRIEVAL_SOURCE_FANOUT_MIN_RESULTS = _env_int("RETRIEVAL_SOURCE_FANOUT_MIN_RESULTS", 60)
RETRIEVAL_SOURCE_FANOUT_FALLBACK_BROAD = _env_bool(
    "RETRIEVAL_SOURCE_FANOUT_FALLBACK_BROAD",
    default=False,
)

# =============================================================================
# Reranker Configuration
# =============================================================================
# Provider: "deepinfra" (default) or "cross-encoder" (self-hosted)
RERANKER_PROVIDER = "deepinfra"  # Fixed: Only DeepInfra supported
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")

# =============================================================================
# Reranker v2 — Configurable Scoring Constants
# =============================================================================
# Set RERANKER_V2_ENABLED=0 in env to roll back to original scoring behaviour.
RERANKER_V2_ENABLED   = _env_bool("RERANKER_V2_ENABLED", default=True)
# Evidence tier multipliers (v2 defaults — less aggressive than original 3.0/1.5/1.0/0.2)
TIER_1_BOOST          = float(os.getenv("TIER_1_BOOST",   "2.00"))   # guidelines
TIER_2_BOOST          = float(os.getenv("TIER_2_BOOST",   "1.25"))   # RCTs/reviews
TIER_3_BOOST          = float(os.getenv("TIER_3_BOOST",   "1.00"))   # standard research
TIER_4_PENALTY        = float(os.getenv("TIER_4_PENALTY", "0.40"))   # case reports
# Combined score weights (v2: reranker 85%, entity 15%; legacy: 70%/30%)
RERANKER_SCORE_WEIGHT = float(os.getenv("RERANKER_SCORE_WEIGHT", "0.85"))
ENTITY_SCORE_WEIGHT   = float(os.getenv("ENTITY_SCORE_WEIGHT",   "0.15"))

# =============================================================================
# Query Preprocessing Configuration
# =============================================================================
QUERY_EXPANSION_COUNT = int(os.getenv("QUERY_EXPANSION_COUNT", "2"))  # Number of expanded query variations
# Bulk retrieval limits (adjusted for larger 2048-token chunks)
BULK_RETRIEVAL_LIMIT = 200  # Balanced: enough diversity without excessive payload overhead
BULK_RETRIEVAL_PER_QUERY = 100  # Reduced from 150 - more efficient with larger chunks
RERANK_TOP_K = FINAL_TOP_ARTICLES  # Final articles after paper-level aggregation
MAX_ABSTRACTS = FINAL_TOP_ARTICLES  # Context article cap
MAX_DAILYMED_PER_DRUG = 2  # Max DailyMed entries per drug (deduplicate by drug name)

# =============================================================================
# Weekly Update QoS Configuration
# =============================================================================
WEEKLY_UPDATE_THROTTLE_SECONDS = _env_float("WEEKLY_UPDATE_THROTTLE_SECONDS", 0.5)
WEEKLY_UPDATE_BATCH_SIZE = _env_int("WEEKLY_UPDATE_BATCH_SIZE", 0)

# =============================================================================
# Query Caching Configuration
# =============================================================================
QUERY_CACHE_ENABLED = _env_bool("QUERY_CACHE_ENABLED", default=True)
QUERY_CACHE_DIR = os.getenv("QUERY_CACHE_DIR", "data/cache")
QUERY_CACHE_EXPIRY_DAYS = int(os.getenv("QUERY_CACHE_EXPIRY_DAYS", "30"))
QUERY_CACHE_NAMESPACE = os.getenv("QUERY_CACHE_NAMESPACE", "default")
QUERY_CACHE_KEY_VERSION = int(os.getenv("QUERY_CACHE_KEY_VERSION", "2"))

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
    if QDRANT_QUERY_BACKEND not in {"batch", "prefetch"}:
        errors.append("QDRANT_QUERY_BACKEND must be one of: batch, prefetch")
    if RETRIEVAL_SOURCE_FANOUT_MODE not in {"parallel"}:
        errors.append("RETRIEVAL_SOURCE_FANOUT_MODE must be: parallel")
    # Only require DeepInfra key when it is actually used for embeddings or LLM
    if EMBEDDING_PROVIDER == "deepinfra" and not DEEPINFRA_API_KEY:
        errors.append("DEEPINFRA_API_KEY not set (required when EMBEDDING_PROVIDER=deepinfra)")
    if EMBEDDING_PROVIDER not in {"deepinfra", "qdrant_cloud_inference", "hf_inference_endpoint"}:
        errors.append(f"EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}' is not supported")
    if EMBEDDING_PROVIDER == "hf_inference_endpoint" and not HF_INFERENCE_ENDPOINT_API_KEY:
        errors.append("HF_INFERENCE_ENDPOINT_API_KEY not set (required when EMBEDDING_PROVIDER=hf_inference_endpoint)")
    if LLM_PROVIDER not in {"groq", "deepinfra"}:
        errors.append("LLM_PROVIDER must be one of: groq, deepinfra")
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        errors.append("GROQ_API_KEY not set")

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
        if GROQ_API_KEY:
            print(f"✅ GROQ_API_KEY: {GROQ_API_KEY[:20]}...")
        print(f"✅ LLM Model: {LLM_MODEL}")
        print(f"✅ LLM Provider: {LLM_PROVIDER}")
        print(f"✅ DeepInfra Base URL: {DEEPINFRA_BASE_URL}")
        print(f"✅ Groq Base URL: {GROQ_BASE_URL}")
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
