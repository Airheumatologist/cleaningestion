#!/bin/bash
# Production Deployment Script for RAG Pipeline
# Run this on the production server

set -e  # Exit on error

echo "=========================================="
echo "🚀 RAG Pipeline Production Deployment"
echo "=========================================="

# Navigate to project directory
cd ~/RAG-pipeline || cd /opt/RAG-pipeline || cd /var/www/RAG-pipeline
echo "📁 Working directory: $(pwd)"

# Pull latest code
echo ""
echo "📥 Pulling latest code from production..."
git pull origin main

# Activate virtual environment
echo ""
echo "🐍 Activating virtual environment..."
source .venv/bin/activate || source venv/bin/activate

# Install updated dependencies
echo ""
echo "📦 Installing updated dependencies..."
pip install -r requirements.txt

# Verify environment variables
echo ""
echo "🔍 Verifying environment configuration..."
if [ -z "$DEEPINFRA_API_KEY" ]; then
    echo "❌ ERROR: DEEPINFRA_API_KEY not set"
    exit 1
fi
if [ -z "$QDRANT_URL" ]; then
    echo "❌ ERROR: QDRANT_URL not set"
    exit 1
fi
if [ -z "$QDRANT_API_KEY" ]; then
    echo "❌ ERROR: QDRANT_API_KEY not set"
    exit 1
fi

echo "✅ Environment variables verified"
echo "   - DEEPINFRA_API_KEY: ${DEEPINFRA_API_KEY:0:10}..."
echo "   - QDRANT_URL: $QDRANT_URL"
echo "   - EMBEDDING_MODEL: ${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B-batch}"

# Test embedding setup
echo ""
echo "🧪 Testing embedding configuration..."
python3 << 'EOF'
import os
os.environ.setdefault('EMBEDDING_PROVIDER', 'deepinfra')
os.environ.setdefault('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B-batch')

from scripts.ingestion_utils import EmbeddingProvider
try:
    provider = EmbeddingProvider()
    test_vectors = provider.embed_batch(["Test medical text"])
    print(f"✅ DeepInfra API test successful - Vector dim: {len(test_vectors[0])}")
except Exception as e:
    print(f"❌ Embedding test failed: {e}")
    exit(1)
EOF

echo ""
echo "=========================================="
echo "✅ Deployment successful!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "   1. Start ingestion: ./start_ingestion.sh"
echo "   2. Monitor logs: tail -f /data/ingestion/*.log"
echo ""
