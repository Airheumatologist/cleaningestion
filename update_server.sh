#!/bin/bash
# Update server and recreate collection

SERVER_IP="65.109.112.253"
REMOTE_DIR="/opt/RAG-pipeline"

echo "Connecting to $SERVER_IP..."
ssh -o BatchMode=yes -o ConnectTimeout=10 root@$SERVER_IP << EOF
    set -e
    echo "Updating repository..."
    cd $REMOTE_DIR
    git pull origin main || git pull https://github.com/Airheumatologist/RagVPS.git main
    
    echo "Running recreation script..."
    chmod +x scripts/recreate_collection_for_cohere.sh
    ./scripts/recreate_collection_for_cohere.sh
EOF

if [ $? -eq 0 ]; then
    echo "✅ Server update complete!"
else
    echo "❌ Failed to update server. Please check SSH connection."
    exit 1
fi
