# Hetzner Self-Hosted Deployment (v4 – Dense + BM25 Sparse)

This folder contains production assets to run Qdrant on a Hetzner dedicated server.

## 1) Prepare server

```bash
sudo apt update && sudo apt upgrade -y
curl -fsSL https://get.docker.com | sh
sudo systemctl enable docker
sudo apt install -y python3-venv python3-pip git ufw fail2ban
```

## 2) Clone repo and prepare directories

```bash
cd /opt
sudo git clone <your-repo-url> RAG-pipeline
sudo chown -R "$USER":"$USER" /opt/RAG-pipeline
mkdir -p /opt/qdrant/qdrant_storage /opt/qdrant/qdrant_config /data/ingestion /data/backups
cp /opt/RAG-pipeline/deploy/hetzner/qdrant-production.yaml /opt/qdrant/qdrant_config/production.yaml
cp /opt/RAG-pipeline/deploy/hetzner/docker-compose.yml /opt/qdrant/docker-compose.yml
```

## 3) Configure environment

Create `/opt/RAG-pipeline/.env` from `env.example`:

```bash
cp /opt/RAG-pipeline/env.example /opt/RAG-pipeline/.env
```

Edit `.env` with production values:

```env
QDRANT_URL=http://localhost:6333
QDRANT_GRPC_URL=localhost:6334
QDRANT_API_KEY=<generate-with-openssl-rand-hex-32>
QDRANT_COLLECTION=rag_pipeline
COLLECTION_NAME=rag_pipeline
QDRANT_CLOUD_INFERENCE=false

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1
EMBEDDING_BATCH_SIZE=64

SPARSE_ENABLED=true
SPARSE_MODE=bm25
USE_HYBRID_SEARCH=true
SPARSE_RETRIEVAL_MODE=bm25

EMBED_FILTER_ENABLED=true
EMBED_FILTER_MODE=conservative
CHUNK_SIZE_TOKENS=384
CHUNK_OVERLAP_TOKENS=64

DATA_DIR=/data/ingestion
```

## 4) Start Qdrant

```bash
cd /opt/qdrant
cp /opt/RAG-pipeline/.env .env
docker compose up -d
curl -H "api-key: ${QDRANT_API_KEY}" http://localhost:6333/healthz
```

## 5) Install Python dependencies

```bash
cd /opt/RAG-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 6) Create collection (with sparse config)

This **must** be run to guarantee the sparse vector config (BM25 with IDF modifier) is applied:

```bash
source .env
python scripts/05_setup_qdrant.py --collection-name rag_pipeline
```

Verify schema includes sparse vector config:
```bash
curl -s -H "api-key: ${QDRANT_API_KEY}" http://localhost:6333/collections/rag_pipeline | python3 -m json.tool | grep -A3 sparse
```

Expected: `"sparse"` key with `"modifier": "idf"`.

## 7) Pilot ingestion

Run a small-scale ingestion to validate the pipeline:

```bash
# PMC (limit to a small XML directory for pilot)
python scripts/06_ingest_pmc.py --xml-dir /data/ingestion/pmc_xml

# DailyMed
python scripts/07_ingest_dailymed.py --xml-dir /data/ingestion/dailymed/xml
```

## 8) Install cron jobs

```bash
sudo cp deploy/hetzner/cron/medical-rag-update.cron /etc/cron.d/medical-rag-update
sudo cp deploy/hetzner/cron/qdrant-backup.cron /etc/cron.d/qdrant-backup
sudo cp deploy/hetzner/cron/qdrant-health.cron /etc/cron.d/qdrant-health
sudo chmod 644 /etc/cron.d/medical-rag-update /etc/cron.d/qdrant-backup /etc/cron.d/qdrant-health
```

## 9) Firewall allowlist

```bash
# Allow SSH
ufw allow OpenSSH

# Allow Qdrant only from specific IPs (ingestion host + prod app)
ufw allow from <INGESTION_IP> to any port 6333
ufw allow from <PROD_APP_IP> to any port 6333

# Block all other Qdrant access
ufw deny 6333/tcp

ufw enable
ufw status
```

## 10) API key rotation

```bash
# Generate new key
NEW_KEY=$(openssl rand -hex 32)
echo "New key: ${NEW_KEY}"

# Update docker-compose env
sed -i "s/QDRANT_API_KEY=.*/QDRANT_API_KEY=${NEW_KEY}/" /opt/qdrant/.env
cd /opt/qdrant && docker compose down && docker compose up -d

# Update application .env
sed -i "s/QDRANT_API_KEY=.*/QDRANT_API_KEY=${NEW_KEY}/" /opt/RAG-pipeline/.env

# Update all external client .env files with the new key
```

## 11) Acceptance checks

| Check | Command |
|---|---|
| Collection has sparse config | `curl -s -H "api-key: $QDRANT_API_KEY" localhost:6333/collections/rag_pipeline \| grep idf` |
| Points have dense + sparse vectors | `curl -s -H "api-key: $QDRANT_API_KEY" "localhost:6333/collections/rag_pipeline/points/scroll?limit=1&with_vectors=true"` |
| Cron installed | `ls /etc/cron.d/medical-rag-update /etc/cron.d/qdrant-backup /etc/cron.d/qdrant-health` |
| Firewall active | `ufw status` |
| Monthly update runs | `tail -f /var/log/rag-update.log` |
