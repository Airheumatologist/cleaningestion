# Hetzner Self-Hosted Deployment

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

Create `/opt/RAG-pipeline/.env`:

```bash
QDRANT_URL=http://localhost:6333
QDRANT_GRPC_URL=localhost:6334
QDRANT_API_KEY=<generate-random-secret>
QDRANT_COLLECTION=medical_rag
QDRANT_CLOUD_INFERENCE=false

# Optional external embedding provider
EMBEDDING_PROVIDER=qdrant_cloud_inference
QDRANT_INFERENCE_URL=
QDRANT_INFERENCE_KEY=

DATA_DIR=/data/ingestion
```

## 4) Start Qdrant

```bash
cd /opt/qdrant
cp /opt/RAG-pipeline/.env .env
docker compose up -d
curl -H "api-key: ${QDRANT_API_KEY}" http://localhost:6333/healthz
```

## 5) Create collection

```bash
cd /opt/RAG-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/05_setup_qdrant.py --collection-name medical_rag --keep-existing
```

## 6) Install cron jobs

```bash
sudo cp deploy/hetzner/cron/medical-rag-update.cron /etc/cron.d/medical-rag-update
sudo cp deploy/hetzner/cron/qdrant-backup.cron /etc/cron.d/qdrant-backup
sudo cp deploy/hetzner/cron/qdrant-health.cron /etc/cron.d/qdrant-health
sudo chmod 644 /etc/cron.d/medical-rag-update /etc/cron.d/qdrant-backup /etc/cron.d/qdrant-health
```

## 7) Verify

```bash
tail -n 200 /var/log/rag-update.log
curl -H "api-key: ${QDRANT_API_KEY}" http://localhost:6333/collections/medical_rag
```
