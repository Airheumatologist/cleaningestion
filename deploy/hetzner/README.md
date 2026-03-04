# Hetzner Deployment: Co-Located RAG API + Qdrant

This folder contains production assets for running the stack on one Hetzner host:
- `qdrant` (internal-only on Docker network)
- `rag-api-1..4` (FastAPI/Gunicorn replicas on internal network)
- `rag-gateway` (Nginx on host port `8000`)

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
```

## 3) Configure environment

Create `/opt/RAG-pipeline/.env` from `env.example`:

```bash
cp /opt/RAG-pipeline/env.example /opt/RAG-pipeline/.env
```

Required production edits in `.env`:

```env
QDRANT_URL=http://qdrant:6333
QDRANT_GRPC_URL=qdrant:6334
QDRANT_PREFER_GRPC=true
QDRANT_GRPC_PORT=6334
QDRANT_HNSW_EF=64
QDRANT_API_KEY=<generate-with-openssl-rand-hex-32>

DEEPINFRA_API_KEY=<deepinfra-key>
API_AUTH_ENABLED=true
API_KEYS_FILE=/opt/RAG-pipeline/api_keys.json
API_MAX_INFLIGHT_REQUESTS=128
API_INFLIGHT_ACQUIRE_TIMEOUT_MS=200

# Browser CORS can stay local-only if callers are backend services.
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

## 4) Create service token hash file

```bash
cd /opt/RAG-pipeline
cp api_keys.example.json api_keys.json

# Generate bcrypt hash from a plaintext token
python3 scripts/hash_service_token.py
```

Place the resulting hash in `api_keys.json` under `token_hash`.

## 5) Start Qdrant + API stack

First, pre-download the massive NLM MeSH Dictionary files directly to the host to prevent concurrent worker download corruption:

```bash
mkdir -p /opt/RAG-pipeline/data/mesh
cd /opt/RAG-pipeline/data/mesh
curl -O https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml
curl -O https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/supp2026.xml
```

Then, start the stack:

```bash
cd /opt/RAG-pipeline/deploy/hetzner
docker compose --env-file ../../.env up -d --build
```

Validate:

```bash
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
```

## 6) Install Python dependencies for ingestion jobs

```bash
cd /opt/RAG-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 7) Create collection (with sparse config)

```bash
cd /opt/RAG-pipeline
source .env
python scripts/05_setup_qdrant.py --collection-name rag_pipeline
```

## 8) Pilot ingestion

```bash
python scripts/06_ingest_pmc.py --xml-dir /data/ingestion/pmc_xml
python scripts/07_ingest_dailymed.py --xml-dir /data/ingestion/dailymed/xml
```

## 9) Install cron jobs

```bash
sudo cp deploy/hetzner/cron/medical-rag-update.cron /etc/cron.d/medical-rag-update
sudo cp deploy/hetzner/cron/qdrant-health.cron /etc/cron.d/qdrant-health
sudo cp deploy/hetzner/cron/api-health.cron /etc/cron.d/api-health
sudo chmod 644 /etc/cron.d/medical-rag-update /etc/cron.d/qdrant-health /etc/cron.d/api-health
```

Optional cleanup if legacy backup cron exists:

```bash
sudo rm -f /etc/cron.d/qdrant-backup
```

## 10) Firewall baseline

```bash
# Allow SSH
ufw allow OpenSSH

# Allow WireGuard (example default port)
ufw allow 51820/udp

# Allow API only from private tunnel/network CIDR
ufw allow from <WIREGUARD_CIDR> to any port 8000 proto tcp
ufw deny 8000/tcp

# Qdrant is not host-published in compose
ufw enable
ufw status
```

## 11) API key rotation

```bash
# 1) Generate a new plaintext service token
openssl rand -hex 32

# 2) Hash it
cd /opt/RAG-pipeline
python3 scripts/hash_service_token.py --token "<new-token>"

# 3) Replace token_hash in /opt/RAG-pipeline/api_keys.json
# 4) Restart API services only (NOTE: isolate API + gateway to avoid touching qdrant)
cd /opt/RAG-pipeline/deploy/hetzner
docker compose --env-file ../../.env up -d --no-deps rag-api-1 rag-api-2 rag-api-3 rag-api-4 rag-gateway
```

## 12) Acceptance checks

| Check | Command |
|---|---|
| API health | `curl -s localhost:8000/api/v1/health` |
| Auth (missing token -> 401) | `curl -i -X POST localhost:8000/api/v1/chat -H "content-type: application/json" -d '{"query":"test"}'` |
| Qdrant internal-only | `docker compose ps` (no host `6333`/`6334` published) |
| Cron installed | `ls /etc/cron.d/medical-rag-update /etc/cron.d/qdrant-health /etc/cron.d/api-health` |
| Stream endpoint | `curl -N -X POST localhost:8000/api/v1/chat/stream -H "Authorization: Bearer <token>" -H "content-type: application/json" -d '{"query":"test", "stream": true}'` |
