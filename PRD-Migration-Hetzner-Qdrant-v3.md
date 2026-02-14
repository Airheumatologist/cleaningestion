# PRD: Migrate Medical RAG to Hetzner Self-Hosted Qdrant

**Version:** 3.0 | **Date:** February 2026  
**Goal:** Replace EC2 + Qdrant Cloud with a single Hetzner dedicated server running self-hosted Qdrant + automated monthly updates via cron. Zero AWS dependency. Reranking via Cohere Rerank 4 Fast API. Production RAG switches by changing one endpoint.

---

## How This Works (Plain English)

### What is "self-hosted Qdrant"?

**Important: Qdrant the software ≠ Qdrant the company's paid products.**

Qdrant's website advertises paid tiers — "Qdrant Cloud" (managed hosting), "Hybrid Cloud" (enterprise), and "Private Cloud" (enterprise). Those are their commercial products and require subscriptions/enterprise accounts. **We are NOT using any of these.**

Qdrant the **software** is fully open-source under the Apache 2.0 license, published on [GitHub](https://github.com/qdrant/qdrant) and [DockerHub](https://hub.docker.com/r/qdrant/qdrant). Anyone can download and run it for free, forever, with:

- **No Qdrant account needed**
- **No license key or subscription**
- **No feature restrictions** — 100% of the functionality, the exact same binary that runs on Qdrant Cloud
- **No connection back to Qdrant's servers**

It's exactly like running PostgreSQL or Redis on your own server — you install the open-source software and use it. The entire "self-hosting" process is one Docker command:

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v /opt/qdrant/storage:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY=your-secret-key \
  qdrant/qdrant:v1.15.0
```

That pulls the free public image from DockerHub, starts Qdrant, and exposes the API. Done. Your RAG app connects to `http://your-server-ip:6333` instead of `https://your-cluster.qdrant.io:6333`.

Here's what we're doing instead of paying Qdrant Cloud:

1. **Rent a dedicated physical server** from Hetzner (~€64/mo) — a real machine with 64GB RAM and 2TB of fast NVMe SSDs sitting in a Hetzner datacenter in Germany
2. **Install Qdrant on it** via Docker — one command, runs as a background service
3. **Qdrant exposes the exact same API** on `http://<your-server-ip>:6333` that Qdrant Cloud exposes on `https://your-cluster.qdrant.io:6333`
4. **Your RAG code changes one line** — the `QDRANT_URL` in your `.env` file. Everything else (search, filtering, BQ, metadata) works identically because it's the same software

### What runs on this server?

```
┌─────────────────────────────────────────────────────┐
│  Hetzner Dedicated Server (AX52)                     │
│  AMD Ryzen 7 7700, 64GB RAM, 2×1TB NVMe             │
│  Ubuntu 24.04, Docker                                │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Qdrant (Docker container) — runs 24/7           │ │
│  │  • Stores all 8M vectors + payloads on NVMe     │ │
│  │  • Serves search queries via REST/gRPC API      │ │
│  │  • Uses ~10-14 GB RAM (2-bit BQ + HNSW index)   │ │
│  │  • Port 6333 (REST) and 6334 (gRPC)             │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Monthly Cron Job — runs ~30-60 min per month    │ │
│  │  • Downloads PubMed/DailyMed updates from NCBI  │ │
│  │  • Parses XML, generates embeddings              │ │
│  │  • Upserts new vectors into Qdrant locally       │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Automated Backups — daily cron                   │ │
│  │  • Qdrant snapshot → Hetzner Object Storage      │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘

Your production app (wherever it's hosted) connects to this
server over the internet, just like it connects to Qdrant Cloud today.
```

### What about the initial 8M document ingestion?

That's a one-time heavy job (~8-16 hours). We spin up a **temporary** cheap Hetzner cloud VPS (CAX41, ~$0.04/hr), run the full PubMed download + parse + upsert pipeline from there into the dedicated server's Qdrant, then delete the VPS. Total cost: <$1.

After that, the dedicated server handles everything — serving queries AND running monthly updates.

---

## 1. Current vs Target

### Current
| Component | Monthly Cost |
|---|---|
| AWS EC2 g5.xlarge (always-on for ingestion) | ~$730 |
| Qdrant Cloud (8M docs, 4 shards, BQ) | ~$100-200+ |
| AWS S3 (~300GB PubMed data) | ~$7 |
| **Total** | **~$850-950/mo** |

### Target
| Component | Monthly Cost |
|---|---|
| Hetzner AX52 dedicated server (Qdrant + cron updates) | €64 (~$69) |
| Hetzner Object Storage (backups) | ~€3-5 |
| Cohere Rerank 4 Fast API | ~$0-20 (usage-based) |
| Qdrant Cloud Inference API (embeddings only, no DB) | ~$0-10 |
| **Total** | **~€70-100/mo (~$75-105)** |

**No AWS. No Qdrant Cloud DB. One server runs Qdrant + monthly updates.**

**Savings: ~$750-875/mo (~$9,000-10,500/yr) — 88-92% reduction**

---

## 2. Capacity Planning for 8M Vectors with 2-Bit Quantization

### Why 2-bit instead of binary (1-bit)?

Your current setup uses binary quantization (1-bit) with mxbai-embed-large-v1 (1024 dimensions). At 1024d, binary quantization can lose meaningful precision — values near zero get rounded poorly. Qdrant 1.15+ introduced **2-bit quantization** which:

- Uses 2 bits per dimension instead of 1 → **16x compression** (vs 32x for binary)
- Explicitly encodes zero values, fixing the main weakness of binary at 1024d
- RAM cost is ~2x more than binary but still tiny compared to full float32
- Better recall with less oversampling/rescoring needed → faster search

### RAM Math

```
8,000,000 vectors × 1024 dimensions

2-bit quantized vectors in RAM (always_ram=true):
  8M × 1024 × 2 bits / 8 = 2,048 MB ≈ 2 GB

HNSW graph index in RAM (on_disk=false):
  8M nodes × m(16) × 2 links × 8 bytes × 1.5 overhead ≈ 3-4 GB

Payload indexes in RAM (year, source, article_type, journal, evidence_grade, country):
  ~2-4 GB depending on cardinality

OS + Docker + Qdrant process:
  ~2-3 GB

Monthly cron job (when running):
  ~2-4 GB (Python parsing + embedding)

TOTAL: ~13-17 GB active
AVAILABLE: 64 GB
HEADROOM: ~47 GB → becomes OS page cache for disk reads (huge speed boost)
```

### Disk Math

```
Original vectors on disk (on_disk=true, kept for rescoring):
  8M × 1024 × 4 bytes = 32 GB

Payloads on disk (on_disk_payload=true):
  ~5-8 KB avg per doc × 8M = ~52 GB
  With 1.5x Qdrant overhead: ~78 GB

HNSW + WAL + optimization temp: ~30 GB

TOTAL Qdrant data: ~150-200 GB
Temporary during initial ingestion: +300 GB (PubMed XML downloads)

Server has 2×1 TB NVMe = 2 TB total → plenty of room
```

### Server: Hetzner Dedicated AX52

| Spec | Value |
|---|---|
| CPU | AMD Ryzen 7 7700 (8 cores / 16 threads, 5.3 GHz boost) |
| RAM | 64 GB DDR5 ECC |
| Storage | 2× 1 TB NVMe SSD |
| Network | 1 Gbit/s, 20 TB included traffic/mo |
| Price | ~€64/mo (~$69) |
| Setup fee | €39 one-time |
| Location | Falkenstein (DE) or Helsinki (FI) |

**Why dedicated, not cloud VPS:**
- Cloud VPS max is 48 GB RAM (CCX series) — not enough headroom
- Dedicated = local NVMe (not network-attached). Qdrant needs high random-read IOPS for HNSW traversal. Local NVMe does 100k-500k IOPS; network storage does 10k-30k
- No noisy neighbors — consistent search latency
- 2× 1TB drives — use one for OS+Qdrant, one for temp ingestion data

**Budget alternative:** AX42 (Ryzen 7 PRO 8700GE, 64GB DDR5, 2×512GB NVMe) at ~€49/mo. Less storage but works if you stream-process during ingestion.

---

## 3. Phase 1 — Set Up the Dedicated Server

### 3A. Order and Provision

1. Go to [Hetzner Robot](https://robot.hetzner.com) → Dedicated Servers → Order AX52
2. Select location (Falkenstein recommended — good EU connectivity)
3. Install Ubuntu 24.04 LTS via their rescue system
4. SSH in:

```bash
ssh root@<your-server-ip>

# Update
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker

# Install tools
apt install -y htop iotop fio ufw fail2ban python3-pip python3-venv git

# Firewall — IMPORTANT: restrict Qdrant ports to your production server IP
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow from YOUR_PRODUCTION_SERVER_IP to any port 6333  # Qdrant REST
ufw allow from YOUR_PRODUCTION_SERVER_IP to any port 6334  # Qdrant gRPC
ufw enable
```

### 3B. Set Up Disk Layout

```bash
# Disk 1 (nvme0n1): OS + Qdrant data (already mounted as /)
# Disk 2 (nvme1n1): Ingestion workspace + backups

mkfs.ext4 /dev/nvme1n1
mkdir -p /data
mount /dev/nvme1n1 /data
echo '/dev/nvme1n1 /data ext4 defaults 0 2' >> /etc/fstab
mkdir -p /data/ingestion /data/backups
```

### 3C. Deploy Qdrant

```bash
mkdir -p /opt/qdrant/qdrant_storage /opt/qdrant/qdrant_config
cd /opt/qdrant
```

**`/opt/qdrant/docker-compose.yml`:**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.15.0  # Pin version; 1.15+ needed for 2-bit quant
    restart: always
    container_name: qdrant
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC (faster for bulk operations)
    volumes:
      - ./qdrant_storage:/qdrant/storage
      - ./qdrant_config:/qdrant/config
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
      - QDRANT__LOG_LEVEL=INFO
```

**`/opt/qdrant/qdrant_config/production.yaml`:**

```yaml
storage:
  performance:
    max_search_threads: 0        # Auto-detect (uses all 16 threads)
    max_optimization_threads: 4  # Background HNSW rebuilds

service:
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
```

**Start it:**

```bash
# Generate API key
export QDRANT_API_KEY=$(openssl rand -hex 32)
echo "QDRANT_API_KEY=${QDRANT_API_KEY}" > .env
echo "SAVE THIS KEY: ${QDRANT_API_KEY}"

# Launch
docker compose up -d

# Verify
curl -H "api-key: ${QDRANT_API_KEY}" http://localhost:6333/healthz
# Should return: {"title":"qdrant - vectorass engine","version":"1.15.0"}
```

### 3D. Create Collection with 2-Bit Quantization

```python
# Run this from the server or any machine with qdrant-client installed
# pip install qdrant-client

from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="http://YOUR_HETZNER_IP:6333",
    api_key="YOUR_QDRANT_API_KEY",
)

# Create collection — matches your existing schema but with 2-bit quant
client.create_collection(
    collection_name="medical_rag",
    vectors_config=models.VectorParams(
        size=1024,                          # mxbai-embed-large-v1
        distance=models.Distance.COSINE,
        on_disk=True,                       # Full vectors on disk (loaded for rescoring)
    ),
    quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            always_ram=True,                # Keep quantized vectors in RAM
        )
    ),
    # NOTE: Qdrant 1.15+ supports 2-bit via update after creation.
    # Create with binary first, then patch to 2-bit:
    hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=128,
        on_disk=False,                      # HNSW graph stays in RAM
    ),
    shard_number=4,                         # Match your current 4-shard setup
    on_disk_payload=True,                   # Heavy payloads (full_text) on disk
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=0,               # DISABLE indexing during bulk upload
    ),
)

# Update to 2-bit quantization (Qdrant 1.15+ API)
client.update_collection(
    collection_name="medical_rag",
    quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            always_ram=True,
        ),
        # If your Qdrant version supports the 2-bit parameter directly:
        # Use the REST API instead:
    ),
)
# Alternative via REST API for 2-bit:
# PATCH /collections/medical_rag
# { "quantization_config": { "binary": { "type": "2bit", "always_ram": true } } }

# Create payload indexes — these stay in RAM for fast filtered search
index_fields = {
    "year": models.PayloadSchemaType.INTEGER,
    "source": models.PayloadSchemaType.KEYWORD,
    "article_type": models.PayloadSchemaType.KEYWORD,
    "journal": models.PayloadSchemaType.KEYWORD,
    "evidence_grade": models.PayloadSchemaType.KEYWORD,
    "country": models.PayloadSchemaType.KEYWORD,
}
for field_name, field_type in index_fields.items():
    client.create_payload_index(
        collection_name="medical_rag",
        field_name=field_name,
        field_schema=field_type,
    )

print("Collection created successfully.")
print(client.get_collection("medical_rag"))
```

> **Important:** Check the exact Qdrant 1.15+ API for 2-bit config syntax. It may require setting `quantization_config` via REST PATCH with `{"binary": {"type": "2bit"}}`. If 2-bit isn't available in your client library version yet, use the REST API directly or start with standard binary and upgrade later — it's a non-destructive change (Qdrant re-quantizes in background).

---

## 4. Phase 2 — Initial Baseline Ingestion (One-Time)

### Temporary Compute: Hetzner Cloud CAX41

| Spec | Value |
|---|---|
| CPU | 16 ARM vCPUs (Ampere Altra) |
| RAM | 32 GB |
| Storage | 320 GB NVMe |
| Price | €0.0384/hr (~$0.042/hr) |
| Est. runtime | 8-16 hours |
| **Total cost** | **<$1** |

> If ARM has compatibility issues with your Python deps (lxml, sentence-transformers), use **CPX51** (16 AMD vCPU, 32 GB RAM, 360 GB NVMe) at €0.097/hr (~$2 total).

### Why a separate ingestion server?

- The initial 8M doc load is CPU/network intensive — don't want it competing with Qdrant serving queries
- 300 GB PubMed download needs temp disk space — keep it off the production Qdrant drives
- It's disposable: create it, run the job, delete it. Costs <$1

### Script Changes Required

**`scripts/config_ingestion.py` — NEW centralized config:**

```python
import os

class IngestionConfig:
    # Self-hosted Qdrant on Hetzner dedicated server
    QDRANT_URL = os.getenv("QDRANT_URL", "http://YOUR_HETZNER_DEDICATED_IP:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_GRPC_URL = os.getenv("QDRANT_GRPC_URL", "http://YOUR_HETZNER_DEDICATED_IP:6334")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_rag")

    # Embedding — keep using Qdrant Cloud Inference API (no DB subscription needed)
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "qdrant_cloud_inference")
    QDRANT_INFERENCE_URL = os.getenv("QDRANT_INFERENCE_URL", "")
    QDRANT_INFERENCE_KEY = os.getenv("QDRANT_INFERENCE_KEY", "")

    # Data paths (local filesystem — NO S3)
    DATA_DIR = os.getenv("DATA_DIR", "/data/ingestion")

    # Tuning
    BATCH_SIZE = 100
    EMBEDDING_BATCH_SIZE = 128
    MAX_WORKERS = 8
    USE_GRPC = True  # gRPC is 2-3x faster for bulk upserts
```

**`scripts/01_download_pmc.py` — Replace S3 with direct NCBI FTP:**

```python
# BEFORE: aws s3 sync s3://pmc-oa-opendata/ /data/pmc_fulltext/
# AFTER: Direct FTP download (free, no AWS needed)

import ftplib, os, gzip

FTP_HOST = "ftp.ncbi.nlm.nih.gov"
BASELINE_PATH = "/pub/pmc/oa_bulk/"

def download_baseline(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()
    ftp.cwd(BASELINE_PATH)
    files = sorted([f for f in ftp.nlst() if f.endswith('.xml.gz')])
    print(f"Found {len(files)} baseline files")

    for i, fname in enumerate(files):
        local_path = os.path.join(output_dir, fname)
        if os.path.exists(local_path):
            continue
        print(f"[{i+1}/{len(files)}] {fname}")
        with open(local_path, 'wb') as f:
            ftp.retrbinary(f'RETR {fname}', f.write)
    ftp.quit()
```

**`scripts/06_ingest_pmc.py` — Stream processing to fit 320 GB disk:**

```python
# KEY CHANGE: Process one .xml.gz at a time instead of extracting everything
#
# Loop:
#   1. Read one .xml.gz file
#   2. Parse articles from it (lxml iterparse)
#   3. Generate embeddings (via API, batched)
#   4. Upsert vectors to Qdrant (via gRPC for speed)
#   5. Delete the .xml.gz file from disk
#   6. Next file
#
# This keeps disk usage under 50 GB at any time
# instead of needing 300 GB of extracted XML
```

**`scripts/05_setup_qdrant.py` — Point to self-hosted:**

```python
# Just change the client initialization:
from config_ingestion import IngestionConfig
from qdrant_client import QdrantClient

client = QdrantClient(
    url=IngestionConfig.QDRANT_URL,
    api_key=IngestionConfig.QDRANT_API_KEY,
)
# Everything else (collection creation) stays the same
```

### Run the Baseline Ingestion

```bash
# 1. Create CAX41 on Hetzner Cloud console (Ubuntu 24.04, Falkenstein)

# 2. SSH in
ssh root@<cax41-ip>
apt update && apt install -y python3-pip python3-venv git

# 3. Clone and setup
git clone https://github.com/Airheumatologist/RAG-pipeline.git
cd RAG-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 4. Set environment
cat > .env << 'EOF'
QDRANT_URL=http://<DEDICATED-SERVER-IP>:6333
QDRANT_API_KEY=<your-qdrant-api-key>
DATA_DIR=/data/ingestion
EMBEDDING_PROVIDER=qdrant_cloud_inference
QDRANT_INFERENCE_URL=https://...
QDRANT_INFERENCE_KEY=...
EOF
export $(cat .env | xargs)

# 5. Run pipeline (use screen/tmux so it survives SSH disconnect)
tmux new -s ingestion

python scripts/01_download_pmc.py          # Download PMC baseline (~4-8 hrs)
python scripts/06_ingest_pmc.py            # Parse + embed + upsert (~4-8 hrs)
python scripts/03_download_dailymed.py     # Download DailyMed (~30 min)
python scripts/04_process_dailymed.py      # Process DailyMed
python scripts/07_ingest_dailymed.py       # Ingest DailyMed (~10 min)

# 6. Re-enable indexing on the dedicated server after all uploads
python -c "
from qdrant_client import QdrantClient, models
c = QdrantClient(url='http://<DEDICATED-SERVER-IP>:6333', api_key='<key>')
c.update_collection('medical_rag',
    optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000))
print('Indexing re-enabled. HNSW will rebuild in background.')
"

# 7. Verify
python -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='http://<DEDICATED-SERVER-IP>:6333', api_key='<key>')
info = c.get_collection('medical_rag')
print(f'Points: {info.points_count:,}')
print(f'Indexed: {info.indexed_vectors_count:,}')
print(f'Status: {info.status}')
"
# Expected: Points: ~8,000,000, Status: green

# 8. DELETE the CAX41 in Hetzner Cloud console — you're done with it
```

### Ingestion Speed Tips

- **Disable indexing first** (`indexing_threshold=0`) — prevents HNSW from rebuilding after every batch. Re-enable after all 8M docs are uploaded for a single optimized build
- **Use gRPC** (port 6334) for upserts — 2-3x faster than REST for bulk
- **Parallelize embedding calls** — 4-8 threads, batch size 128-256
- **Embedding is the bottleneck**: At ~200 embeddings/sec via API, 8M docs ≈ 11 hours. XML parsing is fast in comparison

---

## 5. Phase 3 — Monthly Updates via Cron (On the Dedicated Server)

### No AWS Needed

The dedicated server runs Qdrant using ~15 GB RAM. That leaves ~49 GB free. Monthly PubMed updates are ~100-500 MB of XML — the server handles this trivially alongside Qdrant.

### Setup

```bash
# On the dedicated server:

# 1. Clone your repo (if not already there)
cd /opt
git clone https://github.com/Airheumatologist/RAG-pipeline.git
cd RAG-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Create .env for the update script
cat > /opt/RAG-pipeline/.env << 'EOF'
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=<your-key>
DATA_DIR=/data/ingestion
EMBEDDING_PROVIDER=qdrant_cloud_inference
QDRANT_INFERENCE_URL=https://...
QDRANT_INFERENCE_KEY=...
EOF

# 3. Create the cron job
cat > /etc/cron.d/medical-rag-update << 'EOF'
# Run monthly PubMed + DailyMed update on 1st of each month at 2 AM UTC
SHELL=/bin/bash
0 2 1 * * root cd /opt/RAG-pipeline && source venv/bin/activate && source .env && python scripts/08_monthly_update.py >> /var/log/rag-update.log 2>&1
EOF

# 4. Set up log rotation
cat > /etc/logrotate.d/rag-update << 'EOF'
/var/log/rag-update.log {
    monthly
    rotate 12
    compress
    missingok
    notifempty
}
EOF
```

### `scripts/08_monthly_update.py` Updates

```python
#!/usr/bin/env python3
"""Monthly PubMed + DailyMed update — runs via cron on dedicated server."""

import ftplib
import os
import json
import logging
from datetime import datetime
from config_ingestion import IngestionConfig
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

def get_processed_files():
    """Track which update files we've already processed."""
    tracker_path = os.path.join(IngestionConfig.DATA_DIR, "processed_updates.json")
    if os.path.exists(tracker_path):
        return set(json.load(open(tracker_path)))
    return set()

def save_processed_files(processed):
    tracker_path = os.path.join(IngestionConfig.DATA_DIR, "processed_updates.json")
    json.dump(sorted(processed), open(tracker_path, 'w'))

def fetch_and_process_updates():
    client = QdrantClient(
        url=IngestionConfig.QDRANT_URL,
        api_key=IngestionConfig.QDRANT_API_KEY
    )

    processed = get_processed_files()

    # Download new PubMed update files
    ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov")
    ftp.login()
    ftp.cwd("/pubmed/updatefiles/")
    all_files = sorted([f for f in ftp.nlst() if f.endswith('.xml.gz')])
    new_files = [f for f in all_files if f not in processed]

    log.info(f"Found {len(new_files)} new update files (of {len(all_files)} total)")

    for fname in new_files:
        log.info(f"Processing {fname}...")
        local_path = os.path.join(IngestionConfig.DATA_DIR, fname)

        # Download
        with open(local_path, 'wb') as f:
            ftp.retrbinary(f'RETR {fname}', f.write)

        # Parse + embed + upsert (reuse existing ingestion logic)
        # ... your existing parse/embed/upsert code here ...

        # Cleanup
        os.remove(local_path)
        processed.add(fname)
        save_processed_files(processed)

    ftp.quit()

    # Also check DailyMed for updates
    # ... similar logic for DailyMed API ...

    log.info(f"Monthly update complete. Collection status:")
    info = client.get_collection(IngestionConfig.COLLECTION_NAME)
    log.info(f"  Points: {info.points_count:,}")
    log.info(f"  Status: {info.status}")

if __name__ == "__main__":
    log.info(f"=== Monthly RAG Update Started {datetime.now().isoformat()} ===")
    fetch_and_process_updates()
    log.info(f"=== Monthly RAG Update Finished ===")
```

### Monitoring the Cron Job

```bash
# Check last run
tail -50 /var/log/rag-update.log

# Test manually
cd /opt/RAG-pipeline && source venv/bin/activate
python scripts/08_monthly_update.py

# Optional: send email/Slack on failure
# Add to cron: ... || curl -X POST https://hooks.slack.com/... -d '{"text":"RAG update failed"}'
```

---

## 6. Phase 4 — Switch Production RAG

### What Changes

**Only your `.env` file:**

```bash
# BEFORE (Qdrant Cloud)
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=cloud-api-key

# AFTER (Self-hosted on Hetzner)
QDRANT_URL=http://YOUR_HETZNER_DEDICATED_IP:6333
QDRANT_API_KEY=your-self-hosted-key
```

**No code changes to:**
- `src/retriever_qdrant.py` — uses QdrantClient which is endpoint-agnostic
- `src/rag_pipeline.py` — same
- `src/api_server.py` — same
- `frontend/` — same (talks to your backend, not Qdrant directly)

**One small code change:**
- `src/reranker.py` — update Cohere model from `rerank-v3.5` to `rerank-v4-fast`

### Reranker: Switch to Cohere Rerank 4 Fast

Update your `src/reranker.py` to use Cohere's latest model:

```python
# BEFORE
result = co.rerank(
    model="rerank-v3.5",
    query=query,
    documents=documents,
    top_n=top_n,
)

# AFTER
result = co.rerank(
    model="rerank-v4-fast",
    query=query,
    documents=documents,
    top_n=top_n,
)
```

**Why Rerank 4 Fast:**
- Cohere's newest reranker (2025), optimized for speed while maintaining high accuracy
- Lower latency than v3.5 — critical when handling 10+ concurrent requests
- Better price/performance ratio for high-volume usage
- Drop-in replacement — same API, same input/output format, one parameter change
- Strong performance on domain-specific (medical/scientific) content

### Embedding Strategy

Your current setup uses Qdrant Cloud Inference for embeddings during both ingestion and search. Options going forward:

| Option | What Changes | Cost | Notes |
|---|---|---|---|
| **A. Keep Qdrant Cloud Inference** | Nothing — use their inference API separately from their DB | ~$0-10/mo | Simplest. You only cancel the DB subscription, keep inference |
| **B. Self-host embeddings** | Install sentence-transformers on dedicated server; modify retriever | $0 | AX52's CPU can embed 1 query in <100ms. Fine for search. Slow for bulk ingestion |
| **C. Use FastEmbed** | `pip install fastembed`; smaller/faster CPU-optimized models | $0 | May need different model; test recall first |

**Recommendation:** Option A for now. Zero code changes. Cancel only the Qdrant Cloud database, keep the inference API.

### Cutover Checklist

```
[ ] 8M docs in self-hosted Qdrant (verify exact count matches Cloud)
[ ] Run 20+ test queries — compare top-10 results vs Cloud
[ ] Measure search latency (target: <500ms p95)
[ ] Update .env on production API server (QDRANT_URL → Hetzner IP)
[ ] Update src/reranker.py model to "rerank-v4-fast"
[ ] Restart api_server.py
[ ] Monitor for 24-48 hours
[ ] Decommission Qdrant Cloud DB subscription (keep inference API)
[ ] Terminate AWS EC2 instance
[ ] Delete AWS EBS volumes
[ ] Delete S3 bucket (or keep as cold archive if desired)
```

---

## 7. Scalability — Can This Handle 10,000 Users?

### At 10 req/sec (10K users, moderate usage): Single AX52 is fine

Here's what happens per request in your pipeline and where time is spent:

| Step | What It Does | Latency | Bottleneck? |
|---|---|---|---|
| 1. Query preprocessing | MeSH expansion, LLM decomposition | ~200-500ms | LLM API |
| 2. Embedding (query) | Embed user query for vector search | ~50-100ms | API call |
| 3. **Qdrant vector search** | Search 8M vectors + filtering | **~20-50ms** | **No — trivial** |
| 4. **Cohere Rerank 4 Fast** | Rerank top 100 passages | **~300-600ms** | API rate limit |
| 5. LLM synthesis | Generate final response with citations | ~2-5s | **Main bottleneck** |
| 6. PDF check | Europe PMC API | ~200-500ms | External API |
| **Total per request** | | **~3-7 seconds** | |

**Qdrant at 10 RPS is using ~2% of its capacity.** Qdrant benchmarks show 100-500+ RPS on a single node with 8M vectors and quantization. The AX52's Ryzen 7 with 64GB RAM is massively overpowered for 10 req/sec of vector search.

**The real bottleneck is external API calls** — Cohere reranking, LLM completion, embedding. These are rate-limited by the provider, not by your server. At 10 concurrent requests you need sufficient API quota from each provider.

**FastAPI on the AX52** handles 10 concurrent connections easily with async. Run with multiple workers:

```bash
# Deploy with 4 Uvicorn workers to use multiple cores
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Scaling Tiers

| Scale | Users | RPS | Architecture | Monthly Cost |
|---|---|---|---|---|
| **Current plan** | 10K | ~10 | Single AX52 (Qdrant + cron) | ~€67-69 |
| **Medium** | 50K | ~50 | AX52 (Qdrant) + separate VPS (FastAPI) | ~€80-90 |
| **Large** | 100K+ | ~200+ | AX162 (48-core EPYC, €199) or 2× AX52 with Qdrant replication | ~€200-300 |

**At 50-100 req/sec (if you grow):**
- Move FastAPI to a separate Hetzner VPS (€5-20/mo) so it doesn't share CPU with Qdrant
- Qdrant on AX52 still handles 50-100 RPS easily
- You may need higher-tier API plans from Cohere/OpenRouter
- Add Redis caching for repeated/similar queries (~30-50% cache hit rate for medical queries)

**At 200+ req/sec (major growth):**
- Add a second Qdrant node with replication (double the read RPS)
- Or upgrade to AX162 (AMD EPYC 9454P, 48 cores, 128 GB RAM, ~€199/mo) for a single powerful node
- Load-balance multiple FastAPI workers across VPS instances
- Consider self-hosting the reranker at this scale to eliminate Cohere rate limits

### Network Latency Consideration

If your production app server is US-based and Qdrant is in Germany (Falkenstein), expect ~100-120ms round-trip latency for each Qdrant call. Options:
- **Hetzner Ashburn, VA datacenter** — if available for AX52, co-locates with US infrastructure
- **Accept the latency** — at 100ms per Qdrant call, it's still small vs the 3-7s total pipeline time
- **Co-locate everything** — run FastAPI on the same Hetzner server or a nearby Hetzner VPS

---

## 8. Production Hardening

### Automated Backups

```bash
# /etc/cron.d/qdrant-backup
# Daily at 3 AM — create Qdrant snapshot, upload to Hetzner Object Storage

SHELL=/bin/bash
QDRANT_API_KEY=<your-key>

0 3 * * * root /opt/qdrant/backup.sh >> /var/log/qdrant-backup.log 2>&1
```

**`/opt/qdrant/backup.sh`:**

```bash
#!/bin/bash
set -e

API_KEY="${QDRANT_API_KEY}"
COLLECTION="medical_rag"
BACKUP_DIR="/data/backups"
S3_BUCKET="s3://your-hetzner-object-storage-bucket/qdrant-backups"

# Create snapshot via Qdrant API
SNAP=$(curl -s -X POST \
  -H "api-key: ${API_KEY}" \
  "http://localhost:6333/collections/${COLLECTION}/snapshots" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])")

echo "$(date): Snapshot created: ${SNAP}"

# Copy snapshot to backup disk
cp /opt/qdrant/qdrant_storage/snapshots/${COLLECTION}/${SNAP} ${BACKUP_DIR}/

# Upload to Hetzner Object Storage (S3-compatible)
# Requires: pip install awscli; configure with Hetzner S3 credentials
aws s3 cp ${BACKUP_DIR}/${SNAP} ${S3_BUCKET}/${SNAP} --endpoint-url https://fsn1.your-objectstorage.hetzner.com

# Keep only last 7 local snapshots
ls -t ${BACKUP_DIR}/*.snapshot 2>/dev/null | tail -n +8 | xargs rm -f 2>/dev/null || true

echo "$(date): Backup complete"
```

### Health Monitoring

```bash
# /etc/cron.d/qdrant-health
# Check every 5 minutes

*/5 * * * * root curl -sf -H "api-key: ${QDRANT_API_KEY}" http://localhost:6333/healthz > /dev/null || \
  curl -X POST https://hooks.slack.com/YOUR_WEBHOOK -d '{"text":"⚠️ Qdrant is DOWN on Hetzner"}'
```

### Security Checklist

```
[x] UFW firewall — Qdrant ports open ONLY to production server IP
[x] API key authentication on Qdrant
[x] fail2ban for SSH brute-force protection
[x] unattended-upgrades for OS security patches
[ ] Optional: Caddy reverse proxy for TLS (if exposing publicly)
[ ] Optional: WireGuard VPN between production app and Qdrant server
```

### TLS (if needed)

```bash
apt install caddy

# /etc/caddy/Caddyfile
qdrant.yourdomain.com {
    reverse_proxy localhost:6333
    # Caddy auto-provisions Let's Encrypt TLS
}

systemctl restart caddy

# Now your production app connects to:
# QDRANT_URL=https://qdrant.yourdomain.com
```

---

## 9. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Medical RAG Pipeline (New Architecture)             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Your Production App Server (wherever hosted):                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Query    │→ │ Retrieve │→ │ Rerank   │→ │ Synth    │           │
│  │ Preproc  │  │ (Qdrant) │  │(Cohere4F)│  │ (LLM)   │           │
│  └──────────┘  └────┬─────┘  └──────────┘  └──────────┘           │
│                      │                                               │
│                      │ HTTP/gRPC (just an API call)                  │
│                      ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Hetzner Dedicated Server AX52                               │    │
│  │  (64 GB RAM, 2×1TB NVMe, AMD Ryzen 7 7700)                 │    │
│  │  €64/mo — this is the ONLY infrastructure you pay for        │    │
│  │                                                               │    │
│  │  ┌───────────────────────────────────┐                       │    │
│  │  │ Qdrant (Docker, always running)    │                       │    │
│  │  │ • 8M vectors, 2-bit quantization   │                       │    │
│  │  │ • 4 shards, HNSW in RAM            │                       │    │
│  │  │ • ~15 GB RAM used / 64 GB avail    │                       │    │
│  │  │ • ~200 GB disk / 2 TB avail        │                       │    │
│  │  │ • REST :6333 + gRPC :6334          │                       │    │
│  │  └───────────────────────────────────┘                       │    │
│  │                                                               │    │
│  │  ┌───────────────────────────────────┐                       │    │
│  │  │ Monthly Cron Job (1st of month)    │                       │    │
│  │  │ • Fetch PubMed/DailyMed updates   │                       │    │
│  │  │ • Parse → Embed → Upsert locally  │                       │    │
│  │  │ • ~30-60 min runtime              │                       │    │
│  │  └───────────────────────────────────┘                       │    │
│  │                                                               │    │
│  │  ┌───────────────────────────────────┐                       │    │
│  │  │ Daily Backup (3 AM cron)           │                       │    │
│  │  │ • Qdrant snapshot → Object Storage│                       │    │
│  │  └───────────────────────────────────┘                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  One-Time Initial Load:                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Hetzner Cloud CAX41 (temp, deleted after) → FTP → Qdrant    │    │
│  │ Cost: <$1                                                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Final Cost Summary

| Item | Cost | Frequency |
|---|---|---|
| Hetzner AX52 dedicated server | €64/mo (~$69) | Monthly |
| Hetzner Object Storage (backups) | ~€3-5/mo | Monthly |
| Cohere Rerank 4 Fast API | Usage-based (~$0-20/mo at 10 RPS) | Monthly |
| Qdrant Cloud Inference (embeddings) | ~$0-10/mo | Monthly |
| Hetzner CAX41 (initial ingestion) | <€1 | One-time |
| AX52 setup fee | €39 | One-time |
| **Ongoing total** | **~€70-100/mo (~$75-105)** | |
| **vs current** | **~$850-950/mo** | |
| **Annual savings** | **~$9,000-10,500/yr** | |

---

## 11. File Changes Summary

| File | Action | What |
|---|---|---|
| `scripts/config_ingestion.py` | **NEW** | Centralized config (Qdrant URL, embedding, paths) |
| `scripts/01_download_pmc.py` | **MODIFY** | S3 sync → direct NCBI FTP |
| `scripts/02_extract_pmc.py` | **MODIFY** | Add stream-processing mode (per-file, delete after) |
| `scripts/05_setup_qdrant.py` | **MODIFY** | Self-hosted endpoint; 2-bit quantization config |
| `scripts/06_ingest_pmc.py` | **MODIFY** | Stream processing; gRPC upserts; self-hosted endpoint |
| `scripts/07_ingest_dailymed.py` | **MODIFY** | Self-hosted endpoint |
| `scripts/08_monthly_update.py` | **MODIFY** | Direct FTP; file tracker; runs standalone via cron |
| `src/config.py` | **MODIFY** | Ensure QDRANT_URL is purely env-driven |
| `src/retriever_qdrant.py` | **NO CHANGE** | Endpoint-agnostic |
| `src/reranker.py` | **MODIFY** | Switch model from `rerank-v3.5` to `rerank-v4-fast` |
| `.env` | **MODIFY** | Change QDRANT_URL to Hetzner IP |
| `/opt/qdrant/docker-compose.yml` | **NEW** (on server) | Qdrant deployment |
| `/opt/qdrant/backup.sh` | **NEW** (on server) | Automated backup script |

---

## 12. Implementation Timeline

| Day | Task |
|---|---|
| **Day 1** | Order Hetzner AX52. Install Ubuntu + Docker + Qdrant. Create collection with 2-bit quant. Update repo scripts. |
| **Day 2** | Spin up CAX41. Run baseline ingestion (~8-16 hrs, let it run overnight). |
| **Day 3** | Verify 8M docs. Run test queries. Compare search results with Qdrant Cloud. Re-enable HNSW indexing. |
| **Day 3** | Switch production `.env` → Hetzner. Monitor. |
| **Day 4** | Set up cron (monthly updates + daily backups). Security hardening. |
| **Day 5** | Decommission Qdrant Cloud + EC2 + S3. Delete CAX41. |
