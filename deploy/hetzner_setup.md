# Hetzner Qdrant Setup Guide

## 1. Create Server (Hetzner Cloud)

1.  **Login** to [Hetzner Cloud Console](https://console.hetzner.cloud/).
2.  **Create Project**: Name it `medical-rag`.
3.  **Add Server**:
    *   **Location**: Ashburn, VA (or Falkenstein/Nuremberg if latency isn't critical and GDPR is key). *Ashburn is best for US users.*
    *   **Image**: `Ubuntu 24.04` (or 22.04).
    *   **Type**:
        *   **Recommendation**: **AX52** (AMD Ryzen 7000, 64GB RAM, NVMe). Great performance/price ratio.
        *   **Minimum**: **CX42** (32 GB RAM). Might be tight for 80M+ chunks even with binary quantization.
    *   **Networking**: Select `IPv4` and `IPv6`.
    *   **SSH Key**: Add your local public key (`~/.ssh/id_rsa.pub`).
    *   **Name**: `qdrant-prod-01`.
4.  **Create & Buy**.

## 2. Server Initialization

SSH into your new server:
```bash
ssh root@<SERVER_IP>
```

### Update & Install Docker
```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose plugin (should be included, but verify)
docker compose version
```

### Security Setup (UFW Firewall)
Secure the server to only allow SSH and Qdrant (6333) from allowed IPs.

```bash
# Allow SSH
ufw allow OpenSSH

# Allow Qdrant API
ufw allow 6333/tcp

# Enable Firewall
ufw enable
```

## 3. Deploy Qdrant

Create the deployment directory:
```bash
mkdir -p /opt/qdrant
cd /opt/qdrant
```

Create `docker-compose.yaml`:
```yaml
services:
  qdrant:
   image: qdrant/qdrant:v1.12.0
   restart: always
   container_name: qdrant
   ports:
     - "6333:6333"
   volumes:
     - ./storage:/qdrant/storage:z
     - ./snapshots:/qdrant/snapshots:z
   environment:
     - QDRANT__SERVICE__API_KEY=your-secure-api-key-here
   ulimits:
     nofile:
       soft: 65535
       hard: 65535
```

**IMPORTANT**: Generate a strong API key (e.g., `openssl rand -hex 32`) and replace `your-secure-api-key-here`.

Start the service:
```bash
docker compose up -d
```

Verify it's running:
```bash
docker logs -f qdrant
# OR
curl http://localhost:6333/collections
```

## 4. Client Configuration

Update your local `.env` file (or `env.example` -> `.env`) with the new connection details:

```env
QDRANT_URL=http://<SERVER_IP>:6333
QDRANT_API_KEY=<YOUR_GENERATED_KEY>
QDRANT_COLLECTION=rag_pipeline
```

## 5. Performance Tuning (OS Level)

For high-throughput, optimize sysctl settings:

```bash
# Add to /etc/sysctl.conf
echo "vm.max_map_count=262144" >> /etc/sysctl.conf
echo "vm.swappiness=1" >> /etc/sysctl.conf
sysctl -p
```
