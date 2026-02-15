#!/bin/bash
# Fix "Too many open files" error for Qdrant/RocksDB
# Run this as root on the Hetzner server

set -e

echo "Current ulimit: $(ulimit -n)"

# 1. Set for current session
ulimit -n 65535

# 2. Persist in /etc/security/limits.conf
if ! grep -q "root soft nofile 65535" /etc/security/limits.conf; then
    echo "Adding limits to /etc/security/limits.conf..."
    echo "* soft nofile 65535" >> /etc/security/limits.conf
    echo "* hard nofile 65535" >> /etc/security/limits.conf
    echo "root soft nofile 65535" >> /etc/security/limits.conf
    echo "root hard nofile 65535" >> /etc/security/limits.conf
fi

# 3. Increase fs.file-max
if ! grep -q "fs.file-max" /etc/sysctl.conf; then
    echo "Increasing fs.file-max..."
    echo "fs.file-max = 2097152" >> /etc/sysctl.conf
    sysctl -p
fi

# 4. Update Docker daemon config (optional but good practice)
if [ ! -f /etc/docker/daemon.json ]; then
    echo "Creating /etc/docker/daemon.json..."
    echo '{ "default-ulimits": { "nofile": { "Name": "nofile", "Hard": 65535, "Soft": 65535 } } }' > /etc/docker/daemon.json
    systemctl restart docker
fi

echo "✅ System limits updated."
echo "New ulimit: $(ulimit -n)"

echo "Restarting Qdrant to apply changes..."
cd /opt/RAG-pipeline/deploy/hetzner
docker compose down
docker compose up -d

echo "✅ Qdrant restarted with new limits."
