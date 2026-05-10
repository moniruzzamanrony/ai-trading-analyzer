#!/bin/bash

set -euo pipefail

PASSWORD='AVNSHiB7Cg0kpa4D8JOESP'
REMOTE="root@213.199.36.174"
REMOTE_DIR="/root/trading-ai-analyzer-api"
ARCHIVE="/tmp/trading-ai-analyzer-api.tar.gz"
LOG_FILE="deploy.log"

SSH_OPTS="-o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=20 \
  -o TCPKeepAlive=yes \
  -o ConnectTimeout=30"

SSH="sshpass -p $PASSWORD ssh $SSH_OPTS $REMOTE"
SCP="sshpass -p $PASSWORD scp $SSH_OPTS"
RSYNC_RSH="sshpass -p $PASSWORD ssh $SSH_OPTS"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===================================="
echo "Deployment started at $(date)"
echo "===================================="

# -----------------------------
# Package source
# -----------------------------
echo "Packaging source..."
tar --exclude='.git' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    --exclude='deploy.log' \
    --exclude='.idea' \
    --exclude='.vscode' \
    --exclude='node_modules' \
    --exclude="$(basename "$ARCHIVE")" \
    -czf "$ARCHIVE" .

ARCHIVE_SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo "Archive size: $ARCHIVE_SIZE"

# -----------------------------
# Upload (rsync with resume)
# -----------------------------
echo "Uploading to server..."
$SSH "mkdir -p $REMOTE_DIR"

if command -v rsync >/dev/null 2>&1; then
  rsync -avz --partial --progress \
    -e "$RSYNC_RSH" \
    "$ARCHIVE" "$REMOTE:$REMOTE_DIR/app.tar.gz"
else
  echo "rsync not found, falling back to scp..."
  $SCP "$ARCHIVE" "$REMOTE:$REMOTE_DIR/app.tar.gz"
fi

rm -f "$ARCHIVE"

# -----------------------------
# Build and deploy
# -----------------------------
echo "Building and deploying..."
$SSH "
  set -e
  cd $REMOTE_DIR
  tar -xzf app.tar.gz
  rm -f app.tar.gz
  docker compose down
  docker compose up --build -d
"

echo "===================================="
echo "Deployment finished at $(date)"
echo "===================================="
