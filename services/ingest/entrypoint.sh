#!/usr/bin/env bash

set -euo pipefail

echo "[Entrypoint] Checking models..."
python /app/services/ingest/download_models.py

echo "[Entrypoint] Starting worker..."
exec "$@"
