#!/usr/bin/env bash
# Start experience VDB (see evolver/experience/milvusdb/db_server.py).
# Example:
#   export EMBEDDING_API_URL=http://127.0.0.1:8081/v1
#   export VDB_BASE_DIR=$PWD/outputs/milvus_work
#   export EXPERIMENT_NAME=deepmath_evolver
#   bash scripts/start_milvus_vdb.sh
set -euo pipefail
EVOR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${EVOR}:${PYTHONPATH:-}"
export EMBEDDING_API_URL="${EMBEDDING_API_URL:-http://127.0.0.1:8081/v1}"
export EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-empty}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-bge_m3}"
MILVUS_DIR="$EVOR/evolver/experience/milvusdb"
cd "$MILVUS_DIR"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-default_exp}"
# Match upstream naming (hyphens to underscores in env for Milvus)
export EXPERIMENT_NAME="${EXPERIMENT_NAME//-/_}"
exec python3 db_server.py
