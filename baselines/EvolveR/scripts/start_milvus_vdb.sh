#!/usr/bin/env bash
# Start experience VDB (see evolver/experience/milvusdb/db_server.py).
# Example:
#   export EMBEDDING_API_URL=http://127.0.0.1:8081/v1
#   export VDB_BASE_DIR=$PWD/outputs/milvus_work
#   export EXPERIMENT_NAME=deepmath_evolver
#   bash scripts/start_milvus_vdb.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVOR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/common_env.sh"
MILVUS_DIR="$EVOR/evolver/experience/milvusdb"
cd "$MILVUS_DIR"
# Match upstream naming (hyphens to underscores in env for Milvus)
export EXPERIMENT_NAME="${EXPERIMENT_NAME//-/_}"
export EMBEDDING_MODEL_NAME
mkdir -p "$VDB_BASE_DIR"
exec python3 db_server.py
