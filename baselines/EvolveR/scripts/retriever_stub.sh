#!/usr/bin/env bash
set -euo pipefail
EVOR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${EVOR}:${PYTHONPATH:-}"
export RETRIEVER_STUB_PORT="${RETRIEVER_STUB_PORT:-19999}"
exec python3 "$EVOR/scripts/retriever_stub.py"
