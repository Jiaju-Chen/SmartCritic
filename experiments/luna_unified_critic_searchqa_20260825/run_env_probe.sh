#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
PORT=${PORT:-8010}
LOG_DIR=${LOG_DIR:-/tmp/luna_searchqa_probe}
mkdir -p "$LOG_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/experiments/luna_unified_critic_searchqa_20260825/mini_retrieval_server.py" \
  --port "$PORT" >"$LOG_DIR/retriever.log" 2>&1 &
RETRIEVER_PID=$!
trap 'kill "$RETRIEVER_PID" 2>/dev/null || true; wait "$RETRIEVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$PROJECT_ROOT/experiments/luna_unified_critic_searchqa_20260825/probe_search_env.py" \
  --search-url "http://127.0.0.1:$PORT/retrieve"
