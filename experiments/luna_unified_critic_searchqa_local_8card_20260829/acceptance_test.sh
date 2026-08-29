#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa/local_retriever_8card_20260829}
PORT=${PORT:-18002}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
mkdir -p "$RUN_ROOT"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BASE="$(dirname "$0")"
{
  echo "=== $(date) local SearchQA acceptance ==="
  "$ENV_ROOT/bin/python" "$BASE/benchmark_local_retriever.py" \
    --url "http://127.0.0.1:$PORT/retrieve" \
    --health-url "http://127.0.0.1:$PORT/health"
  "$ENV_ROOT/bin/python" "$BASE/probe_searchqa_env.py" \
    --search-url "http://127.0.0.1:$PORT/retrieve"
  echo "=== $(date) acceptance passed ==="
} 2>&1 | tee "$RUN_ROOT/acceptance.log"
