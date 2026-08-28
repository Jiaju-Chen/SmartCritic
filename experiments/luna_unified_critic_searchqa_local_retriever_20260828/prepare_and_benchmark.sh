#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
ASSET_ROOT=${ASSET_ROOT:-/home/dataset-local/cjj/RL/data/searchR1_official_retriever}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa/local_retriever_20260828}
TRANSFER_SESSION=${TRANSFER_SESSION:-searchqa_local_assets_20260828}
RETRIEVER_SESSION=${RETRIEVER_SESSION:-searchqa_local_retriever_20260828}
PORT=${PORT:-18002}
mkdir -p "$RUN_ROOT"

while tmux has-session -t "$TRANSFER_SESSION" 2>/dev/null; do
  echo "[$(date)] waiting for asset transfer"
  sleep 60
done

test -s "$ASSET_ROOT/index/e5_Flat.index"
test -s "$ASSET_ROOT/corpus/wiki-18.jsonl"
test -s "$ASSET_ROOT/models/e5-base-v2/config.json"

if ! tmux has-session -t "$RETRIEVER_SESSION" 2>/dev/null; then
  tmux new-session -d -s "$RETRIEVER_SESSION" \
    "cd '$PROJECT_ROOT' && PORT='$PORT' FAISS_GPU=0 RETRIEVER_GPU=7 FAISS_OMP_THREADS=16 bash experiments/luna_unified_critic_searchqa_local_retriever_20260828/run_local_retriever.sh 2>&1 | tee '$RUN_ROOT/retriever.log'"
fi

for attempt in $(seq 1 180); do
  if curl --fail --silent --show-error --max-time 5 \
    "http://127.0.0.1:$PORT/health" >"$RUN_ROOT/health.json"; then
    echo "[$(date)] local retriever is healthy"
    break
  fi
  if ! tmux has-session -t "$RETRIEVER_SESSION" 2>/dev/null; then
    echo "Local retriever exited during initialization" >&2
    exit 1
  fi
  if [[ "$attempt" == 180 ]]; then
    echo "Local retriever did not become ready within three hours" >&2
    exit 1
  fi
  echo "[$(date)] waiting for local retriever, attempt $attempt/180"
  sleep 60
done

PYTHON="$PROJECT_ROOT/experiments/luna_unified_critic_searchqa_local_retriever_20260828/benchmark_retriever.py"
ENV_PYTHON=/home/dataset-local/conda/envs/verl-agent-webshop/bin/python
{
  echo "=== sequential ==="
  "$ENV_PYTHON" "$PYTHON" --url "http://127.0.0.1:$PORT/retrieve" --requests 16 --concurrency 1
  echo "=== concurrency 8 ==="
  "$ENV_PYTHON" "$PYTHON" --url "http://127.0.0.1:$PORT/retrieve" --requests 64 --concurrency 8
  echo "=== concurrency 32 ==="
  "$ENV_PYTHON" "$PYTHON" --url "http://127.0.0.1:$PORT/retrieve" --requests 128 --concurrency 32
} 2>&1 | tee "$RUN_ROOT/benchmark.log"

echo "[$(date)] local retriever preparation and benchmark completed"
