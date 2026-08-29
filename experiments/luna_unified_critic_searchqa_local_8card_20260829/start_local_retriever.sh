#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa/local_retriever_8card_20260829}
SESSION=${SESSION:-searchqa_local_retriever_8card_20260829}
PORT=${PORT:-18002}
RETRIEVER_GPU=${RETRIEVER_GPU:-7}
mkdir -p "$RUN_ROOT"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[$(date)] retriever session already exists: $SESSION"
else
  tmux new-session -d -s "$SESSION" \
    "cd '$PROJECT_ROOT' && PORT='$PORT' RETRIEVER_GPU='$RETRIEVER_GPU' bash experiments/luna_unified_critic_searchqa_local_8card_20260829/run_local_gpu_retriever.sh 2>&1 | tee '$RUN_ROOT/retriever.log'"
fi

for attempt in $(seq 1 180); do
  if curl --fail --silent --show-error --max-time 5 \
    "http://127.0.0.1:$PORT/health" >"$RUN_ROOT/health.json"; then
    echo "[$(date)] local retriever healthy on GPU $RETRIEVER_GPU"
    cat "$RUN_ROOT/health.json"
    exit 0
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[$(date)] retriever exited during startup" >&2
    tail -n 80 "$RUN_ROOT/retriever.log" 2>/dev/null || true
    exit 1
  fi
  if (( attempt == 180 )); then
    echo "[$(date)] retriever did not become ready within 3 hours" >&2
    tail -n 80 "$RUN_ROOT/retriever.log" 2>/dev/null || true
    exit 1
  fi
  echo "[$(date)] waiting for local retriever ($attempt/180)"
  sleep 60
done
