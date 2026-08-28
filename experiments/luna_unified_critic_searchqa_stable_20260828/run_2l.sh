#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2l_residual_searchr1_stable_8gpu_seed0_200it_20260828}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/$RUN_NAME}
LOCAL_PORT=${LOCAL_PORT:-18000}

mkdir -p "$LOG_DIR"
export LOG_DIR
export LOCAL_PORT
export TRAIN_PID="$$"
bash "$PROJECT_ROOT/experiments/luna_unified_critic_searchqa_stable_20260828/tunnel_supervisor.sh" &
TUNNEL_SUPERVISOR_PID=$!
cleanup() {
  kill "$TUNNEL_SUPERVISOR_PID" 2>/dev/null || true
  wait "$TUNNEL_SUPERVISOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

for attempt in $(seq 1 180); do
  if curl --fail --silent --show-error --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"query":"Who wrote Pride and Prejudice?","topk":3,"return_scores":false}' \
    "http://127.0.0.1:$LOCAL_PORT/retrieve" \
    >"$LOG_DIR/preflight.json" 2>"$LOG_DIR/preflight.err"; then
    echo "[$(date)] official retriever ready after attempt $attempt" | tee -a "$LOG_DIR/launcher.log"
    break
  fi
  echo "[$(date)] waiting for official retriever, attempt $attempt/180" | tee -a "$LOG_DIR/launcher.log"
  sleep 5
done

curl --fail --silent --show-error --max-time 60 \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who wrote Pride and Prejudice?","topk":3,"return_scores":false}' \
  "http://127.0.0.1:$LOCAL_PORT/retrieve" >/dev/null

cd "$PROJECT_ROOT"
export RUN_NAME
export RUN_ROOT
export WANDB_RUN_ID=${WANDB_RUN_ID:-lunasearchr1_stable2l_0828}
export SEARCH_URL="http://127.0.0.1:$LOCAL_PORT/retrieve"
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export TEST_FREQ=${TEST_FREQ:-50}
export SAVE_FREQ=${SAVE_FREQ:-50}

bash experiments/luna_unified_critic_searchqa_20260825/run_formal.sh \
  +env.search.fail_on_error=true
