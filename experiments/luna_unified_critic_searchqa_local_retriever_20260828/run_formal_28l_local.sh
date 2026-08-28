#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
PORT=${PORT:-18002}
RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified28l_residual_searchr1_localretriever_8gpu_seed0_200it_tokengamma1_20260828}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
ROLLOUT_N=${ROLLOUT_N:-5}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

if [[ ${CONFIG_ONLY:-0} != 1 ]]; then
  curl --fail --silent --show-error --max-time 10 \
    "http://127.0.0.1:$PORT/health" >/dev/null
  curl --fail --silent --show-error --max-time 60 \
    -H 'Content-Type: application/json' \
    -d '{"query":"Who wrote Pride and Prejudice?","topk":3,"return_scores":false}' \
    "http://127.0.0.1:$PORT/retrieve" >/dev/null
fi

export PROJECT_ROOT
export RUN_NAME
export TRAIN_BATCH_SIZE VAL_BATCH_SIZE ROLLOUT_N
export WANDB_RUN_ID=${WANDB_RUN_ID:-lunasearchr1_28l_local_0828}
export SEARCH_URL="http://127.0.0.1:$PORT/retrieve"

exec bash \
  "$PROJECT_ROOT/experiments/luna_unified_critic_searchqa_28l_20260827/run_formal_28l.sh" \
  +env.search.fail_on_error=true \
  "$@"
