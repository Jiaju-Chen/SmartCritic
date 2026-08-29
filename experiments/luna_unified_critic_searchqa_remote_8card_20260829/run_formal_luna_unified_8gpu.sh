#!/usr/bin/env bash
set -euo pipefail

# Standard 8-GPU SearchQA configuration using the independently hosted retriever.
PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
PORT=${PORT:-18003}
RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2l_residual_searchr1_remote_t256_g5_8gpu_seed0_200it_20260829}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}

export RUN_NAME RUN_ROOT RUN_DIR
export CKPT_DIR=${CKPT_DIR:-/home/dataset-local/cjj/RL/checkpoints/luna_unified_searchqa/$RUN_NAME}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:$PORT/retrieve}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export TEST_FREQ=${TEST_FREQ:-50}
export SAVE_FREQ=${SAVE_FREQ:-50}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_ID=${WANDB_RUN_ID:-lunauni8remote_0829}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export RAY_TMP_ROOT=${RAY_TMP_ROOT:-/dev/shm/scq-luna8-remote-ray}
export FAST_TMP_ROOT=${FAST_TMP_ROOT:-/dev/shm/scq-luna8-remote-tmp}

if [[ ${CONFIG_ONLY:-0} != 1 ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy="$NO_PROXY"
  curl --fail --silent --show-error \
    --connect-timeout 5 --max-time 30 \
    -H 'Content-Type: application/json' \
    -d '{"query":"Who wrote Pride and Prejudice?","topk":3,"return_scores":false}' \
    "$SEARCH_URL" >/dev/null
fi

cd "$PROJECT_ROOT"
exec bash experiments/luna_unified_critic_searchqa_20260825/run_formal.sh \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=512 \
  critic.ppo_mini_batch_size=512 \
  trainer.n_gpus_per_node=8 \
  +env.search.fail_on_error=true \
  "$@"
