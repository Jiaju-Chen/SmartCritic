#!/usr/bin/env bash
set -euo pipefail

# Formal SearchQA run: GPUs 0-6 train Luna Unified; GPU 7 serves retrieval.
PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
PORT=${PORT:-18002}
RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2l_residual_searchr1_local_t252_g5_7gpu_seed0_200it_20260829}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}

export RUN_NAME RUN_ROOT RUN_DIR
export CKPT_DIR=${CKPT_DIR:-/home/dataset-local/cjj/RL/checkpoints/luna_unified_searchqa/$RUN_NAME}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}
export SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:$PORT/retrieve}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export TEST_FREQ=${TEST_FREQ:-50}
export SAVE_FREQ=${SAVE_FREQ:-50}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_ID=${WANDB_RUN_ID:-lunauni7sq_0829}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export RAY_TMP_ROOT=${RAY_TMP_ROOT:-/dev/shm/scq-luna7-ray}
export FAST_TMP_ROOT=${FAST_TMP_ROOT:-/dev/shm/scq-luna7-tmp}

if [[ ${CONFIG_ONLY:-0} != 1 ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export NO_PROXY=127.0.0.1,localhost
  export no_proxy="$NO_PROXY"
  curl --fail --silent --show-error --max-time 10 \
    "http://127.0.0.1:$PORT/health" >/dev/null
fi

cd "$PROJECT_ROOT"
exec bash experiments/luna_unified_critic_searchqa_20260825/run_formal.sh \
  data.train_batch_size=252 \
  actor_rollout_ref.actor.ppo_mini_batch_size=504 \
  critic.ppo_mini_batch_size=504 \
  trainer.n_gpus_per_node=7 \
  +env.search.fail_on_error=true \
  "$@"
