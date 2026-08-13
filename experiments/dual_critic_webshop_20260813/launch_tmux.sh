#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF_WebShop}
RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_dualcritic2l_webshop_t128_v128_8gpu_seed0_20260813}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/dual_critic_webshop}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
SESSION=${SESSION:-ppo_dualcritic2l_webshop_20260813}

mkdir -p "$RUN_DIR/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$PROJECT_ROOT' && PROJECT_ROOT='$PROJECT_ROOT' RUN_NAME='$RUN_NAME' RUN_ROOT='$RUN_ROOT' RUN_DIR='$RUN_DIR' CKPT_DIR='${CKPT_DIR:-/home/dataset-local/cjj/RL/checkpoints/dual_critic_webshop/$RUN_NAME}' WANDB_MODE='${WANDB_MODE:-online}' WANDB_RUN_ID='${WANDB_RUN_ID:-dcw2l_0813}' CRITIC_NUM_LAYERS='${CRITIC_NUM_LAYERS:-2}' TURN_CRITIC_NUM_LAYERS='${TURN_CRITIC_NUM_LAYERS:-2}' TRAIN_DATA_SIZE='${TRAIN_DATA_SIZE:-128}' VAL_DATA_SIZE='${VAL_DATA_SIZE:-128}' VAL_BATCH_SIZE='${VAL_BATCH_SIZE:-16}' PPO_MINI_BATCH_SIZE='${PPO_MINI_BATCH_SIZE:-64}' TOTAL_EPOCHS='${TOTAL_EPOCHS:-150}' TEST_FREQ='${TEST_FREQ:-5}' SAVE_FREQ='${SAVE_FREQ:-5}' VAL_BEFORE_TRAIN='${VAL_BEFORE_TRAIN:-True}' NUM_CPUS_PER_ENV_WORKER='${NUM_CPUS_PER_ENV_WORKER:-0.25}' RAY_NUM_CPUS='${RAY_NUM_CPUS:-96}' bash experiments/dual_critic_webshop_20260813/tmux_entry.sh 2>&1 | tee -a '$RUN_DIR/logs/train.tmux.log'"

echo "session=$SESSION"
echo "run_name=$RUN_NAME"
echo "log=$RUN_DIR/logs/train.tmux.log"
echo "ckpt=${CKPT_DIR:-/home/dataset-local/cjj/RL/checkpoints/dual_critic_webshop/$RUN_NAME}"
echo "wandb_project=verl_agent_webshop_critic_ablation"
echo "wandb_run_id=${WANDB_RUN_ID:-dcw2l_0813}"
