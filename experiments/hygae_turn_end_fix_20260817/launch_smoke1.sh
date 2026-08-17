#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_hygae_turnend_critic2l_smoke1_20260817}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/hygae_turn_end_fix_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
SESSION=${SESSION:-ppo_hygae_turnend_smoke1_20260817}

mkdir -p "$RUN_DIR"/logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /home/dataset-local/cjj/RL/GiGPO_PVF_HYGAE_TURNEND && RUN_NAME='$RUN_NAME' RUN_DIR='$RUN_DIR' WANDB_RUN_ID='${WANDB_RUN_ID:-hygaete1smk0817}' TRAIN_DATA_SIZE='8' VAL_DATA_SIZE='4' VAL_BATCH_SIZE='4' PPO_MINI_BATCH_SIZE='8' TOTAL_EPOCHS='1' TEST_FREQ='1' SAVE_FREQ='1' bash experiments/hygae_turn_end_fix_20260817/tmux_entry.sh 2>&1 | tee -a '$RUN_DIR/logs/train.tmux.log'"

echo "session=$SESSION"
echo "run_name=$RUN_NAME"
echo "log=$RUN_DIR/logs/train.tmux.log"
echo "ckpt=/home/dataset-local/cjj/RL/checkpoints/hygae_turn_end_fix_alfworld/$RUN_NAME"
echo "wandb=https://wandb.ai/cjj01-ustc/verl_agent_alfworld_critic_ablation/runs/${WANDB_RUN_ID:-hygaete1smk0817}"
