#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_hygae_unified_critic2l_t128_v140_vb20_8gpu_seed0_pilot5_20260815}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/hygae_unified_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
SESSION=${SESSION:-ppo_hygae_unified_pilot5_20260815}

mkdir -p "$RUN_DIR"/logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /home/dataset-local/cjj/RL/GiGPO_PVF_HYGAE && RUN_NAME='$RUN_NAME' RUN_DIR='$RUN_DIR' WANDB_RUN_ID='${WANDB_RUN_ID:-hygae2lp5_0815}' TOTAL_EPOCHS='5' TEST_FREQ='5' SAVE_FREQ='5' bash experiments/hygae_unified_alfworld_20260815/tmux_entry.sh 2>&1 | tee -a '$RUN_DIR/logs/train.tmux.log'"

echo "session=$SESSION"
echo "run_name=$RUN_NAME"
echo "log=$RUN_DIR/logs/train.tmux.log"
echo "ckpt=/home/dataset-local/cjj/RL/checkpoints/hygae_unified_alfworld/$RUN_NAME"
echo "wandb=https://wandb.ai/cjj01-ustc/verl_agent_alfworld_critic_ablation/runs/${WANDB_RUN_ID:-hygae2lp5_0815}"
