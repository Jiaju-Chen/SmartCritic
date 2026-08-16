#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2h_t128_v140_vb20_8gpu_seed0_full150_20260816}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
SESSION=${SESSION:-ppo_luna_unified_full150_20260816}
WANDB_RUN_ID=${WANDB_RUN_ID:-lunauni2150_0816}

mkdir -p "$RUN_DIR"/logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_UNIFIED && RUN_NAME='$RUN_NAME' RUN_DIR='$RUN_DIR' WANDB_RUN_ID='$WANDB_RUN_ID' TOTAL_EPOCHS='150' TEST_FREQ='5' SAVE_FREQ='5' bash experiments/luna_unified_critic_alfworld_20260816/tmux_entry.sh 2>&1 | tee -a '$RUN_DIR/logs/train.tmux.log'"

echo "session=$SESSION"
echo "run_name=$RUN_NAME"
echo "log=$RUN_DIR/logs/train.tmux.log"
echo "ckpt=/home/dataset-local/cjj/RL/checkpoints/luna_unified_alfworld/$RUN_NAME"
echo "wandb=https://wandb.ai/cjj01-ustc/verl_agent_alfworld_critic_ablation/runs/$WANDB_RUN_ID"
