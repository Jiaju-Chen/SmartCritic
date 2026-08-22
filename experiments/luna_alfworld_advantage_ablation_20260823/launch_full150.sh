#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic_ALFWORLD_DIRECT}
RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_directmix2h_alfworld_t128_v140_vb20_8gpu_seed0_20260823}
SESSION=${SESSION:-ppo_luna_alfworld_directmix_20260823}
WANDB_RUN_ID=${WANDB_RUN_ID:-lunadirectalf_0823}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

printf -v launch_command \
  'env PROJECT_ROOT=%q RUN_NAME=%q WANDB_RUN_ID=%q TOTAL_EPOCHS=150 TEST_FREQ=5 SAVE_FREQ=5 TRAIN_DATA_SIZE=128 VAL_DATA_SIZE=140 VAL_BATCH_SIZE=20 ADVANTAGE_COMPOSITION=direct bash experiments/luna_alfworld_advantage_ablation_20260823/tmux_entry.sh' \
  "$PROJECT_ROOT" "$RUN_NAME" "$WANDB_RUN_ID"

tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" "$launch_command"
echo "session=$SESSION"
echo "run_name=$RUN_NAME"
echo "log=/home/dataset-local/cjj/RL/runs/luna_alfworld_advantage_ablation/$RUN_NAME/logs/train.tmux.log"
echo "ckpt=/home/dataset-local/cjj/RL/checkpoints/luna_alfworld_advantage_ablation/$RUN_NAME"
echo "wandb=https://wandb.ai/cjj01-ustc/verl_agent_alfworld_critic_ablation/runs/$WANDB_RUN_ID"
