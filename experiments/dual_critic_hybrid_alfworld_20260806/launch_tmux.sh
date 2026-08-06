#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_dualcritic_hybrid_t128_v140_8gpu_seed0_20260806}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/dual_critic_hybrid_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
SESSION=${SESSION:-ppo_dualcritic_hybrid_20260806}

mkdir -p "$RUN_DIR"/logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /home/dataset-local/cjj/RL/GiGPO_PVF && RUN_NAME='$RUN_NAME' RUN_DIR='$RUN_DIR' WANDB_RUN_ID='${WANDB_RUN_ID:-dchppo0806}' TRAIN_DATA_SIZE='${TRAIN_DATA_SIZE:-128}' VAL_DATA_SIZE='${VAL_DATA_SIZE:-140}' PPO_MINI_BATCH_SIZE='${PPO_MINI_BATCH_SIZE:-256}' TOTAL_EPOCHS='${TOTAL_EPOCHS:-150}' TEST_FREQ='${TEST_FREQ:-5}' SAVE_FREQ='${SAVE_FREQ:-5}' bash experiments/dual_critic_hybrid_alfworld_20260806/tmux_entry.sh 2>&1 | tee -a '$RUN_DIR/logs/train.tmux.log'"

echo "session=$SESSION"
echo "run_name=$RUN_NAME"
echo "log=$RUN_DIR/logs/train.tmux.log"
echo "ckpt=/home/dataset-local/cjj/RL/checkpoints/dual_critic_hybrid_alfworld/$RUN_NAME"
echo "wandb_project=verl_agent_alfworld_critic_ablation"
echo "wandb_run_id=${WANDB_RUN_ID:-dchppo0806}"
