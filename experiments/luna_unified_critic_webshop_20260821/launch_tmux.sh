#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_WEBSHOP}
RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2h_webshop_t128_v128_8gpu_seed0_20260821}
SESSION=${SESSION:-ppo_luna_unified_webshop_20260821}
WANDB_RUN_ID=${WANDB_RUN_ID:-lunauniws_0821}
WANDB_MODE=${WANDB_MODE:-online}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-128}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

printf -v launch_command \
  'env RUN_NAME=%q WANDB_RUN_ID=%q WANDB_MODE=%q TRAIN_DATA_SIZE=%q VAL_DATA_SIZE=%q VAL_BATCH_SIZE=%q PPO_MINI_BATCH_SIZE=%q TOTAL_EPOCHS=%q TEST_FREQ=%q SAVE_FREQ=%q VAL_BEFORE_TRAIN=%q bash experiments/luna_unified_critic_webshop_20260821/tmux_entry.sh' \
  "$RUN_NAME" "$WANDB_RUN_ID" "$WANDB_MODE" "$TRAIN_DATA_SIZE" "$VAL_DATA_SIZE" \
  "$VAL_BATCH_SIZE" "$PPO_MINI_BATCH_SIZE" "$TOTAL_EPOCHS" "$TEST_FREQ" \
  "$SAVE_FREQ" "$VAL_BEFORE_TRAIN"

tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" "$launch_command"
echo "started tmux session $SESSION for $RUN_NAME"
