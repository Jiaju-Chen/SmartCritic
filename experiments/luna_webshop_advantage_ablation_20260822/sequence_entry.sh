#!/usr/bin/env bash
set -euo pipefail

run_ablation() {
  local composition=$1
  local run_name=$2
  local wandb_id=$3

  echo "[$(date)] sequence starting composition=$composition run=$run_name"
  env \
    RUN_NAME="$run_name" \
    WANDB_RUN_ID="$wandb_id" \
    WANDB_MODE=online \
    ADVANTAGE_COMPOSITION="$composition" \
    TRAIN_DATA_SIZE=128 \
    VAL_DATA_SIZE=128 \
    VAL_BATCH_SIZE=16 \
    PPO_MINI_BATCH_SIZE=64 \
    TOTAL_EPOCHS=150 \
    TEST_FREQ=5 \
    SAVE_FREQ=5 \
    VAL_BEFORE_TRAIN=True \
    bash experiments/luna_webshop_advantage_ablation_20260822/tmux_entry.sh
}

run_ablation \
  direct \
  ppo_qwen25_15b_luna_directmix2h_webshop_t128_v128_8gpu_seed0_20260822 \
  lunawsdirect_0822

run_ablation \
  token_only \
  ppo_qwen25_15b_luna_tokenonly2h_webshop_t128_v128_8gpu_seed0_20260822 \
  lunawstoken_0822

run_ablation \
  turn_only \
  ppo_qwen25_15b_luna_turnonly2h_webshop_t128_v128_8gpu_seed0_20260822 \
  lunawsturn_0822

echo "[$(date)] all WebShop advantage ablations completed"
