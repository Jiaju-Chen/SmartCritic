#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=/root/vepfs-data/chenjiaju
SEQUENCE_RUN_ROOT=$BASE/runs/SmartCritic/luna_alfworld_alpha2_20260906
mkdir -p "$SEQUENCE_RUN_ROOT"

SMOKE_NAME=luna_alfworld_1p5b_residual_alpha2_smoke2_8gpu_20260906
RUN_NAME="$SMOKE_NAME" \
RUN_ROOT="$BASE/runs/SmartCritic/luna_alfworld_alpha2_20260906/smoke" \
CKPT_DIR="$BASE/checkpoints/luna_alfworld_alpha2_20260906/smoke/$SMOKE_NAME" \
WANDB_RUN_ID=lunaalfa2smk_0906 \
TRAIN_DATA_SIZE=16 \
VAL_DATA_SIZE=8 \
VAL_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=16 \
TOTAL_EPOCHS=2 \
TEST_FREQ=2 \
SAVE_FREQ=2 \
VAL_BEFORE_TRAIN=True \
RAY_NUM_CPUS=32 \
WANDB_MODE=offline \
bash "$SCRIPT_DIR/run_lzj.sh" \
  2>&1 | tee "$SEQUENCE_RUN_ROOT/smoke.log"

RUN_NAME=ppo_qwen25_15b_luna_unified2h_residual_alpha2_alfworld_t128_v140_vb20_8gpu_seed0_20260906 \
WANDB_RUN_ID=lunaalf15a2_0906 \
WANDB_MODE=offline \
bash "$SCRIPT_DIR/run_lzj.sh" \
  2>&1 | tee "$SEQUENCE_RUN_ROOT/train.log"
