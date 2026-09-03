#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SMOKE_DATA_ROOT=${SMOKE_DATA_ROOT:-/root/vepfs-data/chenjiaju/datasets/webshop-parquet/smoke8-val4/text}

EXPERIMENT_ID=adaptive_rms_rho05_smoke \
TOTAL_STEPS=2 \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=4 \
PPO_MINI_BATCH_SIZE=8 \
TEST_FREQ=1 \
SAVE_FREQ=1 \
TRAIN_DATA_ROOT="$SMOKE_DATA_ROOT" \
RUN_NAME=luna_webshop_adaptive_rms_rho05_beta09_smoke_8gpu_20260903 \
WANDB_RUN_ID=lzwsrms05smoke_0903 \
"$SCRIPT_DIR/run_train.sh"
