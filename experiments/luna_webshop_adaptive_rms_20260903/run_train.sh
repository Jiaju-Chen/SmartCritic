#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
BASE_TRAIN=$PROJECT_ROOT/experiments/luna_webshop_gamma_alpha_20260902/run_train.sh

RUN_NAME=${RUN_NAME:-luna_webshop_adaptive_rms_rho05_beta09_t128_v128_8gpu_seed0_20260903}
RUN_BASE=${RUN_BASE:-/root/vepfs-data/chenjiaju/runs/SmartCritic/luna_webshop_adaptive_rms_20260903}
CKPT_BASE=${CKPT_BASE:-/root/vepfs-data/chenjiaju/checkpoints/luna_webshop_adaptive_rms_20260903}

env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  EXPERIMENT_ID="${EXPERIMENT_ID:-adaptive_rms_rho05}" \
  ALPHA="${ALPHA:-1.0}" \
  ADAPTIVE_RESIDUAL_SCALE=True \
  ADAPTIVE_TARGET_RATIO="${ADAPTIVE_TARGET_RATIO:-0.5}" \
  ADAPTIVE_EMA_BETA="${ADAPTIVE_EMA_BETA:-0.9}" \
  ADAPTIVE_MIN_SCALE="${ADAPTIVE_MIN_SCALE:-0.0}" \
  ADAPTIVE_MAX_SCALE="${ADAPTIVE_MAX_SCALE:-10.0}" \
  RUN_NAME="$RUN_NAME" \
  WANDB_RUN_ID="${WANDB_RUN_ID:-lzwsrms05_0903}" \
  RUN_BASE="$RUN_BASE" \
  CKPT_BASE="$CKPT_BASE" \
  "$BASE_TRAIN"
