#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2h_webshop_t128_v128_8gpu_seed0_20260821}
export SESSION=${SESSION:-ppo_luna_unified_webshop_full150_20260821}
export WANDB_RUN_ID=${WANDB_RUN_ID:-lunauniws_0821}
export WANDB_MODE=${WANDB_MODE:-online}

exec bash "$(dirname "$0")/launch_tmux.sh"
