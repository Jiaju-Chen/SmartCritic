#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
WEB_SHOP_PATTERN=${WEB_SHOP_PATTERN:-ppo_qwen25_15b_luna_gradient_alpha_nowhiten_webshop_t128_v128_8gpu_seed0_v2_20260825}
WAIT_SECONDS=${WAIT_SECONDS:-60}

echo "Waiting for the active WebShop run to finish: $WEB_SHOP_PATTERN"
while pgrep -f "[m]ain_ppo.*$WEB_SHOP_PATTERN" >/dev/null; do
  date '+%Y-%m-%d %H:%M:%S WebShop is still running'
  sleep "$WAIT_SECONDS"
done

# Let Ray and vLLM release CUDA contexts before the next run initializes.
sleep 120
echo "WebShop process exited; starting Unified Luna SearchQA smoke"
exec bash "$PROJECT_ROOT/experiments/luna_unified_critic_searchqa_20260825/run_smoke.sh"
