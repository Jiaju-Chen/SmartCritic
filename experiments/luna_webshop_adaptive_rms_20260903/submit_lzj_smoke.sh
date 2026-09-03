#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/vepfs-data/chenjiaju/projects/SmartCritic-lzj-webshop-gamma-alpha}
MANAGER=${MANAGER:-/root/vepfs-data/liuzijian/tools/gpu_job_manager.py}
STATE_DIR=${STATE_DIR:-/root/vepfs-data/liuzijian/tools/.gpu_job_manager}

python "$MANAGER" --state-dir "$STATE_DIR" submit \
  --name luna-webshop-adaptive-rms-rho05-smoke-20260903 \
  --num-gpus 8 \
  --gpus 0-7 \
  --cwd "$PROJECT_ROOT" \
  --env PROJECT_ROOT="$PROJECT_ROOT" \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -- \
  /usr/bin/bash "$PROJECT_ROOT/experiments/luna_webshop_adaptive_rms_20260903/run_smoke.sh"
