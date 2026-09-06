#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/vepfs-data/chenjiaju/projects/SmartCritic}
MANAGER=${MANAGER:-/root/vepfs-data/liuzijian/tools/gpu_job_manager.py}
STATE_DIR=${STATE_DIR:-/root/vepfs-data/liuzijian/tools/.gpu_job_manager}

python "$MANAGER" --state-dir "$STATE_DIR" submit \
  --name luna-alfworld-1p5b-residual-alpha2-20260906 \
  --num-gpus 8 \
  --gpus 0-7 \
  --cwd "$PROJECT_ROOT" \
  --env PROJECT_ROOT="$PROJECT_ROOT" \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -- \
  /usr/bin/bash "$PROJECT_ROOT/experiments/luna_alfworld_alpha2_20260906/run_sequence_lzj.sh"

