#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/vepfs-data/chenjiaju/projects/SmartCritic-lzj-webshop-gamma-alpha}
MANAGER=${MANAGER:-/root/vepfs-data/liuzijian/tools/gpu_job_manager.py}
STATE_DIR=${STATE_DIR:-/root/vepfs-data/liuzijian/tools/.gpu_job_manager}
ENTRY=$PROJECT_ROOT/experiments/luna_webshop_gamma_alpha_20260902/run_sequence.sh

for path in "$MANAGER" "$ENTRY"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing submission path: $path" >&2
    exit 2
  fi
done

python "$MANAGER" --state-dir "$STATE_DIR" submit \
  --name luna-webshop-g0-g1-eval500-20260902 \
  --num-gpus 8 \
  --gpus 0-7 \
  --cwd "$PROJECT_ROOT" \
  --env PROJECT_ROOT="$PROJECT_ROOT" \
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -- \
  /usr/bin/bash "$ENTRY"

