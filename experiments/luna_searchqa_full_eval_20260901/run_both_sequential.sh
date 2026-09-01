#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
EVAL_ROOT=${EVAL_ROOT:-/home/dataset-local/cjj/RL/runs/luna_searchqa_full_eval_20260901}
SCRIPT_DIR=$PROJECT_ROOT/experiments/luna_searchqa_full_eval_20260901

mkdir -p "$EVAL_ROOT/logs"

echo "[$(date --iso-8601=seconds)] starting no-whitening full evaluation"
bash "$SCRIPT_DIR/run_one.sh" no_whiten \
  2>&1 | tee "$EVAL_ROOT/logs/no_whiten_full51713.log"

echo "[$(date --iso-8601=seconds)] no-whitening evaluation completed"
echo "[$(date --iso-8601=seconds)] starting whitening full evaluation"
bash "$SCRIPT_DIR/run_one.sh" whiten \
  2>&1 | tee "$EVAL_ROOT/logs/whiten_full51713.log"

echo "[$(date --iso-8601=seconds)] both full evaluations completed"
