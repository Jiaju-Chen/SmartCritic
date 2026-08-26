#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
DATA_ROOT=${DATA_ROOT:-/home/dataset-local/cjj/RL/data/searchR1_official}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
REPO_URL="$HF_ENDPOINT/datasets/PeterJinGo/nq_hotpotqa_train/resolve/main"

mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/processed"
for split in train test; do
  curl --fail --location --retry 20 --retry-delay 5 --continue-at - \
    --output "$DATA_ROOT/raw/$split.parquet" \
    "$REPO_URL/$split.parquet?download=true"
done

"$ENV_ROOT/bin/python" \
  "$PROJECT_ROOT/experiments/luna_unified_critic_searchqa_20260825/process_official_dataset.py" \
  --raw-dir "$DATA_ROOT/raw" \
  --output-dir "$DATA_ROOT/processed"

