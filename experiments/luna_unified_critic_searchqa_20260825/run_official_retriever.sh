#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
ASSET_ROOT=${ASSET_ROOT:-/home/dataset-local/cjj/RL/data/searchR1_official_retriever}
PORT=${PORT:-8000}
LOG_ROOT=${LOG_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa/retriever}
mkdir -p "$LOG_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-$ASSET_ROOT/hf-cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ENV_ROOT/bin/python" "$PROJECT_ROOT/examples/search/retriever/retrieval_server.py" \
  --index_path "$ASSET_ROOT/index/e5_Flat.index" \
  --corpus_path "$ASSET_ROOT/corpus/wiki-18.jsonl" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model "$ASSET_ROOT/models/e5-base-v2" \
  --faiss_gpu \
  --faiss_gpu_temp_memory_mb 512 \
  --faiss_gpu_add_batch_size 100000 \
  --port "$PORT"
