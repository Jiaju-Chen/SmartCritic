#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
ASSET_ROOT=${ASSET_ROOT:-/home/dataset-local/cjj/RL/data/searchR1_official_retriever}
PORT=${PORT:-18002}
RETRIEVER_GPU=${RETRIEVER_GPU:-7}
FAISS_GPU=${FAISS_GPU:-0}
TORCH_GPU_FLAT=${TORCH_GPU_FLAT:-1}
TORCH_GPU_CHUNK_SIZE=${TORCH_GPU_CHUNK_SIZE:-1000000}
FAISS_OMP_THREADS=${FAISS_OMP_THREADS:-16}

test -s "$ASSET_ROOT/index/e5_Flat.index"
test -s "$ASSET_ROOT/corpus/wiki-18.jsonl"
test -s "$ASSET_ROOT/models/e5-base-v2/config.json"

export CUDA_VISIBLE_DEVICES="$RETRIEVER_GPU"
export HF_HOME=${HF_HOME:-$ASSET_ROOT/hf-cache}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$FAISS_OMP_THREADS"
export MKL_NUM_THREADS="$FAISS_OMP_THREADS"
export OPENBLAS_NUM_THREADS="$FAISS_OMP_THREADS"
export NUMEXPR_NUM_THREADS="$FAISS_OMP_THREADS"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

ARGS=(
  --index_path "$ASSET_ROOT/index/e5_Flat.index"
  --corpus_path "$ASSET_ROOT/corpus/wiki-18.jsonl"
  --topk 3
  --retriever_name e5
  --retriever_model "$ASSET_ROOT/models/e5-base-v2"
  --faiss_omp_threads "$FAISS_OMP_THREADS"
  --host 127.0.0.1
  --torch_gpu_chunk_size "$TORCH_GPU_CHUNK_SIZE"
  --port "$PORT"
)
if [[ "$FAISS_GPU" == 1 ]]; then
  ARGS+=(--faiss_gpu --faiss_gpu_temp_memory_mb 512 --faiss_gpu_add_batch_size 100000)
fi
if [[ "$TORCH_GPU_FLAT" == 1 ]]; then
  ARGS+=(--torch_gpu_flat)
fi

exec "$ENV_ROOT/bin/python" \
  "$PROJECT_ROOT/examples/search/retriever/retrieval_server.py" \
  "${ARGS[@]}"
