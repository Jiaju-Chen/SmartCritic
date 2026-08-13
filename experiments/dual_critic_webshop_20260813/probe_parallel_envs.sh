#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF_WebShop}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
export JAVA_HOME=${JAVA_HOME:-$ENV_ROOT}
export JAVA_LD_LIBRARY_PATH=${JAVA_LD_LIBRARY_PATH:-$JAVA_HOME/lib/server}
set +u
source /opt/conda/bin/activate "$ENV_ROOT"
set -u

export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS:-"-Xms32m -Xmx512m -XX:ActiveProcessorCount=1 -XX:+UseSerialGC"}
export WEBSHOP_ENVS_PER_WORKER=${WEBSHOP_ENVS_PER_WORKER:-8}
export WEBSHOP_ENV_INIT_BATCH_SIZE=${WEBSHOP_ENV_INIT_BATCH_SIZE:-4}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_ROOT"
exec python experiments/dual_critic_webshop_20260813/probe_parallel_envs.py
