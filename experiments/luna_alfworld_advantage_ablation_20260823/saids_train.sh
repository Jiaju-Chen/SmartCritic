#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/group_何向南/chenjiaju/luna/worktrees/alfworld-direct-20260823}
SHARED_ROOT=${SHARED_ROOT:-/data2/group_何向南/chenjiaju/luna/shared}
ENV_ARCHIVE=${ENV_ARCHIVE:-$SHARED_ROOT/archives/gigpo-baselines-20260823-v2.tar.zst}
LOCAL_ROOT=${SLURM_TMPDIR:-/tmp/$USER/smartcritic-$SLURM_JOB_ID}
ENV_ROOT=$LOCAL_ROOT/env
RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_directmix2h_alfworld_t128_v140_vb20_8gpu_seed0_saids_20260823}

mkdir -p "$ENV_ROOT" "$LOCAL_ROOT/ray" "$LOCAL_ROOT/tmp"
zstd -dc "$ENV_ARCHIVE" | tar -xf - -C "$ENV_ROOT"

export PROJECT_ROOT
export ENV_ROOT
export SKIP_CONDA_ACTIVATE=1
export RUN_NAME
export RUN_ROOT=${RUN_ROOT:-$SHARED_ROOT/runs/luna_alfworld_advantage_ablation}
export CKPT_ROOT=${CKPT_ROOT:-$SHARED_ROOT/checkpoints/luna_alfworld_advantage_ablation}
export SNAP=${SNAP:-$SHARED_ROOT/models/Qwen2.5-1.5B-Instruct}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$SHARED_ROOT/datasets/alfworld_data}
export ORIGINAL_HOME=${ORIGINAL_HOME:-/home/chenjiaju}
export RAY_TMP_ROOT=$LOCAL_ROOT/ray
export FAST_TMP_ROOT=$LOCAL_ROOT/tmp
export WANDB_RUN_ID=${WANDB_RUN_ID:-lunadirectalf_s_0823}
export WANDB_MODE=${WANDB_MODE:-online}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

cd "$PROJECT_ROOT"
bash experiments/luna_alfworld_advantage_ablation_20260823/run_train.sh
