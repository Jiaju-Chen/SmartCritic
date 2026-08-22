#!/usr/bin/env bash
set -euo pipefail

SHARED_ROOT=${SHARED_ROOT:-/data2/group_何向南/chenjiaju/luna/shared}
ENV_ARCHIVE=${ENV_ARCHIVE:-$SHARED_ROOT/archives/gigpo-baselines-20260823.tar.zst}
LOCAL_ROOT=${SLURM_TMPDIR:-/tmp/$USER/smartcritic-$SLURM_JOB_ID}
ENV_ROOT=$LOCAL_ROOT/env
mkdir -p "$ENV_ROOT"

zstd -dc "$ENV_ARCHIVE" | tar -xf - -C "$ENV_ROOT"
export PATH="$ENV_ROOT/bin:$PATH"
export CONDA_PREFIX=$ENV_ROOT
export PYTHONNOUSERSITE=1
export PYTHONPATH=/data2/group_何向南/chenjiaju/luna/worktrees/alfworld-direct-20260823

python - <<'PY'
import os

import ray
import torch
import transformers
import vllm
import wandb

from verl.trainer.ppo.core_algos import compute_dual_critic_hybrid_gae

assert torch.cuda.is_available(), "CUDA is unavailable on the probe node"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print(
    "SAIDS probe passed",
    {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "ray": ray.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "wandb": wandb.__version__,
        "composition": compute_dual_critic_hybrid_gae.__name__,
        "pythonpath": os.environ["PYTHONPATH"],
    },
)
PY
