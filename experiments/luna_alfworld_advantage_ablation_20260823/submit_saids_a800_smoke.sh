#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/group_何向南/chenjiaju/luna/worktrees/alfworld-direct-20260823}
SHARED_ROOT=${SHARED_ROOT:-/data2/group_何向南/chenjiaju/luna/shared}
LOG_ROOT=${LOG_ROOT:-$SHARED_ROOT/slurm_logs/luna_alfworld_advantage_ablation_20260823}
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

job_id=$(sbatch --parsable \
  --job-name=luna-alf-a8004-smoke \
  --partition=A800 \
  --nodes=1 \
  --gres=gpu:4 \
  --cpus-per-task=64 \
  --mem=480G \
  --time=06:00:00 \
  --output="$LOG_ROOT/a8004-smoke-%j.out" \
  --error="$LOG_ROOT/a8004-smoke-%j.err" \
  --export="ALL,ADVANTAGE_COMPOSITION=residual,RUN_NAME=ppo_qwen25_15b_luna_residual2h_alfworld_t128_v140_vb20_a800x4_smoke1_20260823,WANDB_RUN_ID=lunaa8004smoke_0823,WANDB_MODE=offline,N_GPUS_PER_NODE=4,ROLLOUT_TP_SIZE=2,RAY_NUM_CPUS=64,NUM_CPUS_PER_ENV_WORKER=0.25,TOTAL_EPOCHS=1,TRAIN_DATA_SIZE=128,VAL_DATA_SIZE=140,VAL_BATCH_SIZE=20,PPO_MINI_BATCH_SIZE=256,TEST_FREQ=1,SAVE_FREQ=1,VAL_BEFORE_TRAIN=False" \
  experiments/luna_alfworld_advantage_ablation_20260823/saids_train.sh)

echo "a800_smoke_job_id=$job_id"
echo "partition=A800"
echo "gpus=4"
echo "global_train_batch=128"
echo "global_ppo_mini_batch=256"
echo "logs=$LOG_ROOT"
