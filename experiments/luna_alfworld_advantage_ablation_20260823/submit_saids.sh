#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/group_何向南/chenjiaju/luna/worktrees/alfworld-direct-20260823}
SHARED_ROOT=${SHARED_ROOT:-/data2/group_何向南/chenjiaju/luna/shared}
LOG_ROOT=${LOG_ROOT:-$SHARED_ROOT/slurm_logs/luna_alfworld_advantage_ablation_20260823}
PARTITION=${PARTITION:-A100}
SUBMIT_HOLD=${SUBMIT_HOLD:-0}
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

hold_args=()
if [[ "$SUBMIT_HOLD" == 1 ]]; then
  hold_args+=(--hold)
fi

probe_job_id=$(sbatch --parsable "${hold_args[@]}" \
  --job-name=luna-alf-probe \
  --partition="$PARTITION" \
  --nodes=1 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=00:30:00 \
  --output="$LOG_ROOT/probe-%j.out" \
  --error="$LOG_ROOT/probe-%j.err" \
  experiments/luna_alfworld_advantage_ablation_20260823/saids_probe.sh)

train_job_id=$(sbatch --parsable \
  --job-name=luna-alf-direct \
  --partition="$PARTITION" \
  --dependency="afterok:$probe_job_id" \
  --nodes=1 \
  --gres=gpu:8 \
  --cpus-per-task=96 \
  --mem=800G \
  --time=3-00:00:00 \
  --output="$LOG_ROOT/train-%j.out" \
  --error="$LOG_ROOT/train-%j.err" \
  experiments/luna_alfworld_advantage_ablation_20260823/saids_train.sh)

echo "probe_job_id=$probe_job_id"
echo "train_job_id=$train_job_id"
echo "dependency=afterok:$probe_job_id"
echo "partition=$PARTITION"
echo "probe_held=$SUBMIT_HOLD"
echo "logs=$LOG_ROOT"
