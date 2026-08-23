#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/group_何向南/chenjiaju/luna/worktrees/alfworld-direct-20260823}
SHARED_ROOT=${SHARED_ROOT:-/data2/group_何向南/chenjiaju/luna/shared}
LOG_ROOT=${LOG_ROOT:-$SHARED_ROOT/slurm_logs/luna_alfworld_advantage_ablation_20260823}
PARTITION=${PARTITION:-A100}
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

submit_ablation() {
  local composition=$1
  local job_name=$2
  local run_name=$3
  local wandb_id=$4

  sbatch --parsable \
    --job-name="$job_name" \
    --partition="$PARTITION" \
    --nodes=1 \
    --gres=gpu:8 \
    --cpus-per-task=96 \
    --mem=800G \
    --time=3-00:00:00 \
    --output="$LOG_ROOT/${composition}-%j.out" \
    --error="$LOG_ROOT/${composition}-%j.err" \
    --export="ALL,ADVANTAGE_COMPOSITION=$composition,RUN_NAME=$run_name,WANDB_RUN_ID=$wandb_id,WANDB_MODE=offline" \
    experiments/luna_alfworld_advantage_ablation_20260823/saids_train.sh
}

token_job_id=$(submit_ablation \
  token_only \
  luna-alf-token \
  ppo_qwen25_15b_luna_tokenonly2h_alfworld_t128_v140_vb20_8gpu_seed0_saids_20260823 \
  lunatokenalf_s_0823)

turn_job_id=$(submit_ablation \
  turn_only \
  luna-alf-turn \
  ppo_qwen25_15b_luna_turnonly2h_alfworld_t128_v140_vb20_8gpu_seed0_saids_20260823 \
  lunaturnalf_s_0823)

echo "token_job_id=$token_job_id"
echo "turn_job_id=$turn_job_id"
echo "partition=$PARTITION"
echo "logs=$LOG_ROOT"
