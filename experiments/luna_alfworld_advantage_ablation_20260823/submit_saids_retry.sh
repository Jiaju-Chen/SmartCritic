#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data2/group_何向南/chenjiaju/luna/worktrees/alfworld-direct-20260823}
SHARED_ROOT=${SHARED_ROOT:-/data2/group_何向南/chenjiaju/luna/shared}
LOG_ROOT=${LOG_ROOT:-$SHARED_ROOT/slurm_logs/luna_alfworld_advantage_ablation_20260823}
PARTITION=${PARTITION:-A100}
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

submit_job() {
  local dependency=$1
  local job_name=$2
  local composition=$3
  local run_name=$4
  local wandb_id=$5
  shift 5
  local export_list="ALL,ADVANTAGE_COMPOSITION=$composition,RUN_NAME=$run_name,WANDB_RUN_ID=$wandb_id,WANDB_MODE=offline"
  local export_item
  for export_item in "$@"; do
    export_list+=",$export_item"
  done

  sbatch --parsable \
    --job-name="$job_name" \
    --partition="$PARTITION" \
    --dependency="$dependency" \
    --nodes=1 \
    --gres=gpu:8 \
    --cpus-per-task=96 \
    --mem=800G \
    --time=3-00:00:00 \
    --output="$LOG_ROOT/${job_name}-%j.out" \
    --error="$LOG_ROOT/${job_name}-%j.err" \
    --export="$export_list" \
    experiments/luna_alfworld_advantage_ablation_20260823/saids_train.sh
}

smoke_job_id=$(submit_job \
  singleton \
  luna-alf-smoke-r1 \
  residual \
  ppo_qwen25_15b_luna_residual2h_alfworld_saids_smoke1_r1_20260823 \
  lunasmokealf_s_r1_0823 \
  TOTAL_EPOCHS=1 TRAIN_DATA_SIZE=128 VAL_DATA_SIZE=140 VAL_BATCH_SIZE=20 PPO_MINI_BATCH_SIZE=256 TEST_FREQ=1 SAVE_FREQ=1 VAL_BEFORE_TRAIN=False)

dependency="afterok:$smoke_job_id"

direct_job_id=$(submit_job \
  "$dependency" \
  luna-alf-direct-r1 \
  direct \
  ppo_qwen25_15b_luna_directmix2h_alfworld_t128_v140_vb20_8gpu_seed0_saids_r1_20260823 \
  lunadirectalf_s_r1_0823)

token_job_id=$(submit_job \
  "$dependency" \
  luna-alf-token-r1 \
  token_only \
  ppo_qwen25_15b_luna_tokenonly2h_alfworld_t128_v140_vb20_8gpu_seed0_saids_r1_20260823 \
  lunatokenalf_s_r1_0823)

turn_job_id=$(submit_job \
  "$dependency" \
  luna-alf-turn-r1 \
  turn_only \
  ppo_qwen25_15b_luna_turnonly2h_alfworld_t128_v140_vb20_8gpu_seed0_saids_r1_20260823 \
  lunaturnalf_s_r1_0823)

echo "smoke_job_id=$smoke_job_id"
echo "direct_job_id=$direct_job_id"
echo "token_job_id=$token_job_id"
echo "turn_job_id=$turn_job_id"
echo "formal_dependency=$dependency"
echo "logs=$LOG_ROOT"
