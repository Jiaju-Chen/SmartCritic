#!/usr/bin/env bash
set -euo pipefail

# Submit one clean A100 run and two four-A800 runs with identical PPO semantics.
PROJECT_ROOT=${PROJECT_ROOT:-/data2/group_何向南/chenjiaju/luna/worktrees/alfworld-direct-20260823}
SHARED_ROOT=${SHARED_ROOT:-/data2/group_何向南/chenjiaju/luna/shared}
LOG_ROOT=${LOG_ROOT:-$SHARED_ROOT/slurm_logs/luna_alfworld_advantage_ablation_20260823}
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

submit_ablation() {
  local composition=$1
  local job_name=$2
  local partition=$3
  local gpu_count=$4
  local cpu_count=$5
  local memory=$6
  local run_name=$7
  local wandb_id=$8
  local ray_cpus=$9

  sbatch --parsable \
    --job-name="$job_name" \
    --partition="$partition" \
    --nodes=1 \
    --gres="gpu:$gpu_count" \
    --cpus-per-task="$cpu_count" \
    --mem="$memory" \
    --time=3-00:00:00 \
    --output="$LOG_ROOT/${composition}-%j.out" \
    --error="$LOG_ROOT/${composition}-%j.err" \
    --export="ALL,ADVANTAGE_COMPOSITION=$composition,RUN_NAME=$run_name,WANDB_RUN_ID=$wandb_id,WANDB_MODE=offline,N_GPUS_PER_NODE=$gpu_count,RAY_NUM_CPUS=$ray_cpus" \
    experiments/luna_alfworld_advantage_ablation_20260823/saids_train.sh
}

# Keep the main direct-mix run on eight A100s.
direct_job_id=$(submit_ablation \
  direct \
  luna-alf-direct-a100 \
  A100 \
  8 \
  96 \
  800G \
  ppo_qwen25_15b_luna_directmix2h_alfworld_t128_v140_vb20_a100x8_seed0_20260823 \
  lunadirecta100_0823 \
  96)

# The two ablations use four A800s while keeping the global batch and PPO
# minibatch unchanged; only the distributed world size differs.
token_job_id=$(submit_ablation \
  token_only \
  luna-alf-token-a800 \
  A800 \
  4 \
  64 \
  480G \
  ppo_qwen25_15b_luna_tokenonly2h_alfworld_t128_v140_vb20_a800x4_seed0_20260823 \
  lunatokena800_0823 \
  64)

turn_job_id=$(submit_ablation \
  turn_only \
  luna-alf-turn-a800 \
  A800 \
  4 \
  64 \
  480G \
  ppo_qwen25_15b_luna_turnonly2h_alfworld_t128_v140_vb20_a800x4_seed0_20260823 \
  lunaturna800_0823 \
  64)

echo "direct_a100_job_id=$direct_job_id"
echo "token_a800_job_id=$token_job_id"
echo "turn_a800_job_id=$turn_job_id"
echo "logs=$LOG_ROOT"
