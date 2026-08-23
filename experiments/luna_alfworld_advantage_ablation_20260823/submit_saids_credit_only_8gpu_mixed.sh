#!/usr/bin/env bash
set -euo pipefail

# Submit the remaining Luna ablations on eight GPUs each: token-only on A100
# and turn-only on A800. Global optimization semantics match the direct run.
PROJECT_ROOT=${PROJECT_ROOT:-/data2/group_何向南/chenjiaju/luna/worktrees/alfworld-direct-20260823}
SHARED_ROOT=${SHARED_ROOT:-/data2/group_何向南/chenjiaju/luna/shared}
LOG_ROOT=${LOG_ROOT:-$SHARED_ROOT/slurm_logs/luna_alfworld_advantage_ablation_20260823}
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

submit_ablation() {
  local composition=$1
  local job_name=$2
  local partition=$3
  local run_name=$4
  local wandb_id=$5

  sbatch --parsable \
    --job-name="$job_name" \
    --partition="$partition" \
    --nodes=1 \
    --gres=gpu:8 \
    --cpus-per-task=96 \
    --mem=800G \
    --time=3-00:00:00 \
    --output="$LOG_ROOT/${composition}-%j.out" \
    --error="$LOG_ROOT/${composition}-%j.err" \
    --export="ALL,ADVANTAGE_COMPOSITION=$composition,RUN_NAME=$run_name,WANDB_RUN_ID=$wandb_id,WANDB_MODE=offline,N_GPUS_PER_NODE=8,ROLLOUT_TP_SIZE=2,RAY_NUM_CPUS=96,NUM_CPUS_PER_ENV_WORKER=0.5,TRAIN_DATA_SIZE=128,VAL_DATA_SIZE=140,VAL_BATCH_SIZE=20,PPO_MINI_BATCH_SIZE=256,CRITIC_NUM_LAYERS=2,TURN_LOSS_COEF=1.0,TOTAL_EPOCHS=150,TEST_FREQ=5,SAVE_FREQ=5,VAL_BEFORE_TRAIN=False" \
    experiments/luna_alfworld_advantage_ablation_20260823/saids_train.sh
}

token_job_id=$(submit_ablation \
  token_only \
  luna-alf-token-a100x8 \
  A100 \
  ppo_qwen25_15b_luna_tokenonly2h_alfworld_t128_v140_vb20_a100x8_seed0_20260823 \
  lunatokena100x8_0823)

turn_job_id=$(submit_ablation \
  turn_only \
  luna-alf-turn-a800x8 \
  A800 \
  ppo_qwen25_15b_luna_turnonly2h_alfworld_t128_v140_vb20_a800x8_seed0_20260823 \
  lunaturna800x8_0823)

echo "token_a100x8_job_id=$token_job_id"
echo "turn_a800x8_job_id=$turn_job_id"
echo "logs=$LOG_ROOT"
