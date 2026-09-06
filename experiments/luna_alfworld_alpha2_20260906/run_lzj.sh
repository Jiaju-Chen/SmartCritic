#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
BASE=/root/vepfs-data/chenjiaju
ENV_DIR=${ENV_DIR:-$BASE/conda_envs/smartcritic-webshop-20260902}

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2h_residual_alpha2_alfworld_t128_v140_vb20_8gpu_seed0_20260906}
RUN_ROOT=${RUN_ROOT:-$BASE/runs/SmartCritic/luna_alfworld_alpha2_20260906}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
CKPT_DIR=${CKPT_DIR:-$BASE/checkpoints/luna_alfworld_alpha2_20260906/$RUN_NAME}
MODEL_PATH=${MODEL_PATH:-$BASE/models/Qwen2.5-1.5B-Instruct}
ALFWORLD_DATA=${ALFWORLD_DATA:-$BASE/data/alfworld_data}
PREPARED_DATA_DIR=${PREPARED_DATA_DIR:-$BASE/data/verl-agent/text}

TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-128}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-140}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-20}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-96}
WANDB_RUN_ID=${WANDB_RUN_ID:-lunaalf15a2_0906}

mkdir -p \
  "$RUN_DIR"/{home,logs,wandb,cache,tmp} \
  "$CKPT_DIR" \
  "$BASE/r2" \
  "$BASE/t2"

set +u
source "$ENV_DIR/bin/activate"
set -u
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOME=$RUN_DIR/home
export TMPDIR=$BASE/t2
export TEMP=$TMPDIR
export TMP=$TMPDIR
export RAY_TMPDIR=$BASE/r2
export ALFWORLD_DATA
export HF_HOME=$BASE/cache/smartcritic/huggingface
export XDG_CACHE_HOME=$RUN_DIR/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_ENTITY=${WANDB_ENTITY:-cjj01-ustc}
export WANDB_DIR=$RUN_DIR/wandb
export WANDB_RUN_ID
export WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-30}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.5}
export TRAIN_DATA_SIZE
export VAL_DATA_SIZE
export SKIP_DATA_PREP=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

test -x "$ENV_DIR/bin/python"
test -f "$MODEL_PATH/config.json"
test -d "$ALFWORLD_DATA/json_2.1.1/train"
test -d "$ALFWORLD_DATA/json_2.1.1/valid_seen"
test -f "$PREPARED_DATA_DIR/train.parquet"
test -f "$PREPARED_DATA_DIR/test.parquet"

mkdir -p "$HOME/data/verl-agent/text"
cp "$PREPARED_DATA_DIR/train.parquet" "$HOME/data/verl-agent/text/train.parquet"
cp "$PREPARED_DATA_DIR/test.parquet" "$HOME/data/verl-agent/text/test.parquet"

python - <<'PY'
import alfworld
import textworld
PY

if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then
  echo "ALFWorld alpha=2 launcher preflight passed"
  exit 0
fi

bash examples/ppo_trainer/run_alfworld.sh vllm \
  algorithm.adv_estimator=luna_unified \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.hybrid_advantage.turn_gamma=1.0 \
  algorithm.hybrid_advantage.turn_lam=0.95 \
  algorithm.hybrid_advantage.token_residual_scale=2.0 \
  algorithm.hybrid_advantage.whiten_advantages=True \
  reward_model.use_step_rewards=True \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  env.rollout.n=1 \
  +critic.model.override_config.num_hidden_layers=2 \
  critic.model.num_value_heads=2 \
  critic.unified_luna_turn_loss_coef=1.0 \
  "trainer.logger=['console','wandb']" \
  trainer.project_name=verl_agent_alfworld_critic_ablation \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.checkpoint_slot_mode=best_latest \
  trainer.best_checkpoint_metric=val/success_rate \
  trainer.monitor_validation_size=32 \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.resume_mode=auto \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  critic.model.use_remove_padding=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  critic.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  critic.optim.lr=1e-5 \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_kl_loss=True \
  algorithm.use_kl_in_reward=False \
  env.seed=0 \
  env.max_steps=50 \
  env.resources_per_worker.num_cpus="$NUM_CPUS_PER_ENV_WORKER" \
  ray_init.num_cpus="$RAY_NUM_CPUS" \
  +ray_init._temp_dir="$RAY_TMPDIR"
