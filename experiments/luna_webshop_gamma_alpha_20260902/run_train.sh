#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
ENV_ROOT=${ENV_ROOT:-/root/vepfs-data/chenjiaju/conda_envs/smartcritic-webshop-20260902}
MODEL_ROOT=${MODEL_ROOT:-/root/vepfs-data/chenjiaju/models/Qwen2.5-1.5B-Instruct}
RESOURCE_ROOT=${WEBSHOP_RESOURCE_ROOT:-/root/vepfs-data/chenjiaju/datasets/webshop-small}
TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT:-/root/vepfs-data/chenjiaju/datasets/webshop-parquet/train128-val128/text}
RUN_BASE=${RUN_BASE:-/root/vepfs-data/chenjiaju/runs/SmartCritic/luna_webshop_gamma_alpha_20260902}
CKPT_BASE=${CKPT_BASE:-/root/vepfs-data/chenjiaju/checkpoints/luna_webshop_gamma_alpha_20260902}

EXPERIMENT_ID=${EXPERIMENT_ID:?EXPERIMENT_ID is required, for example g0_alpha1}
ALPHA=${ALPHA:?ALPHA is required}
TURN_GAMMA=${TURN_GAMMA:-0.95}
TURN_LAMBDA=${TURN_LAMBDA:-0.95}
TOKEN_GAMMA=${TOKEN_GAMMA:-1.0}
TOKEN_LAMBDA=${TOKEN_LAMBDA:-1.0}
TOTAL_STEPS=${TOTAL_STEPS:-150}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-5}
CRITIC_NUM_LAYERS=${CRITIC_NUM_LAYERS:-2}
RUN_NAME=${RUN_NAME:-luna_webshop_${EXPERIMENT_ID}_t128_v128_8gpu_seed0_20260902}
WANDB_RUN_ID=${WANDB_RUN_ID:-lzw${EXPERIMENT_ID}_0902}
RUN_DIR=${RUN_DIR:-$RUN_BASE/$RUN_NAME}
CKPT_DIR=${CKPT_DIR:-$CKPT_BASE/$RUN_NAME}
LOG=${LOG:-$RUN_DIR/train.log}
SUMMARY=${SUMMARY:-$RUN_DIR/summary.txt}
# Ray's AF_UNIX socket path is limited to 107 bytes. Keep the Ray root short;
# Ray creates a unique session directory beneath it for each sequential phase.
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/root/vepfs-data/chenjiaju/r}
FAST_TMP_ROOT=${FAST_TMP_ROOT:-/root/vepfs-data/chenjiaju/t/$EXPERIMENT_ID}
PYTHON=$ENV_ROOT/bin/python

for path in "$PYTHON" "$MODEL_ROOT/config.json" "$TRAIN_DATA_ROOT/train.parquet" "$TRAIN_DATA_ROOT/test.parquet"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required training path: $path" >&2
    exit 2
  fi
done

if [[ "$SAVE_FREQ" != "$TEST_FREQ" ]]; then
  echo "best_latest requires SAVE_FREQ == TEST_FREQ; got $SAVE_FREQ and $TEST_FREQ" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"/{home,wandb} "$CKPT_DIR" "$RAY_TMP_ROOT" "$FAST_TMP_ROOT"

if [[ -s "$SUMMARY" ]] && grep -q '^status=finished$' "$SUMMARY" &&
   [[ -s "$CKPT_DIR/latest_checkpointed_iteration.txt" ]] &&
   [[ "$(tr -d '[:space:]' < "$CKPT_DIR/latest_checkpointed_iteration.txt")" == "$TOTAL_STEPS" ]]; then
  echo "Training already finished: $RUN_NAME"
  exit 0
fi

if [[ -s "$CKPT_DIR/latest_checkpointed_iteration.txt" ]]; then
  current_step=$(tr -d '[:space:]' < "$CKPT_DIR/latest_checkpointed_iteration.txt")
  if (( current_step > TOTAL_STEPS )); then
    echo "Checkpoint step $current_step exceeds requested total $TOTAL_STEPS" >&2
    exit 2
  fi
fi

PROJECT_ROOT=$PROJECT_ROOT WEBSHOP_RESOURCE_ROOT=$RESOURCE_ROOT \
  "$SCRIPT_DIR/prepare_resources.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOME=$RUN_DIR/home
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAVA_HOME=$ENV_ROOT
export JAVA_LD_LIBRARY_PATH=$ENV_ROOT/lib/server
export PATH="$ENV_ROOT/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export TMPDIR=$FAST_TMP_ROOT
export TEMP=$FAST_TMP_ROOT
export TMP=$FAST_TMP_ROOT
export RAY_TMPDIR=$RAY_TMP_ROOT
export HF_HOME=${HF_HOME:-/root/vepfs-data/chenjiaju/cache/smartcritic/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_ENTITY=cjj01-ustc
export WANDB_MODE=offline
export WANDB_DIR=$RUN_DIR/wandb
export WANDB_RUN_ID
export WANDB_RESUME=allow
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export JAVA_TOOL_OPTIONS="-Xms32m -Xmx512m -XX:ActiveProcessorCount=1 -XX:+UseSerialGC"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export WEBSHOP_ENVS_PER_WORKER=${WEBSHOP_ENVS_PER_WORKER:-8}
export WEBSHOP_ENV_INIT_BATCH_SIZE=${WEBSHOP_ENV_INIT_BATCH_SIZE:-4}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "$PROJECT_ROOT"
{
  echo "run_name=$RUN_NAME"
  echo "started_at=$(date -Is)"
  echo "turn_gamma=$TURN_GAMMA"
  echo "turn_lambda=$TURN_LAMBDA"
  echo "token_gamma=$TOKEN_GAMMA"
  echo "token_lambda=$TOKEN_LAMBDA"
  echo "alpha=$ALPHA"
  echo "critic_layers=$CRITIC_NUM_LAYERS"
  echo "train_batch_size=$TRAIN_BATCH_SIZE"
  echo "ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  echo "total_steps=$TOTAL_STEPS"
  echo "wandb_mode=$WANDB_MODE"
} | tee -a "$LOG"

set +e
"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=luna_unified \
  algorithm.gamma="$TOKEN_GAMMA" \
  algorithm.lam="$TOKEN_LAMBDA" \
  algorithm.hybrid_advantage.turn_gamma="$TURN_GAMMA" \
  algorithm.hybrid_advantage.turn_lam="$TURN_LAMBDA" \
  algorithm.hybrid_advantage.token_residual_scale="$ALPHA" \
  algorithm.hybrid_advantage.composition_mode=residual \
  algorithm.hybrid_advantage.whiten_advantages=True \
  algorithm.use_kl_in_reward=False \
  reward_model.use_step_rewards=True \
  data.train_files="$TRAIN_DATA_ROOT/train.parquet" \
  data.val_files="$TRAIN_DATA_ROOT/test.parquet" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_ROOT" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  critic.model.path="$MODEL_ROOT" \
  critic.model.use_remove_padding=False \
  critic.model.enable_gradient_checkpointing=True \
  +critic.model.override_config.num_hidden_layers="$CRITIC_NUM_LAYERS" \
  critic.model.num_value_heads=2 \
  critic.unified_luna_turn_loss_coef=1.0 \
  critic.optim.lr=1e-5 \
  critic.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  env.env_name=webshop/WebAgentTextEnv \
  env.seed=0 \
  env.max_steps=15 \
  env.history_length=2 \
  env.rollout.n=1 \
  env.webshop.use_small=True \
  env.webshop.human_goals=False \
  env.resources_per_worker.num_cpus=0.25 \
  trainer.critic_warmup=0 \
  "trainer.logger=['console','wandb']" \
  trainer.project_name=verl_agent_webshop_critic_ablation \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.total_epochs="$TOTAL_STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.checkpoint_slot_mode=best_latest \
  "trainer.best_checkpoint_metric='val/webshop_task_score (not success_rate)'" \
  trainer.monitor_validation_size=32 \
  trainer.val_before_train=True \
  trainer.resume_mode=auto \
  ray_init.num_cpus=96 \
  +ray_init._temp_dir="$RAY_TMPDIR" \
  2>&1 | tee -a "$LOG"
train_status=${PIPESTATUS[0]}
set -e

if (( train_status != 0 )); then
  printf 'status=failed\nexit_code=%s\nfinished_at=%s\n' \
    "$train_status" "$(date -Is)" > "$SUMMARY"
  exit "$train_status"
fi

if [[ ! -s "$CKPT_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "Training exited without a latest checkpoint marker" >&2
  exit 3
fi
latest_step=$(tr -d '[:space:]' < "$CKPT_DIR/latest_checkpointed_iteration.txt")
if [[ "$latest_step" != "$TOTAL_STEPS" ]]; then
  echo "Training exited at checkpoint step $latest_step, expected $TOTAL_STEPS" >&2
  exit 3
fi

{
  echo 'status=finished'
  echo "checkpoint_step=$latest_step"
  echo "alpha=$ALPHA"
  echo "turn_gamma=$TURN_GAMMA"
  echo "turn_lambda=$TURN_LAMBDA"
  echo "token_gamma=$TOKEN_GAMMA"
  echo "token_lambda=$TOKEN_LAMBDA"
  echo "finished_at=$(date -Is)"
  grep -E 'Final validation metrics|val/(full_success_rate|webshop_task_score|evaluated_cases)' "$LOG" | tail -n 3 || true
} > "$SUMMARY"

echo "Training finished: $RUN_NAME at step $latest_step"
