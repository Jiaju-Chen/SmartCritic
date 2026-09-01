#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
ENV_ROOT=${ENV_ROOT:-/root/vepfs-data/chenjiaju/conda_envs/smartcritic-webshop}
MODEL_ROOT=${MODEL_ROOT:-/root/vepfs-data/chenjiaju/models/Qwen2.5-1.5B-Instruct}
RESOURCE_ROOT=${WEBSHOP_RESOURCE_ROOT:-/root/vepfs-data/chenjiaju/datasets/webshop-small}
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-/root/vepfs-data/chenjiaju/datasets/webshop-parquet/eval500/text}
RUN_BASE=${RUN_BASE:-/root/vepfs-data/chenjiaju/runs/SmartCritic/luna_webshop_gamma_alpha_20260902}
CKPT_BASE=${CKPT_BASE:-/root/vepfs-data/chenjiaju/checkpoints/luna_webshop_gamma_alpha_20260902}

EXPERIMENT_ID=${EXPERIMENT_ID:?EXPERIMENT_ID is required}
ALPHA=${ALPHA:?ALPHA is required}
TURN_GAMMA=${TURN_GAMMA:-0.95}
TURN_LAMBDA=${TURN_LAMBDA:-0.95}
TOKEN_GAMMA=${TOKEN_GAMMA:-1.0}
TOKEN_LAMBDA=${TOKEN_LAMBDA:-1.0}
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-luna_webshop_${EXPERIMENT_ID}_t128_v128_8gpu_seed0_20260902}
CKPT_DIR=${CKPT_DIR:-$CKPT_BASE/$TRAIN_RUN_NAME}
EVAL_NAME=${EVAL_NAME:-${TRAIN_RUN_NAME}_latest_eval500}
WANDB_RUN_ID=${WANDB_RUN_ID:-lzw${EXPERIMENT_ID}e500_0902}
OUT_DIR=${OUT_DIR:-$RUN_BASE/eval500/$EVAL_NAME}
LOG=${LOG:-$OUT_DIR/eval.log}
SUMMARY=${SUMMARY:-$OUT_DIR/summary.txt}
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/root/vepfs-data/chenjiaju/tmp/smartcritic/ray/$EVAL_NAME}
FAST_TMP_ROOT=${FAST_TMP_ROOT:-/root/vepfs-data/chenjiaju/tmp/smartcritic/jobs/$EVAL_NAME}
PYTHON=$ENV_ROOT/bin/python

if [[ -s "$SUMMARY" ]] && grep -q '^status=finished$' "$SUMMARY" &&
   grep -q '^evaluated_cases=500$' "$SUMMARY"; then
  echo "Evaluation already finished: $EVAL_NAME"
  exit 0
fi

for path in "$PYTHON" "$MODEL_ROOT/config.json" "$EVAL_DATA_ROOT/train.parquet" "$EVAL_DATA_ROOT/test.parquet" "$CKPT_DIR/latest_checkpointed_iteration.txt"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required evaluation path: $path" >&2
    exit 2
  fi
done

latest_step=$(tr -d '[:space:]' < "$CKPT_DIR/latest_checkpointed_iteration.txt")
if [[ "$latest_step" != "150" ]]; then
  echo "Expected latest checkpoint step 150, found $latest_step" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"/{home,wandb} "$RAY_TMP_ROOT" "$FAST_TMP_ROOT"
PROJECT_ROOT=$PROJECT_ROOT WEBSHOP_RESOURCE_ROOT=$RESOURCE_ROOT \
  "$SCRIPT_DIR/prepare_resources.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOME=$OUT_DIR/home
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
export WANDB_DIR=$OUT_DIR/wandb
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
echo "Evaluating $TRAIN_RUN_NAME latest step $latest_step on 500 cases" | tee -a "$LOG"

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
  data.train_files="$EVAL_DATA_ROOT/train.parquet" \
  data.val_files="$EVAL_DATA_ROOT/test.parquet" \
  data.train_batch_size=128 \
  data.val_batch_size=50 \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
  +critic.model.override_config.num_hidden_layers=2 \
  critic.model.num_value_heads=2 \
  critic.unified_luna_turn_loss_coef=1.0 \
  critic.optim.lr=1e-5 \
  critic.ppo_mini_batch_size=64 \
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
  trainer.project_name=verl_agent_webshop_500_eval \
  trainer.experiment_name="$EVAL_NAME" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.checkpoint_slot_mode=best_latest \
  trainer.monitor_validation_size=32 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.resume_mode=auto \
  ray_init.num_cpus=96 \
  +ray_init._temp_dir="$RAY_TMPDIR" \
  2>&1 | tee -a "$LOG"
eval_status=${PIPESTATUS[0]}
set -e

if (( eval_status != 0 )); then
  printf 'status=failed\nexit_code=%s\nfinished_at=%s\n' \
    "$eval_status" "$(date -Is)" > "$SUMMARY"
  exit "$eval_status"
fi

if ! grep -Eq 'val/evaluated_cases[^0-9]+500([.]0+)?([^0-9]|$)' "$LOG"; then
  echo "Evaluation exited without confirming 500 evaluated cases" >&2
  exit 3
fi

{
  echo 'status=finished'
  echo 'checkpoint=latest'
  echo "checkpoint_step=$latest_step"
  echo 'evaluated_cases=500'
  echo "alpha=$ALPHA"
  echo "finished_at=$(date -Is)"
  grep -E 'Initial validation metrics|val/(full_success_rate|webshop_task_score|evaluated_cases)' "$LOG" | tail -n 3 || true
} > "$SUMMARY"

echo "Evaluation finished: $EVAL_NAME"

