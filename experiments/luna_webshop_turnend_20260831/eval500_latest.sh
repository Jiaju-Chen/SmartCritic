#!/usr/bin/env bash
set -euo pipefail

# Evaluate the action-end ablation's step-150 latest checkpoint on the same
# fixed 500-case WebShop set used by the existing Luna comparison table.

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-webshop-turnend}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
DATA_ROOT=${DATA_ROOT:-/home/dataset-local/cjj/RL/evals/webshop_dualcritic_v4_500/data/verl-agent/text}
MODEL=${MODEL:-/home/dataset-local/cjj/RL/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
CKPT_ROOT=${CKPT_ROOT:-/home/dataset-local/cjj/RL/checkpoints/luna_webshop_turnend/luna_unified2h_webshop_actionend_t128_v128_8gpu_seed0_20260831}
OUT_ROOT=${OUT_ROOT:-/home/dataset-local/cjj/RL/evals/webshop_actionend_500_20260901}
RUN_NAME=${RUN_NAME:-webshop500_luna_action_end_latest_step150_20260901}
WANDB_RUN_ID=${WANDB_RUN_ID:-lunaend500_0901}

PYTHON=$ENV_ROOT/bin/python
WANDB=$ENV_ROOT/bin/wandb
LOG=$OUT_ROOT/eval.log
SUMMARY=$OUT_ROOT/summary.txt
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/home/dataset-local/cjj/lwe500_r}
FAST_TMP_ROOT=${FAST_TMP_ROOT:-/home/dataset-local/cjj/lwe500_t}

if [[ -s "$SUMMARY" ]] && grep -q '^status=finished$' "$SUMMARY"; then
  echo "Evaluation already finished: $SUMMARY"
  exit 0
fi

if [[ ! -f "$CKPT_ROOT/latest_checkpointed_iteration.txt" ]]; then
  echo "Missing latest checkpoint marker: $CKPT_ROOT/latest_checkpointed_iteration.txt" >&2
  exit 2
fi

latest_step=$(tr -d '[:space:]' < "$CKPT_ROOT/latest_checkpointed_iteration.txt")
if [[ "$latest_step" != "150" ]]; then
  echo "Expected latest step 150, found: $latest_step" >&2
  exit 2
fi

if [[ ! -f "$DATA_ROOT/test.parquet" ]]; then
  echo "Missing fixed WebShop-500 parquet: $DATA_ROOT/test.parquet" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"/{home,wandb} "$RAY_TMP_ROOT" "$FAST_TMP_ROOT"

export PROJECT_ROOT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOME=$OUT_ROOT/home
export NETRC=${NETRC:-/home/batchcom/.netrc}
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAVA_HOME=$ENV_ROOT
export JAVA_LD_LIBRARY_PATH=$ENV_ROOT/lib/server
export PATH="$ENV_ROOT/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export TMPDIR=$FAST_TMP_ROOT
export TEMP=$FAST_TMP_ROOT
export TMP=$FAST_TMP_ROOT
export RAY_TMPDIR=$RAY_TMP_ROOT
export HF_HOME=${HF_HOME:-/home/dataset-local/cjj/RL/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_ENTITY=${WANDB_ENTITY:-cjj01-ustc}
export WANDB_MODE=offline
export WANDB_DIR=$OUT_ROOT/wandb
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
export WEBSHOP_ENVS_PER_WORKER=8
export WEBSHOP_ENV_INIT_BATCH_SIZE=4
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "$PROJECT_ROOT"
echo "Starting $RUN_NAME at $(date -Is); checkpoint_step=$latest_step"

set +e
"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=luna_unified \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.hybrid_advantage.turn_gamma=1.0 \
  algorithm.hybrid_advantage.turn_lam=0.95 \
  algorithm.hybrid_advantage.token_residual_scale=1.0 \
  algorithm.hybrid_advantage.composition_mode=residual \
  algorithm.hybrid_advantage.turn_value_position=action_end \
  algorithm.hybrid_advantage.whiten_advantages=True \
  algorithm.use_kl_in_reward=False \
  reward_model.use_step_rewards=True \
  data.train_files="$DATA_ROOT/train.parquet" \
  data.val_files="$DATA_ROOT/test.parquet" \
  data.train_batch_size=128 \
  data.val_batch_size=50 \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL" \
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
  critic.model.path="$MODEL" \
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
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$CKPT_ROOT" \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.checkpoint_slot_mode=best_latest \
  trainer.monitor_validation_size=32 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.resume_mode=auto \
  ray_init.num_cpus=96 \
  +ray_init._temp_dir="$RAY_TMP_ROOT" \
  > "$LOG" 2>&1
eval_status=$?
set -e

if [[ $eval_status -ne 0 ]]; then
  printf 'status=failed\nexit_code=%s\nfinished_at=%s\n' \
    "$eval_status" "$(date -Is)" > "$SUMMARY"
  echo "Evaluation failed with exit code $eval_status; see $LOG" >&2
  exit "$eval_status"
fi

offline_dir=$(find "$WANDB_DIR" -maxdepth 2 -type d -name 'offline-run-*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)
sync_status=0
if [[ -n "$offline_dir" ]]; then
  "$WANDB" sync "$offline_dir" --project verl_agent_webshop_500_eval || sync_status=$?
fi

{
  echo 'status=finished'
  echo 'checkpoint=latest'
  echo "checkpoint_step=$latest_step"
  echo 'evaluated_cases_expected=500'
  echo "wandb_sync_exit_code=$sync_status"
  echo "finished_at=$(date -Is)"
  grep -E 'val/(success_rate|full_success_rate|webshop_task_score|text/test_score|evaluated_cases)' "$LOG" | tail -n 1 || true
} > "$SUMMARY"

echo "Evaluation finished at $(date -Is); sync_status=$sync_status"

