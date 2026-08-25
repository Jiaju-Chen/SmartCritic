#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
if [[ ${SKIP_CONDA_ACTIVATE:-0} == 1 ]]; then
  export PATH="$ENV_ROOT/bin:$PATH"
  export CONDA_PREFIX="$ENV_ROOT"
  export PYTHONNOUSERSITE=1
else
  set +u
  source /opt/conda/bin/activate "$ENV_ROOT"
  set -u
fi
cd "$PROJECT_ROOT"

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified_residual_searchqa_smoke_t8_v4_8gpu_seed0_20260825}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
CKPT_DIR=${CKPT_DIR:-/home/dataset-local/cjj/RL/checkpoints/luna_unified_searchqa/$RUN_NAME}
SNAP=${SNAP:-/home/dataset-local/cjj/RL/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
PORT=${SEARCH_PORT:-8010}
NUM_GPUS=${NUM_GPUS:-8}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}

ORIGINAL_HOME=${ORIGINAL_HOME:-/home/batchcom}
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/dev/shm/cjj_luna_searchqa_smoke_ray}
FAST_TMP_ROOT=${FAST_TMP_ROOT:-/dev/shm/cjj_luna_searchqa_smoke_tmp}
mkdir -p "$RUN_DIR"/{home,logs,wandb,data} "$CKPT_DIR" "$RAY_TMP_ROOT" "$FAST_TMP_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOME="$RUN_DIR/home"
export NETRC=${NETRC:-$ORIGINAL_HOME/.netrc}
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ENV_ROOT/bin:$PATH"
export TMPDIR="$FAST_TMP_ROOT"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export RAY_TMPDIR="$RAY_TMP_ROOT"
export HF_HOME=${HF_HOME:-/home/dataset-local/cjj/RL/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_RUN_ID=${WANDB_RUN_ID:-lunasearchsmoke_0825}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-30}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

python experiments/luna_unified_critic_searchqa_20260825/make_smoke_dataset.py \
  --output-dir "$RUN_DIR/data" --train-size 8 --val-size 4

python experiments/luna_unified_critic_searchqa_20260825/mini_retrieval_server.py \
  --port "$PORT" >"$RUN_DIR/logs/retriever.log" 2>&1 &
RETRIEVER_PID=$!
trap 'kill "$RETRIEVER_PID" 2>/dev/null || true; wait "$RETRIEVER_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

HYDRA_ARGS=()
if [[ ${CONFIG_ONLY:-0} == 1 ]]; then
  HYDRA_ARGS+=(--cfg job)
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=luna_unified \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.hybrid_advantage.turn_gamma=1.0 \
  algorithm.hybrid_advantage.turn_lam=0.95 \
  algorithm.hybrid_advantage.token_residual_scale=1.0 \
  algorithm.hybrid_advantage.composition_mode=residual \
  algorithm.hybrid_advantage.whiten_advantages=False \
  algorithm.use_kl_in_reward=False \
  reward_model.use_step_rewards=True \
  data.train_files="$RUN_DIR/data/train.parquet" \
  data.val_files="$RUN_DIR/data/test.parquet" \
  data.train_batch_size=8 \
  data.val_batch_size=4 \
  data.max_prompt_length=4096 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$SNAP" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_PARALLEL_SIZE" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
  critic.model.path="$SNAP" \
  critic.model.use_remove_padding=False \
  critic.model.enable_gradient_checkpointing=True \
  +critic.model.override_config.num_hidden_layers=2 \
  critic.model.num_value_heads=2 \
  critic.unified_luna_turn_loss_coef=1.0 \
  critic.optim.lr=1e-5 \
  critic.ppo_mini_batch_size=8 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  env.env_name=search \
  env.seed=0 \
  env.max_steps=4 \
  env.history_length=4 \
  env.rollout.n=1 \
  env.search.search_url="http://127.0.0.1:$PORT/retrieve" \
  env.search.topk=3 \
  env.search.timeout=5 \
  env.resources_per_worker.num_cpus=0.25 \
  trainer.critic_warmup=0 \
  "trainer.logger=['console','wandb']" \
  trainer.project_name=verl_agent_searchqa_critic_ablation \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=1 \
  trainer.checkpoint_slot_mode=best_latest \
  trainer.best_checkpoint_metric=val/success_rate \
  trainer.monitor_validation_size=4 \
  trainer.val_before_train=True \
  trainer.resume_mode=disable \
  ray_init.num_cpus=32 \
  +ray_init._temp_dir="$RAY_TMPDIR" \
  "${HYDRA_ARGS[@]}"
