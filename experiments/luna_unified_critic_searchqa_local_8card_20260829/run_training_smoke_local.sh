#!/usr/bin/env bash
set -euo pipefail

# One-update integration smoke test. The local retriever must already own GPU 7.
PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
DATA_ROOT=${DATA_ROOT:-/home/dataset-local/cjj/RL/data/searchR1_official/processed}
SNAP=${SNAP:-/home/dataset-local/cjj/RL/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
PORT=${PORT:-18002}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
NUM_GPUS=${NUM_GPUS:-7}
RUN_NAME=${RUN_NAME:-luna_unified_searchqa_local_train_smoke_20260829}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
CKPT_DIR=${CKPT_DIR:-/home/dataset-local/cjj/RL/checkpoints/luna_unified_searchqa/$RUN_NAME}
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/dev/shm/cjj_luna_searchqa_local_smoke_ray}
FAST_TMP_ROOT=${FAST_TMP_ROOT:-/dev/shm/cjj_luna_searchqa_local_smoke_tmp}

test -s "$DATA_ROOT/train.parquet"
test -s "$DATA_ROOT/test.parquet"
test -s "$SNAP/config.json"
curl --fail --silent --show-error --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null

mkdir -p "$RUN_DIR"/{home,logs,wandb} "$CKPT_DIR" "$RAY_TMP_ROOT" "$FAST_TMP_ROOT"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
export HOME="$RUN_DIR/home"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ENV_ROOT/bin:$PATH"
export TMPDIR="$FAST_TMP_ROOT"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export RAY_TMPDIR="$RAY_TMP_ROOT"
export HF_HOME=${HF_HOME:-/home/dataset-local/cjj/RL/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_DIR="$RUN_DIR/wandb"
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "$PROJECT_ROOT"
exec python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=luna_unified \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.hybrid_advantage.turn_gamma=0.95 \
  algorithm.hybrid_advantage.turn_lam=0.95 \
  algorithm.hybrid_advantage.token_residual_scale=1.0 \
  algorithm.hybrid_advantage.composition_mode=residual \
  algorithm.hybrid_advantage.whiten_advantages=False \
  algorithm.use_kl_in_reward=False \
  reward_model.use_step_rewards=True \
  data.train_files="$DATA_ROOT/train.parquet" \
  data.val_files="$DATA_ROOT/test.parquet" \
  data.train_batch_size=8 \
  data.val_batch_size=4 \
  data.max_prompt_length=4096 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=left \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$SNAP" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
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
  env.search.timeout=30 \
  env.search.fail_on_error=true \
  env.resources_per_worker.num_cpus=0.25 \
  trainer.critic_warmup=0 \
  "trainer.logger=['console']" \
  trainer.project_name=verl_agent_searchqa_critic_ablation \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.test_freq=1 \
  trainer.save_freq=1 \
  trainer.checkpoint_slot_mode=best_latest \
  trainer.best_checkpoint_metric=val/success_rate \
  trainer.monitor_validation_size=4 \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  ray_init.num_cpus=32 \
  +ray_init._temp_dir="$RAY_TMP_ROOT"
