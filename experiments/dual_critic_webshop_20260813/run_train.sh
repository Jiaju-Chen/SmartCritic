#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF_WebShop}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
export JAVA_HOME=${JAVA_HOME:-$ENV_ROOT}
export JAVA_LD_LIBRARY_PATH=${JAVA_LD_LIBRARY_PATH:-$JAVA_HOME/lib/server}
set +u
source /opt/conda/bin/activate "$ENV_ROOT"
set -u
cd "$PROJECT_ROOT"

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_dualcritic2l_webshop_t128_v128_8gpu_seed0_20260813}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/dual_critic_webshop}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
CKPT_DIR=${CKPT_DIR:-/home/dataset-local/cjj/RL/checkpoints/dual_critic_webshop/$RUN_NAME}
SNAP=${SNAP:-/home/dataset-local/cjj/RL/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}

ORIGINAL_HOME=${ORIGINAL_HOME:-/home/batchcom}
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/home/dataset-local/cjj/dcw_r}
FAST_TMP_ROOT=${FAST_TMP_ROOT:-/home/dataset-local/cjj/dcw_t}
mkdir -p "$RUN_DIR"/{home,logs,wandb} "$CKPT_DIR" "$RAY_TMP_ROOT" "$FAST_TMP_ROOT"

bash experiments/dual_critic_webshop_20260813/prepare_resources.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOME=$RUN_DIR/home
export NETRC=${NETRC:-$ORIGINAL_HOME/.netrc}
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$JAVA_HOME/bin:$PATH"
export TMPDIR=$FAST_TMP_ROOT
export TEMP=$TMPDIR
export TMP=$TMPDIR
export RAY_TMPDIR=$RAY_TMP_ROOT
export HF_HOME=${HF_HOME:-/home/dataset-local/cjj/RL/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=$RUN_DIR/wandb
export WANDB_RUN_ID=${WANDB_RUN_ID:-dcw2l_0813}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-30}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS:-"-Xms32m -Xmx256m -XX:ActiveProcessorCount=1 -XX:+UseSerialGC"}
export WEBSHOP_ENV_INIT_BATCH_SIZE=${WEBSHOP_ENV_INIT_BATCH_SIZE:-16}
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-128}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
CRITIC_NUM_LAYERS=${CRITIC_NUM_LAYERS:-2}
TURN_CRITIC_NUM_LAYERS=${TURN_CRITIC_NUM_LAYERS:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.25}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-96}

python -m examples.data_preprocess.prepare \
  --mode text \
  --local_dir "$HOME/data/verl-agent" \
  --train_data_size "$TRAIN_DATA_SIZE" \
  --val_data_size "$VAL_DATA_SIZE"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=dual_critic_hybrid \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.hybrid_advantage.turn_gamma=1.0 \
  algorithm.hybrid_advantage.turn_lam=0.95 \
  algorithm.hybrid_advantage.token_residual_scale=1.0 \
  algorithm.hybrid_advantage.whiten_advantages=True \
  algorithm.use_kl_in_reward=False \
  reward_model.use_step_rewards=True \
  data.train_files="$HOME/data/verl-agent/text/train.parquet" \
  data.val_files="$HOME/data/verl-agent/text/test.parquet" \
  data.train_batch_size="$TRAIN_DATA_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$SNAP" \
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
  critic.model.path="$SNAP" \
  critic.model.use_remove_padding=False \
  critic.model.enable_gradient_checkpointing=True \
  +critic.model.override_config.num_hidden_layers="$CRITIC_NUM_LAYERS" \
  critic.optim.lr=1e-5 \
  critic.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  turn_critic.model.path="$SNAP" \
  turn_critic.model.use_remove_padding=False \
  turn_critic.model.enable_gradient_checkpointing=True \
  +turn_critic.model.override_config.num_hidden_layers="$TURN_CRITIC_NUM_LAYERS" \
  turn_critic.optim.lr=1e-5 \
  turn_critic.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  turn_critic.ppo_micro_batch_size_per_gpu=1 \
  turn_critic.model.fsdp_config.param_offload=False \
  turn_critic.model.fsdp_config.optimizer_offload=False \
  env.env_name=webshop/WebAgentTextEnv \
  env.seed=0 \
  env.max_steps=15 \
  env.history_length=2 \
  env.rollout.n=1 \
  env.webshop.use_small=True \
  env.webshop.human_goals=False \
  env.resources_per_worker.num_cpus="$NUM_CPUS_PER_ENV_WORKER" \
  trainer.critic_warmup=0 \
  "trainer.logger=['console','wandb']" \
  trainer.project_name=verl_agent_webshop_critic_ablation \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.checkpoint_slot_mode=best_latest \
  "trainer.best_checkpoint_metric='val/webshop_task_score (not success_rate)'" \
  trainer.monitor_validation_size=32 \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.resume_mode=auto \
  ray_init.num_cpus="$RAY_NUM_CPUS" \
  +ray_init._temp_dir="$RAY_TMPDIR"
