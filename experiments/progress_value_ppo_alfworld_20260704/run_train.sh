#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/bin/activate /home/dataset-local/cjj/RL/envs/gigpo-baselines
cd /home/dataset-local/cjj/RL/GiGPO_PVF

RUN_NAME=${RUN_NAME:-pvf_ppo_qwen25_15b_seed0_t16_g8_total128_val140_8gpu_envcpu02_ray96_20260704}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/progress_value_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
CKPT_DIR=${CKPT_DIR:-/home/dataset-local/cjj/RL/checkpoints/progress_value_alfworld/$RUN_NAME}
SNAP=${SNAP:-/home/dataset-local/cjj/RL/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}

SHORT_TMP_ROOT=${SHORT_TMP_ROOT:-/tmp/pvfppo}
mkdir -p "$RUN_DIR"/{home,logs,wandb} "$CKPT_DIR" "$SHORT_TMP_ROOT"/{tmp,ray}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOME=$RUN_DIR/home
export TMPDIR=$SHORT_TMP_ROOT/tmp
export RAY_TMPDIR=$SHORT_TMP_ROOT/ray
export ALFWORLD_DATA=${ALFWORLD_DATA:-/home/dataset-local/cjj/RL/alfworld_data}
export HF_HOME=${HF_HOME:-/home/dataset-local/cjj/RL/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$RUN_DIR/wandb
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.2}
export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-140}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-25}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-96}

bash examples/ppo_trainer/run_alfworld.sh vllm \
  algorithm.adv_estimator=progress_value \
  algorithm.progress_value.reward_scale=1.0 \
  algorithm.progress_value.length_penalty=0.02 \
  algorithm.progress_value.remaining_penalty=0.02 \
  algorithm.progress_value.baseline_mode=uid_step \
  algorithm.progress_value.min_group_size=2 \
  algorithm.progress_value.normalize_by_std=False \
  algorithm.progress_value.whiten=True \
  env.rollout.n=8 \
  "trainer.logger=['console','wandb']" \
  trainer.project_name=verl_agent_alfworld_progress_value \
  trainer.experiment_name=$RUN_NAME \
  trainer.default_local_dir=$CKPT_DIR \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.test_freq=$TEST_FREQ \
  trainer.save_freq=$SAVE_FREQ \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  actor_rollout_ref.model.path=$SNAP \
  critic.model.path=$SNAP \
  actor_rollout_ref.model.use_remove_padding=False \
  critic.model.use_remove_padding=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  critic.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_kl_loss=True \
  algorithm.use_kl_in_reward=False \
  env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
  ray_init.num_cpus=$RAY_NUM_CPUS
