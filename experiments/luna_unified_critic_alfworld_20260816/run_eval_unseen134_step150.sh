#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/bin/activate /home/dataset-local/cjj/RL/envs/gigpo-baselines
cd /home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_UNIFIED

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2h_step150_unseen134_v2_20260821}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_unseen_alfworld/$RUN_NAME}
MODEL_PATH=${MODEL_PATH:-/home/dataset-local/cjj/RL/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
CHECKPOINT_SLOT=${CHECKPOINT_SLOT:-/home/dataset-local/cjj/RL/checkpoints/luna_unified_alfworld/ppo_qwen25_15b_luna_unified2h_t128_v140_vb20_8gpu_seed0_full150_20260820/best}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$RUN_ROOT/resume/global_step_150}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOME="$RUN_ROOT/home"
export TMPDIR=${TMPDIR:-/home/dataset-local/cjj/lunau_e_t}
export TEMP=$TMPDIR
export TMP=$TMPDIR
export RAY_TMPDIR=${RAY_TMPDIR:-/home/dataset-local/cjj/lunau_e_r}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/home/dataset-local/cjj/RL/alfworld_data}
export HF_HOME=${HF_HOME:-/home/dataset-local/cjj/RL/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export WANDB_MODE=offline
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.5}
export TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-8}
export VAL_DATA_SIZE=${VAL_DATA_SIZE:-134}

VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-20}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-96}

mkdir -p "$RUN_ROOT"/{home,logs,resume} "$TMPDIR" "$RAY_TMPDIR"
if [[ ! -e "$CHECKPOINT_PATH" ]]; then
  ln -s "$CHECKPOINT_SLOT" "$CHECKPOINT_PATH"
fi

bash examples/ppo_trainer/run_alfworld.sh vllm \
  algorithm.adv_estimator=luna_unified \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.hybrid_advantage.turn_gamma=1.0 \
  algorithm.hybrid_advantage.turn_lam=0.95 \
  algorithm.hybrid_advantage.token_residual_scale=1.0 \
  algorithm.hybrid_advantage.whiten_advantages=True \
  reward_model.use_step_rewards=True \
  data.val_batch_size=$VAL_BATCH_SIZE \
  env.rollout.n=1 \
  +critic.model.override_config.num_hidden_layers=2 \
  critic.model.num_value_heads=2 \
  critic.unified_luna_turn_loss_coef=1.0 \
  "trainer.logger=['console']" \
  trainer.project_name=verl_agent_alfworld_critic_ablation \
  trainer.experiment_name="$RUN_NAME" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_epochs=150 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CHECKPOINT_PATH" \
  trainer.default_local_dir="$RUN_ROOT/dummy_ckpt_dir" \
  trainer.monitor_validation_size=32 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
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
  env.seed=0 \
  env.max_steps=50 \
  env.alfworld.eval_dataset=eval_out_of_distribution \
  env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
  ray_init.num_cpus=$RAY_NUM_CPUS \
  +ray_init._temp_dir=$RAY_TMPDIR
