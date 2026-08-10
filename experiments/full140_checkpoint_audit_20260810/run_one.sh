#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {critic2l|sao_skipobs} {25|50|75|100|125|150}" >&2
  exit 2
fi

METHOD=$1
STEP=$2

case "${STEP}" in
  25|50|75|100|125|150) ;;
  *) echo "unsupported checkpoint step: ${STEP}" >&2; exit 2 ;;
esac

case "${METHOD}" in
  critic2l)
    CHECKPOINT_ROOT=/home/dataset-local/cjj/RL/checkpoints/truncated_qwen_critic_alfworld/ppo_qwen25_15b_critic2l_t128_v32_8gpu_seed0_20260731
    EXTRA_OVERRIDES=()
    ;;
  sao_skipobs)
    CHECKPOINT_ROOT=/home/dataset-local/cjj/RL/checkpoints/truncated_qwen_critic_alfworld/ppo_qwen25_15b_sao_skipobs_t128_v32_8gpu_seed0_20260804
    EXTRA_OVERRIDES=(
      algorithm.adv_estimator=sao_skip_observation
      algorithm.gamma=1.0
      algorithm.lam=1.0
      reward_model.use_step_rewards=True
    )
    ;;
  *) echo "unsupported method: ${METHOD}" >&2; exit 2 ;;
esac

source /opt/conda/bin/activate /home/dataset-local/cjj/RL/envs/gigpo-baselines
cd /home/dataset-local/cjj/RL/GiGPO_PVF

VAL_DATA_SIZE=${VAL_DATA_SIZE:-140}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-20}
EVAL_SEED=${EVAL_SEED:-0}
RUN_TAG=${RUN_TAG:-20260810}
AUDIT_ROOT=${AUDIT_ROOT:-/home/dataset-local/cjj/RL/runs/full140_checkpoint_audit_20260810}
RUN_NAME=${METHOD}_step${STEP}_seen${VAL_DATA_SIZE}_seed${EVAL_SEED}_${RUN_TAG}
RUN_DIR=${AUDIT_ROOT}/${RUN_NAME}
LOG_PATH=${RUN_DIR}/logs/eval.log
STATUS_PATH=${RUN_DIR}/status.txt
CHECKPOINT_PATH=${CHECKPOINT_ROOT}/global_step_${STEP}
MODEL_PATH=/home/dataset-local/cjj/RL/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
if [[ ${METHOD} == critic2l ]]; then
  SHORT_NAME=c${STEP}v${VAL_DATA_SIZE}s${EVAL_SEED}
else
  SHORT_NAME=s${STEP}v${VAL_DATA_SIZE}s${EVAL_SEED}
fi

if [[ ! -d "${CHECKPOINT_PATH}/actor" ]]; then
  echo "missing actor checkpoint: ${CHECKPOINT_PATH}/actor" >&2
  exit 3
fi

if [[ -f "${STATUS_PATH}" ]] && grep -qx 'status=0' "${STATUS_PATH}" && \
   grep -q "val/evaluated_cases:${VAL_DATA_SIZE}.000" "${LOG_PATH}" 2>/dev/null; then
  echo "already complete: ${RUN_NAME}"
  exit 0
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/home" \
  "/home/dataset-local/cjj/t/${SHORT_NAME}" \
  "/home/dataset-local/cjj/r/${SHORT_NAME}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HOME=${RUN_DIR}/home
export TMPDIR=/home/dataset-local/cjj/t/${SHORT_NAME}
export TEMP=${TMPDIR}
export TMP=${TMPDIR}
export RAY_TMPDIR=/home/dataset-local/cjj/r/${SHORT_NAME}
export ALFWORLD_DATA=/home/dataset-local/cjj/RL/alfworld_data
export HF_HOME=/home/dataset-local/cjj/RL/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1
export NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.5}
export TRAIN_DATA_SIZE=8
export VAL_DATA_SIZE
export GROUP_SIZE=1

printf '[%s] start method=%s checkpoint=%s val=%s batch=%s seed=%s\n' \
  "$(date)" "${METHOD}" "${STEP}" "${VAL_DATA_SIZE}" "${VAL_BATCH_SIZE}" "${EVAL_SEED}" | tee -a "${LOG_PATH}"

set +e
bash examples/ppo_trainer/run_alfworld.sh vllm \
  "trainer.logger=['console']" \
  trainer.project_name=verl_agent_alfworld_critic_ablation \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node=8 \
  trainer.total_epochs=150 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.monitor_validation_size=32 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${CHECKPOINT_PATH}" \
  trainer.default_local_dir="${RUN_DIR}/dummy_ckpt" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  critic.model.path="${MODEL_PATH}" \
  +critic.model.override_config.num_hidden_layers=2 \
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
  env.seed="${EVAL_SEED}" \
  env.max_steps=50 \
  env.alfworld.eval_dataset=eval_in_distribution \
  env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER}" \
  ray_init.num_cpus=96 \
  +ray_init._temp_dir="${RAY_TMPDIR}" \
  "${EXTRA_OVERRIDES[@]}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e

printf 'status=%s\n' "${status}" > "${STATUS_PATH}"
printf '[%s] finish method=%s checkpoint=%s status=%s\n' \
  "$(date)" "${METHOD}" "${STEP}" "${status}" | tee -a "${LOG_PATH}"

if [[ ${status} -ne 0 ]]; then
  exit "${status}"
fi

if ! grep -q "val/evaluated_cases:${VAL_DATA_SIZE}.000" "${LOG_PATH}"; then
  echo "evaluation exited successfully but did not report ${VAL_DATA_SIZE} cases" >&2
  exit 4
fi
