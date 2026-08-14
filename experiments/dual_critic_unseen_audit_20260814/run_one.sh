#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {dual2|dual28} {best|latest}" >&2
  exit 2
fi

METHOD=$1
SLOT=$2

case "${METHOD}" in
  dual2)
    CHECKPOINT_ROOT=/home/dataset-local/cjj/RL/checkpoints/dual_critic_hybrid_alfworld/ppo_qwen25_15b_dualcritic_residual_t128_v140_vb20_8gpu_seed0_v2_20260808
    ;;
  dual28)
    CHECKPOINT_ROOT=/home/dataset-local/cjj/RL/checkpoints/dual_critic_full_alfworld/ppo_qwen25_15b_dualcritic_full28_residual_t128_v140_vb20_8gpu_seed0_20260810
    ;;
  *)
    echo "unsupported method: ${METHOD}" >&2
    exit 2
    ;;
esac

case "${SLOT}" in
  best) TRACKER=best_checkpointed_iteration.txt ;;
  latest) TRACKER=latest_checkpointed_iteration.txt ;;
  *)
    echo "unsupported checkpoint slot: ${SLOT}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/cjj/RL/envs/gigpo-baselines}
AUDIT_ROOT=${AUDIT_ROOT:-/home/dataset-local/cjj/RL/runs/dual_critic_unseen_audit_20260814}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-134}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-20}
EVAL_SEED=${EVAL_SEED:-0}
RUN_TAG=${RUN_TAG:-20260814}
MODEL_PATH=${MODEL_PATH:-/home/dataset-local/cjj/RL/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}

if [[ ! -f "${CHECKPOINT_ROOT}/${TRACKER}" ]]; then
  echo "missing checkpoint tracker: ${CHECKPOINT_ROOT}/${TRACKER}" >&2
  exit 3
fi

STEP=$(tr -d '[:space:]' < "${CHECKPOINT_ROOT}/${TRACKER}")
if [[ ! ${STEP} =~ ^[0-9]+$ ]]; then
  echo "invalid checkpoint step in ${TRACKER}: ${STEP}" >&2
  exit 3
fi

CHECKPOINT_PATH=${CHECKPOINT_ROOT}/${SLOT}
if [[ ! -d "${CHECKPOINT_PATH}/actor" ]]; then
  echo "missing actor checkpoint: ${CHECKPOINT_PATH}/actor" >&2
  exit 3
fi

RUN_NAME=${METHOD}_${SLOT}_step${STEP}_unseen${VAL_DATA_SIZE}_seed${EVAL_SEED}_${RUN_TAG}
RUN_DIR=${AUDIT_ROOT}/${RUN_NAME}
LOG_PATH=${RUN_DIR}/logs/eval.log
STATUS_PATH=${RUN_DIR}/status.txt
SHORT_NAME=${METHOD:0:3}${SLOT:0:1}s${STEP}u${VAL_DATA_SIZE}e${EVAL_SEED}
ALIAS_DIR=${RUN_DIR}/checkpoint_links
CHECKPOINT_ALIAS=${ALIAS_DIR}/global_step_${STEP}

if [[ -f "${STATUS_PATH}" ]] && grep -qx 'status=0' "${STATUS_PATH}" && \
   grep -q "val/evaluated_cases:${VAL_DATA_SIZE}.000" "${LOG_PATH}" 2>/dev/null; then
  echo "already complete: ${RUN_NAME}"
  exit 0
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/home" "${ALIAS_DIR}" \
  "/home/dataset-local/cjj/t/${SHORT_NAME}" \
  "/home/dataset-local/cjj/r/${SHORT_NAME}"

if [[ -L "${CHECKPOINT_ALIAS}" ]]; then
  if [[ $(readlink -f "${CHECKPOINT_ALIAS}") != $(readlink -f "${CHECKPOINT_PATH}") ]]; then
    echo "checkpoint alias points to an unexpected target: ${CHECKPOINT_ALIAS}" >&2
    exit 3
  fi
elif [[ -e "${CHECKPOINT_ALIAS}" ]]; then
  echo "checkpoint alias path already exists and is not a symbolic link: ${CHECKPOINT_ALIAS}" >&2
  exit 3
else
  ln -s "${CHECKPOINT_PATH}" "${CHECKPOINT_ALIAS}"
fi

set +u
source /opt/conda/bin/activate "${ENV_ROOT}"
set -u
cd "${PROJECT_ROOT}"

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
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

printf '[%s] start method=%s slot=%s checkpoint_step=%s split=unseen cases=%s batch=%s seed=%s\n' \
  "$(date)" "${METHOD}" "${SLOT}" "${STEP}" "${VAL_DATA_SIZE}" \
  "${VAL_BATCH_SIZE}" "${EVAL_SEED}" | tee -a "${LOG_PATH}"

set +e
bash examples/ppo_trainer/run_alfworld.sh vllm \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  "trainer.logger=['console']" \
  trainer.project_name=verl_agent_alfworld_critic_ablation \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_epochs=150 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.checkpoint_slot_mode=legacy \
  trainer.monitor_validation_size=32 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${CHECKPOINT_ALIAS}" \
  trainer.default_local_dir="${RUN_DIR}/dummy_ckpt" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  env.seed="${EVAL_SEED}" \
  env.max_steps=50 \
  env.alfworld.eval_dataset=eval_out_of_distribution \
  env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER}" \
  ray_init.num_cpus=96 \
  +ray_init._temp_dir="${RAY_TMPDIR}" 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
set -e

if [[ ${status} -ne 0 ]]; then
  printf 'status=%s\n' "${status}" > "${STATUS_PATH}"
  printf '[%s] finish method=%s slot=%s checkpoint_step=%s status=%s\n' \
    "$(date)" "${METHOD}" "${SLOT}" "${STEP}" "${status}" | tee -a "${LOG_PATH}"
  exit "${status}"
fi

if ! grep -q "val/evaluated_cases:${VAL_DATA_SIZE}.000" "${LOG_PATH}"; then
  printf 'status=4\n' > "${STATUS_PATH}"
  echo "evaluation exited successfully but did not report ${VAL_DATA_SIZE} cases" | tee -a "${LOG_PATH}" >&2
  exit 4
fi

printf 'status=0\n' > "${STATUS_PATH}"
printf '[%s] finish method=%s slot=%s checkpoint_step=%s status=0\n' \
  "$(date)" "${METHOD}" "${SLOT}" "${STEP}" | tee -a "${LOG_PATH}"
