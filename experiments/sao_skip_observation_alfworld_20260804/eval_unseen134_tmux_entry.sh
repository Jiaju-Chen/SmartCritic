#!/usr/bin/env bash
set -o pipefail

RUN_NAME=ppo_qwen25_15b_sao_skipobs_step150_unseen134_20260806
RUN_ROOT=/home/dataset-local/cjj/RL/runs/truncated_qwen_critic_alfworld/${RUN_NAME}
LOG_PATH="${RUN_ROOT}/logs/eval.tmux.log"

mkdir -p "${RUN_ROOT}/logs"
printf '[%s] starting SAO skip-observation unseen134 evaluation\n' "$(date)" | tee -a "${LOG_PATH}"
bash /home/dataset-local/cjj/RL/GiGPO_PVF/experiments/sao_skip_observation_alfworld_20260804/run_eval_unseen134.sh 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
printf '[%s] finished status=%s\n' "$(date)" "${status}" | tee -a "${LOG_PATH}"
exit "${status}"
