#!/usr/bin/env bash
set -o pipefail

RUN_ROOT=/home/dataset-local/cjj/RL/runs/truncated_qwen_critic_alfworld/ppo_qwen25_15b_critic2l_step150_unseen134_20260802
LOG_PATH="${RUN_ROOT}/logs/eval.tmux.log"

mkdir -p "${RUN_ROOT}/logs"
printf '[%s] starting truncated critic unseen134 evaluation\n' "$(date)" | tee -a "${LOG_PATH}"
bash /home/dataset-local/cjj/RL/GiGPO_PVF/experiments/truncated_qwen_critic_alfworld_20260731/run_eval_unseen134.sh 2>&1 | tee -a "${LOG_PATH}"
status=${PIPESTATUS[0]}
printf '[%s] finished status=%s\n' "$(date)" "${status}" | tee -a "${LOG_PATH}"
exit "${status}"
