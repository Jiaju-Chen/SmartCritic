#!/usr/bin/env bash
set -o pipefail

RUN_NAME=${RUN_NAME:-ppo_qwen25_15b_luna_unified2h_step150_unseen134_v2_20260821}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_unseen_alfworld/$RUN_NAME}
LOG_PATH="$RUN_ROOT/logs/eval.tmux.log"

mkdir -p "$RUN_ROOT/logs"
printf '[%s] starting Luna unified step-150 unseen-134 evaluation\n' "$(date)" | tee -a "$LOG_PATH"
bash /home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_UNIFIED/experiments/luna_unified_critic_alfworld_20260816/run_eval_unseen134_step150.sh 2>&1 | tee -a "$LOG_PATH"
status=${PIPESTATUS[0]}
printf '[%s] finished status=%s\n' "$(date)" "$status" | tee -a "$LOG_PATH"
exit "$status"
