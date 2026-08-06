#!/usr/bin/env bash
set -euo pipefail

SESSION=sao_skipobs_unseen134_20260806
ENTRY=/home/dataset-local/cjj/RL/GiGPO_PVF/experiments/sao_skip_observation_alfworld_20260804/eval_unseen134_tmux_entry.sh

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "${SESSION}"
  exit 1
fi

tmux new-session -d -s "${SESSION}" "bash ${ENTRY}"
printf 'started tmux session: %s\n' "${SESSION}"
