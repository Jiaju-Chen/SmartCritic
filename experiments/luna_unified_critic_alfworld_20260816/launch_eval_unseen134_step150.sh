#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-luna_unified_unseen134_step150_v2_20260821}
ENTRY=/home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_UNIFIED/experiments/luna_unified_critic_alfworld_20260816/eval_unseen134_step150_tmux_entry.sh

if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" "bash $ENTRY"
printf 'started tmux session: %s\n' "$SESSION"
