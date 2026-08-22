#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF_LUNA_WEBSHOP_ABLATIONS}
SESSION=${SESSION:-luna_webshop_advantage_ablation_20260822}
SEQUENCE_LOG=${SEQUENCE_LOG:-/home/dataset-local/cjj/RL/runs/luna_webshop_advantage_ablation/sequence_20260822.log}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$(dirname "$SEQUENCE_LOG")"
printf -v command \
  'bash experiments/luna_webshop_advantage_ablation_20260822/sequence_entry.sh 2>&1 | tee -a %q' \
  "$SEQUENCE_LOG"

tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" "$command"
echo "started tmux session $SESSION"
echo "sequence log: $SEQUENCE_LOG"
