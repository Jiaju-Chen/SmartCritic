#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic_ALFWORLD_DIRECT}
WAIT_SESSION=${WAIT_SESSION:-luna_webshop_advantage_ablation_20260822}
QUEUE_SESSION=${QUEUE_SESSION:-queue_luna_alfworld_directmix_20260823}
QUEUE_LOG=${QUEUE_LOG:-/home/dataset-local/cjj/RL/runs/luna_alfworld_advantage_ablation/queue_20260823.log}

mkdir -p "$(dirname "$QUEUE_LOG")"
if tmux has-session -t "$QUEUE_SESSION" 2>/dev/null; then
  echo "tmux queue session already exists: $QUEUE_SESSION" >&2
  exit 1
fi

printf -v queue_command \
  'while tmux has-session -t %q 2>/dev/null; do printf "[%%s] waiting for %s\n" "$(date)" | tee -a %q; sleep 60; done; printf "[%%s] launching ALFWorld direct mix\n" "$(date)" | tee -a %q; sleep 30; bash experiments/luna_alfworld_advantage_ablation_20260823/launch_full150.sh 2>&1 | tee -a %q' \
  "$WAIT_SESSION" "$WAIT_SESSION" "$QUEUE_LOG" "$QUEUE_LOG" "$QUEUE_LOG"

tmux new-session -d -s "$QUEUE_SESSION" -c "$PROJECT_ROOT" "$queue_command"
echo "queue_session=$QUEUE_SESSION"
echo "waiting_for=$WAIT_SESSION"
echo "queue_log=$QUEUE_LOG"
