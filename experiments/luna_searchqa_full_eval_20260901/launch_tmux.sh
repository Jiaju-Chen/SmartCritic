#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
SESSION_NAME=${SESSION_NAME:-searchqa_full51713_both_0901}
EVAL_ROOT=${EVAL_ROOT:-/home/dataset-local/cjj/RL/runs/luna_searchqa_full_eval_20260901}

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$EVAL_ROOT/logs"
tmux new-session -d -s "$SESSION_NAME" \
  "cd '$PROJECT_ROOT' && bash experiments/luna_searchqa_full_eval_20260901/run_both_sequential.sh 2>&1 | tee '$EVAL_ROOT/logs/driver.log'"

echo "started tmux session: $SESSION_NAME"
echo "driver log: $EVAL_ROOT/logs/driver.log"
