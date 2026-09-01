#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-webshop-turnend}
SESSION=${SESSION:-webshop_eval500_actionend_0901}
OUT_ROOT=${OUT_ROOT:-/home/dataset-local/cjj/RL/evals/webshop_actionend_500_20260901}
ENTRY=$PROJECT_ROOT/experiments/luna_webshop_turnend_20260831/eval500_latest.sh

mkdir -p "$OUT_ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 2
fi

if [[ -s "$OUT_ROOT/summary.txt" ]] && grep -q '^status=finished$' "$OUT_ROOT/summary.txt"; then
  echo "Evaluation already finished: $OUT_ROOT/summary.txt"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "bash -lc 'set -o pipefail; bash \"$ENTRY\" 2>&1 | tee -a \"$OUT_ROOT/launcher.log\"'"

echo "session=$SESSION"
echo "log=$OUT_ROOT/eval.log"
echo "summary=$OUT_ROOT/summary.txt"

