#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

mkdir -p "$RUN_DIR/logs"
if [[ -e "$RUN_DIR/logs/train.tmux.log" ]]; then
  printf 'Refusing to overwrite an existing training log: %s\n' "$RUN_DIR/logs/train.tmux.log" >&2
  exit 1
fi
date --iso-8601=seconds >"$RUN_DIR/logs/started_at.txt"
set +e
bash "$EXPERIMENT_DIR/run_train.sh" 2>&1 | tee "$RUN_DIR/logs/train.tmux.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" >"$RUN_DIR/logs/exit.status"
date --iso-8601=seconds >"$RUN_DIR/logs/finished_at.txt"
exit "$status"
