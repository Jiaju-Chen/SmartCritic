#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  printf 'Session already exists: %s\n' "$SESSION_NAME" >&2
  exit 1
fi
if [[ -e "$RUN_DIR/logs/train.tmux.log" ]]; then
  printf 'Run already has a training log: %s\n' "$RUN_DIR" >&2
  exit 1
fi
if [[ -d "$CKPT_DIR" ]] && [[ -n "$(find "$CKPT_DIR" -mindepth 1 -print -quit)" ]]; then
  printf 'Checkpoint directory is not empty: %s\n' "$CKPT_DIR" >&2
  exit 1
fi
active_gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$active_gpu_processes" ]]; then
  printf 'GPU compute processes are still present; no training started.\n%s\n' "$active_gpu_processes" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/logs"
stamp=$(date +%Y%m%d-%H%M%S)
config_file="$RUN_DIR/logs/preflight_config-$stamp.yaml"
error_file="$RUN_DIR/logs/preflight-$stamp.stderr.log"
if ! CONFIG_ONLY=1 bash "$EXPERIMENT_DIR/run_train.sh" --resolve >"$config_file" 2>"$error_file"; then
  cat "$error_file" >&2
  exit 1
fi
"$ENV_ROOT/bin/python" "$EXPERIMENT_DIR/verify_config.py" \
  "$config_file" | tee "$RUN_DIR/logs/config_comparison-$stamp.json"

printf -v command '%q ' env PROJECT_ROOT="$PROJECT_ROOT" ENV_ROOT="$ENV_ROOT" \
  RUN_NAME="$RUN_NAME" WANDB_RUN_ID="$WANDB_RUN_ID" SESSION_NAME="$SESSION_NAME" \
  bash "$EXPERIMENT_DIR/tmux_entry.sh"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_ROOT" "$command"
printf 'session=%s\nrun=%s\nlog=%s\n' "$SESSION_NAME" "$RUN_NAME" "$RUN_DIR/logs/train.tmux.log"
