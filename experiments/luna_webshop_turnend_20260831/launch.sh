#!/usr/bin/env bash
set -euo pipefail
mode=${1:-formal}
case "$mode" in formal|smoke|smoke2) ;; *) exit 2 ;; esac
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
session=luna_webshop_turnend_${mode}_0831
logdir=/home/dataset-local/cjj/RL/runs/luna_webshop_turnend/launch_${mode}_20260831
if tmux has-session -t "$session" 2>/dev/null || [[ -e "$logdir/train.log" ]]; then
  echo "Refusing duplicate launch or log overwrite: $session $logdir" >&2
  exit 1
fi
mkdir -p "$logdir"
git -C "$root" rev-parse HEAD > "$logdir/source_commit.txt"
git -C "$root" diff --stat > "$logdir/source_diff.txt"
printf -v command 'bash %q %q %q' \
  "$root/experiments/luna_webshop_turnend_20260831/tmux_entry.sh" "$mode" "$logdir"
tmux new-session -d -s "$session" -c "$root" "$command"
echo "Started $session; log=$logdir/train.log"
