#!/usr/bin/env bash
set -uo pipefail
mode=$1
logdir=$2
date -Is > "$logdir/started_at.txt"
bash experiments/luna_webshop_turnend_20260831/run.sh "$mode" 2>&1 | tee -a "$logdir/train.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$logdir/exit.status"
date -Is > "$logdir/finished_at.txt"
exit "$status"
