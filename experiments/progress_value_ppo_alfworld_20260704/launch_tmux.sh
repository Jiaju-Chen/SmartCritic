#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-pvf_ppo_qwen25_15b_seed0_t16_g8_total128_val32_8gpu_envcpu05_ray96_fdtmp2_20260705}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/progress_value_alfworld}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$RUN_NAME}
SESSION=${SESSION:-pvf_ppo_val32_8gpu_fdtmp2_20260705}

mkdir -p "$RUN_DIR"/logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "attach: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /home/dataset-local/cjj/RL/GiGPO_PVF && RUN_NAME='$RUN_NAME' RUN_DIR='$RUN_DIR' bash experiments/progress_value_ppo_alfworld_20260704/tmux_entry.sh 2>&1 | tee -a '$RUN_DIR/logs/train.tmux.log'"

echo "session=$SESSION"
echo "run_name=$RUN_NAME"
echo "run_dir=$RUN_DIR"
echo "log=$RUN_DIR/logs/train.tmux.log"
echo "ckpt=/home/dataset-local/cjj/RL/checkpoints/progress_value_alfworld/$RUN_NAME"
echo "attach: tmux attach -t $SESSION"
