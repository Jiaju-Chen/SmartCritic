#!/usr/bin/env bash
set -euo pipefail

mode=${1:-formal}
if [[ $# -gt 0 ]]; then shift; fi
export PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export WANDB_ENTITY=cjj01-ustc
export WANDB_MODE=online
export WANDB_RESUME=allow
export RUN_ROOT=/home/dataset-local/cjj/RL/runs/luna_webshop_turnend
export RAY_TMP_ROOT=/home/dataset-local/cjj/lwend_r
export FAST_TMP_ROOT=/home/dataset-local/cjj/lwend_t
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CRITIC_NUM_LAYERS=2
export TRAIN_DATA_SIZE=128 VAL_DATA_SIZE=128 VAL_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=64 TOTAL_EPOCHS=150 TEST_FREQ=5 SAVE_FREQ=5
export VAL_BEFORE_TRAIN=True
case "$mode" in
  formal)
    export RUN_NAME=luna_unified2h_webshop_actionend_t128_v128_8gpu_seed0_20260831
    export WANDB_RUN_ID=lunawsend0831
    ;;
  smoke)
    export RUN_NAME=luna_unified2h_webshop_actionend_smoke_20260831
    export WANDB_RUN_ID=lunawsendprobe0831
    export TRAIN_DATA_SIZE=16 VAL_DATA_SIZE=8 VAL_BATCH_SIZE=8
    export PPO_MINI_BATCH_SIZE=16 TOTAL_EPOCHS=1 TEST_FREQ=1 SAVE_FREQ=1
    export VAL_BEFORE_TRAIN=False
    ;;
  *) echo "Usage: $0 {formal|smoke} [Hydra overrides]" >&2; exit 2 ;;
esac
export RUN_DIR=$RUN_ROOT/$RUN_NAME
export CKPT_DIR=/home/dataset-local/cjj/RL/checkpoints/luna_webshop_turnend/$RUN_NAME
exec bash "$PROJECT_ROOT/experiments/luna_unified_critic_webshop_20260821/run_train.sh" \
  algorithm.hybrid_advantage.turn_value_position=action_end "$@"
