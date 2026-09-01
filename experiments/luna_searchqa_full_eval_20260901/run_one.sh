#!/usr/bin/env bash
set -euo pipefail

VARIANT=${1:?usage: run_one.sh <no_whiten|whiten>}

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
DATA_ROOT=${DATA_ROOT:-/home/dataset-local/cjj/RL/data/searchR1_official/processed}
EVAL_ROOT=${EVAL_ROOT:-/home/dataset-local/cjj/RL/runs/luna_searchqa_full_eval_20260901}
SEARCH_URL=${SEARCH_URL:-http://127.0.0.1:18003/retrieve}
EXPECTED_CASES=${EXPECTED_CASES:-51713}

case "$VARIANT" in
  no_whiten)
    TRAIN_RUN_NAME=luna_unified_searchqa_2l_residual_8gpu_t256_v512_200step_20260830
    EVAL_RUN_NAME=luna_searchqa_2l_no_whiten_step200_full51713_20260901
    WANDB_ID=lunafullnw0901
    WHITEN_ADVANTAGES=False
    ;;
  whiten)
    TRAIN_RUN_NAME=luna_unified_searchqa_2l_residual_whiten_8gpu_t256_v512_200step_20260830
    EVAL_RUN_NAME=luna_searchqa_2l_whiten_step200_full51713_20260901
    WANDB_ID=lunafullw0901
    WHITEN_ADVANTAGES=True
    ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

CHECKPOINT_ROOT=/home/dataset-local/cjj/RL/checkpoints/luna_unified_searchqa/$TRAIN_RUN_NAME
CHECKPOINT_SLOT=$CHECKPOINT_ROOT/latest
RUN_DIR=$EVAL_ROOT/$EVAL_RUN_NAME
RESUME_PATH=$RUN_DIR/resume/global_step_200
FULL_TEST_FILE=$DATA_ROOT/test.parquet

test -s "$FULL_TEST_FILE"
test -d "$CHECKPOINT_SLOT/actor"
test -d "$CHECKPOINT_SLOT/critic"
test "$(tr -d '[:space:]' < "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt")" = 200

ACTUAL_CASES=$(
  /home/dataset-local/conda/envs/verl-agent-webshop/bin/python - "$FULL_TEST_FILE" <<'PY'
import sys
import pyarrow.parquet as pq

print(pq.read_metadata(sys.argv[1]).num_rows)
PY
)
if [[ "$ACTUAL_CASES" != "$EXPECTED_CASES" ]]; then
  echo "expected $EXPECTED_CASES test cases, found $ACTUAL_CASES" >&2
  exit 3
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
curl --fail --silent --show-error \
  --connect-timeout 5 --max-time 30 \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who wrote Pride and Prejudice?","topk":3,"return_scores":false}' \
  "$SEARCH_URL" >/dev/null

mkdir -p "$RUN_DIR"/{home,logs,resume,wandb,dummy_checkpoint_dir}
if [[ ! -e "$RESUME_PATH" ]]; then
  ln -s "$CHECKPOINT_SLOT" "$RESUME_PATH"
fi

export RUN_NAME=$EVAL_RUN_NAME
export RUN_ROOT=$EVAL_ROOT
export RUN_DIR
export CKPT_DIR=$RUN_DIR/dummy_checkpoint_dir
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SEARCH_URL
export WANDB_ENTITY=cjj01-ustc
export WANDB_MODE=online
export WANDB_RUN_ID=$WANDB_ID
export WANDB_RESUME=allow
export RAY_TMP_ROOT=/dev/shm/${WANDB_ID}_ray
export FAST_TMP_ROOT=/dev/shm/${WANDB_ID}_tmp

echo "variant=$VARIANT"
echo "checkpoint=$CHECKPOINT_SLOT"
echo "checkpoint_step=200"
echo "test_file=$FULL_TEST_FILE"
echo "evaluated_cases=$ACTUAL_CASES"
echo "wandb_run_id=$WANDB_ID"

cd "$PROJECT_ROOT"
exec bash experiments/luna_unified_critic_searchqa_20260825/run_formal.sh \
  algorithm.hybrid_advantage.whiten_advantages="$WHITEN_ADVANTAGES" \
  data.val_files="$FULL_TEST_FILE" \
  data.val_batch_size=512 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  env.rollout.n=1 \
  +env.search.fail_on_error=true \
  "trainer.logger=['console','wandb']" \
  trainer.project_name=verl_agent_searchqa_full_eval \
  trainer.experiment_name="$EVAL_RUN_NAME" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$RESUME_PATH" \
  trainer.default_local_dir="$RUN_DIR/dummy_checkpoint_dir" \
  trainer.monitor_validation_size="$EXPECTED_CASES" \
  ray_init.num_cpus=64
