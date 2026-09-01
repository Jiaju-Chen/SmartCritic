#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

test -s "$VAL_FILE"
cd "$PROJECT_ROOT"
exec bash experiments/luna_unified_critic_searchqa_remote_8card_20260829/run_formal_luna_unified_8gpu.sh \
  "data.val_files=$VAL_FILE" \
  env.search.log_requests=false \
  algorithm.hybrid_advantage.composition_mode=direct \
  algorithm.hybrid_advantage.whiten_advantages=False \
  "$@"
