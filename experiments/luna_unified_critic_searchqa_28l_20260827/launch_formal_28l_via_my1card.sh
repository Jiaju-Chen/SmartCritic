#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
REMOTE_HOST=${REMOTE_HOST:-61.172.170.106}
REMOTE_PORT=${REMOTE_PORT:-30245}
REMOTE_USER=${REMOTE_USER:-batchcom}
REMOTE_RETRIEVER_PORT=${REMOTE_RETRIEVER_PORT:-8000}
LOCAL_PORT=${LOCAL_PORT:-18001}
SSH_KEY=${SSH_KEY:-/home/batchcom/.ssh/id_ed25519}
RUN_ROOT=${RUN_ROOT:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/formal_manager_28l_20260827}
RUN_SCRIPT=${RUN_SCRIPT:-experiments/luna_unified_critic_searchqa_28l_20260827/run_formal_28l.sh}
mkdir -p "$LOG_DIR"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

TUNNEL_PID=""
cleanup() {
  if [[ -n "$TUNNEL_PID" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ssh -N \
  -p "$REMOTE_PORT" \
  -i "$SSH_KEY" \
  -o BatchMode=yes \
  -o IdentitiesOnly=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=10 \
  -L "$LOCAL_PORT:127.0.0.1:$REMOTE_RETRIEVER_PORT" \
  "$REMOTE_USER@$REMOTE_HOST" \
  >"$LOG_DIR/tunnel.log" 2>&1 &
TUNNEL_PID=$!

for attempt in $(seq 1 180); do
  if curl --fail --silent --show-error --max-time 30 \
    -H 'Content-Type: application/json' \
    -d '{"query":"Who wrote Pride and Prejudice?","topk":3,"return_scores":false}' \
    "http://127.0.0.1:$LOCAL_PORT/retrieve" >"$LOG_DIR/preflight.json" 2>"$LOG_DIR/preflight.err"; then
    echo "[$(date)] official retriever is ready after attempt $attempt"
    cd "$PROJECT_ROOT"
    SEARCH_URL="http://127.0.0.1:$LOCAL_PORT/retrieve" \
      bash "$RUN_SCRIPT"
    exit 0
  fi
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "[$(date)] SSH tunnel exited before retriever became ready" >&2
    exit 1
  fi
  echo "[$(date)] waiting for official retriever, attempt $attempt/180"
  sleep 60
done

echo "Official retriever did not become ready within three hours" >&2
exit 1
