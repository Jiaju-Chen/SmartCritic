#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-61.172.170.106}
REMOTE_PORT=${REMOTE_PORT:-30056}
REMOTE_USER=${REMOTE_USER:-batchcom}
REMOTE_RETRIEVER_HOST=${REMOTE_RETRIEVER_HOST:-10.233.145.15}
REMOTE_RETRIEVER_PORT=${REMOTE_RETRIEVER_PORT:-8000}
LOCAL_PORT=${LOCAL_PORT:-18003}
SSH_KEY=${SSH_KEY:-/home/batchcom/.ssh/id_ed25519}
LOG_DIR=${LOG_DIR:-/home/dataset-local/cjj/RL/runs/luna_unified_searchqa/remote_retriever_8card_20260829/tunnel}
PROBE_INTERVAL=${PROBE_INTERVAL:-60}
MAX_CONSECUTIVE_FAILURES=${MAX_CONSECUTIVE_FAILURES:-3}

mkdir -p "$LOG_DIR"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

child_pid=""
cleanup() {
  if [[ -n "$child_pid" ]]; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

probe() {
  curl --fail --silent --show-error \
    --connect-timeout 5 --max-time 15 \
    -H 'Content-Type: application/json' \
    -d '{"query":"retriever health probe","topk":1,"return_scores":false}' \
    "http://127.0.0.1:$LOCAL_PORT/retrieve" >/dev/null
}

while true; do
  tunnel_log="$LOG_DIR/tunnel-$(date +%Y%m%d-%H%M%S).log"
  echo "[$(date --iso-8601=seconds)] starting SSH forward" \
    >>"$LOG_DIR/tunnel-supervisor.log"
  ssh -N \
    -p "$REMOTE_PORT" \
    -i "$SSH_KEY" \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -L "127.0.0.1:$LOCAL_PORT:$REMOTE_RETRIEVER_HOST:$REMOTE_RETRIEVER_PORT" \
    "$REMOTE_USER@$REMOTE_HOST" \
    >"$tunnel_log" 2>&1 &
  child_pid=$!

  consecutive_failures=0
  sleep 3
  while kill -0 "$child_pid" 2>/dev/null; do
    if probe; then
      consecutive_failures=0
    else
      consecutive_failures=$((consecutive_failures + 1))
      echo "[$(date --iso-8601=seconds)] probe failure $consecutive_failures/$MAX_CONSECUTIVE_FAILURES" \
        >>"$LOG_DIR/tunnel-supervisor.log"
      if (( consecutive_failures >= MAX_CONSECUTIVE_FAILURES )); then
        echo "[$(date --iso-8601=seconds)] restarting unhealthy SSH forward" \
          >>"$LOG_DIR/tunnel-supervisor.log"
        kill "$child_pid" 2>/dev/null || true
        break
      fi
    fi
    sleep "$PROBE_INTERVAL"
  done

  wait "$child_pid" 2>/dev/null || true
  child_pid=""
  echo "[$(date --iso-8601=seconds)] SSH forward exited; retrying in 5 seconds" \
    >>"$LOG_DIR/tunnel-supervisor.log"
  sleep 5
done
