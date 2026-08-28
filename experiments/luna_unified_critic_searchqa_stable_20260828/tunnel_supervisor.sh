#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=${REMOTE_HOST:-61.172.170.106}
REMOTE_PORT=${REMOTE_PORT:-30245}
REMOTE_USER=${REMOTE_USER:-batchcom}
REMOTE_RETRIEVER_PORT=${REMOTE_RETRIEVER_PORT:-8000}
LOCAL_PORT=${LOCAL_PORT:-18000}
SSH_KEY=${SSH_KEY:-/home/batchcom/.ssh/id_ed25519}
LOG_DIR=${LOG_DIR:?LOG_DIR must be set}
mkdir -p "$LOG_DIR"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

child_pid=""
probe() {
  # Probe only the HTTP forwarding path.  A retrieval query is expensive under
  # rollout load and can falsely trigger a reconnect while the service is busy.
  curl --silent --show-error --max-time 5 \
    "http://127.0.0.1:$LOCAL_PORT/" >/dev/null 2>&1
}
cleanup() {
  if [[ -n "$child_pid" ]]; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

while [[ -z "${TRAIN_PID:-}" ]] || kill -0 "$TRAIN_PID" 2>/dev/null; do
  tunnel_log="$LOG_DIR/tunnel-$(date +%Y%m%d-%H%M%S).log"
  ssh -N \
    -p "$REMOTE_PORT" \
    -i "$SSH_KEY" \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "$LOCAL_PORT:127.0.0.1:$REMOTE_RETRIEVER_PORT" \
    "$REMOTE_USER@$REMOTE_HOST" \
    >"$tunnel_log" 2>&1 &
  child_pid=$!
  sleep 5
  while kill -0 "$child_pid" 2>/dev/null; do
    if ! probe; then
      echo "[$(date)] retriever probe failed; restarting SSH forward" >>"$LOG_DIR/tunnel-supervisor.log"
      kill "$child_pid" 2>/dev/null || true
      break
    fi
    sleep 15
  done
  wait "$child_pid" 2>/dev/null || true
  child_pid=""
  if [[ -n "${TRAIN_PID:-}" ]] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    break
  fi
  echo "[$(date)] SSH forward exited; reconnecting in 5 seconds" >>"$LOG_DIR/tunnel-supervisor.log"
  sleep 5
done
