#!/usr/bin/env bash
set -Eeuo pipefail

# Run this script on a machine that can SSH to both servers. Data is streamed
# through that machine and is never written to its local disk.
SOURCE_HOST=${SOURCE_HOST:-yun-my8card-vscode}
DEST_HOST=${DEST_HOST:-SAIDS}
DEST_ROOT=${DEST_ROOT:-/data2/group_何向南/chenjiaju/luna/shared}

SOURCE_MODEL=${SOURCE_MODEL:-/home/dataset-local/cjj/RL/.cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
SOURCE_ALFWORLD=${SOURCE_ALFWORLD:-/home/dataset-local/cjj/RL/alfworld_data}
SOURCE_WEBSHOP=${SOURCE_WEBSHOP:-/home/dataset-local/xiax/verl-agent/agent_system/environments/env_package/webshop/webshop}

DEST_MODEL=$DEST_ROOT/models/Qwen2.5-1.5B-Instruct
DEST_ALFWORLD=$DEST_ROOT/datasets/alfworld_data
DEST_WEBSHOP=$DEST_ROOT/datasets/webshop

SSH_OPTIONS=(-o ServerAliveInterval=30 -o ServerAliveCountMax=6)

shell_quote() {
  printf '%q' "$1"
}

check_connection() {
  local host=$1
  echo "Checking SSH connection: $host"
  ssh "${SSH_OPTIONS[@]}" "$host" true
}

transfer_directory() {
  local label=$1
  local source=$2
  local destination=$3
  local exclude=${4:-}
  local source_q destination_q source_command destination_command

  source_q=$(shell_quote "$source")
  destination_q=$(shell_quote "$destination")

  source_command="test -d $source_q && tar --checkpoint=200000 --checkpoint-action=dot --dereference -C $source_q -cf - ."
  if [[ -n "$exclude" ]]; then
    source_command="test -d $source_q && tar --checkpoint=200000 --checkpoint-action=dot --dereference --exclude=$(shell_quote "$exclude") -C $source_q -cf - ."
  fi
  destination_command="mkdir -p $destination_q && tar -C $destination_q -xf -"

  echo "Transferring $label"
  echo "  source:      $SOURCE_HOST:$source"
  echo "  destination: $DEST_HOST:$destination"
  if command -v pv >/dev/null 2>&1; then
    ssh "${SSH_OPTIONS[@]}" "$SOURCE_HOST" "$source_command" \
      | pv -brt \
      | ssh "${SSH_OPTIONS[@]}" "$DEST_HOST" "$destination_command"
  else
    ssh "${SSH_OPTIONS[@]}" "$SOURCE_HOST" "$source_command" \
      | ssh "${SSH_OPTIONS[@]}" "$DEST_HOST" "$destination_command"
  fi
}

verify_model() {
  local source_file destination_file source_q destination_q source_hash destination_hash
  source_file=$SOURCE_MODEL/model.safetensors
  destination_file=$DEST_MODEL/model.safetensors
  source_q=$(shell_quote "$source_file")
  destination_q=$(shell_quote "$destination_file")
  source_hash=$(ssh "${SSH_OPTIONS[@]}" "$SOURCE_HOST" "sha256sum $source_q | awk '{print \$1}'")
  destination_hash=$(ssh "${SSH_OPTIONS[@]}" "$DEST_HOST" "sha256sum $destination_q | awk '{print \$1}'")
  if [[ "$source_hash" != "$destination_hash" ]]; then
    echo "Model checksum mismatch" >&2
    return 1
  fi
  echo "Model checksum verified: $source_hash"
}

verify_layout() {
  local root_q webshop_key_q
  root_q=$(shell_quote "$DEST_ROOT")
  webshop_key_q=$(shell_quote "$DEST_WEBSHOP/id_ed25519_xx")
  ssh "${SSH_OPTIONS[@]}" "$DEST_HOST" \
    "test -s $(shell_quote "$DEST_MODEL/config.json") && \
     test -d $(shell_quote "$DEST_ALFWORLD/json_2.1.1") && \
     test -d $(shell_quote "$DEST_WEBSHOP/search_engine") && \
     test ! -e $webshop_key_q && \
     du -sh $root_q/models $root_q/datasets"
  echo "Core asset layout verified; WebShop private key was not transferred."
}

check_connection "$SOURCE_HOST"
check_connection "$DEST_HOST"

transfer_directory "Qwen2.5-1.5B-Instruct" "$SOURCE_MODEL" "$DEST_MODEL"
verify_model
transfer_directory "ALFWorld data" "$SOURCE_ALFWORLD" "$DEST_ALFWORLD"
transfer_directory "WebShop resources" "$SOURCE_WEBSHOP" "$DEST_WEBSHOP" "./id_ed25519_xx"
verify_layout

echo "Core SmartCritic assets are available under: $DEST_ROOT"
