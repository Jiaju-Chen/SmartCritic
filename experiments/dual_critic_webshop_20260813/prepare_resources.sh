#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/GiGPO_PVF_WebShop}
RESOURCE_ROOT=${WEBSHOP_RESOURCE_ROOT:-/home/dataset-local/xiax/verl-agent/agent_system/environments/env_package/webshop/webshop}
LOCAL_ROOT=$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "missing WebShop resource: $1" >&2
    exit 1
  fi
}

link_resource() {
  local source_path=$1
  local target_path=$2

  require_path "$source_path"
  if [[ -L "$target_path" ]]; then
    if [[ "$(readlink -f "$target_path")" != "$(readlink -f "$source_path")" ]]; then
      echo "existing symbolic link points elsewhere: $target_path" >&2
      exit 1
    fi
    return
  fi
  if [[ -e "$target_path" ]]; then
    echo "refusing to replace existing path: $target_path" >&2
    exit 1
  fi
  ln -s "$source_path" "$target_path"
}

mkdir -p "$LOCAL_ROOT/search_engine"
link_resource "$RESOURCE_ROOT/data" "$LOCAL_ROOT/data"

for resource in indexes indexes_100 indexes_1k indexes_100k; do
  link_resource "$RESOURCE_ROOT/search_engine/$resource" "$LOCAL_ROOT/search_engine/$resource"
done

echo "WebShop resources ready under $LOCAL_ROOT"
