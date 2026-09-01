#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
RESOURCE_ROOT=${WEBSHOP_RESOURCE_ROOT:?WEBSHOP_RESOURCE_ROOT is required}
LOCAL_ROOT=$PROJECT_ROOT/agent_system/environments/env_package/webshop/webshop

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing WebShop resource file: $1" >&2
    exit 2
  fi
}

link_resource() {
  local source_path=$1
  local target_path=$2

  if [[ ! -e "$source_path" ]]; then
    echo "Missing WebShop resource: $source_path" >&2
    exit 2
  fi
  if [[ -L "$target_path" ]]; then
    if [[ "$(readlink -f "$target_path")" != "$(readlink -f "$source_path")" ]]; then
      echo "Existing symbolic link points elsewhere: $target_path" >&2
      exit 2
    fi
    return
  fi
  if [[ -e "$target_path" ]]; then
    echo "Refusing to replace existing path: $target_path" >&2
    exit 2
  fi
  ln -s "$source_path" "$target_path"
}

require_file "$RESOURCE_ROOT/data/items_shuffle_1000.json"
require_file "$RESOURCE_ROOT/data/items_ins_v2_1000.json"
require_file "$RESOURCE_ROOT/data/items_human_ins.json"
mkdir -p "$LOCAL_ROOT/search_engine"
link_resource "$RESOURCE_ROOT/data" "$LOCAL_ROOT/data"

# The current WebShop builder passes num_products=None even with use_small=True,
# so the environment opens `indexes`.  Keep all small aliases available to
# make the resource closure explicit and portable across nearby code versions.
for resource in indexes indexes_100 indexes_1k indexes_100k; do
  link_resource "$RESOURCE_ROOT/search_engine/$resource" "$LOCAL_ROOT/search_engine/$resource"
done

echo "WebShop resources ready under $LOCAL_ROOT"
