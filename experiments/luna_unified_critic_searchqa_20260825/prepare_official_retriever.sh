#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
ASSET_ROOT=${ASSET_ROOT:-/home/dataset-local/cjj/RL/data/searchR1_official_retriever}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

INDEX_URL="$HF_ENDPOINT/datasets/PeterJinGo/wiki-18-e5-index/resolve/main"
CORPUS_URL="$HF_ENDPOINT/datasets/PeterJinGo/wiki-18-corpus/resolve/main"
mkdir -p "$ASSET_ROOT/index" "$ASSET_ROOT/corpus" "$ASSET_ROOT/models"

download_exact() {
  local url=$1
  local target=$2
  local expected_size=$3
  if [[ -f "$target" ]] && [[ $(stat -c %s "$target") == "$expected_size" ]]; then
    echo "Using complete asset: $target ($expected_size bytes)"
    return
  fi
  curl --fail --location --retry 30 --retry-delay 10 --continue-at - \
    --output "$target" "$url"
  [[ $(stat -c %s "$target") == "$expected_size" ]]
}

download_exact "$INDEX_URL/part_aa?download=true" \
  "$ASSET_ROOT/index/part_aa" 42949672960
download_exact "$INDEX_URL/part_ab?download=true" \
  "$ASSET_ROOT/index/part_ab" 21609402413
download_exact "$CORPUS_URL/wiki-18.jsonl.gz?download=true" \
  "$ASSET_ROOT/corpus/wiki-18.jsonl.gz" 5123307260

HF_ENDPOINT="$HF_ENDPOINT" HF_HOME="$ASSET_ROOT/hf-cache" \
  MODEL_DIR="$ASSET_ROOT/models/e5-base-v2" "$ENV_ROOT/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="intfloat/e5-base-v2",
    local_dir=os.environ["MODEL_DIR"],
)
PY

ASSET_ROOT="$ASSET_ROOT" "$ENV_ROOT/bin/python" - <<'PY'
import gzip
import os
from pathlib import Path

root = Path(os.environ["ASSET_ROOT"])
index = root / "index/e5_Flat.index"
parts = [root / "index/part_aa", root / "index/part_ab"]
expected = sum(part.stat().st_size for part in parts)
if not index.exists() or index.stat().st_size != expected:
    pending = index.with_suffix(".index.incomplete")
    with pending.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(16 * 1024 * 1024):
                    output.write(chunk)
    pending.replace(index)

compressed = root / "corpus/wiki-18.jsonl.gz"
corpus = root / "corpus/wiki-18.jsonl"
if not corpus.exists():
    pending = corpus.with_suffix(".jsonl.incomplete")
    with gzip.open(compressed, "rb") as source, pending.open("wb") as output:
        while chunk := source.read(16 * 1024 * 1024):
            output.write(chunk)
    pending.replace(corpus)

print(f"index={index} bytes={index.stat().st_size}")
print(f"corpus={corpus} bytes={corpus.stat().st_size}")
PY
