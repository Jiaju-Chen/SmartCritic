#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/dataset-local/cjj/RL/SmartCritic-searchqa}
ENV_ROOT=${ENV_ROOT:-/home/dataset-local/conda/envs/verl-agent-webshop}
ASSET_ROOT=${ASSET_ROOT:-/home/dataset-local/cjj/RL/data/searchR1_official_retriever}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

INDEX_URL="$HF_ENDPOINT/datasets/PeterJinGo/wiki-18-e5-index/resolve/main"
CORPUS_URL="$HF_ENDPOINT/datasets/PeterJinGo/wiki-18-corpus/resolve/main"
mkdir -p "$ASSET_ROOT/index" "$ASSET_ROOT/corpus" "$ASSET_ROOT/models"

for part in part_aa part_ab; do
  curl --fail --location --retry 30 --retry-delay 10 --continue-at - \
    --output "$ASSET_ROOT/index/$part" \
    "$INDEX_URL/$part?download=true"
done

curl --fail --location --retry 30 --retry-delay 10 --continue-at - \
  --output "$ASSET_ROOT/corpus/wiki-18.jsonl.gz" \
  "$CORPUS_URL/wiki-18.jsonl.gz?download=true"

HF_ENDPOINT="$HF_ENDPOINT" HF_HOME="$ASSET_ROOT/hf-cache" \
  "$ENV_ROOT/bin/huggingface-cli" download intfloat/e5-base-v2 \
  --local-dir "$ASSET_ROOT/models/e5-base-v2"

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

