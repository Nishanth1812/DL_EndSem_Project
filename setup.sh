#!/usr/bin/env bash
# Pinned upstream benchmark + baseline, their deps, and the one small patch our runs need.
# Needs git; needs uv unless $PIP is set (Kaggle sets it). Run from the project root.
# GPU box: torch from PyPI on Linux already ships CUDA. On Windows it is CPU-only.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clone() { [ -d external/$1 ] || { git clone -q https://github.com/UKAEA-IBM-STFC-Fusion-FMs/$1.git external/$1 && git -C external/$1 checkout -q $2; }; }
clone tokamark 1a200f4588addaad2ae3d53d438d9e1946c09294
clone tokamark_baseline 1e7c489653716a60ffe0b4c2a6a2e51883228695

# Our only upstream changes: split CSV from configs/baseline.yaml (the shots download.py wrote to ./data;
# upstream reads the full official split), validation once per epoch, DataParallel over visible GPUs.
if git -C external/tokamark_baseline apply --check "$DIR/patches/tokamark_baseline.patch" 2>/dev/null; then
  git -C external/tokamark_baseline apply "$DIR/patches/tokamark_baseline.patch"
  echo "[setup] patched tokamark_baseline (local splits, epochs, multi-GPU)"
else
  echo "[setup] NOTE: patch not applied (already patched, or upstream moved)." >&2
  echo "[setup] runs would use the FULL official split and step-based validation." >&2
fi

if [ -z "$PIP" ]; then uv venv --python 3.11 .venv; PIP="uv pip install --python .venv"; fi
$PIP -e external/tokamark line-profiler==5.0.2 torchinfo==1.8.0
$PIP huggingface_hub hf_xet  # download.py pulls the Hub release with xet's parallel chunk transfer
