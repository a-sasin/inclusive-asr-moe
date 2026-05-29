#!/usr/bin/env bash
set -euo pipefail

ROOT="/lp-dev/amelia"
IN_DIR="$ROOT/data/myst"
OUT_DIR="$IN_DIR/filtered_data"
WORK_DIR="$ROOT/inclusive-asr-moe/preprocessing/myst/work"

mkdir -p "$OUT_DIR" "$WORK_DIR"

PYTHONUNBUFFERED=1 python3 "$ROOT/inclusive-asr-moe/preprocessing/myst/preprocess.py" \
  --train "$IN_DIR/train.json" \
  --val "$IN_DIR/val.json" \
  --test "$IN_DIR/test.json" \
  --output-dir "$OUT_DIR" \
  --work-dir "$WORK_DIR" \
  --model large
