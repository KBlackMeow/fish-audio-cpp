#!/usr/bin/env bash
set -euo pipefail
# CI: build + unit tests + quick smoke test

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/checkpoints/s2-pro}"
JOBS="${JOBS:-$(nproc)}"

require_file() { [[ -f "$1" ]] || { echo "MISSING: $1"; exit 1; }; }

require_file "$ROOT_DIR/CMakeLists.txt"
require_file "$ROOT_DIR/models/s2-pro-fp16/dual_ar.bin"
require_file "$ROOT_DIR/models/s2-pro-fp16/dac.bin"
require_file "$MODEL_DIR/tokenizer.json"

echo "== Configure =="
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "== Build =="
cmake --build "$BUILD_DIR" --target fish-server -j"$JOBS"
cmake --build "$BUILD_DIR" --target test_fish -j"$JOBS" 2>/dev/null

echo "== Unit Tests =="
"$BUILD_DIR/test_fish" 2>&1 | tail -3

echo "== Smoke Test =="
mkdir -p "$ROOT_DIR/output"
"$BUILD_DIR/fish-server" \
  --model-dir "$MODEL_DIR" --dtype int8 \
  --text "Hello world, this is a smoke test." \
  --output "$ROOT_DIR/output/smoke_test.wav" \
  --max-tokens 16 --seed 42 2>&1 | grep -E '(Resolved|GPU|first sem|WAV)'

echo "== Done =="
ls -lh "$ROOT_DIR/output/smoke_test.wav"
