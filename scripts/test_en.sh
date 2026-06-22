#!/usr/bin/env bash
set -euo pipefail
# Quick English voice-clone smoke test

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/checkpoints/s2-pro}"
DTYPE="${DTYPE:-int8}"
OUTPUT="${OUTPUT:-$ROOT_DIR/output/test_en.wav}"

require_file() { [[ -f "$1" ]] || { echo "MISSING: $1"; exit 1; }; }

require_file "$ROOT_DIR/CMakeLists.txt"
require_file "$ROOT_DIR/models/s2-pro-${DTYPE}/dual_ar.bin"
require_file "$ROOT_DIR/models/s2-pro-${DTYPE}/dac.bin"
require_file "$MODEL_DIR/tokenizer.json"
require_file "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav"
require_file "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab"

echo "== Build ($DTYPE) =="
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target fish-server -j"$(nproc)"

mkdir -p "$(dirname "$OUTPUT")"

echo "== Synthesize =="
"$BUILD_DIR/fish-server" \
  --model-dir "$MODEL_DIR" --dtype "$DTYPE" \
  --ref-audio "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav" \
  --ref-text "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab" \
  --text "${TEXT:-Traveler, shall we go on an adventure today? Whether it's the winds of Mondstadt or the mountains of Liyue, I'll be right by your side.}" \
  --output "$OUTPUT" --max-tokens "${MAX_TOKENS:-256}" \
  --seed "${SEED:-42}" 2>&1 | grep -E '(Resolved|GPU|first sem|Generated|Audio|WAV)'

echo "== Output =="
ls -lh "$OUTPUT"
