#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/../checkpoints/s2-pro}"
REF_AUDIO="${REF_AUDIO:-$ROOT_DIR/example/000047.wav}"
REF_TEXT="${REF_TEXT:-$ROOT_DIR/example/000047.lab}"
OUTPUT="${OUTPUT:-$ROOT_DIR/output/000047_mix_test.wav}"
TEXT="${TEXT:-ねえ、你有没有想过——記憶とは何だろう？Sometimes I feel like memories are just fragments of light scattered across different timelines. 触れようとすると、するりと指の間をすり抜けていく。でも、それでも人は前に進むしかない。Because forgetting is not the end, 而是另一种开始。}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
SEED="${SEED:-42}"
JOBS="${JOBS:-$(nproc)}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

require_file "$ROOT_DIR/CMakeLists.txt"
require_file "$MODEL_DIR/dual_ar.bin"
require_file "$MODEL_DIR/dac.bin"
require_file "$MODEL_DIR/dual_ar_config.json"
require_file "$MODEL_DIR/dac_config.json"
require_file "$MODEL_DIR/tokenizer.json"
require_file "$REF_AUDIO"
require_file "$REF_TEXT"

echo "== Configure =="
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "== Build =="
cmake --build "$BUILD_DIR" --target fish-server -j"$JOBS"

mkdir -p "$(dirname "$OUTPUT")"

echo "== Synthesize (Mixed: ZH+EN+JA, speaker=000047) =="
"$BUILD_DIR/fish-server" \
  --model-dir "$MODEL_DIR" \
  --ref-audio "$REF_AUDIO" \
  --ref-text "$REF_TEXT" \
  --text "$TEXT" \
  --output "$OUTPUT" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --seed "$SEED"

echo "== Output =="
ls -lh "$OUTPUT"
if command -v ffprobe >/dev/null 2>&1; then
  ffprobe -hide_banner -loglevel error \
    -show_entries stream=codec_name,sample_rate,channels,duration \
    -of default=noprint_wrappers=1 "$OUTPUT"
fi
