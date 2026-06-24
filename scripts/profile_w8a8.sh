#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
BIN="$BUILD_DIR/fish-server"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output/profile_w8a8}"
TEXT="${TEXT:-Hello world, this is a short W8A8 profiling run.}"
MAX_TOKENS="${MAX_TOKENS:-32}"
SEED="${SEED:-42}"
MODEL_FP16="${MODEL_FP16:-$ROOT_DIR/checkpoints/s2-pro}"
MODEL_INT8_DEFAULT="${MODEL_INT8_DEFAULT:-$ROOT_DIR/models/s2-pro-int8-w8a8-g256}"
MODEL_INT8_G256="${MODEL_INT8_G256:-$ROOT_DIR/models/s2-pro-int8-w8a8-g256}"

mkdir -p "$OUTPUT_DIR"

check() { [[ -e "$1" ]] || { echo "MISSING: $1"; exit 1; }; }
check "$BIN"
check "$MODEL_FP16/tokenizer.json"
check "$MODEL_INT8_DEFAULT/dual_ar.bin"

run_case() {
  local tag="$1"
  local model_dir="$2"
  local dtype="$3"
  local enable_profile="$4"
  local out="$OUTPUT_DIR/${tag}.wav"
  local log="$OUTPUT_DIR/${tag}.log"

  echo "========== ${tag} =========="
  /usr/bin/time -f "wall=%Es rss=%MKB" \
    env FISH_INT8_PROFILE="$enable_profile" \
    "$BIN" \
      --model-dir "$model_dir" \
      --dtype "$dtype" \
      --output "$out" \
      --seed "$SEED" \
      --text "$TEXT" \
      --max-tokens "$MAX_TOKENS" \
      >"$log" 2>&1
  grep -E 'Resolved model|GPU weights|Prefill|Generated|Audio|INT8 profile summary| M=|wall=' "$log" || true
  echo ""
}

run_case "fp16" "$MODEL_FP16" fp16 0
run_case "int8_default" "$MODEL_INT8_DEFAULT" int8 1

if [[ -e "$MODEL_INT8_G256/dual_ar.bin" ]]; then
  if [[ "$(readlink -f "$MODEL_INT8_DEFAULT/dual_ar.bin")" != "$(readlink -f "$MODEL_INT8_G256/dual_ar.bin")" ]]; then
    run_case "int8_g256" "$MODEL_INT8_G256" int8 1
  else
    echo "SKIP int8_g256: same dual_ar target as default W8A8"
  fi
else
  echo "SKIP int8_g256: $MODEL_INT8_G256/dual_ar.bin not found"
fi
