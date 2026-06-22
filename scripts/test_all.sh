#!/usr/bin/env bash
set -euo pipefail
# ---------------------------------------------------------------------------
# test_all.sh — build + full test matrix (FP16 + INT8)
#
# Usage:
#   bash scripts/test_all.sh                  # all tests
#   bash scripts/test_all.sh --no-build        # skip cmake
# ---------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/checkpoints/s2-pro}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
BIN="$BUILD_DIR/fish-server"

SEED=42

# ── helpers ────────────────────────────────────────────────────────────────

check() { [[ -f "$1" ]] || { echo "MISSING: $1"; exit 1; }; }

run_one() {
  local tag="$1" dtype="$2"
  shift 2
  local out="$OUTPUT_DIR/${tag}.wav"
  echo "  → ${tag} [${dtype}]"
  "$BIN" --model-dir "$MODEL_DIR" --dtype "$dtype" \
    --output "$out" --seed "$SEED" \
    "$@" 2>&1 | grep -E '(Resolved|first sem|Generated|Audio|WAV|GPU)'
}

# ── CLI ────────────────────────────────────────────────────────────────────

DO_BUILD=true
for a in "$@"; do
  case "$a" in
    --no-build) DO_BUILD=false ;;
    *) echo "Unknown: $a"; exit 1 ;;
  esac
done

# ── build ──────────────────────────────────────────────────────────────────

if $DO_BUILD; then
  echo "========== BUILD =========="
  check "$ROOT_DIR/CMakeLists.txt"
  check "$ROOT_DIR/models/s2-pro-fp16/dual_ar.bin"
  check "$ROOT_DIR/models/s2-pro-fp16/dac.bin"
  check "$MODEL_DIR/tokenizer.json"

  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" --target fish-server -j"$(nproc)"
  echo ""
fi

mkdir -p "$OUTPUT_DIR"

# ── unit tests ─────────────────────────────────────────────────────────────

echo "========== UNIT TESTS =========="
cmake --build "$BUILD_DIR" --target test_fish -j"$(nproc)" 2>/dev/null
"$BUILD_DIR/test_fish" 2>&1 | tail -2
echo ""

# ── A: pure text, English ──────────────────────────────────────────────────

echo "========== A: pure text (English) =========="

run_one "fp16_baseline" fp16 \
  --text "Hello world, this is a test of the speech synthesis system." \
  --max-tokens 128

run_one "int8_english"  int8 \
  --text "Hello world, this is a test of the speech synthesis system." \
  --max-tokens 128
echo ""

# ── B: pure text, Chinese ──────────────────────────────────────────────────

echo "========== B: pure text (Chinese) =========="

run_one "fp16_chinese"  fp16 \
  --text "你好世界，这是一个语音合成测试。今天天气真好，我们出去散步吧。" \
  --max-tokens 128

run_one "int8_chinese"  int8 \
  --text "你好世界，这是一个语音合成测试。今天天气真好，我们出去散步吧。" \
  --max-tokens 128
echo ""

# ── C: voice clone, Nahida (English) ───────────────────────────────────────

echo "========== C: voice clone — Nahida =========="
check "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav"
check "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab"

run_one "fp16_nahida" fp16 \
  --ref-audio "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav" \
  --ref-text "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab" \
  --text "Traveler, shall we go on an adventure today? Whether it's the winds of Mondstadt or the mountains of Liyue, I'll be right by your side." \
  --max-tokens 256

run_one "int8_nahida" int8 \
  --ref-audio "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav" \
  --ref-text "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab" \
  --text "Traveler, shall we go on an adventure today? Whether it's the winds of Mondstadt or the mountains of Liyue, I'll be right by your side." \
  --max-tokens 256
echo ""

# ── D: voice clone, 000047 (Chinese) ───────────────────────────────────────

echo "========== D: voice clone — 000047 =========="
check "$ROOT_DIR/example/000047.wav"
check "$ROOT_DIR/example/000047.lab"

run_one "fp16_000047" fp16 \
  --ref-audio "$ROOT_DIR/example/000047.wav" \
  --ref-text "$ROOT_DIR/example/000047.lab" \
  --text "今天天气真好，我们出去散步吧。阳光明媚，微风拂面，让人心情愉悦。" \
  --max-tokens 256

run_one "int8_000047" int8 \
  --ref-audio "$ROOT_DIR/example/000047.wav" \
  --ref-text "$ROOT_DIR/example/000047.lab" \
  --text "今天天气真好，我们出去散步吧。阳光明媚，微风拂面，让人心情愉悦。" \
  --max-tokens 256
echo ""

# ── summary ────────────────────────────────────────────────────────────────

echo "========== ALL OUTPUTS =========="
ls -lhS "$OUTPUT_DIR"/*.wav
echo ""
echo "Done → $OUTPUT_DIR/"
