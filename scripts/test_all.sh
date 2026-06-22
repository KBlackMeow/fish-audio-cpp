#!/usr/bin/env bash
set -euo pipefail
# ---------------------------------------------------------------------------
# test_all.sh — Fish Audio S2 Pro: build + full test matrix (FP16 + INT8)
#
# Usage:
#   bash scripts/test_all.sh              # all tests
#   bash scripts/test_all.sh --quick      # 1 scenario only, for fast smoke test
#   bash scripts/test_all.sh --no-build   # skip cmake build
# ---------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/checkpoints/s2-pro}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
BIN="$BUILD_DIR/fish-server"

SEED=42
MAX_TOKENS_PURE=128
MAX_TOKENS_CLONE=256

# ── helpers ────────────────────────────────────────────────────────────────

check() { [[ -f "$1" ]] || { echo "MISSING: $1"; exit 1; }; }

run_one() {
  local tag="$1" dtype="$2" extra_args=()
  shift 2
  # Collect remaining args as extra flags
  local out="$OUTPUT_DIR/${tag}.wav"
  echo "  → ${tag} [${dtype}]"
  "$BIN" --model-dir "$MODEL_DIR" --dtype "$dtype" \
    --output "$out" --seed "$SEED" \
    "$@" 2>&1 | grep -E '(first sem|Generated|Audio|WAV|GPU weights)'
}

# ── CLI flags ───────────────────────────────────────────────────────────────

QUICK=false
DO_BUILD=true
for a in "$@"; do
  case "$a" in
    --quick)   QUICK=true ;;
    --no-build) DO_BUILD=false ;;
    *) echo "Unknown: $a"; exit 1 ;;
  esac
done

# ── build ────────────────────────────────────────────────────────────────────

if $DO_BUILD; then
  echo "========== BUILD =========="
  check "$ROOT_DIR/CMakeLists.txt"
  check "$MODEL_DIR/dual_ar_fp16.bin"
  check "$MODEL_DIR/dac_fp16.bin"
  check "$MODEL_DIR/dual_ar_config.json"
  check "$MODEL_DIR/dac_config.json"
  check "$MODEL_DIR/tokenizer.json"

  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" --target fish-server -j"$(nproc)"
  echo ""
fi

mkdir -p "$OUTPUT_DIR"

# ── unit tests ───────────────────────────────────────────────────────────────

echo "========== UNIT TESTS =========="
cmake --build "$BUILD_DIR" --target test_fish -j"$(nproc)" 2>/dev/null
"$BUILD_DIR/test_fish" 2>&1 | tail -2
echo ""

# ── Scenario A: pure text, English ───────────────────────────────────────────

echo "========== A: pure text (English) =========="
run_one "fp16_baseline" fp16 \
  --text "Hello world, this is a test of the speech synthesis system." \
  --max-tokens "$MAX_TOKENS_PURE"

run_one "int8_english"  int8 \
  --text "Hello world, this is a test of the speech synthesis system." \
  --max-tokens "$MAX_TOKENS_PURE"
echo ""

# ── Scenario B: pure text, Chinese ───────────────────────────────────────────

echo "========== B: pure text (Chinese) =========="
run_one "fp16_chinese"  fp16 \
  --text "你好世界，这是一个语音合成测试。今天天气真好，我们出去散步吧。" \
  --max-tokens "$MAX_TOKENS_PURE"

run_one "int8_chinese"  int8 \
  --text "你好世界，这是一个语音合成测试。今天天气真好，我们出去散步吧。" \
  --max-tokens "$MAX_TOKENS_PURE"
echo ""

# ── Scenario C: voice clone, Nahida (English) ────────────────────────────────

echo "========== C: voice clone — Nahida =========="
check "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav"
check "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab"

run_one "fp16_nahida" fp16 \
  --ref-audio "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav" \
  --ref-text "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab" \
  --text "Traveler, shall we go on an adventure today? Whether it's the winds of Mondstadt or the mountains of Liyue, I'll be right by your side." \
  --max-tokens "$MAX_TOKENS_CLONE"

run_one "int8_nahida" int8 \
  --ref-audio "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav" \
  --ref-text "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab" \
  --text "Traveler, shall we go on an adventure today? Whether it's the winds of Mondstadt or the mountains of Liyue, I'll be right by your side." \
  --max-tokens "$MAX_TOKENS_CLONE"
echo ""

# ── Scenario D: voice clone, 000047 (Chinese) ────────────────────────────────

echo "========== D: voice clone — 000047 =========="
check "$ROOT_DIR/example/000047.wav"
check "$ROOT_DIR/example/000047.lab"

run_one "fp16_000047" fp16 \
  --ref-audio "$ROOT_DIR/example/000047.wav" \
  --ref-text "$ROOT_DIR/example/000047.lab" \
  --text "今天天气真好，我们出去散步吧。阳光明媚，微风拂面，让人心情愉悦。" \
  --max-tokens "$MAX_TOKENS_CLONE"

run_one "int8_000047" int8 \
  --ref-audio "$ROOT_DIR/example/000047.wav" \
  --ref-text "$ROOT_DIR/example/000047.lab" \
  --text "今天天气真好，我们出去散步吧。阳光明媚，微风拂面，让人心情愉悦。" \
  --max-tokens "$MAX_TOKENS_CLONE"
echo ""

# ── quick mode: stop here ────────────────────────────────────────────────────

if $QUICK; then
  echo "========== QUICK MODE: stopping after 1 scenario =========="
  ls -lhS "$OUTPUT_DIR"/*.wav
  exit 0
fi

# ── summary ──────────────────────────────────────────────────────────────────

echo "========== ALL OUTPUTS =========="
ls -lhS "$OUTPUT_DIR"/*.wav
echo ""
echo "Done. 8 files → $OUTPUT_DIR/"
