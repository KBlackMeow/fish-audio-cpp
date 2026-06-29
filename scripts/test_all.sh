#!/usr/bin/env bash
set -euo pipefail
# ---------------------------------------------------------------------------
# test_all.sh — build + full test matrix across all 3 models
#
# Models:
#   s2-pro-fp16
#   s2-pro-int8-w8a8-g256
#   s2-pro-int8-w8a8-g64
#
# Usage:
#   bash scripts/test_all.sh                  # all tests
#   bash scripts/test_all.sh --no-build        # skip cmake
# ---------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
BIN="$BUILD_DIR/fish-server"

SEED=42

# ── models ────────────────────────────────────────────────────────────────────

MODELS=(
  "fp16:$ROOT_DIR/models/s2-pro-fp16"
  "int8-g256:$ROOT_DIR/models/s2-pro-int8-w8a8-g256"
  "int8-g64:$ROOT_DIR/models/s2-pro-int8-w8a8-g64"
)

# ── helpers ────────────────────────────────────────────────────────────────

check() { [[ -f "$1" ]] || { echo "MISSING: $1"; exit 1; }; }

run_one() {
  local tag="$1" model_dir="$2"
  shift 2
  local out="$OUTPUT_DIR/${tag}.wav"
  echo "  → ${tag}"
  "$BIN" --model-dir "$model_dir" \
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
  for m in "${MODELS[@]}"; do
    _dir="${m#*:}"
    check "$_dir/dual_ar.bin"
    check "$_dir/dac.bin"
  done

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

# ── iterate over models ────────────────────────────────────────────────────

for m in "${MODELS[@]}"; do
  m_tag="${m%%:*}"
  m_dir="${m#*:}"

  echo "========== MODEL: ${m_tag} =========="

  # ── pure text, English ──────────────────────────────────────────────────

  echo "--- pure text (English) ---"
  run_one "${m_tag}_english" "$m_dir" \
    --text "Hello world, this is a test of the speech synthesis system." \
    --max-tokens 128

  # ── pure text, Chinese ──────────────────────────────────────────────────

  echo "--- pure text (Chinese) ---"
  run_one "${m_tag}_chinese" "$m_dir" \
    --text "你好世界，这是一个语音合成测试。今天天气真好，我们出去散步吧。" \
    --max-tokens 128

  # ── voice clone, Nahida (English) ───────────────────────────────────────

  echo "--- voice clone: Nahida ---"
  check "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav"
  check "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab"
  run_one "${m_tag}_nahida" "$m_dir" \
    --ref-audio "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.wav" \
    --ref-text "$ROOT_DIR/example/vo_LLZAQ001_4_nahida_03.lab" \
    --text "Traveler, shall we go on an adventure today? Whether it's the winds of Mondstadt or the mountains of Liyue, I'll be right by your side." \
    --max-tokens 256

  # ── voice clone, 000047 (Chinese) ───────────────────────────────────────

  echo "--- voice clone: 000047 ---"
  check "$ROOT_DIR/example/000047.wav"
  check "$ROOT_DIR/example/000047.lab"
  run_one "${m_tag}_000047" "$m_dir" \
    --ref-audio "$ROOT_DIR/example/000047.wav" \
    --ref-text "$ROOT_DIR/example/000047.lab" \
    --text "今天天气真好，我们出去散步吧。阳光明媚，微风拂面，让人心情愉悦。" \
    --max-tokens 256

  echo ""
done

# ── summary ────────────────────────────────────────────────────────────────

echo "========== ALL OUTPUTS =========="
ls -lhS "$OUTPUT_DIR"/*.wav
echo ""
echo "Done → $OUTPUT_DIR/"
