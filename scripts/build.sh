#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
JOBS="${JOBS:-$(nproc)}"

echo "== Configure =="
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "== Build =="
cmake --build "$BUILD_DIR" --target fish-server -j"$JOBS"

echo "== Done =="
echo "fish-server: $BUILD_DIR/fish-server"
