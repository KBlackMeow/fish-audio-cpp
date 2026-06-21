#!/usr/bin/env python3
"""Compare C++ encoded DAC codes against Python tensor dumps or code files."""

import argparse
import struct
from pathlib import Path

import numpy as np


def load_cpp_codes(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        n, t = struct.unpack("<2i", f.read(8))
        data = np.frombuffer(f.read(n * t * 4), dtype=np.int32).copy()
    return data.reshape(n, t)


def load_tensor_codes(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = struct.unpack("<" + "i" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.float32).copy().reshape(shape)
    if len(shape) == 3:
        data = data[0]
    return data.astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-codes", required=True, help="C++ [N,T]+int32 code file")
    ap.add_argument("--py-codes", required=True, help="Python tensor dump, usually *_codes.bin")
    args = ap.parse_args()

    cpp = load_cpp_codes(Path(args.cpp_codes))
    py = load_tensor_codes(Path(args.py_codes))
    if cpp.shape != py.shape:
        raise SystemExit(f"shape mismatch: cpp={cpp.shape} py={py.shape}")

    diff = cpp != py
    total = diff.size
    print(f"shape={cpp.shape} diff={int(diff.sum())}/{total} match={1.0 - diff.mean():.6f}")
    print("per_codebook_diff=" + ",".join(str(int(v)) for v in diff.sum(axis=1)))

    mismatch = np.argwhere(diff)
    if mismatch.size:
        cb, t = mismatch[0]
        print(f"first_mismatch cb={cb} t={t} cpp={cpp[cb, t]} py={py[cb, t]}")


if __name__ == "__main__":
    main()
