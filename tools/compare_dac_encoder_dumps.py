#!/usr/bin/env python3
"""Compare C++ and Python DAC encoder intermediate tensor dumps."""

import argparse
import struct
from pathlib import Path

import numpy as np


DEFAULT_NAMES = [
    "enc_block0",
    "enc_block1",
    "enc_block2",
    "enc_block3",
    "enc_block4",
    "enc_block6",
    "enc_downsample0",
    "enc_downsample1",
    "enc_pre_module_layer0",
    "enc_pre_module_layer1",
    "enc_pre_module_layer2",
    "enc_pre_module_layer3",
    "enc_pre_module_layer4",
    "enc_pre_module_layer5",
    "enc_pre_module_layer6",
    "enc_pre_module_layer7",
    "enc_pre_module",
]


def load(path: Path):
    with path.open("rb") as f:
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = struct.unpack("<" + "i" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.float32).copy().reshape(shape)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-prefix", required=True)
    ap.add_argument("--py-prefix", required=True)
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    args = ap.parse_args()

    for name in args.names:
        cpp_path = Path(f"{args.cpp_prefix}_{name}.bin")
        py_path = Path(f"{args.py_prefix}_{name}.bin")
        if not cpp_path.exists() or not py_path.exists():
            print(f"{name}: missing cpp={cpp_path.exists()} py={py_path.exists()}")
            continue

        c = load(cpp_path)
        p = load(py_path)
        if c.ndim == 3 and p.ndim == 3 and c.shape == (p.shape[0], p.shape[2], p.shape[1]):
            p = np.transpose(p, (0, 2, 1)).copy()
        if c.shape != p.shape:
            print(f"{name}: shape mismatch cpp={c.shape} py={p.shape}")
            continue

        d = np.abs(c - p)
        rel = d / np.maximum(np.abs(p), 1e-6)
        print(
            f"{name}: shape={c.shape} "
            f"max={d.max():.6g} mean={d.mean():.6g} "
            f"rel_max={rel.max():.6g} rel_mean={rel.mean():.6g} "
            f"cpp_range=[{c.min():.6g},{c.max():.6g}] "
            f"py_range=[{p.min():.6g},{p.max():.6g}]"
        )


if __name__ == "__main__":
    main()
