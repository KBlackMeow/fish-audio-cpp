#!/usr/bin/env python3
"""Compare C++ and Python DAC intermediate tensor dumps."""

import argparse
import struct

import numpy as np


def load(path: str):
    with open(path, "rb") as f:
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = struct.unpack("<" + "i" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.float32).copy().reshape(shape)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-prefix", required=True)
    ap.add_argument("--py-prefix", required=True)
    args = ap.parse_args()

    for name in ["rvq", "post", "upsample", "pre_tanh", "audio"]:
        c = load(f"{args.cpp_prefix}_{name}.bin")
        p = load(f"{args.py_prefix}_{name}.bin")
        if c.shape != p.shape:
            print(f"{name}: shape mismatch cpp={c.shape} py={p.shape}")
            continue
        d = np.abs(c - p)
        denom = np.maximum(np.abs(p), 1e-6)
        rel = d / denom
        print(
            f"{name}: shape={c.shape} "
            f"max={d.max():.6g} mean={d.mean():.6g} "
            f"rel_max={rel.max():.6g} rel_mean={rel.mean():.6g} "
            f"cpp_range=[{c.min():.6g},{c.max():.6g}] "
            f"py_range=[{p.min():.6g},{p.max():.6g}]"
        )


if __name__ == "__main__":
    main()
