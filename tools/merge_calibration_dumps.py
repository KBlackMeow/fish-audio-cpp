#!/usr/bin/env python3
"""Merge many small calibration dumps into a few large files per layer/tag."""

import argparse
import re
import struct
from collections import defaultdict
from pathlib import Path


def layer_from_dump_name(path: Path):
    stem = path.stem
    parts = stem.split("_", 1)
    layer = int(parts[0][1:])
    tag = re.sub(r"_\d+$", "", parts[1])
    return f"L{layer:02d}_{tag}"


def merge_group(paths, out_path: Path):
    total_tokens = 0
    dim = None
    payloads = []
    for path in paths:
        with open(path, "rb") as f:
            n_tokens = struct.unpack("<i", f.read(4))[0]
            cur_dim = struct.unpack("<i", f.read(4))[0]
            raw = f.read()
        if dim is None:
            dim = cur_dim
        elif dim != cur_dim:
            raise ValueError(f"Mismatched dim in {path}: {cur_dim} vs {dim}")
        expected = n_tokens * dim * 2
        if len(raw) != expected:
            raise ValueError(f"Truncated dump {path}: expected {expected}, got {len(raw)}")
        total_tokens += n_tokens
        payloads.append(raw)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", total_tokens))
        f.write(struct.pack("<i", dim))
        for raw in payloads:
            f.write(raw)
    return total_tokens, dim


def main():
    parser = argparse.ArgumentParser(description="Merge calibration dump files")
    parser.add_argument("calib_dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--keep-originals", action="store_true")
    args = parser.parse_args()

    calib_dir = Path(args.calib_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else calib_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for path in sorted(calib_dir.glob("L*.bin")):
        groups[layer_from_dump_name(path)].append(path)

    if not groups:
        raise SystemExit(f"No dump files found in {calib_dir}")

    for key, paths in groups.items():
        out_path = output_dir / f"merged_{key}.bin"
        total_tokens, dim = merge_group(paths, out_path)
        print(f"{key}: {len(paths)} file(s) -> {out_path.name} ({total_tokens} tokens x {dim})")
        if not args.keep_originals and output_dir == calib_dir:
            for path in paths:
                path.unlink()


if __name__ == "__main__":
    main()
