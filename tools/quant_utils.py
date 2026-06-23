#!/usr/bin/env python3
"""Shared helpers for fish-audio-cpp quantization/export scripts.

This module supports the streamlined toolchain:
  - `step1_convert_*.py` produces FP16 `.bin`
  - `step2_export_w8a8.py` exports calibrated W8A8 `.bin`
"""

import os
import struct
from typing import BinaryIO, Dict, Iterable, List

import numpy as np

MAGIC = 0x46495348
FORMAT_VERSION = 1
HEADER_SIZE = 344
DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_BF16 = 2
DTYPE_INT8 = 3


def _env_group_size() -> int:
    raw = os.environ.get("FISH_INT8_GROUP_SIZE", "64").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"FISH_INT8_GROUP_SIZE must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"FISH_INT8_GROUP_SIZE must be > 0, got {value}")
    return value


DEFAULT_GROUP_SIZE = _env_group_size()

SKIP_PATTERNS = [
    "_norm.", "norm.", "norm_",
    ".bias", "bias.",
    "codebook.weight",
    "codebook_embeddings",
    "embeddings.weight",
    ".conv.",
    ".dwconv.",
    "freqs_cis",
    "alpha",
    "gamma",
    "out_proj.bias",
    "in_proj.bias",
    "in_proj.weight",
    "out_proj.weight",
]


def should_quantize(name: str) -> bool:
    keep_patterns = os.environ.get("FISH_INT8_KEEP_PATTERNS", "")
    if keep_patterns:
        for pat in keep_patterns.split(","):
            pat = pat.strip()
            if pat and pat in name:
                return False
    for pat in SKIP_PATTERNS:
        if pat in name:
            return False
    return True


def parse_headers(f: BinaryIO) -> List[Dict]:
    f.seek(0)
    prefix = f.read(12)
    if len(prefix) != 12:
        raise ValueError("File too small for header")
    magic, version, num_tensors = struct.unpack("<III", prefix)
    assert magic == MAGIC, f"Bad magic: 0x{magic:08X}"
    assert version == FORMAT_VERSION, f"Unsupported version: {version}"

    tensors = []
    for _ in range(num_tensors):
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError("Truncated tensor header")
        name_bytes = header[:256]
        name = name_bytes.split(b"\x00")[0].decode("utf-8")
        dtype_val = struct.unpack_from("<I", header, 256)[0]
        ndim = struct.unpack_from("<I", header, 260)[0]
        shape = list(struct.unpack_from("<8q", header, 264))[:ndim]
        data_offset_file = struct.unpack_from("<Q", header, 328)[0]
        data_size = struct.unpack_from("<Q", header, 336)[0]
        tensors.append({
            "name": name,
            "dtype_val": dtype_val,
            "ndim": ndim,
            "shape": shape,
            "data_offset": data_offset_file,
            "data_size": data_size,
        })
    return tensors


def read_bin(path: str):
    with open(path, "rb") as f:
        tensors = parse_headers(f)
        f.seek(0)
        data = f.read()
    return tensors, data


def get_tensor_dtype(dtype_val: int):
    if dtype_val in (DTYPE_FP16, DTYPE_BF16):
        return np.float16
    if dtype_val == DTYPE_INT8:
        return np.int8
    return np.float32


def tensor_numel(shape: List[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def tensor_matrix_dims(shape: List[int]):
    if not shape:
        return 1, 1
    m = shape[0]
    k = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    return m, k


def scale_shape_for_weight(shape: List[int], group_size: int = DEFAULT_GROUP_SIZE) -> List[int]:
    m, k = tensor_matrix_dims(shape)
    groups = (k + group_size - 1) // group_size
    if groups <= 1:
        return [m]
    return [m, groups]


def load_tensor(raw: bytes, info: dict) -> np.ndarray:
    off = info["data_offset"]
    size = info["data_size"]
    dt = get_tensor_dtype(info["dtype_val"])
    return np.frombuffer(raw[off:off + size], dtype=dt).reshape(info["shape"])


def load_tensor_from_file(f: BinaryIO, info: dict) -> np.ndarray:
    f.seek(info["data_offset"])
    raw = f.read(info["data_size"])
    dt = get_tensor_dtype(info["dtype_val"])
    return np.frombuffer(raw, dtype=dt).reshape(info["shape"])


def copy_tensor_data(
    fin: BinaryIO,
    fout: BinaryIO,
    info: dict,
    out_offset: int,
    chunk_size: int = 64 * 1024 * 1024,
):
    fin.seek(info["data_offset"])
    fout.seek(out_offset)
    remaining = info["data_size"]
    while remaining > 0:
        chunk = fin.read(min(chunk_size, remaining))
        if not chunk:
            raise ValueError(f"Unexpected EOF while copying tensor {info['name']}")
        fout.write(chunk)
        remaining -= len(chunk)


def quantize_groupwise(w: np.ndarray, group_size: int = DEFAULT_GROUP_SIZE):
    w_f32 = w.astype(np.float32)
    m, k = tensor_matrix_dims(list(w_f32.shape))
    w_flat = w_f32.reshape(m, k)
    groups = (k + group_size - 1) // group_size
    scale = np.empty((m, groups), dtype=np.float16)
    w_int8 = np.empty_like(w_flat, dtype=np.int8)

    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, k)
        chunk = w_flat[:, start:end]
        amax = np.maximum(np.max(np.abs(chunk), axis=1), 1e-8)
        s = (amax / 127.0).astype(np.float16)
        scale[:, g] = s
        w_int8[:, start:end] = np.clip(
            np.round(chunk / s.reshape(m, 1).astype(np.float32)),
            -127,
            127,
        ).astype(np.int8)

    if groups == 1:
        return w_int8.reshape(w.shape), scale[:, 0]
    return w_int8.reshape(w.shape), scale


def align256(x: int) -> int:
    return ((x + 255) // 256) * 256


def write_headers_only(entries: List[Dict], path: str):
    header_total = 12 + HEADER_SIZE * len(entries)
    data_start = align256(header_total)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", len(entries)))

        for entry in entries:
            name_enc = entry["name"].encode("utf-8")
            if len(name_enc) > 255:
                raise ValueError(f"Tensor name too long ({len(name_enc)}): {entry['name']}")
            f.write(name_enc + b"\x00" * (256 - len(name_enc)))

            shape = entry["shape"]
            ndim = len(shape)
            shape_padded = shape + [0] * (8 - ndim)
            f.write(struct.pack("<I", entry["dtype_val"]))
            f.write(struct.pack("<I", ndim))
            for s in shape_padded:
                f.write(struct.pack("<q", s))
            f.write(struct.pack("<Q", entry["data_offset"]))
            f.write(struct.pack("<Q", entry["data_size"]))

        pos = f.tell()
        if pos < data_start:
            f.write(b"\x00" * (data_start - pos))
    return data_start
