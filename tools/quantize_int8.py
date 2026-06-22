#!/usr/bin/env python3
"""Quantize FP16/FP32 .bin model to INT8 W8A16 format.

Symmetric per-channel quantization: W_int8[i,:] = round(W_fp16[i,:] / scale[i])
where scale[i] = max(abs(W_fp16[i,:])) / 127.0

Skips: RMSNorm/LayerNorm weights, biases, codebooks, RoPE freqs,
alpha/gamma scalars — these stay FP16.

Usage:
    python tools/quantize_int8.py dual_ar_fp16.bin dual_ar_int8.bin
"""

import struct
import sys
import numpy as np

MAGIC = 0x46495348
FORMAT_VERSION = 1
HEADER_SIZE = 344
DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_BF16 = 2
DTYPE_INT8 = 3

SKIP_PATTERNS = [
    "_norm.", "norm.", "norm_",
    ".bias", "bias.",
    "codebook.weight",
    "codebook_embeddings",
    "freqs_cis",
    "alpha",
    "gamma",
    "out_proj.bias",
    "in_proj.bias",
]


def should_quantize(name: str) -> bool:
    for pat in SKIP_PATTERNS:
        if pat in name:
            return False
    return True


def read_bin(path: str):
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    magic = struct.unpack_from("<I", data, offset)[0]; offset += 4
    assert magic == MAGIC, f"Bad magic: 0x{magic:08X}"
    version = struct.unpack_from("<I", data, offset)[0]; offset += 4
    assert version == FORMAT_VERSION, f"Unsupported version: {version}"
    num_tensors = struct.unpack_from("<I", data, offset)[0]; offset += 4

    tensors = []
    for _ in range(num_tensors):
        name_bytes = data[offset:offset + 256]
        name = name_bytes.split(b"\x00")[0].decode("utf-8")
        dtype_val = struct.unpack_from("<I", data, offset + 256)[0]
        ndim = struct.unpack_from("<I", data, offset + 260)[0]
        shape = list(struct.unpack_from("<8q", data, offset + 264))[:ndim]
        data_offset_file = struct.unpack_from("<Q", data, offset + 328)[0]
        data_size = struct.unpack_from("<Q", data, offset + 336)[0]
        tensors.append({
            "name": name,
            "dtype_val": dtype_val,
            "ndim": ndim,
            "shape": shape,
            "data_offset": data_offset_file,
            "data_size": data_size,
        })
        offset += HEADER_SIZE
    return tensors, data


def load_tensor(raw: bytes, info: dict) -> np.ndarray:
    off = info["data_offset"]
    size = info["data_size"]
    if info["dtype_val"] in (DTYPE_FP16, DTYPE_BF16):
        dt = np.float16
    else:
        dt = np.float32
    return np.frombuffer(raw[off:off + size], dtype=dt).reshape(info["shape"])


def quantize_rowwise(w: np.ndarray):
    """Symmetric per-channel INT8 quantization.

    Returns:
        w_int8: int8 array, same shape as input
        scale: float16 array, shape [M]
    """
    w_f32 = w.astype(np.float32)
    M = w_f32.shape[0]
    K = int(np.prod(w_f32.shape[1:])) if w_f32.ndim > 1 else 1
    w_flat = w_f32.reshape(M, K)

    amax = np.maximum(np.max(np.abs(w_flat), axis=1), 1e-8)
    scale = (amax / 127.0).astype(np.float16)
    scale_bc = scale.reshape(M, 1).astype(np.float32)
    w_int8 = np.clip(np.round(w_flat / scale_bc), -127, 127).astype(np.int8)
    return w_int8.reshape(w.shape), scale


def write_bin(tensors_out: list, path: str):
    header_total = 4 + 4 + 4 + HEADER_SIZE * len(tensors_out)
    data_start = ((header_total + 255) // 256) * 256

    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", len(tensors_out)))

        cur_offset = data_start
        headers = []
        for t in tensors_out:
            name_enc = t["name"].encode("utf-8")
            if len(name_enc) > 255:
                raise ValueError(f"Tensor name too long ({len(name_enc)}): {t['name']}")
            name_padded = name_enc + b"\x00" * (256 - len(name_enc))
            f.write(name_padded)

            ndim = len(t["shape"])
            shape_padded = list(t["shape"]) + [0] * (8 - ndim)
            f.write(struct.pack("<I", t["dtype_val"]))
            f.write(struct.pack("<I", ndim))
            for s in shape_padded:
                f.write(struct.pack("<q", s))
            f.write(struct.pack("<Q", cur_offset))
            f.write(struct.pack("<Q", t["data_size"]))
            headers.append((cur_offset, t["data"]))
            cur_offset += t["data_size"]
            cur_offset = ((cur_offset + 255) // 256) * 256

        pos = f.tell()
        if pos < data_start:
            f.write(b"\x00" * (data_start - pos))

        for off, dat in headers:
            f.seek(off)
            f.write(dat)
            end = f.tell()
            aligned = ((end + 255) // 256) * 256
            if aligned > end:
                f.write(b"\x00" * (aligned - end))

    print(f"Wrote {len(tensors_out)} tensors to {path}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.bin>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    print(f"Reading: {input_path}")
    tensors, raw = read_bin(input_path)

    quantized = 0
    skipped = 0
    total_in = 0
    total_out = 0
    tensors_out = []

    for info in tensors:
        arr = load_tensor(raw, info)

        if should_quantize(info["name"]) and arr.dtype in (np.float16, np.float32):
            print(f"  INT8: {info['name']}  shape={info['shape']}")
            w_int8, scale = quantize_rowwise(arr)
            quantized += 1
            total_in += info["data_size"]

            # Replace weight tensor with INT8
            info["dtype_val"] = DTYPE_INT8
            info["data"] = w_int8.tobytes()
            info["data_size"] = len(info["data"])
            tensors_out.append(info)

            # Add scale tensor: [M] FP16
            scale_shape = [info["shape"][0]]
            scale_data = scale.tobytes()
            tensors_out.append({
                "name": info["name"] + "_scale",
                "dtype_val": DTYPE_FP16,
                "ndim": len(scale_shape),
                "shape": scale_shape,
                "data": scale_data,
                "data_size": len(scale_data),
            })
            total_out += info["data_size"] + len(scale_data)
        else:
            print(f"  FP16: {info['name']}  shape={info['shape']}  (skip)")
            info["data"] = raw[info["data_offset"]:info["data_offset"] + info["data_size"]]
            tensors_out.append(info)
            skipped += 1
            total_out += info["data_size"]

    write_bin(tensors_out, output_path)

    in_mb = total_in / (1024 * 1024)
    out_mb = total_out / (1024 * 1024)
    print(f"\nQuantized {quantized} tensors, skipped {skipped}")
    print(f"Weights: {in_mb:.1f} MB -> {out_mb:.1f} MB "
          f"({100 * out_mb / max(in_mb, 1):.1f}%)")


if __name__ == "__main__":
    main()
