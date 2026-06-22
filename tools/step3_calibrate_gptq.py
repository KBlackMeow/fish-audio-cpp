#!/usr/bin/env python3
"""GPTQ-style high-precision INT8 calibration.

OPTQ algorithm (Frantar et al. 2022) adapted for per-row symmetric quantization:
  1. Collect calibration activations (FISH_CALIBRATE_DIR)
  2. Compute input Hessian H = X^T @ X per layer
  3. Column-by-column quantization with Hessian-guided error compensation
  4. Output calibrated INT8 .bin model

Usage:
    FISH_CALIBRATE_DIR=/tmp/calib fish-server --dtype fp16 --text "..." --max-tokens 1
    python tools/gptq_calibrate.py /tmp/calib dual_ar_fp16.bin dual_ar_int8.bin
"""

import struct
import sys
import os
import glob
import numpy as np
from step2_quantize_int8 import (
    read_bin, load_tensor, write_bin, should_quantize,
    DTYPE_INT8, DTYPE_FP16
)


def read_calib_dump(path):
    with open(path, 'rb') as f:
        n_tokens = struct.unpack('<i', f.read(4))[0]
        dim = struct.unpack('<i', f.read(4))[0]
        raw = f.read()
    return np.frombuffer(raw, dtype=np.float16).reshape(n_tokens, dim).astype(np.float32)


def layer_from_dump_name(filename):
    base = os.path.basename(filename)
    parts = base.replace('.bin', '').split('_', 1)
    return int(parts[0][1:]), parts[1]


def compute_hessian(acts_list, damping=0.01):
    """Compute regularized inverse Hessian H^-1 from activation matrices.

    Args:
        acts_list: list of [N_i, K] float32 activation matrices
        damping:  regularization strength (percentage of mean diagonal)

    Returns:
        H_inv: [K, K] float32 inverse Hessian
    """
    # Accumulate H = Σ X_i^T @ X_i across calibration samples
    K = acts_list[0].shape[1]
    H = np.zeros((K, K), dtype=np.float32)
    total_N = 0
    for X in acts_list:
        H += X.T @ X
        total_N += X.shape[0]

    # Average + damping
    H /= total_N
    diag_mean = np.mean(np.diag(H))
    H += damping * diag_mean * np.eye(K, dtype=np.float32)

    # Cholesky: H = L @ L^T, then H^-1 = L^-T @ L^-1
    try:
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.inv(L.T) @ np.linalg.inv(L)
    except np.linalg.LinAlgError:
        # Fallback: add more damping
        H += 0.1 * diag_mean * np.eye(K, dtype=np.float32)
        L = np.linalg.cholesky(H)
        H_inv = np.linalg.inv(L.T) @ np.linalg.inv(L)

    return H_inv


def gptq_quantize(W_fp16, H_inv, blocksize=128):
    """GPTQ column-by-column quantization with Hessian error compensation.

    Args:
        W_fp16:   [M, K] float32 weight matrix (converted from FP16)
        H_inv:    [K, K] float32 inverse Hessian
        blocksize: columns per block (larger = faster, smaller = better)

    Returns:
        W_int8: [M, K] int8
        scale:  [M] float16 per-row scale
    """
    M, K = W_fp16.shape
    W = W_fp16.astype(np.float32).copy()
    perm = np.arange(K, dtype=np.int32)

    # Sort columns by diagonal of H_inv (greedy: hardest columns first)
    diag = np.diag(H_inv).copy()
    perm = np.argsort(-diag)  # descending — largest H_inv first
    W = W[:, perm]
    H_inv_perm = H_inv[perm][:, perm]

    # Per-row quantization scales (recomputed within blocks for accuracy)
    scale = np.zeros(M, dtype=np.float32)
    W_int8 = np.zeros((M, K), dtype=np.int8)

    for k_start in range(0, K, blocksize):
        k_end = min(k_start + blocksize, K)
        block_cols = k_end - k_start

        for k in range(k_start, k_end):
            col = W[:, k].copy()

            # Compute optimal scale for this column
            s = np.max(np.abs(col)) / 127.0
            s = max(s, 1e-8)

            # Quantize
            col_q = np.clip(np.round(col / s), -127, 127).astype(np.float32) * s
            err = col_q - col

            W_int8[:, k] = np.clip(np.round(col / s), -127, 127).astype(np.int8)
            scale = np.maximum(scale, s.astype(np.float32))

            # Compensate remaining columns in this block
            if k < k_end - 1:
                remaining = slice(k + 1, k_end)
                denom = H_inv_perm[k, k]
                if denom > 1e-12:
                    coeff = H_inv_perm[k, remaining] / denom  # [remaining]
                    W[:, remaining] -= np.outer(err, coeff)

    # Unpermute
    inv_perm = np.argsort(perm)
    W_int8 = W_int8[:, inv_perm]

    return W_int8, scale.astype(np.float16)


def weight_to_calib_key(name):
    import re
    m = re.search(r'text_model\.model\.layers\.(\d+)\.attention\.wqkv', name)
    if m: return f"L{int(m.group(1)):02d}_attn"
    m = re.search(r'text_model\.model\.layers\.(\d+)\.feed_forward\.w[13]', name)
    if m: return f"L{int(m.group(1)):02d}_ffn"
    m = re.search(r'audio_decoder\.layers\.(\d+)\.attention\.wqkv', name)
    if m: return f"L{int(m.group(1)):02d}_fast_attn"
    m = re.search(r'audio_decoder\.layers\.(\d+)\.feed_forward\.w[13]', name)
    if m: return f"L{int(m.group(1)):02d}_fast_ffn"
    return None


def calibrate_and_quantize(calib_dir, input_bin, output_bin):
    print(f"Reading calibration dumps: {calib_dir}")

    # Group activation dumps by layer key
    act_groups = {}
    for dump_path in sorted(glob.glob(os.path.join(calib_dir, 'L*.bin'))):
        layer, tag = layer_from_dump_name(os.path.basename(dump_path))
        key = f"L{layer:02d}_{tag}"
        arr = read_calib_dump(dump_path)
        act_groups.setdefault(key, []).append(arr)
        print(f"  {key}: {arr.shape[0]} tokens × {arr.shape[1]} dim")

    # Precompute Hessians for layers with calibration data
    hessians = {}
    for key, acts in act_groups.items():
        print(f"  Hessian {key}: {sum(a.shape[0] for a in acts)} total tokens")
        hessians[key] = compute_hessian(acts)

    # Process model
    print(f"\nReading: {input_bin}")
    tensors, raw = read_bin(input_bin)

    calibrated = 0
    total_in = 0
    total_out = 0
    tensors_out = []

    for info in tensors:
        name = info['name']
        skip = not should_quantize(name) or info['dtype_val'] not in (0, 1, 2)

        if skip:
            info['data'] = raw[info['data_offset']:info['data_offset'] + info['data_size']]
            tensors_out.append(info)
            total_out += info['data_size']
            continue

        arr_fp16 = load_tensor(raw, info)
        arr = arr_fp16.astype(np.float32)
        cal_key = weight_to_calib_key(name)

        if cal_key and cal_key in hessians:
            w_int8, scale = gptq_quantize(arr, hessians[cal_key])
            calibrated += 1
            total_in += info['data_size']
            print(f"  GPTQ: {name}  shape={list(arr.shape)}")
        else:
            from step2_quantize_int8 import quantize_rowwise
            w_int8, scale = quantize_rowwise(arr_fp16)
            print(f"  PLAIN:{name}  shape={list(arr.shape)}")

        # INT8 weight
        info['dtype_val'] = DTYPE_INT8
        info['data'] = w_int8.tobytes()
        info['data_size'] = len(info['data'])
        tensors_out.append(info)
        total_out += info['data_size']

        # Per-row scale
        scale_shape = [arr.shape[0]]
        tensors_out.append({
            'name': name + '_scale',
            'dtype_val': DTYPE_FP16,
            'ndim': len(scale_shape),
            'shape': scale_shape,
            'data': scale.tobytes(),
            'data_size': len(scale.tobytes()),
        })
        total_out += len(scale.tobytes())

    write_bin(tensors_out, output_bin)

    in_mb = total_in / (1024 * 1024)
    out_mb = total_out / (1024 * 1024)
    print(f"\nGPTQ calibrated: {calibrated} layers")
    print(f"Weights: {in_mb:.1f} MB -> {out_mb:.1f} MB")
    print(f"Output: {output_bin}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <calib_dir> <input_fp16.bin> <output_int8.bin>")
        sys.exit(1)
    calibrate_and_quantize(sys.argv[1], sys.argv[2], sys.argv[3])
