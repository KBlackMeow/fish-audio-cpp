#!/usr/bin/env python3
"""SmoothQuant-style calibration for INT8 weight-only quantization.

Uses activation statistics (collected via FISH_CALIBRATE_DIR) to compute
per-channel smooth factors that balance weight and activation magnitudes,
reducing quantization error where it matters most.

Usage:
    # 1. Collect calibration data (run once with FP16 model):
    #    FISH_CALIBRATE_DIR=/tmp/calib fish-server --dtype fp16 --text "..." --max-tokens 1

    # 2. Quantize with calibration:
    python tools/calibrate_int8.py /tmp/calib_data dual_ar_fp16.bin dual_ar_int8_cal.bin
"""

import struct
import sys
import os
import glob
import numpy as np
from step2_quantize_int8 import (
    read_bin, load_tensor, write_bin, should_quantize,
    DTYPE_INT8, DTYPE_FP16, HEADER_SIZE
)


def read_calib_dump(path):
    """Read a calibration dump file: [n_tokens, dim] FP16 raw data."""
    with open(path, 'rb') as f:
        n_tokens = struct.unpack('<i', f.read(4))[0]
        dim = struct.unpack('<i', f.read(4))[0]
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.float16).reshape(n_tokens, dim)
    return arr.astype(np.float32)


def layer_from_dump_name(filename):
    """Extract layer index and type from dump filename like 'L05_attn.bin'."""
    base = os.path.basename(filename)
    parts = base.replace('.bin', '').split('_', 1)
    layer = int(parts[0][1:])  # 'L05' -> 5
    tag = parts[1]  # 'attn', 'ffn', 'fast_attn', 'fast_ffn'
    return layer, tag


def compute_smooth_factors(w_fp32, act_stats, alpha=0.5):
    """Compute SmoothQuant channel-wise smooth factors.

    Args:
        w_fp32:  weight matrix [M, K] in float32
        act_stats: activation statistics dict with 'max' key [K]
        alpha:   smoothing strength (0.5 = balanced, default for SmoothQuant)

    Returns:
        smooth:  [K] float32 smooth factors (multiply into weights)
        smooth_inv: [K] float32 inverse (multiply into activations)
    """
    K = w_fp32.shape[1]
    w_max_col = np.max(np.abs(w_fp32), axis=0)  # [K]
    x_max_col = act_stats['max']                 # [K]

    # SmoothQuant formula:
    # smooth[j] = (max(|X[:,j]|)^alpha) / (max(|W[:,j]|)^(1-alpha))
    # This balances quantization difficulty between weights and activations.
    eps = 1e-8
    w_pow = np.maximum(w_max_col, eps) ** (1.0 - alpha)
    x_pow = np.maximum(x_max_col, eps) ** alpha

    smooth = x_pow / w_pow

    # Normalize so mean(smooth) = 1 (avoids overall scale change)
    smooth /= np.mean(smooth)
    smooth = np.clip(smooth, 0.01, 100.0)

    smooth_inv = 1.0 / smooth
    return smooth.astype(np.float16), smooth_inv.astype(np.float16)


def calibrate_and_quantize(calib_dir, input_bin, output_bin):
    """Calibrated INT8 quantization pipeline."""

    # 1. Read activation statistics from calibration dumps
    print(f"Reading calibration dumps from: {calib_dir}")
    act_stats = {}  # key -> {'max': [K], 'count': N}
    for dump_path in sorted(glob.glob(os.path.join(calib_dir, 'L*.bin'))):
        fname = os.path.basename(dump_path)
        layer, tag = layer_from_dump_name(fname)
        arr = read_calib_dump(dump_path)
        key = f"L{layer:02d}_{tag}"

        if key not in act_stats:
            act_stats[key] = {'max': np.zeros(arr.shape[1], dtype=np.float32),
                              'count': 0}
        # Running max
        col_max = np.max(np.abs(arr), axis=0)
        act_stats[key]['max'] = np.maximum(act_stats[key]['max'], col_max)
        act_stats[key]['count'] += arr.shape[0]
        print(f"  {key}: tokens={arr.shape[0]} dim={arr.shape[1]}")

    # 2. Build mapping from weight tensor names to calibration keys
    #    dual_ar naming convention:
    #      text_model.model.layers.{l}.attention.wqkv.weight → L{l:02d}_attn
    #      text_model.model.layers.{l}.attention.wo.weight → N/A (uses attn output)
    #      text_model.model.layers.{l}.feed_forward.w1.weight → L{l:02d}_ffn
    #      text_model.model.layers.{l}.feed_forward.w3.weight → L{l:02d}_ffn
    #      audio_decoder.layers.{l}.attention.wqkv.weight → L{l:02d}_fast_attn
    #      audio_decoder.layers.{l}.feed_forward.w1.weight → L{l:02d}_fast_ffn
    def weight_to_calib_key(name):
        import re
        # Text attention: wqkv (input-side weight)
        m = re.search(r'text_model\.model\.layers\.(\d+)\.attention\.wqkv', name)
        if m: return f"L{int(m.group(1)):02d}_attn"
        # Text FFN: w_gate (w1) and w_up (w3) share ffn_norm input
        # Skip w_down (w2) — it uses post-gate activation, different stats
        m = re.search(r'text_model\.model\.layers\.(\d+)\.feed_forward\.w[13]', name)
        if m: return f"L{int(m.group(1)):02d}_ffn"
        # Fast attention: wqkv
        m = re.search(r'audio_decoder\.layers\.(\d+)\.attention\.wqkv', name)
        if m: return f"L{int(m.group(1)):02d}_fast_attn"
        # Fast FFN: w_gate (w1) and w_up (w3)
        m = re.search(r'audio_decoder\.layers\.(\d+)\.feed_forward\.w[13]', name)
        if m: return f"L{int(m.group(1)):02d}_fast_ffn"
        return None

    # 3. Read input model, compute smooth, quantize, write output
    print(f"\nReading: {input_bin}")
    tensors, raw = read_bin(input_bin)
    print(f"Processing {len(tensors)} tensors...")

    calibrated = 0
    total_in = 0
    total_out = 0
    tensors_out = []

    for info in tensors:
        name = info['name']
        skip = not should_quantize(name) or info['dtype_val'] not in (0, 1, 2)

        if skip:
            info['data'] = raw[info['data_offset']:info['data_offset']+info['data_size']]
            tensors_out.append(info)
            total_out += info['data_size']
            continue

        arr = load_tensor(raw, info).astype(np.float32)
        cal_key = weight_to_calib_key(name)

        if cal_key and cal_key in act_stats:
            # Compute smooth factors
            smooth, smooth_inv = compute_smooth_factors(arr, act_stats[cal_key])

            # Apply smooth to weights
            M, K = arr.shape[0], int(np.prod(arr.shape[1:]))
            arr_2d = arr.reshape(M, K)
            arr_smooth = arr_2d * smooth.reshape(1, K)
            arr_smooth = arr_smooth.reshape(arr.shape)

            # Quantize smoothed weights
            from step2_quantize_int8 import quantize_rowwise
            w_int8, scale = quantize_rowwise(arr_smooth.astype(np.float16))

            # Scale must be adjusted: the smoothed weight W_smooth[m,k] = W[m,k] * smooth[k]
            # W_int8[m,k] = round(W_smooth[m,k] / scale[m])
            # At inference: W_fp16[m,k] = W_int8[m,k] * scale[m] * smooth_inv[k]
            #              = round(W[m,k] * smooth[k] / scale[m]) * scale[m] * smooth_inv[k]
            #              ≈ W[m,k]
            # So we store scale[m] unchanged (it's for smoothed weights)
            # and also store smooth_inv[k] for the dequant step.

            calibrated += 1
            total_in += info['data_size']
        else:
            # No calibration data — use standard quantization
            from step2_quantize_int8 import quantize_rowwise
            w_int8, scale = quantize_rowwise(arr.astype(np.float16))
            smooth_inv = None

        # Store INT8 weight
        info['dtype_val'] = DTYPE_INT8
        info['data'] = w_int8.tobytes()
        info['data_size'] = len(info['data'])
        tensors_out.append(info)
        total_out += info['data_size']

        # Store per-row scale
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

        # Store smooth_inv (per-input-channel) if calibrated
        if smooth_inv is not None:
            smooth_shape = [len(smooth_inv)]
            tensors_out.append({
                'name': name + '_smooth_inv',
                'dtype_val': DTYPE_FP16,
                'ndim': len(smooth_shape),
                'shape': smooth_shape,
                'data': smooth_inv.tobytes(),
                'data_size': len(smooth_inv.tobytes()),
            })
            total_out += len(smooth_inv.tobytes())
            print(f"  CAL:  {name} shape={list(arr.shape)}")
        else:
            print(f"  PLAIN:{name} shape={list(arr.shape)} (no calib data)")

    write_bin(tensors_out, output_bin)

    in_mb = total_in / (1024 * 1024)
    out_mb = total_out / (1024 * 1024)
    print(f"\nCalibrated: {calibrated} layers")
    print(f"Weights: {in_mb:.1f} MB -> {out_mb:.1f} MB ({100*out_mb/max(in_mb,1):.1f}%)")
    print(f"Output: {output_bin}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <calib_dir> <input_fp16.bin> <output_int8.bin>")
        print(f"Example: {sys.argv[0]} /tmp/calib_data dual_ar_fp16.bin dual_ar_int8_cal.bin")
        sys.exit(1)
    calibrate_and_quantize(sys.argv[1], sys.argv[2], sys.argv[3])
