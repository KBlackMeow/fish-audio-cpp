#!/usr/bin/env python3
"""Export a calibrated W8A8 model from an FP16 `.bin`.

Pipeline:
  1. Generate FP16 `.bin` with `step1_convert_dual_ar.py`
  2. Collect calibration dumps with `FISH_CALIBRATE_DIR=... fish-server --dtype fp16`
  3. Export W8A8 with this script

Usage:
    python tools/step2_export_w8a8.py /tmp/calib dual_ar_fp16.bin dual_ar_int8-w8a8.bin --device cuda:0
"""

import glob
import os
import re
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
except Exception:
    torch = None

from quant_utils import (
    DEFAULT_GROUP_SIZE,
    DTYPE_FP16,
    DTYPE_INT8,
    copy_tensor_data,
    load_tensor_from_file,
    parse_headers,
    quantize_groupwise,
    should_quantize,
    tensor_numel,
    write_headers_only,
)


FULL_HESSIAN_MAX_K = 4096
GPTQ_BLOCKSIZE = 128
TORCH_CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())
TORCH_DEVICE = None
AWQ_ALPHA = float(os.environ.get("FISH_AWQ_ALPHA", "0.5"))
AWQ_TOP_FRAC = float(os.environ.get("FISH_AWQ_TOP_FRAC", "0.01"))
AWQ_SMOOTH_MIN = float(os.environ.get("FISH_AWQ_SMOOTH_MIN", "0.25"))
AWQ_SMOOTH_MAX = float(os.environ.get("FISH_AWQ_SMOOTH_MAX", "4.0"))
USE_STATIC_ACT_SCALE = os.environ.get("FISH_INT8_STATIC_ACT_SCALE", "0") == "1"


def read_calib_header(path):
    with open(path, "rb") as f:
        n_tokens = struct.unpack("<i", f.read(4))[0]
        dim = struct.unpack("<i", f.read(4))[0]
    return n_tokens, dim


def configure_torch_device(device_spec):
    global TORCH_CUDA_AVAILABLE
    global TORCH_DEVICE

    if torch is None:
        TORCH_CUDA_AVAILABLE = False
        TORCH_DEVICE = None
        return

    if device_spec in ("auto", "", None):
        if torch.cuda.is_available():
            TORCH_DEVICE = torch.device("cuda")
            TORCH_CUDA_AVAILABLE = True
        else:
            TORCH_DEVICE = torch.device("cpu")
            TORCH_CUDA_AVAILABLE = False
        return

    if device_spec == "cpu":
        TORCH_DEVICE = torch.device("cpu")
        TORCH_CUDA_AVAILABLE = False
        return

    TORCH_DEVICE = torch.device(device_spec)
    TORCH_CUDA_AVAILABLE = TORCH_DEVICE.type == "cuda" and torch.cuda.is_available()
    if TORCH_DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")


def iter_calib_chunks(path, chunk_tokens=8192):
    with open(path, "rb") as f:
        n_tokens = struct.unpack("<i", f.read(4))[0]
        dim = struct.unpack("<i", f.read(4))[0]
        remaining = n_tokens
        while remaining > 0:
            cur = min(chunk_tokens, remaining)
            raw = f.read(cur * dim * 2)
            if len(raw) != cur * dim * 2:
                raise ValueError(f"Truncated calibration dump: {path}")
            yield np.frombuffer(raw, dtype=np.float16).reshape(cur, dim).astype(np.float32)
            remaining -= cur


def layer_from_dump_name(filename):
    base = os.path.basename(filename)
    stem = base.replace(".bin", "")
    parts = stem.split("_", 1)
    layer = int(parts[0][1:])
    tag = parts[1]
    tag = re.sub(r"_\d+$", "", tag)
    return layer, tag


def weight_to_calib_key(name):
    m = re.search(r"text_model\.model\.layers\.(\d+)\.attention\.wqkv", name)
    if m:
        return f"L{int(m.group(1)):02d}_attn"
    m = re.search(r"text_model\.model\.layers\.(\d+)\.feed_forward\.w[13]", name)
    if m:
        return f"L{int(m.group(1)):02d}_ffn"
    m = re.search(r"audio_decoder\.layers\.(\d+)\.attention\.wqkv", name)
    if m:
        return f"L{int(m.group(1)):02d}_fast_attn"
    m = re.search(r"audio_decoder\.layers\.(\d+)\.feed_forward\.w[13]", name)
    if m:
        return f"L{int(m.group(1)):02d}_fast_ffn"
    return None


def scan_calibration_dir(calib_dir):
    act_files = defaultdict(list)
    merged = sorted(glob.glob(os.path.join(calib_dir, "merged_L*.bin")))
    if merged:
        for dump_path in merged:
            base = os.path.basename(dump_path)
            key = base.replace("merged_", "").replace(".bin", "")
            act_files[key].append(dump_path)
    else:
        for dump_path in sorted(glob.glob(os.path.join(calib_dir, "L*.bin"))):
            layer, tag = layer_from_dump_name(os.path.basename(dump_path))
            key = f"L{layer:02d}_{tag}"
            act_files[key].append(dump_path)

    if not act_files:
        raise ValueError(f"No calibration dumps found in {calib_dir}")

    print(f"Reading calibration dump index: {calib_dir}")
    summaries = {}
    for key, paths in act_files.items():
        total_tokens = 0
        dim = None
        for path in paths:
            n_tokens, cur_dim = read_calib_header(path)
            if dim is None:
                dim = cur_dim
            elif dim != cur_dim:
                raise ValueError(f"Mismatched dim for {key}: {dim} vs {cur_dim}")
            total_tokens += n_tokens
        summaries[key] = {"tokens": total_tokens, "dim": dim, "files": len(paths)}
        print(f"  {key}: {total_tokens} tokens x {dim} dim across {len(paths)} dump(s)")
    return act_files, summaries


def compute_activation_max(paths, dim):
    col_max = np.zeros(dim, dtype=np.float32)
    total_tokens = 0
    for path in paths:
        for chunk in iter_calib_chunks(path):
            col_max = np.maximum(col_max, np.max(np.abs(chunk), axis=0))
            total_tokens += chunk.shape[0]
    if total_tokens == 0:
        raise ValueError("Calibration dump is empty")
    return {"max": col_max, "count": total_tokens}


def compute_awq_smooth_factors(w_fp32, act_stats):
    k = w_fp32.shape[1]
    w_max_col = np.maximum(np.max(np.abs(w_fp32), axis=0), 1e-8)
    x_max_col = np.maximum(act_stats["max"], 1e-8)
    importance = x_max_col * w_max_col

    top_k = max(1, int(k * AWQ_TOP_FRAC))
    top_idx = np.argpartition(importance, -top_k)[-top_k:]

    smooth = np.ones(k, dtype=np.float32)
    smooth_sel = (x_max_col[top_idx] ** AWQ_ALPHA) / (w_max_col[top_idx] ** (1.0 - AWQ_ALPHA))
    smooth_sel = np.clip(smooth_sel, AWQ_SMOOTH_MIN, AWQ_SMOOTH_MAX)
    smooth[top_idx] = smooth_sel

    smooth /= np.mean(smooth)
    smooth = np.clip(smooth, AWQ_SMOOTH_MIN, AWQ_SMOOTH_MAX)
    smooth_inv = 1.0 / smooth
    return smooth.astype(np.float16), smooth_inv.astype(np.float16)


def compute_group_act_scales(act_stats, smooth_inv=None, group_size=DEFAULT_GROUP_SIZE):
    act_max = np.maximum(act_stats["max"].astype(np.float32, copy=True), 1e-8)
    if smooth_inv is not None:
        act_max *= smooth_inv.astype(np.float32)
    groups = (act_max.shape[0] + group_size - 1) // group_size
    scales = np.empty(groups, dtype=np.float16)
    for g in range(groups):
        start = g * group_size
        end = min(start + group_size, act_max.shape[0])
        scales[g] = np.float16(max(float(np.max(act_max[start:end])) / 127.0, 1e-8))
    return scales


def build_w8a8_output_plan(tensors, quantize_pred, act_files):
    entries = []
    for info in tensors:
        shape = list(info["shape"])
        quantize = quantize_pred(info)
        cal_key = weight_to_calib_key(info["name"])

        if quantize:
            entries.append({
                "name": info["name"],
                "dtype_val": DTYPE_INT8,
                "shape": shape,
                "data_size": tensor_numel(shape),
                "source": info,
                "kind": "quantized_weight",
            })

            m = shape[0] if shape else 1
            k = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            groups = (k + DEFAULT_GROUP_SIZE - 1) // DEFAULT_GROUP_SIZE
            scale_shape = [m] if groups <= 1 else [m, groups]
            entries.append({
                "name": info["name"] + "_scale",
                "dtype_val": DTYPE_FP16,
                "shape": scale_shape,
                "data_size": tensor_numel(scale_shape) * np.dtype(np.float16).itemsize,
                "source": info,
                "kind": "scale",
            })

            if cal_key and cal_key in act_files:
                entries.append({
                    "name": info["name"] + "_smooth_inv",
                    "dtype_val": DTYPE_FP16,
                    "shape": [k],
                    "data_size": k * np.dtype(np.float16).itemsize,
                    "source": info,
                    "kind": "smooth_inv",
                })
                if USE_STATIC_ACT_SCALE:
                    entries.append({
                        "name": info["name"] + "_act_scale",
                        "dtype_val": DTYPE_FP16,
                        "shape": [groups],
                        "data_size": groups * np.dtype(np.float16).itemsize,
                        "source": info,
                        "kind": "act_scale",
                    })
        else:
            entries.append({
                "name": info["name"],
                "dtype_val": info["dtype_val"],
                "shape": shape,
                "data_size": info["data_size"],
                "source": info,
                "kind": "copy",
            })

    header_total = 12 + 344 * len(entries)
    data_start = ((header_total + 255) // 256) * 256
    cur_offset = data_start
    for entry in entries:
        entry["data_offset"] = cur_offset
        cur_offset = ((cur_offset + entry["data_size"] + 255) // 256) * 256
    return entries


def compute_hessian_from_files(paths, dim, damping=0.01):
    if dim > FULL_HESSIAN_MAX_K:
        print(f"    (K={dim}, using diagonal Hessian approximation)")
        diag = np.zeros(dim, dtype=np.float32)
        total_tokens = 0
        for path in paths:
            for chunk in iter_calib_chunks(path):
                diag += np.sum(chunk * chunk, axis=0)
                total_tokens += chunk.shape[0]
        if total_tokens == 0:
            raise ValueError("Calibration dump is empty")
        diag /= total_tokens
        diag = np.maximum(diag, 1e-8)
        diag_mean = float(np.mean(diag))
        diag += damping * diag_mean
        return {"kind": "diag", "inv_diag": (1.0 / diag).astype(np.float32)}

    if TORCH_CUDA_AVAILABLE:
        device = TORCH_DEVICE
        h = torch.zeros((dim, dim), dtype=torch.float32, device=device)
        total_tokens = 0
        for path in paths:
            for chunk in iter_calib_chunks(path):
                x = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
                h += x.transpose(0, 1) @ x
                total_tokens += x.shape[0]
        if total_tokens == 0:
            raise ValueError("Calibration dump is empty")

        h /= float(total_tokens)
        diag_mean = torch.diag(h).mean()
        h += damping * diag_mean * torch.eye(dim, dtype=torch.float32, device=device)

        try:
            l = torch.linalg.cholesky(h)
            eye = torch.eye(dim, dtype=torch.float32, device=device)
            l_inv = torch.linalg.solve_triangular(l, eye, upper=False)
            h_inv = l_inv.transpose(0, 1) @ l_inv
            return {"kind": "full", "inv": h_inv.cpu().numpy().astype(np.float32)}
        except RuntimeError:
            diag = torch.diag(h).clamp_min(1e-8)
            return {"kind": "diag", "inv_diag": (1.0 / diag).cpu().numpy().astype(np.float32)}

    h = np.zeros((dim, dim), dtype=np.float32)
    total_tokens = 0
    for path in paths:
        for chunk in iter_calib_chunks(path):
            h += chunk.T @ chunk
            total_tokens += chunk.shape[0]
    if total_tokens == 0:
        raise ValueError("Calibration dump is empty")

    h /= total_tokens
    diag_mean = float(np.mean(np.diag(h)))
    h += damping * diag_mean * np.eye(dim, dtype=np.float32)

    try:
        l = np.linalg.cholesky(h)
        l_inv = np.linalg.inv(l)
        h_inv = l_inv.T @ l_inv
    except np.linalg.LinAlgError:
        diag = np.maximum(np.diag(h).copy(), 1e-8)
        return {"kind": "diag", "inv_diag": (1.0 / diag).astype(np.float32)}

    return {"kind": "full", "inv": h_inv.astype(np.float32)}


def gptq_quantize(w_fp32, hessian_info, blocksize=GPTQ_BLOCKSIZE):
    m, k = w_fp32.shape
    w = w_fp32.reshape(m, k).astype(np.float32, copy=True)
    groups = (k + DEFAULT_GROUP_SIZE - 1) // DEFAULT_GROUP_SIZE
    scale = np.empty((m, groups), dtype=np.float32)
    for g in range(groups):
        start = g * DEFAULT_GROUP_SIZE
        end = min(start + DEFAULT_GROUP_SIZE, k)
        amax = np.maximum(np.max(np.abs(w[:, start:end]), axis=1), 1e-8)
        scale[:, g] = amax / 127.0

    if hessian_info["kind"] == "diag":
        inv_diag = hessian_info["inv_diag"]
        perm = np.argsort(-inv_diag)
        w = w[:, perm]
        w_int8_perm = np.empty((m, k), dtype=np.int8)
        perm_groups = perm // DEFAULT_GROUP_SIZE

        for idx in range(k):
            col = w[:, idx]
            q = np.clip(np.round(col / scale[:, perm_groups[idx]]), -127, 127).astype(np.int8)
            w_int8_perm[:, idx] = q

        inv_perm = np.argsort(perm)
        return w_int8_perm[:, inv_perm].reshape(w_fp32.shape), scale.astype(np.float16)

    if TORCH_CUDA_AVAILABLE:
        device = TORCH_DEVICE
        h_inv = torch.from_numpy(hessian_info["inv"]).to(device=device, dtype=torch.float32)
        perm_t = torch.argsort(-torch.diag(h_inv))
        w_t = torch.from_numpy(w).to(device=device, dtype=torch.float32)[:, perm_t]
        h_inv_perm = h_inv.index_select(0, perm_t).index_select(1, perm_t)
        scale_t = torch.from_numpy(scale).to(device=device, dtype=torch.float32)
        perm_groups_t = torch.div(perm_t, DEFAULT_GROUP_SIZE, rounding_mode="floor")
        w_int8_perm = torch.empty((m, k), dtype=torch.int8, device=device)

        for k_start in range(0, k, blocksize):
            k_end = min(k_start + blocksize, k)
            for idx in range(k_start, k_end):
                col = w_t[:, idx]
                scale_col = scale_t[:, perm_groups_t[idx]]
                q = torch.clamp(torch.round(col / scale_col), -127, 127).to(torch.int8)
                col_q = q.to(torch.float32) * scale_col
                err = col_q - col
                w_int8_perm[:, idx] = q

                if idx < k_end - 1:
                    remaining = slice(idx + 1, k_end)
                    denom = h_inv_perm[idx, idx]
                    if float(denom.item()) > 1e-12:
                        coeff = h_inv_perm[idx, remaining] / denom
                        w_t[:, remaining] -= err.unsqueeze(1) * coeff.unsqueeze(0)

        inv_perm_t = torch.argsort(perm_t)
        w_int8 = w_int8_perm.index_select(1, inv_perm_t).cpu().numpy()
        return w_int8.reshape(w_fp32.shape), scale.astype(np.float16)

    h_inv = hessian_info["inv"]
    perm = np.argsort(-np.diag(h_inv))
    w = w[:, perm]
    h_inv_perm = h_inv[perm][:, perm]

    w_int8_perm = np.empty((m, k), dtype=np.int8)
    perm_groups = perm // DEFAULT_GROUP_SIZE

    for k_start in range(0, k, blocksize):
        k_end = min(k_start + blocksize, k)
        for idx in range(k_start, k_end):
            col = w[:, idx].copy()
            scale_col = scale[:, perm_groups[idx]]
            q_int8 = np.clip(np.round(col / scale_col), -127, 127).astype(np.int8)
            col_q = q_int8.astype(np.float32) * scale_col
            err = col_q - col
            w_int8_perm[:, idx] = q_int8

            if idx < k_end - 1:
                remaining = slice(idx + 1, k_end)
                denom = float(h_inv_perm[idx, idx])
                if denom > 1e-12:
                    coeff = h_inv_perm[idx, remaining] / denom
                    w[:, remaining] -= np.outer(err, coeff)

    inv_perm = np.argsort(perm)
    w_int8 = w_int8_perm[:, inv_perm]
    return w_int8.reshape(w_fp32.shape), scale.astype(np.float16)


def export_w8a8(calib_dir, input_bin, output_bin):
    act_files, act_summaries = scan_calibration_dir(calib_dir)
    print(f"Using W8A8 group_size={DEFAULT_GROUP_SIZE} static_act_scale={USE_STATIC_ACT_SCALE}")

    with open(input_bin, "rb") as fin:
        tensors = parse_headers(fin)

    quantize_pred = lambda info: should_quantize(info["name"]) and info["dtype_val"] in (0, 1, 2)
    entries = build_w8a8_output_plan(tensors, quantize_pred, act_files)
    write_headers_only(entries, output_bin)

    future_key_uses = Counter()
    for info in tensors:
        key = weight_to_calib_key(info["name"])
        if key and quantize_pred(info) and key in act_files:
            future_key_uses[key] += 1

    hessian_cache = {}
    act_max_cache = {}
    entry_iter = iter(entries)
    calibrated = 0
    quantized_plain = 0
    awq_smoothed = 0
    total_out = 0
    total = len(tensors)

    with open(input_bin, "rb") as fin, open(output_bin, "r+b") as fout:
        for idx, info in enumerate(tensors):
            out_entry = next(entry_iter)
            assert out_entry["source"]["name"] == info["name"]

            if not quantize_pred(info):
                copy_tensor_data(fin, fout, info, out_entry["data_offset"])
                total_out += out_entry["data_size"]
                continue

            arr = load_tensor_from_file(fin, info)
            arr_fp32 = arr.astype(np.float32)
            cal_key = weight_to_calib_key(info["name"])
            smooth_inv = None

            if cal_key and cal_key in act_files:
                if cal_key not in hessian_cache:
                    summary = act_summaries[cal_key]
                    print(f"  Hessian {cal_key}: {summary['tokens']} tokens x {summary['dim']} dim")
                    hessian_cache[cal_key] = compute_hessian_from_files(act_files[cal_key], summary["dim"])
                    act_max_cache[cal_key] = compute_activation_max(act_files[cal_key], summary["dim"])

                print(f"  [{idx + 1}/{total}] GPTQ+AWQ: {info['name']} shape={list(arr.shape)}", end="", flush=True)
                smooth, smooth_inv = compute_awq_smooth_factors(arr_fp32, act_max_cache[cal_key])
                arr_awq = arr_fp32 * smooth.astype(np.float32).reshape(1, -1)
                w_int8, scale = gptq_quantize(arr_awq, hessian_cache[cal_key])
                act_scale = compute_group_act_scales(act_max_cache[cal_key], smooth_inv) if USE_STATIC_ACT_SCALE else None
                calibrated += 1
                awq_smoothed += 1
                print(" ✓")

                future_key_uses[cal_key] -= 1
                if future_key_uses[cal_key] == 0:
                    del hessian_cache[cal_key]
                    del act_max_cache[cal_key]
            else:
                w_int8, scale = quantize_groupwise(arr)
                quantized_plain += 1
                print(f"  [{idx + 1}/{total}] PLAIN: {info['name']} shape={list(arr.shape)}")

            fout.seek(out_entry["data_offset"])
            fout.write(w_int8.tobytes())
            total_out += out_entry["data_size"]

            scale_entry = next(entry_iter)
            assert scale_entry["kind"] == "scale"
            fout.seek(scale_entry["data_offset"])
            fout.write(scale.tobytes())
            total_out += scale_entry["data_size"]

            if smooth_inv is not None:
                smooth_entry = next(entry_iter)
                assert smooth_entry["kind"] == "smooth_inv"
                fout.seek(smooth_entry["data_offset"])
                fout.write(smooth_inv.tobytes())
                total_out += smooth_entry["data_size"]

                if act_scale is not None:
                    act_scale_entry = next(entry_iter)
                    assert act_scale_entry["kind"] == "act_scale"
                    fout.seek(act_scale_entry["data_offset"])
                    fout.write(act_scale.tobytes())
                    total_out += act_scale_entry["data_size"]

    out_mb = total_out / (1024 * 1024)
    print(f"\nGPTQ calibrated: {calibrated} layers")
    print(f"AWQ-smoothed: {awq_smoothed} layers")
    print(f"Plain INT8 fallback: {quantized_plain} tensors")
    print(f"W8A8 output payload: {out_mb:.1f} MB")
    print(f"Output: {output_bin}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export calibrated W8A8 model")
    parser.add_argument("calib_dir")
    parser.add_argument("input_bin")
    parser.add_argument("output_bin")
    parser.add_argument(
        "--device",
        default=os.environ.get("FISH_GPTQ_DEVICE", "auto"),
        help="Torch device for Hessian/GPTQ math: auto, cpu, cuda, cuda:0 ...",
    )
    args = parser.parse_args()

    configure_torch_device(args.device)
    if torch is not None:
        print(f"Torch device: {TORCH_DEVICE}")
    export_w8a8(args.calib_dir, args.input_bin, args.output_bin)
