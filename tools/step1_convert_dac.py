#!/usr/bin/env python3
"""Convert PyTorch DAC checkpoint to .bin format for fish-audio-cpp."""

import struct
import json
from pathlib import Path

MAGIC = 0x46495348
VERSION = 1
HEADER_FMT = "<256sII8qQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def decompose_weight_norm(state_dict: dict) -> dict:
    """Pre-compute weight_norm and expose the fused weight.

    Handles two PyTorch APIs:
    - Old API  (torch < 2.0): stores ``weight_g`` / ``weight_v`` suffixes.
    - New API  (torch >= 2.0, parametrizations): stores
        ``<base>.parametrizations.weight.original0`` (g) and
        ``<base>.parametrizations.weight.original1`` (v).

    The fused weight is   w = v * (g / ‖v‖₀)
    where ‖v‖₀ means the norm along all dims *except* dim 0
    (consistent with weight_norm default dim=0).
    """
    import torch
    import re

    new_sd: dict = {}
    fused_bases: set = set()

    # --- New PyTorch parametrizations API ---
    param_pattern = re.compile(r'^(.*?)\.parametrizations\.weight\.original([01])$')
    param_bases: dict[str, dict] = {}  # base → {0: g_tensor, 1: v_tensor}
    for name, tensor in state_dict.items():
        m = param_pattern.match(name)
        if m:
            base, idx = m.group(1), int(m.group(2))
            param_bases.setdefault(base, {})[idx] = tensor

    for base, parts in param_bases.items():
        if 0 not in parts or 1 not in parts:
            # Incomplete pair — keep as-is
            continue
        g = parts[0]   # shape [out_channels, 1, 1] (or [out_channels, 1])
        v = parts[1]   # shape [out_channels, in_channels, kernel_size]
        # Norm over all dims except dim 0
        v_flat = v.reshape(v.shape[0], -1)                       # [out, rest]
        norm = torch.linalg.norm(v_flat, dim=1)                  # [out]
        # Broadcast g and norm back to v's shape
        extra_dims = v.dim() - 1
        g_bc   = g.reshape(g.shape[0], *([1] * extra_dims))      # [out, 1, ...]
        norm_bc = norm.reshape(norm.shape[0], *([1] * extra_dims))
        fused = v * (g_bc / norm_bc.clamp(min=1e-8))
        new_sd[base + ".weight"] = fused
        fused_bases.add(base)
        print(f"  Fused weight_norm (parametrizations): {base}.weight → {list(fused.shape)}")

    # --- Old API: weight_g / weight_v suffixes ---
    old_bases = {name[:-2] for name in state_dict if name.endswith("_g")}
    for base in old_bases:
        g = state_dict[f"{base}_g"]
        v = state_dict[f"{base}_v"]
        v_flat = v.reshape(v.shape[0], -1)
        norm = torch.linalg.norm(v_flat, dim=1)
        extra_dims = v.dim() - 1
        g_bc = g.reshape(g.shape[0], *([1] * extra_dims))
        norm_bc = norm.reshape(norm.shape[0], *([1] * extra_dims))
        fused = v * (g_bc / norm_bc.clamp(min=1e-8))
        new_sd[base] = fused
        fused_bases.add(base)
        print(f"  Fused weight_norm (old): {base} → {list(fused.shape)}")

    # Copy through all tensors that are NOT part of a weight-norm pair
    skip_suffixes = ("_g", "_v")
    skip_patterns = set()
    for base in {k for k in param_bases}:
        skip_patterns.add(base + ".parametrizations.weight.original0")
        skip_patterns.add(base + ".parametrizations.weight.original1")

    for name, tensor in state_dict.items():
        if name in skip_patterns:
            continue
        if any(name.endswith(s) for s in skip_suffixes):
            continue
        if name not in new_sd:
            new_sd[name] = tensor

    n_fused = len(fused_bases)
    if n_fused == 0:
        print("  Warning: no weight_norm pairs found — weights copied as-is")
    else:
        print(f"  Fused {n_fused} weight_norm weight(s) total")
    return new_sd


def convert(state_dict: dict, output_path: str) -> None:
    import torch

    dtype_map = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}

    sd = decompose_weight_norm(state_dict)

    tensors = {}
    skipped = 0
    for name, tensor in sd.items():
        # Skip non-weight tensors (bool masks, int indices, etc.)
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            skipped += 1
            continue
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)
        elif tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)

        dtype_val = dtype_map.get(tensor.dtype)
        if dtype_val is None:
            raise ValueError(f"Unsupported dtype {tensor.dtype} for {name}")

        tensors[name] = {
            "data": tensor.contiguous().cpu().numpy().tobytes(),
            "dtype": dtype_val,
            "shape": list(tensor.shape),
        }

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(tensors)))

        header_end = 12 + len(tensors) * HEADER_SIZE
        data_start = ((header_end + 255) // 256) * 256

        offset = data_start
        for name in sorted(tensors.keys()):
            info = tensors[name]
            padded_name = name.encode("utf-8")[:255].ljust(256, b"\x00")
            shape_padded = list(info["shape"]) + [0] * (8 - len(info["shape"]))
            f.write(struct.pack(HEADER_FMT,
                padded_name, info["dtype"], len(info["shape"]),
                *shape_padded, offset, len(info["data"])))
            offset += len(info["data"])
            offset = ((offset + 255) // 256) * 256

        while f.tell() < data_start:
            f.write(b"\x00")

        for name in sorted(tensors.keys()):
            info = tensors[name]
            f.write(info["data"])
            remainder = len(info["data"]) % 256
            if remainder:
                f.write(b"\x00" * (256 - remainder))

    print(f"Converted {len(tensors)} tensors → {output_path}")
    if skipped:
        print(f"  Skipped {skipped} non-weight tensors (bool, int, etc.)")
    print(f"  Size: {Path(output_path).stat().st_size / (1024**3):.2f} GB")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert DAC to .bin")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DAC .pth checkpoint")
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    import torch

    ckpt = Path(args.checkpoint)
    sd = torch.load(str(ckpt), map_location="cpu", mmap=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if any("generator" in k for k in sd):
        sd = {k.replace("generator.", ""): v for k, v in sd.items() if "generator." in k}

    # Write DAC config
    config = {
        "sample_rate": 44100, "block_size": 2048, "n_layer": 8, "n_head": 8,
        "dim": 512, "intermediate_size": 1536, "head_dim": 64,
        "rope_base": 10000.0, "norm_eps": 1e-5,
        "codebook_size": 1024, "num_codebooks": 10, "latent_dim": 1024,
        "channels_first": True, "pos_embed_type": "rope",
        "encoder_rates": [2, 4, 8, 8], "decoder_rates": [8, 8, 4, 2]
    }
    out_dir = Path(args.output_dir)
    with open(out_dir / "dac_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote config → {out_dir / 'dac_config.json'}")

    convert(sd, str(out_dir / "dac.bin"))


if __name__ == "__main__":
    main()
