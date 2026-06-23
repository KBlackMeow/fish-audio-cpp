#!/usr/bin/env python3
"""Convert a PyTorch checkpoint to FP16 `.bin` for fish-audio-cpp.

Supported models:
  - `dual_ar`
  - `dac`
"""

import json
import shutil
import struct
from pathlib import Path

MAGIC = 0x46495348
VERSION = 1
HEADER_FMT = "<256sII8qQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def write_bin_from_state_dict(state_dict: dict, output_path: Path) -> None:
    import torch

    dtype_map = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }

    tensors = {}
    skipped = 0
    for name, tensor in state_dict.items():
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            skipped += 1
            continue
        if tensor.dtype in (torch.float32, torch.bfloat16):
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
            if len(info["shape"]) > 8:
                raise ValueError(f"Tensor {name} has {len(info['shape'])} > 8 dims")

            padded_name = name.encode("utf-8")[:255].ljust(256, b"\x00")
            shape_padded = list(info["shape"]) + [0] * (8 - len(info["shape"]))
            f.write(struct.pack(
                HEADER_FMT,
                padded_name,
                info["dtype"],
                len(info["shape"]),
                *shape_padded,
                offset,
                len(info["data"]),
            ))
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

    size_gb = output_path.stat().st_size / (1024 ** 3)
    print(f"Converted {len(tensors)} tensors → {output_path} ({size_gb:.2f} GB)")
    if skipped:
        print(f"  Skipped {skipped} non-floating tensors")


def load_dual_ar_state_dict(checkpoint: Path) -> dict:
    import torch

    if checkpoint.is_dir():
        try:
            from safetensors.torch import load_file
            sd = {}
            for sf in sorted(checkpoint.glob("*.safetensors")):
                sd.update(load_file(str(sf)))
            print(f"Loaded {len(sd)} keys from safetensors in {checkpoint}")
            return sd
        except ImportError:
            print("safetensors not installed, trying torch.load")
            return torch.load(
                str(checkpoint / "pytorch_model.bin"),
                map_location="cpu",
                weights_only=True,
            )

    sd = torch.load(str(checkpoint), map_location="cpu", mmap=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    elif "model" in sd:
        sd = sd["model"]
    return sd


def write_dual_ar_config(checkpoint: Path, output_dir: Path) -> None:
    config_src = checkpoint / "config.json" if checkpoint.is_dir() else checkpoint.parent / "config.json"
    config_dst = output_dir / "dual_ar_config.json"
    if config_src.exists():
        shutil.copy(config_src, config_dst)
        print(f"Copied config → {config_dst}")
        return

    default_config = {
        "vocab_size": 32000, "n_layer": 32, "n_head": 32, "dim": 4096,
        "intermediate_size": 14336, "n_local_heads": 32, "head_dim": 64,
        "rope_base": 10000.0, "norm_eps": 1e-5, "max_seq_len": 2048,
        "codebook_size": 160, "num_codebooks": 4,
        "semantic_begin_id": 128256, "semantic_end_id": 128415,
    }
    with open(config_dst, "w") as f:
        json.dump(default_config, f, indent=2)
    print(f"Wrote default config → {config_dst}")


def decompose_weight_norm(state_dict: dict) -> dict:
    import re
    import torch

    new_sd = {}
    fused_bases = set()

    param_pattern = re.compile(r"^(.*?)\.parametrizations\.weight\.original([01])$")
    param_bases = {}
    for name, tensor in state_dict.items():
        m = param_pattern.match(name)
        if m:
            base, idx = m.group(1), int(m.group(2))
            param_bases.setdefault(base, {})[idx] = tensor

    for base, parts in param_bases.items():
        if 0 not in parts or 1 not in parts:
            continue
        g = parts[0]
        v = parts[1]
        v_flat = v.reshape(v.shape[0], -1)
        norm = torch.linalg.norm(v_flat, dim=1)
        extra_dims = v.dim() - 1
        g_bc = g.reshape(g.shape[0], *([1] * extra_dims))
        norm_bc = norm.reshape(norm.shape[0], *([1] * extra_dims))
        fused = v * (g_bc / norm_bc.clamp(min=1e-8))
        new_sd[base + ".weight"] = fused
        fused_bases.add(base)
        print(f"  Fused weight_norm (parametrizations): {base}.weight → {list(fused.shape)}")

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

    skip_suffixes = ("_g", "_v")
    skip_patterns = set()
    for base in param_bases:
        skip_patterns.add(base + ".parametrizations.weight.original0")
        skip_patterns.add(base + ".parametrizations.weight.original1")

    for name, tensor in state_dict.items():
        if name in skip_patterns:
            continue
        if any(name.endswith(s) for s in skip_suffixes):
            continue
        if name not in new_sd:
            new_sd[name] = tensor

    if not fused_bases:
        print("  Warning: no weight_norm pairs found — weights copied as-is")
    else:
        print(f"  Fused {len(fused_bases)} weight_norm weight(s) total")
    return new_sd


def load_dac_state_dict(checkpoint: Path) -> dict:
    import torch

    sd = torch.load(str(checkpoint), map_location="cpu", mmap=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if any("generator" in k for k in sd):
        sd = {k.replace("generator.", ""): v for k, v in sd.items() if "generator." in k}
    return decompose_weight_norm(sd)


def write_dac_config(output_dir: Path) -> None:
    config = {
        "sample_rate": 44100, "block_size": 2048, "n_layer": 8, "n_head": 8,
        "dim": 512, "intermediate_size": 1536, "head_dim": 64,
        "rope_base": 10000.0, "norm_eps": 1e-5,
        "codebook_size": 1024, "num_codebooks": 10, "latent_dim": 1024,
        "channels_first": True, "pos_embed_type": "rope",
        "encoder_rates": [2, 4, 8, 8], "decoder_rates": [8, 8, 4, 2],
    }
    config_dst = output_dir / "dac_config.json"
    with open(config_dst, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote config → {config_dst}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert model checkpoint to FP16 .bin")
    parser.add_argument("--model", choices=["dual_ar", "dac"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "dual_ar":
        state_dict = load_dual_ar_state_dict(checkpoint)
        write_dual_ar_config(checkpoint, output_dir)
        write_bin_from_state_dict(state_dict, output_dir / "dual_ar.bin")
        return

    state_dict = load_dac_state_dict(checkpoint)
    write_dac_config(output_dir)
    write_bin_from_state_dict(state_dict, output_dir / "dac.bin")


if __name__ == "__main__":
    main()
