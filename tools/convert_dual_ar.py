#!/usr/bin/env python3
"""Convert PyTorch DualARTransformer checkpoint to .bin format for fish-audio-cpp."""

import struct
import json
import sys
from pathlib import Path

MAGIC = 0x46495348  # "FISH"
VERSION = 1

HEADER_FMT = "<256sII8qQQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 344 bytes


def convert(state_dict: dict, output_path: str) -> None:
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
        # Header
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(tensors)))

        # Compute data start offset = 12 + N*344, aligned to 256
        header_end = 12 + len(tensors) * HEADER_SIZE
        data_start = ((header_end + 255) // 256) * 256

        # Write tensor headers
        offset = data_start
        for name in sorted(tensors.keys()):
            info = tensors[name]
            if len(info["shape"]) > 8:
                raise ValueError(f"Tensor {name} has {len(info['shape'])} > 8 dims")

            padded_name = name.encode("utf-8")[:255].ljust(256, b"\x00")
            shape_padded = list(info["shape"]) + [0] * (8 - len(info["shape"]))

            f.write(struct.pack(HEADER_FMT,
                padded_name,
                info["dtype"],
                len(info["shape"]),
                *shape_padded,
                offset,
                len(info["data"]),
            ))
            offset += len(info["data"])
            offset = ((offset + 255) // 256) * 256  # align to 256

        # Pad header to data_start
        while f.tell() < data_start:
            f.write(b"\x00")

        # Write tensor data
        for name in sorted(tensors.keys()):
            info = tensors[name]
            f.write(info["data"])
            # Pad to 256B alignment
            remainder = len(info["data"]) % 256
            if remainder:
                f.write(b"\x00" * (256 - remainder))

    size_gb = Path(output_path).stat().st_size / (1024 ** 3)
    print(f"Converted {len(tensors)} tensors → {output_path} ({size_gb:.2f} GB)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert DualARTransformer to .bin")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PyTorch checkpoint (.pth) or HF model directory")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for .bin and config files")
    args = parser.parse_args()

    import torch

    checkpoint = Path(args.checkpoint)

    if checkpoint.is_dir():
        # HuggingFace safetensors format
        try:
            from safetensors.torch import load_file
            sd = {}
            for sf in sorted(checkpoint.glob("*.safetensors")):
                sd.update(load_file(str(sf)))
            print(f"Loaded {len(sd)} keys from safetensors in {checkpoint}")
        except ImportError:
            print("safetensors not installed, trying torch.load")
            # Fall back to pytorch_model.bin
            sd = torch.load(str(checkpoint / "pytorch_model.bin"),
                            map_location="cpu", weights_only=True)
    else:
        sd = torch.load(str(checkpoint), map_location="cpu", mmap=True)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]

    # Copy/save config
    config_src = checkpoint / "config.json" if checkpoint.is_dir() else checkpoint.parent / "config.json"
    config_dst = Path(args.output_dir) / "dual_ar_config.json"
    if config_src.exists():
        import shutil
        shutil.copy(config_src, config_dst)
        print(f"Copied config → {config_dst}")
    else:
        # Write default config
        default_config = {
            "vocab_size": 32000, "n_layer": 32, "n_head": 32, "dim": 4096,
            "intermediate_size": 14336, "n_local_heads": 32, "head_dim": 64,
            "rope_base": 10000.0, "norm_eps": 1e-5, "max_seq_len": 2048,
            "codebook_size": 160, "num_codebooks": 4,
            "semantic_begin_id": 128256, "semantic_end_id": 128415
        }
        with open(config_dst, "w") as f:
            json.dump(default_config, f, indent=2)
        print(f"Wrote default config → {config_dst}")

    output = Path(args.output_dir) / "dual_ar.bin"
    convert(sd, str(output))


if __name__ == "__main__":
    main()
