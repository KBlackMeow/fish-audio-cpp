#!/usr/bin/env python3
"""Cross-check DAC RVQ parity by quantizing a dumped C++ latent in Python."""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dump_dac_encoder_intermediates import load_model  # noqa: E402


def load_f32_tensor(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = struct.unpack("<" + "i" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.float32).copy().reshape(shape)


def load_cpp_codes(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        n_codebooks, code_len = struct.unpack("<ii", f.read(8))
        return np.frombuffer(f.read(), dtype=np.int32).copy().reshape(n_codebooks, code_len)


def load_py_codes_dump(path: Path) -> np.ndarray:
    arr = load_f32_tensor(path).astype(np.int32)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def print_compare(label: str, a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        print(f"{label}: shape mismatch a={a.shape} b={b.shape}")
        return
    diff = a != b
    per_codebook = [int(diff[i].sum()) for i in range(diff.shape[0])]
    print(
        f"{label}: diff={int(diff.sum())}/{diff.size} "
        f"match={1.0 - float(diff.sum()) / diff.size:.6f} "
        f"per_codebook={','.join(map(str, per_codebook))}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cpp-latent", required=True, help="C++ enc_pre_module dump")
    ap.add_argument("--cpp-codes", required=True)
    ap.add_argument("--py-codes", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_model(args.checkpoint, device)
    z = torch.from_numpy(load_f32_tensor(Path(args.cpp_latent))).to(device)

    with torch.no_grad():
        semantic_z, semantic_codes, *_ = model.quantizer.semantic_quantizer(z)
        residual = z - semantic_z
        _, acoustic_codes, *_ = model.quantizer.quantizer(residual)
        py_on_cpp_latent = torch.cat([semantic_codes, acoustic_codes], dim=1)

    cross_codes = py_on_cpp_latent.detach().cpu().numpy().astype(np.int32)[0]
    cpp_codes = load_cpp_codes(Path(args.cpp_codes))
    py_codes = load_py_codes_dump(Path(args.py_codes))

    print_compare("python_quantizer(cpp_latent) vs python_codes", cross_codes, py_codes)
    print_compare("python_quantizer(cpp_latent) vs cpp_codes", cross_codes, cpp_codes)
    print_compare("cpp_codes vs python_codes", cpp_codes, py_codes)


if __name__ == "__main__":
    main()
