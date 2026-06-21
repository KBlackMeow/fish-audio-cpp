#!/usr/bin/env python3
"""
decode_audio.py — Python bridge for the DAC codec decoder.

Called by DACEngine::decode() after C++ token generation.
Accepts raw VQ codebook indices, runs the full quantizer.decode + convolutional
decoder (model.from_indices), and writes raw float32 PCM.

Input  (--codes):   raw binary file with header [B, N, T] (3×int32) followed
                    by B*N*T int32 values (C-order / row-major).
                    B=batch, N=num_codebooks (10), T=time frames.
Output (--output):  raw binary file of float32 PCM samples (no header).
                    Mono, 44100 Hz (or as specified by --sample-rate).

Usage (called from C++):
    python3 decode_audio.py \\
        --codes     /tmp/fish_dac_codes.bin \\
        --output    /tmp/fish_dac_audio.f32 \\
        --checkpoint checkpoints/s2-pro/codec.pth \\
        --sample-rate 44100
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def load_codes(path: str):
    """Load [B, N, T] int32 codes written by DACEngine::decode()."""
    with open(path, "rb") as f:
        B, N, T = struct.unpack("<3i", f.read(12))
        data = np.frombuffer(f.read(B * N * T * 4), dtype=np.int32).copy()
    codes = torch.from_numpy(data).reshape(B, N, T).long()
    print(f"[decode_audio] codes shape: {list(codes.shape)}  "
          f"range [{codes.min().item()}, {codes.max().item()}]", file=sys.stderr)
    return codes


def load_dac_model(checkpoint_path: str, device: str = "cuda"):
    """Load the DAC codec model from a PyTorch checkpoint."""
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    script_dir = Path(__file__).resolve().parent
    # fish-audio-cpp/tools  →  repo root  →  fish_speech/configs
    repo_root = script_dir.parent.parent
    config_dir = repo_root / "fish_speech" / "configs"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose("modded_dac_vq")

    model = instantiate(cfg)

    state_dict = torch.load(checkpoint_path, map_location="cpu",
                            mmap=True, weights_only=True)
    # Unwrap nested state-dict keys used by lightning / fish-speech training
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any(k.startswith("generator.") for k in state_dict):
        state_dict = {k.replace("generator.", ""): v
                      for k, v in state_dict.items()
                      if k.startswith("generator.")}

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"[decode_audio] Model loaded: {result}", file=sys.stderr)
    model.eval()
    model.to(device)
    return model


def decode_with_model(model, codes: torch.Tensor, device: str = "cuda") -> np.ndarray:
    """Run quantizer.decode + convolutional decoder on codes → PCM float32."""
    codes = codes.to(device)
    with torch.no_grad():
        # from_indices: codes [B, N, T]  →  audio [B, 1, T_audio]
        audio = model.from_indices(codes)
    # squeeze batch + channel dims → [T_audio]
    audio = audio.squeeze(0).squeeze(0)
    return audio.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="DAC codec decoder bridge")
    parser.add_argument("--codes",      required=True,
                        help="Path to int32 codes binary file from C++")
    parser.add_argument("--output",     required=True,
                        help="Path to write raw float32 PCM audio")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to codec.pth checkpoint")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--device",     default="cuda",
                        help="Device (cuda / cpu)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[decode_audio] CUDA unavailable, falling back to CPU", file=sys.stderr)
        device = "cpu"

    print(f"[decode_audio] Loading codes from {args.codes}", file=sys.stderr)
    codes = load_codes(args.codes)

    print(f"[decode_audio] Loading DAC model from {args.checkpoint}", file=sys.stderr)
    model = load_dac_model(args.checkpoint, device=device)

    print("[decode_audio] Running decoder (quantizer.decode + convolutional)...",
          file=sys.stderr)
    audio = decode_with_model(model, codes, device=device)
    duration = audio.shape[0] / args.sample_rate
    print(f"[decode_audio] Audio: {audio.shape[0]} samples  "
          f"({duration:.2f}s @ {args.sample_rate}Hz)", file=sys.stderr)

    audio.astype(np.float32).tofile(args.output)
    print(f"[decode_audio] Wrote {audio.shape[0]} samples to {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
