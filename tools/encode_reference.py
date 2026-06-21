#!/usr/bin/env python3
"""
Encode a reference audio file → VQ codes using the DAC codec model.
Output: binary file with header [num_codebooks, T] (2×int32) followed by
        num_codebooks * T int32 values (row-major).

Usage:
    python3 encode_reference.py \
        --audio /path/to/ref.wav \
        --checkpoint checkpoints/s2-pro/codec.pth \
        --output /tmp/ref_codes.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


def load_codec(checkpoint_path: str, device: str = "cuda"):
    """Load the DAC codec model."""
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    config_dir = repo_root / "fish_speech" / "configs"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose("modded_dac_vq")

    model = instantiate(cfg)
    sd = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("generator.") for k in sd):
        sd = {k.replace("generator.", ""): v for k, v in sd.items()
              if k.startswith("generator.")}
    model.load_state_dict(sd, strict=False, assign=True)
    model.eval()
    model.to(device)
    return model


def encode_audio(audio_path: str, model, device: str = "cuda") -> torch.Tensor:
    """Encode audio file → VQ codes [num_codebooks, T]."""
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav.to(device), sr, model.sample_rate)[0]

    model_dtype = next(model.parameters()).dtype
    audios = wav[None, None].to(dtype=model_dtype)
    audio_lengths = torch.tensor([len(wav)], device=device, dtype=torch.long)

    with torch.no_grad():
        indices, feature_lengths = model.encode(audios, audio_lengths)
    codes = indices[0, :, : feature_lengths[0]]  # [num_codebooks, T]
    print(f"[encode_reference] Audio: {len(wav)} samples → codes [{codes.shape[0]}, {codes.shape[1]}]",
          file=sys.stderr)
    print(f"[encode_reference] Code range: [{codes.min().item()}, {codes.max().item()}]",
          file=sys.stderr)
    return codes.cpu()


def save_codes(codes: torch.Tensor, path: str):
    """Save [num_codebooks, T] codes as binary with [N, T] header."""
    N, T = codes.shape
    data = codes.numpy().astype(np.int32)
    with open(path, "wb") as f:
        f.write(struct.pack("<2i", N, T))
        f.write(data.tobytes())
    print(f"[encode_reference] Saved [{N}, {T}] → {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Encode reference audio → VQ codes")
    ap.add_argument("--audio", required=True, help="Reference audio file (.wav, .mp3, etc.)")
    ap.add_argument("--checkpoint", required=True, help="Path to codec.pth")
    ap.add_argument("--output", required=True, help="Output binary file path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"[encode_reference] Loading codec from {args.checkpoint}", file=sys.stderr)
    model = load_codec(args.checkpoint, device)
    codes = encode_audio(args.audio, model, device)
    save_codes(codes, args.output)


if __name__ == "__main__":
    main()
