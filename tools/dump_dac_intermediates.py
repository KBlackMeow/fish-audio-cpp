#!/usr/bin/env python3
"""Dump Python DAC intermediate tensors for comparison with C++ DAC."""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_codes(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        b, n, t = struct.unpack("<3i", f.read(12))
        data = np.frombuffer(f.read(b * n * t * 4), dtype=np.int32).copy()
    return torch.from_numpy(data).reshape(b, n, t).long()


def load_model(checkpoint_path: str, device: str):
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = REPO_ROOT / "fish_speech" / "configs"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose("modded_dac_vq")

    model = instantiate(cfg)
    sd = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("generator.") for k in sd):
        sd = {k.replace("generator.", ""): v for k, v in sd.items() if k.startswith("generator.")}
    result = model.load_state_dict(sd, strict=False, assign=True)
    print(f"[dump_dac] loaded: {result}", file=sys.stderr)
    model.eval().to(device)
    return model


def save_tensor(prefix: str, name: str, x: torch.Tensor):
    x = x.detach().float().cpu().contiguous()
    path = f"{prefix}_{name}.bin"
    shape = list(x.shape)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(shape)))
        f.write(struct.pack("<" + "i" * len(shape), *shape))
        f.write(x.numpy().astype(np.float32).tobytes())
    print(f"[dump_dac] {name}: {shape} -> {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    codes = load_codes(args.codes).to(device)
    model = load_model(args.checkpoint, device)

    with torch.no_grad():
        q = model.quantizer
        codes[:, 0] = torch.clamp(codes[:, 0], max=q.semantic_quantizer.codebook_size - 1)
        codes[:, 1:] = torch.clamp(codes[:, 1:], max=q.quantizer.codebook_size - 1)

        z_sem = q.semantic_quantizer.from_codes(codes[:, :1])[0]
        z_res = q.quantizer.from_codes(codes[:, 1:])[0]
        z = z_sem + z_res
        save_tensor(args.prefix, "rvq", z)

        z = q.post_module(z)
        save_tensor(args.prefix, "post", z)

        z = q.upsample(z)
        save_tensor(args.prefix, "upsample", z)

        # Dump pre-tanh final conv output by replaying decoder.model manually.
        x = z
        for layer in model.decoder.model[:-1]:
            x = layer(x)
        save_tensor(args.prefix, "pre_tanh", x)
        audio = model.decoder.model[-1](x)
        save_tensor(args.prefix, "audio", audio)


if __name__ == "__main__":
    main()
