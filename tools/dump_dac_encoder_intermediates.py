#!/usr/bin/env python3
"""Dump Python DAC encoder tensors for C++ parity checks.

The C++ encoder can dump matching tensors by setting FISH_DUMP_DAC_PREFIX.
This script writes the same binary tensor format used by compare_dac_dumps.py:
int32 ndim, int32 shape[ndim], float32 contiguous payload.
"""

import argparse
import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_wav_mono(path: str) -> tuple[torch.Tensor, int]:
    with wave.open(path, "rb") as f:
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        sample_rate = f.getframerate()
        frames = f.readframes(f.getnframes())

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample_width={sample_width}")

    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(audio).unsqueeze(0), sample_rate


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
    print(f"[dump_dac_encoder] loaded: {result}", file=sys.stderr)
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
    print(f"[dump_dac_encoder] {name}: {shape} -> {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    ap.add_argument(
        "--round-weights-fp16",
        action="store_true",
        help="Round floating parameters/buffers through fp16 while keeping float32 compute.",
    )
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = load_model(args.checkpoint, device)
    if args.round_weights_fp16:
        with torch.no_grad():
            for tensor in list(model.parameters()) + list(model.buffers()):
                if tensor.is_floating_point():
                    tensor.copy_(tensor.half().float())
    if args.dtype == "float16":
        model = model.half()
    elif args.dtype == "bfloat16":
        model = model.bfloat16()
    audio, sample_rate = read_wav_mono(args.audio)
    if sample_rate != model.sample_rate:
        raise ValueError(f"Expected {model.sample_rate} Hz audio, got {sample_rate} Hz")

    captures: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def _hook(_module, _inputs, output):
            captures[name] = output.detach().clone()
        return _hook

    handles = []
    enc_blocks = model.encoder.block
    for idx, name in [(0, "enc_block0"), (1, "enc_block1"), (2, "enc_block2"),
                      (3, "enc_block3"), (4, "enc_block4"), (6, "enc_block6")]:
        if idx < len(enc_blocks):
            handles.append(enc_blocks[idx].register_forward_hook(hook(name)))

    handles.append(model.quantizer.downsample[0].register_forward_hook(hook("enc_downsample0")))
    handles.append(model.quantizer.downsample[1].register_forward_hook(hook("enc_downsample1")))
    for idx, stage in enumerate(model.quantizer.downsample):
        conv, convnext = stage[0], stage[1]
        handles.append(conv.register_forward_hook(hook(f"enc_downsample{idx}_conv")))
        handles.append(convnext.dwconv.register_forward_hook(hook(f"enc_downsample{idx}_dwconv")))
        handles.append(convnext.norm.register_forward_hook(hook(f"enc_downsample{idx}_norm")))
        handles.append(convnext.pwconv1.register_forward_hook(hook(f"enc_downsample{idx}_pwconv1")))
        handles.append(convnext.act.register_forward_hook(hook(f"enc_downsample{idx}_gelu")))
        handles.append(convnext.pwconv2.register_forward_hook(hook(f"enc_downsample{idx}_pwconv2")))
        handles.append(convnext.register_forward_hook(hook(f"enc_downsample{idx}_out")))
    handles.append(model.quantizer.pre_module.register_forward_hook(hook("enc_pre_module")))
    if hasattr(model.quantizer.pre_module, "layers"):
        for idx, layer in enumerate(model.quantizer.pre_module.layers):
            handles.append(layer.register_forward_hook(hook(f"enc_pre_module_layer{idx}")))
            handles.append(layer.attention.register_forward_hook(hook(f"enc_pre_module_layer{idx}_attention")))
            handles.append(layer.feed_forward.register_forward_hook(hook(f"enc_pre_module_layer{idx}_ffn")))
            if idx == 0:
                handles.append(layer.attention_norm.register_forward_hook(hook("enc_pre_module_layer0_attention_norm")))
                handles.append(layer.attention.wqkv.register_forward_hook(hook("enc_pre_module_layer0_wqkv")))

    try:
        with torch.no_grad():
            audio = audio.to(device)
            if args.dtype == "float16":
                audio = audio.half()
            elif args.dtype == "bfloat16":
                audio = audio.bfloat16()
            length = audio.shape[-1]
            right_pad = math.ceil(length / model.frame_length) * model.frame_length - length
            padded = torch.nn.functional.pad(audio.unsqueeze(1), (0, right_pad))
            save_tensor(args.prefix, "input_audio", padded)
            codes, code_lens = model.encode(audio)
            save_tensor(args.prefix, "codes", codes)
            save_tensor(args.prefix, "code_lens", code_lens)
    finally:
        for h in handles:
            h.remove()

    for name in [
        "enc_block0", "enc_block1", "enc_block2", "enc_block3", "enc_block4",
        "enc_block6", "enc_downsample0", "enc_downsample1", "enc_pre_module",
        "enc_pre_module_layer0", "enc_pre_module_layer1", "enc_pre_module_layer2",
        "enc_pre_module_layer3", "enc_pre_module_layer4", "enc_pre_module_layer5",
        "enc_pre_module_layer6", "enc_pre_module_layer7",
    ]:
        if name in captures:
            save_tensor(args.prefix, name, captures[name])

    for idx in range(2):
        for suffix in ("conv", "dwconv", "norm", "pwconv1", "gelu", "pwconv2", "out"):
            name = f"enc_downsample{idx}_{suffix}"
            if name in captures:
                save_tensor(args.prefix, name, captures[name])

    for idx in range(8):
        for suffix in ("attention", "ffn"):
            name = f"enc_pre_module_layer{idx}_{suffix}"
            if name in captures:
                save_tensor(args.prefix, name, captures[name])

    for name in ["enc_pre_module_layer0_attention_norm", "enc_pre_module_layer0_wqkv"]:
        if name in captures:
            save_tensor(args.prefix, name, captures[name])


if __name__ == "__main__":
    main()
