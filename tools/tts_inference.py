#!/usr/bin/env python3
"""
tts_inference.py — 单进程全流程 TTS:
  text → (DualAR) → VQ codes → (DAC) → PCM → WAV

在同一进程内顺序完成，DualAR 推理结束后立即释放显存再加载 DAC，
避免两个大模型同时驻留 GPU/RAM 导致 WSL OOM。
"""

import argparse
import struct
import sys
import gc
from pathlib import Path

import numpy as np
import torch
import soundfile as sf


# ── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent          # fish-audio-cpp/tools → repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: text → VQ codes  (DualARTransformer)
# ═════════════════════════════════════════════════════════════════════════════

def build_prompt(text: str, model, device: str) -> torch.Tensor:
    tok = model.tokenizer
    prompt_text = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|voice|>"
    )
    ids = tok.encode(prompt_text)
    num_cb = model.config.num_codebooks
    T = len(ids)
    prompt = torch.zeros(num_cb + 1, T, dtype=torch.int, device=device)
    prompt[0] = torch.tensor(ids, dtype=torch.int, device=device)
    return prompt


def run_dualar(text: str, model_dir: str, max_new_tokens: int,
               temperature: float, top_p: float, top_k: int,
               device: str, seed: int) -> torch.Tensor:
    from fish_speech.models.text2semantic.inference import init_model, generate

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    print("[tts] Loading DualAR model …", file=sys.stderr)
    model, decode_one_token = init_model(
        checkpoint_path=model_dir,
        device=device,
        precision=torch.bfloat16,
        compile=False,
    )

    prompt = build_prompt(text, model, device)
    print(f"[tts] Prompt: {prompt.shape[1]} tokens  →  generating …", file=sys.stderr)

    num_cb = model.config.num_codebooks
    audio_masks = torch.zeros(1, dtype=torch.bool, device=device)
    audio_parts = torch.zeros(1, model.config.dim, dtype=torch.bfloat16, device=device)

    seq = generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    T_prompt = prompt.shape[1]
    generated = seq[:, T_prompt:]          # [num_cb+1, T_gen]
    codes = generated[1:, :].cpu()         # [num_cb, T_gen]  — drop row-0 (text sem tokens)

    print(f"[tts] Generated {codes.shape[1]} frames × {codes.shape[0]} codebooks  "
          f"range [{codes.min().item()}, {codes.max().item()}]", file=sys.stderr)

    # ── release DualAR from GPU ──────────────────────────────────────────────
    del model, seq, generated, prompt, audio_masks, audio_parts
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[tts] DualAR released from GPU", file=sys.stderr)

    return codes   # CPU tensor


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: VQ codes → audio  (DAC model.from_indices)
# ═════════════════════════════════════════════════════════════════════════════

def load_dac(checkpoint_path: str, device: str):
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
        sd = {k.replace("generator.", ""): v for k, v in sd.items()
              if k.startswith("generator.")}

    result = model.load_state_dict(sd, strict=False, assign=True)
    print(f"[tts] DAC loaded: {result}", file=sys.stderr)
    model.eval()
    model.to(device)
    return model


def run_dac(codes: torch.Tensor, codec_path: str, device: str) -> np.ndarray:
    print("[tts] Loading DAC model …", file=sys.stderr)
    model = load_dac(codec_path, device)

    codes_gpu = codes.to(device).long().unsqueeze(0)   # [1, N, T]
    print(f"[tts] DAC decode  codes {list(codes_gpu.shape)} …", file=sys.stderr)

    with torch.no_grad():
        audio = model.from_indices(codes_gpu)           # [1, 1, T_audio]

    audio_np = audio.squeeze().float().cpu().numpy()
    print(f"[tts] Audio: {audio_np.shape[0]} samples  "
          f"({audio_np.shape[0]/44100:.2f}s)", file=sys.stderr)

    del model, audio, codes_gpu
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return audio_np


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Fish S2 Pro TTS single-process pipeline")
    ap.add_argument("--text",           required=True)
    ap.add_argument("--model-dir",      required=True)
    ap.add_argument("--output",         default="output/speech.wav")
    ap.add_argument("--max-new-tokens", type=int,   default=512)
    ap.add_argument("--temperature",    type=float, default=0.7)
    ap.add_argument("--top-p",          type=float, default=0.7)
    ap.add_argument("--top-k",          type=int,   default=256)
    ap.add_argument("--device",         default="cuda")
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[tts] CUDA unavailable, using CPU", file=sys.stderr)
        device = "cpu"

    model_dir   = args.model_dir
    codec_path  = str(Path(model_dir) / "codec.pth")
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Step 1
    codes = run_dualar(
        text=args.text,
        model_dir=model_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=device,
        seed=args.seed,
    )

    # Step 2
    audio = run_dac(codes, codec_path, device)

    # Step 3: write WAV
    sf.write(output_path, audio, 44100)
    print(f"[tts] Saved: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
