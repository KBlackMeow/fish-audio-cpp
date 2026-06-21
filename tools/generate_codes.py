#!/usr/bin/env python3
"""
generate_codes.py — Python bridge for text-to-semantic/codebook generation.

Uses the fish_speech reference DualARTransformer to generate VQ codes from text.
Called by the C++ side to replace the (currently broken) C++ DualAR engine.

Input:  --text       text to synthesize
Output: --codes      path to write int32 codes binary file [B=1, N, T]
                     Same format as /tmp/fish_dac_codes.bin used by decode_audio.py

Usage:
    python3 generate_codes.py \\
        --text     "你好" \\
        --codes    /tmp/fish_dac_codes.bin \\
        --model-dir checkpoints/s2-pro \\
        --max-new-tokens 512 \\
        --temperature 0.7 \\
        --top-p 0.7
"""

import argparse
import struct
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(text: str, model, device: str) -> torch.Tensor:
    """
    Build the prompt tensor  [num_codebooks+1, T]  for the DualAR model.

    Format (chat template):
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        {text}<|im_end|>
        <|im_start|>assistant
        <|voice|>
    """
    tok = model.tokenizer

    # Encode the whole prompt as a flat sequence of token ids
    prompt_text = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|voice|>"
    )
    ids = tok.encode(prompt_text)
    num_codebooks = model.config.num_codebooks

    # Shape: [num_codebooks+1, T]
    # Row 0  = token ids
    # Rows 1..num_codebooks = 0 (pad — no VQ codes in the prompt)
    T = len(ids)
    prompt = torch.zeros(num_codebooks + 1, T, dtype=torch.int, device=device)
    prompt[0] = torch.tensor(ids, dtype=torch.int, device=device)
    return prompt


def generate_codes(
    text: str,
    model_dir: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.7,
    top_k: int = 256,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """Run DualAR inference; return codes tensor [N, T] on CPU."""
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from fish_speech.models.text2semantic.inference import init_model, generate

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    print(f"[generate_codes] Loading model from {model_dir}", file=sys.stderr)
    model, decode_one_token = init_model(
        checkpoint_path=model_dir,
        device=device,
        precision=torch.bfloat16,
        compile=False,
    )

    print(f"[generate_codes] Building prompt for: '{text}'", file=sys.stderr)
    prompt = build_prompt(text, model, device)
    print(f"[generate_codes] Prompt length: {prompt.shape[1]} tokens", file=sys.stderr)

    num_codebooks = model.config.num_codebooks
    audio_masks = torch.zeros(1, dtype=torch.bool, device=device)
    audio_parts = torch.zeros(1, model.config.dim, dtype=torch.bfloat16, device=device)

    print(f"[generate_codes] Generating (max_new_tokens={max_new_tokens})...", file=sys.stderr)
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
    # seq shape: [num_codebooks+1, T_total]
    # Columns 0..prompt_len-1 are the prompt; columns prompt_len.. are generated tokens
    T_prompt = prompt.shape[1]
    generated = seq[:, T_prompt:]  # [num_codebooks+1, T_gen]

    # Row 0 = semantic token ids (text vocab), rows 1..N = codebook indices
    # Discard row 0 (semantic text tokens), keep rows 1..num_codebooks
    codes = generated[1:, :]  # [num_codebooks, T_gen]

    print(f"[generate_codes] Generated {codes.shape[1]} frames × {codes.shape[0]} codebooks", file=sys.stderr)
    print(f"[generate_codes] Code range: [{codes.min().item()}, {codes.max().item()}]", file=sys.stderr)

    return codes.cpu()


def save_codes(codes: torch.Tensor, path: str):
    """Save codes [N, T] as binary file with [B=1, N, T] header."""
    N, T = codes.shape
    data = codes.numpy().astype("int32")
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", 1, N, T))   # B=1, N, T
        f.write(data.tobytes())
    print(f"[generate_codes] Saved codes [1, {N}, {T}] to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="DualAR text-to-codes bridge")
    parser.add_argument("--text",           required=True, help="Text to synthesize")
    parser.add_argument("--codes",          required=True, help="Output codes binary path")
    parser.add_argument("--model-dir",      required=True, help="Path to model checkpoint dir")
    parser.add_argument("--max-new-tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--top-p",          type=float, default=0.7)
    parser.add_argument("--top-k",          type=int,   default=256)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[generate_codes] CUDA unavailable, falling back to CPU", file=sys.stderr)
        device = "cpu"

    codes = generate_codes(
        text=args.text,
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=device,
        seed=args.seed,
    )

    save_codes(codes, args.codes)


if __name__ == "__main__":
    main()
