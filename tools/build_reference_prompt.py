#!/usr/bin/env python3
"""
Generate a complete prompt file with reference audio for voice cloning.

Output format (binary):
    int32: num_codebooks
    int32: prompt_len (T)
    int32[num_codebooks+1][T]: prompt tensor (row-major, row 0 = text tokens)

The C++ pipeline reads this via --prompt-file and skips its own tokenization.

Usage:
    python3 build_reference_prompt.py \
        --text "你好世界" \
        --ref-audio /path/to/speaker.wav \
        --ref-text "参考音频对应的文字" \
        --model-dir checkpoints/s2-pro \
        --output /tmp/prompt.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def load_tokenizer(model_dir: str):
    from fish_speech.tokenizer import FishTokenizer
    return FishTokenizer(model_dir)


def load_codec(checkpoint_path: str, device: str = "cuda"):
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


def encode_ref_audio(audio_path: str, codec_model, device: str = "cuda") -> torch.Tensor:
    """Encode reference audio → VQ codes [num_codebooks, T_codes]."""
    import torchaudio
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav.to(device), sr, codec_model.sample_rate)[0]

    dtype = next(codec_model.parameters()).dtype
    audios = wav[None, None].to(dtype=dtype)
    audio_lengths = torch.tensor([len(wav)], device=device, dtype=torch.long)

    with torch.no_grad():
        indices, feature_lengths = codec_model.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]].cpu()


def main():
    ap = argparse.ArgumentParser(description="Build reference prompt file for C++ pipeline")
    ap.add_argument("--text", required=True, help="Text to synthesize")
    ap.add_argument("--ref-audio", required=True, help="Reference speaker audio file")
    ap.add_argument("--ref-text", required=True, help="Text content of reference audio")
    ap.add_argument("--model-dir", required=True, help="Model checkpoint directory")
    ap.add_argument("--output", required=True, help="Output prompt binary file")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model_dir = Path(args.model_dir)
    codec_path = model_dir / "codec.pth"

    # 1. Load tokenizer
    print("[build_prompt] Loading tokenizer...", file=sys.stderr)
    tok = load_tokenizer(str(model_dir))

    # 2. Load codec and encode reference audio
    print("[build_prompt] Loading codec & encoding reference audio...", file=sys.stderr)
    codec = load_codec(str(codec_path), device)
    ref_codes = encode_ref_audio(args.ref_audio, codec, device)  # [N, T_codes]
    num_codebooks = ref_codes.shape[0]
    print(f"[build_prompt] Ref codes: [{num_codebooks}, {ref_codes.shape[1]}]  "
          f"range [{ref_codes.min().item()}, {ref_codes.max().item()}]", file=sys.stderr)

    # 3. Build prompt with reference audio format (matches inference.py generate_long)
    # The VQ codes go in the system message after "Speech:\n"
    system_prompt = (
        "convert the provided text to speech reference to the following:\n\n"
        "Text:\n"
        f"<|speaker:0|>{args.ref_text}\n\n"
        "Speech:\n"
    )
    prompt_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{args.text}<|im_end|>\n"
        f"<|im_start|>assistant\n<|voice|>"
    )

    # 4. Tokenize — find where VQ codes should be placed
    # The VQ codes replace the "<|im_end|>" after the system message.
    # We need to find the position of the first "<|im_end|>" token.
    sys_part = f"<|im_start|>system\n{system_prompt}<|im_end|>"
    sys_ids = tok.encode(sys_part, add_special_tokens=False)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    # Find the LAST <|im_end|> in the system part (this is where VQ codes go)
    vq_positions = []
    for i, tid in enumerate(sys_ids):
        if tid == im_end_id:
            vq_positions.append(i)

    # The VQ codes go at the position of the LAST <|im_end|> in the system prompt
    # Actually, looking at Python: vq_mask_tokens marks positions where TextPart ends
    # and VQPart begins. The VQPart appears after "Speech:\n" and before <|im_end|>.
    # In the token sequence: [..., "Speech:", "\n", <|im_end|>, <|im_start|>, "user", ...]
    # The VQ codes go AT the <|im_end|> position (replacing it in rows 1..N).
    # Row 0 keeps <|im_end|>, rows 1..N get the VQ codes.

    # Actually, in Python's content_sequence.encode_for_inference:
    #   values[1:, encoded.vq_mask_tokens] = all_vq_codes
    # vq_mask_tokens is True at positions where VQPart appears.
    # The VQPart in the system message is the LAST part, right before <|im_end|>.
    # In the token sequence, the VQPart tokens occupy some positions before <|im_end|>.
    # Wait — VQPart doesn't add text tokens. It only adds VQ codes to rows 1..N
    # at the vq_mask_tokens positions. The text tokens come from the TextParts only.

    # Let me think about this differently. The Python encode method produces:
    #   tokens: all text tokens (from TextParts)
    #   vq_mask_tokens: boolean mask of length len(tokens), True where VQ codes go
    #
    # The VQPart is the last part in the system message. In the token sequence,
    # it's placed AFTER "Speech:\n" and BEFORE the system message's <|im_end|>.
    # But VQPart has NO text tokens — it only contributes VQ codes.
    #
    # So the tokens are: [sys_start, ..., "Speech:", "\n", <|im_end|>, user_start, ...]
    # And vq_mask_tokens is: [False, ..., False, True, True, ..., True, False, ...]
    # where the True positions correspond to the VQ codes.
    #
    # But WHERE exactly? The VQ codes are placed at positions where the VQPart would be
    # if it had text tokens. Looking at content_sequence.encode():
    #   for part in self.parts:
    #     if isinstance(part, VQPart):
    #       # The VQ codes are placed at the CURRENT position in the sequence
    #       # No text tokens are added for VQPart
    #
    # Actually, I think the VQ codes are placed at the positions IMMEDIATELY AFTER
    # the last text token from the previous TextPart. So they go right after "Speech:\n"
    # and before <|im_end|>.
    #
    # For our purposes, let me find the position of "Speech:" in the tokenized output
    # and place the VQ codes right after the newline that follows "Speech:".

    # Full tokenization to find positions
    full_ids = tok.encode(prompt_text, add_special_tokens=False)

    # Find "Speech:" token position
    speech_id = tok.convert_tokens_to_ids("Speech")
    speech_pos = None
    for i, tid in enumerate(full_ids):
        if tid == speech_id:
            speech_pos = i
            break

    # After "Speech:" there should be ":" token and "\n", then VQ codes start
    # Actually "Speech:" might be one token. The "\n" after it is at speech_pos + 1 or +2.
    # VQ codes start at the position of the first <|im_end|> after "Speech:\n"
    # Wait — in Python, the VQPart is placed between TextParts. The system message has:
    #   1. TextPart: "convert the provided text to speech..."
    #   2. TextPart: "{ref_text}\n\nSpeech:\n"
    #   3. VQPart: codes
    #   4. (im_end added by add_im_end=True)
    #
    # In the encoding, the VQ codes occupy positions at the END of the system message,
    # before the <|im_end|>. The positions are determined by where the VQPart sits
    # in the sequence — it doesn't add text tokens, so the positions are "virtual".
    # The vq_mask_tokens marks these virtual positions.

    # Let me take a completely different approach. Instead of figuring out the exact
    # token positions, I'll use the Python content_sequence to build the prompt properly.

    print("[build_prompt] Building prompt via content_sequence...", file=sys.stderr)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from fish_speech.conversation import Conversation, Message
    from fish_speech.content_sequence import TextPart, VQPart

    conv = Conversation()
    conv.append(Message(
        role="system",
        parts=[
            TextPart(text="convert the provided text to speech reference to the following:\n\nText:\n", cal_loss=False),
            TextPart(text=f"<|speaker:0|>{args.ref_text}\n\nSpeech:\n", cal_loss=False),
            VQPart(codes=ref_codes, cal_loss=False),
        ],
        cal_loss=False,
        add_im_start=True,
        add_im_end=True,
    ))
    conv.append(Message(
        role="user",
        parts=[TextPart(text=args.text, cal_loss=False)],
        cal_loss=False,
        add_im_start=True,
        add_im_end=True,
    ))
    conv.append(Message(
        role="assistant",
        parts=[],
        cal_loss=False,
        modality="voice",
        add_im_start=True,
        add_im_end=False,
    ))

    encoded, audio_masks, audio_parts = conv.encode_for_inference(
        tok, num_codebooks=num_codebooks
    )
    # encoded: [num_codebooks+1, T] tensor with text tokens in row 0 and VQ codes in rows 1..N
    print(f"[build_prompt] Prompt tensor: {list(encoded.shape)}", file=sys.stderr)

    # 5. Save as binary
    N = encoded.shape[0] - 1  # num_codebooks
    T = encoded.shape[1]
    data = encoded.numpy().astype(np.int32)
    with open(args.output, "wb") as f:
        f.write(struct.pack("<2i", N, T))  # header: num_codebooks, prompt_len
        f.write(data.tobytes())             # [N+1, T] row-major

    size_kb = Path(args.output).stat().st_size / 1024
    print(f"[build_prompt] Saved [{N}+1, {T}] → {args.output} ({size_kb:.0f} KB)", file=sys.stderr)

    # Print first few semantic tokens from ref for verification
    print(f"[build_prompt] Ref cb0 first 5: {ref_codes[0, :5].tolist()}", file=sys.stderr)


if __name__ == "__main__":
    main()
