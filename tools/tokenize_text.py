#!/usr/bin/env python3
"""
Thin tokenizer shim called once by fish-server.
Usage:  tokenize_text.py <model_dir> <text>
Output: space-separated token IDs on stdout (plain integers).
"""
import sys
import os

model_dir = sys.argv[1]
text      = sys.argv[2]

# Load directly from tokenizer.json — avoids AutoTokenizer hub lookups.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

tok_file = os.path.join(model_dir, "tokenizer.json")
cfg_file  = os.path.join(model_dir, "tokenizer_config.json")
print(f"[tokenize_text] cwd={os.getcwd()} tok={tok_file} exists={os.path.exists(tok_file)}", file=sys.stderr)

from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(
    tokenizer_file=tok_file,
    tokenizer_config_file=cfg_file,
)

# Fish-speech S2 Pro TTS prompt (matches inference.py generate_long):
# System prompt instructs the model to "convert the provided text to speech",
# NOT to act as a chatbot assistant.
prompt = (
    "<|im_start|>system\n"
    "convert the provided text to speech<|im_end|>\n"
    "<|im_start|>user\n"
    f"{text}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<|voice|>"
)
ids = tok.encode(prompt, add_special_tokens=False)
print(" ".join(map(str, ids)))
