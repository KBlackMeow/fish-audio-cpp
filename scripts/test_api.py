#!/usr/bin/env python3
"""Stream TTS with reference audio, play it, then exit.

Usage:
    python scripts/test_api.py [--host HOST] [--port PORT] [--text TEXT]

Requires:
    - requests
    - ffplay
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


DEFAULT_TEXT = (
    "你好，world! 今天我们用中文、English、日本語一起测试流式参考音频语音合成。"
)
DEFAULT_REF_WAV = "example/vo_LLZAQ001_4_nahida_03.wav"
DEFAULT_REF_TEXT = "example/vo_LLZAQ001_4_nahida_03.lab"


def load_ref_audio_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_ref_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def stream_tts_with_ref(
    base_url: str,
    text: str,
    ref_wav: str,
    ref_text_path: str,
    play_audio: bool = True,
    max_new_tokens: int = 300,
    chunk_length: int = 0,
    history_frames: int = 96,
    playback_buffer_ms: int = 500,
) -> None:
    payload = {
        "text": text,
        "ref_audio": load_ref_audio_b64(ref_wav),
        "ref_audio_format": "wav",
        "ref_text": load_ref_text(ref_text_path),
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 42,
        "chunk_length": chunk_length,
        "history_frames": history_frames,
    }

    print(f"[INFO] Reference audio: {ref_wav}")
    print(f"[INFO] Reference text: {ref_text_path}")
    print("[INFO] Starting streaming synthesis...")
    request_start = time.perf_counter()

    response = requests.post(
        f"{base_url}/v1/tts/with-ref/stream",
        json=payload,
        stream=True,
        timeout=300,
    )
    response.raise_for_status()

    ffplay_proc = None
    sample_rate = 44100
    received_audio = False
    done = False
    error_message = None
    first_audio_ms = None
    chunk_count = 0
    profiling = None
    pending_audio = bytearray()
    playback_started = False
    buffered_ms = 0.0

    def ensure_player(cur_sample_rate: int) -> None:
        nonlocal ffplay_proc
        if ffplay_proc is not None:
            return
        ffplay_proc = subprocess.Popen(
            [
                "ffplay",
                "-autoexit",
                "-nodisp",
                "-loglevel",
                "quiet",
                "-f",
                "f32le",
                "-ar",
                str(cur_sample_rate),
                "-ac",
                "1",
                "-i",
                "-",
            ],
            stdin=subprocess.PIPE,
        )

    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            event = json.loads(line[6:])
            event_type = event.get("type")

            if event_type == "audio":
                raw = base64.b64decode(event["data"])
                chunk_count += 1
                sample_rate = event.get("sample_rate", sample_rate)
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - request_start) * 1000.0
                    print(f"[INFO] First audio chunk at {first_audio_ms:.1f} ms")
                if play_audio:
                    pending_audio.extend(raw)
                    buffered_ms = (len(pending_audio) / 4.0) * 1000.0 / sample_rate
                    if not playback_started and buffered_ms >= playback_buffer_ms:
                        ensure_player(sample_rate)
                        if ffplay_proc is not None and ffplay_proc.stdin is None:
                            raise RuntimeError("ffplay stdin is not available")
                        if ffplay_proc is not None and ffplay_proc.stdin is not None:
                            ffplay_proc.stdin.write(pending_audio)
                        pending_audio.clear()
                        playback_started = True
                    elif playback_started:
                        if ffplay_proc is not None and ffplay_proc.stdin is None:
                            raise RuntimeError("ffplay stdin is not available")
                        if ffplay_proc is not None and ffplay_proc.stdin is not None:
                            ffplay_proc.stdin.write(raw)
                received_audio = True
            elif event_type == "done":
                done = True
                profiling = event.get("profiling")
                print(f"[INFO] Stream finished: {event.get('total_samples', 0)} samples")
            elif event_type == "error":
                error_message = event.get("message", "unknown error")
                break
    finally:
        if play_audio and pending_audio and not playback_started:
            ensure_player(sample_rate)
            if ffplay_proc is not None and ffplay_proc.stdin is not None:
                ffplay_proc.stdin.write(pending_audio)
            pending_audio.clear()
        if ffplay_proc and ffplay_proc.stdin:
            ffplay_proc.stdin.close()
        if ffplay_proc:
            ffplay_proc.wait()
        response.close()

    if error_message:
        raise RuntimeError(f"Streaming error: {error_message}")
    if not received_audio:
        raise RuntimeError("No audio received from stream")
    if not done:
        raise RuntimeError("Stream ended without a done event")

    total_ms = (time.perf_counter() - request_start) * 1000.0
    print(
        "[METRIC] client total_ms={:.1f} first_audio_ms={} chunks={}".format(
            total_ms,
            "n/a" if first_audio_ms is None else f"{first_audio_ms:.1f}",
            chunk_count,
        )
    )
    if profiling:
        print("[METRIC] server profiling:")
        for key, value in profiling.items():
            print(f"  - {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream and play TTS with reference audio"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", default=8080, type=int, help="Server port")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--ref-wav", default=DEFAULT_REF_WAV, help="Reference WAV path")
    parser.add_argument(
        "--ref-text",
        default=DEFAULT_REF_TEXT,
        help="Reference transcript text file path",
    )
    parser.add_argument("--repeat", default=1, type=int, help="Repeat requests")
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not play audio with ffplay",
    )
    parser.add_argument(
        "--max-new-tokens",
        default=300,
        type=int,
        help="Generation limit",
    )
    parser.add_argument(
        "--chunk-length",
        default=120,
        type=int,
        help="Target UTF-8 bytes per streaming text chunk; 0 disables chunking",
    )
    parser.add_argument(
        "--history-frames",
        default=96,
        type=int,
        help="How many generated VQ frames to carry into the next text chunk",
    )
    parser.add_argument(
        "--playback-buffer-ms",
        default=500,
        type=int,
        help="How much audio to buffer before local playback starts",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ref_wav):
        raise FileNotFoundError(f"Reference WAV not found: {args.ref_wav}")
    if not os.path.exists(args.ref_text):
        raise FileNotFoundError(f"Reference text not found: {args.ref_text}")

    base_url = f"http://{args.host}:{args.port}"
    for idx in range(args.repeat):
        print(f"[INFO] Run {idx + 1}/{args.repeat}")
        stream_tts_with_ref(
            base_url,
            args.text,
            args.ref_wav,
            args.ref_text,
            play_audio=not args.no_play,
            max_new_tokens=args.max_new_tokens,
            chunk_length=args.chunk_length,
            history_frames=args.history_frames,
            playback_buffer_ms=args.playback_buffer_ms,
        )


if __name__ == "__main__":
    main()
