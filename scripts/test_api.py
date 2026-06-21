#!/usr/bin/env python3
"""Test script for fish-audio-cpp REST API.

Usage:
    python scripts/test_api.py [--host HOST] [--port PORT]

Requires: requests (pip install requests)
Starts by checking health, then runs non-streaming and streaming TTS tests.
"""

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import wave

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


def test_health(base_url: str) -> bool:
    """GET /health — should return {"status": "ok"}"""
    print("[TEST] Health check...", end=" ")
    r = requests.get(f"{base_url}/health", timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok", f"Expected status=ok, got {data}"
    print("PASS")
    return True


def test_info(base_url: str) -> bool:
    """GET /v1/info — should return engine info"""
    print("[TEST] Server info...", end=" ")
    r = requests.get(f"{base_url}/v1/info", timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert "engine" in data, f"Missing 'engine' field: {data}"
    assert data.get("backend") == "CUDA", f"Expected backend=CUDA, got {data}"
    print(f"PASS (engine={data['engine']}, version={data.get('version','?')})")
    return True


def test_non_streaming(base_url: str, output_dir: str) -> bool:
    """POST /v1/tts — non-streaming TTS, save WAV"""
    print("[TEST] Non-streaming TTS...", end=" ")
    payload = {
        "text": "你好世界",
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 42,
    }
    r = requests.post(f"{base_url}/v1/tts", json=payload, timeout=120)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()

    assert "audio" in data, f"Missing 'audio' field: {data.keys()}"
    assert "sample_rate" in data, "Missing 'sample_rate'"
    assert data["num_samples"] > 0, f"Expected num_samples > 0, got {data['num_samples']}"

    # Decode base64 → float32 PCM → 16-bit WAV
    raw = base64.b64decode(data["audio"])
    num_samples = len(raw) // 4  # float32 = 4 bytes
    samples = struct.unpack(f"{num_samples}f", raw)

    wav_path = os.path.join(output_dir, "test_non_streaming.wav")
    _write_wav(wav_path, samples, data["sample_rate"])

    dur = data["num_samples"] / data["sample_rate"]
    print(f"PASS ({num_samples} samples, {dur:.2f}s) → {wav_path}")
    return True


def test_streaming(base_url: str, output_dir: str, play: bool = False) -> bool:
    """POST /v1/tts/stream — SSE streaming TTS, optionally play in real-time"""
    print("[TEST] Streaming TTS...")
    payload = {
        "text": "今天天气真不错，适合出门散步。",
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 123,
    }

    r = requests.post(f"{base_url}/v1/tts/stream", json=payload,
                      stream=True, timeout=300)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    progress_count = 0
    audio_chunks = []
    sample_rate = 44100
    done = False
    error = None
    ffplay_proc = None

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if not line.startswith("data: "):
            continue

        ev = json.loads(line[6:])
        ev_type = ev.get("type", "?")

        if ev_type == "progress":
            progress_count += 1
            if progress_count <= 3 or progress_count % 20 == 0:
                print(f"  Progress: {ev['current']}/{ev['total']}")
        elif ev_type == "audio":
            raw = base64.b64decode(ev["data"])
            chunk = struct.unpack(f"{len(raw)//4}f", raw)
            audio_chunks.extend(chunk)
            sample_rate = ev.get("sample_rate", sample_rate)
            print(f"  Audio chunk #{ev['chunk_index']}: {len(raw)//4} samples")

            # Real-time playback: spawn ffplay on first chunk, feed subsequent ones
            if play:
                if ffplay_proc is None:
                    ffplay_proc = subprocess.Popen(
                        ["ffplay", "-f", "f32le", "-ar", str(sample_rate),
                         "-ac", "1", "-nodisp", "-loglevel", "quiet", "-i", "-"],
                        stdin=subprocess.PIPE)
                ffplay_proc.stdin.write(raw)
                ffplay_proc.stdin.flush()

        elif ev_type == "done":
            done = True
            dur = ev.get("duration", 0)
            total = ev.get("total_samples", 0)
            print(f"  Done: {total} samples, {dur:.2f}s")
        elif ev_type == "error":
            error = ev.get("message", "unknown error")
            print(f"  ERROR: {error}")
            break
        else:
            print(f"  Unknown event: {ev_type}")

    # Close ffplay stdin → it finishes playback
    if ffplay_proc:
        ffplay_proc.stdin.close()
        ffplay_proc.wait()

    assert done, "Did not receive 'done' event"
    assert error is None, f"Received error event: {error}"
    assert progress_count > 0, "No progress events received"
    assert len(audio_chunks) > 0, "No audio chunks received"

    wav_path = os.path.join(output_dir, "test_streaming.wav")
    _write_wav(wav_path, audio_chunks, sample_rate)
    print(f"PASS ({len(audio_chunks)} samples, progress={progress_count}) → {wav_path}")
    return True


def test_error_missing_text(base_url: str) -> bool:
    """POST /v1/tts without 'text' field → 400"""
    print("[TEST] Error: missing text...", end=" ")
    r = requests.post(f"{base_url}/v1/tts", json={}, timeout=10)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    data = r.json()
    assert "error" in data, f"Expected error field: {data}"
    print(f"PASS ({data['error']})")
    return True


def test_error_empty_text(base_url: str) -> bool:
    """POST /v1/tts with empty text → 400"""
    print("[TEST] Error: empty text...", end=" ")
    r = requests.post(f"{base_url}/v1/tts", json={"text": ""}, timeout=10)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print("PASS")
    return True


def test_with_ref(base_url: str, output_dir: str) -> bool:
    """POST /v1/tts/with-ref — voice cloning with reference audio"""
    print("[TEST] Voice cloning with-ref...")
    # Use example files from the project
    import glob
    ref_wav = "example/vo_LLZAQ001_4_nahida_03.wav"
    ref_lab = "example/vo_LLZAQ001_4_nahida_03.lab"

    if not os.path.exists(ref_wav) or not os.path.exists(ref_lab):
        # Try to find any example files
        wavs = sorted(glob.glob("example/*.wav"))
        labs = sorted(glob.glob("example/*.lab"))
        if wavs and labs:
            ref_wav = wavs[0]
            ref_lab = labs[0]
        else:
            print("  SKIP: no example reference audio found")
            return True

    # Read reference text
    with open(ref_lab) as f:
        ref_text = f.read().strip()
    print(f"  Ref audio: {ref_wav}")
    print(f"  Ref text: '{ref_text[:60]}...'")

    # Read WAV → float32 PCM
    w = wave.open(ref_wav)
    assert w.getnchannels() == 1, "Ref audio must be mono"
    sampwidth = w.getsampwidth()
    raw_pcm = w.readframes(w.getnframes())
    w.close()

    # Convert to float32
    if sampwidth == 2:  # 16-bit
        pcm_int = struct.unpack(f"{len(raw_pcm)//2}h", raw_pcm)
        audio_f32 = [v / 32768.0 for v in pcm_int]
    else:
        print(f"  SKIP: unsupported sample width {sampwidth*8}-bit")
        return True

    ref_audio_b64 = base64.b64encode(struct.pack(f"{len(audio_f32)}f", *audio_f32)).decode()

    payload = {
        "text": "今天天气真好，我们一起出去玩吧。",
        "ref_audio": ref_audio_b64,
        "ref_text": ref_text,
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 42,
    }

    r = requests.post(f"{base_url}/v1/tts/with-ref", json=payload, timeout=120)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
    data = r.json()

    assert "audio" in data, f"Missing 'audio' field"
    assert data["num_samples"] > 0, f"Expected num_samples > 0"

    raw = base64.b64decode(data["audio"])
    num = len(raw) // 4
    samples = struct.unpack(f"{num}f", raw)
    wav_path = os.path.join(output_dir, "test_with_ref.wav")
    _write_wav(wav_path, samples, data["sample_rate"])
    dur = data["num_samples"] / data["sample_rate"]
    print(f"PASS ({num} samples, {dur:.2f}s) → {wav_path}")
    return True


def test_with_ref_streaming(base_url: str, output_dir: str, play: bool = False) -> bool:
    """POST /v1/tts/with-ref/stream — streaming voice cloning"""
    print("[TEST] Voice cloning with-ref streaming...")
    ref_wav = "example/vo_LLZAQ001_4_nahida_03.wav"
    ref_lab = "example/vo_LLZAQ001_4_nahida_03.lab"

    if not os.path.exists(ref_wav) or not os.path.exists(ref_lab):
        print("  SKIP: no example reference audio found")
        return True

    with open(ref_lab) as f:
        ref_text = f.read().strip()

    w = wave.open(ref_wav)
    raw_pcm = w.readframes(w.getnframes())
    w.close()
    import struct
    pcm_int = struct.unpack(f"{len(raw_pcm)//2}h", raw_pcm)
    audio_f32 = [v / 32768.0 for v in pcm_int]
    ref_audio_b64 = base64.b64encode(struct.pack(f"{len(audio_f32)}f", *audio_f32)).decode()

    payload = {
        "text": "今天天气真好，我们一起出去玩吧。",
        "ref_audio": ref_audio_b64,
        "ref_text": ref_text,
        "max_new_tokens": 200,
        "temperature": 0.7, "top_p": 0.9, "top_k": 50, "seed": 42,
    }

    r = requests.post(f"{base_url}/v1/tts/with-ref/stream", json=payload,
                      stream=True, timeout=300)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    audio_chunks = []
    sample_rate = 44100
    done = False
    ffplay_proc = None

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        t = ev.get("type", "?")
        if t == "audio":
            raw = base64.b64decode(ev["data"])
            chunk = struct.unpack(f"{len(raw)//4}f", raw)
            audio_chunks.extend(chunk)
            sample_rate = ev.get("sample_rate", sample_rate)
            print(f"  Audio chunk #{ev['chunk_index']}: {len(raw)//4} samples")
            if play:
                if ffplay_proc is None:
                    ffplay_proc = subprocess.Popen(
                        ["ffplay", "-f", "f32le", "-ar", str(sample_rate),
                         "-ac", "1", "-nodisp", "-loglevel", "quiet", "-i", "-"],
                        stdin=subprocess.PIPE)
                ffplay_proc.stdin.write(raw)
                ffplay_proc.stdin.flush()
        elif t == "done":
            done = True
            print(f"  Done: {ev['total_samples']} samples, {ev.get('duration', 0):.2f}s")
        elif t == "error":
            print(f"  ERROR: {ev.get('message')}")
            break

    if ffplay_proc:
        ffplay_proc.stdin.close()
        ffplay_proc.wait()

    assert done, "No done event"
    assert len(audio_chunks) > 0, "No audio chunks"

    wav_path = os.path.join(output_dir, "test_with_ref_streaming.wav")
    _write_wav(wav_path, audio_chunks, sample_rate)
    print(f"PASS ({len(audio_chunks)} samples) → {wav_path}")
    return True


def _write_wav(path: str, samples: list, sample_rate: int):
    """Write float32 samples as 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    pcm = []
    for s in samples:
        v = max(-1.0, min(1.0, s))
        pcm.append(int(v * 32767))
    pcm_bytes = struct.pack(f"{len(pcm)}h", *pcm)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def main():
    parser = argparse.ArgumentParser(description="Test fish-audio-cpp REST API")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", default=8080, type=int, help="Server port")
    parser.add_argument("--output-dir", default="output", help="Output directory for WAV files")
    parser.add_argument("--play", action="store_true", help="Play streaming audio in real-time via ffplay")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    os.makedirs(args.output_dir, exist_ok=True)

    tests = [
        ("Health", lambda: test_health(base_url)),
        ("Info", lambda: test_info(base_url)),
        ("Error: missing text", lambda: test_error_missing_text(base_url)),
        ("Error: empty text", lambda: test_error_empty_text(base_url)),
        ("Non-streaming TTS", lambda: test_non_streaming(base_url, args.output_dir)),
        ("Streaming TTS", lambda: test_streaming(base_url, args.output_dir, args.play)),
        ("Voice clone with-ref", lambda: test_with_ref(base_url, args.output_dir)),
        ("Voice clone with-ref stream", lambda: test_with_ref_streaming(base_url, args.output_dir, args.play)),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
