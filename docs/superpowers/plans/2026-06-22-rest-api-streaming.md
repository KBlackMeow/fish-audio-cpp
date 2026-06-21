# REST API Streaming — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SSE streaming endpoint (`POST /v1/tts/stream`) to the HTTP server and a Python test script.

**Architecture:** Refactor `InferencePipeline::run()` to delegate to a new `run_streaming()` method that accepts progress/audio-chunk callbacks. The HTTP server uses cpp-httplib's `set_chunked_content_provider` to emit SSE events. A `--server` CLI flag is added so the server can be started without a full CLI TTS run.

**Tech Stack:** C++17, CUDA, cpp-httplib v0.18.1, nlohmann/json, Python 3 stdlib + requests

## Global Constraints

- C++17, CUDA 17, CMake ≥3.24
- cpp-httplib v0.18.1 (via FetchContent, already in CMakeLists.txt)
- Audio chunk size: 2205 samples (~50ms at 44.1kHz)
- SSE format: `data: {json}\n\n`
- Audio encoding: base64 (float32 PCM bytes) — reuse existing `encode_pcm_base64()`
- `run()` remains unchanged as public API, internally delegates to `run_streaming()` with no-op callbacks

---

### Task 1: Add `--server` flag to main.cc

**Files:**
- Modify: `src/main.cc:337-370` (options), after line 389 (parse), after line 629 (run server instead of CLI TTS)

**Interfaces:**
- Produces: `--server` flag (bool), `--host` (string, default "0.0.0.0"), `--port` (int, default 8080)
- Consumes: Existing `InferencePipeline` construction, `HttpServer` class

- [ ] **Step 1: Add CLI options for server mode**

In `src/main.cc`, add after line 369 (`decode-codes` option):
```cpp
("server",          "Run in HTTP server mode instead of CLI",
 cxxopts::value<bool>()->default_value("false"))
("host",            "Server bind address",
 cxxopts::value<std::string>()->default_value("0.0.0.0"))
("port",            "Server port",
 cxxopts::value<int>()->default_value("8080"))
```

- [ ] **Step 2: Parse new args**

After line 389 (`decode_codes`), add:
```cpp
const bool        server_mode  = args["server"].as<bool>();
const std::string server_host  = args["host"].as<std::string>();
const int         server_port  = args["port"].as<int>();
```

- [ ] **Step 3: Add server-mode branch**

After line 629 (`result = pipeline.run(...)` closing brace `}`), add the `#include "server/http_server.h"` at top of file. Then replace the output block (lines 629-644) with:

```cpp
if (server_mode) {
    fish::HttpServer::Config srv_cfg;
    srv_cfg.host = server_host;
    srv_cfg.port = server_port;
    fish::HttpServer srv(srv_cfg, &pipeline);
    spdlog::info("Starting HTTP server on {}:{}", server_host, server_port);
    srv.start();  // blocking
    // server stopped — cleanup below
} else {
    fish::TTSOutput result;
    if (!prompt_file.empty()) {
        spdlog::info("Using pre-built prompt file: {}", prompt_file);
        result = pipeline.run_with_prompt_file(prompt_file, max_tokens, temperature, top_p, top_k, seed);
    } else {
        result = pipeline.run(text, max_tokens, temperature, top_p, top_k, seed);
    }

    if (result.audio_samples.empty()) {
        spdlog::error("No audio generated");
        cudnnDestroy(cudnn);
        cublasDestroy(cublas);
        cudaStreamDestroy(stream);
        return 1;
    }

    write_wav(output_wav, result.audio_samples.data(),
              static_cast<int>(result.audio_samples.size()), result.sample_rate);
}
```

And add the include near the top (line 8 area):
```cpp
#include "server/http_server.h"
```

- [ ] **Step 4: Build and verify compilation**

```bash
cd /home/illya/fish-audio-cpp/build && cmake .. && make -j$(nproc) fish-server 2>&1 | tail -20
```

Expected: successful compilation, no errors.

- [ ] **Step 5: Quick smoke test**

```bash
./build/fish-server --server --help 2>&1 | grep -q server
```

Expected: `--server` appears in help output.

- [ ] **Step 6: Commit**

```bash
git add src/main.cc
git commit -m "feat: add --server flag to run HTTP server mode"
```

---

### Task 2: Add `StreamCallback` and `run_streaming()` to InferencePipeline

**Files:**
- Modify: `src/engine/inference_pipeline.h:11-14` (add StreamCallback struct), `:34-42` (add run_streaming declaration)
- Modify: `src/engine/inference_pipeline.cc:54-547` (refactor run() to delegate to run_streaming())

**Interfaces:**
- Produces: `StreamCallback { on_progress, on_audio_chunk }`, `TTSOutput run_streaming(text, max_new_tokens, temperature, top_p, top_k, seed, StreamCallback)`
- Consumes: DualAREngine, DACEngine, BlockManager, Tokenizer (all existing, unchanged)

- [ ] **Step 1: Add StreamCallback struct to header**

In `src/engine/inference_pipeline.h`, after `TTSOutput` (line 14), add:
```cpp
// Callbacks for streaming inference.
// All callbacks are called from the inference thread.
struct StreamCallback {
    // Called once per AR decode step. current is 0-indexed step number.
    std::function<void(int current, int total)> on_progress;
    // Called for each audio chunk after DAC decode completes.
    // Chunks are ~50ms (2205 samples at 44.1kHz).
    std::function<void(const float* samples, int count)> on_audio_chunk;
};
```

- [ ] **Step 2: Add run_streaming() declaration**

In `src/engine/inference_pipeline.h`, after `run_with_prompt_file()` declaration (before `private:`), add:
```cpp
    // Streaming variant of run(). Reports progress during AR generation
    // and streams audio chunks after DAC decode. Returns full TTSOutput.
    TTSOutput run_streaming(
        const std::string& text,
        int max_new_tokens,
        float temperature, float top_p, int top_k, int seed,
        StreamCallback callback
    );
```

- [ ] **Step 3: Refactor inference_pipeline.cc — extract shared logic**

In `src/engine/inference_pipeline.cc`:

Replace the body of `TTSOutput InferencePipeline::run(...)` (lines 54-547) with a thin wrapper:

```cpp
TTSOutput InferencePipeline::run(
    const std::string& text,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    int seed
) {
    StreamCallback noop;
    return run_streaming(text, max_new_tokens, temperature, top_p, top_k, seed, noop);
}
```

Then rename the existing `run()` body to `run_streaming()` by changing the method signature (line 54) from:
```cpp
TTSOutput InferencePipeline::run(
```
to:
```cpp
TTSOutput InferencePipeline::run_streaming(
    const std::string& text,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    int seed,
    StreamCallback callback
)
```

- [ ] **Step 4: Add progress callback in AR loop**

In the same function (now `run_streaming`), inside the decode loop (around line 447, the `for (int step = 0; step < max_new_tokens; step++)` block), add after the diagnostic log block (after line 453, the `// Record current frame` comment area):

After the line `generated.push_back(sem);` (around line 445):
```cpp
        // Streaming: report progress
        if (callback.on_progress) {
            callback.on_progress(step, max_new_tokens);
        }
```

- [ ] **Step 5: Add audio chunk streaming after DAC decode**

After the DAC decode section where `h_audio` is populated (around line 523-525, after `cudaMemcpy` of audio to host), and before the cleanup block (line 531), add:

```cpp
    // Streaming: emit audio chunks (~50ms each)
    if (callback.on_audio_chunk) {
        constexpr int kChunkSize = 2205;  // ~50ms at 44.1kHz
        int offset = 0;
        while (offset < audio_len) {
            int chunk = std::min(kChunkSize, audio_len - offset);
            callback.on_audio_chunk(h_audio.data() + offset, chunk);
            offset += chunk;
        }
    }
```

- [ ] **Step 6: Build and verify**

```bash
cd /home/illya/fish-audio-cpp/build && cmake .. && make -j$(nproc) fish_core fish-server 2>&1 | tail -20
```

Expected: no compilation errors.

- [ ] **Step 7: Commit**

```bash
git add src/engine/inference_pipeline.h src/engine/inference_pipeline.cc
git commit -m "feat: add run_streaming() with progress and audio-chunk callbacks"
```

---

### Task 3: Add SSE streaming endpoint to HttpServer

**Files:**
- Modify: `src/server/routes.cc:1-37` (add sse_event helper)
- Modify: `src/server/http_server.cc:1-112` (add POST /v1/tts/stream handler)

**Interfaces:**
- Consumes: `StreamCallback`, `InferencePipeline::run_streaming()`, `encode_pcm_base64()` from routes.cc
- Produces: `POST /v1/tts/stream` SSE endpoint

- [ ] **Step 1: Add `sse_event()` helper to routes.cc**

In `src/server/routes.cc`, after the existing `encode_pcm_base64()` function, add:

```cpp
// Format a Server-Sent Events message line.
// Each call returns one "data: {json}\n\n" frame.
inline std::string sse_event(const std::string& type, const nlohmann::json& data) {
    nlohmann::json j = data;
    j["type"] = type;
    return "data: " + j.dump() + "\n\n";
}
```

Also add `#include <functional>` at the top (already may be included transitively).

- [ ] **Step 2: Add streaming handler to http_server.cc**

In `src/server/http_server.cc`, after the existing `POST /v1/tts` handler (after line 91 `})`), and before the `/v1/info` handler (line 94), add:

```cpp
    // TTS streaming endpoint (SSE)
    server_->Post("/v1/tts/stream", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = nlohmann::json::parse(req.body);
            std::string text = body.value("text", "");
            if (text.empty()) {
                res.status = 400;
                res.set_content(routes::error_json("Missing 'text' field"), "application/json");
                return;
            }

            int max_tokens  = body.value("max_new_tokens", 512);
            float temp      = body.value("temperature", 0.7f);
            float top_p     = body.value("top_p", 0.9f);
            int top_k       = body.value("top_k", 50);
            int seed        = body.value("seed", 42);

            spdlog::info("TTS stream: '{}' (max_tokens={})", text.substr(0, 60), max_tokens);

            // SSE headers
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_header("X-Accel-Buffering", "no");  // disable nginx buffering

            // Use chunked content provider for SSE
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, text, max_tokens, temp, top_p, top_k, seed](
                    size_t /*offset*/, httplib::DataSink& sink
                ) -> bool {
                    bool done_flag = false;
                    std::string error_msg;

                    StreamCallback cb;
                    cb.on_progress = [&sink](int current, int total) {
                        nlohmann::json ev;
                        ev["current"] = current;
                        ev["total"] = total;
                        sink.write(routes::sse_event("progress", ev).data(),
                                   routes::sse_event("progress", ev).size());
                    };

                    cb.on_audio_chunk = [&sink, chunk_idx = 0](
                        const float* samples, int count
                    ) mutable {
                        std::string raw(reinterpret_cast<const char*>(samples),
                                       count * sizeof(float));
                        std::string b64 = httplib::detail::base64_encode(raw);
                        nlohmann::json ev;
                        ev["data"] = b64;
                        ev["sample_rate"] = 44100;
                        ev["chunk_index"] = chunk_idx++;
                        ev["num_samples"] = count;
                        sink.write(routes::sse_event("audio", ev).data(),
                                   routes::sse_event("audio", ev).size());
                    };

                    try {
                        auto result = pipeline_->run_streaming(
                            text, max_tokens, temp, top_p, top_k, seed, cb);

                        nlohmann::json done_ev;
                        done_ev["total_samples"] = static_cast<int>(result.audio_samples.size());
                        done_ev["duration"] = static_cast<double>(result.audio_samples.size())
                                              / result.sample_rate;
                        done_ev["sample_rate"] = result.sample_rate;
                        sink.write(routes::sse_event("done", done_ev).data(),
                                   routes::sse_event("done", done_ev).size());
                    } catch (const std::exception& e) {
                        nlohmann::json err_ev;
                        err_ev["message"] = e.what();
                        sink.write(routes::sse_event("error", err_ev).data(),
                                   routes::sse_event("error", err_ev).size());
                        spdlog::error("TTS stream error: {}", e.what());
                    }

                    sink.done();
                    return true;
                }
            );
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(routes::error_json(e.what()), "application/json");
            spdlog::error("TTS stream setup error: {}", e.what());
        }
    });
```

- [ ] **Step 3: Add routes.h include if needed**

Check that `src/server/http_server.cc` includes `routes.cc` or that the functions are accessible. The current `http_server.cc` doesn't include `routes.cc`. The `routes.cc` file is compiled separately as part of `fish_core` in CMakeLists.txt. We need a header for it.

Create `src/server/routes.h`:
```cpp
// src/server/routes.h
#pragma once
#include <nlohmann/json.hpp>
#include <string>

namespace fish::routes {

std::string error_json(const std::string& msg);
std::string encode_pcm_base64(const float* samples, int n);
std::string sse_event(const std::string& type, const nlohmann::json& data);

}  // namespace fish::routes
```

Then modify `src/server/routes.cc` to:
```cpp
// src/server/routes.cc — shared route utilities for the HTTP TTS server
#include "server/routes.h"

namespace fish::routes {

std::string error_json(const std::string& msg) {
    nlohmann::json j;
    j["error"] = msg;
    return j.dump();
}

std::string encode_pcm_base64(const float* samples, int n) {
    static const char kBase64[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    auto* bytes = reinterpret_cast<const uint8_t*>(samples);
    int nbytes = n * static_cast<int>(sizeof(float));
    std::string out;
    out.reserve(((nbytes + 2) / 3) * 4);
    for (int i = 0; i < nbytes; i += 3) {
        uint32_t val = static_cast<uint32_t>(bytes[i]) << 16;
        if (i + 1 < nbytes) val |= static_cast<uint32_t>(bytes[i + 1]) << 8;
        if (i + 2 < nbytes) val |= static_cast<uint32_t>(bytes[i + 2]);
        out.push_back(kBase64[(val >> 18) & 0x3F]);
        out.push_back(kBase64[(val >> 12) & 0x3F]);
        out.push_back((i + 1 < nbytes) ? kBase64[(val >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < nbytes) ? kBase64[val & 0x3F] : '=');
    }
    return out;
}

std::string sse_event(const std::string& type, const nlohmann::json& data) {
    nlohmann::json j = data;
    j["type"] = type;
    return "data: " + j.dump() + "\n\n";
}

}  // namespace fish::routes
```

And in `src/server/http_server.cc`, replace any direct use of `routes::` functions (they already work since routes.cc is compiled into the same library) — just add the include:
```cpp
#include "server/routes.h"
```

In `CMakeLists.txt`, add the header to the library (optional since headers aren't compiled):
No CMakeLists.txt changes needed — the header is in the include path via `target_include_directories(fish_core PUBLIC ${CMAKE_SOURCE_DIR}/src)`.

- [ ] **Step 4: Build and verify**

```bash
cd /home/illya/fish-audio-cpp/build && cmake .. && make -j$(nproc) fish-server 2>&1 | tail -30
```

Expected: successful compilation. If `routes::` functions are not found from http_server.cc, verify the include is correct.

- [ ] **Step 5: Commit**

```bash
git add src/server/routes.h src/server/routes.cc src/server/http_server.cc
git commit -m "feat: add SSE streaming endpoint POST /v1/tts/stream"
```

---

### Task 4: Write test script

**Files:**
- Create: `scripts/test_api.py`

**Interfaces:**
- Consumes: HTTP server on localhost:8080 (`/health`, `/v1/info`, `/v1/tts`, `/v1/tts/stream`)
- Produces: test output WAV files in a temp directory

- [ ] **Step 1: Write the test script**

Create `scripts/test_api.py`:

```python
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
import sys
import tempfile
import wave
from pathlib import Path

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


def test_streaming(base_url: str, output_dir: str) -> bool:
    """POST /v1/tts/stream — SSE streaming TTS"""
    print("[TEST] Streaming TTS...")
    payload = {
        "text": "今天天气真不错，适合出门散步。",
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "seed": 123,
    }

    # Use stream=True to read SSE events as they arrive
    r = requests.post(f"{base_url}/v1/tts/stream", json=payload,
                      stream=True, timeout=300)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    progress_count = 0
    audio_chunks = []
    sample_rate = 44100
    done = False
    error = None

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if not line.startswith("data: "):
            continue

        data_str = line[6:]  # strip "data: " prefix
        try:
            ev = json.loads(data_str)
        except json.JSONDecodeError:
            print(f"  [WARN] Invalid JSON in SSE: {data_str[:80]}")
            continue

        ev_type = ev.get("type", "?")
        if ev_type == "progress":
            progress_count += 1
            if progress_count <= 3 or progress_count % 20 == 0:
                print(f"  Progress: {ev['current']}/{ev['total']}")
        elif ev_type == "audio":
            raw = base64.b64decode(ev["data"])
            num = len(raw) // 4
            chunk = struct.unpack(f"{num}f", raw)
            audio_chunks.extend(chunk)
            sample_rate = ev.get("sample_rate", sample_rate)
            print(f"  Audio chunk #{ev['chunk_index']}: {num} samples")
        elif ev_type == "done":
            done = True
            dur = ev.get("duration", 0)
            print(f"  Done: {ev['total_samples']} samples, {dur:.2f}s")
        elif ev_type == "error":
            error = ev.get("message", "unknown error")
            print(f"  ERROR: {error}")
            break
        else:
            print(f"  Unknown event: {ev_type}")

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


def _write_wav(path: str, samples: list, sample_rate: int):
    """Write float32 samples as 16-bit PCM WAV."""
    import numpy as np  # optional: fall back to manual clipping
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
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tests = [
            ("Health", lambda: test_health(base_url)),
            ("Info", lambda: test_info(base_url)),
            ("Error: missing text", lambda: test_error_missing_text(base_url)),
            ("Error: empty text", lambda: test_error_empty_text(base_url)),
            ("Non-streaming TTS", lambda: test_non_streaming(base_url, tmpdir)),
            ("Streaming TTS", lambda: test_streaming(base_url, tmpdir)),
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
```

- [ ] **Step 2: Make script executable**

```bash
chmod +x /home/illya/fish-audio-cpp/scripts/test_api.py
```

- [ ] **Step 3: Verify script syntax**

```bash
python3 -c "import ast; ast.parse(open('scripts/test_api.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/test_api.py
git commit -m "test: add API test script for REST and SSE streaming endpoints"
```

---

### Task 5: Build and integration test

**Files:** None (verification only)

- [ ] **Step 1: Full rebuild**

```bash
cd /home/illya/fish-audio-cpp/build && cmake .. && make -j$(nproc) 2>&1 | tail -30
```

Expected: all targets build successfully.

- [ ] **Step 2: Run existing unit tests to verify no regressions**

```bash
cd /home/illya/fish-audio-cpp/build && ./test_fish 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 3: Commit if any fixes were needed**

Only if changes were required to fix build or test failures.
