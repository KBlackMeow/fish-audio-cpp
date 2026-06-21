# REST API Streaming — Design Spec

**Date:** 2026-06-21
**Status:** approved
**Scope:** Add SSE streaming endpoint to existing HTTP server, plus test script.

## Overview

The project already has a basic REST API (`src/server/`) with `POST /v1/tts` returning
base64-encoded PCM audio in a single JSON response. This spec adds a streaming endpoint
using Server-Sent Events (SSE) so clients receive progress updates during generation and
audio chunks as soon as DAC decoding completes — enabling playback before the full
response body is transferred.

## Endpoints

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| `GET` | `/health` | existing | Health check |
| `GET` | `/v1/info` | existing | Server info |
| `POST` | `/v1/tts` | existing | Non-streaming TTS (JSON response) |
| `POST` | `/v1/tts/stream` | **new** | Streaming TTS (SSE response) |

## SSE Event Protocol

`POST /v1/tts/stream` accepts the same JSON body as `/v1/tts`:
```json
{"text": "...", "max_new_tokens": 512, "temperature": 0.7, "top_p": 0.9, "top_k": 50, "seed": 42}
```

Response: `Content-Type: text/event-stream`

Events:
```
data: {"type":"progress","current":0,"total":200}

data: {"type":"progress","current":1,"total":200}
...

data: {"type":"audio","data":"<base64>","sample_rate":44100,"chunk_index":0,"num_samples":2205}

data: {"type":"audio","data":"<base64>","sample_rate":44100,"chunk_index":1,"num_samples":2205}
...

data: {"type":"done","total_samples":88200,"duration":2.0}

```

- `progress`: emitted once per AR decode step. `current` is 0-indexed step number, `total` is `max_new_tokens`.
- `audio`: emitted after DAC decode completes. Audio is split into ~50ms chunks (~2205 samples at 44.1kHz) for network-friendly streaming.
- `done`: final event with summary stats.
- On error: `data: {"type":"error","message":"..."}` then connection closes.

Key design decisions:
- **No incremental DAC decode.** The AR loop generates codes; DAC decode runs once at the end on the full code sequence. This avoids O(N²) DAC compute.
- **Audio chunk streaming after decode.** The full PCM buffer is split into ~50ms chunks and streamed via SSE. This gives network-level streaming (client starts playback sooner) without compromising audio quality.
- **Progress events during generation.** The slow part (AR loop, 5–30s) provides real-time feedback.

## Code Changes

### 1. `src/engine/inference_pipeline.h` — add streaming callback

```cpp
// Callback for streaming inference.
// Called from the inference thread — must be thread-safe if server is multi-threaded.
struct StreamCallback {
    // Called once per AR decode step. current is 0-indexed.
    std::function<void(int current, int total)> on_progress;
    // Called for each audio chunk after DAC decode.
    std::function<void(const float* samples, int count)> on_audio_chunk;
};

// New method: same as run() but reports progress and streams audio chunks.
TTSOutput run_streaming(
    const std::string& text,
    int max_new_tokens,
    float temperature, float top_p, int top_k, int seed,
    StreamCallback callback
);
```

### 2. `src/engine/inference_pipeline.cc` — implement `run_streaming()`

The implementation reuses the exact same AR loop and DAC decode as `run()`.
Differences:
- After each decode step: `if (callback.on_progress) callback.on_progress(step, max_new_tokens);`
- After DAC decode → h_audio is ready: split into ~50ms chunks (2205 samples), call `callback.on_audio_chunk(ptr, len)` for each.

To avoid duplicating the ~400-line inference loop, the implementation refactors
`run()` to call `run_streaming()` with a no-op callback. The shared logic lives in
`run_streaming()`, and `run()` becomes a thin wrapper.

### 3. `src/server/http_server.cc` — add streaming endpoint

New handler for `POST /v1/tts/stream`:

```cpp
server_->Post("/v1/tts/stream", [this](const httplib::Request& req, httplib::Response& res) {
    // Parse body (same as /v1/tts)
    // Validate text field
    // Set headers for SSE
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    // Use chunked content provider for SSE
    res.set_chunked_content_provider([this, params...](size_t offset, httplib::DataSink& sink) {
        // Call pipeline_->run_streaming() with callbacks that write SSE events to sink
        // on_progress → write progress SSE event
        // on_audio_chunk → write audio SSE event (base64-encoded)
        // After completion → write done SSE event, then sink.done()
        // On error → write error SSE event, then sink.done()
        return true; // all data sent
    });
});
```

Helper: `write_sse_event(sink, json_str)` — writes `"data: {json}\n\n"` to the data sink.

### 4. `src/server/routes.cc` — add SSE helper

Add `encode_pcm_base64(const float*, int)` already exists. Add:

```cpp
inline std::string sse_event(const std::string& type, const nlohmann::json& data) {
    nlohmann::json j = data;
    j["type"] = type;
    return "data: " + j.dump() + "\n\n";
}
```

## Test Script: `scripts/test_api.py`

Python script using only stdlib + `requests`. Tests:

1. **Health check** — `GET /health` → assert 200, `status: ok`
2. **Server info** — `GET /v1/info` → assert 200, `engine` field present
3. **Non-streaming TTS** — `POST /v1/tts` with text → assert 200, save WAV file
4. **Streaming TTS** — `POST /v1/tts/stream` → read SSE events, print progress, reconstruct audio, save WAV
5. **Error: missing text** — `POST /v1/tts` without text → assert 400
6. **Error: empty text** — `POST /v1/tts` with empty text → assert 400

The streaming test uses raw `http.client` (or `requests` with `stream=True`) to read the SSE
response line by line, parse JSON from each `data:` line, and accumulate audio chunks.

WAV output: 16-bit PCM, 44100 Hz mono, using Python's `wave` module.

## Files Changed

| File | Action |
|------|--------|
| `src/engine/inference_pipeline.h` | Add `StreamCallback`, `run_streaming()` declaration |
| `src/engine/inference_pipeline.cc` | Refactor `run()` → delegate to `run_streaming()` |
| `src/server/http_server.cc` | Add `POST /v1/tts/stream` handler |
| `src/server/routes.cc` | Add `sse_event()` helper |
| `scripts/test_api.py` | **New** — test script |

## Non-Goals (excluded from this spec)

- Voice cloning endpoint (`/v1/tts/with-ref` etc.) — deferred to follow-up
- WebSocket support — not needed given SSE
- Batching/multi-request server — uses existing single-request model
- Authentication, rate-limiting — out of scope
