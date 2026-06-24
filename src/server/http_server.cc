// src/server/http_server.cc
#include "server/http_server.h"
#include "server/routes.h"
#include <spdlog/spdlog.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <chrono>

namespace fish {

namespace {

nlohmann::json profiling_json(const TTSProfiling& p) {
    return {
        {"total_ms", p.total_ms},
        {"tokenize_ms", p.tokenize_ms},
        {"prefill_ms", p.prefill_ms},
        {"ar_decode_ms", p.ar_decode_ms},
        {"decode_embed_ms", p.decode_embed_ms},
        {"decode_step_ms", p.decode_step_ms},
        {"decode_logits_ms", p.decode_logits_ms},
        {"semantic_sample_ms", p.semantic_sample_ms},
        {"codebook_decode_ms", p.codebook_decode_ms},
        {"codebook_sample_ms", p.codebook_sample_ms},
        {"seq_update_ms", p.seq_update_ms},
        {"dac_decode_ms", p.dac_decode_ms},
        {"audio_copy_ms", p.audio_copy_ms},
        {"first_audio_ms", p.first_audio_ms},
        {"ref_decode_ms", p.ref_decode_ms},
        {"ref_encode_ms", p.ref_encode_ms},
        {"prompt_build_ms", p.prompt_build_ms},
        {"response_encode_ms", p.response_encode_ms},
        {"prompt_tokens", p.prompt_tokens},
        {"generated_frames", p.generated_frames},
        {"ref_code_frames", p.ref_code_frames},
        {"streamed_dac_calls", p.streamed_dac_calls},
        {"streamed_audio_chunks", p.streamed_audio_chunks},
        {"ref_cache_hit", p.ref_cache_hit},
    };
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

}  // namespace

HttpServer::HttpServer(const Config& cfg, InferencePipeline* pipeline)
    : cfg_(cfg), pipeline_(pipeline)
{
    server_ = std::make_unique<httplib::Server>();
    spdlog::info("HttpServer: {}:{} ({} threads)", cfg_.host, cfg_.port, cfg_.num_threads);
}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::start() {
    // CORS middleware
    server_->set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Health check
    server_->Get("/health", [](const httplib::Request&, httplib::Response& res) {
        nlohmann::json j;
        j["status"] = "ok";
        res.set_content(j.dump(), "application/json");
    });

    // TTS endpoint
    server_->Post("/v1/tts", [this](const httplib::Request& req, httplib::Response& res) {
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
            std::string response_format =
                body.value("response_format", body.value("format", std::string("wav")));

            spdlog::info("TTS: '{}' (max_tokens={})", text.substr(0, 60), max_tokens);

            TTSOutput result;
            {
                std::lock_guard<std::mutex> lock(inference_mutex_);
                result = pipeline_->run(text, max_tokens, temp, top_p, top_k, seed);
            }

            if (result.audio_samples.empty()) {
                res.status = 500;
                res.set_content(routes::error_json("Inference produced no audio"), "application/json");
                return;
            }

            nlohmann::json resp;
            const auto encode_start = std::chrono::steady_clock::now();
            if (response_format == "pcm_f32" || response_format == "float32") {
                resp["audio"] = routes::encode_pcm_f32_base64(
                    result.audio_samples.data(),
                    static_cast<int>(result.audio_samples.size()));
                resp["audio_format"] = "pcm_f32";
                resp["audio_bytes"] = static_cast<int>(result.audio_samples.size() * sizeof(float));
            } else if (response_format == "wav") {
                resp["audio"] = routes::encode_wav_base64(
                    result.audio_samples.data(),
                    static_cast<int>(result.audio_samples.size()),
                    result.sample_rate);
                resp["audio_format"] = "wav";
                resp["audio_bytes"] = static_cast<int>(44 + result.audio_samples.size() * 2);
            } else {
                res.status = 400;
                res.set_content(routes::error_json(
                    "Unsupported response format: use 'wav' or 'pcm_f32'"), "application/json");
                return;
            }
            result.profiling.response_encode_ms = elapsed_ms(encode_start);
            resp["sample_rate"] = result.sample_rate;
            resp["num_samples"] = static_cast<int>(result.audio_samples.size());
            resp["duration_s"] = static_cast<double>(result.audio_samples.size())
                                 / result.sample_rate;
            resp["profiling"] = profiling_json(result.profiling);
            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(routes::error_json(e.what()), "application/json");
            spdlog::error("TTS error: {}", e.what());
        }
    });

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
            int sr = pipeline_->sample_rate();
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, text, max_tokens, temp, top_p, top_k, seed, sr](
                    size_t /*offset*/, httplib::DataSink& sink
                ) -> bool {
                    StreamCallback cb;
                    cb.on_progress = [&sink](int current, int total) {
                        nlohmann::json ev;
                        ev["current"] = current;
                        ev["total"] = total;
                        auto se = routes::sse_event("progress", ev);
                        sink.write(se.data(), se.size());
                    };

                    cb.on_audio_chunk = [&sink, sr, chunk_idx = 0](
                        const float* samples, int count
                    ) mutable {
                        nlohmann::json ev;
                        ev["data"] = routes::encode_pcm_f32_base64(samples, count);
                        ev["audio_format"] = "pcm_f32";
                        ev["sample_rate"] = sr;
                        ev["chunk_index"] = chunk_idx++;
                        ev["num_samples"] = count;
                        auto se = routes::sse_event("audio", ev);
                        sink.write(se.data(), se.size());
                    };

                    try {
                        TTSOutput result;
                        {
                            std::lock_guard<std::mutex> lock(inference_mutex_);
                            result = pipeline_->run_streaming(
                                text, max_tokens, temp, top_p, top_k, seed, cb);
                        }

                        nlohmann::json done_ev;
                        done_ev["total_samples"] = static_cast<int>(result.audio_samples.size());
                        done_ev["duration"] = static_cast<double>(result.audio_samples.size())
                                              / result.sample_rate;
                        done_ev["sample_rate"] = result.sample_rate;
                        done_ev["profiling"] = profiling_json(result.profiling);
                        auto se = routes::sse_event("done", done_ev);
                        sink.write(se.data(), se.size());
                    } catch (const std::exception& e) {
                        nlohmann::json err_ev;
                        err_ev["message"] = e.what();
                        auto se = routes::sse_event("error", err_ev);
                        sink.write(se.data(), se.size());
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

    // Voice cloning endpoint (reference audio)
    server_->Post("/v1/tts/with-ref", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = nlohmann::json::parse(req.body);
            std::string text = body.value("text", "");
            std::string ref_text = body.value("ref_text", "");
            std::string ref_audio_b64 = body.value("ref_audio", "");

            if (text.empty() || ref_text.empty() || ref_audio_b64.empty()) {
                res.status = 400;
                res.set_content(routes::error_json(
                    "Missing fields: 'text', 'ref_text', and 'ref_audio' (base64 WAV or float32 PCM) are required"),
                    "application/json");
                return;
            }

            int max_tokens = body.value("max_new_tokens", 512);
            float temp     = body.value("temperature", 0.7f);
            float tp       = body.value("top_p", 0.9f);
            int top_k      = body.value("top_k", 50);
            int seed       = body.value("seed", 42);
            std::string response_format =
                body.value("response_format", body.value("format", std::string("wav")));
            std::string ref_audio_format = body.value("ref_audio_format", std::string("auto"));
            int ref_sample_rate = body.value("ref_sample_rate", pipeline_->sample_rate());

            const auto ref_decode_start = std::chrono::steady_clock::now();
            routes::DecodedAudio ref_audio = routes::decode_audio_base64(
                ref_audio_b64, pipeline_->sample_rate(), ref_sample_rate, ref_audio_format);
            const double ref_decode_ms = elapsed_ms(ref_decode_start);

            spdlog::info("TTS with-ref: '{}' ref_text='{}' ref_format={} ref_samples={}",
                         text.substr(0, 40), ref_text.substr(0, 40),
                         ref_audio.format, ref_audio.samples.size());

            TTSOutput result;
            {
                std::lock_guard<std::mutex> lock(inference_mutex_);
                result = pipeline_->run_with_ref_audio(
                    ref_audio.samples.data(),
                    static_cast<int>(ref_audio.samples.size()), ref_text, text,
                    max_tokens, temp, tp, top_k, seed);
            }

            if (result.audio_samples.empty()) {
                res.status = 500;
                res.set_content(routes::error_json("Inference produced no audio"), "application/json");
                return;
            }

            nlohmann::json resp;
            const auto encode_start = std::chrono::steady_clock::now();
            if (response_format == "pcm_f32" || response_format == "float32") {
                resp["audio"] = routes::encode_pcm_f32_base64(
                    result.audio_samples.data(),
                    static_cast<int>(result.audio_samples.size()));
                resp["audio_format"] = "pcm_f32";
                resp["audio_bytes"] = static_cast<int>(result.audio_samples.size() * sizeof(float));
            } else if (response_format == "wav") {
                resp["audio"] = routes::encode_wav_base64(
                    result.audio_samples.data(),
                    static_cast<int>(result.audio_samples.size()),
                    result.sample_rate);
                resp["audio_format"] = "wav";
                resp["audio_bytes"] = static_cast<int>(44 + result.audio_samples.size() * 2);
            } else {
                res.status = 400;
                res.set_content(routes::error_json(
                    "Unsupported response format: use 'wav' or 'pcm_f32'"), "application/json");
                return;
            }
            result.profiling.ref_decode_ms = ref_decode_ms;
            result.profiling.response_encode_ms = elapsed_ms(encode_start);
            resp["sample_rate"] = result.sample_rate;
            resp["num_samples"] = static_cast<int>(result.audio_samples.size());
            resp["duration_s"] = static_cast<double>(result.audio_samples.size()) / result.sample_rate;
            resp["profiling"] = profiling_json(result.profiling);
            res.set_content(resp.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(routes::error_json(e.what()), "application/json");
            spdlog::error("TTS with-ref error: {}", e.what());
        }
    });

    // Voice cloning streaming endpoint
    server_->Post("/v1/tts/with-ref/stream", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = nlohmann::json::parse(req.body);
            std::string text = body.value("text", "");
            std::string ref_text = body.value("ref_text", "");
            std::string ref_audio_b64 = body.value("ref_audio", "");

            if (text.empty() || ref_text.empty() || ref_audio_b64.empty()) {
                res.status = 400;
                res.set_content(routes::error_json(
                    "Missing fields: 'text', 'ref_text', and 'ref_audio' (base64 WAV or float32 PCM) are required"),
                    "application/json");
                return;
            }

            int max_tokens = body.value("max_new_tokens", 512);
            float temp     = body.value("temperature", 0.7f);
            float tp       = body.value("top_p", 0.9f);
            int top_k      = body.value("top_k", 50);
            int seed       = body.value("seed", 42);
            int chunk_length = body.value("chunk_length", 0);
            int history_frames = body.value("history_frames", 96);
            std::string ref_audio_format = body.value("ref_audio_format", std::string("auto"));
            int ref_sample_rate = body.value("ref_sample_rate", pipeline_->sample_rate());

            const auto ref_decode_start = std::chrono::steady_clock::now();
            routes::DecodedAudio ref_audio = routes::decode_audio_base64(
                ref_audio_b64, pipeline_->sample_rate(), ref_sample_rate, ref_audio_format);
            const double ref_decode_ms = elapsed_ms(ref_decode_start);

            spdlog::info("TTS with-ref stream: '{}' ref_text='{}' ref_format={} ref_samples={}",
                         text.substr(0, 40), ref_text.substr(0, 40),
                         ref_audio.format, ref_audio.samples.size());

            int sr = pipeline_->sample_rate();
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_header("X-Accel-Buffering", "no");

            res.set_chunked_content_provider(
                "text/event-stream",
                [this, text, ref_text, ref_audio = std::move(ref_audio),
                 max_tokens, temp, tp, top_k, seed, chunk_length, history_frames, sr, ref_decode_ms](
                    size_t, httplib::DataSink& sink) -> bool {

                    StreamCallback cb;
                    cb.on_audio_chunk = [&sink, sr, chunk_idx = 0](const float* samples, int count) mutable {
                        nlohmann::json ev;
                        ev["data"] = routes::encode_pcm_f32_base64(samples, count);
                        ev["audio_format"] = "pcm_f32";
                        ev["sample_rate"] = sr;
                        ev["chunk_index"] = chunk_idx++;
                        ev["num_samples"] = count;
                        auto se = routes::sse_event("audio", ev);
                        sink.write(se.data(), se.size());
                    };

                    try {
                        TTSOutput result;
                        {
                            std::lock_guard<std::mutex> lock(inference_mutex_);
                            result = pipeline_->run_with_ref_audio_streaming(
                                ref_audio.samples.data(),
                                static_cast<int>(ref_audio.samples.size()), ref_text, text,
                                max_tokens, temp, tp, top_k, seed, chunk_length, history_frames, cb);
                        }

                        nlohmann::json done_ev;
                        done_ev["total_samples"] = static_cast<int>(result.audio_samples.size());
                        done_ev["duration"] = static_cast<double>(result.audio_samples.size()) / result.sample_rate;
                        done_ev["sample_rate"] = result.sample_rate;
                        result.profiling.ref_decode_ms = ref_decode_ms;
                        done_ev["profiling"] = profiling_json(result.profiling);
                        auto se = routes::sse_event("done", done_ev);
                        sink.write(se.data(), se.size());
                    } catch (const std::exception& e) {
                        auto se = routes::sse_event("error", {{"message", e.what()}});
                        sink.write(se.data(), se.size());
                        spdlog::error("TTS with-ref stream error: {}", e.what());
                    }

                    sink.done();
                    return true;
                });
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(routes::error_json(e.what()), "application/json");
            spdlog::error("TTS with-ref stream setup error: {}", e.what());
        }
    });

    // Server info
    server_->Get("/v1/info", [](const httplib::Request&, httplib::Response& res) {
        nlohmann::json j;
        j["engine"] = "fish-audio-cpp";
        j["version"] = "0.1.0";
        j["backend"] = "CUDA";
        res.set_content(j.dump(), "application/json");
    });

    spdlog::info("HttpServer listening on {}:{}", cfg_.host, cfg_.port);
    server_->new_task_queue = [this]() -> httplib::TaskQueue* {
        return new httplib::ThreadPool(cfg_.num_threads);
    };
    server_->listen(cfg_.host.c_str(), cfg_.port);
}

void HttpServer::stop() {
    if (server_) {
        server_->stop();
    }
}

}  // namespace fish
