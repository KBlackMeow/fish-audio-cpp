// src/server/http_server.cc
#include "server/http_server.h"
#include "server/routes.h"
#include <spdlog/spdlog.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace fish {

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

            // Return PCM float32 as base64
            std::string raw(reinterpret_cast<const char*>(result.audio_samples.data()),
                            result.audio_samples.size() * sizeof(float));
            std::string b64 = httplib::detail::base64_encode(raw);

            nlohmann::json resp;
            resp["audio"] = b64;
            resp["audio_bytes"] = static_cast<int>(result.audio_samples.size() * sizeof(float));
            resp["sample_rate"] = result.sample_rate;
            resp["num_samples"] = static_cast<int>(result.audio_samples.size());
            resp["duration_s"] = static_cast<double>(result.audio_samples.size())
                                 / result.sample_rate;
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
                        sink.write(routes::sse_event("progress", ev).data(),
                                   routes::sse_event("progress", ev).size());
                    };

                    cb.on_audio_chunk = [&sink, sr, chunk_idx = 0](
                        const float* samples, int count
                    ) mutable {
                        std::string raw(reinterpret_cast<const char*>(samples),
                                       count * sizeof(float));
                        std::string b64 = httplib::detail::base64_encode(raw);
                        nlohmann::json ev;
                        ev["data"] = b64;
                        ev["sample_rate"] = sr;
                        ev["chunk_index"] = chunk_idx++;
                        ev["num_samples"] = count;
                        sink.write(routes::sse_event("audio", ev).data(),
                                   routes::sse_event("audio", ev).size());
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
                    "Missing fields: 'text', 'ref_text', and 'ref_audio' (base64 float32 PCM) are required"),
                    "application/json");
                return;
            }

            // Decode ref audio from base64
            std::string ref_raw = routes::base64_decode(ref_audio_b64);
            int ref_num = static_cast<int>(ref_raw.size()) / static_cast<int>(sizeof(float));

            int max_tokens = body.value("max_new_tokens", 512);
            float temp     = body.value("temperature", 0.7f);
            float tp       = body.value("top_p", 0.9f);
            int top_k      = body.value("top_k", 50);
            int seed       = body.value("seed", 42);

            spdlog::info("TTS with-ref: '{}' ref_text='{}' ref_samples={}",
                         text.substr(0, 40), ref_text.substr(0, 40), ref_num);

            TTSOutput result;
            {
                std::lock_guard<std::mutex> lock(inference_mutex_);
                result = pipeline_->run_with_ref_audio(
                    reinterpret_cast<const float*>(ref_raw.data()),
                    ref_num, ref_text, text,
                    max_tokens, temp, tp, top_k, seed);
            }

            if (result.audio_samples.empty()) {
                res.status = 500;
                res.set_content(routes::error_json("Inference produced no audio"), "application/json");
                return;
            }

            std::string raw(reinterpret_cast<const char*>(result.audio_samples.data()),
                           result.audio_samples.size() * sizeof(float));
            std::string b64 = httplib::detail::base64_encode(raw);

            nlohmann::json resp;
            resp["audio"] = b64;
            resp["audio_bytes"] = static_cast<int>(result.audio_samples.size() * sizeof(float));
            resp["sample_rate"] = result.sample_rate;
            resp["num_samples"] = static_cast<int>(result.audio_samples.size());
            resp["duration_s"] = static_cast<double>(result.audio_samples.size()) / result.sample_rate;
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
                    "Missing fields: 'text', 'ref_text', and 'ref_audio' (base64 float32 PCM) are required"),
                    "application/json");
                return;
            }

            std::string ref_raw = routes::base64_decode(ref_audio_b64);
            int ref_num = static_cast<int>(ref_raw.size()) / static_cast<int>(sizeof(float));

            int max_tokens = body.value("max_new_tokens", 512);
            float temp     = body.value("temperature", 0.7f);
            float tp       = body.value("top_p", 0.9f);
            int top_k      = body.value("top_k", 50);
            int seed       = body.value("seed", 42);

            spdlog::info("TTS with-ref stream: '{}' ref_text='{}'",
                         text.substr(0, 40), ref_text.substr(0, 40));

            int sr = pipeline_->sample_rate();
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_header("X-Accel-Buffering", "no");

            res.set_chunked_content_provider(
                "text/event-stream",
                [this, text, ref_text, ref_raw, ref_num, max_tokens, temp, tp, top_k, seed, sr](
                    size_t, httplib::DataSink& sink) -> bool {

                    StreamCallback cb;
                    cb.on_audio_chunk = [&sink, sr, chunk_idx = 0](const float* samples, int count) mutable {
                        std::string r(reinterpret_cast<const char*>(samples), count * sizeof(float));
                        std::string b = httplib::detail::base64_encode(r);
                        nlohmann::json ev;
                        ev["data"] = b;
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
                                reinterpret_cast<const float*>(ref_raw.data()),
                                ref_num, ref_text, text,
                                max_tokens, temp, tp, top_k, seed, cb);
                        }

                        nlohmann::json done_ev;
                        done_ev["total_samples"] = static_cast<int>(result.audio_samples.size());
                        done_ev["duration"] = static_cast<double>(result.audio_samples.size()) / result.sample_rate;
                        done_ev["sample_rate"] = result.sample_rate;
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
