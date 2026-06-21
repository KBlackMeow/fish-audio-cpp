// src/server/http_server.cc
#include "server/http_server.h"
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
                nlohmann::json err;
                err["error"] = "Missing 'text' field";
                res.set_content(err.dump(), "application/json");
                return;
            }

            int max_tokens  = body.value("max_new_tokens", 512);
            float temp      = body.value("temperature", 0.7f);
            float top_p     = body.value("top_p", 0.9f);
            int top_k       = body.value("top_k", 50);
            int seed        = body.value("seed", 42);

            spdlog::info("TTS: '{}' (max_tokens={})", text.substr(0, 60), max_tokens);

            auto result = pipeline_->run(text, max_tokens, temp, top_p, top_k, seed);

            if (result.audio_samples.empty()) {
                res.status = 500;
                nlohmann::json err;
                err["error"] = "Inference produced no audio";
                res.set_content(err.dump(), "application/json");
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
            nlohmann::json err;
            err["error"] = e.what();
            res.set_content(err.dump(), "application/json");
            spdlog::error("TTS error: {}", e.what());
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
    server_->listen(cfg_.host.c_str(), cfg_.port);
}

void HttpServer::stop() {
    if (server_) {
        server_->stop();
    }
}

}  // namespace fish
