// src/server/http_server.h
#pragma once
#include "engine/inference_pipeline.h"
#include <memory>
#include <mutex>
#include <string>
#include <functional>

namespace httplib { class Server; }

namespace fish {

// Simple HTTP REST server wrapping InferencePipeline.
// Uses cpp-httplib under the hood.
class HttpServer {
public:
    struct Config {
        std::string host = "0.0.0.0";
        int port = 8080;
        int num_threads = 4;
    };

    HttpServer(const Config& cfg, InferencePipeline* pipeline);
    ~HttpServer();

    // Start listening (blocking). Returns when the server shuts down.
    void start();

    // Request shutdown from another thread.
    void stop();

private:
    Config cfg_;
    InferencePipeline* pipeline_;
    std::unique_ptr<httplib::Server> server_;
    mutable std::mutex inference_mutex_;
};

}  // namespace fish
