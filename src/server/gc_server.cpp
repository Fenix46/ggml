#include "gc_server.h"

#include "vendor/httplib.h"
#include "vendor/json.hpp"

#include <chrono>

using json = nlohmann::json;

struct gc_server_t::impl_t {
    httplib::Server http;
};

gc_server_t::gc_server_t(const gc_server_params_t & params)
    : params_(params), impl_(new impl_t()) {
    impl_->http.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
        res.status = 200;
    });

    impl_->http.Post("/v1/chat/completions", [](const httplib::Request & req, httplib::Response & res) {
        json in;
        try {
            in = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"invalid JSON body"}})", "application/json");
            return;
        }

        if (!in.contains("model") || !in["model"].is_string()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"missing or invalid 'model'"}})", "application/json");
            return;
        }
        if (!in.contains("messages") || !in["messages"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"missing or invalid 'messages'"}})", "application/json");
            return;
        }

        const bool stream = in.value("stream", false);

        if (stream) {
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [](size_t, httplib::DataSink & sink) {
                    json c1 = {
                        {"id", "chatcmpl-gc-1"},
                        {"object", "chat.completion.chunk"},
                        {"choices", {{{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}}}
                    };
                    std::string e1 = "data: " + c1.dump() + "\n\n";
                    sink.write(e1.data(), e1.size());

                    json c2 = {
                        {"id", "chatcmpl-gc-1"},
                        {"object", "chat.completion.chunk"},
                        {"choices", {{{"index", 0}, {"delta", {{"content", "Not implemented yet."}}}, {"finish_reason", "stop"}}}}
                    };
                    std::string e2 = "data: " + c2.dump() + "\n\n";
                    sink.write(e2.data(), e2.size());

                    const char done[] = "data: [DONE]\n\n";
                    sink.write(done, sizeof(done) - 1);
                    sink.done();
                    return false;
                });
            return;
        }

        json out = {
            {"id", "chatcmpl-gc-1"},
            {"object", "chat.completion"},
            {"model", in["model"]},
            {"choices", {{
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", "Not implemented yet."}}},
                {"finish_reason", "stop"}
            }}}
        };
        res.status = 200;
        res.set_content(out.dump(), "application/json");
    });
}

gc_server_t::~gc_server_t() {
    stop();
}

bool gc_server_t::start_async() {
    if (running_.exchange(true)) {
        return false;
    }

    server_thread_ = std::thread([this]() {
        impl_->http.listen(params_.host, params_.port);
        running_.store(false);
    });

    // Give the listener a short warmup window.
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    return true;
}

void gc_server_t::stop() {
    if (!running_.load()) return;
    impl_->http.stop();
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    running_.store(false);
}
