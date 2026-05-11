#include "gc_server.h"

#include "vendor/httplib.h"
#include "vendor/json.hpp"

#include <chrono>
#include <sstream>

using json = nlohmann::json;

namespace {

static std::string gc__extract_prompt(const json & in) {
    std::ostringstream oss;
    const auto & msgs = in["messages"];
    for (const auto & m : msgs) {
        if (!m.is_object()) continue;
        const std::string role = m.value("role", "");
        const std::string content = m.value("content", "");
        if (!role.empty() && !content.empty()) {
            oss << role << ": " << content << "\n";
        }
    }
    return oss.str();
}

static std::string gc__token_to_text(int32_t tok) {
    return "<tok:" + std::to_string(tok) + ">";
}

class gc_mock_runtime_t : public gc_server_runtime_t {
public:
    std::string submit(const gc_server_request_t & req) override {
        std::lock_guard<std::mutex> lg(mu_);
        state_t st;
        st.req_id = req.id;
        st.cursor = 0;
        st.tokens = {101, 102, 103};
        states_[req.id] = std::move(st);
        return req.id;
    }

    bool cancel(const std::string & req_id) override {
        std::lock_guard<std::mutex> lg(mu_);
        return states_.erase(req_id) > 0;
    }

    std::vector<gc_server_token_event_t> step() override {
        std::lock_guard<std::mutex> lg(mu_);
        std::vector<gc_server_token_event_t> out;
        out.reserve(states_.size());

        std::vector<std::string> done;
        for (auto & it : states_) {
            auto & st = it.second;
            gc_server_token_event_t ev;
            ev.req_id = st.req_id;
            if (st.cursor < st.tokens.size()) {
                ev.token = st.tokens[st.cursor++];
                ev.finished = (st.cursor >= st.tokens.size());
            } else {
                ev.token = -1;
                ev.finished = true;
            }
            out.push_back(ev);
            if (ev.finished) done.push_back(st.req_id);
        }
        for (const auto & id : done) states_.erase(id);
        return out;
    }

private:
    struct state_t {
        std::string req_id;
        std::vector<int32_t> tokens;
        size_t cursor = 0;
    };

    std::mutex mu_;
    std::unordered_map<std::string, state_t> states_;
};

} // namespace

struct gc_server_t::impl_t {
    httplib::Server http;
    std::unique_ptr<gc_mock_runtime_t> owned_mock_runtime;
    gc_server_runtime_t * runtime = nullptr;
    std::atomic<uint64_t> next_req_id{1};
};

gc_server_t::gc_server_t(const gc_server_params_t & params)
    : params_(params), impl_(new impl_t()) {
    if (params_.runtime) {
        impl_->runtime = params_.runtime;
    } else {
        impl_->owned_mock_runtime.reset(new gc_mock_runtime_t());
        impl_->runtime = impl_->owned_mock_runtime.get();
    }

    impl_->http.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
        res.status = 200;
    });

    impl_->http.Post("/v1/chat/completions", [this](const httplib::Request & req, httplib::Response & res) {
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
        gc_server_request_t reqv;
        reqv.id = "req-" + std::to_string(impl_->next_req_id.fetch_add(1));
        reqv.model = in["model"].get<std::string>();
        reqv.prompt_text = gc__extract_prompt(in);
        reqv.max_tokens = in.value("max_tokens", 16);
        reqv.stream = stream;

        const std::string rid = impl_->runtime->submit(reqv);
        const std::string cid = "chatcmpl-" + rid;

        if (stream) {
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection", "keep-alive");
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, cid, rid](size_t, httplib::DataSink & sink) {
                    json role_chunk = {
                        {"id", cid},
                        {"object", "chat.completion.chunk"},
                        {"choices", {{{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}}}
                    };
                    std::string role_evt = "data: " + role_chunk.dump() + "\n\n";
                    sink.write(role_evt.data(), role_evt.size());

                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
                    bool finished = false;
                    while (std::chrono::steady_clock::now() < deadline && !finished) {
                        auto evs = impl_->runtime->step();
                        for (const auto & ev : evs) {
                            if (ev.req_id != rid) continue;
                            if (ev.token >= 0) {
                                json tok_chunk = {
                                    {"id", cid},
                                    {"object", "chat.completion.chunk"},
                                    {"choices", {{{"index", 0}, {"delta", {{"content", gc__token_to_text(ev.token)}}}, {"finish_reason", nullptr}}}}
                                };
                                std::string evt = "data: " + tok_chunk.dump() + "\n\n";
                                sink.write(evt.data(), evt.size());
                            }
                            finished = ev.finished;
                        }
                        if (!finished) std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }

                    json end_chunk = {
                        {"id", cid},
                        {"object", "chat.completion.chunk"},
                        {"choices", {{{"index", 0}, {"delta", json::object()}, {"finish_reason", "stop"}}}}
                    };
                    std::string end_evt = "data: " + end_chunk.dump() + "\n\n";
                    sink.write(end_evt.data(), end_evt.size());

                    const char done[] = "data: [DONE]\n\n";
                    sink.write(done, sizeof(done) - 1);
                    sink.done();
                    return false;
                });
            return;
        }

        std::string text;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        bool finished = false;
        while (std::chrono::steady_clock::now() < deadline && !finished) {
            auto evs = impl_->runtime->step();
            for (const auto & ev : evs) {
                if (ev.req_id != rid) continue;
                if (ev.token >= 0) text += gc__token_to_text(ev.token);
                finished = ev.finished;
            }
            if (!finished) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        json out = {
            {"id", cid},
            {"object", "chat.completion"},
            {"model", reqv.model},
            {"choices", {{
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", text}}},
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

