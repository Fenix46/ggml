#include "gc_server.h"
#include "gc_engine.h"

#include "vendor/httplib.h"
#include "vendor/json.hpp"

#include <chrono>
#include <ctime>
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

static std::string gc__token_to_text(int32_t tok, const gc_server_detokenize_fn_t & detok) {
    if (detok) {
        try {
            return detok(tok);
        } catch (...) {
            // Fallback keeps serving responses even if tokenizer cannot
            // detokenize a specific sampled id.
        }
    }
    return "<tok:" + std::to_string(tok) + ">";
}

static std::string gc__detokenize_full(
        const std::vector<int32_t> & toks,
        const gc_server_detokenize_ids_fn_t & detok_ids,
        const gc_server_detokenize_fn_t & detok_single) {
    if (detok_ids) {
        try {
            return detok_ids(toks);
        } catch (...) {
            // fallback below
        }
    }
    std::string out;
    out.reserve(toks.size() * 4);
    for (int32_t tok : toks) {
        out += gc__token_to_text(tok, detok_single);
    }
    return out;
}

static int64_t gc__unix_time_now() {
    return static_cast<int64_t>(std::time(nullptr));
}

static bool gc__validate_chat_request(const json & in, std::string & err) {
    if (!in.contains("model") || !in["model"].is_string()) {
        err = "missing or invalid 'model'";
        return false;
    }
    if (!in.contains("messages") || !in["messages"].is_array()) {
        err = "missing or invalid 'messages'";
        return false;
    }
    if (in.contains("stream") && !in["stream"].is_boolean()) {
        err = "invalid 'stream': expected boolean";
        return false;
    }
    if (in.contains("max_tokens") && !in["max_tokens"].is_number_integer()) {
        err = "invalid 'max_tokens': expected integer";
        return false;
    }
    return true;
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

gc_engine_server_runtime_t::gc_engine_server_runtime_t(
        gc_engine_t * engine,
        gc_server_tokenize_fn_t tokenize_fn,
        int32_t eos_token_id)
    : engine_(engine), tokenize_fn_(std::move(tokenize_fn)), eos_token_id_(eos_token_id) {}

std::string gc_engine_server_runtime_t::submit(const gc_server_request_t & req) {
    std::lock_guard<std::mutex> lg(mu_);
    if (!engine_) return "";

    auto toks = tokenize_fn_ ? tokenize_fn_(req.prompt_text) : std::vector<int32_t>{};
    if (toks.empty()) {
        // Keep request valid even with an empty prompt.
        toks.push_back(1);
    }

    gc_sampling_params_t sp;
    sp.max_tokens = req.max_tokens > 0 ? req.max_tokens : 16;
    sp.eos_token_id = eos_token_id_;
    sp.ignore_eos = false;

    engine_->add_request(std::make_unique<gc_request_t>(req.id, toks, sp));
    return req.id;
}

bool gc_engine_server_runtime_t::cancel(const std::string & req_id) {
    std::lock_guard<std::mutex> lg(mu_);
    if (!engine_) return false;
    engine_->abort_request(req_id);
    return true;
}

std::vector<gc_server_token_event_t> gc_engine_server_runtime_t::step() {
    std::lock_guard<std::mutex> lg(mu_);
    std::vector<gc_server_token_event_t> out;
    if (!engine_ || !engine_->has_work()) return out;

    const gc_step_output_t s = engine_->step();
    out.reserve(s.outputs.size());
    for (const auto & o : s.outputs) {
        gc_server_token_event_t ev;
        ev.req_id = o.req_id;
        ev.token = o.token;
        ev.finished = o.finished;
        out.push_back(ev);
    }
    return out;
}

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
            res.set_content(R"({"error":{"message":"invalid JSON body","type":"invalid_request_error"}})", "application/json");
            return;
        }

        std::string err_msg;
        if (!gc__validate_chat_request(in, err_msg)) {
            res.status = 400;
            json e = {{"error", {{"message", err_msg}, {"type", "invalid_request_error"}}}};
            res.set_content(e.dump(), "application/json");
            return;
        }

        const bool stream = in.value("stream", false);
        gc_server_request_t reqv;
        reqv.id = "req-" + std::to_string(impl_->next_req_id.fetch_add(1));
        reqv.model = in["model"].get<std::string>();
        reqv.prompt_text = gc__extract_prompt(in);
        reqv.max_tokens = in.value("max_tokens", 16);
        if (reqv.max_tokens <= 0) reqv.max_tokens = 1;
        if (reqv.max_tokens > 4096) reqv.max_tokens = 4096;
        reqv.stream = stream;

        const std::string rid = impl_->runtime->submit(reqv);
        if (rid.empty()) {
            res.status = 500;
            res.set_content(R"({"error":{"message":"runtime submit failed","type":"server_error"}})", "application/json");
            return;
        }
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
                        {"created", gc__unix_time_now()},
                        {"model", "gc-server"},
                        {"choices", {{{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}}}
                    };
                    std::string role_evt = "data: " + role_chunk.dump() + "\n\n";
                    sink.write(role_evt.data(), role_evt.size());

                    std::vector<int32_t> emitted_tokens;
                    std::string rendered_text;
                    int idle_polls = 0;
                    const int max_idle_polls = 1500; // ~15s at 10ms
                    bool finished = false;
                    while (!finished && idle_polls < max_idle_polls) {
                        bool had_event = false;
                        auto evs = impl_->runtime->step();
                        for (const auto & ev : evs) {
                            if (ev.req_id != rid) continue;
                            had_event = true;
                            if (ev.token >= 0) {
                                emitted_tokens.push_back(ev.token);
                                const std::string full = gc__detokenize_full(
                                    emitted_tokens, params_.detokenize_ids_fn, params_.detokenize_fn);
                                std::string delta = full;
                                if (delta.rfind(rendered_text, 0) == 0) {
                                    delta.erase(0, rendered_text.size());
                                }
                                rendered_text = full;
                                if (delta.empty()) {
                                    continue;
                                }
                                json tok_chunk = {
                                    {"id", cid},
                                    {"object", "chat.completion.chunk"},
                                    {"created", gc__unix_time_now()},
                                    {"model", "gc-server"},
                                    {"choices", {{{"index", 0}, {"delta", {{"content", delta}}}, {"finish_reason", nullptr}}}}
                                };
                                std::string evt = "data: " + tok_chunk.dump() + "\n\n";
                                sink.write(evt.data(), evt.size());
                            }
                            finished = ev.finished;
                        }
                        if (!had_event && !finished) {
                            ++idle_polls;
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        } else {
                            idle_polls = 0;
                        }
                    }

                    json end_chunk = {
                        {"id", cid},
                        {"object", "chat.completion.chunk"},
                        {"created", gc__unix_time_now()},
                        {"model", "gc-server"},
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
        int completion_tokens = 0;
        std::vector<int32_t> emitted_tokens;
        int idle_polls = 0;
        const int max_idle_polls = 1500; // ~15s at 10ms
        bool finished = false;
        while (!finished && idle_polls < max_idle_polls) {
            bool had_event = false;
            auto evs = impl_->runtime->step();
            for (const auto & ev : evs) {
                if (ev.req_id != rid) continue;
                had_event = true;
                if (ev.token >= 0) {
                    emitted_tokens.push_back(ev.token);
                    text = gc__detokenize_full(emitted_tokens, params_.detokenize_ids_fn, params_.detokenize_fn);
                    completion_tokens++;
                }
                finished = ev.finished;
            }
            if (!had_event && !finished) {
                ++idle_polls;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                idle_polls = 0;
            }
        }

        json out = {
            {"id", cid},
            {"object", "chat.completion"},
            {"created", gc__unix_time_now()},
            {"model", reqv.model},
            {"choices", {{
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", text}}},
                {"finish_reason", "stop"}
            }}},
            {"usage", {
                {"prompt_tokens", 0},
                {"completion_tokens", completion_tokens},
                {"total_tokens", completion_tokens}
            }}
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
