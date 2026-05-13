#include "gc_server.h"
#include "gc_chat.h"
#include "gc_engine.h"
#include "gc_request.h"

#include "vendor/httplib.h"
#include "vendor/json.hpp"

#include <ctime>
#include <sstream>

using json = nlohmann::json;

namespace {

static std::string gc__extract_prompt(const json & in, gc_chat_template_t * tmpl) {
    const auto & msgs = in["messages"];

    std::vector<std::string> role_store, content_store;
    role_store.reserve(msgs.size());
    content_store.reserve(msgs.size());
    for (const auto & m : msgs) {
        if (!m.is_object()) continue;
        role_store.push_back(m.value("role", ""));
        content_store.push_back(m.value("content", ""));
    }

    // Build parallel arrays
    std::vector<const char*> role_ptrs, content_ptrs;
    for (size_t i = 0; i < role_store.size(); i++) {
        role_ptrs.push_back(role_store[i].c_str());
        content_ptrs.push_back(content_store[i].c_str());
    }

    if (tmpl) {
        std::string out;
        gc_status_t st = gc_chat_render(tmpl, role_ptrs.data(), content_ptrs.data(),
                                        role_ptrs.size(), "", true, &out);
        if (st == GC_OK && !out.empty()) return out;
    }

    // Fallback: plain concatenation
    std::ostringstream oss;
    for (size_t i = 0; i < role_store.size(); ++i) {
        if (!role_store[i].empty() && !content_store[i].empty()) {
            oss << role_store[i] << ": " << content_store[i] << "\n";
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
        ev.req_id   = o.req_id;
        ev.token    = o.token;
        ev.finished = o.finished;
        if (o.finished) {
            switch (o.finish_reason) {
                case GC_REQ_FINISHED_LENGTH_CAPPED:
                    ev.finish_reason = "length";
                    break;
                case GC_REQ_FINISHED_ABORTED:
                    ev.finish_reason = "abort";
                    break;
                case GC_REQ_FINISHED_STOPPED:
                default:
                    ev.finish_reason = "stop";
                    break;
            }
        }
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
        reqv.prompt_text = gc__extract_prompt(in, params_.chat_template);
        reqv.max_tokens = in.value("max_tokens", 16);
        if (reqv.max_tokens <= 0) reqv.max_tokens = 1;
        if (reqv.max_tokens > 4096) reqv.max_tokens = 4096;
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
                        {"created", gc__unix_time_now()},
                        {"model", "gc-server"},
                        {"choices", {{{"index", 0}, {"delta", {{"role", "assistant"}}}, {"finish_reason", nullptr}}}}
                    };
                    std::string role_evt = "data: " + role_chunk.dump() + "\n\n";
                    sink.write(role_evt.data(), role_evt.size());

                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
                    bool finished = false;
                    std::string final_reason = "stop";
                    while (std::chrono::steady_clock::now() < deadline && !finished) {
                        auto evs = impl_->runtime->step();
                        for (const auto & ev : evs) {
                            if (ev.req_id != rid) continue;
                            if (ev.token >= 0) {
                                json tok_chunk = {
                                    {"id", cid},
                                    {"object", "chat.completion.chunk"},
                                    {"created", gc__unix_time_now()},
                                    {"model", "gc-server"},
                                    {"choices", {{{"index", 0}, {"delta", {{"content", gc__token_to_text(ev.token, params_.detokenize_fn)}}}, {"finish_reason", nullptr}}}}
                                };
                                std::string evt = "data: " + tok_chunk.dump() + "\n\n";
                                sink.write(evt.data(), evt.size());
                            }
                            if (ev.finished) {
                                finished = true;
                                if (!ev.finish_reason.empty()) final_reason = ev.finish_reason;
                            }
                        }
                        if (!finished) std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }

                    json end_chunk = {
                        {"id", cid},
                        {"object", "chat.completion.chunk"},
                        {"created", gc__unix_time_now()},
                        {"model", "gc-server"},
                        {"choices", {{{"index", 0}, {"delta", json::object()}, {"finish_reason", final_reason}}}}
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
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
        bool finished = false;
        std::string final_reason = "stop";
        while (std::chrono::steady_clock::now() < deadline && !finished) {
            auto evs = impl_->runtime->step();
            for (const auto & ev : evs) {
                if (ev.req_id != rid) continue;
                if (ev.token >= 0) {
                    text += gc__token_to_text(ev.token, params_.detokenize_fn);
                    completion_tokens++;
                }
                if (ev.finished) {
                    finished = true;
                    if (!ev.finish_reason.empty()) final_reason = ev.finish_reason;
                }
            }
            if (!finished) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        json out = {
            {"id", cid},
            {"object", "chat.completion"},
            {"created", gc__unix_time_now()},
            {"model", reqv.model},
            {"choices", {{
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", text}}},
                {"finish_reason", final_reason}
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
