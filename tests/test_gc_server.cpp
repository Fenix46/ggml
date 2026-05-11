#include "gc_server.h"
#include "gc_engine.h"

#include "../src/server/vendor/httplib.h"
#include "../src/server/vendor/json.hpp"

#include <cassert>
#include <cstdio>
#include <string>

using json = nlohmann::json;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

class ServerTestRunner : public gc_model_runner_t {
public:
    int num_layers() const override { return 1; }
    gc_model_output_t execute(const gc_batch_t & batch) override {
        gc_model_output_t out;
        for (const auto & e : batch.entries) {
            out.req_ids.push_back(e.req_id);
            out.sampled_tokens.push_back(e.is_prefill ? -1 : 2); // EOS token
        }
        return out;
    }
};

int main() {
    gc_server_params_t p;
    p.host = "127.0.0.1";
    p.port = 18123;

    gc_server_t server(p);
    CHECK(server.start_async());

    httplib::Client cli(p.host, p.port);

    // Health endpoint
    auto health = cli.Get("/health");
    CHECK(health != nullptr);
    CHECK(health->status == 200);
    CHECK(health->body.find("\"ok\"") != std::string::npos);

    // Non-stream chat completion
    json req = {
        {"model", "test-model"},
        {"messages", {{{"role", "user"}, {"content", "hi"}}}},
        {"stream", false}
    };
    auto resp = cli.Post("/v1/chat/completions", req.dump(), "application/json");
    CHECK(resp != nullptr);
    CHECK(resp->status == 200);
    auto out = json::parse(resp->body);
    CHECK(out["object"] == "chat.completion");
    CHECK(out["choices"].is_array());

    // Stream chat completion
    req["stream"] = true;
    auto sresp = cli.Post("/v1/chat/completions", req.dump(), "application/json");
    CHECK(sresp != nullptr);
    CHECK(sresp->status == 200);
    CHECK(sresp->body.find("data: [DONE]") != std::string::npos);

    server.stop();
    CHECK(!server.is_running());

    // Engine-backed runtime path (no slot/context semantics)
    gc_engine_params_t ep;
    ep.sched.num_blocks = 64;
    ep.sched.block_size = 4;
    ep.sched.max_num_running = 8;
    ep.sched.max_num_tokens = 64;
    ep.sched.max_model_len = 256;
    ep.sched.enable_prefix_cache = true;

    ServerTestRunner runner;
    gc_engine_t eng(ep, &runner);
    gc_engine_server_runtime_t rt(&eng, [](const std::string & prompt) {
        std::vector<int32_t> toks;
        toks.reserve(prompt.size());
        for (char c : prompt) toks.push_back((int32_t)(unsigned char)c + 1);
        return toks;
    });

    gc_server_params_t p2;
    p2.host = "127.0.0.1";
    p2.port = 18124;
    p2.runtime = &rt;

    gc_server_t server2(p2);
    CHECK(server2.start_async());

    httplib::Client cli2(p2.host, p2.port);
    auto resp2 = cli2.Post("/v1/chat/completions", req.dump(), "application/json");
    CHECK(resp2 != nullptr);
    CHECK(resp2->status == 200);
    auto out2 = json::parse(resp2->body);
    CHECK(out2["object"] == "chat.completion");
    CHECK(out2["choices"].is_array());
    CHECK(out2["choices"][0]["message"]["role"] == "assistant");

    server2.stop();
    CHECK(!server2.is_running());

    if (g_fail) {
        fprintf(stderr, "\n%d checks failed, %d passed\n", g_fail, g_pass);
        return 1;
    }
    fprintf(stdout, "PASS: %d checks\n", g_pass);
    return 0;
}
