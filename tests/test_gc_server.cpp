#include "gc_server.h"

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

    if (g_fail) {
        fprintf(stderr, "\n%d checks failed, %d passed\n", g_fail, g_pass);
        return 1;
    }
    fprintf(stdout, "PASS: %d checks\n", g_pass);
    return 0;
}

