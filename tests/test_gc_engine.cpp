// test_gc_engine.cpp — unit tests for Phase 8 engine

#include "gc_engine.h"

#include <cassert>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

// ── Mock runner ───────────────────────────────────────────────────────────────
// Returns token_id = entry_index + 100 for decode steps, -1 for prefill-only.

class MockRunner : public gc_model_runner_t {
public:
    int num_layers() const override { return 4; }

    gc_model_output_t execute(const gc_batch_t & batch) override {
        gc_model_output_t out;
        for (const auto & e : batch.entries) {
            out.req_ids.push_back(e.req_id);
            // Produce a token only for decode steps (is_prefill == false).
            // For simplicity return token 42 for every decode entry.
            out.sampled_tokens.push_back(e.is_prefill ? -1 : 42);
        }
        return out;
    }
};

// ── Helpers ───────────────────────────────────────────────────────────────────

static gc_engine_params_t base_params() {
    gc_engine_params_t p;
    p.sched.num_blocks        = 64;
    p.sched.block_size        = 4;
    p.sched.max_num_running   = 8;
    p.sched.max_num_tokens    = 64;
    p.sched.max_model_len     = 256;
    p.sched.policy            = GC_SCHED_FCFS;
    p.sched.enable_prefix_cache = true;
    return p;
}

static std::unique_ptr<gc_request_t> make_req(
        const std::string & id, int n_tokens, int max_tokens = 32, int eos_token_id = 2) {
    gc_sampling_params_t sp;
    sp.max_tokens = max_tokens;
    sp.eos_token_id = eos_token_id;
    std::vector<int32_t> toks(n_tokens);
    for (int i = 0; i < n_tokens; ++i) toks[i] = i + 1;
    return std::make_unique<gc_request_t>(id, toks, sp);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_engine_constructor() {
    gc_default_model_runner_t::config_t cfg;
    cfg.num_layers = 4;
    cfg.decode_token = 17;
    gc_default_model_runner_t runner(cfg);
    auto p = base_params();
    gc_engine_t eng(p, &runner);
    CHECK(!eng.has_work());
    CHECK(eng.num_waiting() == 0);
    CHECK(eng.num_running() == 0);
    CHECK(eng.kv_usage() == 0.0f);
}

static void test_engine_add_request() {
    MockRunner runner;
    gc_engine_t eng(base_params(), &runner);

    eng.add_request(make_req("r0", 4, 8));
    CHECK(eng.has_work());
    CHECK(eng.num_waiting() == 1);
}

static void test_engine_step_empty() {
    MockRunner runner;
    gc_engine_t eng(base_params(), &runner);

    // No requests: step() should return empty output.
    auto out = eng.step();
    CHECK(!out.had_work);
    CHECK(out.total_scheduled_tokens == 0);
    CHECK(out.outputs.empty());
}

static void test_engine_step_prefill() {
    MockRunner runner;
    gc_engine_t eng(base_params(), &runner);

    eng.add_request(make_req("r0", 4, 32));
    auto out = eng.step();

    // Prefill step: had_work == true, tokens scheduled.
    CHECK(out.had_work);
    CHECK(out.total_scheduled_tokens > 0);
    // After prefill, request moves to running.
    CHECK(eng.num_running() == 1);
}

static void test_engine_step_decode() {
    gc_default_model_runner_t::config_t cfg;
    cfg.num_layers = 4;
    cfg.decode_token = 42;
    gc_default_model_runner_t runner(cfg);
    gc_engine_t eng(base_params(), &runner);

    eng.add_request(make_req("r0", 4, 32));
    // Step 1: prefill
    eng.step();
    CHECK(eng.num_running() == 1);

    // Step 2: decode — mock returns token 42 (non-EOS)
    auto out2 = eng.step();
    CHECK(out2.had_work);
    // At least one output produced
    CHECK(!out2.outputs.empty());
    CHECK(out2.outputs[0].req_id == "r0");
    CHECK(out2.outputs[0].token == 42);
    CHECK(!out2.outputs[0].finished);  // token 42 ≠ configured EOS
}

static void test_default_runner_prefill_behavior() {
    gc_default_model_runner_t::config_t cfg;
    cfg.num_layers = 3;
    cfg.decode_token = 9;
    cfg.emit_token_on_prefill = false;
    gc_default_model_runner_t runner(cfg);
    gc_batch_t b;
    gc_batch_entry_t e0;
    e0.req_id = "r0";
    e0.token_ids = {1, 2};
    e0.block_ids = {0};
    e0.num_computed = 0;
    e0.num_new_tokens = 2;
    e0.is_prefill = true;
    b.entries.push_back(e0);

    gc_batch_entry_t e1;
    e1.req_id = "r1";
    e1.token_ids = {3};
    e1.block_ids = {1};
    e1.num_computed = 2;
    e1.num_new_tokens = 1;
    e1.is_prefill = false;
    b.entries.push_back(e1);

    const gc_model_output_t out = runner.execute(b);
    CHECK(runner.num_layers() == 3);
    CHECK(out.req_ids.size() == 2);
    CHECK(out.sampled_tokens.size() == 2);
    CHECK(out.sampled_tokens[0] == -1);
    CHECK(out.sampled_tokens[1] == 9);
}

static void test_ggml_runner_sampling_basic() {
    gc_ggml_model_runner_t::config_t cfg;
    cfg.num_layers = 2;
    cfg.vocab_size = 256;
    cfg.seed = 123;
    cfg.temperature = 0.8f;
    cfg.top_k = 32;
    cfg.top_p = 0.9f;
    cfg.repetition_penalty = 1.1f;
    cfg.repetition_window = 16;
    gc_ggml_model_runner_t runner(cfg);

    gc_batch_t b;
    gc_batch_entry_t e0;
    e0.req_id = "s0";
    e0.token_ids = {1, 2, 3};
    e0.is_prefill = true;
    b.entries.push_back(e0);

    gc_batch_entry_t e1;
    e1.req_id = "s1";
    e1.token_ids = {10};
    e1.is_prefill = false;
    b.entries.push_back(e1);

    const gc_model_output_t out = runner.execute(b);
    CHECK(runner.num_layers() == 2);
    CHECK(out.req_ids.size() == 2);
    CHECK(out.sampled_tokens.size() == 2);
    CHECK(out.sampled_tokens[0] == -1);
    CHECK(out.sampled_tokens[1] >= 0);
    CHECK(out.sampled_tokens[1] < 256);
}

static void test_engine_step_finish_eos() {
    // Use a runner that always returns token 0 and configure EOS=0.
    struct EosRunner : public gc_model_runner_t {
        int num_layers() const override { return 4; }
        gc_model_output_t execute(const gc_batch_t & batch) override {
            gc_model_output_t out;
            for (auto & e : batch.entries) {
                out.req_ids.push_back(e.req_id);
                out.sampled_tokens.push_back(e.is_prefill ? -1 : 0);
            }
            return out;
        }
    } eos_runner;

    gc_engine_t eng(base_params(), &eos_runner);
    eng.add_request(make_req("r0", 4, 32, /*eos_token_id=*/0));

    // Prefill
    eng.step();
    CHECK(eng.num_running() == 1);

    // Decode: EOS → request finishes
    auto out = eng.step();
    CHECK(out.had_work);
    // Request should be marked finished
    bool found_finished = false;
    for (auto & o : out.outputs) {
        if (o.req_id == "r0" && o.finished) found_finished = true;
    }
    CHECK(found_finished);
    CHECK(!eng.has_work());
}

static void test_engine_step_finish_max_tokens() {
    struct TokRunner : public gc_model_runner_t {
        int num_layers() const override { return 4; }
        gc_model_output_t execute(const gc_batch_t & batch) override {
            gc_model_output_t out;
            for (auto & e : batch.entries) {
                out.req_ids.push_back(e.req_id);
                out.sampled_tokens.push_back(e.is_prefill ? -1 : 7);  // non-EOS
            }
            return out;
        }
    } tok_runner;

    auto p = base_params();
    gc_engine_t eng(p, &tok_runner);
    eng.add_request(make_req("r0", 4, /*max_tokens=*/2));

    // Prefill
    eng.step();

    // Decode step 1: token 7 generated, not done yet
    eng.step();
    CHECK(eng.num_running() == 1);

    // Decode step 2: second token → max_tokens reached
    eng.step();
    CHECK(eng.num_running() == 0);
    CHECK(!eng.has_work());
}

static void test_engine_abort_waiting() {
    MockRunner runner;
    gc_engine_t eng(base_params(), &runner);

    eng.add_request(make_req("r0", 4, 8));
    eng.add_request(make_req("r1", 4, 8));
    eng.abort_request("r0");

    CHECK(eng.num_waiting() == 1);
    // Step should only process r1
    auto out = eng.step();
    CHECK(out.had_work);
    CHECK(eng.num_running() == 1);
    CHECK(eng.num_waiting() == 0);
}

static void test_engine_abort_running() {
    MockRunner runner;
    gc_engine_t eng(base_params(), &runner);

    eng.add_request(make_req("r0", 4, 32));
    eng.step();  // prefill → running
    CHECK(eng.num_running() == 1);

    eng.abort_request("r0");
    CHECK(eng.num_running() == 0);
    CHECK(!eng.has_work());
}

static void test_engine_multiple_requests() {
    MockRunner runner;
    gc_engine_t eng(base_params(), &runner);

    for (int i = 0; i < 4; ++i) {
        eng.add_request(make_req("r" + std::to_string(i), 4, 16));
    }
    CHECK(eng.num_waiting() == 4);

    auto out = eng.step();  // prefill all 4
    CHECK(out.had_work);
    CHECK(eng.num_running() == 4);
}

static void test_engine_kv_usage_changes() {
    MockRunner runner;
    gc_engine_t eng(base_params(), &runner);

    CHECK(eng.kv_usage() == 0.0f);
    eng.add_request(make_req("r0", 4, 8));
    eng.step();
    CHECK(eng.kv_usage() > 0.0f);
}

static void test_engine_build_batch_new_req() {
    // Verify batch entries for new (prefill) requests have is_prefill=true.
    struct InspectRunner : public gc_model_runner_t {
        bool saw_prefill = false;
        int num_layers() const override { return 4; }
        gc_model_output_t execute(const gc_batch_t & batch) override {
            gc_model_output_t out;
            for (auto & e : batch.entries) {
                if (e.is_prefill) saw_prefill = true;
                out.req_ids.push_back(e.req_id);
                out.sampled_tokens.push_back(e.is_prefill ? -1 : 5);
            }
            return out;
        }
    } insp;

    gc_engine_t eng(base_params(), &insp);
    eng.add_request(make_req("r0", 4, 32));
    eng.step();
    CHECK(insp.saw_prefill);
}

static void test_engine_build_batch_token_ids() {
    // Verify token_ids in prefill batch match prompt tokens.
    struct TokenCheckRunner : public gc_model_runner_t {
        std::vector<int32_t> seen_tokens;
        int num_layers() const override { return 4; }
        gc_model_output_t execute(const gc_batch_t & batch) override {
            gc_model_output_t out;
            for (auto & e : batch.entries) {
                if (e.is_prefill) seen_tokens = e.token_ids;
                out.req_ids.push_back(e.req_id);
                out.sampled_tokens.push_back(e.is_prefill ? -1 : 5);
            }
            return out;
        }
    } tc;

    gc_engine_t eng(base_params(), &tc);
    // Prompt tokens: 1,2,3,4
    eng.add_request(make_req("r0", 4, 32));
    eng.step();
    CHECK(tc.seen_tokens.size() == 4);
    CHECK(tc.seen_tokens[0] == 1);
    CHECK(tc.seen_tokens[3] == 4);
}

static void test_engine_step_no_work_after_empty() {
    MockRunner runner;
    gc_engine_t eng(base_params(), &runner);

    auto out = eng.step();
    CHECK(!out.had_work);
    // Repeated empty steps safe.
    out = eng.step();
    CHECK(!out.had_work);
}

static void test_engine_chunked_prefill() {
    MockRunner runner;
    auto p = base_params();
    p.sched.max_num_tokens = 4;  // only 4 tokens per step
    gc_engine_t eng(p, &runner);

    // 16-token prompt: needs 4 prefill steps
    eng.add_request(make_req("r0", 16, 64));
    auto out = eng.step();
    CHECK(out.had_work);
    CHECK(out.total_scheduled_tokens <= 4);
    CHECK(eng.num_running() == 1);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_engine_constructor();
    test_engine_add_request();
    test_engine_step_empty();
    test_engine_step_prefill();
    test_engine_step_decode();
    test_default_runner_prefill_behavior();
    test_ggml_runner_sampling_basic();
    test_engine_step_finish_eos();
    test_engine_step_finish_max_tokens();
    test_engine_abort_waiting();
    test_engine_abort_running();
    test_engine_multiple_requests();
    test_engine_kv_usage_changes();
    test_engine_build_batch_new_req();
    test_engine_build_batch_token_ids();
    test_engine_step_no_work_after_empty();
    test_engine_chunked_prefill();

    fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
