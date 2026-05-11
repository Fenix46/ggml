// test_gc_scheduler.cpp — unit tests for Phase 7 scheduler

#include "gc_scheduler.h"

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

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::unique_ptr<gc_request_t> make_req(
        const std::string & id,
        int n_tokens,
        int max_tokens = 32,
        int priority   = 0,
        int eos_token_id = 2) {
    gc_sampling_params_t sp;
    sp.max_tokens = max_tokens;
    sp.eos_token_id = eos_token_id;
    std::vector<int32_t> toks(n_tokens);
    for (int i = 0; i < n_tokens; ++i) toks[i] = i + 1;
    return std::make_unique<gc_request_t>(id, toks, sp, priority, 0.0);
}

static gc_scheduler_params_t base_params() {
    gc_scheduler_params_t p;
    p.num_blocks        = 64;
    p.block_size        = 4;
    p.max_num_running   = 8;
    p.max_num_tokens    = 64;
    p.max_model_len     = 256;
    p.policy            = GC_SCHED_FCFS;
    p.enable_prefix_cache = true;
    return p;
}

// ── Request tests ─────────────────────────────────────────────────────────────

static void test_request_basic() {
    gc_sampling_params_t sp;
    sp.max_tokens = 16;
    std::vector<int32_t> toks = {1, 2, 3, 4, 5, 6, 7, 8};
    gc_request_t req("r0", toks, sp);
    CHECK(req.num_prompt_tokens() == 8);
    CHECK(req.num_output_tokens() == 0);
    CHECK(req.num_tokens() == 8);
    CHECK(req.max_tokens() == 16);
    CHECK(req.status == GC_REQ_WAITING);
}

static void test_request_block_hashes() {
    gc_sampling_params_t sp;
    sp.max_tokens = 4;
    // 8 tokens, block_size=4 → 2 full blocks
    std::vector<int32_t> toks = {1, 2, 3, 4, 5, 6, 7, 8};
    gc_request_t req("r1", toks, sp);
    req.update_block_hashes(4);
    CHECK((int)req.block_hashes.size() == 2);
    CHECK(req.block_hashes[0] != GC_BLOCK_HASH_NONE);
    CHECK(req.block_hashes[1] != req.block_hashes[0]);
}

static void test_request_block_hashes_partial() {
    gc_sampling_params_t sp;
    sp.max_tokens = 4;
    // 5 tokens, block_size=4 → 1 full block, 1 partial
    std::vector<int32_t> toks = {1, 2, 3, 4, 5};
    gc_request_t req("r2", toks, sp);
    req.update_block_hashes(4);
    CHECK((int)req.block_hashes.size() == 1);
}

static void test_request_append_output() {
    gc_sampling_params_t sp; sp.max_tokens = 16;
    std::vector<int32_t> toks = {1, 2, 3};
    gc_request_t req("r3", toks, sp);
    req.update_block_hashes(4);
    CHECK((int)req.block_hashes.size() == 0);  // 3 tokens, no full block

    req.append_output_token(4, 4);  // now 4 tokens → 1 full block
    CHECK((int)req.block_hashes.size() == 1);
    CHECK(req.num_output_tokens() == 1);
    CHECK(req.num_tokens() == 4);
}

static void test_request_priority_order() {
    gc_sampling_params_t sp; sp.max_tokens = 4;
    gc_request_t r1("r1", {1}, sp, /*priority=*/2, 1.0);
    gc_request_t r2("r2", {1}, sp, /*priority=*/1, 2.0);
    gc_request_t r3("r3", {1}, sp, /*priority=*/1, 1.0);
    // r3 < r2 < r1 (lower priority value = higher priority; earlier arrival breaks ties)
    CHECK(r3 < r2);
    CHECK(r2 < r1);
    CHECK(!(r1 < r3));
}

// ── Scheduler: basic scheduling ───────────────────────────────────────────────

static void test_scheduler_schedule_new_req() {
    auto p = base_params();
    gc_scheduler_t sched(p);

    sched.add_request(make_req("r0", 8, 16));
    CHECK(sched.num_waiting() == 1);
    CHECK(sched.num_running() == 0);

    auto out = sched.schedule();
    CHECK((int)out.new_reqs.size() == 1);
    CHECK(out.new_reqs[0].req_id == "r0");
    CHECK(out.total_scheduled_tokens > 0);
    CHECK(sched.num_running() == 1);
    CHECK(sched.num_waiting() == 0);
}

static void test_scheduler_schedule_multiple() {
    auto p = base_params();
    gc_scheduler_t sched(p);

    for (int i = 0; i < 4; ++i) {
        sched.add_request(make_req("r" + std::to_string(i), 4, 8));
    }
    auto out = sched.schedule();
    CHECK((int)out.new_reqs.size() == 4);
    CHECK(sched.num_running() == 4);
    CHECK(sched.num_waiting() == 0);
}

static void test_scheduler_respects_max_running() {
    auto p = base_params();
    p.max_num_running = 2;
    gc_scheduler_t sched(p);

    for (int i = 0; i < 4; ++i) {
        sched.add_request(make_req("r" + std::to_string(i), 4, 8));
    }
    auto out = sched.schedule();
    CHECK(sched.num_running() <= 2);
    CHECK(sched.num_waiting() >= 2);
}

static void test_scheduler_respects_token_budget() {
    auto p = base_params();
    p.max_num_tokens = 8;  // only 8 tokens per step
    gc_scheduler_t sched(p);

    // One request with 16 tokens: chunked prefill
    sched.add_request(make_req("r0", 16, 32));
    auto out = sched.schedule();
    CHECK(out.total_scheduled_tokens <= 8);
    // Request is now running with partial schedule
    CHECK(sched.num_running() == 1);
}

// ── Scheduler: continuation (running reqs) ────────────────────────────────────

static void test_scheduler_running_continuation() {
    auto p = base_params();
    p.max_num_tokens = 4;
    gc_scheduler_t sched(p);

    sched.add_request(make_req("r0", 4, 16));
    auto out1 = sched.schedule();
    CHECK(!out1.new_reqs.empty());
    CHECK(sched.num_running() == 1);

    // Simulate output token
    sched.update_from_output({{"r0", 10}});
    CHECK(sched.num_running() == 1);  // not finished yet

    auto out2 = sched.schedule();
    // r0 is now in cached_reqs (running continuation)
    bool found = false;
    for (auto & e : out2.cached_reqs) {
        if (e.req_id == "r0") found = true;
    }
    CHECK(found);
}

// ── Scheduler: finish detection ───────────────────────────────────────────────

static void test_scheduler_finish_eos() {
    auto p = base_params();
    gc_scheduler_t sched(p);

    sched.add_request(make_req("r0", 4, 32));
    sched.schedule();
    CHECK(sched.num_running() == 1);

    // Token id = 2 (configured EOS) → finish
    sched.update_from_output({{"r0", 2}});
    CHECK(sched.num_running() == 0);

    auto out = sched.schedule();
    CHECK(out.finished_req_ids.count("r0") == 1);
}

static void test_scheduler_finish_max_tokens() {
    auto p = base_params();
    p.max_num_tokens = 4;
    gc_scheduler_t sched(p);

    sched.add_request(make_req("r0", 4, 2));  // max_tokens=2
    sched.schedule();

    // Generate 2 non-EOS tokens to hit max_tokens
    sched.update_from_output({{"r0", 5}});
    CHECK(sched.num_running() == 1);
    sched.update_from_output({{"r0", 6}});
    CHECK(sched.num_running() == 0);
}

// ── Scheduler: abort ─────────────────────────────────────────────────────────

static void test_scheduler_abort_waiting() {
    auto p = base_params();
    gc_scheduler_t sched(p);

    sched.add_request(make_req("r0", 4, 8));
    sched.add_request(make_req("r1", 4, 8));
    sched.abort_request("r0");
    CHECK(sched.num_waiting() == 1);

    auto out = sched.schedule();
    CHECK(out.new_reqs.size() == 1);
    CHECK(out.new_reqs[0].req_id == "r1");
}

static void test_scheduler_abort_running() {
    auto p = base_params();
    gc_scheduler_t sched(p);

    sched.add_request(make_req("r0", 4, 8));
    sched.schedule();
    CHECK(sched.num_running() == 1);

    sched.abort_request("r0");
    CHECK(sched.num_running() == 0);
    CHECK(sched.kv_usage() == 0.0f);
}

// ── Scheduler: preemption ─────────────────────────────────────────────────────

static void test_scheduler_preemption_on_oom() {
    auto p = base_params();
    p.num_blocks     = 4;  // only 3 usable (1 null)
    p.block_size     = 4;
    p.max_num_tokens = 128;
    gc_scheduler_t sched(p);

    // Two requests needing 2 blocks each = 4 blocks total; only 3 available.
    sched.add_request(make_req("r0", 8, 32));   // 2 blocks
    sched.add_request(make_req("r1", 8, 32));   // 2 blocks

    // First schedule: r0 admitted (2 blocks), r1 can't fit → either preempted or not admitted
    auto out = sched.schedule();

    // At least r0 was scheduled
    bool r0_scheduled = false;
    for (auto & nr : out.new_reqs) if (nr.req_id == "r0") r0_scheduled = true;
    for (auto & cr : out.cached_reqs) if (cr.req_id == "r0") r0_scheduled = true;
    CHECK(r0_scheduled || !out.preempted_req_ids.empty());
}

// ── Scheduler: priority policy ────────────────────────────────────────────────

static void test_scheduler_priority_order() {
    auto p = base_params();
    p.policy = GC_SCHED_PRIORITY;
    p.max_num_tokens = 4;   // only 1 request per step to see ordering
    p.max_num_running = 1;
    gc_scheduler_t sched(p);

    // Higher priority (lower value) should be scheduled first
    sched.add_request(make_req("low",  4, 8, /*priority=*/10));
    sched.add_request(make_req("high", 4, 8, /*priority=*/1));

    auto out = sched.schedule();
    CHECK((int)out.new_reqs.size() == 1);
    CHECK(out.new_reqs[0].req_id == "high");
}

// ── Scheduler: prefix cache ───────────────────────────────────────────────────

static void test_scheduler_prefix_cache_reuse() {
    auto p = base_params();
    p.num_blocks = 32;
    gc_scheduler_t sched(p);

    // First request: 8-token prompt, compute it fully
    sched.add_request(make_req("r0", 8, 4));
    sched.schedule();
    // advance computed tokens
    sched.update_from_output({{"r0", 5}});
    sched.update_from_output({{"r0", 6}});
    sched.update_from_output({{"r0", 7}});
    sched.update_from_output({{"r0", 2}});  // EOS
    CHECK(sched.num_running() == 0);

    float usage_after_r0 = sched.kv_usage();

    // Second request: same 8-token prefix → prefix cache hit, should use fewer blocks
    sched.add_request(make_req("r1", 8, 4));
    sched.schedule();
    // KV usage should not increase much (blocks reused via prefix cache)
    // (cached blocks get pinned again, so usage = same or slightly above)
    float usage_after_r1 = sched.kv_usage();
    CHECK(usage_after_r1 <= usage_after_r0 + 0.1f);
}

// ── Scheduler: kv_usage ──────────────────────────────────────────────────────

static void test_scheduler_kv_usage_increases() {
    auto p = base_params();
    gc_scheduler_t sched(p);
    CHECK(sched.kv_usage() == 0.0f);

    sched.add_request(make_req("r0", 8, 8));
    sched.schedule();
    CHECK(sched.kv_usage() > 0.0f);
}

static void test_scheduler_kv_usage_decreases_after_free() {
    auto p = base_params();
    gc_scheduler_t sched(p);

    sched.add_request(make_req("r0", 4, 2));
    sched.schedule();
    float before = sched.kv_usage();
    sched.update_from_output({{"r0", 2}});  // EOS → free
    CHECK(sched.kv_usage() < before || sched.kv_usage() == 0.0f);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_request_basic();
    test_request_block_hashes();
    test_request_block_hashes_partial();
    test_request_append_output();
    test_request_priority_order();

    test_scheduler_schedule_new_req();
    test_scheduler_schedule_multiple();
    test_scheduler_respects_max_running();
    test_scheduler_respects_token_budget();
    test_scheduler_running_continuation();
    test_scheduler_finish_eos();
    test_scheduler_finish_max_tokens();
    test_scheduler_abort_waiting();
    test_scheduler_abort_running();
    test_scheduler_preemption_on_oom();
    test_scheduler_priority_order();
    test_scheduler_prefix_cache_reuse();
    test_scheduler_kv_usage_increases();
    test_scheduler_kv_usage_decreases_after_free();

    fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
