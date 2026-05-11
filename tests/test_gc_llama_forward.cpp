// test_gc_llama_forward.cpp
//
// Manual validation lane for the real GGML Llama forward path.
// Usage:
//   test_gc_llama_forward <model.gguf> [--prompt TEXT] [--chat]
//       [--max-tokens N] [--expect-token ID] [--check-chunking]
//       [--save-logits PATH] [--compare-logits PATH]
//
// This is intentionally CPU-only and single-request. It exists to validate
// Phase 5 before trusting paged KV, continuous batching, or the HTTP server.

#include "gc_gguf_loader.h"
#include "gc_hparams.h"
#include "gc_arch.h"
#include "gc_vocab.h"
#include "gc_chat.h"
#include "gc_engine.h"
#include "gc_graph_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

struct run_result_t {
    int32_t token = -1;
    std::string piece;
    std::vector<int32_t> tokens;
    std::string text;
    std::vector<float> logits;
};

static bool arg_starts(const std::string & arg, const char * prefix) {
    return arg.rfind(prefix, 0) == 0;
}

static std::string build_chat_prompt(gc_model_loader_t & loader, const std::string & user_text) {
    std::string raw_tmpl;
    gc_chat_template_t tmpl = GC_CHAT_TEMPLATE_LLAMA_3;
    if (loader.get_str("tokenizer.chat_template", raw_tmpl, false) && !raw_tmpl.empty()) {
        tmpl = gc_chat_detect_template(raw_tmpl);
    }

    if (tmpl == GC_CHAT_TEMPLATE_UNKNOWN) {
        return std::string("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n") +
               user_text +
               "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    gc_chat_message_t msg{ "user", user_text.c_str() };
    std::vector<const gc_chat_message_t *> chat{ &msg };
    std::string out;
    const int32_t n = gc_chat_apply_template(tmpl, chat, out, true);
    if (n < 0 || out.empty()) {
        return user_text;
    }
    return out;
}

static bool write_logits_bin(const std::string & path, const std::vector<float> & logits) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char *>(logits.data()),
              (std::streamsize)(logits.size() * sizeof(float)));
    return (bool)out;
}

static bool read_logits_bin(const std::string & path, std::vector<float> & logits) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const std::streamsize n = in.tellg();
    if (n < 0 || (n % (std::streamsize)sizeof(float)) != 0) return false;
    in.seekg(0, std::ios::beg);
    logits.resize((size_t)n / sizeof(float));
    in.read(reinterpret_cast<char *>(logits.data()), n);
    return (bool)in;
}

static std::vector<int> topk_indices(const std::vector<float> & logits, int k) {
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, (int)idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; });
    idx.resize((size_t)k);
    return idx;
}

static void compare_logits(const std::vector<float> & got, const std::vector<float> & ref) {
    CHECK(!got.empty());
    CHECK(got.size() == ref.size());
    if (got.empty() || got.size() != ref.size()) {
        std::fprintf(stderr, "INFO logits_shape got=%zu ref=%zu\n", got.size(), ref.size());
        return;
    }

    double sum_abs = 0.0;
    float max_abs = 0.0f;
    int max_idx = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        sum_abs += d;
        if (d > max_abs) {
            max_abs = d;
            max_idx = (int)i;
        }
    }

    const std::vector<int> got_top = topk_indices(got, 10);
    const std::vector<int> ref_top = topk_indices(ref, 10);
    int overlap = 0;
    for (int a : got_top) {
        for (int b : ref_top) {
            if (a == b) ++overlap;
        }
    }

    std::fprintf(stderr, "INFO logits_cmp size=%zu mean_abs=%.6g max_abs=%.6g max_idx=%d top10_overlap=%d/10\n",
                 got.size(), sum_abs / (double)got.size(), max_abs, max_idx, overlap);
    std::fprintf(stderr, "INFO gc_top10:");
    for (int i : got_top) std::fprintf(stderr, " %d(%.4g)", i, got[(size_t)i]);
    std::fprintf(stderr, "\nINFO ref_top10:");
    for (int i : ref_top) std::fprintf(stderr, " %d(%.4g)", i, ref[(size_t)i]);
    std::fprintf(stderr, "\n");

    CHECK(overlap > 0);
}

static run_result_t run_once(
        gc_model_loader_t & loader,
        const gc_hparams_t & hp,
        const gc_vocab_t & vocab,
        const std::vector<int32_t> & prompt_tokens,
        int max_tokens,
        int max_num_tokens,
        int max_prefill_chunk_tokens) {
    gc_graph_runner_t::config_t rcfg;
    rcfg.num_kv_blocks = 128;
    rcfg.kv_block_size = 16;
    rcfg.seed = 1;
    rcfg.temperature = 0.0f;
    rcfg.top_k = 1;
    rcfg.top_p = 1.0f;
    rcfg.repetition_penalty = 1.0f;
    rcfg.n_threads = 4;
    rcfg.force_cpu = true;

    gc_graph_runner_t runner(&loader, hp, rcfg);
    CHECK(runner.ok());
    if (!runner.ok()) {
        std::fprintf(stderr, "runner error: %s\n", runner.error().c_str());
        return {};
    }

    gc_engine_params_t ep;
    ep.sched.num_blocks = 128;
    ep.sched.block_size = 16;
    ep.sched.max_num_running = 1;
    ep.sched.max_num_tokens = max_num_tokens;
    ep.sched.max_prefill_chunk_tokens = max_prefill_chunk_tokens;
    ep.sched.max_model_len = hp.n_ctx_train > 0 ? (int)hp.n_ctx_train : 4096;
    ep.sched.enable_prefix_cache = false;

    gc_engine_t eng(ep, &runner);

    gc_sampling_params_t sp;
    sp.max_tokens = max_tokens;
    sp.eos_token_id = vocab.token_eos() == GC_TOKEN_NULL ? -1 : vocab.token_eos();
    sp.ignore_eos = false;
    sp.temperature = 0.0f;
    sp.top_k = 1;
    sp.top_p = 1.0f;
    sp.repetition_penalty = 1.0f;

    eng.add_request(std::make_unique<gc_request_t>("forward", prompt_tokens, sp));

    run_result_t result;
    int guard = 4096;
    while (eng.has_work() && guard-- > 0) {
        const gc_step_output_t out = eng.step();
        for (const gc_req_output_t & o : out.outputs) {
            if (o.req_id == "forward" && o.token >= 0) {
                const std::string piece = vocab.detokenize({o.token}, false);
                if (result.tokens.empty()) {
                    result.token = o.token;
                    result.piece = piece;
                }
                result.tokens.push_back(o.token);
                result.text += piece;
            }
        }
    }
    CHECK(guard > 0);
    CHECK(result.token >= 0);
    CHECK(!result.tokens.empty());
    result.logits = runner.debug_last_logits();
    CHECK(!result.logits.empty());
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--prompt TEXT] [--chat] [--max-tokens N] [--expect-token ID] [--check-chunking] [--save-logits PATH] [--compare-logits PATH]\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = "Hello";
    bool chat = false;
    bool check_chunking = false;
    int max_tokens = 1;
    int32_t expect_token = -1;
    std::string save_logits_path;
    std::string compare_logits_path;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--chat") {
            chat = true;
        } else if (arg == "--check-chunking") {
            check_chunking = true;
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg_starts(arg, "--prompt=")) {
            prompt = arg.substr(std::string("--prompt=").size());
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::atoi(argv[++i]);
        } else if (arg_starts(arg, "--max-tokens=")) {
            max_tokens = std::atoi(arg.substr(std::string("--max-tokens=").size()).c_str());
        } else if (arg == "--expect-token" && i + 1 < argc) {
            expect_token = (int32_t)std::atoi(argv[++i]);
        } else if (arg_starts(arg, "--expect-token=")) {
            expect_token = (int32_t)std::atoi(arg.substr(std::string("--expect-token=").size()).c_str());
        } else if (arg == "--save-logits" && i + 1 < argc) {
            save_logits_path = argv[++i];
        } else if (arg_starts(arg, "--save-logits=")) {
            save_logits_path = arg.substr(std::string("--save-logits=").size());
        } else if (arg == "--compare-logits" && i + 1 < argc) {
            compare_logits_path = argv[++i];
        } else if (arg_starts(arg, "--compare-logits=")) {
            compare_logits_path = arg.substr(std::string("--compare-logits=").size());
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (max_tokens < 1) {
        std::fprintf(stderr, "--max-tokens must be >= 1\n");
        return 1;
    }

    try {
        gc_loader_params_t lp;
        lp.use_mmap = true;
        gc_model_loader_t loader(model_path, lp);

        gc_hparams_t hp;
        CHECK(gc_hparams_load(hp, loader));
        CHECK(hp.arch == GC_ARCH_LLAMA);

        gc_vocab_t vocab;
        CHECK(vocab.load(loader) == GC_OK);
        CHECK(vocab.n_tokens() > 0);
        CHECK(vocab.token_bos() == 128000 || vocab.token_bos() != GC_TOKEN_NULL);
        CHECK(vocab.text_to_token("<|eot_id|>") == 128009 || vocab.text_to_token("<|eot_id|>") == GC_TOKEN_NULL);
        if (vocab.text_to_token("<|eot_id|>") != GC_TOKEN_NULL) {
            CHECK(vocab.token_eot() == vocab.text_to_token("<|eot_id|>"));
        }

        const std::string rendered = chat ? build_chat_prompt(loader, prompt) : prompt;
        const std::vector<int32_t> prompt_tokens = vocab.tokenize(rendered, chat ? false : true, true);
        CHECK(!prompt_tokens.empty());

        std::fprintf(stderr, "INFO prompt=%s\n", rendered.c_str());
        std::fprintf(stderr, "INFO prompt_tokens=%zu", prompt_tokens.size());
        for (int32_t t : prompt_tokens) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");

        const run_result_t full = run_once(loader, hp, vocab, prompt_tokens,
                                          max_tokens,
                                          /*max_num_tokens=*/std::max(64, (int)prompt_tokens.size()),
                                          /*max_prefill_chunk_tokens=*/std::max(64, (int)prompt_tokens.size()));
        std::fprintf(stderr, "INFO first_token=%d piece='%s'\n", full.token, full.piece.c_str());
        std::fprintf(stderr, "INFO generated_tokens=%zu", full.tokens.size());
        for (int32_t t : full.tokens) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\nINFO generated_text=%s\n", full.text.c_str());

        if (expect_token >= 0) {
            CHECK(full.token == expect_token);
        }

        if (!save_logits_path.empty()) {
            CHECK(write_logits_bin(save_logits_path, full.logits));
            std::fprintf(stderr, "INFO saved_logits=%s n=%zu\n", save_logits_path.c_str(), full.logits.size());
        }

        if (!compare_logits_path.empty()) {
            CHECK(max_tokens == 1);
            std::vector<float> ref;
            CHECK(read_logits_bin(compare_logits_path, ref));
            compare_logits(full.logits, ref);
        }

        if (check_chunking) {
            const run_result_t chunked = run_once(loader, hp, vocab, prompt_tokens,
                                                 max_tokens,
                                                 /*max_num_tokens=*/1,
                                                 /*max_prefill_chunk_tokens=*/1);
            std::fprintf(stderr, "INFO chunked_first_token=%d piece='%s'\n", chunked.token, chunked.piece.c_str());
            CHECK(chunked.token == full.token);
            CHECK(chunked.tokens == full.tokens);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }

    std::fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
