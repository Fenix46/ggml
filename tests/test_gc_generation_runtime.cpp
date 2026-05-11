// test_gc_generation_runtime.cpp
//
// Usage:
//   test_gc_generation_runtime <model.gguf> [--rounds=N] [--max-tokens=N]
//
// Purpose:
//   Runtime lifecycle test on a real GGUF model:
//   - load metadata/vocab from real model
//   - run repeated generation rounds through scheduler+engine
//   - verify requests always finish and engine state returns idle

#include "gc_gguf_loader.h"
#include "gc_arch.h"
#include "gc_hparams.h"
#include "gc_vocab.h"
#include "gc_engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

static int gc__parse_opt_int(const std::string & arg, const char * prefix, int fallback) {
    const std::string p(prefix);
    if (arg.rfind(p, 0) != 0) return fallback;
    const std::string v = arg.substr(p.size());
    if (v.empty()) return fallback;
    return std::max(1, std::atoi(v.c_str()));
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--rounds=N] [--max-tokens=N]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    int rounds = 5;
    int max_tokens = 64;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        rounds = gc__parse_opt_int(arg, "--rounds=", rounds);
        max_tokens = gc__parse_opt_int(arg, "--max-tokens=", max_tokens);
    }

    try {
        gc_loader_params_t lp;
        lp.use_mmap = true;
        gc_model_loader_t loader(model_path, lp);

        gc_hparams_t hp;
        CHECK(gc_hparams_load(hp, loader));
        CHECK(hp.arch != GC_ARCH_UNKNOWN);

        gc_vocab_t vocab;
        CHECK(vocab.load(loader) == GC_OK);
        CHECK(vocab.n_tokens() > 0);

        const int32_t eos_id = vocab.token_eos() == GC_TOKEN_NULL ? 2 : vocab.token_eos();
        struct EosRunner final : public gc_model_runner_t {
            explicit EosRunner(int layers, int32_t eos) : layers_(std::max(1, layers)), eos_(eos) {}
            int num_layers() const override { return layers_; }
            gc_model_output_t execute(const gc_batch_t & batch) override {
                gc_model_output_t out;
                for (const auto & e : batch.entries) {
                    out.req_ids.push_back(e.req_id);
                    out.sampled_tokens.push_back(e.is_prefill ? -1 : eos_);
                }
                return out;
            }
            int layers_ = 1;
            int32_t eos_ = 2;
        } runner((int) hp.n_layer, eos_id);

        gc_engine_params_t ep;
        ep.sched.num_blocks = 256;
        ep.sched.block_size = 16;
        ep.sched.max_num_running = 8;
        ep.sched.max_num_tokens = 512;
        ep.sched.max_model_len = hp.n_ctx_train > 0 ? (int) hp.n_ctx_train : 4096;
        ep.sched.enable_prefix_cache = true;
        gc_engine_t eng(ep, &runner);

        for (int r = 0; r < rounds; ++r) {
            const std::string prompt = "Round " + std::to_string(r) + ": runtime generation lifecycle.";
            auto prompt_tokens = vocab.tokenize(prompt, false, false);
            CHECK(!prompt_tokens.empty());

            gc_sampling_params_t sp;
            sp.max_tokens = max_tokens;
            sp.eos_token_id = eos_id;
            sp.ignore_eos = false;

            const std::string req_id = "req-" + std::to_string(r);
            eng.add_request(std::make_unique<gc_request_t>(req_id, prompt_tokens, sp));

            int produced = 0;
            int guard = 512;
            while (eng.has_work() && guard-- > 0) {
                const gc_step_output_t out = eng.step();
                for (const auto & o : out.outputs) {
                    if (o.req_id == req_id && o.token >= 0) {
                        produced++;
                    }
                }
            }

            CHECK(guard > 0);
            CHECK(produced >= 1);
            CHECK(!eng.has_work());
            CHECK(eng.num_running() == 0);
            CHECK(eng.num_waiting() == 0);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
