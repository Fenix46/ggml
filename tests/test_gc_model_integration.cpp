// test_gc_model_integration.cpp
//
// Usage:
//   test_gc_model_integration <model.gguf> [--expect-arch=<name>]
//
// Purpose:
//   Real-model integration smoke test across phases 1,2,3,4,7,8.

#include "gc_gguf_loader.h"
#include "gc_arch.h"
#include "gc_hparams.h"
#include "gc_vocab.h"
#include "gc_chat.h"
#include "gc_engine.h"

#include <cstdio>
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

class EosRunner : public gc_model_runner_t {
public:
    explicit EosRunner(int32_t eos_id) : eos_id_(eos_id) {}

    int num_layers() const override { return 1; }

    gc_model_output_t execute(const gc_batch_t & batch) override {
        gc_model_output_t out;
        for (const auto & e : batch.entries) {
            out.req_ids.push_back(e.req_id);
            out.sampled_tokens.push_back(e.is_prefill ? -1 : eos_id_);
        }
        return out;
    }

private:
    int32_t eos_id_ = 2;
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--expect-arch=<name>]\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string expect_arch;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const std::string k = "--expect-arch=";
        if (arg.rfind(k, 0) == 0) {
            expect_arch = arg.substr(k.size());
        }
    }

    try {
        gc_loader_params_t lp;
        lp.use_mmap = true;
        gc_model_loader_t loader(model_path, lp);

        CHECK(loader.n_tensors > 0);
        CHECK(loader.n_kv > 0);

        gc_hparams_t hp;
        CHECK(gc_hparams_load(hp, loader));
        CHECK(hp.arch != GC_ARCH_UNKNOWN);
        CHECK(hp.n_layer > 0);
        CHECK(hp.n_embd > 0);
        if (!expect_arch.empty()) {
            CHECK(gc_arch_name(hp.arch) == expect_arch);
        }

        gc_vocab_t vocab;
        CHECK(vocab.load(loader) == GC_OK);
        CHECK(vocab.n_tokens() > 0);

        // Phase 3: real tokenizer path (ASCII + CJK)
        {
            auto t1 = vocab.tokenize("Hello, world!", false, false);
            CHECK(!t1.empty());
            auto d1 = vocab.detokenize(t1, false);
            CHECK(!d1.empty());

            auto t2 = vocab.tokenize("こんにちは世界", false, false);
            CHECK(!t2.empty());
            auto d2 = vocab.detokenize(t2, false);
            CHECK(!d2.empty());
        }

        // Phase 4: detect+apply template when metadata contains one
        {
            std::string tmpl;
            if (loader.get_str("tokenizer.chat_template", tmpl, /*required=*/false)) {
                gc_chat_template_t t = gc_chat_detect_template(tmpl);
                if (t == GC_CHAT_TEMPLATE_UNKNOWN) {
                    fprintf(stderr, "INFO  unsupported chat template; skipping render check\n");
                } else {
                    std::vector<gc_chat_message_t> msgs = {
                        {"user", "Say hi briefly."}
                    };
                    std::vector<const gc_chat_message_t *> ptrs;
                    for (auto & m : msgs) ptrs.push_back(&m);

                    std::string prompt;
                    const int32_t n = gc_chat_apply_template(t, ptrs, prompt, true);
                    CHECK(n >= 0);
                    CHECK(!prompt.empty());
                }
            } else {
                fprintf(stderr, "INFO  tokenizer.chat_template not present; skipping chat-template check\n");
            }
        }

        // Phase 7/8: schedule+engine step with real prompt tokens.
        {
            auto prompt_tokens = vocab.tokenize("Hello from integration test.", false, false);
            CHECK(!prompt_tokens.empty());

            gc_engine_params_t ep;
            ep.sched.num_blocks = 64;
            ep.sched.block_size = 16;
            ep.sched.max_num_running = 4;
            ep.sched.max_num_tokens = 128;
            ep.sched.max_model_len = 4096;
            ep.sched.enable_prefix_cache = true;

            const int32_t eos_id = vocab.token_eos() == GC_TOKEN_NULL ? 2 : vocab.token_eos();
            EosRunner runner(eos_id);
            gc_engine_t eng(ep, &runner);

            gc_sampling_params_t sp;
            sp.max_tokens = 8;
            sp.eos_token_id = eos_id;
            sp.ignore_eos = false;

            eng.add_request(std::make_unique<gc_request_t>("req0", prompt_tokens, sp));
            CHECK(eng.has_work());

            // Prefill
            auto s1 = eng.step();
            CHECK(s1.had_work);

            // Decode (runner emits EOS)
            auto s2 = eng.step();
            CHECK(s2.had_work);
            CHECK(!eng.has_work());
        }

    } catch (const std::exception & e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
