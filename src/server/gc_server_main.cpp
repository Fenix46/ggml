#include "gc_server.h"
#include "gc_engine.h"
#include "gc_graph_runner.h"
#include "gc_gguf_loader.h"
#include "gc_hparams.h"
#include "gc_vocab.h"
#include "gc_chat.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_stop{false};

void gc__signal_handler(int) {
    g_stop.store(true);
}

int gc__parse_int_arg(const char * s, int fallback) {
    if (!s || !*s) return fallback;
    const int v = std::atoi(s);
    return v > 0 ? v : fallback;
}

void gc__print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <path.gguf> [--host 127.0.0.1] [--port 8080]\n"
        "       [--max-running N] [--max-batched-tokens N] [--blocks N] [--block-size N]\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;

    int max_running = 8;
    int max_batched_tokens = 512;
    int num_blocks = 256;
    int block_size = 16;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = gc__parse_int_arg(argv[++i], port);
        } else if (std::strcmp(argv[i], "--max-running") == 0 && i + 1 < argc) {
            max_running = gc__parse_int_arg(argv[++i], max_running);
        } else if (std::strcmp(argv[i], "--max-batched-tokens") == 0 && i + 1 < argc) {
            max_batched_tokens = gc__parse_int_arg(argv[++i], max_batched_tokens);
        } else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            num_blocks = gc__parse_int_arg(argv[++i], num_blocks);
        } else if (std::strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
            block_size = gc__parse_int_arg(argv[++i], block_size);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            gc__print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            gc__print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "error: --model is required\n");
        gc__print_usage(argv[0]);
        return 1;
    }

    try {
        gc_loader_params_t lp;
        lp.use_mmap = true;
        gc_model_loader_t loader(model_path, lp);

        gc_hparams_t hp;
        if (!gc_hparams_load(hp, loader) || hp.arch == GC_ARCH_UNKNOWN) {
            std::fprintf(stderr, "error: failed to load model hparams/architecture from %s\n", model_path.c_str());
            return 1;
        }

        gc_vocab_t vocab;
        if (vocab.load(loader) != GC_OK || vocab.n_tokens() <= 0) {
            std::fprintf(stderr, "error: failed to load tokenizer vocab from %s\n", model_path.c_str());
            return 1;
        }

        gc_chat_template_t chat_template = GC_CHAT_TEMPLATE_UNKNOWN;
        std::string raw_template;
        if (loader.get_str("tokenizer.chat_template", raw_template, /*required=*/false)) {
            chat_template = gc_chat_detect_template(raw_template);
        }
        if (chat_template == GC_CHAT_TEMPLATE_UNKNOWN) {
            switch (hp.arch) {
                case GC_ARCH_GEMMA:
                case GC_ARCH_GEMMA2:
                case GC_ARCH_GEMMA3:
                case GC_ARCH_GEMMA3N:
                case GC_ARCH_GEMMA4:
                    chat_template = GC_CHAT_TEMPLATE_GEMMA;
                    break;
                default:
                    break;
            }
        }

        gc_engine_params_t ep;
        ep.sched.num_blocks = num_blocks;
        ep.sched.block_size = block_size;
        ep.sched.max_num_running = max_running;
        ep.sched.max_num_tokens = max_batched_tokens;
        ep.sched.max_model_len = hp.n_ctx_train > 0 ? (int) hp.n_ctx_train : 4096;
        ep.sched.enable_prefix_cache = true;

        std::unique_ptr<gc_model_runner_t> runner_holder;

        gc_graph_runner_t::config_t runner_cfg;
        runner_cfg.num_kv_blocks      = num_blocks;
        runner_cfg.kv_block_size      = block_size;
        runner_cfg.seed               = 1;
        runner_cfg.temperature        = 0.8f;
        runner_cfg.top_k              = 40;
        runner_cfg.top_p              = 0.95f;
        runner_cfg.repetition_penalty = 1.1f;
        runner_cfg.repetition_window  = 64;
        auto graph_runner = std::make_unique<gc_graph_runner_t>(&loader, hp, runner_cfg);
        if (!graph_runner->ok()) {
            std::fprintf(stderr, "error: graph runner init failed: %s\n",
                         graph_runner->error().c_str());
            return 1;
        }
        runner_holder = std::move(graph_runner);
        gc_engine_t engine(ep, runner_holder.get());

        const int32_t eos_id = vocab.token_eos() == GC_TOKEN_NULL ? 2 : vocab.token_eos();
        gc_engine_server_runtime_t runtime(&engine, [&vocab](const std::string & text) {
            auto toks = vocab.tokenize(text, false, false);
            if (toks.empty()) {
                toks.push_back(1);
            }
            return toks;
        }, eos_id);

        gc_server_params_t sp;
        sp.host = host;
        sp.port = port;
        sp.runtime = &runtime;
        sp.chat_template = chat_template;
        sp.detokenize_fn = [&vocab](int32_t tok) {
            std::string s = vocab.detokenize({tok}, false);
            if (s.empty()) {
                return std::string("<tok:") + std::to_string(tok) + ">";
            }
            return s;
        };
        sp.detokenize_ids_fn = [&vocab](const std::vector<int32_t> & toks) {
            return vocab.detokenize(toks, false);
        };

        gc_server_t server(sp);
        if (!server.start_async()) {
            std::fprintf(stderr, "error: failed to start server on %s:%d\n", host.c_str(), port);
            return 1;
        }

        std::signal(SIGINT, gc__signal_handler);
        std::signal(SIGTERM, gc__signal_handler);

        std::fprintf(stderr,
            "gc_server_main listening on %s:%d | model=%s | arch=%s | vocab=%d | chat_template=%d\n",
            host.c_str(), port, model_path.c_str(), gc_arch_name(hp.arch), vocab.n_tokens(), (int) chat_template);

        while (!g_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        server.stop();
        std::fprintf(stderr, "gc_server_main stopped\n");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
