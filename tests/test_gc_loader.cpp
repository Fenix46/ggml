//
// Phase 1 test — load a GGUF file and verify metadata + tensor index.
//
// Usage:  test_gc_loader <path/to/model.gguf>
//

#include "loader/gc_gguf_loader.h"
#include "loader/gc_common.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

static void log_cb(ggml_log_level level, const char * text, void *) {
    const char * prefix = "";
    switch (level) {
        case GGML_LOG_LEVEL_WARN:  prefix = "[WARN]  "; break;
        case GGML_LOG_LEVEL_ERROR: prefix = "[ERROR] "; break;
        case GGML_LOG_LEVEL_DEBUG: prefix = "[DEBUG] "; break;
        default:                   prefix = "[INFO]  "; break;
    }
    fputs(prefix, stdout);
    fputs(text,   stdout);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [no-mmap]\n", argv[0]);
        return 1;
    }

    gc_log_set_callback(log_cb, nullptr);

    gc_loader_params_t params;
    params.use_mmap = (argc < 3 || std::string(argv[2]) != "no-mmap");

    fprintf(stdout, "Loading: %s  (mmap=%s)\n\n", argv[1], params.use_mmap ? "yes" : "no");

    try {
        gc_model_loader_t loader(argv[1], params);
        loader.print_info();

        // ── basic sanity checks ────────────────────────────────────────────────

        // 1. at least 1 tensor
        if (loader.n_tensors < 1) {
            fprintf(stderr, "FAIL: no tensors found\n");
            return 1;
        }

        // 2. architecture key present (most GGUF models have it)
        if (loader.arch_name.empty()) {
            fprintf(stdout, "WARN: general.architecture not set — unusual but not fatal\n");
        } else {
            fprintf(stdout, "\nPASS: architecture = '%s'\n", loader.arch_name.c_str());
        }

        // 3. load first tensor into memory
        {
            const gc_tensor_weight_t * w0 = nullptr;
            for (auto & kv : loader.weights) { w0 = &kv.second; break; }

            const size_t nb = ggml_nbytes(w0->tensor);
            std::vector<uint8_t> buf(nb);
            gc_status_t rc = loader.load_tensor_data(*w0, buf.data());
            if (rc != GC_OK) {
                fprintf(stderr, "FAIL: load_tensor_data returned %d\n", (int)rc);
                return 1;
            }
            fprintf(stdout, "PASS: loaded first tensor '%s'  (%zu bytes)\n",
                    ggml_get_name(w0->tensor), nb);
        }

        fprintf(stdout, "\nPhase 1 PASS — %d tensors, %d KV pairs, arch='%s'\n",
                loader.n_tensors, loader.n_kv, loader.arch_name.c_str());

    } catch (const std::exception & e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }

    return 0;
}
