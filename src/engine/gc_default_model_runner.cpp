#include "gc_engine.h"

gc_default_model_runner_t::gc_default_model_runner_t(config_t cfg)
    : cfg_(cfg) {
    if (cfg_.num_layers < 1) {
        cfg_.num_layers = 1;
    }

    gc_ggml_model_runner_t::config_t ggml_cfg;
    ggml_cfg.num_layers = cfg_.num_layers;
    ggml_cfg.vocab_size = 32000;
    ggml_cfg.seed = 1;
    ggml_cfg.temperature = 1.0f;
    ggml_cfg.top_k = 40;
    ggml_cfg.top_p = 0.95f;
    ggml_cfg.repetition_penalty = 1.1f;
    ggml_cfg.repetition_window = 64;
    impl_.reset(new gc_ggml_model_runner_t(ggml_cfg));
}

gc_default_model_runner_t::gc_default_model_runner_t()
    : gc_default_model_runner_t(config_t{}) {}

gc_model_output_t gc_default_model_runner_t::execute(const gc_batch_t & batch) {
    return impl_->execute(batch);
}

int gc_default_model_runner_t::num_layers() const {
    return cfg_.num_layers;
}
