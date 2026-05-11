#include "gc_engine.h"

#include <algorithm>

gc_default_model_runner_t::gc_default_model_runner_t(config_t cfg)
    : cfg_(cfg) {
    if (cfg_.num_layers < 1) {
        cfg_.num_layers = 1;
    }
}

gc_default_model_runner_t::gc_default_model_runner_t()
    : gc_default_model_runner_t(config_t{}) {}

gc_model_output_t gc_default_model_runner_t::execute(const gc_batch_t & batch) {
    gc_model_output_t out;
    out.req_ids.reserve(batch.entries.size());
    out.sampled_tokens.reserve(batch.entries.size());

    for (const auto & e : batch.entries) {
        out.req_ids.push_back(e.req_id);
        if (e.is_prefill && !cfg_.emit_token_on_prefill) {
            out.sampled_tokens.push_back(-1);
        } else {
            out.sampled_tokens.push_back(cfg_.decode_token);
        }
    }
    return out;
}

int gc_default_model_runner_t::num_layers() const {
    return cfg_.num_layers;
}
