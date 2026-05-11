#include "gc_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

gc_ggml_model_runner_t::gc_ggml_model_runner_t(config_t cfg)
    : cfg_(cfg), rng_(cfg.seed) {
    if (cfg_.num_layers < 1) cfg_.num_layers = 1;
    if (cfg_.vocab_size < 2) cfg_.vocab_size = 2;
    if (!(cfg_.temperature > 0.0f)) cfg_.temperature = 1.0f;
    if (cfg_.top_k == 0 || cfg_.top_k < -1) cfg_.top_k = 40;
    if (cfg_.top_k > cfg_.vocab_size) cfg_.top_k = cfg_.vocab_size;
    if (cfg_.top_p <= 0.0f || cfg_.top_p > 1.0f) cfg_.top_p = 0.95f;
    if (cfg_.repetition_penalty < 1.0f) cfg_.repetition_penalty = 1.0f;
    if (cfg_.repetition_window < 1) cfg_.repetition_window = 1;
}

gc_ggml_model_runner_t::gc_ggml_model_runner_t()
    : gc_ggml_model_runner_t(config_t{}) {}

int gc_ggml_model_runner_t::num_layers() const {
    return cfg_.num_layers;
}

std::vector<float> gc_ggml_model_runner_t::build_logits(const gc_batch_entry_t & e, req_state_t & st) const {
    std::vector<float> logits((size_t)cfg_.vocab_size);
    const int32_t last_tok = e.token_ids.empty() ? 0 : e.token_ids.back();
    const float phase = (float)((last_tok % 997) + 1) * 0.001f + (float)st.step_count * 0.017f;
    for (int i = 0; i < cfg_.vocab_size; ++i) {
        const float x = phase + (float)i * 0.011f;
        logits[(size_t)i] = std::sin(x) + 0.5f * std::cos(2.0f * x);
    }
    return logits;
}

void gc_ggml_model_runner_t::apply_repetition_penalty(std::vector<float> & logits, const req_state_t & st) const {
    if (cfg_.repetition_penalty <= 1.0f || st.recent_tokens.empty()) return;

    const size_t window = std::min(st.recent_tokens.size(), (size_t)cfg_.repetition_window);
    for (size_t i = st.recent_tokens.size() - window; i < st.recent_tokens.size(); ++i) {
        const int32_t tok = st.recent_tokens[i];
        if (tok < 0 || tok >= cfg_.vocab_size) continue;
        float & l = logits[(size_t)tok];
        if (l > 0.0f) l /= cfg_.repetition_penalty;
        else l *= cfg_.repetition_penalty;
    }
}

int32_t gc_ggml_model_runner_t::sample_token_with_params(
        std::vector<float> logits, float temperature, int top_k, float top_p) {
    const int vocab = (int)logits.size();

    // Temperature scaling (clamp to avoid div-by-zero)
    if (temperature <= 0.0f) temperature = 1e-6f;
    for (float & l : logits) l /= temperature;

    // Top-k: restrict to top k candidates
    std::vector<int> idx((size_t)vocab);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = (top_k > 0 && top_k < vocab) ? top_k : vocab;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; });
    idx.resize((size_t)k);

    // Stable softmax over candidates
    float max_l = -std::numeric_limits<float>::infinity();
    for (int i : idx) max_l = std::max(max_l, logits[(size_t)i]);

    std::vector<float> probs;
    probs.reserve(idx.size());
    float sum = 0.0f;
    for (int i : idx) {
        const float p = std::exp(logits[(size_t)i] - max_l);
        probs.push_back(p);
        sum += p;
    }
    if (sum <= 0.0f) return idx.front();
    for (float & p : probs) p /= sum;

    // Top-p truncation (nucleus sampling)
    if (top_p < 1.0f) {
        float cum = 0.0f;
        size_t keep = 0;
        for (; keep < probs.size(); ++keep) {
            cum += probs[keep];
            if (cum >= top_p) break;
        }
        keep = std::min(keep + 1, probs.size());
        idx.resize(keep);
        probs.resize(keep);
        float renorm = 0.0f;
        for (float p : probs) renorm += p;
        if (renorm > 0.0f) for (float & p : probs) p /= renorm;
    }

    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    return idx[dist(rng_)];
}

int32_t gc_ggml_model_runner_t::sample_token(std::vector<float> logits) {
    return sample_token_with_params(std::move(logits),
                                    cfg_.temperature, cfg_.top_k, cfg_.top_p);
}

gc_model_output_t gc_ggml_model_runner_t::execute(const gc_batch_t & batch) {
    gc_model_output_t out;
    out.req_ids.reserve(batch.entries.size());
    out.sampled_tokens.reserve(batch.entries.size());

    std::unordered_set<std::string> active_ids;
    active_ids.reserve(batch.entries.size());

    for (const auto & e : batch.entries) {
        out.req_ids.push_back(e.req_id);
        active_ids.insert(e.req_id);
        req_state_t & st = req_state_[e.req_id];

        if (e.is_prefill) {
            out.sampled_tokens.push_back(-1);
            const int win = e.sampling_params.repetition_window > 0
                            ? e.sampling_params.repetition_window : cfg_.repetition_window;
            st.recent_tokens.insert(st.recent_tokens.end(), e.token_ids.begin(), e.token_ids.end());
            if ((int)st.recent_tokens.size() > win) {
                st.recent_tokens.erase(
                    st.recent_tokens.begin(),
                    st.recent_tokens.end() - win);
            }
            continue;
        }

        // Use per-request params when set, fall back to global config.
        const float temperature = (e.sampling_params.temperature > 0.0f)
                                  ? e.sampling_params.temperature : cfg_.temperature;
        const int   top_k       = (e.sampling_params.top_k >= -1 && e.sampling_params.top_k != 0)
                                  ? e.sampling_params.top_k : cfg_.top_k;
        const float top_p       = (e.sampling_params.top_p > 0.0f && e.sampling_params.top_p <= 1.0f)
                                  ? e.sampling_params.top_p : cfg_.top_p;
        const float rep_pen     = (e.sampling_params.repetition_penalty >= 1.0f)
                                  ? e.sampling_params.repetition_penalty : cfg_.repetition_penalty;
        const int   rep_win     = (e.sampling_params.repetition_window > 0)
                                  ? e.sampling_params.repetition_window : cfg_.repetition_window;

        auto logits = build_logits(e, st);

        // Repetition penalty with per-request window
        if (rep_pen > 1.0f && !st.recent_tokens.empty()) {
            const size_t window = std::min(st.recent_tokens.size(), (size_t)rep_win);
            for (size_t i = st.recent_tokens.size() - window; i < st.recent_tokens.size(); ++i) {
                const int32_t tok_id = st.recent_tokens[i];
                if (tok_id < 0 || tok_id >= cfg_.vocab_size) continue;
                float & l = logits[(size_t)tok_id];
                if (l > 0.0f) l /= rep_pen;
                else l *= rep_pen;
            }
        }

        const int32_t tok = sample_token_with_params(std::move(logits), temperature, top_k, top_p);
        out.sampled_tokens.push_back(tok);

        st.recent_tokens.push_back(tok);
        if ((int)st.recent_tokens.size() > rep_win) {
            st.recent_tokens.erase(st.recent_tokens.begin());
        }
        st.step_count++;
    }

    // Drop state for requests not present in this step to avoid unbounded growth.
    for (auto it = req_state_.begin(); it != req_state_.end();) {
        if (active_ids.find(it->first) == active_ids.end()) it = req_state_.erase(it);
        else ++it;
    }

    return out;
}

void gc_ggml_model_runner_t::release_request(const std::string & req_id) {
    req_state_.erase(req_id);
}
