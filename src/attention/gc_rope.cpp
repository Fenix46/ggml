#include "gc_rope.h"

struct ggml_tensor * gc_rope_apply(
        struct ggml_context   * ctx,
        struct ggml_tensor    * a,
        struct ggml_tensor    * pos,
        struct ggml_tensor    * factors,
        const gc_rope_params_t & p) {
    return ggml_rope_ext(
            ctx, a, pos, factors,
            p.n_dims, p.mode, p.n_ctx_orig,
            p.freq_base, p.freq_scale,
            p.ext_factor, p.attn_factor,
            p.beta_fast,  p.beta_slow);
}

struct ggml_tensor * gc_rope_apply_multi(
        struct ggml_context   * ctx,
        struct ggml_tensor    * a,
        struct ggml_tensor    * pos,
        struct ggml_tensor    * factors,
        const gc_rope_params_t & p,
        int sections[GGML_MROPE_SECTIONS]) {
    return ggml_rope_multi(
            ctx, a, pos, factors,
            p.n_dims, sections, p.mode, p.n_ctx_orig,
            p.freq_base, p.freq_scale,
            p.ext_factor, p.attn_factor,
            p.beta_fast,  p.beta_slow);
}
