#pragma once

#include "gc_graph.h"

// Llama-family graph builder (includes Llama 3, Llama 4, and variants).
// Standard decoder-only transformer: pre-attention norm → Q/K/V → RoPE →
// KV write → MHA → output proj → residual → pre-FFN norm → SiLU-gated FFN →
// residual → output norm → lm_head.
class GcGraphLlama final : public GcGraphBuilder {
public:
    ggml_tensor * build(const GcGraphBuildParams & params) override;
    const char * name() const override { return "Llama"; }
};
