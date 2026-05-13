#pragma once

#include "gc_engine.h"       // gc_batch_entry_t
#include "gc_hparams.h"      // gc_hparams_t
#include "gc_kvcache.h"      // gc_block_hash_t

#include "ggml.h"

#include <functional>

struct gc_kv_pool_t;  // defined in gc_graph_runner.h

// Weight lookup: returns a ggml_tensor* view of a model weight.
using gc_graph_weight_fn_t = std::function<ggml_tensor*(
    ggml_context * ctx,
    gc_tensor_role_t role,
    const char * suffix,
    int layer     // -1 = global (non-layer) weight
)>;

struct GcGraphBuildParams {
    ggml_context          * ctx        = nullptr;
    ggml_cgraph           * gf         = nullptr;
    const gc_batch_entry_t * entry     = nullptr;
    const gc_kv_pool_t    * kv_pool    = nullptr;
    const gc_hparams_t    * hp         = nullptr;
    gc_graph_weight_fn_t    get_weight;
    ggml_tensor           * inp_tokens = nullptr;  // I32 [n_new] CPU-backed
    ggml_tensor           * inp_pos    = nullptr;  // I32 [n_new] CPU-backed
    ggml_tensor          ** kq_mask_out = nullptr;
};

class GcGraphBuilder {
public:
    virtual ~GcGraphBuilder() = default;
    virtual ggml_tensor * build(const GcGraphBuildParams & params) = 0;
    virtual const char * name() const = 0;
};
