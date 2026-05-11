#pragma once

#include "gc_common.h"
#include "gc_mmap.h"

#include "gguf.h"
#include "ggml.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// ── GGUF file version ─────────────────────────────────────────────────────────

typedef enum gc_gguf_version_t {
    GC_GGUF_VERSION_V1 = 1,
    GC_GGUF_VERSION_V2 = 2,
    GC_GGUF_VERSION_V3 = 3,
} gc_gguf_version_t;

const char * gc_gguf_version_name(gc_gguf_version_t v);

// ── Tensor weight record ──────────────────────────────────────────────────────

struct gc_tensor_weight_t {
    uint16_t       shard_idx; // which shard file owns this tensor
    size_t         file_offs; // byte offset of tensor data in that shard
    ggml_tensor *  tensor;    // metadata tensor (no data allocated)

    gc_tensor_weight_t(const gc_file_t * file,
                       uint16_t shard_idx,
                       const gguf_context * gguf_ctx,
                       ggml_tensor * tensor);
};

// ── Loader configuration ──────────────────────────────────────────────────────

struct gc_loader_params_t {
    bool use_mmap     = true;
    bool check_tensors = false;
};

// ── Main loader ───────────────────────────────────────────────────────────────

struct gc_model_loader_t {
    // custom comparator: sort weights by layer index for nicer printing
    struct weight_name_cmp {
        bool operator()(const std::string & a, const std::string & b) const {
            int la = -1, lb = -1;
            sscanf(a.c_str(), "blk.%d.", &la);
            sscanf(b.c_str(), "blk.%d.", &lb);
            return la != lb ? la < lb : a < b;
        }
    };

    // ── state ─────────────────────────────────────────────────────────────────
    int      n_kv       = 0;
    int      n_tensors  = 0;
    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    std::string        arch_name;
    gc_gguf_version_t  fver = GC_GGUF_VERSION_V3;
    std::string        ftype_name;   // human-readable quantization label

    gc_files_t  files;    // one per shard
    gc_mmaps_t  mmaps;    // one per shard (when use_mmap = true)

    std::map<std::string, gc_tensor_weight_t, weight_name_cmp> weights;

    // gguf context for the first (or only) shard
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf_ctx {nullptr, gguf_free};
    // ggml contexts that hold tensor metadata (one per shard)
    std::vector<std::unique_ptr<ggml_context, decltype(&ggml_free)>> ggml_ctxs;

    // ── construction ─────────────────────────────────────────────────────────
    gc_model_loader_t(const std::string & path, const gc_loader_params_t & params = {});
    ~gc_model_loader_t() = default;

    gc_model_loader_t(const gc_model_loader_t &) = delete;
    gc_model_loader_t & operator=(const gc_model_loader_t &) = delete;

    // ── metadata access ───────────────────────────────────────────────────────

    // Get scalar KV values.  Returns false if key absent and required=false,
    // throws std::runtime_error if required=true and missing or wrong type.
    bool get_str    (const std::string & key, std::string & out, bool required = true) const;
    bool get_u32    (const std::string & key, uint32_t   & out, bool required = true) const;
    bool get_u64    (const std::string & key, uint64_t   & out, bool required = true) const;
    bool get_i32    (const std::string & key, int32_t    & out, bool required = true) const;
    bool get_f32    (const std::string & key, float      & out, bool required = true) const;
    bool get_bool   (const std::string & key, bool       & out, bool required = true) const;

    // Get arrays
    bool get_arr_u32(const std::string & key, std::vector<uint32_t>   & out, bool required = true) const;
    bool get_arr_i32(const std::string & key, std::vector<int32_t>    & out, bool required = true) const;
    bool get_arr_f32(const std::string & key, std::vector<float>      & out, bool required = true) const;
    bool get_arr_str(const std::string & key, std::vector<std::string>& out, bool required = true) const;

    // Raw gguf lookup (use only when typed helpers don't cover your case)
    int find_key(const std::string & key) const;

    // ── tensor access ─────────────────────────────────────────────────────────

    const gc_tensor_weight_t * find_weight(const char * name) const;

    // Loads raw tensor bytes into *dst.  dst must be large enough (ggml_nbytes(tensor)).
    // Uses mmap if available, else fread.
    gc_status_t load_tensor_data(const gc_tensor_weight_t & w, void * dst) const;

    // ── info ──────────────────────────────────────────────────────────────────
    void print_info() const;

private:
    gc_loader_params_t params_;

    void load_shard(const std::string & path, uint16_t shard_idx);
    void determine_ftype();
    static std::vector<std::string> list_shards(const std::string & first_path,
                                                uint16_t n_shards);
};
