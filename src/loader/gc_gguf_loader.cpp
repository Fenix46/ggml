#include "gc_gguf_loader.h"

#include "gguf.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <map>
#include <regex>
#include <stdexcept>
#include <vector>

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string gc__gguf_kv_to_str(const gguf_context * ctx, int i) {
    const gguf_type type = gguf_get_kv_type(ctx, i);
    switch (type) {
        case GGUF_TYPE_STRING:  return gguf_get_val_str(ctx, i);
        case GGUF_TYPE_UINT8:   return gc__format("%u",   (unsigned)gguf_get_val_u8 (ctx, i));
        case GGUF_TYPE_INT8:    return gc__format("%d",   (int)     gguf_get_val_i8 (ctx, i));
        case GGUF_TYPE_UINT16:  return gc__format("%u",   (unsigned)gguf_get_val_u16(ctx, i));
        case GGUF_TYPE_INT16:   return gc__format("%d",   (int)     gguf_get_val_i16(ctx, i));
        case GGUF_TYPE_UINT32:  return gc__format("%u",             gguf_get_val_u32(ctx, i));
        case GGUF_TYPE_INT32:   return gc__format("%d",             gguf_get_val_i32(ctx, i));
        case GGUF_TYPE_UINT64:  return gc__format("%" PRIu64,       gguf_get_val_u64(ctx, i));
        case GGUF_TYPE_INT64:   return gc__format("%" PRId64,       gguf_get_val_i64(ctx, i));
        case GGUF_TYPE_FLOAT32: return gc__format("%.6f",           gguf_get_val_f32(ctx, i));
        case GGUF_TYPE_FLOAT64: return gc__format("%.10f",          gguf_get_val_f64(ctx, i));
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(ctx, i) ? "true" : "false";
        case GGUF_TYPE_ARRAY: {
            const gguf_type at = gguf_get_arr_type(ctx, i);
            const size_t    n  = gguf_get_arr_n(ctx, i);
            return gc__format("[%s × %zu]", gguf_type_name(at), n);
        }
        default: return "(unknown)";
    }
}

// Shard filename scheme: <base>-00001-of-00004.gguf
static std::vector<std::string> gc__list_shards(const std::string & first, uint16_t n) {
    // try to detect the pattern
    std::regex re(R"((-\d{5}-of-\d{5})(\.gguf)$)", std::regex::icase);
    std::smatch m;
    if (!std::regex_search(first, m, re)) {
        // fallback: only one shard
        return {first};
    }
    std::string prefix = first.substr(0, m.position());
    std::string ext    = m[2].str();
    std::vector<std::string> paths;
    for (uint16_t i = 1; i <= n; ++i) {
        paths.push_back(gc__format("%s-%05d-of-%05d%s", prefix.c_str(), (int)i, (int)n, ext.c_str()));
    }
    return paths;
}

static std::string gc__tensor_shape_str(const ggml_tensor * t) {
    std::string s = "[";
    for (int i = ggml_n_dims(t) - 1; i >= 0; --i) {
        if (i < ggml_n_dims(t) - 1) s += ", ";
        s += gc__format("%" PRId64, t->ne[i]);
    }
    s += "]";
    return s;
}

// ── gc_tensor_weight_t ────────────────────────────────────────────────────────

gc_tensor_weight_t::gc_tensor_weight_t(const gc_file_t * file,
                                       uint16_t idx,
                                       const gguf_context * gguf_ctx,
                                       ggml_tensor * t)
    : shard_idx(idx), tensor(t)
{
    const int ti = gguf_find_tensor(gguf_ctx, ggml_get_name(t));
    if (ti < 0) {
        throw std::runtime_error(
            gc__format("tensor '%s' not found in GGUF metadata", ggml_get_name(t)));
    }
    file_offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, ti);
    const size_t nb = ggml_nbytes(t);
    if (file_offs + nb < file_offs || file_offs + nb > file->size()) {
        throw std::runtime_error(
            gc__format("tensor '%s' data out of file bounds — model corrupt", ggml_get_name(t)));
    }
}

// ── Version name ─────────────────────────────────────────────────────────────

const char * gc_gguf_version_name(gc_gguf_version_t v) {
    switch (v) {
        case GC_GGUF_VERSION_V1: return "GGUF V1";
        case GC_GGUF_VERSION_V2: return "GGUF V2";
        case GC_GGUF_VERSION_V3: return "GGUF V3";
    }
    return "unknown";
}

// ── gc_model_loader_t ─────────────────────────────────────────────────────────

gc_model_loader_t::gc_model_loader_t(const std::string & path,
                                     const gc_loader_params_t & params)
    : params_(params)
{
    // load first (or only) shard
    load_shard(path, 0);

    // check for multi-shard model
    uint32_t n_shards = 0;
    if (get_u32("split.count", n_shards, /*required=*/false) && n_shards > 1) {
        uint32_t shard_no = 0;
        get_u32("split.no", shard_no, /*required=*/true);
        if (shard_no != 0) {
            throw std::runtime_error(
                gc__format("must load first shard (index 0), got index %u", shard_no));
        }

        auto shard_paths = gc__list_shards(path, (uint16_t)n_shards);
        for (uint16_t i = 1; i < (uint16_t)n_shards; ++i) {
            load_shard(shard_paths[i], i);
        }

        // verify expected total
        uint32_t expected = 0;
        if (get_u32("split.tensors.count", expected, /*required=*/false)) {
            if ((uint32_t)weights.size() != expected) {
                throw std::runtime_error(
                    gc__format("shard count mismatch: expected %u tensors, got %zu",
                               expected, weights.size()));
            }
        }

        GC_LOG_INFO("gc_model_loader: loaded %u shards\n", n_shards);
    }

    n_kv      = gguf_get_n_kv(gguf_ctx.get());
    n_tensors = (int)weights.size();
    fver      = (gc_gguf_version_t)gguf_get_version(gguf_ctx.get());

    get_str("general.architecture", arch_name, /*required=*/false);

    determine_ftype();

    GC_LOG_INFO("gc_model_loader: %d KV pairs, %d tensors, arch='%s', format=%s\n",
                n_kv, n_tensors, arch_name.c_str(), gc_gguf_version_name(fver));
}

// ── load_shard ────────────────────────────────────────────────────────────────

void gc_model_loader_t::load_shard(const std::string & path, uint16_t shard_idx) {
    // open file
    auto file = std::make_unique<gc_file_t>(path.c_str(), "rb");

    // parse GGUF header + tensor index (no tensor data allocated)
    ggml_context * ctx_ggml = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = true;
    ip.ctx      = &ctx_ggml;

    gguf_context * ctx_gguf = gguf_init_from_file(path.c_str(), ip);
    if (!ctx_gguf) {
        throw std::runtime_error(gc__format("failed to init GGUF from '%s'", path.c_str()));
    }

    // validate magic / version
    {
        uint32_t ver = gguf_get_version(ctx_gguf);
        if (ver < GC_GGUF_VERSION_V1 || ver > GC_GGUF_VERSION_V3) {
            gguf_free(ctx_gguf);
            ggml_free(ctx_ggml);
            throw std::runtime_error(gc__format("unsupported GGUF version %u in '%s'", ver, path.c_str()));
        }
    }

    // keep first shard's gguf context as the authoritative metadata source
    if (shard_idx == 0) {
        gguf_ctx.reset(ctx_gguf);
    } else {
        // subsidiary shards: we only need their tensor index, then free
        // but we must build weights_map first (below), so defer free
    }
    ggml_ctxs.emplace_back(ctx_ggml, ggml_free);

    // mmap the shard
    if (params_.use_mmap && gc_mmap_t::SUPPORTED) {
        mmaps.emplace_back(std::make_unique<gc_mmap_t>(file.get()));
    }

    // enumerate tensors
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx_ggml);
         cur != nullptr;
         cur = ggml_get_next_tensor(ctx_ggml, cur))
    {
        std::string name = ggml_get_name(cur);
        if (weights.find(name) != weights.end()) {
            throw std::runtime_error(gc__format("duplicate tensor '%s'", name.c_str()));
        }
        n_elements += ggml_nelements(cur);
        n_bytes    += ggml_nbytes(cur);
        weights.emplace(name, gc_tensor_weight_t(file.get(), shard_idx, ctx_gguf, cur));
    }

    files.emplace_back(std::move(file));

    // free subsidiary shard gguf context (tensors are in weights_map now)
    if (shard_idx > 0) {
        gguf_free(ctx_gguf);
    }
}

// ── determine_ftype ───────────────────────────────────────────────────────────

void gc_model_loader_t::determine_ftype() {
    std::map<ggml_type, uint32_t> counts;
    ggml_type dominant = GGML_TYPE_F32;
    uint32_t  max_cnt  = 0;

    for (auto & kv : weights) {
        ggml_type t = kv.second.tensor->type;
        uint32_t  c = ++counts[t];
        if (c > max_cnt) { max_cnt = c; dominant = t; }
    }

    ftype_name = ggml_type_name(dominant);

    // try to read explicit ftype from metadata
    uint32_t ftype_val = 0;
    if (get_u32("general.file_type", ftype_val, /*required=*/false)) {
        ftype_name = gc__format("%s (explicit)", ggml_type_name(dominant));
    }
}

// ── metadata getters ──────────────────────────────────────────────────────────

int gc_model_loader_t::find_key(const std::string & key) const {
    return gguf_find_key(gguf_ctx.get(), key.c_str());
}

static void gc__require(const std::string & key, bool found, bool required) {
    if (!found && required) {
        throw std::runtime_error(gc__format("GGUF key not found: '%s'", key.c_str()));
    }
}

bool gc_model_loader_t::get_str(const std::string & key, std::string & out, bool required) const {
    int k = find_key(key);
    if (k < 0) { gc__require(key, false, required); return false; }
    if (gguf_get_kv_type(gguf_ctx.get(), k) != GGUF_TYPE_STRING) {
        throw std::runtime_error(gc__format("GGUF key '%s' is not a string", key.c_str()));
    }
    out = gguf_get_val_str(gguf_ctx.get(), k);
    return true;
}

bool gc_model_loader_t::get_u32(const std::string & key, uint32_t & out, bool required) const {
    int k = find_key(key);
    if (k < 0) { gc__require(key, false, required); return false; }
    const gguf_type t = gguf_get_kv_type(gguf_ctx.get(), k);
    if (t == GGUF_TYPE_UINT32) { out = gguf_get_val_u32(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_UINT16) { out = gguf_get_val_u16(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_UINT8)  { out = gguf_get_val_u8 (gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_INT32)  {
        const int32_t v = gguf_get_val_i32(gguf_ctx.get(), k);
        if (v < 0) throw std::runtime_error(gc__format("GGUF key '%s' negative int32 cannot convert to uint32", key.c_str()));
        out = (uint32_t)v;
        return true;
    }
    if (t == GGUF_TYPE_INT16)  {
        const int16_t v = gguf_get_val_i16(gguf_ctx.get(), k);
        if (v < 0) throw std::runtime_error(gc__format("GGUF key '%s' negative int16 cannot convert to uint32", key.c_str()));
        out = (uint32_t)v;
        return true;
    }
    if (t == GGUF_TYPE_INT8)   {
        const int8_t v = gguf_get_val_i8(gguf_ctx.get(), k);
        if (v < 0) throw std::runtime_error(gc__format("GGUF key '%s' negative int8 cannot convert to uint32", key.c_str()));
        out = (uint32_t)v;
        return true;
    }
    if (t == GGUF_TYPE_UINT64) {
        const uint64_t v = gguf_get_val_u64(gguf_ctx.get(), k);
        if (v > UINT32_MAX) throw std::runtime_error(gc__format("GGUF key '%s' uint64 overflows uint32", key.c_str()));
        out = (uint32_t)v;
        return true;
    }
    if (t == GGUF_TYPE_INT64)  {
        const int64_t v = gguf_get_val_i64(gguf_ctx.get(), k);
        if (v < 0 || v > (int64_t)UINT32_MAX) throw std::runtime_error(gc__format("GGUF key '%s' int64 out of uint32 range", key.c_str()));
        out = (uint32_t)v;
        return true;
    }
    if (!required) return false;
    throw std::runtime_error(gc__format("GGUF key '%s' expected uint type", key.c_str()));
}

bool gc_model_loader_t::get_u64(const std::string & key, uint64_t & out, bool required) const {
    int k = find_key(key);
    if (k < 0) { gc__require(key, false, required); return false; }
    if (gguf_get_kv_type(gguf_ctx.get(), k) != GGUF_TYPE_UINT64) {
        throw std::runtime_error(gc__format("GGUF key '%s' expected uint64", key.c_str()));
    }
    out = gguf_get_val_u64(gguf_ctx.get(), k);
    return true;
}

bool gc_model_loader_t::get_i32(const std::string & key, int32_t & out, bool required) const {
    int k = find_key(key);
    if (k < 0) { gc__require(key, false, required); return false; }
    const gguf_type t = gguf_get_kv_type(gguf_ctx.get(), k);
    if (t == GGUF_TYPE_INT32) { out = gguf_get_val_i32(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_INT16) { out = gguf_get_val_i16(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_INT8)  { out = gguf_get_val_i8 (gguf_ctx.get(), k); return true; }
    throw std::runtime_error(gc__format("GGUF key '%s' expected int type", key.c_str()));
}

bool gc_model_loader_t::get_f32(const std::string & key, float & out, bool required) const {
    int k = find_key(key);
    if (k < 0) { gc__require(key, false, required); return false; }
    const gguf_type t = gguf_get_kv_type(gguf_ctx.get(), k);
    if (t == GGUF_TYPE_FLOAT32) {
        out = gguf_get_val_f32(gguf_ctx.get(), k);
        return true;
    }
    if (t == GGUF_TYPE_FLOAT64) {
        out = (float)gguf_get_val_f64(gguf_ctx.get(), k);
        return true;
    }
    if (t == GGUF_TYPE_UINT32) { out = (float)gguf_get_val_u32(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_INT32)  { out = (float)gguf_get_val_i32(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_UINT64) { out = (float)gguf_get_val_u64(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_INT64)  { out = (float)gguf_get_val_i64(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_UINT16) { out = (float)gguf_get_val_u16(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_INT16)  { out = (float)gguf_get_val_i16(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_UINT8)  { out = (float)gguf_get_val_u8(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_INT8)   { out = (float)gguf_get_val_i8(gguf_ctx.get(), k); return true; }
    if (t == GGUF_TYPE_BOOL)   { out = gguf_get_val_bool(gguf_ctx.get(), k) ? 1.0f : 0.0f; return true; }
    if (!required) return false;
    throw std::runtime_error(gc__format("GGUF key '%s' expected float32", key.c_str()));
}

bool gc_model_loader_t::get_bool(const std::string & key, bool & out, bool required) const {
    int k = find_key(key);
    if (k < 0) { gc__require(key, false, required); return false; }
    if (gguf_get_kv_type(gguf_ctx.get(), k) != GGUF_TYPE_BOOL) {
        throw std::runtime_error(gc__format("GGUF key '%s' expected bool", key.c_str()));
    }
    out = gguf_get_val_bool(gguf_ctx.get(), k);
    return true;
}

bool gc_model_loader_t::get_arr_u32(const std::string & key, std::vector<uint32_t> & out, bool required) const {
    int k = find_key(key);
    if (k < 0) { gc__require(key, false, required); return false; }
    if (gguf_get_kv_type(gguf_ctx.get(), k) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(gc__format("GGUF key '%s' is not an array", key.c_str()));
    }
    const size_t n = gguf_get_arr_n(gguf_ctx.get(), k);
    const gguf_type at = gguf_get_arr_type(gguf_ctx.get(), k);
    if (at != GGUF_TYPE_UINT32 && at != GGUF_TYPE_INT32) {
        throw std::runtime_error(gc__format("GGUF key '%s' array not uint32/int32", key.c_str()));
    }
    const uint32_t * data = (const uint32_t *)gguf_get_arr_data(gguf_ctx.get(), k);
    out.assign(data, data + n);
    return true;
}

bool gc_model_loader_t::get_arr_i32(const std::string & key, std::vector<int32_t> & out, bool required) const {
    std::vector<uint32_t> tmp;
    if (!get_arr_u32(key, tmp, required)) return false;
    out.resize(tmp.size());
    std::memcpy(out.data(), tmp.data(), tmp.size() * sizeof(int32_t));
    return true;
}

bool gc_model_loader_t::get_arr_f32(const std::string & key, std::vector<float> & out, bool required) const {
    int k = find_key(key);
    if (k < 0) { gc__require(key, false, required); return false; }
    if (gguf_get_kv_type(gguf_ctx.get(), k) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(gc__format("GGUF key '%s' is not an array", key.c_str()));
    }
    if (gguf_get_arr_type(gguf_ctx.get(), k) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error(gc__format("GGUF key '%s' array is not float32", key.c_str()));
    }
    const size_t  n    = gguf_get_arr_n(gguf_ctx.get(), k);
    const float * data = (const float *)gguf_get_arr_data(gguf_ctx.get(), k);
    out.assign(data, data + n);
    return true;
}

bool gc_model_loader_t::get_arr_str(const std::string & key, std::vector<std::string> & out, bool required) const {
    int k = find_key(key);
    if (k < 0) { gc__require(key, false, required); return false; }
    if (gguf_get_kv_type(gguf_ctx.get(), k) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(gc__format("GGUF key '%s' is not an array", key.c_str()));
    }
    if (gguf_get_arr_type(gguf_ctx.get(), k) != GGUF_TYPE_STRING) {
        throw std::runtime_error(gc__format("GGUF key '%s' array is not string", key.c_str()));
    }
    const size_t n = gguf_get_arr_n(gguf_ctx.get(), k);
    out.clear();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(gguf_ctx.get(), k, (int)i));
    }
    return true;
}

// ── tensor access ─────────────────────────────────────────────────────────────

const gc_tensor_weight_t * gc_model_loader_t::find_weight(const char * name) const {
    auto it = weights.find(name);
    return it != weights.end() ? &it->second : nullptr;
}

gc_status_t gc_model_loader_t::load_tensor_data(const gc_tensor_weight_t & w, void * dst) const {
    const size_t nb = ggml_nbytes(w.tensor);
    const uint16_t si = w.shard_idx;

    if (si < mmaps.size() && mmaps[si]) {
        // fast path: just memcpy from the mmap region
        const uint8_t * src = (const uint8_t *)mmaps[si]->addr() + w.file_offs;
        std::memcpy(dst, src, nb);
        return GC_OK;
    }

    // fallback: seek + read
    if (si >= files.size() || !files[si]) return GC_ERR_INVALID;
    try {
        files[si]->seek(w.file_offs, SEEK_SET);
        files[si]->read_raw(dst, nb);
    } catch (const std::exception & e) {
        GC_LOG_ERROR("load_tensor_data: %s\n", e.what());
        return GC_ERR_IO;
    }
    return GC_OK;
}

// ── print_info ────────────────────────────────────────────────────────────────

void gc_model_loader_t::print_info() const {
    GC_LOG_INFO("=== GGUF.CORE model info ===\n");
    GC_LOG_INFO("  format  : %s\n", gc_gguf_version_name(fver));
    GC_LOG_INFO("  arch    : %s\n", arch_name.empty() ? "(unknown)" : arch_name.c_str());
    GC_LOG_INFO("  ftype   : %s\n", ftype_name.c_str());
    GC_LOG_INFO("  kv pairs: %d\n", n_kv);
    GC_LOG_INFO("  tensors : %d  (%.2f GiB)\n", n_tensors,
                (double)n_bytes / (1024.0 * 1024.0 * 1024.0));
    GC_LOG_INFO("  shards  : %zu\n", files.size());

    GC_LOG_INFO("--- KV metadata ---\n");
    for (int i = 0; i < n_kv; ++i) {
        const char * name      = gguf_get_key(gguf_ctx.get(), i);
        const gguf_type type   = gguf_get_kv_type(gguf_ctx.get(), i);
        std::string type_str   = gguf_type_name(type);
        if (type == GGUF_TYPE_ARRAY) {
            type_str += gc__format("[%s,%zu]",
                gguf_type_name(gguf_get_arr_type(gguf_ctx.get(), i)),
                gguf_get_arr_n(gguf_ctx.get(), i));
        }
        std::string val = gc__gguf_kv_to_str(gguf_ctx.get(), i);
        if (val.size() > 60) val = val.substr(0, 57) + "...";
        GC_LOG_INFO("  [%3d] %-40s  %-20s = %s\n", i, name, type_str.c_str(), val.c_str());
    }

    GC_LOG_INFO("--- tensors ---\n");
    for (auto & kv : weights) {
        const gc_tensor_weight_t & w = kv.second;
        GC_LOG_INFO("  shard=%d  offs=%-12zu  %-8s  %-12s  %s\n",
            w.shard_idx, w.file_offs,
            ggml_type_name(w.tensor->type),
            gc__tensor_shape_str(w.tensor).c_str(),
            kv.first.c_str());
    }
}
