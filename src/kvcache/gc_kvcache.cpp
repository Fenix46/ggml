#include "gc_kvcache.h"

#include <cassert>
#include <cstring>
#include <stdexcept>

// ── Block hash ────────────────────────────────────────────────────────────────
// FNV-1a 64-bit: deterministic, no external deps, fast.

static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

static inline uint64_t fnv1a_bytes(const void * data, size_t len, uint64_t h = FNV_OFFSET) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= FNV_PRIME;
    }
    return h;
}

gc_block_hash_t gc_hash_block_tokens(
        gc_block_hash_t parent_hash,
        const int32_t * token_ids,
        size_t          n_tokens,
        uint64_t        extra_key) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a_bytes(&parent_hash, sizeof(parent_hash), h);
    h = fnv1a_bytes(token_ids,    sizeof(int32_t) * n_tokens, h);
    if (extra_key != 0) {
        h = fnv1a_bytes(&extra_key, sizeof(extra_key), h);
    }
    return h;
}

// ── Free queue ────────────────────────────────────────────────────────────────

gc_free_queue_t::gc_free_queue_t() {
    head_sentinel.block_id = -1;
    tail_sentinel.block_id = -1;
    head_sentinel.next_free = &tail_sentinel;
    tail_sentinel.prev_free = &head_sentinel;
    head_sentinel.prev_free = nullptr;
    tail_sentinel.next_free = nullptr;
}

gc_kv_block_t * gc_free_queue_t::popleft() {
    assert(num_free > 0);
    gc_kv_block_t * blk = head_sentinel.next_free;
    assert(blk != &tail_sentinel);

    head_sentinel.next_free = blk->next_free;
    blk->next_free->prev_free = &head_sentinel;
    blk->prev_free = blk->next_free = nullptr;
    --num_free;
    return blk;
}

std::vector<gc_kv_block_t *> gc_free_queue_t::popleft_n(int n) {
    assert(n >= 0 && n <= num_free);
    std::vector<gc_kv_block_t *> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        out.push_back(popleft());
    }
    return out;
}

void gc_free_queue_t::remove(gc_kv_block_t * blk) {
    assert(blk->prev_free && blk->next_free);
    blk->prev_free->next_free = blk->next_free;
    blk->next_free->prev_free = blk->prev_free;
    blk->prev_free = blk->next_free = nullptr;
    --num_free;
}

void gc_free_queue_t::append(gc_kv_block_t * blk) {
    gc_kv_block_t * last = tail_sentinel.prev_free;
    last->next_free        = blk;
    blk->prev_free         = last;
    blk->next_free         = &tail_sentinel;
    tail_sentinel.prev_free = blk;
    ++num_free;
}

void gc_free_queue_t::append_n(const std::vector<gc_kv_block_t *> & blks) {
    for (gc_kv_block_t * b : blks) {
        append(b);
    }
}

// ── Block pool ────────────────────────────────────────────────────────────────

gc_block_pool_t::gc_block_pool_t(int num_blocks) {
    assert(num_blocks >= 1);
    blocks_.resize(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        blocks_[i].block_id = i;
    }

    // Block 0 is the null block — never freed, never in free list.
    blocks_[0].is_null = true;
    blocks_[0].ref_cnt = 1; // permanently pinned

    // All other blocks start in the free queue.
    for (int i = 1; i < num_blocks; ++i) {
        free_queue_.append(&blocks_[i]);
    }
}

std::vector<gc_kv_block_t *> gc_block_pool_t::get_new_blocks(int n) {
    if (n <= 0) return {};
    if (free_queue_.size() < n) return {};

    auto blks = free_queue_.popleft_n(n);
    for (gc_kv_block_t * b : blks) {
        // Evict from prefix cache if this block had a hash.
        if (b->block_hash.has_value()) {
            hash_to_block_.erase(b->block_hash.value());
            b->block_hash.reset();
        }
        assert(b->ref_cnt == 0);
        b->ref_cnt = 1;
    }
    return blks;
}

void gc_block_pool_t::touch(const std::vector<gc_kv_block_t *> & blks) {
    for (gc_kv_block_t * b : blks) {
        if (b->is_null) continue;
        // ref_cnt==0 means block is in the free queue — remove it.
        if (b->ref_cnt == 0 && b->prev_free != nullptr) {
            free_queue_.remove(b);
        }
        ++b->ref_cnt;
    }
}

void gc_block_pool_t::free_blocks(const std::vector<gc_kv_block_t *> & blks) {
    for (gc_kv_block_t * b : blks) {
        if (b->is_null) continue;
        assert(b->ref_cnt > 0);
        --b->ref_cnt;
        if (b->ref_cnt == 0) {
            free_queue_.append(b);
        }
    }
}

void gc_block_pool_t::cache_block(gc_block_hash_t hash, gc_kv_block_t * blk) {
    if (blk->is_null) return;

    // Already cached under same hash, nothing to do.
    if (blk->block_hash.has_value() && blk->block_hash.value() == hash) return;

    // Remove stale entry for this block if it had a different hash.
    if (blk->block_hash.has_value()) {
        hash_to_block_.erase(blk->block_hash.value());
    }

    // If another block already holds this hash, keep the existing one.
    // Block tables are append-only: we never steal a hash slot from an
    // active block (matches vllm NOTE #1 in BlockHashToBlockMap).
    auto it = hash_to_block_.find(hash);
    if (it != hash_to_block_.end()) return;

    blk->block_hash = hash;
    hash_to_block_[hash] = blk;
}

gc_kv_block_t * gc_block_pool_t::get_cached_block(gc_block_hash_t hash) const {
    auto it = hash_to_block_.find(hash);
    if (it == hash_to_block_.end()) return nullptr;
    return it->second;
}

void gc_block_pool_t::evict_blocks(const std::vector<int> & block_ids) {
    for (int id : block_ids) {
        if (id < 0 || id >= (int)blocks_.size()) continue;
        gc_kv_block_t * b = &blocks_[id];
        if (b->block_hash.has_value()) {
            hash_to_block_.erase(b->block_hash.value());
            b->block_hash.reset();
        }
    }
}

bool gc_block_pool_t::reset_prefix_cache() {
    // Can only reset when no block is actively held (ref_cnt > 1 means
    // held by a request beyond the pinned null block).
    for (auto & b : blocks_) {
        if (!b.is_null && b.ref_cnt > 0) return false;
    }
    hash_to_block_.clear();
    for (auto & b : blocks_) {
        if (!b.is_null) b.block_hash.reset();
    }
    return true;
}

int gc_block_pool_t::num_free_blocks() const {
    return free_queue_.size();
}

float gc_block_pool_t::usage() const {
    int total = (int)blocks_.size() - 1; // exclude null block
    if (total <= 0) return 1.0f;
    int used = total - free_queue_.size();
    return (float)used / (float)total;
}

// ── gc_req_blocks_t ───────────────────────────────────────────────────────────

std::vector<int32_t> gc_req_blocks_t::block_ids() const {
    std::vector<int32_t> ids;
    ids.reserve(blocks.size());
    for (const gc_kv_block_t * b : blocks) ids.push_back(b->block_id);
    return ids;
}

// ── KV manager ───────────────────────────────────────────────────────────────
//
// LIFECYCLE INVARIANTS (mirrors vllm SingleTypeKVCacheManager):
//
//   1. req_blocks_[id].blocks is the COMPLETE logical block table for the
//      request, indexed 0..N-1 where N = ceil(num_tokens / block_size).
//      Logical block i covers tokens [i*block_size, (i+1)*block_size).
//
//   2. Block table is APPEND-ONLY after first allocation.
//      allocate_slots() for a running request only appends new blocks;
//      it never clears or rebuilds the existing table.
//
//   3. Prefix-cached blocks returned by find_cache_hit() are prepended
//      once on first admission via allocate_new_request(). After that,
//      the request is "running" and only gets extensions.
//
//   4. ref_cnt > 0  <=>  block is held by at least one request or
//      pinned (null block always ref_cnt=1).
//      ref_cnt == 0 <=>  block is in the free queue (eviction candidate).
//
//   5. free() decrements ref_counts in reverse order (LRU-friendly)
//      then removes the entry from req_blocks_.
//
//   6. No dangling pointer: req_blocks_[id].blocks contains only pointers
//      into pool_.blocks_ which is a stable std::vector (never resized
//      after construction).

gc_kv_manager_t::gc_kv_manager_t(const gc_kv_manager_params_t & p)
    : params_(p), pool_(p.num_blocks) {}

int gc_kv_manager_t::blocks_needed(int num_tokens) const {
    if (num_tokens <= 0) return 0;
    return (num_tokens + params_.block_size - 1) / params_.block_size;
}

gc_kv_manager_t::cache_hit_t gc_kv_manager_t::find_cache_hit(
        const std::vector<gc_block_hash_t> & block_hashes) const {
    cache_hit_t hit;
    if (!params_.enable_prefix) return hit;

    for (gc_block_hash_t h : block_hashes) {
        gc_kv_block_t * b = pool_.get_cached_block(h);
        if (!b) break;
        hit.blocks.push_back(b);
        hit.num_tokens += params_.block_size;
    }
    return hit;
}

// ── allocate_slots ────────────────────────────────────────────────────────────
// Called on every scheduling step for every scheduled request.
//
// For NEW requests (not yet in req_blocks_):
//   - Prepend cached_blocks (prefix hits) — touch them to pin
//   - Append fresh blocks to reach num_tokens capacity
//
// For RUNNING requests (already in req_blocks_):
//   - Block table is extended (append-only)
//   - cached_blocks MUST be empty (running requests have no new prefix hits)
//   - Only fresh blocks are allocated to cover new token slots
//
// Returns nullptr if not enough free blocks are available.

const gc_req_blocks_t * gc_kv_manager_t::allocate_slots(
        const std::string                  & request_id,
        int                                  num_tokens,
        const std::vector<gc_kv_block_t *> & cached_blocks,
        const std::vector<gc_block_hash_t> & block_hashes,
        int                                  num_computed) {
    int total_needed = blocks_needed(num_tokens);

    auto it = req_blocks_.find(request_id);
    bool is_running = (it != req_blocks_.end());

    if (is_running) {
        // Running request: block table already exists, only extend it.
        assert(cached_blocks.empty() &&
               "prefix cache hits impossible for already-running request");

        gc_req_blocks_t & rb = it->second;
        int already_have = (int)rb.blocks.size();
        int new_needed   = total_needed - already_have;
        if (new_needed <= 0) return &rb; // enough slots already

        if (new_needed > pool_.num_free_blocks()) return nullptr;

        auto new_blks = pool_.get_new_blocks(new_needed);
        if ((int)new_blks.size() < new_needed) {
            // Should not happen given the check above.
            pool_.free_blocks(new_blks);
            return nullptr;
        }
        rb.blocks.insert(rb.blocks.end(), new_blks.begin(), new_blks.end());

        if (params_.enable_prefix) {
            cache_blocks(request_id, block_hashes, num_computed);
        }
        return &rb;
    }

    // New request: build block table from scratch.
    int already_have    = (int)cached_blocks.size();
    int new_blocks_need = total_needed - already_have;
    if (new_blocks_need < 0) new_blocks_need = 0;

    // Count evictable cached blocks — they may need to come off the free queue.
    int evictable = 0;
    for (auto * b : cached_blocks) {
        if (!b->is_null && b->ref_cnt == 0) evictable++;
    }
    int free_needed = new_blocks_need + evictable;
    if (free_needed > pool_.num_free_blocks()) return nullptr;

    // Touch (pin) the cached hit blocks so they won't be evicted.
    pool_.touch(cached_blocks);

    // Allocate fresh blocks for the remainder.
    auto new_blks = pool_.get_new_blocks(new_blocks_need);
    if ((int)new_blks.size() < new_blocks_need) {
        pool_.free_blocks(new_blks);
        // Undo touch — decrement ref_counts we just incremented.
        pool_.free_blocks(cached_blocks);
        return nullptr;
    }

    gc_req_blocks_t & rb = req_blocks_[request_id];
    // Block table must be empty for a new request.
    assert(rb.blocks.empty() && "block table must be empty for new request");
    rb.blocks.insert(rb.blocks.end(), cached_blocks.begin(), cached_blocks.end());
    rb.blocks.insert(rb.blocks.end(), new_blks.begin(),     new_blks.end());

    // Validate: no duplicate writable block ownership.
    // Each block must appear at most once in this table.
    // (prefix-cached blocks can be shared across requests but not duplicated
    //  within a single request's table.)
    assert([&]() {
        std::unordered_set<int32_t> seen;
        for (auto * b : rb.blocks) {
            if (b->is_null) continue;
            if (!seen.insert(b->block_id).second) return false;
        }
        return true;
    }() && "duplicate block_id in request block table");

    if (params_.enable_prefix) {
        cache_blocks(request_id, block_hashes, num_computed);
    }
    return &rb;
}

void gc_kv_manager_t::cache_blocks(
        const std::string                  & request_id,
        const std::vector<gc_block_hash_t> & block_hashes,
        int                                  num_computed_tokens) {
    auto it = req_blocks_.find(request_id);
    if (it == req_blocks_.end()) return;

    const auto & blks = it->second.blocks;
    int full_blocks = num_computed_tokens / params_.block_size;
    full_blocks = std::min(full_blocks, (int)blks.size());
    full_blocks = std::min(full_blocks, (int)block_hashes.size());

    for (int i = 0; i < full_blocks; ++i) {
        pool_.cache_block(block_hashes[i], blks[i]);
    }
}

void gc_kv_manager_t::free(const std::string & request_id) {
    auto it = req_blocks_.find(request_id);
    if (it == req_blocks_.end()) return;

    // Free in reverse order so tail blocks are evicted first (LRU-friendly).
    auto & blks = it->second.blocks;
    std::vector<gc_kv_block_t *> rev(blks.rbegin(), blks.rend());
    pool_.free_blocks(rev);
    req_blocks_.erase(it);
}

const gc_req_blocks_t * gc_kv_manager_t::get_blocks(const std::string & request_id) const {
    auto it = req_blocks_.find(request_id);
    if (it == req_blocks_.end()) return nullptr;
    return &it->second;
}
