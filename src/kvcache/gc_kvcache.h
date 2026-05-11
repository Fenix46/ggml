#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>
#include <optional>

// ── Block hash ───────────────────────────────────────────────────────────────
// 64-bit rolling hash: FNV-1a over (parent_hash, token_ids[], extra_key).
// Mirrors vllm's hash_block_tokens() but without Python pickle overhead.

typedef uint64_t gc_block_hash_t;

// Sentinel: "no parent" / empty hash
static constexpr gc_block_hash_t GC_BLOCK_HASH_NONE = 0ULL;

gc_block_hash_t gc_hash_block_tokens(
    gc_block_hash_t       parent_hash,
    const int32_t       * token_ids,
    size_t                n_tokens,
    uint64_t              extra_key = 0);   // LoRA id, salt, etc.

// ── KV block ─────────────────────────────────────────────────────────────────

struct gc_kv_block_t {
    int32_t  block_id   = -1;
    int32_t  ref_cnt    = 0;
    bool     is_null    = false;

    // Set once a full block is committed to prefix cache; nullopt otherwise.
    std::optional<gc_block_hash_t> block_hash;

    // Intrusive doubly-linked list for FreeQueue.
    // nullptr when block is NOT in the free list.
    gc_kv_block_t * prev_free = nullptr;
    gc_kv_block_t * next_free = nullptr;
};

// ── Free block queue (LRU doubly-linked list) ─────────────────────────────────
// Head = LRU (oldest, evicted first).  Tail = MRU (most recently freed).
// Fake sentinel nodes so every real block always has valid prev/next.

struct gc_free_queue_t {
    gc_free_queue_t();
    ~gc_free_queue_t() = default;

    // Non-copyable
    gc_free_queue_t(const gc_free_queue_t &)            = delete;
    gc_free_queue_t & operator=(const gc_free_queue_t &) = delete;

    // Pop LRU block (head side).
    gc_kv_block_t * popleft();

    // Pop n LRU blocks.
    std::vector<gc_kv_block_t *> popleft_n(int n);

    // O(1) removal of an arbitrary block (e.g. cache hit → promote).
    void remove(gc_kv_block_t * blk);

    // Append block to tail (MRU).
    void append(gc_kv_block_t * blk);

    // Append multiple blocks to tail.
    void append_n(const std::vector<gc_kv_block_t *> & blks);

    int  size()  const { return num_free; }
    bool empty() const { return num_free == 0; }

private:
    gc_kv_block_t head_sentinel; // fake, never in block array
    gc_kv_block_t tail_sentinel; // fake, never in block array
    int           num_free = 0;
};

// ── Block pool ────────────────────────────────────────────────────────────────
// Owns all gc_kv_block_t objects.  Prefix-cache map lives here.

struct gc_block_pool_t {
    // num_blocks includes the null block (block_id=0).
    explicit gc_block_pool_t(int num_blocks);
    ~gc_block_pool_t() = default;

    gc_block_pool_t(const gc_block_pool_t &)            = delete;
    gc_block_pool_t & operator=(const gc_block_pool_t &) = delete;

    // ── Allocation ──────────────────────────────────────────────────────────

    // Get n free blocks (LRU-evict cached if necessary).
    // Returns empty vector if not enough blocks available.
    std::vector<gc_kv_block_t *> get_new_blocks(int n);

    // Increment ref_cnt, remove from free queue if it was there.
    void touch(const std::vector<gc_kv_block_t *> & blks);

    // Decrement ref_cnt; if reaches 0, append to free queue tail.
    void free_blocks(const std::vector<gc_kv_block_t *> & blks);

    // ── Prefix cache ────────────────────────────────────────────────────────

    // Register a fully computed block under hash.
    // If a different block already holds the hash, ref-count logic applies.
    void cache_block(gc_block_hash_t hash, gc_kv_block_t * blk);

    // Look up a cached block by hash.  Returns nullptr on miss.
    gc_kv_block_t * get_cached_block(gc_block_hash_t hash) const;

    // Evict specific block IDs from prefix cache (for KV transfer / RLHF).
    void evict_blocks(const std::vector<int> & block_ids);

    // Invalidate all prefix-cached blocks.
    bool reset_prefix_cache();

    // ── Stats ────────────────────────────────────────────────────────────────

    int    num_free_blocks()  const;
    float  usage()            const;   // 0.0 – 1.0

    // Null block accessor (block_id = 0, is_null = true).
    gc_kv_block_t * null_block() { return &blocks_[0]; }

    int total_blocks() const { return (int)blocks_.size(); }

private:
    std::vector<gc_kv_block_t>                        blocks_;
    gc_free_queue_t                                   free_queue_;
    std::unordered_map<gc_block_hash_t, gc_kv_block_t *> hash_to_block_;
};

// ── Per-request block table ───────────────────────────────────────────────────
// Mirrors vllm's per-request block list (single KV cache group, simple models).
// Phase 8 will wrap this into a multi-group coordinator.

struct gc_req_blocks_t {
    std::vector<gc_kv_block_t *> blocks;  // ordered: block[0] = first tokens

    int  num_blocks() const { return (int)blocks.size(); }
    bool empty()      const { return blocks.empty(); }

    // Collect block_ids for dispatch to GGML backend.
    std::vector<int32_t> block_ids() const;
};

// ── KV cache manager ─────────────────────────────────────────────────────────
// Thin coordinator.  All state lives in gc_block_pool_t.
// Maps request_id → gc_req_blocks_t.

struct gc_kv_manager_params_t {
    int  num_blocks    = 0;
    int  block_size    = 16;    // tokens per block
    bool enable_prefix = true;
};

struct gc_kv_manager_t {
    explicit gc_kv_manager_t(const gc_kv_manager_params_t & p);
    ~gc_kv_manager_t() = default;

    gc_kv_manager_t(const gc_kv_manager_t &)            = delete;
    gc_kv_manager_t & operator=(const gc_kv_manager_t &) = delete;

    // ── Prefix cache ────────────────────────────────────────────────────────

    // Find longest prefix-cache hit for request's block_hashes.
    // Returns {cached_blocks, num_cached_tokens}.
    // num_cached_tokens is always block_size-aligned.
    struct cache_hit_t {
        std::vector<gc_kv_block_t *> blocks;
        int                          num_tokens = 0;
    };
    cache_hit_t find_cache_hit(
        const std::vector<gc_block_hash_t> & block_hashes) const;

    // ── Slot allocation ──────────────────────────────────────────────────────

    // Allocate or extend slots for a request.
    //   request_id    : unique string id
    //   num_tokens    : total tokens needed (including already computed)
    //   cached_blocks : from find_cache_hit()
    //   block_hashes  : full list of hashes for this request (for caching)
    //   num_computed  : tokens already computed (used to cache full blocks)
    // Returns nullptr if not enough free blocks.
    const gc_req_blocks_t * allocate_slots(
        const std::string                  & request_id,
        int                                  num_tokens,
        const std::vector<gc_kv_block_t *> & cached_blocks,
        const std::vector<gc_block_hash_t> & block_hashes,
        int                                  num_computed);

    // Cache newly completed blocks for a request.
    void cache_blocks(
        const std::string                  & request_id,
        const std::vector<gc_block_hash_t> & block_hashes,
        int                                  num_computed_tokens);

    // Free all blocks for a request.
    void free(const std::string & request_id);

    // ── Stats / misc ─────────────────────────────────────────────────────────

    int   num_free_blocks()  const { return pool_.num_free_blocks(); }
    float usage()            const { return pool_.usage(); }
    bool  reset_prefix_cache()     { return pool_.reset_prefix_cache(); }

    const gc_kv_manager_params_t & params() const { return params_; }

    // Get current blocks for a request (nullptr if not found).
    const gc_req_blocks_t * get_blocks(const std::string & request_id) const;

private:
    gc_kv_manager_params_t                              params_;
    gc_block_pool_t                                     pool_;
    std::unordered_map<std::string, gc_req_blocks_t>    req_blocks_;

    int blocks_needed(int num_tokens) const;
};
