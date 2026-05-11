// test_gc_kvcache.cpp — unit tests for Phase 6 paged KV cache

#include "gc_kvcache.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
} while (0)

// ── Hash ─────────────────────────────────────────────────────────────────────

static void test_hash_deterministic() {
    int32_t toks[] = {1, 2, 3, 4};
    gc_block_hash_t h1 = gc_hash_block_tokens(GC_BLOCK_HASH_NONE, toks, 4);
    gc_block_hash_t h2 = gc_hash_block_tokens(GC_BLOCK_HASH_NONE, toks, 4);
    CHECK(h1 == h2);
    CHECK(h1 != GC_BLOCK_HASH_NONE);
}

static void test_hash_chaining() {
    int32_t toks[] = {1, 2, 3, 4};
    gc_block_hash_t h0 = gc_hash_block_tokens(GC_BLOCK_HASH_NONE, toks, 4);
    gc_block_hash_t h1 = gc_hash_block_tokens(h0,               toks, 4);
    gc_block_hash_t h2 = gc_hash_block_tokens(GC_BLOCK_HASH_NONE, toks, 4);
    // chained hash must differ from non-chained
    CHECK(h1 != h2);
    CHECK(h0 != h1);
}

static void test_hash_extra_key() {
    int32_t toks[] = {5, 6, 7};
    gc_block_hash_t h_no_extra  = gc_hash_block_tokens(GC_BLOCK_HASH_NONE, toks, 3, 0);
    gc_block_hash_t h_extra     = gc_hash_block_tokens(GC_BLOCK_HASH_NONE, toks, 3, 42);
    CHECK(h_no_extra != h_extra);
}

// ── Free queue ────────────────────────────────────────────────────────────────

static void test_free_queue_basic() {
    gc_kv_block_t a, b, c;
    a.block_id = 1; b.block_id = 2; c.block_id = 3;

    gc_free_queue_t q;
    CHECK(q.empty());
    CHECK(q.size() == 0);

    q.append(&a);
    q.append(&b);
    q.append(&c);
    CHECK(q.size() == 3);

    gc_kv_block_t * p = q.popleft();
    CHECK(p == &a);
    CHECK(q.size() == 2);

    p = q.popleft();
    CHECK(p == &b);
    CHECK(q.size() == 1);
}

static void test_free_queue_remove_middle() {
    gc_kv_block_t a, b, c;
    a.block_id = 1; b.block_id = 2; c.block_id = 3;

    gc_free_queue_t q;
    q.append(&a); q.append(&b); q.append(&c);

    q.remove(&b);
    CHECK(q.size() == 2);

    gc_kv_block_t * p1 = q.popleft();
    gc_kv_block_t * p2 = q.popleft();
    CHECK(p1 == &a);
    CHECK(p2 == &c);
    CHECK(q.empty());
}

static void test_free_queue_popleft_n() {
    gc_free_queue_t q;
    const int N = 5;
    gc_kv_block_t blks[N];
    for (int i = 0; i < N; ++i) { blks[i].block_id = i; q.append(&blks[i]); }

    auto got = q.popleft_n(3);
    CHECK((int)got.size() == 3);
    CHECK(got[0] == &blks[0]);
    CHECK(got[1] == &blks[1]);
    CHECK(got[2] == &blks[2]);
    CHECK(q.size() == 2);
}

// ── Block pool ────────────────────────────────────────────────────────────────

static void test_pool_null_block() {
    gc_block_pool_t pool(8);
    gc_kv_block_t * nb = pool.null_block();
    CHECK(nb->block_id == 0);
    CHECK(nb->is_null == true);
    CHECK(nb->ref_cnt == 1);
}

static void test_pool_free_count() {
    gc_block_pool_t pool(8); // 1 null + 7 free
    CHECK(pool.num_free_blocks() == 7);
    CHECK(pool.total_blocks() == 8);
}

static void test_pool_get_new_blocks() {
    gc_block_pool_t pool(8);
    auto blks = pool.get_new_blocks(3);
    CHECK((int)blks.size() == 3);
    CHECK(pool.num_free_blocks() == 4);
    for (auto * b : blks) {
        CHECK(b->block_id != 0);
        CHECK(b->ref_cnt == 1);
        CHECK(!b->is_null);
    }
}

static void test_pool_get_too_many() {
    gc_block_pool_t pool(4); // 3 free
    auto blks = pool.get_new_blocks(4);
    CHECK(blks.empty());
    CHECK(pool.num_free_blocks() == 3);
}

static void test_pool_free_returns_to_queue() {
    gc_block_pool_t pool(8);
    auto blks = pool.get_new_blocks(3);
    CHECK(pool.num_free_blocks() == 4);
    pool.free_blocks(blks);
    CHECK(pool.num_free_blocks() == 7);
}

static void test_pool_usage() {
    gc_block_pool_t pool(8); // 7 usable
    CHECK(pool.usage() == 0.0f);
    auto blks = pool.get_new_blocks(7);
    CHECK(pool.usage() == 1.0f);
    pool.free_blocks(blks);
    CHECK(pool.usage() == 0.0f);
}

// ── Prefix cache ──────────────────────────────────────────────────────────────

static void test_pool_cache_and_lookup() {
    gc_block_pool_t pool(8);
    auto blks = pool.get_new_blocks(2);
    gc_block_hash_t h = 0xDEADBEEFCAFE;

    pool.cache_block(h, blks[0]);
    gc_kv_block_t * found = pool.get_cached_block(h);
    CHECK(found == blks[0]);
    CHECK(blks[0]->block_hash.has_value());
    CHECK(blks[0]->block_hash.value() == h);

    // Miss on unknown hash
    CHECK(pool.get_cached_block(0xBADBADBAD) == nullptr);
}

static void test_pool_evict() {
    gc_block_pool_t pool(8);
    auto blks = pool.get_new_blocks(1);
    gc_block_hash_t h = 0x1234;
    pool.cache_block(h, blks[0]);
    CHECK(pool.get_cached_block(h) == blks[0]);

    pool.evict_blocks({blks[0]->block_id});
    CHECK(pool.get_cached_block(h) == nullptr);
    CHECK(!blks[0]->block_hash.has_value());
}

static void test_pool_reset_prefix_cache() {
    gc_block_pool_t pool(4);
    auto blks = pool.get_new_blocks(2);
    pool.cache_block(0xAA, blks[0]);
    pool.cache_block(0xBB, blks[1]);
    pool.free_blocks(blks);   // ref_cnt back to 0, in free queue

    bool ok = pool.reset_prefix_cache();
    CHECK(ok);
    CHECK(pool.get_cached_block(0xAA) == nullptr);
    CHECK(pool.get_cached_block(0xBB) == nullptr);
}

static void test_pool_evict_lru_when_full() {
    // Pool has 3 usable blocks; fill all, then get_new_blocks should evict LRU cached.
    gc_block_pool_t pool(4); // 3 free
    auto blks = pool.get_new_blocks(3);
    gc_block_hash_t h0 = 0x11, h1 = 0x22, h2 = 0x33;
    pool.cache_block(h0, blks[0]);
    pool.cache_block(h1, blks[1]);
    pool.cache_block(h2, blks[2]);
    pool.free_blocks(blks); // all go back to free queue (cached, ref=0)
    CHECK(pool.num_free_blocks() == 3);

    // Now allocate 1 — should pop LRU (blks[0], oldest freed)
    auto nb = pool.get_new_blocks(1);
    CHECK((int)nb.size() == 1);
    CHECK(nb[0] == blks[0]);
    // Its hash must be evicted from map
    CHECK(pool.get_cached_block(h0) == nullptr);
    // Others still cached
    CHECK(pool.get_cached_block(h1) == blks[1]);
    CHECK(pool.get_cached_block(h2) == blks[2]);
}

// ── KV manager ───────────────────────────────────────────────────────────────

static std::vector<gc_block_hash_t> make_hashes(int n) {
    int32_t dummy[4] = {0, 1, 2, 3};
    std::vector<gc_block_hash_t> hs;
    gc_block_hash_t prev = GC_BLOCK_HASH_NONE;
    for (int i = 0; i < n; ++i) {
        dummy[0] = i;
        prev = gc_hash_block_tokens(prev, dummy, 4);
        hs.push_back(prev);
    }
    return hs;
}

static void test_manager_allocate_basic() {
    gc_kv_manager_params_t p;
    p.num_blocks   = 16;
    p.block_size   = 4;
    p.enable_prefix = true;

    gc_kv_manager_t mgr(p);
    auto hashes = make_hashes(4);

    auto hit = mgr.find_cache_hit(hashes);
    CHECK(hit.blocks.empty());
    CHECK(hit.num_tokens == 0);

    const gc_req_blocks_t * rb = mgr.allocate_slots("req0", 16, hit.blocks, hashes, 0);
    CHECK(rb != nullptr);
    CHECK(rb->num_blocks() == 4);
    for (auto * b : rb->blocks) CHECK(b->block_id != 0);

    CHECK(mgr.num_free_blocks() == 11); // 15 - 4
}

static void test_manager_free() {
    gc_kv_manager_params_t p;
    p.num_blocks   = 8;
    p.block_size   = 4;
    p.enable_prefix = false;

    gc_kv_manager_t mgr(p);
    auto hashes = make_hashes(1);
    mgr.allocate_slots("req1", 4, {}, hashes, 0);
    CHECK(mgr.num_free_blocks() == 6);
    mgr.free("req1");
    CHECK(mgr.num_free_blocks() == 7);
}

static void test_manager_prefix_cache_hit() {
    gc_kv_manager_params_t p;
    p.num_blocks   = 16;
    p.block_size   = 4;
    p.enable_prefix = true;

    gc_kv_manager_t mgr(p);
    auto hashes = make_hashes(3);

    // First request: allocate + compute all 3 blocks
    mgr.allocate_slots("req_a", 12, {}, hashes, 12 /*num_computed*/);
    mgr.free("req_a");
    // Blocks are freed (ref_cnt=0) but cached in prefix map

    CHECK(mgr.num_free_blocks() == 15);

    // Second request with same prefix: should get cache hit
    auto hit = mgr.find_cache_hit(hashes);
    CHECK(hit.num_tokens == 12);
    CHECK((int)hit.blocks.size() == 3);

    // Allocate for second request — no new blocks needed
    const gc_req_blocks_t * rb = mgr.allocate_slots("req_b", 12, hit.blocks, hashes, 12);
    CHECK(rb != nullptr);
    CHECK(rb->num_blocks() == 3);
    // Blocks were pinned via touch(), so free count goes down
    CHECK(mgr.num_free_blocks() == 12);
}

static void test_manager_partial_prefix_hit() {
    gc_kv_manager_params_t p;
    p.num_blocks   = 16;
    p.block_size   = 4;
    p.enable_prefix = true;

    gc_kv_manager_t mgr(p);
    auto hashes4 = make_hashes(4);
    auto hashes2 = std::vector<gc_block_hash_t>(hashes4.begin(), hashes4.begin() + 2);

    // Cache 2 blocks
    mgr.allocate_slots("req_x", 8, {}, hashes2, 8);
    mgr.free("req_x");

    // New request needs 4 blocks, first 2 are cached
    auto hit = mgr.find_cache_hit(hashes4);
    CHECK(hit.num_tokens == 8);
    CHECK((int)hit.blocks.size() == 2);

    const gc_req_blocks_t * rb = mgr.allocate_slots("req_y", 16, hit.blocks, hashes4, 8);
    CHECK(rb != nullptr);
    CHECK(rb->num_blocks() == 4);
}

static void test_manager_insufficient_blocks() {
    gc_kv_manager_params_t p;
    p.num_blocks   = 4; // only 3 usable
    p.block_size   = 4;
    p.enable_prefix = false;

    gc_kv_manager_t mgr(p);
    auto hashes = make_hashes(4);
    const gc_req_blocks_t * rb = mgr.allocate_slots("req_z", 16, {}, hashes, 0);
    CHECK(rb == nullptr);
    CHECK(mgr.num_free_blocks() == 3);
}

static void test_manager_get_blocks() {
    gc_kv_manager_params_t p;
    p.num_blocks   = 8;
    p.block_size   = 4;
    p.enable_prefix = false;

    gc_kv_manager_t mgr(p);
    CHECK(mgr.get_blocks("nonexistent") == nullptr);

    auto hashes = make_hashes(1);
    mgr.allocate_slots("req_g", 4, {}, hashes, 0);
    CHECK(mgr.get_blocks("req_g") != nullptr);
    mgr.free("req_g");
    CHECK(mgr.get_blocks("req_g") == nullptr);
}

static void test_manager_block_ids() {
    gc_kv_manager_params_t p;
    p.num_blocks   = 8;
    p.block_size   = 4;
    p.enable_prefix = false;

    gc_kv_manager_t mgr(p);
    auto hashes = make_hashes(2);
    const gc_req_blocks_t * rb = mgr.allocate_slots("req_ids", 8, {}, hashes, 0);
    CHECK(rb != nullptr);
    auto ids = rb->block_ids();
    CHECK((int)ids.size() == 2);
    for (int32_t id : ids) CHECK(id > 0); // no null block
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_hash_deterministic();
    test_hash_chaining();
    test_hash_extra_key();

    test_free_queue_basic();
    test_free_queue_remove_middle();
    test_free_queue_popleft_n();

    test_pool_null_block();
    test_pool_free_count();
    test_pool_get_new_blocks();
    test_pool_get_too_many();
    test_pool_free_returns_to_queue();
    test_pool_usage();
    test_pool_cache_and_lookup();
    test_pool_evict();
    test_pool_reset_prefix_cache();
    test_pool_evict_lru_when_full();

    test_manager_allocate_basic();
    test_manager_free();
    test_manager_prefix_cache_hit();
    test_manager_partial_prefix_hit();
    test_manager_insufficient_blocks();
    test_manager_get_blocks();
    test_manager_block_ids();

    fprintf(stderr, "\n%s  pass=%d  fail=%d\n",
            g_fail ? "FAILED" : "PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
