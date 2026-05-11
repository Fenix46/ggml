#include "gc_request.h"

#include <cassert>
#include <cmath>

gc_request_t::gc_request_t(
        std::string          request_id_,
        std::vector<int32_t> prompt_token_ids_,
        gc_sampling_params_t sampling_params_,
        int                  priority_,
        double               arrival_time_)
    : request_id    (std::move(request_id_))
    , priority      (priority_)
    , arrival_time  (arrival_time_)
    , sampling_params(sampling_params_)
    , prompt_token_ids(std::move(prompt_token_ids_))
{}

void gc_request_t::update_block_hashes(int block_size) {
    // Compute hashes for all full blocks in prompt+output tokens.
    // Incremental: only hashes for blocks beyond block_hashes.size().
    assert(block_size > 0);

    // Build the combined token view lazily.
    // all_tokens = prompt_token_ids + output_token_ids
    // We index into them by position.
    auto token_at = [&](int idx) -> int32_t {
        if (idx < (int)prompt_token_ids.size())
            return prompt_token_ids[idx];
        return output_token_ids[idx - (int)prompt_token_ids.size()];
    };

    int total = num_tokens();
    int num_full_blocks = total / block_size;
    int already_hashed  = (int)block_hashes.size();

    for (int i = already_hashed; i < num_full_blocks; ++i) {
        gc_block_hash_t parent = (i == 0) ? GC_BLOCK_HASH_NONE : block_hashes[i - 1];
        // Collect block's token ids into a contiguous array.
        std::vector<int32_t> toks(block_size);
        for (int j = 0; j < block_size; ++j) {
            toks[j] = token_at(i * block_size + j);
        }
        block_hashes.push_back(gc_hash_block_tokens(parent, toks.data(), (size_t)block_size));
    }
}

void gc_request_t::append_output_token(int32_t token_id, int block_size) {
    output_token_ids.push_back(token_id);
    // Update block_hashes if a new full block is now complete.
    update_block_hashes(block_size);
}
