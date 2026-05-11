#include "gc_engine.h"

#include <algorithm>
#include <cassert>

// ── Constructor ───────────────────────────────────────────────────────────────

gc_engine_t::gc_engine_t(const gc_engine_params_t & p, gc_model_runner_t * runner)
    : params_(p), sched_(p.sched), runner_(runner)
{
    assert(runner_ != nullptr);
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

void gc_engine_t::add_request(std::unique_ptr<gc_request_t> req) {
    sched_.add_request(std::move(req));
}

void gc_engine_t::abort_request(const std::string & req_id) {
    sched_.abort_request(req_id);
}

// ── build_batch() ─────────────────────────────────────────────────────────────

gc_batch_t gc_engine_t::build_batch(const gc_sched_output_t & sched_out) const {
    gc_batch_t batch;

    // New requests: send full prompt tokens (or the scheduled chunk).
    for (const gc_new_req_data_t & nr : sched_out.new_reqs) {
        auto it = sched_out.num_scheduled_tokens.find(nr.req_id);
        int n = (it != sched_out.num_scheduled_tokens.end()) ? it->second : 0;
        if (n <= 0) continue;

        gc_batch_entry_t e;
        e.req_id             = nr.req_id;
        e.block_ids          = nr.block_ids;  // full block table from allocate_slots
        e.num_computed       = nr.num_computed_tokens;
        e.num_new_tokens     = n;
        e.num_prompt_tokens  = (int)nr.prompt_token_ids.size();
        e.is_prefill         = true;

        // token_ids = prompt[num_computed : num_computed + n]
        int start = nr.num_computed_tokens;
        int end   = std::min(start + n, (int)nr.prompt_token_ids.size());
        e.token_ids.assign(
            nr.prompt_token_ids.begin() + start,
            nr.prompt_token_ids.begin() + end);

        // Final prefill chunk when this step covers the last prompt token.
        e.is_last_prefill = ((e.num_computed + (int)e.token_ids.size()) >= e.num_prompt_tokens);

        // Absolute position ids for RoPE
        e.position_ids.resize(e.token_ids.size());
        for (int i = 0; i < (int)e.token_ids.size(); ++i) {
            e.position_ids[i] = nr.num_computed_tokens + i;
        }

        batch.total_tokens += n;
        batch.entries.push_back(std::move(e));
    }

    // Cached (running / resumed) requests.
    for (const gc_cached_req_entry_t & cr : sched_out.cached_reqs) {
        auto it = sched_out.num_scheduled_tokens.find(cr.req_id);
        int n = (it != sched_out.num_scheduled_tokens.end()) ? it->second : 0;
        if (n <= 0) continue;

        gc_batch_entry_t e;
        e.req_id            = cr.req_id;
        e.block_ids         = cr.block_ids;  // COMPLETE logical block table
        e.num_computed      = cr.num_computed_tokens;
        e.num_new_tokens    = n;
        e.num_prompt_tokens = cr.num_prompt_tokens;
        e.is_prefill        = (cr.num_computed_tokens < cr.num_prompt_tokens);

        if (e.is_prefill) {
            // Prefill continuation: next chunk of prompt tokens
            int start = cr.num_computed_tokens;
            int end   = std::min(start + n, (int)cr.all_token_ids.size());
            e.token_ids.assign(cr.all_token_ids.begin() + start,
                               cr.all_token_ids.begin() + end);
            // Final prefill when this chunk reaches the end of the prompt
            e.is_last_prefill = ((e.num_computed + (int)e.token_ids.size()) >= e.num_prompt_tokens);
        } else {
            // Decode step: forward the last generated token
            e.token_ids      = { cr.last_token };
            e.is_last_prefill = false;
        }

        // Absolute position ids for RoPE
        e.position_ids.resize(e.token_ids.size());
        for (int i = 0; i < (int)e.token_ids.size(); ++i) {
            e.position_ids[i] = cr.num_computed_tokens + i;
        }

        batch.total_tokens += n;
        batch.entries.push_back(std::move(e));
    }

    return batch;
}

// ── process_output() ─────────────────────────────────────────────────────────

gc_step_output_t gc_engine_t::process_output(
        const gc_sched_output_t & sched_out,
        const gc_model_output_t & model_out) {

    // Build req_id → sampled_token map from model output (skip prefill-only entries).
    std::unordered_map<std::string, int32_t> new_tokens;
    for (size_t i = 0; i < model_out.req_ids.size(); ++i) {
        if (i < model_out.sampled_tokens.size() && model_out.sampled_tokens[i] >= 0) {
            new_tokens[model_out.req_ids[i]] = model_out.sampled_tokens[i];
        }
    }

    // Update scheduler (finish detection, block caching).
    sched_.update_from_output(new_tokens);

    // Build step output.
    gc_step_output_t out;
    out.total_scheduled_tokens = sched_out.total_scheduled_tokens;
    out.had_work = (sched_out.total_scheduled_tokens > 0);

    // Requests that produced a real token this step.
    for (auto & [req_id, tok] : new_tokens) {
        if (tok < 0) continue;  // prefill-only steps return -1

        gc_req_output_t ro;
        ro.req_id = req_id;
        ro.token  = tok;

        // A request is finished if update_from_output() moved it out of the
        // running set (kv_mgr_ entry freed) OR it appears in finished_req_ids.
        // kv_mgr_proxy returns nullptr both for unknown requests AND for
        // finished ones — both mean "not running", which is what we want here.
        const bool in_finished_set = (sched_out.finished_req_ids.count(req_id) > 0);
        const bool no_longer_running = (sched_.kv_mgr_proxy(req_id) == nullptr);
        ro.finished = in_finished_set || no_longer_running;
        if (ro.finished) {
            ro.finish_reason = GC_REQ_FINISHED_STOPPED;
        }
        out.outputs.push_back(ro);
    }

    // Also report requests that finished this step without generating a token
    // (aborted, or finished by the scheduler for other reasons).
    for (const auto & fid : sched_out.finished_req_ids) {
        bool already = false;
        for (auto & o : out.outputs) { if (o.req_id == fid) { already = true; break; } }
        if (!already) {
            gc_req_output_t ro;
            ro.req_id        = fid;
            ro.token         = -1;
            ro.finished      = true;
            ro.finish_reason = GC_REQ_FINISHED_STOPPED;
            out.outputs.push_back(ro);
        }
    }

    return out;
}

// ── step() ───────────────────────────────────────────────────────────────────

gc_step_output_t gc_engine_t::step() {
    if (!has_work()) return {};

    gc_sched_output_t sched_out = sched_.schedule();
    if (sched_out.empty()) return {};

    gc_batch_t batch = build_batch(sched_out);

    gc_model_output_t model_out;
    if (!batch.empty()) {
        model_out = runner_->execute(batch);
        // Advance num_computed_tokens AFTER confirmed execution.
        sched_.update_computed_tokens(sched_out.num_scheduled_tokens);
    }

    return process_output(sched_out, model_out);
}

// ── get_blocks proxy ─────────────────────────────────────────────────────────

const gc_req_blocks_t * gc_engine_t::get_blocks(const std::string & req_id) const {
    return sched_.kv_mgr_proxy(req_id);
}
