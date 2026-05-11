#include "gc_scheduler.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

// ── Constructor ───────────────────────────────────────────────────────────────

gc_scheduler_t::gc_scheduler_t(const gc_scheduler_params_t & p)
    : params_(p)
    , kv_mgr_({p.num_blocks, p.block_size, p.enable_prefix_cache})
{}

// ── Waiting queue helpers ─────────────────────────────────────────────────────

bool gc_scheduler_t::waiting_empty() const {
    if (params_.policy == GC_SCHED_PRIORITY) return waiting_prio_.empty();
    return waiting_fcfs_.empty();
}

gc_request_t * gc_scheduler_t::waiting_front() const {
    if (params_.policy == GC_SCHED_PRIORITY) return waiting_prio_.top();
    return waiting_fcfs_.front();
}

gc_request_t * gc_scheduler_t::waiting_pop() {
    if (params_.policy == GC_SCHED_PRIORITY) {
        gc_request_t * r = waiting_prio_.top();
        waiting_prio_.pop();
        return r;
    }
    gc_request_t * r = waiting_fcfs_.front();
    waiting_fcfs_.pop_front();
    return r;
}

void gc_scheduler_t::waiting_push_front(gc_request_t * req) {
    // For PRIORITY queue there's no "front" concept — just re-insert.
    if (params_.policy == GC_SCHED_PRIORITY) {
        waiting_prio_.push(req);
    } else {
        waiting_fcfs_.push_front(req);
    }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

void gc_scheduler_t::add_request(std::unique_ptr<gc_request_t> req) {
    req->status = GC_REQ_WAITING;
    req->update_block_hashes(params_.block_size);
    gc_request_t * raw = req.get();
    all_reqs_[req->request_id] = std::move(req);
    if (params_.policy == GC_SCHED_PRIORITY) {
        waiting_prio_.push(raw);
    } else {
        waiting_fcfs_.push_back(raw);
    }
}

void gc_scheduler_t::abort_request(const std::string & req_id) {
    auto it = all_reqs_.find(req_id);
    if (it == all_reqs_.end()) return;

    gc_request_t * req = it->second.get();

    if (req->status == GC_REQ_RUNNING) {
        kv_mgr_.free(req_id);
        running_.erase(std::remove(running_.begin(), running_.end(), req), running_.end());
    } else if (req->status == GC_REQ_WAITING || req->status == GC_REQ_PREEMPTED) {
        if (params_.policy == GC_SCHED_FCFS) {
            waiting_fcfs_.erase(
                std::remove(waiting_fcfs_.begin(), waiting_fcfs_.end(), req),
                waiting_fcfs_.end());
        }
        // Priority queue removal is O(n); rebuild.
        if (params_.policy == GC_SCHED_PRIORITY) {
            std::vector<gc_request_t *> tmp;
            while (!waiting_prio_.empty()) {
                gc_request_t * r = waiting_prio_.top(); waiting_prio_.pop();
                if (r != req) tmp.push_back(r);
            }
            for (auto * r : tmp) waiting_prio_.push(r);
        }
    }

    finished_req_ids_.insert(req_id);
    all_reqs_.erase(it);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

int gc_scheduler_t::num_new_tokens_for(const gc_request_t * req, int token_budget) const {
    int remaining = req->num_tokens() - req->num_computed_tokens;
    if (remaining <= 0) remaining = 1;  // at least one decoding step
    int n = std::min(remaining, token_budget);
    n = std::min(n, params_.max_model_len - 1 - req->num_computed_tokens);
    return n;
}

void gc_scheduler_t::preempt(gc_request_t * req) {
    assert(req->status == GC_REQ_RUNNING);
    kv_mgr_.free(req->request_id);
    req->status = GC_REQ_PREEMPTED;
    req->num_computed_tokens = 0;
    req->num_preemptions++;
    // Put back at front of waiting queue.
    waiting_push_front(req);
}

void gc_scheduler_t::finish_request(gc_request_t * req, gc_req_status_t status) {
    req->status = status;
    kv_mgr_.free(req->request_id);
    running_.erase(std::remove(running_.begin(), running_.end(), req), running_.end());
    finished_req_ids_.insert(req->request_id);
}

gc_new_req_data_t gc_scheduler_t::make_new_req_data(const gc_request_t * req) const {
    gc_new_req_data_t d;
    d.req_id             = req->request_id;
    d.prompt_token_ids   = req->prompt_token_ids;
    d.sampling_params    = req->sampling_params;
    d.num_computed_tokens = req->num_computed_tokens;

    const gc_req_blocks_t * rb = kv_mgr_.get_blocks(req->request_id);
    if (rb) {
        auto ids = rb->block_ids();
        d.block_ids.assign(ids.begin(), ids.end());
    }
    return d;
}

gc_cached_req_entry_t gc_scheduler_t::make_cached_entry(
        const gc_request_t * req,
        const std::vector<int32_t> & new_block_ids) const {
    gc_cached_req_entry_t e;
    e.req_id              = req->request_id;
    e.new_block_ids       = new_block_ids;
    e.num_computed_tokens = req->num_computed_tokens;
    e.num_output_tokens   = req->num_output_tokens();
    e.num_prompt_tokens   = req->num_prompt_tokens();
    return e;
}

// ── Main schedule() ───────────────────────────────────────────────────────────

gc_sched_output_t gc_scheduler_t::schedule() {
    gc_sched_output_t out;
    out.finished_req_ids = std::move(finished_req_ids_);
    finished_req_ids_.clear();

    int token_budget = params_.max_num_tokens;

    // ── 1. Schedule RUNNING requests ─────────────────────────────────────────
    // Iterate running_; for each request try to allocate slots.
    // If allocation fails, preempt the lowest-priority running request.

    std::vector<gc_request_t *> scheduled_running;
    // Track which running reqs were scheduled (req → new block ids)
    std::unordered_map<std::string, std::vector<int32_t>> running_new_blocks;

    int idx = 0;
    while (idx < (int)running_.size() && token_budget > 0) {
        gc_request_t * req = running_[idx];

        int num_new = num_new_tokens_for(req, token_budget);
        if (num_new <= 0) { ++idx; continue; }

        // Try to allocate.
        while (true) {
            const gc_req_blocks_t * rb = kv_mgr_.allocate_slots(
                req->request_id,
                req->num_computed_tokens + num_new,
                {},                      // no cached blocks for running reqs
                req->block_hashes,
                req->num_computed_tokens);

            if (rb != nullptr) break;

            // Allocation failed: preempt lowest-priority running request.
            // For FCFS: preempt last (youngest). For PRIORITY: preempt worst.
            if (running_.empty()) break;

            gc_request_t * victim;
            if (params_.policy == GC_SCHED_PRIORITY) {
                victim = *std::max_element(
                    running_.begin(), running_.end(),
                    [](const gc_request_t * a, const gc_request_t * b){ return *a < *b; });
            } else {
                victim = running_.back();
            }

            // If the only candidate is the request itself, give up.
            if (victim == req) { goto cannot_schedule_running; }

            // Un-schedule victim if it was already scheduled this step.
            auto vi = running_new_blocks.find(victim->request_id);
            if (vi != running_new_blocks.end()) {
                auto ns_it = out.num_scheduled_tokens.find(victim->request_id);
                if (ns_it != out.num_scheduled_tokens.end()) {
                    token_budget += ns_it->second;
                    out.total_scheduled_tokens -= ns_it->second;
                    out.num_scheduled_tokens.erase(ns_it);
                }
                running_new_blocks.erase(vi);
                scheduled_running.erase(
                    std::remove(scheduled_running.begin(), scheduled_running.end(), victim),
                    scheduled_running.end());
                if (victim == req) { idx--; }
            }

            running_.erase(std::remove(running_.begin(), running_.end(), victim), running_.end());
            out.preempted_req_ids.insert(victim->request_id);
            preempt(victim);
            if (victim == req) goto cannot_schedule_running;
        }

        {
            // Collect newly allocated block ids.
            const gc_req_blocks_t * rb2 = kv_mgr_.get_blocks(req->request_id);
            std::vector<int32_t> new_blk_ids;
            if (rb2) {
                // New blocks = those beyond what was allocated before.
                // Simplification: send all block ids (engine deduplicates).
                auto all_ids = rb2->block_ids();
                new_blk_ids.assign(all_ids.begin(), all_ids.end());
            }

            scheduled_running.push_back(req);
            running_new_blocks[req->request_id] = new_blk_ids;
            out.num_scheduled_tokens[req->request_id] = num_new;
            out.total_scheduled_tokens += num_new;
            token_budget -= num_new;
        }
        ++idx;
        continue;

    cannot_schedule_running:
        break;
    }

    // ── 2. Schedule WAITING requests ─────────────────────────────────────────
    if (!out.preempted_req_ids.empty()) goto build_output; // no new admits after preemption

    while (!waiting_empty()
        && (int)running_.size() < params_.max_num_running
        && token_budget > 0) {

        gc_request_t * req = waiting_front();

        // Prefix-cache lookup.
        auto hit = kv_mgr_.find_cache_hit(req->block_hashes);
        int num_computed = hit.num_tokens;

        int num_new = req->num_tokens() - num_computed;
        num_new = std::min(num_new, token_budget);
        num_new = std::min(num_new, params_.max_model_len - num_computed);
        if (num_new <= 0) {
            waiting_pop();
            continue;
        }

        const gc_req_blocks_t * rb = kv_mgr_.allocate_slots(
            req->request_id,
            num_computed + num_new,
            hit.blocks,
            req->block_hashes,
            num_computed);

        if (rb == nullptr) break;  // not enough blocks; stop admitting

        waiting_pop();
        req->status = GC_REQ_RUNNING;
        req->num_computed_tokens = num_computed;
        running_.push_back(req);

        auto all_ids = rb->block_ids();
        std::vector<int32_t> blk_ids(all_ids.begin(), all_ids.end());

        bool is_new = (req->num_preemptions == 0 && req->num_output_tokens() == 0);
        if (is_new) {
            out.new_reqs.push_back(make_new_req_data(req));
        } else {
            // resumed after preemption
            out.cached_reqs.push_back(make_cached_entry(req, blk_ids));
        }

        out.num_scheduled_tokens[req->request_id] = num_new;
        out.total_scheduled_tokens += num_new;
        token_budget -= num_new;
    }

build_output:
    // Fill cached_reqs for running requests scheduled in step 1.
    for (gc_request_t * req : scheduled_running) {
        auto & new_blks = running_new_blocks[req->request_id];
        out.cached_reqs.push_back(make_cached_entry(req, new_blks));
    }

    // Advance num_computed_tokens post-schedule (mirrors vllm _update_after_schedule).
    for (auto & [req_id, n] : out.num_scheduled_tokens) {
        auto it = all_reqs_.find(req_id);
        if (it != all_reqs_.end()) {
            it->second->num_computed_tokens += n;
        }
    }

    return out;
}

// ── update_from_output() ──────────────────────────────────────────────────────

void gc_scheduler_t::update_from_output(
        const std::unordered_map<std::string, int32_t> & new_tokens) {
    for (auto & [req_id, tok] : new_tokens) {
        auto it = all_reqs_.find(req_id);
        if (it == all_reqs_.end()) continue;
        gc_request_t * req = it->second.get();
        if (req->status != GC_REQ_RUNNING) continue;

        req->append_output_token(tok, params_.block_size);

        // Cache newly completed blocks.
        kv_mgr_.cache_blocks(req_id, req->block_hashes, req->num_computed_tokens);

        // Check stop conditions.
        bool stop = false;
        gc_req_status_t fin_status = GC_REQ_FINISHED_STOPPED;

        if (!req->sampling_params.ignore_eos &&
            req->sampling_params.eos_token_id >= 0 &&
            tok == req->sampling_params.eos_token_id) {
            stop = true;
            fin_status = GC_REQ_FINISHED_STOPPED;
        } else if (req->num_output_tokens() >= req->max_tokens()) {
            stop = true;
            fin_status = GC_REQ_FINISHED_LENGTH_CAPPED;
        }

        if (stop) {
            finish_request(req, fin_status);
        }
    }
}

// ── Queries ───────────────────────────────────────────────────────────────────

int gc_scheduler_t::num_waiting() const {
    if (params_.policy == GC_SCHED_PRIORITY) return (int)waiting_prio_.size();
    return (int)waiting_fcfs_.size();
}

int gc_scheduler_t::num_running() const {
    return (int)running_.size();
}
