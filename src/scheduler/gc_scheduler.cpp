#include "gc_scheduler.h"
#include "gc_debug.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

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
    GC_LOG_REQ("admit req=%s waiting=%d", raw->request_id.c_str(), num_waiting());
    if (params_.enable_sched_trace) {
        GC_LOG_SCHED("admit req=%s waiting=%d", raw->request_id.c_str(), num_waiting());
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
        if (params_.policy == GC_SCHED_PRIORITY) {
            std::vector<gc_request_t *> tmp;
            while (!waiting_prio_.empty()) {
                gc_request_t * r = waiting_prio_.top(); waiting_prio_.pop();
                if (r != req) tmp.push_back(r);
            }
            for (auto * r : tmp) waiting_prio_.push(r);
        }
    }

    finished_reqs_[req_id] = GC_REQ_FINISHED_ABORTED;
    all_reqs_.erase(it);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

void gc_scheduler_t::preempt(gc_request_t * req) {
    assert(req != nullptr && "preempt: null request");
    assert(req->status == GC_REQ_RUNNING && "preempt: request not in RUNNING state");
    assert(gc_req_transition_valid(req->status, GC_REQ_PREEMPTED) &&
           "preempt: invalid state transition");
    kv_mgr_.free(req->request_id);
    req->status = GC_REQ_PREEMPTED;
    req->num_computed_tokens = 0;
    req->num_preemptions++;
    waiting_push_front(req);
    GC_LOG_REQ("preempt req=%s preemptions=%d", req->request_id.c_str(), req->num_preemptions);
    if (params_.enable_sched_trace) {
        GC_LOG_SCHED("preempt req=%s preemptions=%d", req->request_id.c_str(), req->num_preemptions);
    }
}

void gc_scheduler_t::finish_request(gc_request_t * req, gc_req_status_t status) {
    assert(req != nullptr && "finish_request: null request");
    assert(gc_req_is_finished(status) && "finish_request: target status is not a finished status");
    assert(gc_req_transition_valid(req->status, status) &&
           "finish_request: invalid state transition");
    req->status = status;
    kv_mgr_.free(req->request_id);
    running_.erase(std::remove(running_.begin(), running_.end(), req), running_.end());
    finished_reqs_[req->request_id] = status;
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
        const std::vector<int32_t> & /*new_block_ids_unused*/) const {
    gc_cached_req_entry_t e;
    e.req_id              = req->request_id;
    e.num_computed_tokens = req->num_computed_tokens;
    e.num_output_tokens   = req->num_output_tokens();
    e.num_prompt_tokens   = req->num_prompt_tokens();

    const gc_req_blocks_t * rb = kv_mgr_.get_blocks(req->request_id);
    if (rb) {
        auto ids = rb->block_ids();
        e.block_ids.assign(ids.begin(), ids.end());
    }

    if (!req->output_token_ids.empty()) {
        e.last_token = req->output_token_ids.back();
    } else if (!req->prompt_token_ids.empty()) {
        e.last_token = req->prompt_token_ids.back();
    }

    e.all_token_ids.reserve(req->prompt_token_ids.size() + req->output_token_ids.size());
    e.all_token_ids.insert(e.all_token_ids.end(),
                           req->prompt_token_ids.begin(), req->prompt_token_ids.end());
    e.all_token_ids.insert(e.all_token_ids.end(),
                           req->output_token_ids.begin(), req->output_token_ids.end());
    e.sampling_params = req->sampling_params;
    return e;
}

// How many new tokens to schedule for a running request this step.
// decode: always 1; prefill: min(remaining, chunk_cap, budget, model_len_limit)
int gc_scheduler_t::num_new_tokens_for(const gc_request_t * req, int token_budget) const {
    const bool is_decode = (req->num_computed_tokens >= req->num_prompt_tokens());
    if (is_decode) {
        // Decode: exactly 1 token, always.
        return (token_budget >= 1) ? 1 : 0;
    }
    // Prefill continuation: respect chunk cap.
    int remaining = req->num_prompt_tokens() - req->num_computed_tokens;
    if (remaining <= 0) remaining = 1;
    int n = std::min(remaining, token_budget);
    n = std::min(n, params_.max_prefill_chunk_tokens);
    n = std::min(n, params_.max_model_len - 1 - req->num_computed_tokens);
    return n;
}

// ── Preempt lowest-priority running request ───────────────────────────────────
// Returns the request that was preempted, or nullptr if nothing could be freed.
gc_request_t * gc_scheduler_t::preempt_one(
        gc_sched_output_t & out,
        std::vector<gc_request_t *> & scheduled_running,
        gc_request_t * protect) {

    if (running_.empty()) return nullptr;

    gc_request_t * victim;
    if (params_.policy == GC_SCHED_PRIORITY) {
        victim = *std::max_element(
            running_.begin(), running_.end(),
            [](const gc_request_t * a, const gc_request_t * b){ return *a < *b; });
    } else {
        victim = running_.back();
    }

    if (victim == protect) return nullptr;

    // Defensive: victim must be in RUNNING state when preempted.
    assert(victim->status == GC_REQ_RUNNING &&
           "preempt_one: victim is not in RUNNING state");
    assert(!gc_req_is_finished(victim->status) &&
           "preempt_one: attempt to preempt an already-finished request");

    // Un-schedule victim if already scheduled this step.
    auto ns_it = out.num_scheduled_tokens.find(victim->request_id);
    if (ns_it != out.num_scheduled_tokens.end()) {
        out.total_scheduled_tokens -= ns_it->second;
        out.num_scheduled_tokens.erase(ns_it);
        scheduled_running.erase(
            std::remove(scheduled_running.begin(), scheduled_running.end(), victim),
            scheduled_running.end());
    }

    running_.erase(std::remove(running_.begin(), running_.end(), victim), running_.end());
    out.preempted_req_ids.insert(victim->request_id);
    preempt(victim);
    return victim;
}

// ── Main schedule() ───────────────────────────────────────────────────────────
// Priority:
//   P1 – decode tokens (running requests past prefill phase)
//   P2 – prefill continuations (running requests still in prefill)
//   P3 – new admissions from waiting queue

gc_sched_output_t gc_scheduler_t::schedule() {
    gc_sched_output_t out;
    out.finished_reqs = std::move(finished_reqs_);
    finished_reqs_.clear();

    int token_budget = params_.max_num_tokens;

    std::vector<gc_request_t *> scheduled_running;

    // ── P1: Decode requests ───────────────────────────────────────────────────
    // Schedule all running requests that are in decode phase first.
    // Decode requests consume exactly 1 token each and must not be starved.
    for (gc_request_t * req : running_) {
        const bool is_decode = (req->num_computed_tokens >= req->num_prompt_tokens());
        if (!is_decode) continue;
        if (token_budget <= 0) break;

        while (true) {
            const gc_req_blocks_t * rb = kv_mgr_.allocate_slots(
                req->request_id,
                req->num_computed_tokens + 1,
                {},
                req->block_hashes,
                req->num_computed_tokens);

            if (rb != nullptr) break;

            gc_request_t * v = preempt_one(out, scheduled_running, req);
            if (v == nullptr) goto done_running;
        }

        scheduled_running.push_back(req);
        out.num_scheduled_tokens[req->request_id] = 1;
        out.total_scheduled_tokens += 1;
        out.metrics.num_decode_tokens += 1;
        token_budget -= 1;

        GC_LOG_SCHED("decode req=%s computed=%d budget_left=%d",
                     req->request_id.c_str(), req->num_computed_tokens, token_budget);
    }

    // ── P2: Prefill continuations ─────────────────────────────────────────────
    // Running requests still in prefill phase, capped by max_prefill_chunk_tokens.
    for (gc_request_t * req : running_) {
        const bool is_decode = (req->num_computed_tokens >= req->num_prompt_tokens());
        if (is_decode) continue;  // already handled above
        if (token_budget <= 0) break;

        int num_new = num_new_tokens_for(req, token_budget);
        if (num_new <= 0) continue;

        while (true) {
            const gc_req_blocks_t * rb = kv_mgr_.allocate_slots(
                req->request_id,
                req->num_computed_tokens + num_new,
                {},
                req->block_hashes,
                req->num_computed_tokens);

            if (rb != nullptr) break;

            gc_request_t * v = preempt_one(out, scheduled_running, req);
            if (v == nullptr) goto done_running;
        }

        scheduled_running.push_back(req);
        out.num_scheduled_tokens[req->request_id] = num_new;
        out.total_scheduled_tokens += num_new;
        out.metrics.num_prefill_tokens += num_new;
        token_budget -= num_new;

        GC_LOG_SCHED("prefill-cont req=%s computed=%d chunk=%d budget_left=%d",
                     req->request_id.c_str(), req->num_computed_tokens, num_new, token_budget);
    }

done_running:

    // ── P3: New admissions ────────────────────────────────────────────────────
    // Only admit new requests if no preemption happened this step.
    if (!out.preempted_req_ids.empty()) goto build_output;

    while (!waiting_empty()
        && (int)running_.size() < params_.max_num_running
        && token_budget > 0) {

        gc_request_t * req = waiting_front();

        auto hit = kv_mgr_.find_cache_hit(req->block_hashes);
        int num_computed = hit.num_tokens;

        int num_new = req->num_tokens() - num_computed;
        num_new = std::min(num_new, token_budget);
        num_new = std::min(num_new, params_.max_prefill_chunk_tokens);
        num_new = std::min(num_new, params_.max_model_len - num_computed);
        if (num_new <= 0) {
            break;
        }

        const gc_req_blocks_t * rb = kv_mgr_.allocate_slots(
            req->request_id,
            num_computed + num_new,
            hit.blocks,
            req->block_hashes,
            num_computed);

        if (rb == nullptr) break;

        waiting_pop();
        assert(gc_req_transition_valid(req->status, GC_REQ_RUNNING) &&
               "schedule: invalid WAITING→RUNNING transition");
        req->status = GC_REQ_RUNNING;
        req->num_computed_tokens = num_computed;
        running_.push_back(req);

        bool is_new = (req->num_preemptions == 0 && req->num_output_tokens() == 0);
        if (is_new) {
            out.new_reqs.push_back(make_new_req_data(req));
        } else {
            out.cached_reqs.push_back(make_cached_entry(req, {}));
        }

        out.num_scheduled_tokens[req->request_id] = num_new;
        out.total_scheduled_tokens += num_new;
        out.metrics.num_prefill_tokens += num_new;
        out.metrics.num_new_admitted += 1;
        token_budget -= num_new;

        GC_LOG_SCHED("admit-run req=%s prefix_hit=%d chunk=%d budget_left=%d",
                     req->request_id.c_str(), num_computed, num_new, token_budget);
    }

build_output:
    // Fill cached_reqs for running requests scheduled in P1/P2 above.
    for (gc_request_t * req : scheduled_running) {
        out.cached_reqs.push_back(make_cached_entry(req, {}));
    }

    // Populate metrics snapshot.
    out.metrics.num_running   = (int)running_.size();
    out.metrics.num_waiting   = num_waiting();
    out.metrics.num_preempted = (int)out.preempted_req_ids.size();
    out.metrics.kv_usage      = kv_mgr_.usage();

    GC_LOG_SCHED("step: running=%d waiting=%d decode_tok=%d prefill_tok=%d "
                 "new_admitted=%d preempted=%d kv=%.2f%%",
                 out.metrics.num_running, out.metrics.num_waiting,
                 out.metrics.num_decode_tokens, out.metrics.num_prefill_tokens,
                 out.metrics.num_new_admitted, out.metrics.num_preempted,
                 out.metrics.kv_usage * 100.f);

    // NOTE: num_computed_tokens is NOT advanced here.
    // It is advanced in update_computed_tokens(), called by the engine AFTER
    // confirmed model execution. This prevents state corruption on exec failure.

    return out;
}

// ── update_computed_tokens() ──────────────────────────────────────────────────
// Called by engine AFTER model execution confirms all scheduled tokens processed.

void gc_scheduler_t::update_computed_tokens(
        const std::unordered_map<std::string, int> & scheduled_tokens) {
    for (auto & [req_id, n] : scheduled_tokens) {
        auto it = all_reqs_.find(req_id);
        if (it != all_reqs_.end()) {
            it->second->num_computed_tokens += n;
        }
    }
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
        kv_mgr_.cache_blocks(req_id, req->block_hashes, req->num_computed_tokens);

        bool stop = false;
        gc_req_status_t fin_status = GC_REQ_FINISHED_STOPPED;

        // EOS-by-id
        if (!stop &&
            !req->sampling_params.ignore_eos &&
            req->sampling_params.eos_token_id >= 0 &&
            tok == req->sampling_params.eos_token_id) {
            stop = true;
        }

        // Multi-token stop sequences: check if the tail of output_token_ids
        // matches any configured stop sequence.
        if (!stop && !req->sampling_params.stop_token_ids.empty()) {
            const auto & out_toks = req->output_token_ids;
            for (const auto & seq : req->sampling_params.stop_token_ids) {
                if (seq.empty()) continue;
                const int slen = (int)seq.size();
                const int olen = (int)out_toks.size();
                if (olen < slen) continue;
                bool match = true;
                for (int si = 0; si < slen; ++si) {
                    if (out_toks[(size_t)(olen - slen + si)] != seq[(size_t)si]) {
                        match = false;
                        break;
                    }
                }
                if (match) { stop = true; break; }
            }
        }

        // Max-tokens budget
        if (!stop && req->num_output_tokens() >= req->max_tokens()) {
            stop = true;
            fin_status = GC_REQ_FINISHED_LENGTH_CAPPED;
        }

        if (stop) {
            GC_LOG_REQ("finish req=%s out_tokens=%d reason=%d",
                       req_id.c_str(), req->num_output_tokens(), (int)fin_status);
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
