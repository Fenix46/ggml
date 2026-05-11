#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class gc_engine_t;

struct gc_server_request_t {
    std::string              id;
    std::string              model;
    std::string              prompt_text;
    int                      max_tokens = 16;
    bool                     stream = false;
};

struct gc_server_token_event_t {
    std::string req_id;
    int32_t     token = -1;
    bool        finished = false;
};

class gc_server_runtime_t {
public:
    virtual ~gc_server_runtime_t() = default;
    virtual std::string submit(const gc_server_request_t & req) = 0;
    virtual bool cancel(const std::string & req_id) = 0;
    virtual std::vector<gc_server_token_event_t> step() = 0;
};

using gc_server_tokenize_fn_t = std::function<std::vector<int32_t>(const std::string &)>;

class gc_engine_server_runtime_t final : public gc_server_runtime_t {
public:
    gc_engine_server_runtime_t(
        gc_engine_t * engine,
        gc_server_tokenize_fn_t tokenize_fn);

    std::string submit(const gc_server_request_t & req) override;
    bool cancel(const std::string & req_id) override;
    std::vector<gc_server_token_event_t> step() override;

private:
    gc_engine_t * engine_ = nullptr; // non-owning
    gc_server_tokenize_fn_t tokenize_fn_;
    std::mutex mu_;
};

struct gc_server_params_t {
    std::string host = "127.0.0.1";
    int         port = 8080;
    gc_server_runtime_t * runtime = nullptr; // non-owning; optional
};

class gc_server_t {
public:
    explicit gc_server_t(const gc_server_params_t & params = {});
    ~gc_server_t();

    gc_server_t(const gc_server_t &) = delete;
    gc_server_t & operator=(const gc_server_t &) = delete;

    bool start_async();
    void stop();
    bool is_running() const { return running_.load(); }

private:
    gc_server_params_t params_;
    std::atomic<bool>  running_{false};
    std::thread        server_thread_;

    struct impl_t;
    std::unique_ptr<impl_t> impl_;
};
