#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

struct gc_server_params_t {
    std::string host = "127.0.0.1";
    int         port = 8080;
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

