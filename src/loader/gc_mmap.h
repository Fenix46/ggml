#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

// ── File I/O ──────────────────────────────────────────────────────────────────

struct gc_file_t {
    gc_file_t(const char * path, const char * mode);
    explicit gc_file_t(FILE * fp);
    ~gc_file_t();

    // non-copyable
    gc_file_t(const gc_file_t &) = delete;
    gc_file_t & operator=(const gc_file_t &) = delete;

    size_t   tell()    const;
    size_t   size()    const;
    int      file_id() const;

    void seek(size_t offset, int whence) const;

    void     read_raw(void * dst, size_t len);
    uint32_t read_u32();

    void write_raw(const void * src, size_t len) const;
    void write_u32(uint32_t val) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using gc_files_t = std::vector<std::unique_ptr<gc_file_t>>;

// ── Memory-mapped file ────────────────────────────────────────────────────────

struct gc_mmap_t {
    gc_mmap_t(gc_file_t * file, size_t prefetch = (size_t)-1);
    ~gc_mmap_t();

    gc_mmap_t(const gc_mmap_t &) = delete;
    gc_mmap_t & operator=(const gc_mmap_t &) = delete;

    size_t  size() const;
    void *  addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using gc_mmaps_t = std::vector<std::unique_ptr<gc_mmap_t>>;

// ── Memory-locked region ──────────────────────────────────────────────────────

struct gc_mlock_t {
    gc_mlock_t();
    ~gc_mlock_t();

    gc_mlock_t(const gc_mlock_t &) = delete;
    gc_mlock_t & operator=(const gc_mlock_t &) = delete;

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using gc_mlocks_t = std::vector<std::unique_ptr<gc_mlock_t>>;

size_t gc_path_max();
