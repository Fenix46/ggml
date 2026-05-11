#include "gc_mmap.h"
#include "gc_common.h"

#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <stdexcept>

#ifdef __has_include
#  if __has_include(<unistd.h>)
#    include <unistd.h>
#    include <fcntl.h>
#    include <sys/stat.h>
#    ifdef _POSIX_MAPPED_FILES
#      include <sys/mman.h>
#    endif
#    ifdef _POSIX_MEMLOCK_RANGE
#      include <sys/resource.h>
#    endif
#  endif
#endif

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  ifndef PATH_MAX
#    define PATH_MAX MAX_PATH
#  endif
#  include <io.h>
#endif

#ifdef _WIN32
#  define gc__ftell _ftelli64
#  define gc__fseek _fseeki64
#else
#  define gc__ftell ftello
#  define gc__fseek fseeko
#endif

// ── gc_file_t::impl ───────────────────────────────────────────────────────────

struct gc_file_t::impl {
#ifdef _WIN32
    HANDLE fp_win32 = INVALID_HANDLE_VALUE;
    bool   owns_fp  = true;
    FILE * fp       = nullptr;
    size_t sz       = 0;

    static std::string win32_errmsg(DWORD err) {
        LPSTR buf = nullptr;
        DWORD len = FormatMessageA(
            FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, nullptr);
        if (!len) { return gc__format("Win32 error 0x%lx", err); }
        std::string s(buf, len);
        LocalFree(buf);
        return s;
    }

    impl(const char * path, const char * mode) {
        fp = ggml_fopen(path, mode);
        if (!fp) { throw std::runtime_error(gc__format("failed to open %s: %s", path, strerror(errno))); }
        fp_win32 = (HANDLE)_get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        sz = tell();
        seek(0, SEEK_SET);
    }

    explicit impl(FILE * f) : owns_fp(false), fp(f) {
        fp_win32 = (HANDLE)_get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        sz = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        LARGE_INTEGER li{}; li.QuadPart = 0;
        if (!SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT))
            throw std::runtime_error(gc__format("tell error: %s", win32_errmsg(GetLastError()).c_str()));
        return (size_t)li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        static_assert(SEEK_SET == FILE_BEGIN);
        static_assert(SEEK_CUR == FILE_CURRENT);
        static_assert(SEEK_END == FILE_END);
        LARGE_INTEGER li{}; li.QuadPart = (LONGLONG)offset;
        if (!SetFilePointerEx(fp_win32, li, nullptr, whence))
            throw std::runtime_error(gc__format("seek error: %s", win32_errmsg(GetLastError()).c_str()));
    }

    void read_raw(void * dst, size_t len) {
        size_t done = 0;
        while (done < len) {
            size_t chunk = std::min<size_t>(len - done, 64 * 1024 * 1024);
            DWORD  got   = 0;
            if (!ReadFile(fp_win32, (char *)dst + done, (DWORD)chunk, &got, nullptr))
                throw std::runtime_error(gc__format("read error: %s", win32_errmsg(GetLastError()).c_str()));
            if (got == 0) throw std::runtime_error("unexpected EOF");
            done += got;
        }
    }

    uint32_t read_u32() { uint32_t v; read_raw(&v, sizeof(v)); return v; }

    void write_raw(const void * src, size_t len) const {
        size_t done = 0;
        while (done < len) {
            size_t chunk = std::min<size_t>(len - done, 64 * 1024 * 1024);
            DWORD  wrote = 0;
            if (!WriteFile(fp_win32, (const char *)src + done, (DWORD)chunk, &wrote, nullptr))
                throw std::runtime_error(gc__format("write error: %s", win32_errmsg(GetLastError()).c_str()));
            if (wrote == 0) throw std::runtime_error("write failed");
            done += wrote;
        }
    }

    void write_u32(uint32_t v) const { write_raw(&v, sizeof(v)); }

    int file_id() const { return _fileno(fp); }
    size_t file_size() const { return sz; }

    ~impl() { if (fp && owns_fp) std::fclose(fp); }

#else
    FILE * fp      = nullptr;
    bool   owns_fp = true;
    size_t sz      = 0;
    std::string fname;

    impl(const char * path, const char * mode) : fname(path) {
        fp = ggml_fopen(path, mode);
        if (!fp) { throw std::runtime_error(gc__format("failed to open %s: %s", path, strerror(errno))); }
        seek(0, SEEK_END);
        sz = tell();
        seek(0, SEEK_SET);
    }

    explicit impl(FILE * f) : fp(f), owns_fp(false), fname("(FILE*)") {
        seek(0, SEEK_END);
        sz = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        off_t r = gc__ftell(fp);
        if (r == -1) throw std::runtime_error(gc__format("ftell: %s", strerror(errno)));
        return (size_t)r;
    }

    void seek(size_t offset, int whence) const {
        if (gc__fseek(fp, (off_t)offset, whence) != 0)
            throw std::runtime_error(gc__format("fseek: %s", strerror(errno)));
    }

    void read_raw(void * dst, size_t len) {
        if (!len) return;
        errno = 0;
        if (std::fread(dst, len, 1, fp) != 1) {
            if (ferror(fp)) throw std::runtime_error(gc__format("read error: %s", strerror(errno)));
            throw std::runtime_error("unexpected EOF");
        }
    }

    uint32_t read_u32() { uint32_t v; read_raw(&v, sizeof(v)); return v; }

    void write_raw(const void * src, size_t len) const {
        if (!len) return;
        errno = 0;
        if (std::fwrite(src, len, 1, fp) != 1)
            throw std::runtime_error(gc__format("write error: %s", strerror(errno)));
    }

    void write_u32(uint32_t v) const { write_raw(&v, sizeof(v)); }

    int file_id() const {
#if defined(fileno)
        return fileno(fp);
#else
        return ::fileno(fp);
#endif
    }

    size_t file_size() const { return sz; }

    ~impl() { if (fp && owns_fp) std::fclose(fp); }
#endif
};

gc_file_t::gc_file_t(const char * path, const char * mode) : pimpl(std::make_unique<impl>(path, mode)) {}
gc_file_t::gc_file_t(FILE * fp)                             : pimpl(std::make_unique<impl>(fp)) {}
gc_file_t::~gc_file_t() = default;

size_t   gc_file_t::tell()    const { return pimpl->tell(); }
size_t   gc_file_t::size()    const { return pimpl->file_size(); }
int      gc_file_t::file_id() const { return pimpl->file_id(); }

void     gc_file_t::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void     gc_file_t::read_raw(void * dst, size_t len)      { pimpl->read_raw(dst, len); }
uint32_t gc_file_t::read_u32()                            { return pimpl->read_u32(); }
void     gc_file_t::write_raw(const void * src, size_t len) const { pimpl->write_raw(src, len); }
void     gc_file_t::write_u32(uint32_t v) const           { pimpl->write_u32(v); }

// ── gc_mmap_t::impl ───────────────────────────────────────────────────────────

struct gc_mmap_t::impl {
#ifdef _POSIX_MAPPED_FILES
    void * addr_ = MAP_FAILED;
    size_t size_ = 0;
    std::vector<std::pair<size_t, size_t>> fragments;

    impl(gc_file_t * file, size_t prefetch) {
        size_ = file->size();
        int fd = file->file_id();
        addr_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
        if (addr_ == MAP_FAILED) throw std::runtime_error(gc__format("mmap failed: %s", strerror(errno)));
        if (prefetch > 0) {
            if (posix_madvise(addr_, std::min(size_, prefetch), POSIX_MADV_WILLNEED))
                GC_LOG_WARN("posix_madvise WILLNEED failed: %s\n", strerror(errno));
        }
        fragments.emplace_back(0, size_);
    }

    static void align_range(size_t * first, size_t * last, size_t page) {
        size_t off = *first & (page - 1);
        *first += off ? page - off : 0;
        *last  &= ~(page - 1);
        if (*last <= *first) *last = *first;
    }

    void unmap_fragment(size_t first, size_t last) {
        size_t page = (size_t)sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page);
        size_t len = last - first;
        if (!len) return;
        munmap((char *)addr_ + first, len);

        std::vector<std::pair<size_t, size_t>> updated;
        for (auto & f : fragments) {
            if (f.first < first && f.second > last) {
                updated.emplace_back(f.first, first);
                updated.emplace_back(last, f.second);
            } else if (f.first < first && f.second > first) {
                updated.emplace_back(f.first, first);
            } else if (f.first < last && f.second > last) {
                updated.emplace_back(last, f.second);
            } else if (f.first >= first && f.second <= last) {
                // consumed — drop
            } else {
                updated.push_back(f);
            }
        }
        fragments = std::move(updated);
    }

    void * addr() const { return addr_; }
    size_t size() const { return size_; }

    ~impl() {
        for (auto & f : fragments)
            munmap((char *)addr_ + f.first, f.second - f.first);
    }

#elif defined(_WIN32)
    HANDLE  hmap  = nullptr;
    void *  addr_ = nullptr;
    size_t  size_ = 0;

    impl(gc_file_t * file, size_t prefetch) {
        size_ = file->size();
        HANDLE hf = (HANDLE)_get_osfhandle(file->file_id());
        hmap = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hmap) throw std::runtime_error(gc__format("CreateFileMapping failed"));
        addr_ = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
        if (!addr_) { CloseHandle(hmap); throw std::runtime_error(gc__format("MapViewOfFile failed")); }
        (void)prefetch;
    }

    void unmap_fragment(size_t, size_t) {}

    void * addr() const { return addr_; }
    size_t size() const { return size_; }

    ~impl() {
        if (addr_) UnmapViewOfFile(addr_);
        if (hmap)  CloseHandle(hmap);
    }
#else
    impl(gc_file_t *, size_t) { throw std::runtime_error("mmap not supported on this platform"); }
    void unmap_fragment(size_t, size_t) { throw std::runtime_error("mmap not supported"); }
    void * addr() const { return nullptr; }
    size_t size() const { return 0; }
#endif
};

gc_mmap_t::gc_mmap_t(gc_file_t * file, size_t prefetch) : pimpl(std::make_unique<impl>(file, prefetch)) {}
gc_mmap_t::~gc_mmap_t() = default;

size_t gc_mmap_t::size() const { return pimpl->size(); }
void * gc_mmap_t::addr() const { return pimpl->addr(); }
void   gc_mmap_t::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

#if defined(_POSIX_MAPPED_FILES) || defined(_WIN32)
const bool gc_mmap_t::SUPPORTED = true;
#else
const bool gc_mmap_t::SUPPORTED = false;
#endif

// ── gc_mlock_t::impl ──────────────────────────────────────────────────────────

struct gc_mlock_t::impl {
    void * addr_  = nullptr;
    size_t size_  = 0;
    bool   failed = false;

#if defined(_POSIX_MEMLOCK_RANGE)
    static size_t granularity() { return (size_t)sysconf(_SC_PAGESIZE); }

    bool raw_lock(const void * a, size_t n) const {
        if (!mlock(a, n)) return true;
        GC_LOG_WARN("mlock failed for %zu bytes: %s\n", n, strerror(errno));
        return false;
    }
    static void raw_unlock(void * a, size_t n) { munlock(a, n); }

#elif defined(_WIN32)
    static size_t granularity() {
        SYSTEM_INFO si; GetSystemInfo(&si); return (size_t)si.dwPageSize;
    }
    bool raw_lock(void * a, size_t n) const {
        for (int t = 1; ; ++t) {
            if (VirtualLock(a, n)) return true;
            if (t == 2) { GC_LOG_WARN("VirtualLock failed\n"); return false; }
            SIZE_T mn, mx;
            GetProcessWorkingSetSize(GetCurrentProcess(), &mn, &mx);
            mn += n + 1048576; mx += n + 1048576;
            SetProcessWorkingSetSize(GetCurrentProcess(), mn, mx);
        }
    }
    static void raw_unlock(void * a, size_t n) { VirtualUnlock(a, n); }
#else
    static size_t granularity() { return 65536; }
    bool raw_lock(const void *, size_t) const { GC_LOG_WARN("mlock not supported\n"); return false; }
    static void raw_unlock(const void *, size_t) {}
#endif

    void init(void * ptr) { addr_ = ptr; }

    void grow_to(size_t target) {
        if (!addr_ || failed) return;
        size_t g = granularity();
        target = (target + g - 1) & ~(g - 1);
        if (target > size_) {
            if (raw_lock((char *)addr_ + size_, target - size_)) {
                size_ = target;
            } else {
                failed = true;
            }
        }
    }
};

gc_mlock_t::gc_mlock_t()  : pimpl(std::make_unique<impl>()) {}
gc_mlock_t::~gc_mlock_t() = default;

void gc_mlock_t::init(void * ptr)             { pimpl->init(ptr); }
void gc_mlock_t::grow_to(size_t target_size)  { pimpl->grow_to(target_size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool gc_mlock_t::SUPPORTED = true;
#else
const bool gc_mlock_t::SUPPORTED = false;
#endif

size_t gc_path_max() { return PATH_MAX; }
