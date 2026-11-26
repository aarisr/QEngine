#pragma once
// Shared PREFIX objects: a pinned system/tool prefix (Pin) published as one shared-memory object in a directory on a
// host tmpfs mount, mapped by every engine on the host and (when it can) registered with CUDA, so a restore DMAs the
// shards straight from the shared pages instead of prefilling the prefix. Off when QE_SHARED_L2_DIR is unset: nothing
// here runs by default.
//
// Object (one file per prefix, name = content key over fingerprint + steering id + exact tokens):
//   header: magic, version, body length, body crc32c; body: fingerprint, steering id, exact tokens, per-card KV shard
//   (offset, length, crc32c) and state shard (offset, length, crc32c), writer id; payload starts page-aligned, every
//   shard 64-byte aligned, so each rank's DMA starts on its own boundary.
//   A reader opens, fstats, mmaps MAP_SHARED, verifies the header and every shard's crc32c BEFORE first use, then reads
//   the mapping directly. An unlinked object stays valid while mapped (Linux), so eviction never corrupts a restore.
// Multi-writer: each writer lays out a private name (O_EXCL <key>.tmp.<pid>.<seq>), fills it, then link()s it to the
// content key; an existing identical key means done. Publishing takes no lock. Readers never delete. Only the flock'd
// Evictor unlinks shared objects or stale writer temps.
// Eviction: a writer prunes its own objects against its quota; one flock'd evictor role per directory enforces the
// global cap. This header has no CUDA types: the caller installs the registration hooks (flash_capi.cpp), so tests run
// without a card and a failed registration falls back to a pageable copy that the caller logs.
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/inotify.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "shared_prefix_crc.h"  // SSE4.2 when available, table fallback; both standard CRC-32C

namespace sp {

constexpr char kMagic[8] = {'Q', 'P', 'I', 'N', '0', '0', '0', '2'};
constexpr uint32_t kVersion = 2;
constexpr char kExt[] = ".qpin";
constexpr char kTmpTag[] = ".tmp.";
constexpr size_t kMaxBody = 1u << 22;    // fingerprint + steering + 16K tokens + a few cards
constexpr size_t kMaxTokens = 1u << 20;  // sanity bound for a header
constexpr int kMaxCards = 64;
constexpr double kTempAgeS = 3600.0;     // a private name older than this was left by a killed writer
constexpr size_t kAlignShard = 64;

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
inline uint64_t hash64(const void* d, size_t n) { const uint8_t* p = (const uint8_t*)d; uint64_t h = 0xcbf29ce484222325ull; for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ull; } return h; }
inline size_t page_size() {
    static const size_t ps = (size_t)sysconf(_SC_PAGESIZE);
    return ps;
}
inline std::string key(const std::string& fp, const std::string& steer, const std::vector<int>& toks) {
    uint64_t h[3];
    h[0] = hash64(fp.data(), fp.size());
    h[1] = hash64(steer.data(), steer.size());
    h[2] = hash64(toks.data(), toks.size() * sizeof(int));
    char b[40];
    snprintf(b, sizeof b, "%016llx%016llx", (unsigned long long)h[0], (unsigned long long)hash64(h, sizeof h));
    return b;
}
inline std::string path_of(const std::string& dir, const std::string& k) { return dir + "/" + k + kExt; }

// ---- registration hooks (CUDA in the caller)
using RegFn = bool (*)(void* p, size_t n, void* ctx, int* err);
using UnregFn = void (*)(void* p, size_t n, void* ctx);
struct Hooks {
    RegFn reg = nullptr;
    UnregFn unreg = nullptr;
    void* ctx = nullptr;
    size_t budget = 0;      // bytes this process may pin; 0: unlimited
    size_t pinned = 0;
    // Retry a refused read-only registration on a writable view of the same pages,
    // then drop CPU write access again (cards without HostRegisterReadOnly, e.g. the
    // RTX 3090s, refuse a PROT_READ mapping). Off: QE_SHARED_L2_RW_PIN=0.
    bool rw_pin = true;
    std::atomic<long long> n_reg{0}, n_reg_rw{0}, n_reg_fail{0}, n_fallback{0};
};
inline Hooks& hooks() {
    static Hooks h;
    return h;
}
inline void set_hooks(RegFn r, UnregFn u, void* ctx, size_t budget = 0) {
    hooks().reg = r;
    hooks().unreg = u;
    hooks().ctx = ctx;
    hooks().budget = budget;
}

struct Shard {
    uint64_t off = 0, len = 0;
    uint32_t crc = 0;
};
struct Desc {
    std::string fp, steer;
    std::vector<int> toks;
    uint64_t writer = 0;
    int ng = 0;
    std::vector<Shard> kv, st;
    uint64_t payload_off = 0, total = 0;
};

inline std::vector<char> encode_header(const Desc& d, uint32_t* body_crc = nullptr) {
    std::vector<char> b;
    auto put = [&](const void* p, size_t n) { b.insert(b.end(), (const char*)p, (const char*)p + n); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    u32((uint32_t)d.fp.size());
    put(d.fp.data(), d.fp.size());
    u32((uint32_t)d.steer.size());
    put(d.steer.data(), d.steer.size());
    u32((uint32_t)d.toks.size());
    u32((uint32_t)d.ng);
    u64(d.writer);
    for (int r = 0; r < d.ng; r++) { u64(d.kv[r].off); u64(d.kv[r].len); u32(d.kv[r].crc); }
    for (int r = 0; r < d.ng; r++) { u64(d.st[r].off); u64(d.st[r].len); u32(d.st[r].crc); }
    put(d.toks.data(), d.toks.size() * 4);
    if (body_crc) *body_crc = crc32c(b.data(), b.size());
    return b;
}
// parse a header; the payload offsets are validated against `file_size` and the alignment rules
inline bool decode_header(const char* p, size_t n, uint64_t file_size, Desc& d, const char** why) {
    auto bad = [&](const char* w) { if (why) *why = w; return false; };
    if (n < 12 + 4 || memcmp(p, kMagic, 8) != 0) return bad("magic");
    uint32_t ver = 0, blen = 0, bcrc = 0;
    memcpy(&ver, p + 8, 4);
    memcpy(&blen, p + 12, 4);
    if (ver != kVersion) return bad("version");
    if (blen > kMaxBody || 20 + (size_t)blen > n) return bad("body length");
    memcpy(&bcrc, p + 16, 4);
    if (crc32c(p + 20, blen) != bcrc) return bad("header checksum");
    const char* q = p + 20;
    const char* qe = q + blen;
    auto get = [&](void* dst, size_t k) { if (q + k > qe) return false; memcpy(dst, q, k); q += k; return true; };
    uint32_t fl = 0, sl = 0;
    if (!get(&fl, 4) || fl > (size_t)(qe - q)) return bad("fingerprint");
    d.fp.assign(q, fl);
    q += fl;
    if (!get(&sl, 4) || sl > (size_t)(qe - q)) return bad("steering");
    d.steer.assign(q, sl);
    q += sl;
    uint32_t nt = 0, ng = 0;
    if (!get(&nt, 4) || !get(&ng, 4) || nt == 0 || nt > kMaxTokens || ng <= 0 || ng > kMaxCards) return bad("counts");
    uint64_t wr = 0;
    if (!get(&wr, 8)) return bad("writer");
    d.writer = wr;
    d.ng = (int)ng;
    d.kv.assign(ng, Shard{});
    d.st.assign(ng, Shard{});
    for (auto& s : d.kv)
        if (!get(&s.off, 8) || !get(&s.len, 8) || !get(&s.crc, 4)) return bad("kv shard");
    for (auto& s : d.st)
        if (!get(&s.off, 8) || !get(&s.len, 8) || !get(&s.crc, 4)) return bad("state shard");
    d.toks.resize(nt);
    if (!get(d.toks.data(), (size_t)nt * 4) || q != qe) return bad("tokens");
    d.payload_off = align_up(20 + (size_t)blen, page_size());
    d.total = file_size;
    if (file_size < d.payload_off) return bad("short file");
    for (auto* v : {&d.kv, &d.st})
        for (auto& s : *v) {
            if (s.len == 0 || s.off % kAlignShard != 0 || s.off < d.payload_off || s.off > file_size || s.len > file_size - s.off)
                return bad("shard layout");
        }
    return true;
}

// ---- an attached object: mmap, verified header, shards, optional registration
enum class Rd { OK, MISS, OTHER, BAD };
class Map {
public:
    Map() = default;
    Map(const Map&) = delete;
    Map& operator=(const Map&) = delete;
    ~Map() {
        if (base_ && registered_) {
            if (hooks().unreg) hooks().unreg(base_, size_, hooks().ctx);
            std::lock_guard<std::mutex> l(hooks_mutex());
            hooks().pinned -= std::min(hooks().pinned, size_);
        }
        if (base_) munmap(base_, size_);
        if (fd_ >= 0) close(fd_);
    }
    bool ok() const { return base_ != nullptr; }
    const Desc& desc() const { return d_; }
    size_t size() const { return size_; }
    const char* kv(int r) const { return base_ + d_.kv[r].off; }
    size_t kv_len(int r) const { return (size_t)d_.kv[r].len; }
    const char* st(int r) const { return base_ + d_.st[r].off; }
    size_t st_len(int r) const { return (size_t)d_.st[r].len; }
    // verify one shard's crc before its first use; false: damaged (the caller treats it as a miss)
    bool verify_kv(int r) const { return d_.kv[r].crc == crc32c(kv(r), kv_len(r)); }
    bool verify_st(int r) const { return d_.st[r].crc == crc32c(st(r), st_len(r)); }
#include "shared_prefix_verify.inc"
    bool registered() const { return registered_; }
    // register the whole mapping (once); false: the caller logs the fallback to pageable copies, never fatal
    bool attach() {
        if (!base_ || registered_) return registered_;
        auto& h = hooks();
        if (!h.reg) return registered_ = false;
        std::lock_guard<std::mutex> l(hooks_mutex());
        if (h.budget && h.pinned + size_ > h.budget) {
            h.n_fallback++;
            return registered_ = false;
        }
        int err = 0;
        bool ok = h.reg(base_, size_, h.ctx, &err);
        if (!ok && h.rw_pin && rw_fd_) {
            ok = remap(PROT_READ | PROT_WRITE) && h.reg(base_, size_, h.ctx, &err);
            // the pin survives mprotect; DMA keeps reading, stray CPU writes fault
            if (ok && mprotect(base_, size_, PROT_READ) != 0) {
                if (h.unreg) h.unreg(base_, size_, h.ctx);
                ok = false;
            }
            if (ok) h.n_reg_rw++;
            else if (!remap(PROT_READ)) {
                munmap(base_, size_);  // a hole is safer than a writable view
                base_ = nullptr;
            }
        }
        if (!ok) {
            h.n_reg_fail++;
            h.n_fallback++;
            return registered_ = false;
        }
        h.pinned += size_;
        h.n_reg++;
        return registered_ = true;
    }
    static std::mutex& hooks_mutex() {
        static std::mutex m;
        return m;
    }
    bool advise(int advice) const {
        if (advice == 0) return true;
        return madvise(base_, size_, advice) == 0;
    }

private:
    friend Rd open_map(const std::string&, const std::string&, Map&, const char**);
    // replace the view in place (same address, same pages)
    bool remap(int prot) {
        void* p = mmap(base_, size_, prot, MAP_SHARED | MAP_FIXED, fd_, 0);
        return p == (void*)base_;
    }
    int fd_ = -1;
    bool rw_fd_ = false;
    char* base_ = nullptr;
    size_t size_ = 0;
    bool registered_ = false;
    Desc d_;
};

inline Rd open_map(const std::string& path, const std::string& fp, Map& out, const char** why = nullptr) {
    // O_RDWR only so attach() can register a writable view; the mapping stays PROT_READ
    bool rw = true;
    int fd = open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0 && (errno == EACCES || errno == EROFS || errno == EPERM)) {
        rw = false;
        fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    }
    if (fd < 0) return Rd::MISS;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 20) {
        close(fd);
        return Rd::MISS;
    }
    void* p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        close(fd);
        return Rd::MISS;
    }
    Desc d;
    const char* w = "";
    if (!decode_header((const char*)p, (size_t)st.st_size, (uint64_t)st.st_size, d, &w)) {
        munmap(p, (size_t)st.st_size);
        close(fd);
        if (why) *why = w;
        return Rd::BAD;
    }
    if (d.fp != fp) {
        munmap(p, (size_t)st.st_size);
        close(fd);
        return Rd::OTHER;
    }
    out.fd_ = fd;
    out.rw_fd_ = rw;
    out.base_ = (char*)p;
    out.size_ = (size_t)st.st_size;
    out.d_ = std::move(d);
    return Rd::OK;
}

// ---- a private layout being filled (the writer side); one per writer, never visible before seal/publish
class Builder {
public:
    Builder() = default;
    Builder(const Builder&) = delete;
    ~Builder() { discard(); }
    bool advise(int advice) const {
        if (advice == 0) return true;
        return madvise(base_, size_, advice) == 0;
    }
    // sizes per card of the KV shards and the state shards; the header fields come from `desc`
    bool create(const std::string& dir, Desc desc, int& err) {
        d_ = std::move(desc);
        d_.ng = (int)d_.kv.size();
        if (d_.ng <= 0 || d_.st.size() != d_.kv.size()) return fail(EINVAL, err);
        size_t off = 0;
        for (int r = 0; r < d_.ng; r++) {
            d_.kv[r].off = align_up(off, kAlignShard);
            off = (size_t)(d_.kv[r].off + d_.kv[r].len);
            d_.st[r].off = align_up(off, kAlignShard);
            off = (size_t)(d_.st[r].off + d_.st[r].len);
        }
        body_ = encode_header(d_);
        if (body_.size() > kMaxBody) return fail(ENOMEM, err);
        const size_t hdr = align_up(20 + body_.size(), page_size());
        const size_t total = align_up(hdr + off, page_size());
        // the shard offsets in the header are relative; add the page-aligned payload base
        d_.payload_off = hdr;
        d_.total = total;
        for (auto* v : {&d_.kv, &d_.st})
            for (auto& s : *v) s.off += hdr;
        body_ = encode_header(d_);
        temp_ = dir + "/." + key(d_.fp,d_.steer,d_.toks) + ".tmp." + std::to_string(getpid()) + "." + std::to_string(++seq_);
        fd_ = open(temp_.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (fd_ < 0) return fail(errno, err);
        if (ftruncate(fd_, (off_t)total) != 0) return fail(errno, err);
        void* p = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) return fail(errno, err);
        base_ = (char*)p;
        size_ = total;
        return true;
    }
    char* kv(int r) { return base_ + d_.kv[r].off; }
    char* st(int r) { return base_ + d_.st[r].off; }
    size_t kv_len(int r) const { return (size_t)d_.kv[r].len; }
    size_t st_len(int r) const { return (size_t)d_.st[r].len; }
    // checksum every shard, write the header, then hand the object to the key: link() (an existing key wins), rename()
    // as the fallback for a filesystem without hard links, then the private name is gone
    bool publish(const std::string& dir, const std::string& key, int& err) {
        if (!base_) return fail(EINVAL, err);
        if (key != sp::key(d_.fp, d_.steer, d_.toks)) return fail(EINVAL, err);
        for (int r = 0; r < d_.ng; r++) {
            d_.kv[r].crc = crc32c(kv(r), kv_len(r));
            d_.st[r].crc = crc32c(st(r), st_len(r));
        }
        body_ = encode_header(d_);
        const uint32_t crc = crc32c(body_.data(), body_.size());
        memcpy(base_, kMagic, 8);
        const uint32_t ver = kVersion, blen = (uint32_t)body_.size();
        memcpy(base_ + 8, &ver, 4);
        memcpy(base_ + 12, &blen, 4);
        memcpy(base_ + 16, &crc, 4);
        memcpy(base_ + 20, body_.data(), body_.size());
        const std::string final = dir + "/" + key + kExt;
        if (link(temp_.c_str(), final.c_str()) != 0) {
            const int e = errno;
            if (e == EEXIST) {  // another writer won: only trust its object if its key and every shard still verify
                Map existing;
                const Rd rd = open_map(final, d_.fp, existing);
                bool same = rd == Rd::OK && existing.desc().steer == d_.steer && existing.desc().toks == d_.toks &&
                            existing.desc().ng == d_.ng && existing.verify_all();
                for (int r = 0; same && r < d_.ng; r++)
                    same = existing.desc().kv[r].len == d_.kv[r].len && existing.desc().st[r].len == d_.st[r].len;
                if (!same) return fail(EIO, err);  // corrupt/colliding file: miss, never trust it
                detach();
                unlink(temp_.c_str());
                temp_.clear();
                return true;
            }
            // rename may replace an existing object, so only an unsupported link() allows it.
            if (e != EPERM && e != EOPNOTSUPP && e != ENOSYS) return fail(e, err);
            static std::once_flag unsupported_link_once;
            std::call_once(unsupported_link_once, [] { fprintf(stderr, "hard-link publication is unsupported and rename is being used\n"); });
            if (rename(temp_.c_str(), final.c_str()) != 0) return fail(errno, err);
            temp_.clear();
            detach();
            return true;
        }
        unlink(temp_.c_str());
        temp_.clear();
        detach();
        return true;
    }
    void detach() {
        if (base_) {
            munmap(base_, size_);
            base_ = nullptr;
            size_ = 0;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }
    void discard() {
        detach();
        if (!temp_.empty()) {
            unlink(temp_.c_str());
            temp_.clear();
        }
    }
    const Desc& desc() const { return d_; }

private:
    bool fail(int e, int& err) {
        err = e;
        discard();
        return false;
    }
    static std::atomic<uint64_t> seq_;
    Desc d_;
    std::vector<char> body_;
    std::string temp_;
    int fd_ = -1;
    char* base_ = nullptr;
    size_t size_ = 0;
};
inline std::atomic<uint64_t> Builder::seq_{0};

// ---- directory index: headers only, refreshed by inotify events and by a periodic rescan
struct Entry {
    std::string path, fp, steer;
    std::vector<int> toks;
    uint64_t writer = 0, bytes = 0;
    double mtime = 0;
};
class Index {
public:
    Index() = default;
    Index(const Index&) = delete;
    ~Index() { if (ifd_ >= 0) close(ifd_); }
    bool init(const std::string& dir) {
        dir_ = dir;
        ifd_ = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
        if (ifd_ >= 0 && inotify_add_watch(ifd_, dir.c_str(), IN_MOVED_TO | IN_MOVED_FROM | IN_CREATE | IN_DELETE | IN_CLOSE_WRITE) < 0) {
            close(ifd_);
            ifd_ = -1;
        }
        dirty_ = true;
        return ifd_ >= 0;
    }
    // drain pending events; any change (or a lost queue) forces a rescan
    void poke() {
        if (ifd_ < 0) return;
        char buf[4096] __attribute__((aligned(__alignof__(struct inotify_event))));
        for (;;) {
            const ssize_t n = read(ifd_, buf, sizeof buf);
            if (n > 0) {
                dirty_ = true;
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            if (n < 0 && errno == EAGAIN) return;
            if (n < 0) { dirty_ = true; return; }  // queue overflow or read error: rescan
        }
    }
    // current entries of this fingerprint; a rescan at most every max_age_s, or immediately when events said so
    const std::vector<Entry>& entries(const std::string& fp, double max_age_s = 2.0) {
        poke();
        const double t = now();
        if (dirty_ || t - last_ >= max_age_s) {
            rescan(fp);
            dirty_ = false;
            last_ = t;
        }
        return e_;
    }
    void rescan(const std::string& fp) {
        e_.clear();
        skipped_ = 0;
        DIR* d = opendir(dir_.c_str());
        if (!d) return;
        while (dirent* de = readdir(d)) {
            const std::string nm = de->d_name, path = dir_ + "/" + nm;
            if (nm.find(kTmpTag) != std::string::npos) continue;  // only the flock'd evictor may remove stale writer temps
            const size_t e = strlen(kExt);
            if (nm.size() <= e || nm.compare(nm.size() - e, e, kExt) != 0) continue;
            if (e_.size() >= kMaxEntries) continue;
            Map m;
            const char* why = "";
            const Rd r = open_map(path, fp, m, &why);
            if (r != Rd::OK) { skipped_++; continue; }
            struct stat st;
            if (stat(path.c_str(), &st) != 0) continue;
            Entry en;
            en.path = path;
            en.fp = m.desc().fp;
            en.steer = m.desc().steer;
            en.toks = m.desc().toks;
            en.writer = m.desc().writer;
            en.bytes = (uint64_t)st.st_size;
            en.mtime = (double)st.st_mtime;
            e_.push_back(std::move(en));
        }
        closedir(d);
    }
    size_t skipped() const { return skipped_; }
    static constexpr size_t kMaxEntries = 256;
    static double now() {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

private:
    std::string dir_;
    int ifd_ = -1;
    bool dirty_ = true;
    double last_ = -1e18;
    size_t skipped_ = 0;
    std::vector<Entry> e_;
};

// ---- eviction: one flock'd evictor role per directory; writers prune their own objects only
class Evictor {
public:
    ~Evictor() {
        stop_ = true;
        if (th_.joinable()) th_.join();
        if (fd_ >= 0) close(fd_);
    }
    // true: this process is the directory's evictor (the flock is held until destruction)
    bool acquire(const std::string& dir) {
        const std::string p = dir + "/.qpin.evict";
        fd_ = open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ < 0) return false;
        if (flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            close(fd_);
            fd_ = -1;
            return false;
        }
        dir_ = dir;
        return true;
    }
    void start(uint64_t cap) {
        if (fd_ < 0 || th_.joinable()) return;
        stop_ = false;
        th_ = std::thread([this, cap] {
            while (!stop_.load()) {
                enforce(cap);
                for (int i = 0; i < 10 && !stop_.load(); i++) usleep(100000);
            }
        });
    }
    // unlink the oldest objects of any writer until the directory's total is <= cap; returns bytes removed
    uint64_t enforce(uint64_t cap) {
        std::lock_guard<std::mutex> guard(mu_);
        if (fd_ < 0) return 0;
        struct Obj { std::string path; uint64_t bytes; double mtime; };
        std::vector<Obj> v;
        uint64_t total = 0;
        DIR* d = opendir(dir_.c_str());
        if (!d) return 0;
        while (dirent* de = readdir(d)) {
            const std::string nm = de->d_name;
            if (nm.find(kTmpTag) != std::string::npos) {
                const std::string tmp = dir_ + "/" + nm;
                struct stat ts;
                if (stat(tmp.c_str(), &ts) == 0 && Index::now() - (double)ts.st_mtime > kTempAgeS)
                    unlink(tmp.c_str());
                continue;
            }
            const size_t e = strlen(kExt);
            if (nm.size() <= e || nm.compare(nm.size() - e, e, kExt) != 0) continue;
            const std::string path = dir_ + "/" + nm;
            struct stat st;
            if (stat(path.c_str(), &st) != 0) continue;
            v.push_back({path, (uint64_t)st.st_size, (double)st.st_mtime});
            total += (uint64_t)st.st_size;
        }
        closedir(d);
        std::sort(v.begin(), v.end(), [](const Obj& a, const Obj& b) { return a.mtime < b.mtime; });
        uint64_t removed = 0;
        for (auto& o : v) {
            if (total <= cap) break;
            if (unlink(o.path.c_str()) == 0) {
                total -= std::min(total, o.bytes);
                removed += o.bytes;
                n_evicted++;
            }
        }
        return removed;
    }
    long long evicted() const { return n_evicted; }

private:
    std::string dir_;
    int fd_ = -1;
    std::mutex mu_;
    std::atomic<bool> stop_{false};
    std::thread th_;
    long long n_evicted = 0;
};

// remove this writer's oldest objects until its own total fits; paths are the ones this process published
inline uint64_t prune_own(std::vector<std::string>& own, uint64_t& own_bytes, uint64_t cap) {
    uint64_t removed = 0;
    while (own_bytes > cap && !own.empty()) {
        struct stat st;
        const std::string path = own.front();
        uint64_t b = 0;
        if (stat(path.c_str(), &st) == 0) b = (uint64_t)st.st_size;
        unlink(path.c_str());
        own.erase(own.begin());
        own_bytes -= std::min(own_bytes, b);
        removed += b;
    }
    return removed;
}
}  // namespace sp
