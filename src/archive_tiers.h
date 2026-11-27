#pragma once
// Tiered persistent archive store: N independently capped tiers (RAM /dev/shm, NVMe) behind one API. Host-only, no CUDA.
// The engine keeps the tokens/KV; this index owns the per-tier byte accounting, the hard caps, the free-space floor,
// per-session identity (latest checkpoint only, delete by session) and TTL expiry. Every object lives in exactly one tier.
#include "archive_store_policy.h"

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace qe_archive_store {

struct Tier {
    std::string name, dir;
    bool ram = false;
    size_t cap = 0;
    unsigned max_pct = 85;  // free-space floor of a non-RAM tier: no write that would pass this filesystem usage
    size_t bytes = 0, objects = 0;
    unsigned long long evictions = 0;
};

struct Obj {
    int tier = 0;
    size_t bytes = 0;
    uint64_t seq = 0;       // LRU / recency order inside the store (higher = newer)
    uint64_t session = 0;   // x-session-hash (0: none: pinned prefixes, anonymous requests)
    int64_t expires = 0;    // wall-clock seconds; 0: never
};

inline bool statvfs_usage(const std::string& dir, size_t& total, size_t& avail) {
    struct statvfs fs{};
    if (statvfs(dir.c_str(), &fs) != 0) return false;
    total = (size_t)fs.f_blocks * fs.f_frsize;
    avail = (size_t)fs.f_bavail * fs.f_frsize;
    return true;
}

// The engine-side selection rules over a snapshot of its archives (index = position in the archive list). `protect`:
// pinned prefix, held by a restore, or a sibling arm's read-only file: never a victim.
struct View {
    uint64_t session = 0;
    int64_t expires = 0;
    uint64_t used = 0;  // recency tick (higher = newer)
    bool protect = false;
};
inline std::vector<size_t> delete_targets(const std::vector<View>& v, uint64_t session) {
    std::vector<size_t> r;
    if (session)
        for (size_t i = 0; i < v.size(); i++)
            if (v[i].session == session && !v[i].protect) r.push_back(i);
    return r;
}
inline std::vector<size_t> expired_targets(const std::vector<View>& v, int64_t now) {
    std::vector<size_t> r;
    for (size_t i = 0; i < v.size(); i++)
        if (v[i].expires > 0 && v[i].expires <= now && !v[i].protect) r.push_back(i);
    return r;
}
// the session has a newer checkpoint than v[i]: a flush skips v[i] (latest only)
inline bool superseded_in_session(const std::vector<View>& v, size_t i) {
    if (!v[i].session) return false;
    for (size_t j = 0; j < v.size(); j++)
        if (j != i && !v[j].protect && v[j].session == v[i].session && v[j].used > v[i].used) return true;
    return false;
}
// the session's older checkpoints once v[keep] is complete
inline std::vector<size_t> older_in_session(const std::vector<View>& v, size_t keep) {
    std::vector<size_t> r;
    if (v[keep].session)
        for (size_t j = 0; j < v.size(); j++)
            if (j != keep && !v[j].protect && v[j].session == v[keep].session && v[j].used < v[keep].used) r.push_back(j);
    return r;
}

class TieredStore {
public:
    std::function<int64_t()> clock = [] { return (int64_t)time(nullptr); };
    std::function<bool(const std::string&, size_t&, size_t&)> usage = statvfs_usage;

    int add_tier(const std::string& name, const std::string& dir, bool ram, size_t cap, unsigned max_pct = 85) {
        Tier t;
        t.name = name, t.dir = dir, t.ram = ram, t.cap = cap, t.max_pct = max_pct;
        tiers_.push_back(t);
        return (int)tiers_.size() - 1;
    }
    int tiers() const { return (int)tiers_.size(); }
    const Tier& tier(int t) const { return tiers_[t]; }
    int find_tier(const std::string& name) const {
        for (int i = 0; i < tiers(); i++)
            if (tiers_[i].name == name) return i;
        return -1;
    }

    // --- accounting (the engine writes its own files and registers them here)
    bool fits(int t, size_t need, size_t extra_used = 0) const {
        const size_t used = tiers_[t].bytes > ~(size_t)0 - extra_used ? ~(size_t)0 : tiers_[t].bytes + extra_used;
        return qe_archive_store::fits(tiers_[t].cap, used, need);
    }
    bool floor_ok(int t, size_t need) const {
        if (tiers_[t].ram) return true;
        size_t total = 0, avail = 0;
        return usage(tiers_[t].dir, total, avail) && below_usage_limit(total, avail, need, tiers_[t].max_pct);
    }
    void add(const std::string& path, Obj o) {
        remove(path, false);
        o.seq = ++seq_;
        objs_[path] = o;
        tiers_[o.tier].bytes += o.bytes, tiers_[o.tier].objects++;
    }
    void reset_objects() {  // forget every object (files stay), keep tiers and eviction counters
        objs_.clear();
        for (auto& t : tiers_) t.bytes = 0, t.objects = 0;
    }
    bool has(const std::string& path) const { return objs_.count(path) != 0; }
    const Obj* get(const std::string& path) const {
        auto it = objs_.find(path);
        return it == objs_.end() ? nullptr : &it->second;
    }
    void touch(const std::string& path) {
        auto it = objs_.find(path);
        if (it != objs_.end()) it->second.seq = ++seq_;
    }
    bool remove(const std::string& path, bool evicted) {
        auto it = objs_.find(path);
        if (it == objs_.end()) return false;
        Tier& t = tiers_[it->second.tier];
        t.bytes -= it->second.bytes, t.objects--;
        if (evicted) t.evictions++;
        objs_.erase(it);
        return true;
    }
    // oldest object of a tier that `locked` does not protect ("" if none)
    std::string lru(int t, const std::function<bool(const std::string&)>& locked = nullptr) const {
        std::string best;
        uint64_t bs = ~0ull;
        for (auto& [p, o] : objs_)
            if (o.tier == t && o.seq < bs && !(locked && locked(p))) best = p, bs = o.seq;
        return best;
    }

    // --- victim selection (paths; the caller drops them through its own path, or unlink_all below)
    std::vector<std::string> session_paths(uint64_t session) const {
        std::vector<std::string> v;
        if (session)
            for (auto& [p, o] : objs_)
                if (o.session == session) v.push_back(p);
        return v;
    }
    // everything of the session except `keep`: the latest checkpoint wins
    std::vector<std::string> superseded(uint64_t session, const std::string& keep) const {
        std::vector<std::string> v;
        for (auto& p : session_paths(session))
            if (p != keep) v.push_back(p);
        return v;
    }
    std::vector<std::string> expired(int64_t now) const {
        std::vector<std::string> v;
        for (auto& [p, o] : objs_)
            if (o.expires > 0 && o.expires <= now) v.push_back(p);
        return v;
    }

    // --- file-level API (put/get/evict/delete); flush = the engine's flush_archives, which calls put-equivalents
    // Write `n` bytes as <tier dir>/<name>: expired objects, then LRU ones, go first until it fits under the hard cap;
    // refused (false, nothing written) when it cannot fit or would pass the free-space floor.
    bool put(int t, const std::string& name, const char* data, size_t n, uint64_t session = 0, int64_t ttl_s = 0,
             std::string* out = nullptr) {
        if (t < 0 || t >= tiers() || n > tiers_[t].cap || !floor_ok(t, n)) return false;
        const std::string path = tiers_[t].dir + "/" + name;
        gc(clock());
        while (!fits(t, n)) {
            const std::string v = lru(t);
            if (v.empty() || v == path) return false;
            unlink_remove(v, true);
        }
        const std::string tmp = path + ".tmp";
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) return false;
        const bool ok = fwrite(data, 1, n, f) == n && fflush(f) == 0;
        fclose(f);
        if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
            unlink(tmp.c_str());
            return false;
        }
        Obj o;
        o.tier = t, o.bytes = n, o.session = session, o.expires = ttl_s > 0 ? clock() + ttl_s : 0;
        add(path, o);
        for (auto& old : superseded(session, path)) unlink_remove(old, true);
        if (out) *out = path;
        return true;
    }
    // path of a stored object, recency refreshed ("" when absent)
    std::string get_path(int t, const std::string& name) {
        const std::string p = tiers_[t].dir + "/" + name;
        if (!has(p)) return "";
        touch(p);
        return p;
    }
    size_t evict(int t, size_t need) {  // free at least `need` bytes of room (LRU); returns objects evicted
        size_t k = 0;
        while (!fits(t, need)) {
            const std::string v = lru(t);
            if (v.empty()) break;
            unlink_remove(v, true), k++;
        }
        return k;
    }
    size_t delete_session(uint64_t session) {
        size_t k = 0;
        for (auto& p : session_paths(session)) unlink_remove(p, false), k++;
        return k;
    }
    size_t gc(int64_t now) {
        size_t k = 0;
        for (auto& p : expired(now)) unlink_remove(p, true), k++;
        return k;
    }

private:
    void unlink_remove(const std::string& path, bool evicted) {
        unlink(path.c_str());
        remove(path, evicted);
    }
    std::vector<Tier> tiers_;
    std::map<std::string, Obj> objs_;
    uint64_t seq_ = 0;
};

}  // namespace qe_archive_store
