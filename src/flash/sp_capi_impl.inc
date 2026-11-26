// Out-of-class definitions of FE's shared-PREFIX methods (declared in flash_capi.cpp, included after class FE).
// QE_SHARED_L2_DIR unset: every entry point returns immediately, so the feature is off by default.
#include "flash_adapter.h"

#define sp_dir flash_adapter::state(this).sp_dir
#define sp_fp flash_adapter::state(this).sp_fp
#define sp_index flash_adapter::state(this).sp_index
#define sp_evictor flash_adapter::state(this).sp_evictor
#define sp_own flash_adapter::state(this).sp_own
#define sp_quota flash_adapter::state(this).sp_quota
#define sp_own_bytes flash_adapter::state(this).sp_own_bytes
#define sp_writer flash_adapter::state(this).sp_writer
#define sp_global flash_adapter::state(this).sp_global
#define sp_pages_advice flash_adapter::state(this).sp_pages_advice
#define n_sp_write flash_adapter::state(this).n_sp_write
#define n_sp_hit flash_adapter::state(this).n_sp_hit
#define n_sp_miss flash_adapter::state(this).n_sp_miss
#define n_sp_fail flash_adapter::state(this).n_sp_fail
#define n_sp_fallback flash_adapter::state(this).n_sp_fallback
void FE::sp_fail_log(const char* at, int err) {
    if (n_sp_fail++ < 3)
        fprintf(stderr, "[qflash] shared L2: prefix %s failed: %s (a miss, never fatal)\n", at, strerror(err));
}

void FE::sp_init(const std::string& model) {
    const char* dir = env_str("QE_SHARED_L2_DIR");
    if (!dir || !*dir) return;
    sp_dir = dir;
    for (size_t i = 1; i <= sp_dir.size(); i++)
        if (i == sp_dir.size() || sp_dir[i] == '/') mkdir(sp_dir.substr(0, i).c_str(), 0755);
    if (access(sp_dir.c_str(), R_OK | W_OK) != 0) {
        fprintf(stderr, "[qflash] shared L2: prefix sharing off: %s: %s\n", sp_dir.c_str(), strerror(errno));
        sp_dir.clear();
        return;
    }
    sp_fp = dk_fingerprint(model);
    if (getrandom(&sp_writer, sizeof sp_writer, 0) != (ssize_t)sizeof sp_writer || sp_writer == 0)
        sp_writer = ((uint64_t)getpid() << 32) ^ (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() ^ 1;
    sp_quota = (size_t)(std::max(0.0, env_double("QE_SHARED_L2_WRITER_GB", 8.0)) * (1ull << 30));
    sp_global = (uint64_t)(std::max(0.0, env_double("QE_SHARED_L2_GB", 32.0)) * (1ull << 30));
    #include "sp_pages_init.inc"
    const size_t pin_cap = (size_t)(std::max(0.0, env_double("QE_SHARED_L2_PIN_GB", 16.0)) * (1ull << 30));
    sp_index = std::make_shared<sp::Index>();
    const bool ino = sp_index->init(sp_dir);
    sp_evictor = std::make_unique<sp::Evictor>();
    const bool ev = sp_evictor->acquire(sp_dir);
    sp::set_hooks(
        [](void* p, size_t n, void* ctx, int* err) -> bool {
            FE* e = (FE*)ctx;
            cudaError_t x = cudaSuccess;
            if (!e->devs.empty()) x = cudaSetDevice(e->devs[0]);
            #include "sp_register_body.inc"
            if (x != cudaSuccess) {
                *err = (int)x;
                return false;
            }
            return true;
        },
        [](void* p, size_t, void*) { cudaHostUnregister(p); }, this, pin_cap);
    sp::hooks().rw_pin = env_double("QE_SHARED_L2_RW_PIN", 1.0) != 0.0;
    fprintf(stderr,
            "[qflash] shared L2: prefix objects in %s (%s index, %s evictor; writer quota %.1f GB, dir cap %.1f GB, "
            "registration cap %.1f GB; fingerprint %.16s)\n",
            sp_dir.c_str(), ino ? "inotify" : "rescan-only", ev ? "this process" : "another", sp_quota / 1e9,
            sp_global / 1e9, pin_cap / 1e9, sp_fp.c_str());
    if (ev) sp_evictor->start(sp_global);
    const size_t queue_depth = (size_t)std::max(0, env_int("QE_SHARED_L2_PUBLISH_QUEUE", 2));
    sp_publish_queue = std::make_unique<sp::AsyncWriter<Pin>>(
        queue_depth,
        [](void* context, Pin&& queued) {
            FE* e = (FE*)context;
            try {
                e->sp_publish_sync(std::move(queued));
            } catch (...) {
                e->sp_fail_log("async publish", EIO);
            }
        },
        this);
    fprintf(stderr, "[qflash] shared L2: async publisher queue depth %zu\n", queue_depth);
}

// Queue a copy of the Pin so its shared Seg/State references keep the payload alive through the write.
void FE::sp_publish(const Pin& p) {
    if (sp_dir.empty() || !budget || !sp_publish_queue) return;
    const int P = (int)p.toks.size();
    if (P <= 0 || p.st->pos != P || p.st->pend || p.st->kind != 1 || p.st->prev_m != 0) return;
    if (!sp_publish_queue->try_push(p)) {
        const uint64_t dropped = sp_publish_queue->dropped();
        if (dropped == 1 || (dropped & (dropped - 1)) == 0)
            fprintf(stderr, "[qflash] shared L2: async publish queue full (dropped %llu)\n", (unsigned long long)dropped);
    }
}

// The single background writer performs the former synchronous publish work.
// Publish a new pin as one shared object. The bytes are copied from the pin's private host blobs: this is the writer
// side (once per system prefix); a reader maps the object and DMAs from the shared pages, there is no read() path.
void FE::sp_publish_sync(Pin&& p) {
    if (sp_dir.empty() || !budget) return;
    const int P = (int)p.toks.size();
    if (P <= 0 || p.st->pos != P || p.st->pend || p.st->kind != 1 || p.st->prev_m != 0) return;
    const std::string sid = dk_steer_id(p.steer);
    sp::Desc d;
    d.fp = sp_fp;
    d.steer = sid;
    d.toks = std::move(p.toks);
    d.writer = sp_writer;
    d.ng = ng;
    d.kv.resize(ng);
    d.st.resize(ng);
    size_t need = 0;
    for (int r = 0; r < ng; r++) {
        d.kv[r].len = p.kv->n[r];
        d.st[r].len = p.st->n[r];
        need += sp::align_up((size_t)p.kv->n[r], 4096) + sp::align_up((size_t)p.st->n[r], 4096);
    }
    const double t0 = now_ms();
    sp_own_bytes = 0;
    for (auto& x : sp_own) {
        struct stat st;
        if (stat(x.c_str(), &st) == 0) sp_own_bytes += (size_t)st.st_size;
    }
    while (sp_quota && sp_own_bytes + need > sp_quota && !sp_own.empty()) {
        const std::string path = sp_own.front();
        struct stat st;
        size_t bytes = 0;
        if (stat(path.c_str(), &st) == 0) {
            bytes = (size_t)st.st_size;
            sp::Map owned;
            if (sp::open_map(path, sp_fp, owned) == sp::Rd::OK && owned.desc().writer == sp_writer)
                unlink(path.c_str());
        }
        sp_own.erase(sp_own.begin());
        sp_own_bytes -= std::min(sp_own_bytes, bytes);
    }
    sp::Builder b;
    int err = 0;
    if (!b.create(sp_dir, d, err)) return sp_fail_log("create", err);
    if (!b.advise(sp_pages_advice)) {
        static bool warned_advice = false;
        if (!warned_advice) {
            warned_advice = true;
            fprintf(stderr, "[qflash] shared L2: prefix page advice failed (ignored)\n");
        }
    }
    for (int r = 0; r < ng; r++) {
        memcpy(b.kv(r), p.kv->p[r], (size_t)p.kv->n[r]);
        memcpy(b.st(r), p.st->p[r], (size_t)p.st->n[r]);
    }
    const std::string k = sp::key(sp_fp, sid, d.toks);
    if (!b.publish(sp_dir, k, err)) return sp_fail_log("publish", err);
    sp_own.push_back(sp::path_of(sp_dir, k));
    sp_own_bytes += need;
    n_sp_write++;
    if (sp_evictor) sp_evictor->enforce(sp_global);
    if (log || n_sp_write <= 3)
        fprintf(stderr, "[qflash] shared L2: published prefix %d tokens (%zu MB object, %d cards; %.1f ms, %lld written)\n",
                P, need >> 20, ng, now_ms() - t0, (long long)n_sp_write.load());
}

bool FE::sp_fits(const sp::Desc& d) {
    if (d.ng != ng) return false;
    for (int r = 0; r < ng; r++) {
        if (W[r]->kv_bytes(0, (int)d.toks.size()) != d.kv[r].len) return false;
        std::vector<Worker::Piece> ps;
        W[r]->state_pieces_of(0, false, ps);
        if (bytes_of(ps) != d.st[r].len) return false;
    }
    return true;
}

// Restore a shared prefix: mapped registered shards DMA directly; registration refusal uses an explicit, logged private copy.
#include "sp_capi_load.inc"

// The longest shared prefix for ids[0, n) that beats every local source; loaded as a pin (subject to the pin budget).
Pin* FE::sp_best(const int* ids, int n, const Src& b, const Pin* keep) {
    if (sp_dir.empty() || !pin_budget || !budget) return nullptr;
    const std::vector<sp::Entry>& es = sp_index->entries(sp_fp);
    const std::string want = dk_steer_id(steer_key());
    const sp::Entry* pick = nullptr;
    int bestP = -1;
    for (auto& e : es) {
        const int P = (int)e.toks.size();
        if (e.steer != want || P <= 0 || P >= n || P < ckpt_min) continue;
        if (P < b.L || (P == b.L && (b.c || b.pin))) continue;
        if (!std::equal(e.toks.begin(), e.toks.end(), ids)) continue;
        if (P > bestP) {
            bestP = P;
            pick = &e;
        }
    }
    return pick ? sp_load(*pick, keep) : nullptr;
}
#undef sp_dir
#undef sp_fp
#undef sp_index
#undef sp_evictor
#undef sp_own
#undef sp_quota
#undef sp_own_bytes
#undef sp_writer
#undef sp_global
#undef sp_pages_advice
#undef n_sp_write
#undef n_sp_hit
#undef n_sp_miss
#undef n_sp_fail
#undef n_sp_fallback
