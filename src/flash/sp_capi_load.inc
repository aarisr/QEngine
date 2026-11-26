#include "flash_adapter.h"
// Restore a shared prefix. The normal path keeps the MAP_SHARED view alive as the Pin and CUDA DMAs from the registered
// shards. If registration is unavailable/over budget, copy the mapped shards once into this process's HostPool (logged);
// if neither path fits the pin budget/headroom, the caller treats this as a miss and prefills.
Pin* FE::sp_load(const sp::Entry& en, const Pin* keep) {
    const double t0 = now_ms();
    flash_adapter::KeepPin<FE, Pin> keep_guard(this, keep);
    auto mp = std::make_shared<sp::Map>();
    const char* why = "";
    if (sp::open_map(en.path, sp_fp, *mp, &why) != sp::Rd::OK) {
        n_sp_miss++;
        return nullptr;  // vanished/replaced/foreign file: normal cache miss
    }
    if (!mp->advise(sp_pages_advice)) sp_fail_log("madvise", errno);
    const sp::Desc& d = mp->desc();
    if (d.steer != en.steer || d.toks != en.toks || !sp_fits(d)) return nullptr;

    size_t copy_need = 0;
    std::vector<size_t> kv_sizes(ng), st_sizes(ng);
    for (int r = 0; r < ng; r++) {
        kv_sizes[r] = (size_t)d.kv[r].len;
        st_sizes[r] = (size_t)d.st[r].len;
        copy_need += sp::align_up(kv_sizes[r], 4096) + sp::align_up(st_sizes[r], 4096);
    }
    const size_t map_need = mp->size();
    const bool map_budget = map_need <= pin_budget;
    const bool copy_budget = copy_need <= pin_budget;
    if (!map_budget && !copy_budget) { sp_fail_log("pin budget", ENOMEM); return nullptr; }
    mem_guard();
    const bool map_fits = map_budget && qe_hostmem::headroom() >= map_need;
    const bool copy_fits = copy_budget && qe_hostmem::headroom() >= copy_need;
    if (!map_fits && !copy_fits) { sp_fail_log("headroom", ENOMEM); return nullptr; }
    if (!mp->verify_all()) { n_sp_miss++; sp_fail_log("shard checksum", EIO); return nullptr; }  // reject before use
    std::shared_ptr<Seg> kv;
    std::shared_ptr<State> st;
    bool registered_map = false;
    size_t need = 0;
    if (map_fits) {
        while (pin_used + map_need > pin_budget)
            if (!drop_pin("shared prefix budget")) break;
        if (pin_used + map_need <= pin_budget) {
            registered_map = mp->attach();
            if (registered_map) {
                need = map_need;
                kv = flash_adapter::mapped_blob<Seg>(mp);
                st = flash_adapter::mapped_blob<State>(mp);
                kv->layout = st->layout = layout;
                kv->p.resize(ng);
                kv->n.resize(ng);
                kv->pinned.assign(ng, 1);
                st->p.resize(ng);
                st->n.resize(ng);
                st->pinned.assign(ng, 1);
                kv->t0 = 0;
                kv->t1 = (int)d.toks.size();
                for (int r = 0; r < ng; r++) {
                    kv->p[r] = (char*)mp->kv(r);
                    kv->n[r] = kv_sizes[r];
                    kv->bytes += kv->n[r];
                    st->p[r] = (char*)mp->st(r);
                    st->n[r] = st_sizes[r];
                    st->bytes += st->n[r];
                }
            }
        }
    }
    if (!registered_map) {
        if (!copy_fits) { sp_fail_log("private-copy budget/headroom", ENOMEM); return nullptr; }
        while (pin_used + copy_need > pin_budget)
            if (!drop_pin("shared prefix copy budget")) { sp_fail_log("private-copy pin budget", ENOMEM); return nullptr; }
        if (pin_used + copy_need > pin_budget || qe_hostmem::headroom() < copy_need) {
            sp_fail_log("private-copy headroom", ENOMEM);
            return nullptr;
        }
        if (n_sp_fallback++ < 3)
            fprintf(stderr, "[qflash] shared L2: registration unavailable/over budget; copying mapped shards into private host buffers\n");
        kv = std::make_shared<Seg>();
        st = std::make_shared<State>();
        kv->layout = st->layout = layout;
        if (!kv->alloc(&hpool, kv_sizes) || !st->alloc(&hpool, st_sizes)) {
            n_sp_miss++;
            sp_fail_log("private-copy allocation", ENOMEM);
            return nullptr;
        }
        kv->t0 = 0;
        kv->t1 = (int)d.toks.size();
        for (int r = 0; r < ng; r++) {
            memcpy(kv->p[r], mp->kv(r), kv_sizes[r]);
            memcpy(st->p[r], mp->st(r), st_sizes[r]);
        }
        kv->bytes = 0;
        st->bytes = 0;
        for (int r = 0; r < ng; r++) { kv->bytes += kv_sizes[r]; st->bytes += st_sizes[r]; }
        need = copy_need;
    }

    st->pos = (int)d.toks.size();
    st->prev_m = 0;
    st->pend = false;
    st->kind = 1;
    Pin p;
    p.toks = d.toks;
    p.steer = steer_key();
    p.kv = kv;
    p.st = st;
    p.bytes = need;
    p.used = ++tick;
    pins.push_back(std::move(p));
    pin_used += need;
    n_pin++;
    n_sp_hit++;
    (void)utimensat(AT_FDCWD, en.path.c_str(), nullptr, 0);
    auto it = std::find(sp_own.begin(), sp_own.end(), en.path);
    if (it != sp_own.end()) { sp_own.erase(it); sp_own.push_back(en.path); }
    if (log)
        fprintf(stderr, "[qflash] archive: shared prefix %zu tokens (%s, %.2f GB; %.1f ms, %lld hit(s))\n",
                d.toks.size(), registered_map ? "mmap+registered DMA" : "private copy", need / 1e9,
                now_ms() - t0, n_sp_hit.load());
    return &pins.back();
}
