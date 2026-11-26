// Included inside FE: the opt-in path owns physical periods independently of slot addresses.
bool live_on = false;
int live_stride = 0;
int live_last_init_tokens = 0;
std::unique_ptr<flash_live::Arena> live_arena;
flash_live::PrefixIndex live_index;
std::vector<std::unique_ptr<flash_live::View>> live_views;
std::vector<unsigned char> live_marks;

void live_init() {
    if (!env_int("FLASH_KV_SHARE_PREFIX", 0)) return;
    CHECK(!ppb && !pph && W[0]->kv_stream(), "FLASH_KV_SHARE_PREFIX requires text TP RAM KV4 (no hybrid pipeline)");
    live_stride = periods_for(ctx + MAXP + 2);
    CHECK(live_stride > 0 && (long long)live_stride * sl.size() * period < INT_MAX,
          "FLASH_KV_SHARE_PREFIX: virtual address space exceeds int token offsets");
    live_arena = std::make_unique<flash_live::Arena>(np);
    live_views.resize(sl.size());
    live_marks.resize(np);
    job = [&](int r) { W[r]->live_init(live_stride, np); };
    run("live prefix init");
    live_on = true;
    fprintf(stderr, "[qflash] live prefix sharing ON: %d physical periods, %d virtual periods/slot\n", np, live_stride);
}
int live_shared_tokens(int s) const {
    return live_on && live_views[s] ? live_views[s]->shared_periods() * period : 0;
}
void live_update_top() {
    std::fill(live_marks.begin(), live_marks.end(), 0);
    long long active = 0;
    for (size_t s = 0; s < sl.size(); ++s)
        if (sl[s].active && live_views[s])
            for (int p : live_views[s]->periods())
                if (!live_marks[p]) { live_marks[p] = 1; active += period; }
    const int top = live_arena->high_water();
    job = [&](int r) {
        W[r]->set_kv_active(active);
        W[r]->set_kv_top(top);
        W[r]->kv_claim(0);
    };
    run("live prefix pool top");
}
void live_drop(int s) {
    if (!live_on || !live_views[s]) return;
    job = [&](int r) { W[r]->live_map(s, {}, 0, {}); };
    run("live prefix unmap");  // all consumers complete before the last extent can release periods
    live_views[s].reset();
}
bool live_ensure(int s, int tokens, const std::vector<int>& protect) {
    if (tokens > ctx + MAXP + 2 || tokens < 0) return false;
    const int need = std::min(live_stride, periods_for(tokens + reserve));
    if (!live_views[s]) live_views[s] = std::make_unique<flash_live::View>(*live_arena);
    auto& view = *live_views[s];
    const size_t old = view.periods().size();
    if (old >= (size_t)need && sl[s].cap) return true;  // steady decode: no allocations, copies, or rebind
    while (!view.ensure(need)) {
        const int v = idle_victim(s, protect);
        if (v < 0) {
            if (!old) live_views[s].reset();
            return false;
        }
        evict_slot(v, true, "live suffix needed physical periods");
    }
    if (view.periods().size() != old) {
        std::vector<int> added(view.periods().begin() + old, view.periods().end());
        job = [&](int r) { W[r]->live_map(s, view.periods(), view.shared_periods(), added); };
        run("live suffix growth");
    }
    sl[s].base = s * live_stride;
    sl[s].cap = (int)view.periods().size();
    rebind();
    live_update_top();
    return true;
}
int live_acquire(int n, const std::vector<int>& protect, Pin* pin) {
    live_last_init_tokens = 0;
    int s = free_slot();
    if (s < 0) {
        s = idle_victim(-1, protect, true);
        if (s < 0) return -1;
        evict_slot(s, true, "live prefix start needed a slot");
    }
    std::shared_ptr<const void> owner;
    flash_live::ExtentPtr prefix;
    // Keep the MTP warm-up/replay window private, including every partially filled four-row block.
    const int shared = pin ? std::max(0, ((int)pin->toks.size() - MAXP - 2) / period) : 0;
    bool fresh = false;
    if (shared) {
        owner = std::static_pointer_cast<const void>(pin->kv);
        prefix = live_index.find(owner, shared);
        if (!prefix) {
            prefix = live_arena->allocate(shared, owner);
            while (!prefix) {
                const int v = idle_victim(s, protect);
                if (v < 0) return -1;
                evict_slot(v, true, "live prefix extent needed physical periods");
                prefix = live_arena->allocate(shared, owner);
            }
            fresh = true;
        }
    }
    live_drop(s);
    live_views[s] = std::make_unique<flash_live::View>(*live_arena, prefix);
    auto& view = *live_views[s];
    if (!live_ensure(s, n, protect)) { live_drop(s); return -1; }
    // live_ensure published the suffix additions; install/initialize a fresh immutable prefix before discovery.
    if (prefix) {
        const std::vector<int> added = fresh ? prefix->periods() : std::vector<int>{};
        job = [&](int r) { W[r]->live_map(s, view.periods(), shared, added); };
        run("live prefix bind");
        if (fresh) {
            job = [&](int r) { W[r]->live_restore(pin->kv->p[r], sl[s].base, 0,
                                                    (int)pin->toks.size(), shared * period, true); };
            run("live prefix initialize");
            live_last_init_tokens = shared * period;
            live_index.insert(owner, shared, prefix);
        }
    }
    if (log) fprintf(stderr, "[qflash] live prefix: slot %d shared %d tokens (%s), physical %d/%d periods\n",
                     s, shared * period, fresh ? "initialized" : "retained", live_arena->used(), live_arena->capacity());
    return s;
}

// Preserve the public admission policy, but count active physical periods only once.
int live_active_periods(bool paused_floors, const std::vector<unsigned char>* skip = nullptr) {
    std::fill(live_marks.begin(), live_marks.end(), 0);
    int used = 0;
    for (size_t s = 0; s < sl.size(); ++s) {
        if (!sl[s].active) continue;
        if (skip && (*skip)[s]) continue;
        if (sl[s].paused) { if (paused_floors) used += floor_of(sl[s]); continue; }
        if (live_views[s])
            for (int p : live_views[s]->periods())
                if (!live_marks[p]) { live_marks[p] = 1; ++used; }
    }
    return used;
}
bool live_can_admit(int n) {
    int active = 0;
    for (const auto& q : sl) active += q.active;
    if (!batch_api) return active == 0;
    return active < (int)sl.size() &&
           live_active_periods(true) + std::min(np, periods_for(n + reserve)) <= np;
}
int live_pause_for(int n) {
    const int need = std::min(np, periods_for(n + reserve));
    std::vector<int> victims;
    int active = 0;
    for (int s = 0; s < (int)sl.size(); ++s) {
        active += sl[s].active;
        if (sl[s].active && !sl[s].paused && sl[s].conv && sl[s].cap &&
            sl[s].pending >= 0 && !prefilling(s)) victims.push_back(s);
    }
    if (active >= (int)sl.size()) return 0;
    std::sort(victims.begin(), victims.end(), [&](int a, int b) { return sl[a].admit > sl[b].admit; });
    std::vector<unsigned char> skip(sl.size(), 0);
    size_t take = 0;
    while (live_active_periods(false, &skip) + need > np) {
        if (take == victims.size()) return 0;  // no mutation if the planned pauses cannot make room
        skip[victims[take++]] = 1;
    }
    for (size_t i = 0; i < take; ++i)
        if (!pause_slot(victims[i])) return 0;
    return 1;
}
