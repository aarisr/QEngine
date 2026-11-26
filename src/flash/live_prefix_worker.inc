// Included by flash.cu inside namespace flash, after the KV kernels are defined.
void Worker::live_init(int stride, int capacity) {
    CHECK(kvs_ && kv4_ && !p_.pp() && !kvh_lazy_ && p_.own % IDX_R == 0,
          "FLASH_KV_SHARE_PREFIX requires eager RAM KV4, TP, and complete four-row ownership blocks");
    CHECK(stride > 0 && (long long)stride * slots_.size() * p_.own < INT_MAX,
          "live prefix: virtual owned-row addresses overflow");
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaDeviceSynchronize());
    drop_graphs(nullptr, 1.0);  // old captures must not retain identity-map kernel arguments
    live_stride_ = stride;
    live_capacity_ = capacity;
    live_period_h_.assign((size_t)stride * slots_.size(), -1);
    live_shared_h_.assign(slots_.size(), 0);
    live_period_d_ = (int*)dalloc(live_period_h_.size() * sizeof(int));
    live_shared_d_ = (int*)dalloc(live_shared_h_.size() * sizeof(int));
    CUDA_CHECK(cudaMemset(live_period_d_, 0xff, live_period_h_.size() * sizeof(int)));
    CUDA_CHECK(cudaMemset(live_shared_d_, 0, live_shared_h_.size() * sizeof(int)));
    for (int li = 0; li <= LAYERS; ++li) {
        if (li == LAYERS && !has_mtp_) break;
        Layer& layer = li < LAYERS ? L_[li] : mtp_;
        if (!layer.kvp.ctl) continue;
        layer.kvp.m.period_map = live_period_d_;
        layer.kvp.m.map_own = p_.own;
        layer.kvp.m.prefix_periods = live_shared_d_;
        layer.kvp.m.slot_periods = stride;
    }
}
void Worker::live_map(int slot, const std::vector<int>& periods, int shared, const std::vector<int>& added) {
    CHECK(slot >= 0 && slot < (int)slots_.size() && periods.size() <= (size_t)live_stride_ &&
          shared >= 0 && shared <= (int)periods.size(), "live prefix: invalid slot mapping");
    CUDA_CHECK(cudaSetDevice(dev_));
    // Ownership transitions may follow prefill on another stream. No mapping is recycled while a consumer runs.
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int physical : periods) CHECK(physical >= 0 && physical < live_capacity_, "live prefix: invalid physical period");
    for (size_t i = 0; i < added.size();) {
        size_t end = i + 1;
        while (end < added.size() && added[end] == added[end-1] + 1) ++end;
        const int b0 = added[i] * p_.own / IDX_R, b1 = (added[end-1] + 1) * p_.own / IDX_R;
        for (int li = 0; li <= LAYERS; ++li) {
            if (li == LAYERS && !has_mtp_) break;
            Layer& layer = li < LAYERS ? L_[li] : mtp_;
            if (layer.kvp.ctl) k_kv_inval<<<std::min(256, (b1-b0+255)/256), 256, 0, st_>>>(layer.kvp,b0,b1);
        }
        i = end;
    }
    int* first = live_period_h_.data() + (size_t)slot * live_stride_;
    std::fill(first, first + live_stride_, -1);
    std::copy(periods.begin(), periods.end(), first);
    live_shared_h_[slot] = shared;
    CUDA_CHECK(cudaMemcpyAsync(live_period_d_ + (size_t)slot * live_stride_, first,
                               (size_t)live_stride_ * sizeof(int), cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(live_shared_d_ + slot, &live_shared_h_[slot], sizeof(int), cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}
void Worker::live_pieces(int base, int t0, int t1, std::vector<Piece>& out) {
    const int r0 = kv_rows(t0), r1 = kv_rows(t1);
    if (r1 <= r0) return;
    const size_t logical = (size_t)base * p_.own;
    auto field = [&](char* storage, size_t bytes, int width, int first, int last, size_t logical_base) {
        flash_live::runs(live_period_h_, width, logical_base + first, (size_t)(last-first),
            [&](size_t, size_t physical, size_t n) { out.push_back({storage + physical*bytes, n*bytes}); });
    };
    for (const KvBase& k : kvb_) {
        field((char*)k.hm->hk, NKV*kv_hb(), p_.own, r0, r1, logical);
        field((char*)k.hm->hv, NKV*kv_hb(), p_.own, r0, r1, logical);
        field((char*)k.hm->hks, NKV*kv_sf()*sizeof(float), p_.own, r0, r1, logical);
        field((char*)k.hm->hvs, NKV*kv_sf()*sizeof(float), p_.own, r0, r1, logical);
        field((char*)k.ikb, IDX_HD*sizeof(bf16), p_.own/IDX_R, r0/IDX_R, (r1+IDX_R-1)/IDX_R, logical/IDX_R);
    }
}
void Worker::live_restore(const char* source, int base, int a, int z, int upto, bool initialize) {
    CHECK(base >= 0 && base % live_stride_ == 0, "live prefix: unaligned virtual slot base");
    const int slot = base/live_stride_;
    CHECK(slot < (int)slots_.size(), "live prefix: virtual slot out of range");
    const int begin = std::max(a, initialize ? 0 : live_shared_h_[slot]*p_.period);
    const int stop = std::min(z, upto);
    if (stop <= begin) return;
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaDeviceSynchronize());
    const int r0=kv_rows(a), r1=kv_rows(z), rb=kv_rows(begin), re=kv_rows(stop);
    const size_t logical=(size_t)base*p_.own;
    size_t offset=0;
    auto field = [&](char* storage, size_t bytes, int width, int first, int last,
                     int wanted_first, int wanted_last, size_t logical_base, bool host) {
        flash_live::runs(live_period_h_, width, logical_base+wanted_first, (size_t)(wanted_last-wanted_first),
            [&](size_t off, size_t physical, size_t n) {
                const char* src=source+offset+((size_t)(wanted_first-first)+off)*bytes;
                char* dst=storage+physical*bytes;
                if (host) memcpy(dst,src,n*bytes);
                else CUDA_CHECK(cudaMemcpyAsync(dst,src,n*bytes,cudaMemcpyHostToDevice,st_));
            });
        offset+=(size_t)(last-first)*bytes;
    };
    for (const KvBase& k : kvb_) {
        field((char*)k.hm->hk,NKV*kv_hb(),p_.own,r0,r1,rb,re,logical,true);
        field((char*)k.hm->hv,NKV*kv_hb(),p_.own,r0,r1,rb,re,logical,true);
        field((char*)k.hm->hks,NKV*kv_sf()*sizeof(float),p_.own,r0,r1,rb,re,logical,true);
        field((char*)k.hm->hvs,NKV*kv_sf()*sizeof(float),p_.own,r0,r1,rb,re,logical,true);
        field((char*)k.ikb,IDX_HD*sizeof(bf16),p_.own/IDX_R,r0/IDX_R,(r1+IDX_R-1)/IDX_R,
              rb/IDX_R,(re+IDX_R-1)/IDX_R,logical/IDX_R,false);
    }
    live_inval(base,begin,stop);
    CUDA_CHECK(cudaStreamSynchronize(st_));
}
void Worker::live_inval(int base, int t0, int t1) {
    const int first=kv_rows(t0)/IDX_R, last=(kv_rows(t1)+IDX_R-1)/IDX_R;
    if (last<=first) return;
    flash_live::runs(live_period_h_,p_.own/IDX_R,(size_t)base*p_.own/IDX_R+first,(size_t)(last-first),
        [&](size_t,size_t physical,size_t count) {
            for (int li=0;li<=LAYERS;++li) {
                if (li==LAYERS && !has_mtp_) break;
                Layer& layer=li<LAYERS?L_[li]:mtp_;
                if (layer.kvp.ctl) k_kv_inval<<<std::min<size_t>(256,(count+255)/256),256,0,st_>>>(layer.kvp,(int)physical,(int)(physical+count));
            }
        });
}
void Worker::live_move(int from, int to, int ntok) {
    if (from==to || ntok<=0) return;
    CUDA_CHECK(cudaSetDevice(dev_));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK(to%live_stride_==0 && live_shared_h_.at(to/live_stride_)==0,
          "live prefix: copying into an immutable prefix is forbidden");
    std::vector<Piece> src,dst;
    live_pieces(from,0,ntok,src);
    live_pieces(to,0,ntok,dst);
    size_t si=0,di=0,so=0,doff=0;
    while(si<src.size() && di<dst.size()) {
        const size_t n=std::min(src[si].n-so,dst[di].n-doff);
        // RAM pieces and device indexer pieces have matching canonical field boundaries.
        cudaPointerAttributes attr{};
        CUDA_CHECK(cudaPointerGetAttributes(&attr,src[si].p+so));
        if(attr.type==cudaMemoryTypeHost) memmove(dst[di].p+doff,src[si].p+so,n);
        else CUDA_CHECK(cudaMemcpyAsync(dst[di].p+doff,src[si].p+so,n,cudaMemcpyDeviceToDevice,st_));
        so+=n; doff+=n;
        if(so==src[si].n) {++si;so=0;}
        if(doff==dst[di].n) {++di;doff=0;}
    }
    CHECK(si==src.size() && di==dst.size(),"live prefix: move byte-layout mismatch");
    live_inval(to,0,ntok);
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

size_t Worker::kv_bytes(int t0, int t1) const {
    const int r0 = kv_rows(t0), r1 = kv_rows(t1);
    if (r1 <= r0) return 0;
    const size_t rows = (size_t)(r1 - r0);
    const size_t blocks = (size_t)((r1 + IDX_R - 1) / IDX_R - r0 / IDX_R);
    return kvb_.size() * (rows * NKV * 2 * (kv_hb() + kv_sf() * sizeof(float)) +
                          blocks * IDX_HD * sizeof(bf16));
}
