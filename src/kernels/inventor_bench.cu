// Microbench hooks for experiments (not used by the engine): an AR-latency stand-in
// (one block spinning on the global timer) and L2 prefetch variants of the next GEMV's weights.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "gemv.cuh"

namespace {

__global__ void spin_kernel(unsigned long long ns) {
    if (threadIdx.x) return;
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

// prefix pattern (like prefetch_l2_kernel) with a configurable grid
__global__ void pf_prefix_kernel(const char* __restrict__ p, size_t bytes) {
    for (size_t off = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 128; off < bytes;
         off += (size_t)gridDim.x * blockDim.x * 128)
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p + off));
}

// spread pattern: the first `per` bytes of every one of `nseg` equal segments of the matrix (each
// segment = one warp's K slice of one tile in the GEMV's tile-major layout), so every warp of the
// consumer GEMV finds the start of its slice in L2
__global__ void pf_spread_kernel(const char* __restrict__ p, size_t seg_bytes, int nseg, size_t per) {
    const size_t lines = per / 128;
    const size_t total = lines * nseg;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t s = i / lines, l = i % lines;
        asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p + s * seg_bytes + l * 128));
    }
}

// plain loads (data discarded) instead of prefetch instructions: 8 independent 16-byte loads in
// flight per thread per iteration
__global__ void pf_load_kernel(const uint4* __restrict__ p, size_t n16) {
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += 8 * stride) {
        uint4 v[8];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = i + j * stride < n16 ? __ldcg(p + i + j * stride) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < 8; j++) acc ^= v[j].x;
    }
    if (acc == 0x12345678u) asm volatile("trap;");
}

// spread pattern with real loads (one 16-byte load per 128-byte line)
__global__ void pf_spread_ld_kernel(const char* __restrict__ p, size_t seg_bytes, int nseg, size_t per) {
    const size_t lines = per / 128;
    const size_t total = lines * nseg;
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += 4 * stride) {
        uint4 v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const size_t ii = i + j * stride;
            v[j] = ii < total ? __ldcg(reinterpret_cast<const uint4*>(p + (((ii / lines) * seg_bytes + (ii % lines) * 128) & ~(size_t)127))) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) acc ^= v[j].x;
    }
    if (acc == 0x12345678u) asm volatile("trap;");
}

}  // namespace

void flag_set_after(unsigned* flag, unsigned v, double us, cudaStream_t st, int blocks, int threads,
                    const void* src = nullptr, void* dst = nullptr, size_t bytes = 0);

extern "C" {
void qe_gemv_w4h_early(const void* q, const void* s, int n, int k, const __half* xh, float* y, int m, int ldx,
                       int depth, const unsigned* flag, int nw, void* st) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k, false};
    gemv_w4h_early(w, xh, ldx, y, n, m, depth, flag, nw, (cudaStream_t)st);
}
void qe_flag_set_after(unsigned* flag, unsigned v, double us, void* st, int blocks, int threads) { flag_set_after(flag, v, us, (cudaStream_t)st, blocks, threads); }
void qe_flag_copy_after(unsigned* flag, unsigned v, double us, void* st, int blocks, int threads, const void* src, void* dst, size_t bytes) {
    flag_set_after(flag, v, us, (cudaStream_t)st, blocks, threads, src, dst, bytes);
}
void qe_spin_ns(unsigned long long ns, void* st) { spin_kernel<<<1, 32, 0, (cudaStream_t)st>>>(ns); }
void qe_prefetch_engine(const void* p, size_t bytes, void* st) { prefetch_l2(p, bytes, (cudaStream_t)st); }
void qe_prefetch_prefix(const void* p, size_t bytes, int blocks, int threads, void* st) {
    pf_prefix_kernel<<<blocks, threads, 0, (cudaStream_t)st>>>((const char*)p, bytes);
}
void qe_prefetch_spread(const void* p, size_t seg_bytes, int nseg, size_t per, int blocks, int threads, void* st) {
    pf_spread_kernel<<<blocks, threads, 0, (cudaStream_t)st>>>((const char*)p, seg_bytes, nseg, per);
}
void qe_prefetch_spread_ld(const void* p, size_t seg_bytes, int nseg, size_t per, int blocks, int threads, void* st) {
    pf_spread_ld_kernel<<<blocks, threads, 0, (cudaStream_t)st>>>((const char*)p, seg_bytes, nseg, per);
}
void qe_prefetch_load(const void* p, size_t bytes, int blocks, int threads, void* st) {
    pf_load_kernel<<<blocks, threads, 0, (cudaStream_t)st>>>((const uint4*)p, bytes / 16);
}
int qe_pick_warps(int tiles, int units) { return pick_warps(tiles, units, true); }
void qe_gemv_w4hx_s(const void* q, const void* s, int n, int k, const __half* xh, float* y, int m, int ldx, void* st) {
    W4 w{(const uint4*)q, (const bf16*)s, n, k, false};
    gemv_w4h(w, xh, ldx, y, n, m, (cudaStream_t)st);
}
}

// ---- decode attention bit-identity check: the qe_bench_attn_seg setup with
// pseudo-random q / K / V (fp8 bytes); returns an FNV-1a hash of every segment's bf16 output
#include "ops.cuh"
namespace {
__global__ void fill_rand_kernel(uint8_t* p, size_t n, uint32_t seed, uint8_t mask) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = (uint8_t)x & mask;
    }
}
__global__ void fill_bf16_kernel(bf16* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = __float2bfloat16(((int)(x % 2001) - 1000) * 0.002f);
    }
}
}  // namespace
extern "C" unsigned long long qe_check_attn_seg(int nseg, int ctx, int m) {
    AttnShape s{};
    s.nq = 8; s.nkv = 2;
    s.kv_qb[0] = 0; s.kv_qe[0] = 6; s.kv_qb[1] = 6; s.kv_qe[1] = 8;
    for (int h = 0; h < 8; h++) s.q2kv[h] = h < 6 ? 0 : 1;
    s.max_t = ctx + 64; s.fp8 = 1; s.ks = 0.02f; s.vs = 0.02f; s.dchunk_min = -1;
    const int ld = 8 * 512 + 2 * 512;
    const size_t kvb = (size_t)2 * s.max_t * 256;
    const size_t wf = attn_work_floats(s);
    AttnSegs g{};
    g.n = nseg; g.max_m = m;
    std::vector<void*> bufs;
    auto al = [&](size_t b, int v) { void* p; CUDA_CHECK(cudaMalloc(&p, b)); CUDA_CHECK(cudaMemset(p, v, b)); bufs.push_back(p); return p; };
    bf16* qn_w = (bf16*)al(512, 0); bf16* kn_w = (bf16*)al(512, 0); float* invf = (float*)al(1024, 0);
    { const float one = 1.f; CUDA_CHECK(cudaMemcpy(invf + 32, &one, 4, cudaMemcpyHostToDevice)); }  // q factor (native)
    for (int i = 0; i < nseg; i++) {
        AttnSeg& c = g.c[i];
        c.proj = (float*)al((size_t)m * ld * 4, 0);
        fill_bf16_kernel<<<64, 256>>>((bf16*)c.proj, (size_t)m * ld * 2, 77u + i);  // gates / q / k / v of the new rows
        c.q = (bf16*)al((size_t)m * 8 * 256 * 2, 0);
        fill_bf16_kernel<<<64, 256>>>(c.q, (size_t)m * 8 * 256, 11u + i);
        c.kc = al(kvb, 0); c.vc = al(kvb, 0);
        fill_rand_kernel<<<256, 256>>>((uint8_t*)c.kc, kvb, 1234u + i, 0x77);  // finite e4m3 (no NaN pattern)
        fill_rand_kernel<<<256, 256>>>((uint8_t*)c.vc, kvb, 4321u + i, 0x77);
        const int p0 = ctx - m;
        int* pos = (int*)al(4, 0);
        CUDA_CHECK(cudaMemcpy(pos, &p0, 4, cudaMemcpyHostToDevice));
        c.pos = pos; c.rdelta = nullptr; c.m = m; c.max_t = s.max_t;
        c.work = (float*)al(wf * 4, 0);
        c.out = (bf16*)al((size_t)m * 8 * 256 * 2, 0);
        c.xh = nullptr;
    }
    attention_prefill_init();
    attention_decode_tc_init();
    const int G = 6 * m;
    attention_dec8_seg(g, s, G, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long h = 1469598103934665603ull;
    for (int i = 0; i < nseg; i++) {
        // partials and (m, l) of every split: what the combine reads
        std::vector<uint8_t> hb(wf * 4);
        CUDA_CHECK(cudaMemcpy(hb.data(), g.c[i].work, wf * 4, cudaMemcpyDeviceToHost));
        const int ns = kAttnSplits;
        // only the entries the kernel writes: [kv][split][row < G][256] and (m, l)
        const float* w = reinterpret_cast<const float*>(hb.data());
        for (int kv = 0; kv < 2; kv++)
            for (int sp = 0; sp < ns; sp++)
                for (int r = 0; r < (kv == 0 ? 6 : 2) * m; r++) {
                    const float* ml = w + (size_t)s.nkv * kAttnSplits * kMaxGroup * 256 + ((size_t)(kv * kAttnSplits + sp) * kMaxGroup + r) * 2;
                    if (ml[0] == -FLT_MAX) continue;  // split without keys
                    const uint8_t* b = reinterpret_cast<const uint8_t*>(w + ((size_t)(kv * kAttnSplits + sp) * kMaxGroup + r) * 256);
                    for (int x = 0; x < 1024; x++) h = (h ^ b[x]) * 1099511628211ull;
                    const uint8_t* bm = reinterpret_cast<const uint8_t*>(ml);
                    for (int x = 0; x < 8; x++) h = (h ^ bm[x]) * 1099511628211ull;
                }
    }
    for (void* p : bufs) cudaFree(p);
    return h;
}

// ---- multi-row INT4 GEMV bit-identity check and timing: pseudo-random packed weights, scales and XH
// activations at (n, k, m), run through the production dispatch gemv_w4h (gemv_x_init first, so the tile-pair
// and wide kernels are reachable). Returns an FNV-1a hash of y (m x n fp32 bits); *us gets the mean time of
// `iters` back-to-back launches with the weights cycled over a 256 MiB pool so each launch reads from DRAM.
namespace {
__global__ void fill_u32_kernel(uint32_t* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = x;
    }
}
__global__ void fill_half_kernel(__half* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = __float2half(((int)(x % 2001) - 1000) * 0.001f);
    }
}
}  // namespace

extern "C" unsigned long long qe_check_gemv(int n, int k, int m, int iters, float* us) {
    gemv_x_init();
    const size_t wbytes = (size_t)n * k / 2;
    const size_t sb = (size_t)(n / 16) * (k / 128) * 16 * 2;  // bf16 [tile][group][16]
    const int ldx = (k + 63) / 64 * 64;
    const int copies = (int)std::max<size_t>(1, ((size_t)256 << 20) / wbytes);
    uint8_t *q, *s; __half* xh; float* y;
    CUDA_CHECK(cudaMalloc(&q, wbytes * copies)); CUDA_CHECK(cudaMalloc(&s, sb * copies));
    CUDA_CHECK(cudaMalloc(&xh, (size_t)m * ldx * 2 + 256)); CUDA_CHECK(cudaMalloc(&y, (size_t)m * n * 4));
    fill_u32_kernel<<<1024, 256>>>((uint32_t*)q, wbytes * copies / 4, 99u);
    fill_half_kernel<<<256, 256>>>((__half*)s, sb * copies / 2, 5u);   // scale bits: bf16 of small magnitude after the cast below
    fill_half_kernel<<<256, 256>>>(xh, (size_t)m * ldx, 31u);
    CUDA_CHECK(cudaDeviceSynchronize());
    // scales must be sane bf16: map the fp16 values to bf16 in place (|v| <= 1)
    {
        std::vector<__half> hs(sb * copies / 2);
        CUDA_CHECK(cudaMemcpy(hs.data(), s, sb * copies, cudaMemcpyDeviceToHost));
        std::vector<bf16> bs(hs.size());
        for (size_t i = 0; i < hs.size(); i++) bs[i] = __float2bfloat16(0.01f * __half2float(hs[i]));
        CUDA_CHECK(cudaMemcpy(s, bs.data(), sb * copies, cudaMemcpyHostToDevice));
    }
    cudaStream_t st; CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    auto run = [&](int c) {
        W4 w{(const uint4*)(q + wbytes * c), (const bf16*)(s + sb * c), n, k, false};
        gemv_w4h(w, xh, ldx, y, n, m, st);
    };
    run(0);
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<float> hy((size_t)m * n);
    CUDA_CHECK(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
    unsigned long long h = 1469598103934665603ull;
    for (float f : hy) { uint32_t b; memcpy(&b, &f, 4); for (int x = 0; x < 4; x++) h = (h ^ ((b >> (8 * x)) & 255)) * 1099511628211ull; }
    if (iters > 0) {
        // 20 launches per graph (host launch cost is not what is measured), `iters` launches in all
        cudaGraph_t gr; cudaGraphExec_t ge;
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < 20; i++) run(i % copies);
        CUDA_CHECK(cudaStreamEndCapture(st, &gr));
        CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
        const int reps = std::max(1, iters / 20);
        // warm-up until the clocks have ramped (the card idles in P5 between runs), then the best of 5 timed passes
        for (int i = 0; i < 4000; i++) {  // ~2 s of load: the memory clock follows the load with a delay
            CUDA_CHECK(cudaGraphLaunch(ge, st));
            if (i % 64 == 63) CUDA_CHECK(cudaStreamSynchronize(st));
        }
        CUDA_CHECK(cudaStreamSynchronize(st));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f;
        for (int pass = 0; pass < 5; pass++) {
            cudaEventRecord(e0, st);
            for (int i = 0; i < reps; i++) CUDA_CHECK(cudaGraphLaunch(ge, st));
            cudaEventRecord(e1, st); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            best = std::min(best, ms * 1000.f / (reps * 20));
        }
        if (us) *us = best;
        cudaGraphExecDestroy(ge); cudaGraphDestroy(gr);
    }
    cudaStreamDestroy(st);
    cudaFree(q); cudaFree(s); cudaFree(xh); cudaFree(y);
    return h;
}
