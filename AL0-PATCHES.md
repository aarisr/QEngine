# al0 patch series on QEngine

This branch is upstream [bpx03/QEngine](https://github.com/bpx03/QEngine) **v1.6.2** plus our patches, one commit per patch, in numeric order. We run this exact series in production: Qwen3.8-27B INT4 on RTX 3090s, two TP2 instances behind a scheduler.

Upstream is the reference. We send a patch upstream only when it is proven:
- it applies on its own;
- it has its own test or reproduction;
- a performance change has before/after numbers.
Everything else stays here.

| Patch | What it does | Status |
|---|---|---|
| 0001 | flash: shared, read-only prefix objects across engine instances | fork-only (multi-instance) |
| 0011 | flash: publish shared prefixes asynchronously | fork-only (multi-instance) |
| 0012 | server: request classes (interactive / background); background cannot pause interactive | cooking (tied to our request header; needs a generic interface first) |
| 0013 | tools: parity and qtrace replay tooling | fork-only (internal tooling) |
| 0014 | cuda: report the first failing CUDA operation | cooking (debug aid; could become an opt-in flag) |
| 0015 | flash: widen KV capacity and row-block arithmetic to 64 bit | ready, unposted (Flash-Next; overflow only near INT_MAX, not seen in practice) |
| 0016 | build: host build flags (parallelism cap, spill diagnostics) | fork-only (host-specific) |
| 0017 | flash: slot-state ownership window unit test | fork-only (tests 0018) |
| 0018 | flash: share a live prefix between instances | fork-only (multi-instance) |
| 0019 | flash: live-prefix compile fixes | empty on 1.6.2 (folded into 0011 and 0018) |
| 0023 | 27b: lift the register cap on the default i8v prefill tile (40-byte spill -> 0 on sm_86) | deployed 2026-11-27; no regression in the full-series 60-min arm comparison (per-active decode D +5.8% vs A); patch-specific prefill effect not isolated, so no upstream PR yet |
| 0025 | server: log the x-session-hash request header | fork-only (our gateway header) |
| 0026 | 27b: verify only as many DFlash2 drafts as pay (`QE_DF_ADAPT`) | deployed 2026-11-27; full-series arm rollout passed production comparison; `QE_DF_BATCH=0` in both arms (single-sequence DFlash2 only), adaptation effect not isolated, so no upstream PR yet |
| 0027 | 27b: batched MTP steps share rows by each conversation's acceptance | cooking (needs its own A/B) |
| 0028 | metrics: Prometheus series for speculative decoding, prompt-cache source, thinking tokens | cooking (better as an issue first; upstream has its own stats path) |
| 0029 | server: reserve N slots for background requests (`QE_BG_RESERVED_SEQS`, default 0 = off); a waiting background request may park the newest interactive seq | fork-only for now (tied to our request-class header; candidate for a generic proposal later) |
| 0030 | GEMM: reuse tables across builds keyed by model, GPU/card count, kernel-set hash and runtime setup; migrate only matching 1.6.2 legacy tables | QEngine build + CPU key/header/device-set tests passed; live no-retune confirmation awaits planned arm restart |
| 0031 | 27b: idle VRAM poll gate (`QE_IDLE_POLL_S`, default 30, 0 = old behaviour): the server's 1/s `mem_guard` skips `vram_tick` (cudaMemGetInfo per card, NVML) while no sequence or prefill is active and the pool is not shrunk; steps and prefills still tick | cooking, unproven on production, ships as a bet (hypothesis: idle driver traffic holds the TP3 cards at P2; mechanism not measured); also gates the metrics NVML 1/s poll while no job is open; CPU test `idle_poll_unit`; revert with `QE_IDLE_POLL_S=0` |
| 0032 | event-driven GPU polling with rate gates: `mem_guard`/`vram_tick` makes no GPU call while idle (admission in `start_begin`, KV growth in `ensure_cap`, every step and prefill chunk tick, `vram_tick` keeps its 200 ms gate, a shrunk pool keeps its cadence); `QE_IDLE_POLL_S` is now an optional idle heartbeat, default 0 = none (0031 default was 30); metrics NVML is read on scrape only, cached `QE_NVML_CACHE_S` (5 s) while busy, never while no job is open; CPU tests `idle_poll_unit`, `tests/test_metrics_nvml_gate.py` | cooking, unproven on production, ships as a bet (revert: `QE_IDLE_POLL_S=30` and `QE_NVML_CACHE_S=0`, or drop the patch) |
| 0035 | 27b: `attn_dec8d` (K4 4-bit cache, 3 row tiles, `attn_dec8d_seg_kernel<3, true>`) without register spills: Q fragments reloaded from `q` per tile instead of 48 live registers, the P x V-scale fragments built per 32-dim group from shared memory (24 live instead of 72), softmax state (`mx`, `l`, `qpos`) kept only for the row tile the warp owns; stack 144 B -> 16 B, local-memory traffic 35.7 MB ld + 35.9 MB st -> 3.5 MB + 0.46 MB per launch (ncu, 3 sequences, 10K context); outputs bit-identical (`qe_check_attn_seg_kv4`, the 4-bit twin of `qe_check_attn_seg`) | cooking, measured on TP1 (GPU 2) only; kernel time unchanged within noise (104-110 us vs 107-112 us), DRAM write 17.2 -> 5.3 MB; not deployed |

Dropped: 0024 (pinned/split archives to disk). Upstream fixed the same problem in 1.6.0 (#47) without a new file format.

Already upstream from our reports: #40 and #41 (1.5.1), #47 (1.6.0).
