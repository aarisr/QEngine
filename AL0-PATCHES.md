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

Dropped: 0024 (pinned/split archives to disk). Upstream fixed the same problem in 1.6.0 (#47) without a new file format.

Already upstream from our reports: #40 and #41 (1.5.1), #47 (1.6.0).
