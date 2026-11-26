// [q27-dfa] 27B: how many of the DFlash2 block drafter's 7 drafts a step verifies (QE_DF_ADAPT), header-only so that
// tests/df_adapt_unit.cpp checks the decision on the CPU.
//
// A block step always got all 7 drafts verified (8 rows per sequence), whatever the text. On long agentic contexts with
// mostly reasoning the model accepts 1.6..2.9 of them (measured 2026-11-23: 23..41 % of the verified drafts), so most
// rows of a step are computed and thrown away. The MTP drafts have had a draft count that follows the acceptance since
// 1.0 (Engine::next_k); block drafts were left out of it. This is the same rule for block drafts, per conversation:
//  - per depth i (1..7) the share of steps that accepted draft i when drafts 1..i-1 were accepted (decayed counts, as
//    Engine::observe), per slot, with the engine-wide counts as the prior of a slot that has seen little;
//  - verify the k drafts with the most expected tokens per millisecond: (1 + sum of P(accept >= i), i <= k) /
//    (c0 + c1 * k); in a batched step of n sequences the fixed part is shared (c0 / n);
//  - the cost of a row (c1) is measured where it cannot be confounded with the context length: the same slot's step
//    time at k drafts against a step of it at another count (the probe steps below, and every change of k);
//  - every `period` steps a slot verifies one draft more than its pick (a probe), so a depth that was given up is tried
//    again and the count grows when the text turns predictable (code after reasoning);
//  - a slot keeps its count unless another one is worth `hyst` more: every new row count is a new CUDA graph.
// The drafter still makes 7 drafts (its cost does not depend on the count); only the verification shrinks. Output is
// unchanged: a step that verifies fewer drafts emits fewer tokens of the same sequence.
#pragma once
#include <algorithm>
#include <cmath>

namespace qe_dfa {

constexpr int kDepth = 7;  // the block drafter's drafts per step (kDflDrafts)

// decayed acceptance counts per depth: depth i is tried only if drafts 1..i-1 were accepted, and only tried depths
// decay (a depth not verified lately keeps its last estimate)
struct Rates {
    double num[kDepth + 1] = {}, den[kDepth + 1] = {};
    void observe(int kv, int accepted, double decay = 0.95) {
        kv = std::min(kv, kDepth);
        for (int i = 1; i <= kv; i++) {
            num[i] *= decay;
            den[i] *= decay;
            den[i] += 1;
            if (i > accepted) break;
            num[i] += 1;
        }
    }
};

// the conditional acceptance of depth i: the slot's counts over the engine's, over an optimistic prior that decays
// slowly with the depth (Engine::next_k's), so depths nobody has tried get tried
inline double rate(const Rates& slot, const Rates& all, int i) {
    const double prior = 0.9 * std::pow(0.97, i - 1);
    const double g = (all.num[i] + 3.0 * prior) / (all.den[i] + 3.0);
    return (slot.num[i] + 4.0 * g) / (slot.den[i] + 4.0);
}

// expected tokens of a step that verifies k drafts (the token after the last accepted draft included)
inline double tokens(const Rates& slot, const Rates& all, int k) {
    double e = 1, prod = 1;
    for (int i = 1; i <= k; i++) {
        prod *= rate(slot, all, i);
        e += prod;
    }
    return e;
}

inline double score(const Rates& slot, const Rates& all, int k, double c0, double c1) {
    return tokens(slot, all, k) / (c0 + c1 * k);
}

// the drafts to verify, kmin..kmax: the best tokens per millisecond; `prev` (the slot's last pick, < 0: none) stays
// unless the best is worth `hyst` more
inline int pick(const Rates& slot, const Rates& all, double c0, double c1, int kmin, int kmax, int prev, double hyst = 0.03) {
    kmax = std::max(1, std::min(kmax, kDepth));
    kmin = std::max(1, std::min(kmin, kmax));
    c0 = std::max(c0, 0.1);
    c1 = std::max(c1, 0.01);
    int bk = kmin;
    double best = 0;
    for (int k = kmin; k <= kmax; k++) {
        const double s = score(slot, all, k, c0, c1);
        if (s > best) best = s, bk = k;
    }
    if (prev >= kmin && prev <= kmax && best <= score(slot, all, prev, c0, c1) * (1.0 + hyst)) return prev;
    return bk;
}

// a probe step: one draft more than the pick, every `period` steps of the slot (0: never)
inline bool probe_due(unsigned long long tick, int period) { return period > 0 && tick % period == (unsigned)period - 1; }

// the measured cost of one more verified row (ms): decayed mean of the samples, within 0.2..3 ms as Engine::next_k's
struct RowCost {
    double sum = 0, w = 0;
    void add(double ms_per_row) {
        if (!(ms_per_row > -2.0 && ms_per_row < 6.0)) return;  // (a step that waited for something else)
        sum = 0.9 * sum + ms_per_row;
        w = 0.9 * w + 1;
    }
    double get(double def) const { return w < 3 ? def : std::min(3.0, std::max(0.2, sum / w)); }
    // a slot's own samples over the engine's value (a long context pays more per row than a short one)
    double get_over(double engine) const { return std::min(3.0, std::max(0.2, (sum + 2.0 * engine) / (w + 2.0))); }
};

// a slot's block-step time at its current count (`base` ms at `k` drafts, a fast mean: the context grows); a step at
// another count gives one sample of the row cost against it
struct StepTime {
    double base = 0;
    int k = -1, n = 0;
    // probe: a one-off count (the base stays); else a step at a new count restarts the base there.
    // Returns true and the sample when this step measured the row cost.
    bool add(int kv, double ms, bool probe, double* sample) {
        if (!(ms > 0)) return false;
        if (kv == k) {
            if (n > 0 && ms > 2.0 * base + 2.0) return false;  // outlier (host time in the step)
            base = n > 0 ? 0.8 * base + 0.2 * ms : ms;
            n++;
            return false;
        }
        const bool got = k >= 0 && n >= 4 && ms <= 2.0 * base + 2.0;
        if (got) *sample = (ms - base) / (kv - k);
        if (!probe) base = ms, k = kv, n = 1;
        return got;
    }
    bool known() const { return n >= 4; }
    // the fixed part of the slot's step: its time at k drafts minus the k rows
    double c0(double c1) const { return std::max(1.0, base - c1 * k); }
};

// everything a slot keeps
struct Slot {
    Rates r;
    RowCost row;
    StepTime t;
    int k[2] = {kDepth, kDepth};  // the last pick: single-sequence steps, batched steps (all 7 until the counts say less)
    unsigned long long tick = 0;
};

}  // namespace qe_dfa
