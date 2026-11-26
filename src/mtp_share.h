// [q27-mba] 27B: how the sequences of a batched step share its verification rows for their MTP drafts
// (QE_MTP_BATCH_ADAPT), header-only so that tests/mtp_share_unit.cpp checks the decision on the CPU.
//
// A batched step gave every sequence the same number of MTP drafts: min(Engine::next_k(), rows / n - 1), 2 at 7..10
// sequences with the 24-row budget. next_k() is one count for the whole engine, and the rows a 7-sequence step leaves
// over (24 - 21) were not used. Conversations differ: reasoning text accepts few drafts, code and quoted text many, so
// with an even split the second draft of one sequence is verified at 10 % while the third of another, never made,
// would be accepted at 60 %. Here the same rows go where they are expected to be accepted:
//  - per slot and depth the share of verified MTP drafts the model accepted (qe_dfa::Rates, the block drafts' rule),
//    with the engine's batched MTP counts as the prior of a slot that has seen little;
//  - the step's drafts (the even split's total plus the rows it leaves over) are handed out one at a time to the
//    sequence whose next draft is most likely accepted (the gain of a sequence's d-th draft is P(drafts 1..d accepted):
//    it falls with d, so this greedy order is the best split); a draft expected below `floor` tokens is not verified;
//  - a sequence may get up to `extra` drafts more than the even split's cap; the step then makes that many MTP drafts
//    for every sequence, which costs one draft pass per extra depth: a depth is made only when the split it allows is
//    worth `pass` more of the step's tokens per pass (and a depth nobody would use is not made);
//  - every sequence keeps `kmin` drafts (1) when they are worth the floor: the split favours the predictable text, it
//    does not take a conversation's only draft away;
//  - a share stays unless another split is worth `hyst` more (every new set of row counts is a new CUDA graph), the
//    split is planned again every `period` launches or when the step's sequences change, and one sequence in turn
//    verifies one draft more than its share for a period, so a depth that was given up is tried again.
// Output is unchanged: a sequence that verifies fewer drafts emits fewer tokens of the same sequence per step.
#pragma once
#include "df_adapt.h"

namespace qe_mba {

constexpr int kDepth = qe_dfa::kDepth;  // = kMaxSpecK
constexpr int kMaxSeq = 32;             // = kMaxSegs (sequences of a batched step)
using qe_dfa::Rates;

// everything a slot keeps: its MTP acceptance per depth in batched steps, its share of the last plan (-1: none yet)
struct Slot {
    Rates r;
    int k = -1;
};

// g.m[i][d]: P(sequence i accepts its drafts 1..d) = the expected tokens its d-th draft adds to the step
struct Gain {
    double m[kMaxSeq][kDepth + 2];
};

inline void gains(const Rates* const* slot, const Rates& all, int n, Gain& g) {
    for (int i = 0; i < n; i++) {
        g.m[i][0] = 1;
        for (int d = 1; d <= kDepth; d++) g.m[i][d] = g.m[i][d - 1] * qe_dfa::rate(*slot[i], all, d);
        g.m[i][kDepth + 1] = 0;
    }
}

// Shares `drafts` verification rows among n sequences: k[i] <= lim[i] drafts for sequence i. prev[i]: its share of the
// last plan (< 0: none), kept unless another sequence's draft is worth `hyst` more. A draft expected below `floor`
// tokens is not handed out. explore (>= 0): that sequence verifies one draft more than its share (taken from the
// cheapest draft of another sequence when the budget is used up). kmin: drafts every sequence gets first when they are
// worth the floor. Returns the expected accepted drafts of the split.
inline double share(const Gain& g, int n, const int* lim, const int* prev, int drafts, double floor, double hyst, int explore,
                    int* k, int kmin = 0) {
    for (int i = 0; i < n; i++) k[i] = 0;
    int left = drafts;
    for (int d = 1; d <= kmin && d <= kDepth; d++)
        for (int i = 0; i < n && left > 0; i++)
            if (k[i] == d - 1 && d <= lim[i] && g.m[i][d] >= floor) k[i]++, left--;
    while (left > 0) {
        int bi = -1;
        double best = 0;
        for (int i = 0; i < n; i++) {
            if (k[i] >= lim[i] || k[i] >= kDepth) continue;
            const double w = g.m[i][k[i] + 1] * (k[i] + 1 <= prev[i] ? 1.0 + hyst : 1.0);
            if (w >= floor && w > best) best = w, bi = i;
        }
        if (bi < 0) break;
        k[bi]++;
        left--;
    }
    if (explore >= 0 && explore < n && k[explore] < lim[explore] && k[explore] < kDepth) {
        bool room = left > 0;
        if (!room) {  // the draft least likely accepted among the other sequences' last ones makes room
            int wi = -1;
            for (int i = 0; i < n; i++)
                if (i != explore && k[i] > kmin && (wi < 0 || g.m[i][k[i]] < g.m[wi][k[wi]])) wi = i;
            if (wi >= 0) k[wi]--, room = true;
        }
        if (room) k[explore]++;
    }
    double e = 0;
    for (int i = 0; i < n; i++)
        for (int d = 1; d <= k[i]; d++) e += g.m[i][d];
    return e;
}

// The MTP drafts a step makes per sequence (1..kmax) so that the next step can share its rows: the depth limit whose
// best split gives the most expected tokens after `pass` (the cost of one more draft pass, as a share of the step) per
// depth; prevd (the last choice, < 1: none) stays unless the best is worth 1 % more.
inline int depth(const Gain& g, int n, const int* prev, int drafts, double floor, double hyst, int kmax, double pass, int prevd,
                 int kmin = 0) {
    kmax = std::max(1, std::min(kmax, kDepth));
    int lim[kMaxSeq], k[kMaxSeq], bd = 1;
    double net[kDepth + 1] = {}, best = 0;
    for (int d = 1; d <= kmax; d++) {
        for (int i = 0; i < n; i++) lim[i] = d;
        int used = 0;
        const double e = share(g, n, lim, prev, drafts, floor, hyst, -1, k, kmin);
        for (int i = 0; i < n; i++) used = std::max(used, k[i]);
        if (used < d && d > 1) break;  // nobody would verify a d-th draft (nor a deeper one)
        net[d] = (n + e) / (1.0 + pass * (d - 1));
        if (net[d] > best) best = net[d], bd = d;
    }
    if (prevd >= 1 && prevd <= kmax && net[prevd] > 0 && best <= net[prevd] * 1.01) return prevd;
    return bd;
}

}  // namespace qe_mba
