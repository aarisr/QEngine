// 27B [q27-mba]: how the sequences of a batched step share its rows for their MTP drafts (src/mtp_share.h), on a model
// of the MTP head: no GPU. The engine call sites (Engine::mba_plan / mba_observe, launch_batch) use these functions.
#include "mtp_share.h"
#include <cstdio>

#define EXPECT(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)
static int fails = 0;
using namespace qe_mba;

static unsigned long long rng_state = 88172645463325252ull;
static double rnd() {  // xorshift64, [0, 1)
    rng_state ^= rng_state << 13, rng_state ^= rng_state >> 7, rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / 9007199254740992.0;
}
// drafts accepted of kv verified when every depth is accepted with probability p given the ones before
static int accept(int kv, double p) {
    int a = 0;
    while (a < kv && rnd() < p) a++;
    return a;
}

// rates that say exactly "p per depth" (no prior left)
static Rates exact(double p) {
    Rates r;
    for (int d = 1; d <= kDepth; d++) r.num[d] = 1e6 * p, r.den[d] = 1e6;
    return r;
}

static double value(const Gain& g, int n, const int* k) {
    double e = 0;
    for (int i = 0; i < n; i++)
        for (int d = 1; d <= k[i]; d++) e += g.m[i][d];
    return e;
}

// the best split by trying all of them (n <= 4)
static double brute(const Gain& g, int n, const int* lim, int drafts, int i, int* k) {
    if (i == n) return value(g, n, k);
    double best = -1;
    for (int c = 0; c <= lim[i] && c <= drafts; c++) {
        k[i] = c;
        best = std::max(best, brute(g, n, lim, drafts - c, i + 1, k));
    }
    return best;
}

struct Sim {
    double tok = 0, rows = 0, passes = 0;
    int kend[kMaxSeq] = {}, kd = 0;
};

// n sequences in batched steps for `steps` launches: the engine's loop (Engine::mba_plan: plan when the available
// drafts or the period's epoch change, verify, observe). p(i, step): sequence i's acceptance per depth at that step.
// cap: the even split's drafts per sequence, spare: the rows it leaves over. adaptive = false: the even split.
template <class P>
static Sim run(int n, int steps, int cap, int spare, int extra, int period, bool adaptive, P p, int kmin = 1,
               double floor = 0.10, double hyst = 0.10, double pass = 0.02) {
    Sim s;
    Slot slot[kMaxSeq];
    Rates all;
    int avail[kMaxSeq], kd = cap, lastd = 0;
    unsigned long long sig = 0;
    for (int i = 0; i < n; i++) avail[i] = cap;
    for (int t = 1; t <= steps; t++) {
        int k[kMaxSeq];
        for (int i = 0; i < n; i++) k[i] = std::min(avail[i], cap);
        if (adaptive) {
            const int kmax = std::min(kDepth, cap + extra), drafts = n * cap + spare;
            int lim[kMaxSeq], prev[kMaxSeq];
            const Rates* rs[kMaxSeq];
            const unsigned long long epoch = period > 0 ? t / period : 0;
            unsigned long long g2 = epoch * 1000003ull + 17;
            for (int i = 0; i < n; i++) lim[i] = std::min(avail[i], kmax), prev[i] = slot[i].k, rs[i] = &slot[i].r, g2 = g2 * 31 + lim[i];
            if (g2 != sig) {
                Gain g;
                int kk[kMaxSeq];
                gains(rs, all, n, g);
                share(g, n, lim, prev, drafts, floor, hyst, period > 0 ? (int)(epoch % n) : -1, kk, kmin);
                lastd = depth(g, n, prev, drafts, floor, hyst, kmax, pass, lastd, kmin);
                for (int i = 0; i < n; i++) slot[i].k = kk[i];
                sig = g2;
            }
            for (int i = 0; i < n; i++) k[i] = std::min(slot[i].k, lim[i]);
            kd = std::max(1, lastd);
        }
        int used = 0;
        for (int i = 0; i < n; i++) {
            const int a = accept(k[i], p(i, t));
            s.tok += a + 1;
            used += k[i] + 1;
            if (adaptive) slot[i].r.observe(k[i], a, 0.97), all.observe(k[i], a, 0.97);
            avail[i] = kd;
        }
        if (adaptive) EXPECT(used <= n * (cap + 1) + spare);  // never more rows than the step's budget
        s.rows += used;
        s.passes += kd;
    }
    for (int i = 0; i < n; i++) s.kend[i] = adaptive ? slot[i].k : cap;
    s.kd = kd;
    return s;
}

int main() {
    Rates none;
    // 1. equal sequences: the even split, and no deeper drafts are made than it uses
    {
        Rates r[8];
        const Rates* rs[8];
        for (int i = 0; i < 8; i++) r[i] = exact(0.7), rs[i] = &r[i];
        Gain g;
        gains(rs, none, 8, g);
        int lim[8], prev[8], k[8];
        for (int i = 0; i < 8; i++) lim[i] = 4, prev[i] = -1;
        const double e = share(g, 8, lim, prev, 16, 0.10, 0.10, -1, k);
        for (int i = 0; i < 8; i++) EXPECT(k[i] == 2);
        EXPECT(std::fabs(e - 8 * (0.7 + 0.49)) < 1e-3);
        EXPECT(depth(g, 8, prev, 16, 0.10, 0.10, 4, 0.02, 0) == 2);
        // 7 sequences in the 24-row budget: the 3 rows left over are used (3 third drafts at 0.34 tokens each pay a pass)
        EXPECT(std::fabs(share(g, 7, lim, prev, 14 + 3, 0.10, 0.10, -1, k) - (7 * 1.19 + 3 * 0.343)) < 1e-3);
        EXPECT(depth(g, 7, prev, 14 + 3, 0.10, 0.10, 4, 0.02, 0) == 3);
    }
    // 2. different sequences: the rows go to the drafts most likely accepted, within the budget and each limit
    {
        const double ps[8] = {0.25, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 0.92};
        Rates r[8];
        const Rates* rs[8];
        for (int i = 0; i < 8; i++) r[i] = exact(ps[i]), rs[i] = &r[i];
        Gain g;
        gains(rs, none, 8, g);
        int lim[8], prev[8], k[8], even[8], sum = 0;
        for (int i = 0; i < 8; i++) lim[i] = 4, prev[i] = -1, even[i] = 2;
        const double e = share(g, 8, lim, prev, 16, 0.10, 0.0, -1, k);
        for (int i = 0; i < 8; i++) {
            sum += k[i];
            EXPECT(k[i] >= 0 && k[i] <= 4);
        }
        EXPECT(sum <= 16);
        EXPECT(k[0] <= 1 && k[1] <= 1 && k[6] == 4 && k[7] == 4);
        for (int i = 1; i < 8; i++) EXPECT(k[i] >= k[i - 1]);
        const double e0 = value(g, 8, even);
        printf("8 sequences 0.25..0.92, 24 rows: even split %.2f tokens per step, shared %.2f (%+.1f %%), shares", 8 + e0, 8 + e,
               100.0 * (e - e0) / (8 + e0));
        for (int i = 0; i < 8; i++) printf(" %d", k[i]);
        printf("\n");
        EXPECT(e > e0 * 1.10);
        // every sequence keeps one draft (QE_MTP_BATCH_MIN=1, the default): still most of the gain
        const double e1 = share(g, 8, lim, prev, 16, 0.10, 0.0, -1, k, 1);
        printf("  with one draft kept per sequence: %.2f (%+.1f %%), shares", 8 + e1, 100.0 * (e1 - e0) / (8 + e0));
        for (int i = 0; i < 8; i++) {
            printf(" %d", k[i]);
            EXPECT(k[i] >= 1);
        }
        printf("\n");
        EXPECT(e1 > e0 * 1.08 && e1 <= e + 1e-9);
        // depth: 4 drafts are worth two more passes here; an expensive pass keeps the cap
        EXPECT(depth(g, 8, prev, 16, 0.10, 0.0, 4, 0.02, 0) == 4);
        EXPECT(depth(g, 8, prev, 16, 0.10, 0.0, 4, 0.50, 0) <= 2);
        // the available drafts bound a share (drafts are made one step before)
        for (int i = 0; i < 8; i++) lim[i] = 2;
        share(g, 8, lim, prev, 16, 0.10, 0.0, -1, k);
        for (int i = 0; i < 8; i++) EXPECT(k[i] <= 2);
        EXPECT(k[0] == 1 && k[7] == 2);  // (0.25^2 is below the floor: that row is not verified at all)
    }
    // 3. the greedy split is the best one (all splits tried, 4 sequences, random rates, no floor / hysteresis)
    for (int rep = 0; rep < 200; rep++) {
        Rates r[4];
        const Rates* rs[4];
        for (int i = 0; i < 4; i++) {
            for (int d = 1; d <= kDepth; d++) r[i].den[d] = 1e6, r[i].num[d] = 1e6 * (0.05 + 0.9 * rnd());
            rs[i] = &r[i];
        }
        Gain g;
        gains(rs, none, 4, g);
        int lim[4], prev[4] = {-1, -1, -1, -1}, k[4], kb[4];
        for (int i = 0; i < 4; i++) lim[i] = 1 + (int)(rnd() * 5);
        const int drafts = 1 + (int)(rnd() * 12);
        const double e = share(g, 4, lim, prev, drafts, 0.0, 0.0, -1, k);
        EXPECT(std::fabs(e - brute(g, 4, lim, drafts, 0, kb)) < 1e-9);
    }
    // 4. a share stays when the rates are about equal (no new graph for noise), and moves when they are not
    {
        Rates r[3] = {exact(0.70), exact(0.72), exact(0.71)};
        const Rates* rs[3] = {&r[0], &r[1], &r[2]};
        Gain g;
        gains(rs, none, 3, g);
        int lim[3] = {4, 4, 4}, prev[3] = {2, 1, 2}, k[3];
        share(g, 3, lim, prev, 5, 0.10, 0.0, -1, k);  // (without the last shares: the two best second drafts)
        EXPECT(k[0] == 1 && k[1] == 2 && k[2] == 2);
        share(g, 3, lim, prev, 5, 0.10, 0.10, -1, k);
        EXPECT(k[0] == 2 && k[1] == 1 && k[2] == 2);
        r[0] = exact(0.30);
        gains(rs, none, 3, g);
        share(g, 3, lim, prev, 5, 0.10, 0.10, -1, k);
        EXPECT(k[0] == 0 && k[1] == 3 && k[2] == 2);
    }
    // 5. the explored sequence verifies one draft more, inside the budget; without room at its limit nothing changes
    {
        Rates r[4] = {exact(0.2), exact(0.8), exact(0.8), exact(0.8)};
        const Rates* rs[4] = {&r[0], &r[1], &r[2], &r[3]};
        Gain g;
        gains(rs, none, 4, g);
        int lim[4] = {3, 3, 3, 3}, prev[4] = {-1, -1, -1, -1}, k0[4], k[4];
        share(g, 4, lim, prev, 8, 0.10, 0.0, -1, k0);
        share(g, 4, lim, prev, 8, 0.10, 0.0, 0, k);
        EXPECT(k[0] == k0[0] + 1);
        EXPECT(k[0] + k[1] + k[2] + k[3] <= 8);
        share(g, 4, lim, prev, 8, 0.10, 0.0, 1, k);  // (sequence 1 is at its limit already)
        for (int i = 0; i < 4; i++) EXPECT(k[i] == k0[i]);
        // a floor above every gain: only the explored draft is verified
        share(g, 4, lim, prev, 8, 2.0, 0.0, 2, k);
        EXPECT(k[0] == 0 && k[1] == 0 && k[2] == 1 && k[3] == 0);
        // the explored draft never takes another sequence's kept draft
        int lim1[4] = {2, 1, 1, 1};
        share(g, 4, lim1, prev, 4, 0.10, 0.0, 0, k, 1);
        EXPECT(k[0] == 1 && k[1] == 1 && k[2] == 1 && k[3] == 1);
    }
    // 6. the engine's loop on a model: 8 conversations, half reasoning (0.35 per depth) and half code (0.85), cap 2
    {
        const auto p = [](int i, int) { return i < 4 ? 0.35 : 0.85; };
        const Sim e = run(8, 6000, 2, 0, 2, 32, false, p), a = run(8, 6000, 2, 0, 2, 32, true, p);
        const double te = e.tok / 6000, ta = a.tok / 6000;
        printf("8 conversations (4 x 0.35, 4 x 0.85), cap 2: even %.2f tokens per step at %.1f rows, shared %.2f at %.1f rows, "
               "%.2f drafts made (%+.1f %% tokens per step; %+.1f %% with 2 %% per extra draft pass)\n", te, e.rows / 6000, ta,
               a.rows / 6000, a.passes / 6000, 100.0 * (ta / te - 1),
               100.0 * (ta / (1.0 + 0.02 * (a.passes / 6000 - 2)) / te - 1));
        EXPECT(a.rows <= e.rows + 1e-9);
        EXPECT(ta > te * 1.08);
        for (int i = 0; i < 4; i++) EXPECT(a.kend[i] >= 1 && a.kend[i] <= 2);
        for (int i = 4; i < 8; i++) EXPECT(a.kend[i] >= 2);
    }
    // 7. all alike: nothing to share, the split stays even (within the explored draft) and tokens do not drop
    {
        const auto p = [](int, int) { return 0.6; };
        const Sim e = run(8, 6000, 2, 0, 2, 32, false, p), a = run(8, 6000, 2, 0, 2, 32, true, p);
        printf("8 conversations alike (0.60), cap 2: even %.2f tokens per step, shared %.2f at %.1f rows, %.2f drafts made\n",
               e.tok / 6000, a.tok / 6000, a.rows / 6000, a.passes / 6000);
        EXPECT(a.tok > e.tok * 0.97);
        EXPECT(a.passes / 6000 < 2.6);
    }
    // 8. 7 conversations: the 3 rows of the 24 the even split leaves over are used
    {
        const auto p = [](int i, int) { return i < 3 ? 0.4 : 0.8; };
        const Sim e = run(7, 6000, 2, 0, 2, 32, false, p), a = run(7, 6000, 2, 3, 2, 32, true, p);
        printf("7 conversations (3 x 0.40, 4 x 0.80), cap 2 + 3 rows: even %.2f tokens per step at %.1f rows, shared %.2f at %.1f\n",
               e.tok / 6000, e.rows / 6000, a.tok / 6000, a.rows / 6000);
        EXPECT(a.tok > e.tok * 1.08);
        EXPECT(a.rows / 6000 <= 24.0);
    }
    // 9. the text changes: a conversation that turns unpredictable gives its rows up, one that turns predictable gets
    // them back through the explored drafts
    {
        const auto p = [](int i, int t) { return i == 0 ? (t < 3000 ? 0.9 : 0.2) : i == 1 ? (t < 3000 ? 0.2 : 0.9) : 0.6; };
        const Sim a1 = run(6, 2920, 3, 0, 2, 16, true, p);  // (its last plan explores sequence 2)
        printf("6 conversations, cap 3: shares before the change %d %d %d %d %d %d", a1.kend[0], a1.kend[1], a1.kend[2], a1.kend[3], a1.kend[4], a1.kend[5]);
        EXPECT(a1.kend[0] >= 4 && a1.kend[1] <= 1);
        rng_state = 88172645463325252ull;
        const Sim a2 = run(6, 6000, 3, 0, 2, 16, true, p);
        printf(", after %d %d %d %d %d %d\n", a2.kend[0], a2.kend[1], a2.kend[2], a2.kend[3], a2.kend[4], a2.kend[5]);
        EXPECT(a2.kend[0] <= 2 && a2.kend[1] >= 4);
    }
    // 10. terrible acceptance everywhere: almost no drafts verified, one draft still made (the explored one finds a change)
    {
        const auto p = [](int, int) { return 0.05; };
        const Sim a = run(8, 3000, 2, 0, 2, 16, true, p);
        EXPECT(a.rows / 3000 < 8 + 2.5);
        EXPECT(a.kd == 1);
    }
    if (fails) {
        printf("%d checks failed\n", fails);
        return 1;
    }
    printf("mtp_share_unit: ok\n");
    return 0;
}
