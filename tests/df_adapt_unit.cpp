// 27B [q27-dfa]: how many DFlash2 block drafts a step verifies (src/df_adapt.h), on a model of the drafter: no GPU.
// The engine call sites (Engine::dfa_kv / dfa_observe, launch_step, launch_batch) use these functions directly.
#include "df_adapt.h"
#include <cstdio>

#define EXPECT(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)
static int fails = 0;
using namespace qe_dfa;

static unsigned long long rng_state = 88172645463325252ull;
static double rnd() {  // xorshift64, [0, 1)
    rng_state ^= rng_state << 13, rng_state ^= rng_state >> 7, rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) / 9007199254740992.0;
}
// drafts accepted of kv verified when each depth is accepted with probability p[i] given the ones before
static int accept(int kv, const double* p) {
    int a = 0;
    while (a < kv && rnd() < p[a + 1]) a++;
    return a;
}

// a slot decoding alone for `steps` steps: the engine's loop (pick, probe, verify, observe, time = c0 + c1 * kv).
// Returns tokens per millisecond; *kend: the slot's count at the end.
static double run(Slot& s, Rates& all, RowCost& row, const double* p, int steps, double c0, double c1, int kmin, int period,
                  bool adaptive, int* kend, double def_c0 = 8.0, double def_c1 = 0.65) {
    double tok = 0, ms = 0;
    for (int i = 0; i < steps; i++) {
        int kv = kDepth;
        bool probe = false;
        if (adaptive) {
            const double e1 = s.row.get_over(row.get(def_c1));
            const double e0 = s.t.known() ? s.t.c0(e1) : def_c0;
            s.k[0] = pick(s.r, all, e0, e1, kmin, kDepth, s.k[0]);
            probe = s.k[0] < kDepth && probe_due(s.tick++, period);
            kv = s.k[0] + probe;
        }
        const int a = accept(kv, p);
        const double t = c0 + c1 * kv;
        tok += a + 1, ms += t;
        if (adaptive) {
            s.r.observe(kv, a), all.observe(kv, a);
            double smp = 0;
            if (s.t.add(kv, t, probe, &smp)) s.row.add(smp), row.add(smp);
        }
    }
    if (kend) *kend = s.k[0];
    return tok / ms;
}

int main() {
    const double low[8] = {0, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68};   // about 2 of 7 accepted (the live reasoning text)
    const double high[8] = {0, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97};  // predictable text
    const double cliff[8] = {0, 0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.9};        // two good drafts, then mostly wrong

    {  // nothing seen: all 7, as before the rule existed
        Rates s, all;
        Slot fresh;
        EXPECT(fresh.k[0] == kDepth && fresh.k[1] == kDepth);
        EXPECT(pick(s, all, 8.0, 0.65, 2, kDepth, fresh.k[0]) == kDepth);
        EXPECT(pick(s, all, 8.0, 0.65, 2, kDepth, -1) >= kDepth - 1);
        EXPECT(tokens(s, all, 0) == 1.0);
        EXPECT(tokens(s, all, 7) > tokens(s, all, 6));
    }
    {  // only tried depths decay; a rejection at depth i says nothing about deeper ones
        Rates r;
        r.observe(7, 2);
        EXPECT(r.num[1] == 1 && r.num[2] == 1 && r.num[3] == 0 && r.den[3] == 1 && r.den[4] == 0);
        r.observe(2, 2);
        EXPECT(r.den[3] == 1 && r.den[1] == 1.95 && r.num[2] == 1.95);
        r.observe(9, 9);  // (never past the block)
        EXPECT(r.den[7] == 1 && r.num[7] == 1);
    }
    {  // low acceptance: fewer drafts, more tokens per millisecond than verifying all 7 (same text, same costs)
        Slot s, f;
        Rates all, fall;
        RowCost row, frow;
        int k = 0;
        rng_state = 1234567;
        const double ad = run(s, all, row, low, 4000, 8.0, 0.65, 2, 16, true, &k);
        rng_state = 1234567;
        const double fx = run(f, fall, frow, low, 4000, 8.0, 0.65, 2, 16, false, nullptr);
        printf("low acceptance, c0 8 c1 0.65: count %d, %.4f vs %.4f tok/ms (%+.1f %%)\n", k, ad, fx, 100 * (ad / fx - 1));
        EXPECT(k >= 3 && k <= 5);
        EXPECT(ad > fx * 1.03);
        EXPECT(row.get(0.0) > 0.6 && row.get(0.0) < 0.7);  // the probes measured the row cost
        // a long context (rows cost more): fewer still, and a larger gain
        Slot s2, f2;
        Rates all2, fall2;
        RowCost row2, frow2;
        int k2 = 0;
        rng_state = 1234567;
        const double ad2 = run(s2, all2, row2, low, 4000, 10.0, 1.5, 2, 16, true, &k2);
        rng_state = 1234567;
        const double fx2 = run(f2, fall2, frow2, low, 4000, 10.0, 1.5, 2, 16, false, nullptr);
        printf("low acceptance, c0 10 c1 1.5: count %d, %.4f vs %.4f tok/ms (%+.1f %%)\n", k2, ad2, fx2, 100 * (ad2 / fx2 - 1));
        EXPECT(k2 <= k && k2 >= 2);
        EXPECT(ad2 > fx2 * 1.10);
        EXPECT(row2.get(0.0) > 1.4 && row2.get(0.0) < 1.6);
        EXPECT(s2.t.known() && s2.t.c0(row2.get(0.0)) > 9.5 && s2.t.c0(row2.get(0.0)) < 10.5);
    }
    {  // predictable text: stays at 7 and loses nothing
        Slot s, f;
        Rates all, fall;
        RowCost row, frow;
        int k = 0;
        rng_state = 99;
        const double ad = run(s, all, row, high, 3000, 8.0, 0.65, 2, 16, true, &k);
        rng_state = 99;
        const double fx = run(f, fall, frow, high, 3000, 8.0, 0.65, 2, 16, false, nullptr);
        EXPECT(k == kDepth);
        EXPECT(ad > fx * 0.995);
    }
    {  // a depth that is mostly wrong: stop before it
        Slot s;
        Rates all;
        RowCost row;
        int k = 0;
        rng_state = 7;
        run(s, all, row, cliff, 3000, 8.0, 0.65, 1, 16, true, &k);
        EXPECT(k == 2);
    }
    {  // the text changes: reasoning, then code, then reasoning. The count follows it up and down
        Slot s;
        Rates all;
        RowCost row;
        int k = 0;
        rng_state = 42;
        run(s, all, row, low, 2000, 8.0, 0.65, 2, 16, true, &k);
        EXPECT(k < kDepth);
        run(s, all, row, high, 1500, 8.0, 0.65, 2, 16, true, &k);
        EXPECT(k == kDepth);
        run(s, all, row, low, 1500, 8.0, 0.65, 2, 16, true, &k);
        EXPECT(k >= 2 && k <= 5);
        // what the probes are for: a depth given up as mostly wrong is only tried again by a probe
        Slot p, np;
        Rates pall, nall;
        RowCost prow, nrow;
        int kp = 0, kn = 0;
        rng_state = 42;
        run(p, pall, prow, cliff, 2000, 8.0, 0.65, 1, 16, true, &kp);
        run(p, pall, prow, high, 1500, 8.0, 0.65, 1, 16, true, &kp);
        rng_state = 42;
        run(np, nall, nrow, cliff, 2000, 8.0, 0.65, 1, 0, true, &kn);
        EXPECT(kn == 2);
        run(np, nall, nrow, high, 1500, 8.0, 0.65, 1, 0, true, &kn);
        EXPECT(kp == kDepth && kn == 2);
    }
    {  // a batched step shares its fixed part: never more drafts than alone; QE_DF_ADAPT_MIN is kept
        Slot s;
        Rates all;
        RowCost row;
        rng_state = 5;
        run(s, all, row, low, 2000, 8.0, 0.65, 2, 16, true, nullptr);
        const int k1 = pick(s.r, all, 8.0, 0.65, 1, kDepth, -1), k4 = pick(s.r, all, 8.0 / 4, 0.65, 1, kDepth, -1);
        EXPECT(k4 <= k1 && k4 >= 1);
        EXPECT(pick(s.r, all, 8.0 / 4, 3.0, 2, kDepth, -1) == 2);
        EXPECT(pick(s.r, all, 8.0 / 4, 3.0, 5, kDepth, -1) == 5);
        EXPECT(pick(s.r, all, 8.0, 0.65, 2, 3, -1) <= 3);
        EXPECT(pick(s.r, all, 8.0, 0.65, 9, 9, -1) == kDepth);  // (bounds outside the block)
        // costlier rows never mean more drafts
        int last = kDepth;
        for (double c1 : {0.2, 0.5, 1.0, 2.0, 3.0}) {
            const int k = pick(s.r, all, 8.0, c1, 1, kDepth, -1);
            EXPECT(k <= last);
            last = k;
        }
    }
    {  // the last pick stays unless another count is worth 3 % more (every new count is a new step graph)
        Rates s, all;
        for (int i = 0; i < 200; i++) s.observe(7, i % 3 == 0 ? 7 : i % 3 == 1 ? 3 : 1);
        const int b = pick(s, all, 8.0, 0.65, 1, kDepth, -1);
        for (int k = 1; k <= kDepth; k++) {
            const bool close = score(s, all, b, 8.0, 0.65) <= score(s, all, k, 8.0, 0.65) * 1.03;
            EXPECT(pick(s, all, 8.0, 0.65, 1, kDepth, k) == (close ? k : b));
        }
        EXPECT(pick(s, all, 8.0, 0.65, 1, kDepth, 1) == b);  // (1 draft is far from the best here)
        EXPECT(pick(s, all, 8.0, 0.65, 2, kDepth, 1) != 1);  // a last pick outside the bounds is not kept
    }
    {  // a slot that has seen nothing starts from the engine's acceptance, not from 7
        Slot old;
        Rates all;
        RowCost row;
        rng_state = 11;
        run(old, all, row, low, 2000, 8.0, 0.65, 2, 16, true, nullptr);
        Rates fresh;
        const int k = pick(fresh, all, 8.0, 0.65, 2, kDepth, -1);
        EXPECT(k < kDepth && k >= 2);
        // and its own text wins over the engine's after a few dozen steps
        Slot mine;
        int km = 0;
        run(mine, all, row, high, 300, 8.0, 0.65, 2, 16, true, &km);
        EXPECT(km == kDepth);
    }
    {  // probes: one in `period` steps, none with 0
        int n = 0;
        for (unsigned long long t = 0; t < 160; t++) n += probe_due(t, 16);
        EXPECT(n == 10 && !probe_due(0, 16) && probe_due(15, 16) && !probe_due(15, 0));
    }
    {  // step time: the base follows the context (slow drift is no row cost), outliers and unknown bases give no sample
        StepTime t;
        double smp = -1;
        EXPECT(!t.add(4, 10.0, false, &smp) && !t.known());
        EXPECT(!t.add(5, 11.0, true, &smp));  // (a probe before the base is known: no sample, the base stays)
        EXPECT(t.k == 4 && t.n == 1);
        for (int i = 0; i < 5; i++) EXPECT(!t.add(4, 10.0 + 0.01 * i, false, &smp));
        EXPECT(t.known());
        EXPECT(!t.add(4, 60.0, false, &smp) && t.base < 10.1);  // an outlier leaves the base alone
        EXPECT(t.add(5, t.base + 1.2, true, &smp) && smp > 1.19 && smp < 1.21 && t.k == 4);
        EXPECT(!t.add(5, 80.0, true, &smp));                    // an outlier probe: no sample
        EXPECT(t.add(2, t.base - 2.4, false, &smp) && smp > 1.19 && smp < 1.21);  // a new count: a sample, and the base moves
        EXPECT(t.k == 2 && t.n == 1 && !t.known());
        EXPECT(!t.add(2, 0.0, false, &smp));
        RowCost r;
        EXPECT(r.get(0.65) == 0.65);  // too few samples: the default
        for (int i = 0; i < 2; i++) r.add(1.2);
        EXPECT(r.get(0.65) == 0.65);
        for (int i = 0; i < 2; i++) r.add(1.2);
        EXPECT(r.get(0.65) > 1.19 && r.get(0.65) < 1.21);
        r.add(50.0), r.add(-9.0);  // (not row costs)
        EXPECT(r.get(0.65) > 1.19 && r.get(0.65) < 1.21);
        for (int i = 0; i < 50; i++) r.add(5.5);
        EXPECT(r.get(0.65) == 3.0);  // bounded like the MTP step-time model
        RowCost none;
        EXPECT(none.get_over(0.65) == 0.65);
        RowCost own;
        for (int i = 0; i < 20; i++) own.add(2.0);
        EXPECT(own.get_over(0.65) > 1.5 && own.get_over(0.65) < 2.0);  // a slot's own samples outweigh the engine's
    }
    if (fails) {
        printf("df_adapt_unit: %d FAILED\n", fails);
        return 1;
    }
    printf("df_adapt_unit: ok\n");
    return 0;
}
