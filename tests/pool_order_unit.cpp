// Both engines: the KV pool's pick orders (src/pool_order.h), every
// conversation alike, on a model of the slots: no GPU. The engine call sites
// (27B room_or_pause / pause_for / the step_batch resume / pick_slot /
// archive_more, Flash-Next room_or_pause / qe_pause_for / resume_paused) use
// these comparators directly.
#include "pool_order.h"
#include <cstdio>
#include <vector>

#define EXPECT(c)                                                              \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("FAIL line %d: %s\n", __LINE__, #c);                              \
      fails++;                                                                 \
    }                                                                          \
  } while (0)
static int fails = 0;
using V = std::vector<int>;

struct Slot {
  int prio;
  unsigned long long admit, paused;
  int pos;
  unsigned long long used;
};

int main() {
  // a main (prio 2) admitted last pauses first; prio is not read (M1: pause_for
  // uses the same order)
  std::vector<Slot> s = {{1, 10, 0, 0, 0}, {2, 30, 0, 0, 0}, {0, 20, 0, 0, 0}};
  EXPECT(qe_pool::pause_order({0, 1, 2}, [&](int i) { return s[i].admit; }) ==
         V({1, 2, 0}));
  // the room_or_pause scan: the first pauses_first winner over the slots
  int v = -1;
  for (int j = 0; j < 3; j++)
    if (v < 0 || qe_pool::pauses_first(s[j].admit, s[v].admit))
      v = j;
  EXPECT(v == 1);
  // resume: first paused first, whatever the prio (a main paused after two
  // subagents waits behind them)
  s = {{2, 0, 50, 0, 0}, {1, 0, 40, 0, 0}, {0, 0, 45, 0, 0}};
  EXPECT(qe_pool::resume_order({0, 1, 2}, [&](int i) { return s[i].paused; }) ==
         V({1, 2, 0}));
  // 27B pick_slot overwrite: fewest tokens, then least recently used; a short
  // main goes before a long finished subagent
  s = {{2, 0, 0, 1000, 9}, {0, 0, 0, 90000, 1}, {1, 0, 0, 1000, 5}};
  int b = -1;
  for (int i = 0; i < 3; i++)
    if (b < 0 ||
        qe_pool::overwrite_first(s[i].pos, s[i].used, s[b].pos, s[b].used))
      b = i;
  EXPECT(b == 2);
  EXPECT(qe_pool::overwrite_first(10, 9, 11, 1));
  EXPECT(!qe_pool::overwrite_first(10, 9, 10, 1));
  // 27B archive_more: longest first; a refused write tries the next (review
  // M2), none accepted: -1; ties: the lower slot
  s = {{0, 0, 0, 40000, 0}, {0, 0, 0, 100000, 0}, {0, 0, 0, 40000, 0}};
  auto pos = [&](int i) { return s[i].pos; };
  std::vector<int> tried;
  int got = qe_pool::longest_accepted({0, 1, 2}, pos, [&](int i) {
    tried.push_back(i);
    return i != 1;
  });
  EXPECT(got == 0 && tried == V({1, 0}));
  tried.clear();
  got = qe_pool::longest_accepted({0, 1, 2}, pos, [&](int i) {
    tried.push_back(i);
    return false;
  });
  EXPECT(got == -1 && tried == V({1, 0, 2}));
  EXPECT(qe_pool::longest_accepted({}, pos, [](int) { return true; }) == -1);
  // QoS classes (bead 2b03): a background start never pauses an interactive
  // decoder; class 0 may pause both
  EXPECT(qe_pool::pausable_by(qe_pool::kClassBackground,
                              qe_pool::kClassBackground));
  EXPECT(!qe_pool::pausable_by(qe_pool::kClassBackground,
                               qe_pool::kClassInteractive));
  EXPECT(qe_pool::pausable_by(qe_pool::kClassInteractive,
                              qe_pool::kClassInteractive));
  EXPECT(qe_pool::pausable_by(qe_pool::kClassInteractive,
                              qe_pool::kClassBackground));
  // N=0 preserves the old admission and pause rules.
  EXPECT(qe_pool::interactive_limit(8, 0) == 8);
  EXPECT(qe_pool::may_admit(qe_pool::kClassInteractive, 7, 8, 0));
  EXPECT(qe_pool::pausable_by(qe_pool::kClassBackground,
                              qe_pool::kClassInteractive, 0, 0, false) ==
         qe_pool::pausable_by(qe_pool::kClassBackground,
                              qe_pool::kClassInteractive));
  // Reserve two of eight seats for background; a background below N may
  // displace interactive work.
  EXPECT(qe_pool::interactive_limit(8, 2) == 6);
  EXPECT(!qe_pool::may_admit(qe_pool::kClassInteractive, 6, 8, 2));
  EXPECT(qe_pool::may_admit(qe_pool::kClassBackground, 8, 8, 2));
  EXPECT(qe_pool::pausable_by(qe_pool::kClassBackground,
                              qe_pool::kClassInteractive, 1, 2, false));
  const std::vector<unsigned long long> interactive_admits = {7, 11};
  const V newest_interactive = qe_pool::pause_order(
      {0, 1}, [&](int i) { return interactive_admits[i]; });
  EXPECT(newest_interactive.front() == 1);
  EXPECT(!qe_pool::pausable_by(qe_pool::kClassBackground,
                               qe_pool::kClassInteractive, 2, 2, false));
  EXPECT(!qe_pool::pausable_by(qe_pool::kClassInteractive,
                               qe_pool::kClassBackground, 2, 2, true));
  EXPECT(qe_pool::pausable_by(qe_pool::kClassInteractive,
                              qe_pool::kClassBackground, 3, 2, false));
  printf(fails ? "pool_order_unit: %d FAILED\n" : "pool_order_unit: ok\n",
         fails);
  // [al0-0041] class-aware: background pauses first whatever its age, then recency
  {
    using namespace qe_pool;
    EXPECT(pauses_first_cls(kClassBackground, 1, kClassInteractive, 99));
    EXPECT(!pauses_first_cls(kClassInteractive, 99, kClassBackground, 1));
    EXPECT(pauses_first_cls(kClassInteractive, 30, kClassInteractive, 20));
    EXPECT(pauses_first_cls(kClassBackground, 30, kClassBackground, 20));
    // resume: a paused lane before paused bulk, whoever paused first
    EXPECT(resumes_first_cls(kClassInteractive, 99, kClassBackground, 1));
    EXPECT(!resumes_first_cls(kClassBackground, 1, kClassInteractive, 99));
    EXPECT(resumes_first_cls(kClassBackground, 1, kClassBackground, 2));
    // room: only a paused interactive sequence keeps its room
    EXPECT(paused_room_reserved(kClassInteractive));
    EXPECT(!paused_room_reserved(kClassBackground));
  }
  return fails ? 1 : 0;
}
