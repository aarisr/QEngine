// CPU integration of the actual driver ownership seam; Worker is the CUDA
// boundary.
#include "flash/live_prefix.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#define CHECK(ok, ...)                                                         \
  do {                                                                         \
    if (!(ok))                                                                 \
      throw std::runtime_error("driver invariant");                            \
  } while (0)
static constexpr int MAXP = 2;
static int env_int(const char *, int fallback) { return fallback; }
struct Seg {
  std::vector<char *> p;
};
struct Pin {
  std::vector<int> toks;
  std::shared_ptr<Seg> kv;
};
struct Slot {
  int base = 0, cap = 0, pending = 0;
  bool active = false, paused = false;
  void *conv = nullptr;
  unsigned long long admit = 0;
};
struct Worker {
  int stride = 0, period = 4, top = 0;
  std::vector<std::vector<int>> maps;
  std::vector<int> shared;
  std::vector<char> bytes = std::vector<char>(64, 0);
  std::vector<int> restored;
  bool kv_stream() const { return true; }
  void live_init(int s, int) {
    stride = s;
    maps.resize(4);
    shared.resize(4);
  }
  void live_map(int slot, const std::vector<int> &map, int sh,
                const std::vector<int> &) {
    maps.at(slot) = map;
    shared.at(slot) = sh;
  }
  void live_restore(const char *source, int base, int a, int z, int upto,
                    bool initialize) {
    const int slot = base / stride;
    const int first = std::max(a, initialize ? 0 : shared.at(slot) * period);
    const int last = std::min(z, upto);
    restored.push_back(std::max(0, last - first));
    for (int i = first; i < last; ++i)
      bytes.at(maps.at(slot).at(i / period) * period + i % period) = source[i];
  }
  void set_kv_active(long long) {}
  void set_kv_top(int t) { top = t; }
  void kv_claim(int) {}
};
struct FE {
  std::vector<Slot> sl = std::vector<Slot>(4);
  Worker worker;
  std::vector<Worker *> W{&worker};
  std::function<void(int)> job;
  void *ppb = nullptr;
  void *pph = nullptr;
  int ctx = 48, np = 16, period = 4, reserve = 4, ng = 1, log = 0, cs = 0;
  bool batch_api = true;
  int periods_for(int n) const { return (n + period - 1) / period; }
  void run(const char *) { job(0); }
  int free_slot() const {
    for (int s = 0; s < (int)sl.size(); ++s)
      if (!sl[s].conv && !sl[s].cap)
        return s;
    return -1;
  }
  int idle_victim(int s, const std::vector<int> &protect,
                  bool any = false) const {
    for (int v = 0; v < (int)sl.size(); ++v)
      if (v != s && !sl[v].active && (any || sl[v].cap) &&
          std::find(protect.begin(), protect.end(), v) == protect.end())
        return v;
    return -1;
  }
  void evict_slot(int s, bool, const char *) {
    live_drop(s);
    sl[s] = Slot{};
  }
  void rebind() {}
  int floor_of(const Slot &) const { return 1; }
  bool prefilling(int) const { return false; }
  bool pause_slot(int s) {
    live_drop(s);
    sl[s].cap = 0;
    sl[s].paused = true;
    return true;
  }
#include "flash/live_prefix_fe.inc"
  void enable() {
    live_stride = periods_for(ctx + MAXP + 2);
    live_arena = std::make_unique<flash_live::Arena>(np);
    live_views.resize(sl.size());
    live_marks.resize(np);
    worker.live_init(live_stride, np);
    live_on = true;
  }
};
static void expect(bool ok) {
  if (!ok)
    throw std::runtime_error("CPU live driver test failed");
}
int main() {
  FE e;
  e.enable();
  std::vector<char> source(16);
  for (int i = 0; i < 16; ++i)
    source[i] = (char)(i + 1);
  Pin pin;
  pin.toks.resize(16);
  pin.kv = std::make_shared<Seg>();
  pin.kv->p.push_back(source.data());
  const int a = e.live_acquire(20, {}, &pin);
  expect(a == 0);
  expect(e.live_shared_tokens(a) == 12);
  expect(e.live_last_init_tokens == 12);
  e.worker.live_restore(source.data(), e.sl[a].base, 0, 16, 16, false);
  e.sl[a].conv = &e;
  e.sl[a].active = true;
  const int b = e.live_acquire(20, {}, &pin);
  expect(b == 1);
  expect(e.live_last_init_tokens == 0);
  e.worker.live_restore(source.data(), e.sl[b].base, 0, 16, 16, false);
  e.sl[b].conv = &e;
  e.sl[b].active = true;
  expect((e.worker.restored == std::vector<int>{12, 4, 4}));
  expect(e.live_arena->used() ==
         9); // three immutable + two independent three-period tails
  for (int i = 0; i < 3; ++i)
    expect(e.worker.maps[a][i] == e.worker.maps[b][i]);
  expect(e.worker.maps[a][3] != e.worker.maps[b][3]);
  const int pa = e.worker.maps[a][3] * 4, pb = e.worker.maps[b][3] * 4;
  e.worker.bytes[pa] = 61;
  e.worker.bytes[pb] = 62;
  expect(e.worker.bytes[pa] == 61 && e.worker.bytes[pb] == 62);
  const auto prefix = e.worker.maps[b][0];
  e.live_index.clear();
  pin.kv.reset();
  e.evict_slot(a, true, "test source eviction");
  expect(e.worker.bytes[prefix * 4] == 1);
  expect(e.live_arena->used() == 6);
  e.evict_slot(b, true, "last consumer");
  e.live_update_top();
  expect(e.live_arena->used() == 0 && e.worker.top == 0);
  // Failed acquisition releases its temporary immutable prefix and never
  // publishes a hit.
  FE small;
  small.np = 4;
  small.enable();
  pin.kv = std::make_shared<Seg>();
  pin.kv->p.push_back(source.data());
  expect(small.live_acquire(20, {}, &pin) == -1);
  expect(small.live_arena->used() == 0);
  expect(small.live_index.find(std::static_pointer_cast<const void>(pin.kv),
                               3) == nullptr);
  std::puts("live_prefix_driver_unit: PASS");
}
