// CPU test of the shared PREFIX object module (src/flash/shared_prefix.h): key
// derivation, round trip, checksum and fingerprint refusal, two-writer races, a
// writer killed mid-layout, unlink while mapped, the inotify index with a
// forced rescan, the flock'd evictor, per-writer quota pruning, and the
// registration-failure fallback, async queue/drop/lifetime, and published-object
// visibility. Multi-process cases re-exec this binary with a child argument (no
// a pipe). No GPU: registration is a hook the test installs and can make fail.
// Measurements (n, p50/p95) cover the CPU-side numbers CI can produce:
// build/publish, map+verify, attach policy, and the bytes per prefix; the
// register/DMA numbers need a GPU and are printed as pending with the
// constraints.
#include "flash/shared_prefix.h"

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>
#include <chrono>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace sp;
static int fails = 0;
#define EXPECT(c)                                                              \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("FAIL line %d: %s\n", __LINE__, #c);                              \
      fails++;                                                                 \
    }                                                                          \
  } while (0)

static std::string g_dir;
static std::string g_self;
#include "shared_prefix_test_dir.inc"

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static std::vector<int> toks(int n, int seed = 1) {
  std::vector<int> v(n);
  std::mt19937 r(seed);
  for (auto &x : v)
    x = (int)(r() % 100000);
  return v;
}
static std::vector<char> bytes(size_t n, unsigned seed) {
  std::vector<char> v(n);
  unsigned x = seed;
  for (auto &c : v) {
    x = x * 1103515245u + 12345u;
    c = (char)(x >> 16);
  }
  return v;
}
static Desc make_desc(const std::string &fp, const std::string &steer,
                      const std::vector<int> &t, int ng, size_t kv_per_card,
                      size_t st_per_card) {
  Desc d;
  d.fp = fp;
  d.steer = steer;
  d.toks = t;
  d.writer = 7;
  d.ng = ng;
  d.kv.assign(ng, Shard{0, (uint64_t)kv_per_card, 0});
  d.st.assign(ng, Shard{0, (uint64_t)st_per_card, 0});
  return d;
}
static std::vector<std::vector<char>> payloads(const Desc &d) {
  std::vector<std::vector<char>> v(2 * d.ng);
  for (int r = 0; r < d.ng; r++) {
    v[2 * r] = bytes((size_t)d.kv[r].len, 11 + r);
    v[2 * r + 1] = bytes((size_t)d.st[r].len, 101 + r);
  }
  return v;
}
static bool write_obj(const Desc &d, const std::string &fp,
                      const std::string &steer, const std::vector<int> &t,
                      const std::vector<std::vector<char>> &pay,
                      std::string *out_key = nullptr) {
  Desc dd = d;
  dd.fp = fp;
  dd.steer = steer;
  dd.toks = t;
  Builder b;
  int err = 0;
  if (!b.create(g_dir, dd, err))
    return false;
  for (int r = 0; r < dd.ng; r++) {
    memcpy(b.kv(r), pay[2 * r].data(), pay[2 * r].size());
    memcpy(b.st(r), pay[2 * r + 1].data(), pay[2 * r + 1].size());
  }
  const std::string k = key(fp, steer, t);
  if (!b.publish(g_dir, k, err))
    return false;
  if (out_key)
    *out_key = k;
  return true;
}
static bool same_payload(const Map &m,
                         const std::vector<std::vector<char>> &pay) {
  for (int r = 0; r < m.desc().ng; r++)
    if (memcmp(m.kv(r), pay[2 * r].data(), m.kv_len(r)) != 0)
      return false;
  for (int r = 0; r < m.desc().ng; r++)
    if (memcmp(m.st(r), pay[2 * r + 1].data(), m.st_len(r)) != 0)
      return false;
  return true;
}
static double q(std::vector<double> v, double p) {
  if (v.empty())
    return 0;
  std::sort(v.begin(), v.end());
  const size_t i = std::min(v.size() - 1, (size_t)(p * (v.size() - 1)));
  return v[i];
}
#include "shared_prefix_crc_test.inc"
#include "shared_prefix_unlink_test.inc"

// ---- async publication queue fixtures: AsyncWriter takes a plain function
// pointer plus a context, so its callbacks reach the state through this sink.
// Two gates park a callback inside the writer (show: before it looks at the
// item, finish: before it returns), so every ordering here is a
// condition_variable wait with a bound: no sleep-based ordering and no path
// that can hang.
static constexpr char kAwText[] = "prefix-payload";

struct AwSink {
  std::mutex m;
  std::condition_variable cv;
  bool show = true;   // false parks the callback before it inspects the item
  bool finish = true; // false parks the callback before it returns
  int entered = 0;    // callbacks that reached the callback body
  std::vector<int> values;           // completed writes, in callback order
  std::vector<std::thread::id> tids; // the thread that ran each callback
  bool payload_intact = false;
  int alive_at_entry = -1, alive_at_exit = -1;
  std::string payload_text;

  // bounded predicate wait: false on timeout, so a worker that never runs fails
  // the test instead of blocking CI
  template <class F> bool until(double timeout_s, F pred) {
    std::unique_lock<std::mutex> lock(m);
    return cv.wait_for(lock, std::chrono::duration<double>(timeout_s), pred);
  }
  void show_now() {
    {
      std::lock_guard<std::mutex> lock(m);
      show = true;
    }
    cv.notify_all();
  }
  void finish_now() {
    {
      std::lock_guard<std::mutex> lock(m);
      finish = true;
    }
    cv.notify_all();
  }
};

// a payload that counts its live instances, so the test can watch a queued
// shared_ptr keep it alive until the worker's callback is done with it
struct AwPayload {
  explicit AwPayload(std::atomic<int> *live, std::string text)
      : live(live), text(std::move(text)) {
    live->fetch_add(1);
  }
  ~AwPayload() { live->fetch_add(-1); }
  std::atomic<int> *live;
  const std::string text;
};

static void aw_write_int(void *ctx, int &&v) {
  AwSink *s = (AwSink *)ctx;
  {
    std::unique_lock<std::mutex> lock(s->m);
    s->tids.push_back(std::this_thread::get_id());
    s->entered++;
    s->cv.notify_all(); // bounded waits may already be parked on entered
    s->cv.wait(lock, [s] { return s->finish; });
    s->values.push_back(v);
  }
  s->cv.notify_all();
}

static void aw_write_payload(void *ctx, std::shared_ptr<AwPayload> &&p) {
  AwSink *s = (AwSink *)ctx;
  {
    std::unique_lock<std::mutex> lock(s->m);
    s->tids.push_back(std::this_thread::get_id());
    s->entered++; // the caller now parks before looking at the item
    s->cv.notify_all();
    s->cv.wait(lock, [s] { return s->show; });
    s->alive_at_entry = p ? p->live->load() : 0;
    s->payload_intact = p && p->text == kAwText;
    s->cv.notify_all(); // the entry observations are visible now
    s->cv.wait(lock, [s] { return s->finish; });
    s->alive_at_exit = p ? p->live->load() : 0; // still alive mid-callback
    if (p)
      s->payload_text = p->text;
  }
  s->cv.notify_all();
}

// ---- async publication of a real object: the queued item owns the Desc and
// its payload shards, so the worker lays the object out and publishes it
// itself, and the caller only ever sees it through the public open_map path.
// One gate parks the callback before it touches the item, so "not published
// yet" is an ordered observation and never a sleep.
struct ApJob {
  ApJob(std::atomic<int> *live, Desc desc,
        std::vector<std::vector<char>> shards, std::string obj_key)
      : live(live), d(std::move(desc)), pay(std::move(shards)),
        k(std::move(obj_key)) {
    live->fetch_add(1);
  }
  ~ApJob() { live->fetch_add(-1); }
  std::atomic<int> *live;
  const Desc d;
  const std::vector<std::vector<char>> pay;
  const std::string k; // the object key this job publishes under
};

// the shards folded into one number, so the callback can report the exact
// bytes it copied without sharing the caller's vectors
static uint64_t pay_hash(const std::vector<std::vector<char>> &pay) {
  uint64_t h = 1469598103934665603ull;
  for (const auto &v : pay)
    for (const char c : v)
      h = (h ^ (unsigned char)c) * 1099511628211ull;
  return h;
}

struct ApSink {
  std::mutex m;
  std::condition_variable cv;
  bool release = false; // false parks after ownership checks, before publish
  int entered = 0;      // callbacks that reached the callback body
  int alive_at_entry = -1, alive_at_exit = -1;
  bool shards_match = false; // shard count agrees with the description
  uint64_t seen_hash = 0, seen_bytes = 0; // payload the writer copied
  bool published = false;                 // publish() returned, either way
  bool ok = false;
  int err = 0;
  std::thread::id worker;
  std::string path; // where the writer says the object lives

  // bounded predicate wait: false on timeout, so a worker that never runs fails
  // the test instead of blocking CI
  template <class F> bool until(double timeout_s, F pred) {
    std::unique_lock<std::mutex> lock(m);
    return cv.wait_for(lock, std::chrono::duration<double>(timeout_s), pred);
  }
  void release_now() {
    {
      std::lock_guard<std::mutex> lock(m);
      release = true;
    }
    cv.notify_all();
  }
};

static void ap_publish_job(void *ctx, std::shared_ptr<ApJob> &&job) {
  ApSink *s = (ApSink *)ctx;
  {
    std::unique_lock<std::mutex> lock(s->m);
    s->worker = std::this_thread::get_id();
    s->entered++; // report entry while publication is still parked
    s->alive_at_entry = job ? job->live->load() : 0;
    s->shards_match = job && job->pay.size() == (size_t)job->d.ng * 2;
    s->cv.notify_all();
    s->cv.wait(lock, [s] { return s->release; });
  }
  s->cv.notify_all(); // wake the test after the callback returns from its gate
  if (job) {
    int err = 0;
    bool ok = false;
    {
      Builder b; // owns the mapping until the object is handed to its key
      ok = b.create(g_dir, job->d, err);
      for (int r = 0; ok && r < job->d.ng; r++) {
        memcpy(b.kv(r), job->pay[2 * r].data(), job->pay[2 * r].size());
        memcpy(b.st(r), job->pay[2 * r + 1].data(), job->pay[2 * r + 1].size());
      }
      if (ok)
        ok = b.publish(g_dir, job->k, err);
    }
    std::lock_guard<std::mutex> lock(s->m);
    s->seen_hash = pay_hash(job->pay);
    for (const auto &v : job->pay)
      s->seen_bytes += v.size();
    s->ok = ok;
    s->err = err;
    s->path = path_of(g_dir, job->k);
    s->alive_at_exit = job->live->load(); // shards live through the publish
    s->published = true;
  }
  s->cv.notify_all(); // released on every path: the drain and join cannot hang
}

// ---- child modes (multi-process cases)
static int child_writer(const std::string &fp, const std::string &steer,
                        int iters, int fd_ready) {
  Desc d = make_desc(fp, steer, toks(256, 3), 2, 1 << 16, 1 << 14);
  auto pay = payloads(d);
  if (write(fd_ready, "r", 1) != 1)
    return 2;
  for (int i = 0; i < iters; i++) {
    pay[0][0] = (char)i;
    if (!write_obj(d, fp, steer, d.toks, pay))
      return 3;
  }
  return 0;
}
static int child_killed_writer(const std::string &fp, const std::string &steer,
                               int fd_ready) {
  // big enough that the parent can kill it between create and publish
  Desc d = make_desc(fp, steer, toks(512, 4), 2, 1 << 21, 1 << 18);
  Builder b;
  int err = 0;
  if (!b.create(g_dir, d, err))
    return 3;
  memset(b.kv(0), 0, (size_t)d.kv[0].len / 2);
  if (write(fd_ready, "r", 1) != 1)
    return 2;
  for (;;)
    pause();
}

int main(int argc, char **argv) {
  char *exe = realpath(argv[0], nullptr);
  g_self = exe ? exe : argv[0];
  free(exe);
  if (argc >= 2 && std::string(argv[1]).rfind("--child-", 0) == 0) {
    const char *shared = getenv("QE_SP_TEST_DIR");
    if (!shared || !*shared)
      return 2;
    g_dir = shared;
    if (std::string(argv[1]) == "--child-writer")
      return child_writer(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]));
    if (std::string(argv[1]) == "--child-killed")
      return child_killed_writer(argv[2], argv[3], atoi(argv[4]));
    return 2;
  }
  if (!init_test_dir()) {
    printf("cannot create isolated test dir\\n");
    return 1;
  }
  setenv("QE_SP_TEST_DIR", g_dir.c_str(), 1);
  test_crc_performance(fails);
  const std::string fp = "fp|qpin-test|g2|kv4", steer = "vector:0|0|0|-1|1.0";

  // 1. key derivation: deterministic, and every input changes it
  {
    auto t = toks(64, 9);
    const std::string k1 = key(fp, steer, t), k2 = key(fp, steer, t);
    EXPECT(k1 == k2);
    auto t2 = t;
    t2[10] ^= 1;
    EXPECT(key(fp, steer, t2) != k1);
    EXPECT(key(fp + "x", steer, t) != k1);
    EXPECT(key(fp, steer + "x", t) != k1);
    auto t3 = t;
    t3.push_back(1);
    EXPECT(key(fp, steer, t3) != k1);
  }

  // 2. round trip: publish, map, verify every shard, exact payload
  {
    auto t = toks(4096, 5);
    Desc d = make_desc(fp, steer, t, 4, 1 << 16, 1 << 15);
    auto pay = payloads(d);
    std::string k;
    EXPECT(write_obj(d, fp, steer, t, pay, &k));
    EXPECT(k.size() == 32);
    Map m;
    const char *why = "";
    EXPECT(open_map(g_dir + "/" + k + kExt, fp, m, &why) == Rd::OK);
    EXPECT(m.desc().toks == t);
    EXPECT(m.desc().ng == 4);
    EXPECT(m.verify_all());
    EXPECT(same_payload(m, pay));
    EXPECT(!m.registered());
    EXPECT(!m.attach()); // no hooks: fallback path, not an error
  }

  // 3. fingerprint refusal and checksum refusal
  {
    auto t = toks(128, 6);
    Desc d = make_desc(fp, steer, t, 2, 1 << 14, 1 << 12);
    auto pay = payloads(d);
    std::string k;
    EXPECT(write_obj(d, fp, steer, t, pay, &k));
    Map m;
    EXPECT(open_map(g_dir + "/" + k + kExt, "other-fingerprint", m) ==
           Rd::OTHER);
    // flip one payload byte: the shard checksum must refuse it
    const std::string p = g_dir + "/" + k + kExt;
    Map base;
    EXPECT(open_map(p, fp, base) == Rd::OK);
    const long off = (long)(base.desc().kv[1].off + base.kv_len(1) / 2);
    FILE *f = fopen(p.c_str(), "r+b");
    if (!f) {
      EXPECT(false);
    } else {
      fseek(f, off, SEEK_SET);
      int c = fgetc(f);
      fseek(f, off, SEEK_SET);
      fputc(c ^ 0x5a, f);
      fclose(f);
    }
    Map bad;
    EXPECT(open_map(p, fp, bad) == Rd::OK); // header still valid
    EXPECT(!bad.verify_kv(1));              // payload refused
  }

  // 4. two writers racing on one key (two processes), then the object is whole
  // and equals one payload
  {
    int pfd[2];
    if (pipe(pfd) == 0) {
      pid_t p1 = fork();
      if (p1 == 0) {
        close(pfd[0]);
        if (pfd[1] < 0)
          _exit(2);
        std::string fd = std::to_string(pfd[1]);
        execl(g_self.c_str(), g_self.c_str(), "--child-writer", fp.c_str(),
              (steer + "-race").c_str(), "300", fd.c_str(), (char *)nullptr);
        _exit(5);
      }
      pid_t p2 = fork();
      if (p2 == 0) {
        close(pfd[0]);
        std::string fd = std::to_string(pfd[1]);
        execl(g_self.c_str(), g_self.c_str(), "--child-writer", fp.c_str(),
              (steer + "-race").c_str(), "300", fd.c_str(), (char *)nullptr);
        _exit(5);
      }
      close(pfd[1]);
      char buf[2];
      int got = 0;
      while (got < 2 && read(pfd[0], buf + got, 2 - got) > 0)
        got++;
      close(pfd[0]);
      int st1 = 0, st2 = 0;
      waitpid(p1, &st1, 0);
      waitpid(p2, &st2, 0);
      EXPECT(WIFEXITED(st1) && WEXITSTATUS(st1) == 0);
      EXPECT(WIFEXITED(st2) && WEXITSTATUS(st2) == 0);
      Desc d =
          make_desc(fp, steer + "-race", toks(256, 3), 2, 1 << 16, 1 << 14);
      Map m;
      const auto opened =
          open_map(g_dir + "/" + key(d.fp, d.steer, d.toks) + kExt, fp, m);
      EXPECT(opened == Rd::OK);
      if (opened == Rd::OK) {
        for (int r = 0; r < 2; r++) {
          EXPECT(m.verify_kv(r));
          EXPECT(m.verify_st(r));
        }
      }
      struct stat st;
      for (auto &de : std::vector<std::string>{})
        (void)de;
      DIR *dir = opendir(g_dir.c_str());
      int tmps = 0;
      if (dir) {
        while (dirent *e = readdir(dir))
          if (std::string(e->d_name).find(kTmpTag) != std::string::npos)
            tmps++;
        closedir(dir);
      }
      EXPECT(tmps == 0);
      (void)st;
    }
  }

  // 5. a writer killed between layout and publish leaves no final object and
  // only a private name
  {
    int pfd[2];
    if (pipe(pfd) == 0) {
      pid_t p = fork();
      if (p == 0) {
        close(pfd[0]);
        std::string fd = std::to_string(pfd[1]);
        execl(g_self.c_str(), g_self.c_str(), "--child-killed", fp.c_str(),
              (steer + "-kill").c_str(), fd.c_str(), (char *)nullptr);
        _exit(5);
      }
      close(pfd[1]);
      char b = 0;
      if (read(pfd[0], &b, 1) != 1)
        EXPECT(false);
      close(pfd[0]);
      kill(p, SIGKILL);
      int stt = 0;
      waitpid(p, &stt, 0);
      const std::string k = key(fp, steer + "-kill", toks(512, 4));
      struct stat st;
      EXPECT(stat((g_dir + "/" + k + kExt).c_str(), &st) != 0); // never visible
      Index ix;
      ix.init(g_dir);
      (void)ix.entries(fp); // a scan must not choke on the private name
      Map m;
      EXPECT(open_map(g_dir + "/" + k + kExt, fp, m) == Rd::MISS);
    }
  }

  // 6. unlink while mapped: the mapping stays valid and readable
  {
    auto t = toks(512, 8);
    Desc d = make_desc(fp, steer, t, 2, 1 << 16, 1 << 14);
    auto pay = payloads(d);
    std::string k;
    EXPECT(write_obj(d, fp, steer, t, pay, &k));
    Map m;
    EXPECT(open_map(g_dir + "/" + k + kExt, fp, m) == Rd::OK);
    EXPECT(concurrent_unlink_test(g_dir + "/" + k + kExt, fp));
    EXPECT(m.verify_kv(0) && m.verify_st(0));
    EXPECT(same_payload(m, pay));
    Map again;
    EXPECT(open_map(g_dir + "/" + k + kExt, fp, again) == Rd::MISS);
  }

  // 7. inotify index: a publish becomes visible, a delete disappears; a forced
  // rescan heals a missed event
  {
    Index ix;
    ix.init(g_dir);
    auto t = toks(64, 12);
    Desc d = make_desc(fp, steer, t, 1, 1 << 12, 1 << 12);
    auto pay = payloads(d);
    std::string k;
    EXPECT(write_obj(d, fp, steer, t, pay, &k));
    bool seen = false;
    for (int i = 0; i < 50 && !seen; i++) {
      for (auto &e : ix.entries(fp, 0.05))
        if (e.toks == t)
          seen = true;
      usleep(2000);
    }
    EXPECT(seen);
    EXPECT(unlink((g_dir + "/" + k + kExt).c_str()) == 0);
    bool gone = false;
    for (int i = 0; i < 50 && !gone; i++) {
      gone = true;
      for (auto &e : ix.entries(fp, 0.05))
        if (e.toks == t)
          gone = false;
      if (!gone)
        usleep(2000);
    }
    EXPECT(gone);
    ix.rescan(fp); // the periodic path
    for (auto &e : ix.entries(fp, 0.0))
      EXPECT(e.toks != t);
  }

  // 8. evictor: one flock'd role, global cap enforced oldest-first; a reader
  // never deletes
  {
    for (int i = 0; i < 5; i++) {
      auto t = toks(64, 20 + i);
      Desc d = make_desc(fp, steer, t, 1, 1 << 13, 1 << 12);
      EXPECT(write_obj(d, fp, steer, t, payloads(d)));
      usleep(1100000); // distinct mtimes for the LRU order
    }
    Evictor e1, e2;
    EXPECT(e1.acquire(g_dir));
    EXPECT(!e2.acquire(g_dir)); // the flock'd role is single
    const uint64_t cap = 2 * (uint64_t)(1 << 13) + 4096;
    const uint64_t removed = e1.enforce(cap);
    EXPECT(removed > 0);
    uint64_t total = 0, n = 0;
    DIR *dir = opendir(g_dir.c_str());
    if (dir) {
      while (dirent *de = readdir(dir)) {
        const std::string nm = de->d_name;
        if (nm.size() > strlen(kExt) &&
            nm.compare(nm.size() - strlen(kExt), strlen(kExt), kExt) == 0) {
          struct stat st;
          if (stat((g_dir + "/" + nm).c_str(), &st) == 0) {
            total += (uint64_t)st.st_size;
            n++;
          }
        }
      }
      closedir(dir);
    }
    EXPECT(total <= cap);
    EXPECT(n >= 1);
  }

  // 9. registration hook: success counts and pins, failure falls back without
  // an error
  {
    static int reg_mode = 0;
    set_hooks(
        [](void *p, size_t, void *ctx, int *err) -> bool {
          (void)p;
          if (*(int *)ctx == 0) {
            *err = 1;
            return false;
          }
          return true;
        },
        [](void *, size_t, void *) {}, &reg_mode, 1ull << 30);
    auto t = toks(96, 30);
    Desc d = make_desc(fp, steer, t, 1, 1 << 13, 1 << 12);
    std::string k;
    EXPECT(write_obj(d, fp, steer, t, payloads(d), &k));
    Map m;
    EXPECT(open_map(g_dir + "/" + k + kExt, fp, m) == Rd::OK);
    reg_mode = 0;
    auto fails0 = hooks().n_reg_fail.load();
    EXPECT(!m.attach());
    EXPECT(hooks().n_reg_fail.load() == fails0 + 1);
    Map m2;
    EXPECT(open_map(g_dir + "/" + k + kExt, fp, m2) == Rd::OK);
    reg_mode = 1;
    EXPECT(m2.attach());
    EXPECT(m2.registered());

    // A card without HostRegisterReadOnly refuses the PROT_READ view: attach
    // retries on a writable view of the same pages, then drops write access.
    static int calls = 0;
    set_hooks(
        [](void *, size_t, void *, int *err) -> bool {
          if (calls++ == 0) {
            *err = 1;
            return false;
          }
          return true;
        },
        [](void *, size_t, void *) {}, nullptr, 1ull << 30);
    auto perms_at = [](const void *p) {
      FILE *f = fopen("/proc/self/maps", "r");
      char line[512], perms[8] = "";
      unsigned long lo, hi;
      while (f && fgets(line, sizeof line, f))
        if (sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) == 3 &&
            (unsigned long)p >= lo && (unsigned long)p < hi)
          break;
      if (f) fclose(f);
      return std::string(perms);
    };
    auto rw0 = hooks().n_reg_rw.load();
    Map m3;
    EXPECT(open_map(g_dir + "/" + k + kExt, fp, m3) == Rd::OK);
    EXPECT(m3.attach());
    EXPECT(calls == 2);
    EXPECT(hooks().n_reg_rw.load() == rw0 + 1);
    EXPECT(perms_at(m3.kv(0)) == "r--s");
    EXPECT(m3.verify_kv(0));

    // QE_SHARED_L2_RW_PIN=0: one refused attempt, then the private-copy fallback.
    calls = 0;
    hooks().rw_pin = false;
    Map m4;
    EXPECT(open_map(g_dir + "/" + k + kExt, fp, m4) == Rd::OK);
    EXPECT(!m4.attach());
    EXPECT(calls == 1);
    EXPECT(perms_at(m4.kv(0)) == "r--s");
    hooks().rw_pin = true;
    set_hooks(nullptr, nullptr, nullptr, 0);
  }

  // 10. measurements: bytes per prefix, build/publish and map+verify times (n,
  // p50/p95); GPU numbers are pending
  {
    const int ng = 4, tokens = 8192;
    const size_t kv_per_card = (size_t)((tokens + ng - 1) / ng) * 13 *
                               576; // KV4: 576 B/token/layer, 13 layers
    const size_t st_per_card = 30u
                               << 20; // committed state, measured ~30 MB/card
    auto t = toks(tokens, 77);
    Desc d = make_desc(fp, steer + "-measure", t, ng, kv_per_card, st_per_card);
    auto pay = payloads(d);
    std::vector<double> w, r;
    std::string k;
    for (int i = 0; i < 7; i++) {
      const double t0 = now_s();
      EXPECT(write_obj(d, fp, steer + "-measure", t, pay, &k));
      w.push_back((now_s() - t0) * 1000);
      const double t1 = now_s();
      Map m;
      EXPECT(open_map(g_dir + "/" + k + kExt, fp, m) == Rd::OK);
      EXPECT(m.verify_all());
      r.push_back((now_s() - t1) * 1000);
    }
    const double mb = (double)((kv_per_card + st_per_card) * ng) / 1e6;
    printf("MEASURE prefix tokens=%d cards=%d object=%.1f MB kv/card=%.2f MB "
           "state/card=%.2f MB\n",
           tokens, ng, (double)(kv_per_card + st_per_card) * ng / 1e6,
           kv_per_card / 1e6, st_per_card / 1e6);
    printf("MEASURE publish_ms n=%zu p50=%.2f p95=%.2f (%.1f MB/s p50)\n",
           w.size(), q(w, 0.5), q(w, 0.95),
           mb / std::max(0.001, q(w, 0.5) / 1000));
    printf("MEASURE map_verify_ms n=%zu p50=%.2f p95=%.2f (%.1f MB/s p50)\n",
           r.size(), q(r, 0.5), q(r, 0.95),
           mb / std::max(0.001, q(r, 0.5) / 1000));
    const double prefill_tok_s = getenv("QE_TEST_PREFILL_TOK_S")
                                     ? atof(getenv("QE_TEST_PREFILL_TOK_S"))
                                     : 100.0;
    printf("MEASURE prefill_estimate_ms tokens=%d at %.0f tok/s = %.1f "
           "(assumption, env QE_TEST_PREFILL_TOK_S)\n",
           tokens, prefill_tok_s, tokens / prefill_tok_s * 1000);
    printf("MEASURE gpu: cudaHostRegister time and H2D DMA bandwidth require a "
           "GPU; pending (GPU 7, <=256 MB, <=60 s, "
           "only at queue 0)\n");
  }

  // 11. async queue: writes run on the worker thread, never on the caller, one
  // writer at a time, accepted items keep push order
  {
    const std::thread::id caller = std::this_thread::get_id();
    AwSink sink; // gates open: the callback never parks here
    uint64_t dropped_before = 0;
    {
      AsyncWriter<int> w(4, aw_write_int, &sink);
      EXPECT(w.try_push(1));
      EXPECT(w.try_push(2));
      EXPECT(w.try_push(3));
      EXPECT(sink.until(5.0, [&] { return sink.entered == 3; }));
      w.stop(); // drain: every accepted item is written before the join
      dropped_before = w.dropped();
      EXPECT(
          (sink.values == std::vector<int>{1, 2, 3})); // FIFO, single consumer
      EXPECT(sink.tids.size() == 3);
      for (size_t i = 0; i < sink.tids.size(); i++) {
        EXPECT(sink.tids[i] != caller);       // away from the publishing thread
        EXPECT(sink.tids[i] == sink.tids[0]); // and always the same worker
      }
    } // ~AsyncWriter stops again: the second stop must stay clean
    EXPECT(dropped_before == 0);
  }

  // 12. bounded queue: a full queue rejects and counts, stop(drain=true) writes
  // everything already accepted and nothing else
  {
    AwSink sink;
    sink.finish = false; // park the worker inside the first callback
    size_t accepted = 0;
    {
      AsyncWriter<int> w(3, aw_write_int, &sink);
      EXPECT(w.try_push(1));
      // the worker has taken item 1 and sits in the callback, so the queue is
      // empty here and the capacity below is the whole queue
      EXPECT(sink.until(5.0, [&] { return sink.entered == 1; }));
      EXPECT(w.try_push(2));
      EXPECT(w.try_push(3));
      EXPECT(w.try_push(4)); // queue now holds exactly capacity items
      EXPECT(w.dropped() == 0);
      EXPECT(!w.try_push(5)); // full: rejected, not queued
      EXPECT(w.dropped() == 1);
      EXPECT(!w.try_push(6));
      EXPECT(w.dropped() == 2);
      accepted = 4;
      sink.finish_now(); // release on every path so the join cannot hang
      w.stop();          // drain the accepted items, then join
      EXPECT(sink.values.size() == accepted);
      EXPECT((sink.values == std::vector<int>{1, 2, 3, 4}));
      EXPECT(w.dropped() == 2); // the two rejections, and only those
    }
  }

  // 13. a queued shared_ptr keeps its payload alive for the whole callback and
  // is released when the worker is done with it
  {
    std::atomic<int> live{0};
    AwSink sink;
    sink.show = false;   // park before the callback inspects the item
    sink.finish = false; // park again before the callback returns
    {
      AsyncWriter<std::shared_ptr<AwPayload>> w(2, aw_write_payload, &sink);
      auto p = std::make_shared<AwPayload>(&live, kAwText);
      EXPECT(live.load() == 1);
      EXPECT(w.try_push(p));
      p.reset();
      EXPECT(live.load() == 1); // the queue owns the only reference
      EXPECT(sink.until(5.0, [&] { return sink.entered == 1; }));
      EXPECT(live.load() == 1); // alive while the worker holds it
      sink.show_now();          // the callback now reads through the pointer
      EXPECT(sink.until(5.0, [&] { return sink.alive_at_entry == 1; }));
      EXPECT(sink.payload_intact); // payload reaches the writer intact
      EXPECT(live.load() == 1);    // and outlives the inspection
      sink.finish_now();           // callback returns; release on every path
      EXPECT(sink.until(5.0, [&] { return sink.alive_at_exit == 1; }));
      EXPECT(sink.payload_text == kAwText);
      w.stop();
      EXPECT(live.load() == 0); // the drained item was released, nothing leaked
      EXPECT(w.dropped() == 0);
    }
    EXPECT(live.load() == 0);
  }

  // 14. async publication of a real object: the item owns the description and
  // the shards, the worker builds and publishes on its own thread, and the
  // object is only visible through open_map once that writer is done
  {
    auto t = toks(256, 41);
    Desc d = make_desc(fp, steer + "-async", t, 2, 1 << 15, 1 << 13);
    auto pay = payloads(d);
    const std::string k = key(d.fp, d.steer, d.toks);
    const std::string want = path_of(g_dir, k);
    std::atomic<int> live{0};
    ApSink sink;
    {
      AsyncWriter<std::shared_ptr<ApJob>> w(2, ap_publish_job, &sink);
      auto job = std::make_shared<ApJob>(&live, d, pay, k);
      EXPECT(live.load() == 1);
      EXPECT(w.try_push(job));
      job.reset(); // the queue owns the description and the shards now
      EXPECT(live.load() == 1);
      EXPECT(sink.until(5.0, [&] { return sink.entered == 1; }));
      Map too_early;
      EXPECT(open_map(want, fp, too_early) == Rd::MISS); // not built yet
      EXPECT(sink.until(5.0, [&] { return sink.alive_at_entry == 1; }));
      EXPECT(sink.shards_match);
      sink.release_now(); // the worker now builds, copies and publishes
      EXPECT(sink.until(5.0, [&] { return sink.published; }));
      EXPECT(sink.ok);
      EXPECT(sink.err == 0);
      EXPECT(sink.path == want);
      EXPECT(sink.seen_hash == pay_hash(pay)); // bytes reach the writer intact
      EXPECT(sink.seen_bytes == ((size_t)d.kv[0].len + d.st[0].len) * 2);
      EXPECT(sink.alive_at_exit == 1); // and outlive the publish
      w.stop(); // drain, then join: the callback ran to completion
      EXPECT(w.dropped() == 0);
      EXPECT(sink.worker != std::this_thread::get_id()); // away from the caller
      EXPECT(live.load() == 0); // released with the callback, nothing leaked
    }
    Map m;
    const char *why = "";
    EXPECT(open_map(want, fp, m, &why) == Rd::OK);
    EXPECT(m.desc().steer == steer + "-async");
    EXPECT(m.desc().toks == t);
    EXPECT(m.desc().ng == 2);
    EXPECT(m.desc().writer == 7);
    EXPECT(m.verify_all());
    EXPECT(same_payload(m, pay));
    Map other;
    EXPECT(open_map(want, "other-fingerprint", other) == Rd::OTHER);
  }

  cleanup_test_dir();
  printf(fails ? "FAILED %d\n" : "OK shared_prefix_unit\n", fails);
  return fails ? 1 : 0;
}
