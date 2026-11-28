// CPU test of the tiered archive store (src/archive_tiers.h): real files in temp directories, injected clock and
// filesystem usage. Per tier overfill stays under the cap; delete by session; TTL expiry; latest checkpoint only.
#include "archive_tiers.h"

#include <dirent.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <string>

using qe_archive_store::TieredStore;

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

static std::string mkdtemp_dir(const char* tag) {
    std::string t = std::string("/tmp/q27-tiers-") + tag + "-XXXXXX";
    std::vector<char> b(t.begin(), t.end());
    b.push_back(0);
    return mkdtemp(b.data()) ? std::string(b.data()) : std::string();
}

// bytes of the regular files in a directory (what the filesystem really holds)
static size_t dir_bytes(const std::string& d, int* count = nullptr) {
    size_t sum = 0;
    int n = 0;
    if (DIR* dir = opendir(d.c_str())) {
        while (dirent* e = readdir(dir)) {
            struct stat st{};
            const std::string p = d + "/" + e->d_name;
            if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) sum += st.st_size, n++;
        }
        closedir(dir);
    }
    if (count) *count = n;
    return sum;
}

static void rm_dir(const std::string& d) {
    if (DIR* dir = opendir(d.c_str())) {
        while (dirent* e = readdir(dir)) unlink((d + "/" + e->d_name).c_str());
        closedir(dir);
    }
    rmdir(d.c_str());
}

// characterization: reset_objects forgets the objects, the files stay, the tier and the eviction counters are kept
// [al0-0038] A sibling arm's archive (QE_DISK_CACHE_READ_DIRS) is a protected view: no selection rule ever names it,
// and a failed read-back never unlinks it. Mirrors Engine::archive_views (protect = pinned || hold || disk_readonly).
static void test_read_only_sibling_never_targeted() {
    using namespace qe_archive_store;
    const uint64_t s = 0x5e55;
    // index 0: our older checkpoint; 1: a sibling arm's file of the same session, expired and newest; 2: our newest
    std::vector<View> v = {{s, 100, 1, false}, {s, 100, 9, true}, {s, 0, 5, false}};
    const int64_t now = 1000;  // every expiry above is in the past
    for (size_t i : delete_targets(v, s)) EXPECT(i != 1);
    EXPECT(delete_targets(v, s) == (std::vector<size_t>{0, 2}));
    for (size_t i : expired_targets(v, now)) EXPECT(i != 1);
    EXPECT(expired_targets(v, now) == std::vector<size_t>{0});
    for (size_t i : older_in_session(v, 2)) EXPECT(i != 1);  // keep our newest: the sibling is not "older" prey
    EXPECT(older_in_session(v, 2) == std::vector<size_t>{0});
    // a protected newer sibling never makes our own checkpoint "superseded" (it would skip our flush)
    EXPECT(!superseded_in_session(std::vector<View>{{s, 0, 5, false}, {s, 0, 9, true}}, 0));
    // a failed read-back: our own file may go, a read-only sibling file never
    EXPECT(may_unlink_unreadable(false));
    EXPECT(!may_unlink_unreadable(true));
}

static void test_reset_objects() {
    TieredStore s;
    const int64_t now = 1000000;
    s.clock = [&] { return now; };
    const std::string d = mkdtemp_dir("ro");
    const int t = s.add_tier("ram", d, true, 250);
    std::string blob(100, 'x');
    EXPECT(s.put(t, "k1.qkv", blob.data(), 100, 0x71));
    EXPECT(s.put(t, "k2.qkv", blob.data(), 100, 0x72));
    EXPECT(s.put(t, "k3.qkv", blob.data(), 100, 0x73));  // the third over the 250 cap: the first is evicted
    EXPECT(s.tier(t).evictions == 1 && s.tier(t).objects == 2);
    s.reset_objects();
    EXPECT(s.tier(t).objects == 0 && s.tier(t).bytes == 0);  // the objects are forgotten
    EXPECT(s.tier(t).evictions == 1);  // the eviction counter is kept
    EXPECT(s.tiers() == 1 && s.find_tier("ram") == 0);  // the tier is kept
    EXPECT(!s.has(d + "/k2.qkv") && !s.has(d + "/k3.qkv"));
    EXPECT(access((d + "/k2.qkv").c_str(), F_OK) == 0 && access((d + "/k3.qkv").c_str(), F_OK) == 0);  // the files stay
    int files = 0;
    EXPECT(dir_bytes(d, &files) == 200 && files == 2);
    rm_dir(d);
}

// characterization: get returns the registered object, or nullptr when absent
static void test_get() {
    TieredStore s;
    const int64_t now = 1000000;
    s.clock = [&] { return now; };
    const std::string d = mkdtemp_dir("gt");
    const int t = s.add_tier("ram", d, true, 10000);
    std::string blob(100, 'x');
    EXPECT(s.put(t, "g1.qkv", blob.data(), 100, 0x81, 1234));
    const qe_archive_store::Obj* o = s.get(d + "/g1.qkv");
    EXPECT(o != nullptr && o->tier == t && o->bytes == 100 && o->session == 0x81 && o->expires == now + 1234 && o->seq == 1);
    EXPECT(s.get(d + "/nope.qkv") == nullptr);
    rm_dir(d);
}

// characterization: evict runs the LRU loop until the tier fits, and returns the evicted count
static void test_evict_count() {
    TieredStore s;
    const int64_t now = 1000000;
    s.clock = [&] { return now; };
    const std::string d = mkdtemp_dir("ev");
    const int t = s.add_tier("ram", d, true, 250);
    std::string blob(100, 'x');
    for (int i = 1; i <= 3; i++) EXPECT(s.put(t, "e" + std::to_string(i) + ".qkv", blob.data(), 100, 0));
    EXPECT(s.tier(t).objects == 2 && s.tier(t).evictions == 1);  // the third put already evicted e1
    EXPECT(s.evict(t, 150) == 1);  // 200 + 150 > 250: e2 goes, then it fits
    EXPECT(s.tier(t).objects == 1 && s.tier(t).evictions == 2 && !s.has(d + "/e2.qkv"));
    EXPECT(s.evict(t, 0) == 0);  // already fits: nothing goes
    EXPECT(s.tier(t).objects == 1 && s.tier(t).evictions == 2);
    EXPECT(s.evict(t, 1000) == 1);  // evicts until the tier is empty and returns the count
    EXPECT(s.tier(t).objects == 0 && s.tier(t).evictions == 3);
    int files = 0;
    EXPECT(dir_bytes(d, &files) == 0 && files == 0);
    rm_dir(d);
}

// characterization: delete_session goes through the private unlink_remove: the file is unlinked and the index entry gone
static void test_delete_session_remove_path() {
    TieredStore s;
    const int64_t now = 1000000;
    s.clock = [&] { return now; };
    const std::string d = mkdtemp_dir("ds");
    const int t = s.add_tier("ram", d, true, 10000);
    std::string blob(100, 'x');
    EXPECT(s.put(t, "s1.qkv", blob.data(), 100, 0x91));
    // a second object of the same session, registered the way the restore does (a put would supersede s1)
    const int fd = ::open((d + "/s2.qkv").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    EXPECT(fd >= 0 && ::write(fd, blob.data(), blob.size()) == (ssize_t)blob.size());
    ::close(fd);
    qe_archive_store::Obj o;
    o.tier = t, o.bytes = 100, o.session = 0x91;
    s.add(d + "/s2.qkv", o);
    EXPECT(s.put(t, "s3.qkv", blob.data(), 100, 0x92));
    EXPECT(s.session_paths(0x91).size() == 2);
    EXPECT(s.delete_session(0x91) == 2);
    EXPECT(!s.has(d + "/s1.qkv") && !s.has(d + "/s2.qkv") && s.has(d + "/s3.qkv"));
    EXPECT(access((d + "/s1.qkv").c_str(), F_OK) != 0 && access((d + "/s2.qkv").c_str(), F_OK) != 0);
    EXPECT(s.tier(t).objects == 1 && s.tier(t).bytes == 100 && s.tier(t).evictions == 0);  // a delete is not an eviction
    EXPECT(s.delete_session(0x91) == 0);  // the session is empty: nothing to delete
    rm_dir(d);
}

int main() {
    const std::string ram = mkdtemp_dir("ram"), nvme = mkdtemp_dir("nvme");
    EXPECT(!ram.empty() && !nvme.empty());
    int64_t now = 1000000;
    TieredStore st;
    st.clock = [&] { return now; };
    const int r = st.add_tier("ram", ram, true, 1000);
    const int n = st.add_tier("nvme", nvme, false, 4000, 85);
    EXPECT(st.find_tier("nvme") == n && st.find_tier("none") == -1);
    std::string blob(300, 'x');

    // overfill each tier with 40 objects: bytes and the real directory stay at or under the cap, evictions are counted
    for (int i = 0; i < 40; i++) {
        EXPECT(st.put(r, "r" + std::to_string(i) + ".qkv", blob.data(), 300));
        EXPECT(st.tier(r).bytes <= 1000 && dir_bytes(ram) <= 1000);
    }
    for (int i = 0; i < 40; i++) {
        EXPECT(st.put(n, "n" + std::to_string(i) + ".qkv", blob.data(), 300));
        EXPECT(st.tier(n).bytes <= 4000 && dir_bytes(nvme) <= 4000);
    }
    int files = 0;
    dir_bytes(ram, &files);
    EXPECT(files == 3 && st.tier(r).objects == 3 && st.tier(r).bytes == 900);
    EXPECT(st.tier(r).evictions == 37 && st.tier(n).evictions == 27);
    EXPECT(!st.put(r, "big.qkv", blob.data(), 1001));  // larger than the cap: refused, nothing written
    EXPECT(dir_bytes(ram) == 900);
    EXPECT(st.get_path(r, "r39.qkv") == ram + "/r39.qkv" && st.get_path(r, "r0.qkv").empty());

    // free-space floor: a non-RAM tier refuses a write that would pass 85 % usage; the RAM tier ignores it
    size_t total = 1000, avail = 160;
    st.usage = [&](const std::string&, size_t& t, size_t& a) { t = total, a = avail; return true; };
    EXPECT(!st.put(n, "floor.qkv", blob.data(), 100));  // 840 used + 100 > 850
    EXPECT(st.put(n, "floor.qkv", blob.data(), 10));
    EXPECT(st.put(r, "ramfloor.qkv", blob.data(), 100));

    // delete by session: only that session's objects go (both tiers), the others stay
    TieredStore s2;
    s2.clock = [&] { return now; };
    const std::string d2 = mkdtemp_dir("s2"), d3 = mkdtemp_dir("s3");
    const int a = s2.add_tier("ram", d2, true, 10000), b = s2.add_tier("nvme", d3, false, 10000);
    s2.usage = [](const std::string&, size_t& t, size_t& av) { t = 1 << 30, av = 1 << 30; return true; };
    EXPECT(s2.put(a, "a1.qkv", blob.data(), 100, 0xaaa) && s2.put(b, "b1.qkv", blob.data(), 100, 0xbbb));
    EXPECT(s2.put(b, "c1.qkv", blob.data(), 100, 0) && s2.put(a, "pin.qkv", blob.data(), 100, 0));
    EXPECT(s2.delete_session(0) == 0);  // session 0 is "none": never a delete target
    EXPECT(s2.delete_session(0xaaa) == 1);
    EXPECT(s2.tier(a).objects == 1 && s2.tier(a).bytes == 100 && s2.tier(a).evictions == 0);
    EXPECT(access((d2 + "/a1.qkv").c_str(), F_OK) != 0 && access((d3 + "/b1.qkv").c_str(), F_OK) == 0);
    EXPECT(s2.delete_session(0xbbb) == 1 && s2.tier(b).objects == 1);
    EXPECT(s2.delete_session(0xbbb) == 0);

    // latest checkpoint only: each put of a session removes its older checkpoints, wherever they sit
    TieredStore s3;
    s3.clock = [&] { return now; };
    const int p3 = s3.add_tier("ram", d2, true, 10000), q3 = s3.add_tier("nvme", d3, false, 10000);
    s3.usage = [](const std::string&, size_t& t, size_t& av) { t = 1 << 30, av = 1 << 30; return true; };
    EXPECT(s3.put(p3, "k1.qkv", blob.data(), 100, 0x51));
    EXPECT(s3.put(q3, "k2.qkv", blob.data(), 150, 0x51));
    EXPECT(s3.put(p3, "k3.qkv", blob.data(), 200, 0x51));
    EXPECT(s3.put(p3, "other.qkv", blob.data(), 100, 0x52));
    EXPECT(s3.session_paths(0x51).size() == 1 && s3.has(d2 + "/k3.qkv") && !s3.has(d2 + "/k1.qkv") && !s3.has(d3 + "/k2.qkv"));
    EXPECT(access((d2 + "/k1.qkv").c_str(), F_OK) != 0 && access((d3 + "/k2.qkv").c_str(), F_OK) != 0);
    EXPECT(s3.has(d2 + "/other.qkv"));

    // TTL expiry: objects past their class are dropped by gc (and before a put makes room)
    TieredStore s4;
    s4.clock = [&] { return now; };
    const int t4 = s4.add_tier("ram", d2, true, 10000);
    EXPECT(s4.put(t4, "short.qkv", blob.data(), 100, 0x61, 60));
    EXPECT(s4.put(t4, "long.qkv", blob.data(), 100, 0x62, 3600));
    EXPECT(s4.put(t4, "forever.qkv", blob.data(), 100, 0x63, 0));
    EXPECT(s4.gc(now + 30) == 0);
    EXPECT(s4.gc(now + 60) == 1 && !s4.has(d2 + "/short.qkv") && access((d2 + "/short.qkv").c_str(), F_OK) != 0);
    now += 4000;
    EXPECT(s4.put(t4, "new.qkv", blob.data(), 100, 0x64, 60));  // put runs the gc first
    EXPECT(!s4.has(d2 + "/long.qkv") && s4.has(d2 + "/forever.qkv") && s4.has(d2 + "/new.qkv"));
    EXPECT(s4.tier(t4).objects == 2 && s4.tier(t4).bytes == 200);

    // engine-side selection rules (delete_session / archive_gc / flush latest-only), driven the way Engine drives them:
    // a snapshot of View rows, victims applied to real files through the store
    {
        using namespace qe_archive_store;
        TieredStore s5;
        s5.clock = [&] { return now; };
        const std::string d5 = mkdtemp_dir("s5");
        const int t5 = s5.add_tier("ram", d5, true, 100000);
        struct Row { std::string name; View v; };
        std::vector<Row> rows = {
            {"a_old", {0xa1, 0, 1, false}}, {"a_new", {0xa1, 0, 5, false}}, {"b", {0xb1, now + 10, 3, false}},
            {"pinned", {0xa1, 0, 9, true}}, {"anon", {0, 0, 4, false}}, {"b_exp", {0xb2, now - 1, 2, false}}};
        for (auto& r : rows) EXPECT(s5.put(t5, r.name + ".qkv", blob.data(), 100, 0, 0));
        auto views = [&] { std::vector<View> v; for (auto& r : rows) v.push_back(r.v); return v; };
        auto drop = [&](const std::vector<size_t>& idx) {  // highest index first so the rest stay valid
            for (size_t k = idx.size(); k-- > 0;) {
                s5.remove(d5 + "/" + rows[idx[k]].name + ".qkv", false);
                unlink((d5 + "/" + rows[idx[k]].name + ".qkv").c_str());
                rows.erase(rows.begin() + idx[k]);
            }
        };
        EXPECT(superseded_in_session(views(), 0) && !superseded_in_session(views(), 1));  // a_old: a_new is newer
        EXPECT(!superseded_in_session(views(), 4));  // anonymous: never superseded
        EXPECT(older_in_session(views(), 1) == std::vector<size_t>{0});  // pinned (protected) is not "older"
        EXPECT(delete_targets(views(), 0).empty());  // session 0 is no target
        EXPECT(expired_targets(views(), now) == std::vector<size_t>{5});
        drop(expired_targets(views(), now));
        EXPECT(s5.tier(t5).objects == 5);
        drop(delete_targets(views(), 0xa1));  // a_old + a_new go, the pinned row of the same session stays
        EXPECT(rows.size() == 3 && s5.has(d5 + "/pinned.qkv") && !s5.has(d5 + "/a_new.qkv"));
        EXPECT(access((d5 + "/a_old.qkv").c_str(), F_OK) != 0 && access((d5 + "/pinned.qkv").c_str(), F_OK) == 0);
        EXPECT(s5.tier(t5).bytes == 300 && s5.tier(t5).evictions == 0);
        now += 20;
        drop(expired_targets(views(), now));  // b ran out
        EXPECT(rows.size() == 2 && s5.tier(t5).objects == 2);
        rm_dir(d5);
    }

    // characterization: the methods the coverage file marks uncovered, one test function each
    test_read_only_sibling_never_targeted();
    test_reset_objects();
    test_get();
    test_evict_count();
    test_delete_session_remove_path();

    for (const auto& d : {ram, nvme, d2, d3}) rm_dir(d);
    if (fails) return 1;
    std::printf("archive_tiers_unit: ok\n");
    return 0;
}
