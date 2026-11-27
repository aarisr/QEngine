// CPU test of the tiered archive store (src/archive_tiers.h): real files in temp directories, injected clock and
// filesystem usage. Per tier overfill stays under the cap; delete by session; TTL expiry; latest checkpoint only.
#include "archive_tiers.h"

#include <dirent.h>
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

    for (const auto& d : {ram, nvme, d2, d3}) rm_dir(d);
    if (fails) return 1;
    std::printf("archive_tiers_unit: ok\n");
    return 0;
}
