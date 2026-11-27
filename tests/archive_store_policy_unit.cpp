#include "archive_store_policy.h"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <initializer_list>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); fails++; } } while (0)

int main() {
    using qe_archive_store::below_usage_limit;
    using qe_archive_store::fits;
    using qe_archive_store::nvme_cap;

    EXPECT(fits(16, 12, 4));
    EXPECT(!fits(16, 13, 4));
    EXPECT(!fits(16, 0, 17));
    EXPECT(!fits(16, std::numeric_limits<size_t>::max(), 1));
    EXPECT(below_usage_limit(100, 20, 5));
    EXPECT(!below_usage_limit(100, 20, 6));
    EXPECT(!below_usage_limit(100, 0, 1));

    // Repeated writes overfill a real CPU-only archive directory; oldest objects are evicted before each write.
    char dir_template[] = "/tmp/q27-store-cap-XXXXXX";
    char* dir = mkdtemp(dir_template);
    EXPECT(dir != nullptr);
    if (!dir) return 1;
    constexpr size_t cap = 23;
    struct Stored { std::string path; size_t bytes; };
    std::vector<Stored> stored;
    size_t used = 0;
    int seq = 0;
    for (size_t bytes : {7, 8, 9, 10, 11, 12, 13}) {
        while (!fits(cap, used, bytes) && !stored.empty()) {
            EXPECT(unlink(stored.front().path.c_str()) == 0);
            used -= stored.front().bytes;
            stored.erase(stored.begin());
        }
        if (fits(cap, used, bytes)) {
            const std::string path = std::string(dir) + "/object-" + std::to_string(seq++);
            const int fd = open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
            EXPECT(fd >= 0);
            if (fd < 0) break;
            EXPECT(ftruncate(fd, bytes) == 0);
            EXPECT(fsync(fd) == 0);
            EXPECT(close(fd) == 0);
            stored.push_back({path, bytes});
            used += bytes;
        }
        size_t on_disk = 0;
        for (const auto& item : stored) {
            struct stat st{};
            if (stat(item.path.c_str(), &st) == 0)
                on_disk += static_cast<size_t>(st.st_size);
            else
                EXPECT(false);
        }
        EXPECT(on_disk == used);
        EXPECT(on_disk <= cap);
    }
    EXPECT(!fits(cap, used, cap + 1));
    for (const auto& item : stored) EXPECT(unlink(item.path.c_str()) == 0);
    EXPECT(rmdir(dir) == 0);
    EXPECT(nvme_cap(40, 30, 10) == 100);
    EXPECT(nvme_cap(std::numeric_limits<size_t>::max(), 1, 1) == std::numeric_limits<size_t>::max());
    std::printf(fails ? "archive_store_policy_unit: %d FAILED\n" : "archive_store_policy_unit: ok\n", fails);
    return fails ? 1 : 0;
}
