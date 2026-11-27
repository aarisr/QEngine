#include "archive_map.h"

#include <cassert>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <unistd.h>

int main() {
    char path[] = "/tmp/q27-archive-map-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    const std::string payload = "versioned archive bytes survive a process restart";
    assert(write(fd, payload.data(), payload.size()) == static_cast<ssize_t>(payload.size()));
    assert(fsync(fd) == 0);
    close(fd);

    ArchiveMap map;
    assert(map.open_readonly(path));
    assert(map.size() == payload.size());
    assert(std::memcmp(map.data(), payload.data(), payload.size()) == 0);
    assert(unlink(path) == 0);  // a shared mapping remains readable after LRU unlink
    assert(map.size() == payload.size());
    assert(std::memcmp(map.data(), payload.data(), payload.size()) == 0);

    assert(!map.open_readonly(path));  // failed reopen clears the previous mapping
    assert(!map && map.size() == 0);
    char empty[] = "/tmp/q27-archive-map-empty-XXXXXX";
    const int empty_fd = mkstemp(empty);
    assert(empty_fd >= 0);
    close(empty_fd);
    assert(!map.open_readonly(empty));
    assert(unlink(empty) == 0);
    return 0;
}
