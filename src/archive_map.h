#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Read-only MAP_SHARED view of one persistent archive object. The mapping stays valid after the
// caller closes the descriptor or an LRU evictor unlinks the file; archive_load keeps the view
// alive through validation and the copy into the engine's pinned HostPool buffers.
class ArchiveMap {
public:
    ArchiveMap() = default;
    ArchiveMap(const ArchiveMap&) = delete;
    ArchiveMap& operator=(const ArchiveMap&) = delete;
    ~ArchiveMap() { reset(); }

    bool open_readonly(const std::string& path) {
        reset();
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 ||
            static_cast<std::uintmax_t>(st.st_size) > std::numeric_limits<size_t>::max()) {
            ::close(fd);
            return false;
        }
        const size_t size = static_cast<size_t>(st.st_size);
        void* view = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        const int map_errno = errno;
        ::close(fd);
        if (view == MAP_FAILED) {
            errno = map_errno;
            return false;
        }
        data_ = static_cast<const char*>(view);
        size_ = size;
        return true;
    }

    const char* data() const { return data_; }
    size_t size() const { return size_; }
    explicit operator bool() const { return data_ != nullptr; }

private:
    void reset() {
        if (data_) munmap(const_cast<char*>(data_), size_);
        data_ = nullptr;
        size_ = 0;
    }

    const char* data_ = nullptr;
    size_t size_ = 0;
};
