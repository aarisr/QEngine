#pragma once

#include <cstddef>
#include <limits>

namespace qe_archive_store {

// Overflow-safe cap check used before reserving bytes for a persistent archive write.
inline bool fits(size_t cap, size_t used, size_t need) {
    return need <= cap && used <= cap - need;
}

// The NVMe store must hold one full eviction of the KV pool, archive and pinned-prefix budgets,
// plus 25% headroom. Saturate rather than wrap if a configuration is unreasonable.
inline size_t nvme_cap(size_t kv_pool, size_t archive, size_t pins) {
    const size_t max = std::numeric_limits<size_t>::max();
    if (kv_pool > max - archive) return max;
    const size_t base = kv_pool + archive;
    if (base > max - pins) return max;
    const size_t total = base + pins;
    return total > max - total / 4 ? max : total + total / 4;
}

// Conservatively keep a filesystem at or below `percent` usage after a write.
inline bool below_usage_limit(size_t total, size_t available, size_t need, unsigned percent = 85) {
    if (total == 0 || available > total || percent > 100) return false;
    const size_t used = total - available;
    const size_t limit = (total / 100) * percent + ((total % 100) * percent) / 100;
    return used <= limit && need <= limit - used;
}

}  // namespace qe_archive_store
