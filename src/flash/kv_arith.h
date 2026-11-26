#pragma once
#include <cstdint>

namespace flash {
namespace kv_math {

#if defined(__CUDACC__)
#define FLASH_KV_HOST_DEVICE __host__ __device__
#else
#define FLASH_KV_HOST_DEVICE
#endif

// period and block are positive engine constants; quotient/remainder avoids overflow-prone ceil additions.
FLASH_KV_HOST_DEVICE inline std::int64_t periods_for_tokens(std::int64_t tokens, int period) {
    if (tokens <= 0) return 1;
    return tokens / period + (tokens % period != 0);
}
FLASH_KV_HOST_DEVICE inline std::int64_t capacity_tokens(std::int64_t periods, int period) {
    return periods * period;
}
FLASH_KV_HOST_DEVICE inline int blocks_for_rows(int rows, int block) {
    return rows <= 0 ? 0 : rows / block + (rows % block != 0);
}
FLASH_KV_HOST_DEVICE inline int complete_block_end(int pos, int block) {
    const std::int64_t end = static_cast<std::int64_t>(pos) + 1;
    return static_cast<int>((end / block) * block - 1);
}

#undef FLASH_KV_HOST_DEVICE
}  // namespace kv_math
}  // namespace flash
