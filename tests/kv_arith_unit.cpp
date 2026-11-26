#include "flash/kv_arith.h"
#include <climits>
#include <cstdint>

int main() {
    using flash::kv_math::blocks_for_rows;
    using flash::kv_math::capacity_tokens;
    using flash::kv_math::complete_block_end;
    using flash::kv_math::periods_for_tokens;

    if (periods_for_tokens(INT_MAX, 1536) != 1398102) return 1;
    if (periods_for_tokens(static_cast<std::int64_t>(INT_MAX) * 2, 1536) != 2796203) return 2;
    if (capacity_tokens(INT_MAX, 1536) != INT64_C(3298534881792)) return 3;
    if (blocks_for_rows(INT_MAX, 128) != 16777216) return 4;
    if (complete_block_end(INT_MAX, 128) != INT_MAX) return 5;
    if (complete_block_end(INT_MAX - 1, 128) != 2147483519) return 6;
    return 0;
}
