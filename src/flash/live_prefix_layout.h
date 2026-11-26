#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace flash_live {
// Visit contiguous physical runs in canonical logical order, without
// allocating. Offset/count units may be owned rows or four-row pooled-indexer
// blocks.
template <class F>
void runs(const std::vector<int> &map, int units_per_period, size_t first,
          size_t count, F &&visit) {
  if (units_per_period <= 0)
    throw std::invalid_argument("live prefix: invalid period width");
  size_t done = 0;
  while (done < count) {
    const size_t logical = first + done;
    const size_t period = logical / (size_t)units_per_period;
    if (period >= map.size() || map[period] < 0)
      throw std::out_of_range("live prefix: unmapped logical period");
    const size_t within = logical % (size_t)units_per_period;
    const size_t physical = (size_t)map[period] * units_per_period + within;
    size_t length = std::min(count - done, (size_t)units_per_period - within);
    while (length < count - done) {
      const size_t next = (logical + length) / (size_t)units_per_period;
      if (next >= map.size() || map[next] < 0)
        throw std::out_of_range("live prefix: unmapped logical period");
      if ((size_t)map[next] * units_per_period != physical + length)
        break;
      length += std::min(count - done - length, (size_t)units_per_period);
    }
    visit(done, physical, length);
    done += length;
  }
}
} // namespace flash_live
