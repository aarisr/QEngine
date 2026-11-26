#include "../src/flash/live_prefix_layout.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <tuple>

int main() {
  using Run = std::tuple<size_t, size_t, size_t>;
  const std::vector<int> fragmented{3, 4, 0, 7};
  std::vector<Run> got;
  flash_live::runs(fragmented, 4, 2, 12,
                   [&](size_t off, size_t physical, size_t n) {
                     got.emplace_back(off, physical, n);
                   });
  const std::vector<Run> expected{{0, 14, 6}, {6, 0, 4}, {10, 28, 2}};
  assert(got == expected);
  // A canonical archive serializes logical order, independently of physical
  // placement.
  int physical[32] = {};
  const int source[]{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  flash_live::runs(fragmented, 4, 2, 12, [&](size_t off, size_t p, size_t n) {
    std::memcpy(physical + p, source + off, n * sizeof(int));
  });
  int restored[12] = {};
  flash_live::runs(fragmented, 4, 2, 12, [&](size_t off, size_t p, size_t n) {
    std::memcpy(restored + off, physical + p, n * sizeof(int));
  });
  for (int i = 0; i < 12; ++i)
    assert(restored[i] == 10 + i);
  // Pooled indexer blocks use the same periods, with a four-times smaller
  // width.
  std::vector<Run> blocks;
  flash_live::runs(fragmented, 1, 0, 4, [&](size_t off, size_t p, size_t n) {
    blocks.emplace_back(off, p, n);
  });
  assert((blocks == std::vector<Run>{{0, 3, 2}, {2, 0, 1}, {3, 7, 1}}));
  // The final partial period is private: restoring its tail must not touch
  // prefix cells.
  int split[12];
  for (int &x : split)
    x = -1;
  flash_live::runs(std::vector<int>{1, 0}, 4, 4, 2,
                   [&](size_t, size_t p, size_t n) {
                     for (size_t j = 0; j < n; ++j)
                       split[p + j] = 42;
                   });
  assert(split[0] == 42 && split[1] == 42 && split[2] == -1 && split[4] == -1);
  bool refused = false;
  try {
    flash_live::runs(std::vector<int>{-1}, 4, 0, 1,
                     [](size_t, size_t, size_t) {});
  } catch (const std::out_of_range &) {
    refused = true;
  }
  assert(refused);
  std::cout << "live_prefix_layout_unit: PASS\n";
}
