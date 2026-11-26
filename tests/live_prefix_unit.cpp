// CUDA-free ownership and mapping tests for flash/live_prefix.h.
#include "flash/live_prefix.h"

#include <cstdio>
#include <memory>
#include <vector>

using namespace flash_live;

static int failures = 0;
#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,             \
                   #condition);                                                \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

static void shared_prefix_has_private_writable_tails() {
  Arena arena(8);
  auto prefix = arena.allocate(2);
  EXPECT(prefix != nullptr);
  View first(arena, prefix);
  View second(arena, prefix);
  EXPECT(first.ensure(4));
  EXPECT(second.ensure(4));
  EXPECT(first.shared_periods() == 2);
  EXPECT(first.periods().size() == 4);
  EXPECT(second.periods().size() == 4);
  EXPECT(first.periods()[0] == second.periods()[0]);
  EXPECT(first.periods()[1] == second.periods()[1]);
  EXPECT(first.periods()[2] != second.periods()[2]);
  EXPECT(first.periods()[3] != second.periods()[3]);

  std::vector<int> physical_bytes(8, -1);
  physical_bytes[first.periods()[2]] = 101;
  physical_bytes[second.periods()[2]] = 202;
  EXPECT(physical_bytes[first.periods()[2]] == 101);
  EXPECT(physical_bytes[second.periods()[2]] == 202);
}

static void index_eviction_does_not_invalidate_live_extent() {
  Arena arena(4);
  PrefixIndex index;
  auto owner = std::make_shared<int>(7);
  std::weak_ptr<int> owner_observer = owner;
  std::shared_ptr<const void> source = owner;
  auto extent = arena.allocate(2, source);
  index.insert(source, 2, extent);
  EXPECT(index.find(source, 2) == extent);

  index.clear();
  EXPECT(index.find(source, 2) == nullptr);
  source.reset();
  owner.reset();
  EXPECT(!owner_observer.expired());
  EXPECT(extent->periods().size() == 2);
  EXPECT(arena.used() == 2);
  extent.reset();
  EXPECT(owner_observer.expired());
  EXPECT(arena.used() == 0);
}
static void index_is_weak_and_keys_by_owner_and_period_count() {
  Arena arena(3);
  PrefixIndex index;
  auto owner = std::make_shared<int>(1);
  auto other_owner = std::make_shared<int>(1);
  std::shared_ptr<const void> source = owner;
  std::shared_ptr<const void> other_source = other_owner;
  auto extent = arena.allocate(1, source);
  index.insert(source, 1, extent);
  EXPECT(index.find(other_source, 1) == nullptr);
  EXPECT(index.find(source, 2) == nullptr);
  EXPECT(index.find(source, 1) == extent);
  extent.reset();
  EXPECT(arena.used() == 0);
  EXPECT(index.find(source, 1) == nullptr);
}

static void shrinking_ensure_does_not_change_a_view() {
  Arena arena(4);
  View view(arena);
  EXPECT(view.ensure(3));
  const auto original = view.periods();
  EXPECT(view.ensure(1));
  EXPECT(view.periods() == original);
  EXPECT(view.shared_periods() == 0);
}

static void final_release_reuses_physical_periods() {
  Arena arena(2);
  int released_period = -1;
  {
    auto extent = arena.allocate(2);
    EXPECT(extent != nullptr);
    released_period = extent->periods()[1];
    EXPECT(arena.used() == 2);
  }
  EXPECT(arena.used() == 0);
  auto replacement = arena.allocate(2);
  EXPECT(replacement != nullptr);
  EXPECT(std::find(replacement->periods().begin(), replacement->periods().end(),
                   released_period) != replacement->periods().end());
}

static void exhaustion_is_atomic_and_fragmentation_is_supported() {
  Arena arena(5);
  auto prefix = arena.allocate(1);
  View view(arena, prefix);
  EXPECT(view.ensure(2));
  const auto original = view.periods();
  const int used_before_failure = arena.used();
  auto blocker_a = arena.allocate(1);
  auto blocker_b = arena.allocate(1);
  EXPECT(blocker_a != nullptr);
  EXPECT(blocker_b != nullptr);
  EXPECT(!view.ensure(4));
  EXPECT(view.periods() == original);
  EXPECT(arena.used() == used_before_failure + 2);
  blocker_a.reset();
  EXPECT(view.ensure(4));
  EXPECT(view.periods().size() == 4);
  EXPECT(view.periods()[0] == original[0]);
  EXPECT(view.periods()[1] == original[1]);
  EXPECT(view.periods()[2] != original[1]);
  EXPECT(view.periods()[3] != original[1]);
  EXPECT(view.periods()[2] != view.periods()[3]);
}

static void failed_view_growth_preserves_mapping_and_releases_reservations() {
  Arena arena(4);
  auto prefix = arena.allocate(1);
  View view(arena, prefix);
  EXPECT(view.ensure(2));
  const auto original = view.periods();
  auto blocker = arena.allocate(1);
  EXPECT(blocker != nullptr);
  const int used = arena.used();
  EXPECT(!view.ensure(4));
  EXPECT(view.periods() == original);
  EXPECT(arena.used() == used);
  blocker.reset();
  view.clear();
  EXPECT(view.periods().empty());
  EXPECT(arena.used() == 1);
}

static void source_owner_survives_until_last_consumer() {
  Arena arena(5);
  auto owner = std::make_shared<int>(11);
  std::weak_ptr<int> observer = owner;
  std::shared_ptr<const void> source = owner;
  auto prefix = arena.allocate(2, source);
  View first(arena, prefix);
  View second(arena, prefix);
  owner.reset();
  source.reset();
  prefix.reset();
  EXPECT(!observer.expired());
  first.clear();
  EXPECT(!observer.expired());
  second.clear();
  EXPECT(observer.expired());
  EXPECT(arena.used() == 0);
}

static void pool_top_covers_owned_addresses_and_shrinks_on_release() {
  Arena arena(4);
  auto low = arena.allocate(2);
  auto high = arena.allocate(2);
  EXPECT(arena.high_water() == 4);
  low.reset();
  EXPECT(arena.used() == 2);
  EXPECT(arena.high_water() == 4);
  high.reset();
  EXPECT(arena.high_water() == 0);
  auto reused = arena.allocate(2);
  const int highest =
      *std::max_element(reused->periods().begin(), reused->periods().end());
  EXPECT(arena.high_water() == highest + 1);
  reused.reset();
  EXPECT(arena.high_water() == 0);
}

int main() {
  shared_prefix_has_private_writable_tails();
  index_eviction_does_not_invalidate_live_extent();
  index_is_weak_and_keys_by_owner_and_period_count();
  shrinking_ensure_does_not_change_a_view();
  final_release_reuses_physical_periods();
  exhaustion_is_atomic_and_fragmentation_is_supported();
  failed_view_growth_preserves_mapping_and_releases_reservations();
  source_owner_survives_until_last_consumer();
  pool_top_covers_owned_addresses_and_shrinks_on_release();
  if (failures != 0) {
    std::fprintf(stderr, "%d live-prefix test(s) failed\n", failures);
    return 1;
  }
  std::puts("live-prefix CPU tests passed");
  return 0;
}
