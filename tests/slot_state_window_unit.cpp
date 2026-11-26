#include "flash/slot_state.h"

#include <cstdio>

int main() {
  slot_state::Owner owner(7);
  const auto &initial = owner.snapshot();
  if (!initial.slot_live || !initial.archive_present || !initial.prefix_present)
    return 1;

  // Crash-A window: forced release, archive-full checkpoint drop, then the
  // pinned-prefix persist later in the same tick.
  owner.release_slot();
  const auto &released = owner.snapshot();
  if (released.slot_live || released.slot_epoch != 8 ||
      released.archive_epoch != 7 || released.prefix_epoch != 7)
    return 2;

  owner.drop_archive();
  const auto &dropped = owner.snapshot();
  if (dropped.archive_present || dropped.archive_epoch != 8 ||
      dropped.slot_epoch != 8)
    return 3;

  owner.persist_prefix();
  const auto &persisted = owner.snapshot();
  if (!persisted.prefix_present || persisted.prefix_epoch != 8 ||
      persisted.slot_epoch != 8 || persisted.archive_epoch != 8)
    return 4;

  std::puts(
      "OK slot_state_window_unit (release, archive drop, prefix persist)");
  return 0;
}
