#pragma once

// CPU-only model of the ownership hand-off observed in Crash A. This type is
// deliberately not wired into FE: the current event report does not establish
// a safe replacement for the live archive ordering.
#include <cstdint>

namespace slot_state {

struct Snapshot {
  uint64_t slot_epoch = 0;
  uint64_t archive_epoch = 0;
  uint64_t prefix_epoch = 0;
  bool slot_live = false;
  bool archive_present = false;
  bool prefix_present = false;
};

// A value-only ledger for the three independently-owned representations. Each
// event advances its own epoch; releasing a slot or dropping an archive must
// not silently invalidate an already-persisted prefix.
class Owner {
public:
  explicit Owner(uint64_t epoch = 1) {
    state_.slot_epoch = state_.archive_epoch = state_.prefix_epoch = epoch;
    state_.slot_live = state_.archive_present = state_.prefix_present = true;
  }

  void release_slot() {
    state_.slot_live = false;
    ++state_.slot_epoch;
  }

  void drop_archive() {
    state_.archive_present = false;
    ++state_.archive_epoch;
  }

  void persist_prefix() {
    state_.prefix_present = true;
    state_.prefix_epoch = state_.slot_epoch;
  }

  const Snapshot &snapshot() const { return state_; }

private:
  Snapshot state_;
};

} // namespace slot_state
