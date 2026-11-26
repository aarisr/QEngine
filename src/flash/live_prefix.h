#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace flash_live {

namespace detail {
struct ArenaState {
  explicit ArenaState(int period_capacity) : capacity(period_capacity) {
    if (period_capacity < 0)
      throw std::invalid_argument("arena capacity must be non-negative");
    occupied.resize(static_cast<std::size_t>(period_capacity), 0);
    free_periods.reserve(static_cast<std::size_t>(period_capacity));
    for (int period = period_capacity; period > 0; --period)
      free_periods.push_back(period - 1);
  }

  const int capacity;
  mutable std::mutex mutex;
  std::vector<int> free_periods;
  std::vector<unsigned char> occupied;
  int used = 0;
  int high_water = 0;
};
} // namespace detail

class Extent {
public:
  const std::vector<int> &periods() const noexcept { return periods_; }

  ~Extent() {
    if (!state_ || periods_.empty())
      return;
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (int period : periods_) {
      state_->free_periods.push_back(period);
      state_->occupied[period] = 0;
    }
    state_->used -= static_cast<int>(periods_.size());
    while (state_->high_water > 0 && !state_->occupied[state_->high_water - 1])
      --state_->high_water;
  }

  Extent(const Extent &) = delete;
  Extent &operator=(const Extent &) = delete;

private:
  friend class Arena;
  Extent(std::shared_ptr<detail::ArenaState> state, std::vector<int> periods,
         std::shared_ptr<const void> owner)
      : state_(std::move(state)), periods_(std::move(periods)),
        owner_(std::move(owner)) {}

  std::shared_ptr<detail::ArenaState> state_;
  std::vector<int> periods_;
  std::shared_ptr<const void> owner_;
};

using ExtentPtr = std::shared_ptr<const Extent>;

class Arena {
public:
  explicit Arena(int capacity)
      : state_(std::make_shared<detail::ArenaState>(capacity)) {}

  ExtentPtr allocate(int count, std::shared_ptr<const void> owner = {}) const {
    if (count < 0)
      throw std::invalid_argument("extent size must be non-negative");

    std::vector<int> reserved;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      if (count > static_cast<int>(state_->free_periods.size()))
        return nullptr;
      reserved.reserve(static_cast<std::size_t>(count));
      for (int i = 0; i < count; ++i) {
        const int period = state_->free_periods.back();
        reserved.push_back(period);
        state_->free_periods.pop_back();
        state_->occupied[period] = 1;
        state_->high_water = std::max(state_->high_water, period + 1);
      }
      state_->used += count;
    }

    Extent *raw = nullptr;
    try {
      raw = new Extent(state_, std::move(reserved), std::move(owner));
    } catch (...) {
      std::lock_guard<std::mutex> lock(state_->mutex);
      // reserve() above ensured this rollback cannot allocate.
      for (int period : reserved) {
        state_->free_periods.push_back(period);
        state_->occupied[period] = 0;
      }
      state_->used -= count;
      while (state_->high_water > 0 &&
             !state_->occupied[state_->high_water - 1])
        --state_->high_water;
      throw;
    }
    // If the shared_ptr control-block allocation fails, it deletes raw and the
    // extent destructor returns its reservation.
    return ExtentPtr(raw);
  }

  int capacity() const noexcept { return state_->capacity; }

  int used() const noexcept {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->used;
  }

  int high_water() const noexcept {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->high_water;
  }

private:
  std::shared_ptr<detail::ArenaState> state_;
};

class View {
public:
  explicit View(Arena arena, ExtentPtr prefix = {})
      : arena_(std::move(arena)), prefix_(std::move(prefix)),
        shared_periods_(prefix_ ? static_cast<int>(prefix_->periods().size())
                                : 0) {
    if (prefix_)
      periods_ = prefix_->periods();
  }

  View(const View &) = delete;
  View &operator=(const View &) = delete;
  View(View &&) noexcept = default;
  View &operator=(View &&) noexcept = default;

  bool ensure(int total_periods) {
    if (total_periods < 0)
      return false;
    if (total_periods <= static_cast<int>(periods_.size()))
      return true;

    const int additional = total_periods - static_cast<int>(periods_.size());
    periods_.reserve(static_cast<std::size_t>(total_periods));
    tails_.reserve(tails_.size() + 1);
    ExtentPtr tail = arena_.allocate(additional);
    if (!tail)
      return false;

    periods_.insert(periods_.end(), tail->periods().begin(),
                    tail->periods().end());
    tails_.push_back(std::move(tail));
    return true;
  }

  int shared_periods() const noexcept { return shared_periods_; }
  const std::vector<int> &periods() const noexcept { return periods_; }

  void clear() noexcept {
    tails_.clear();
    periods_.clear();
    prefix_.reset();
    shared_periods_ = 0;
  }

private:
  Arena arena_;
  ExtentPtr prefix_;
  std::vector<ExtentPtr> tails_;
  std::vector<int> periods_;
  int shared_periods_ = 0;
};

class PrefixIndex {
public:
  ExtentPtr find(const std::shared_ptr<const void> &owner, int periods) {
    for (auto it = entries_.begin(); it != entries_.end();) {
      if (it->extent.expired()) {
        it = entries_.erase(it);
        continue;
      }
      if (it->periods == periods && same_owner(it->owner, owner))
        return it->extent.lock();
      ++it;
    }
    return nullptr;
  }

  void insert(const std::shared_ptr<const void> &owner, int periods,
              ExtentPtr extent) {
    for (auto &entry : entries_) {
      if (entry.periods == periods && same_owner(entry.owner, owner)) {
        entry.extent = std::move(extent);
        return;
      }
    }
    entries_.push_back(Entry{owner, periods, std::move(extent)});
  }

  void clear() noexcept { entries_.clear(); }

private:
  struct Entry {
    std::weak_ptr<const void> owner;
    int periods;
    std::weak_ptr<const Extent> extent;
  };

  static bool same_owner(const std::weak_ptr<const void> &weak,
                         const std::shared_ptr<const void> &shared) noexcept {
    return !weak.owner_before(shared) && !shared.owner_before(weak);
  }

  std::vector<Entry> entries_;
};

} // namespace flash_live
