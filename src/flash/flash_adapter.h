#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shared_prefix.h"

namespace flash_adapter {

template <class Blob>
std::shared_ptr<Blob> mapped_blob(std::shared_ptr<sp::Map> mapping) {
    return std::shared_ptr<Blob>(
        new Blob, [mapping = std::move(mapping)](Blob* blob) {
            blob->p.clear();
            delete blob;
        });
}
template <class Engine, class Pin>
class KeepPin {
    Engine& engine_;
    Pin* previous_;
public:
    KeepPin(Engine* engine, Pin* current)
        : engine_(*engine), previous_(engine->hold_pin) {
        engine->hold_pin = current;
    }
    ~KeepPin() { engine_.hold_pin = previous_; }
};


struct State {
    std::string sp_dir, sp_fp;
    std::shared_ptr<sp::Index> sp_index;
    std::unique_ptr<sp::Evictor> sp_evictor;
    std::vector<std::string> sp_own;
    size_t sp_quota = 0, sp_own_bytes = 0;
    uint64_t sp_writer = 0, sp_global = 0;
    int sp_pages_advice = 0;
    long long n_sp_write = 0, n_sp_hit = 0, n_sp_miss = 0, n_sp_fail = 0;
    long long n_sp_fallback = 0;
};

template <class Engine>
std::unordered_map<Engine*, State>& registry() {
    static std::unordered_map<Engine*, State> states;
    return states;
}

template <class Engine>
State& state(Engine* engine) {
    return registry<Engine>()[engine];
}

template <class Engine>
void shutdown(Engine* engine) {
    registry<Engine>().erase(engine);
}

} // namespace flash_adapter
