#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace gemm_tune_key {

struct Device {
    std::string name;
    int multiprocessors = 0;
    int major = 0;
    int minor = 0;
};

// Stable across device ordering; includes the multiset of GPU models and the card count.
inline std::string hardware(const std::vector<Device>& devices, const std::string& kernel_set_hash) {
    std::map<std::string, size_t> counts;
    for (const Device& device : devices) {
        std::ostringstream profile;
        profile << device.name << " sm" << device.multiprocessors << " cc" << device.major << '.' << device.minor;
        ++counts[profile.str()];
    }
    std::ostringstream out;
    out << "cards " << devices.size() << '\n';
    for (const auto& entry : counts) out << "gpu " << entry.second << ' ' << entry.first << '\n';
    out << "kernel-set " << kernel_set_hash << '\n';
    return out.str();
}

}  // namespace gemm_tune_key
