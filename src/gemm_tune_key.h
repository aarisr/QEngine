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

inline bool legacy_header_matches(const std::vector<std::string>& lines, const std::string& legacy_setup) {
    if (lines.size() < 2 || lines[0] != "qengine-gemm-tune 1" || lines[1].rfind("built ", 0) != 0) return false;
    std::string setup;
    for (size_t i = 2; i < lines.size(); ++i) setup += lines[i] + "\n";
    return setup == legacy_setup;
}

inline bool legacy_devices_match(std::vector<std::string> expected, std::vector<std::string> actual) {
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    return expected == actual;
}

}  // namespace gemm_tune_key
