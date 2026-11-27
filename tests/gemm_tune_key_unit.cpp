#include "../src/gemm_tune_key.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
    using gemm_tune_key::Device;
    const Device a{"NVIDIA GeForce RTX 3090", 82, 8, 6};
    const Device b{"NVIDIA RTX 4090", 128, 8, 9};
    const std::string two_3090 = gemm_tune_key::hardware({a, a}, "kernel-hash-a");
    assert(two_3090 == "cards 2\ngpu 2 NVIDIA GeForce RTX 3090 sm82 cc8.6\nkernel-set kernel-hash-a\n");
    assert(two_3090 == gemm_tune_key::hardware({a, a}, "kernel-hash-a"));
    assert(two_3090 != gemm_tune_key::hardware({a}, "kernel-hash-a"));
    assert(two_3090 != gemm_tune_key::hardware({a, a}, "kernel-hash-b"));
    assert(two_3090 != gemm_tune_key::hardware({a, b}, "kernel-hash-a"));
    const std::string mixed = gemm_tune_key::hardware({a, b}, "kernel-hash-a");
    assert(mixed == gemm_tune_key::hardware({b, a}, "kernel-hash-a"));
    const std::string setup = "build /src/src/kernels/gemm.cu\ncands 32x1 64x1\n";
    const std::vector<std::string> legacy = {"qengine-gemm-tune 1", "built Nov 27 2026 08:15:10",
                                             "build /src/src/kernels/gemm.cu", "cands 32x1 64x1"};
    assert(gemm_tune_key::legacy_header_matches(legacy, setup));
    assert(!gemm_tune_key::legacy_header_matches({"qengine-gemm-tune 2", "built Nov 27 2026 08:15:10",
                                                   "build /src/src/kernels/gemm.cu", "cands 32x1 64x1"},
                                                  setup));
    assert(!gemm_tune_key::legacy_header_matches(legacy, "build other.cu\ncands 32x1 64x1\n"));
    assert(gemm_tune_key::legacy_devices_match({"device GPU0", "device GPU1"}, {"device GPU1", "device GPU0"}));
    assert(!gemm_tune_key::legacy_devices_match({"device GPU0", "device GPU1"}, {"device GPU0"}));
    return 0;
}
