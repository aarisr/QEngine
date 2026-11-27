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
    return 0;
}
