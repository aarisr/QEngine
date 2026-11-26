#pragma once
// CRC-32C for the shared PREFIX objects: the SSE4.2 crc32 instruction when the CPU has it (EPYC 7663: yes; 15-20 GB/s
// per core) and a reflected-table fallback otherwise. Both produce the standard CRC-32C value (the test asserts it on
// a known vector and on random buffers). The bytes are verified per shard BEFORE first use, so this is on the restore
// path; it never guesses.
#include <stdint.h>
#include <string.h>

#include <mutex>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <nmmintrin.h>
#define SP_HAVE_SSE42 1
#endif

namespace sp {

inline bool crc32c_hardware() {
#if SP_HAVE_SSE42
    static const bool ok = __builtin_cpu_supports("sse4.2");
    return ok;
#else
    return false;
#endif
}

inline uint32_t crc32c_soft(const void* data, size_t n, uint32_t crc = 0) {
    static std::once_flag once;
    static uint32_t tab[256];
    std::call_once(once, [] {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
            tab[i] = c;
        }
    });
    const uint8_t* p = (const uint8_t*)data;
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = tab[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

__attribute__((target("sse4.2"))) inline uint32_t crc32c_hw(const void* data, size_t n, uint32_t crc = 0) {
#if SP_HAVE_SSE42
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = (uint32_t)~crc;
    while (n >= 8) {
        uint64_t w;
        memcpy(&w, p, 8);
        h = (uint64_t)_mm_crc32_u64(h, w);
        p += 8;
        n -= 8;
    }
    while (n) {
        h = (uint32_t)_mm_crc32_u8((uint32_t)h, *p);
        p++;
        n--;
    }
    return ~(uint32_t)h;
#else
    return crc32c_soft(data, n, crc);
#endif
}

inline uint32_t crc32c(const void* data, size_t n, uint32_t crc = 0) {
    return crc32c_hardware() ? crc32c_hw(data, n, crc) : crc32c_soft(data, n, crc);
}

}  // namespace sp
