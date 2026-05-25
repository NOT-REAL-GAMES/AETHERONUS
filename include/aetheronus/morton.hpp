#pragma once

#include <cstdint>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

namespace ae {

inline uint64_t morton_split_by_3(uint32_t value) {
    uint64_t x = static_cast<uint64_t>(value) & 0x1fffffu;
    x = (x | (x << 32u)) & 0x1f00000000ffffull;
    x = (x | (x << 16u)) & 0x1f0000ff0000ffull;
    x = (x | (x << 8u)) & 0x100f00f00f00f00full;
    x = (x | (x << 4u)) & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2u)) & 0x1249249249249249ull;
    return x;
}

inline uint32_t morton_compact_by_3(uint64_t value) {
    uint64_t x = value & 0x1249249249249249ull;
    x = (x ^ (x >> 2u)) & 0x10c30c30c30c30c3ull;
    x = (x ^ (x >> 4u)) & 0x100f00f00f00f00full;
    x = (x ^ (x >> 8u)) & 0x1f0000ff0000ffull;
    x = (x ^ (x >> 16u)) & 0x1f00000000ffffull;
    x = (x ^ (x >> 32u)) & 0x1fffffull;
    return static_cast<uint32_t>(x);
}

inline uint64_t morton_encode(uint32_t x, uint32_t y, uint32_t z) {
#if defined(__BMI2__)
    return _pdep_u64(x, 0x1249249249249249ull) |
           _pdep_u64(y, 0x2492492492492492ull) |
           _pdep_u64(z, 0x4924924924924924ull);
#else
    return morton_split_by_3(x) |
           (morton_split_by_3(y) << 1u) |
           (morton_split_by_3(z) << 2u);
#endif
}

inline void morton_decode(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
#if defined(__BMI2__)
    x = static_cast<uint32_t>(_pext_u64(code, 0x1249249249249249ull));
    y = static_cast<uint32_t>(_pext_u64(code, 0x2492492492492492ull));
    z = static_cast<uint32_t>(_pext_u64(code, 0x4924924924924924ull));
#else
    x = morton_compact_by_3(code);
    y = morton_compact_by_3(code >> 1u);
    z = morton_compact_by_3(code >> 2u);
#endif
}

inline uint64_t morton_parent(uint64_t code) {
    return code >> 3u;
}

inline uint64_t morton_child(uint64_t code, uint32_t child_index) {
    return (code << 3u) | static_cast<uint64_t>(child_index & 7u);
}

} // namespace ae
