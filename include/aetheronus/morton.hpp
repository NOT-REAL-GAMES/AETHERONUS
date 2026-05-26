#pragma once

#include <cstdint>

#if defined(__BMI2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
#include <immintrin.h>
#endif
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
#include <intrin.h>
#endif

namespace ae {

#if defined(_MSC_VER)
#define AE_FORCE_INLINE __forceinline
#else
#define AE_FORCE_INLINE inline __attribute__((always_inline))
#endif

AE_FORCE_INLINE uint64_t morton_split_by_3(uint32_t value) {
    uint64_t x = static_cast<uint64_t>(value) & 0x1fffffu;
    x = (x | (x << 32u)) & 0x1f00000000ffffull;
    x = (x | (x << 16u)) & 0x1f0000ff0000ffull;
    x = (x | (x << 8u)) & 0x100f00f00f00f00full;
    x = (x | (x << 4u)) & 0x10c30c30c30c30c3ull;
    x = (x | (x << 2u)) & 0x1249249249249249ull;
    return x;
}

AE_FORCE_INLINE uint32_t morton_compact_by_3(uint64_t value) {
    uint64_t x = value & 0x1249249249249249ull;
    x = (x ^ (x >> 2u)) & 0x10c30c30c30c30c3ull;
    x = (x ^ (x >> 4u)) & 0x100f00f00f00f00full;
    x = (x ^ (x >> 8u)) & 0x1f0000ff0000ffull;
    x = (x ^ (x >> 16u)) & 0x1f00000000ffffull;
    x = (x ^ (x >> 32u)) & 0x1fffffull;
    return static_cast<uint32_t>(x);
}

AE_FORCE_INLINE uint64_t morton_encode_scalar(uint32_t x, uint32_t y, uint32_t z) {
    return morton_split_by_3(x) |
           (morton_split_by_3(y) << 1u) |
           (morton_split_by_3(z) << 2u);
}

AE_FORCE_INLINE void morton_decode_scalar(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
    x = morton_compact_by_3(code);
    y = morton_compact_by_3(code >> 1u);
    z = morton_compact_by_3(code >> 2u);
}

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
inline bool morton_cpu_has_bmi2() {
    int registers[4] = {};
    __cpuidex(registers, 0, 0);
    if (registers[0] < 7) {
        return false;
    }
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 8)) != 0;
}
#endif

AE_FORCE_INLINE uint64_t morton_encode(uint32_t x, uint32_t y, uint32_t z) {
#if defined(__BMI2__)
    return _pdep_u64(x, 0x1249249249249249ull) |
           _pdep_u64(y, 0x2492492492492492ull) |
           _pdep_u64(z, 0x4924924924924924ull);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
    static const bool has_bmi2 = morton_cpu_has_bmi2();
    if (has_bmi2) {
        return _pdep_u64(x, 0x1249249249249249ull) |
               _pdep_u64(y, 0x2492492492492492ull) |
               _pdep_u64(z, 0x4924924924924924ull);
    }
    return morton_encode_scalar(x, y, z);
#else
    return morton_encode_scalar(x, y, z);
#endif
}

AE_FORCE_INLINE void morton_decode(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
#if defined(__BMI2__)
    x = static_cast<uint32_t>(_pext_u64(code, 0x1249249249249249ull));
    y = static_cast<uint32_t>(_pext_u64(code, 0x2492492492492492ull));
    z = static_cast<uint32_t>(_pext_u64(code, 0x4924924924924924ull));
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
    static const bool has_bmi2 = morton_cpu_has_bmi2();
    if (has_bmi2) {
        x = static_cast<uint32_t>(_pext_u64(code, 0x1249249249249249ull));
        y = static_cast<uint32_t>(_pext_u64(code, 0x2492492492492492ull));
        z = static_cast<uint32_t>(_pext_u64(code, 0x4924924924924924ull));
        return;
    }
    morton_decode_scalar(code, x, y, z);
#else
    morton_decode_scalar(code, x, y, z);
#endif
}

AE_FORCE_INLINE uint64_t morton_parent(uint64_t code) {
    return code >> 3u;
}

AE_FORCE_INLINE uint64_t morton_child(uint64_t code, uint32_t child_index) {
    return (code << 3u) | static_cast<uint64_t>(child_index & 7u);
}

#undef AE_FORCE_INLINE

} // namespace ae
