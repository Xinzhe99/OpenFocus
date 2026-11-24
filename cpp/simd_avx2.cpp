#include "simd_dispatch.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define OPENFOCUS_X86_64 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define OPENFOCUS_X86_64 0
#endif

namespace openfocus {
namespace simd {
namespace {

std::atomic<bool> g_force_scalar{false};

bool detect_avx2_hw() {
#if OPENFOCUS_X86_64
    int info1[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    __cpuid(info1, 0);
    const int max_leaf = info1[0];
    if (max_leaf < 7) {
        return false;
    }
    __cpuidex(info1, 1, 0);
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx) || eax < 7) {
        return false;
    }
    __cpuid_count(1, 0, eax, ebx, ecx, edx);
    info1[0] = static_cast<int>(eax);
    info1[1] = static_cast<int>(ebx);
    info1[2] = static_cast<int>(ecx);
    info1[3] = static_cast<int>(edx);
#endif
    const uint32_t ecx1 = static_cast<uint32_t>(info1[2]);
    // Bit 27 = OSXSAVE, Bit 28 = AVX
    if ((ecx1 & (1u << 27)) == 0 || (ecx1 & (1u << 28)) == 0) {
        return false;
    }
    uint64_t xcr0 = 0;
#if defined(_MSC_VER)
    xcr0 = _xgetbv(0);
#else
    uint32_t xlo = 0, xhi = 0;
    __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    xcr0 = (static_cast<uint64_t>(xhi) << 32) | xlo;
#endif
    // XMM (bit 1) + YMM (bit 2) state enabled by OS
    if ((xcr0 & 0x6ull) != 0x6ull) {
        return false;
    }
    uint32_t ebx7 = 0;
#if defined(_MSC_VER)
    int info7[4] = {0, 0, 0, 0};
    __cpuidex(info7, 7, 0);
    ebx7 = static_cast<uint32_t>(info7[1]);
#else
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    ebx7 = ebx;
#endif
    // Leaf 7 EBX bit 5 = AVX2
    return (ebx7 & (1u << 5)) != 0;
#else
    return false;
#endif
}

bool cpu_has_avx2() {
    static const bool kSupported = detect_avx2_hw();
    return kSupported;
}

#if OPENFOCUS_X86_64
inline uint16_t encode_weight_scalar(float strength, bool is_border, float w_scale) {
    if (is_border) {
        return 0;
    }
    const float scaled = strength * w_scale;
    if (scaled >= 65535.0f) {
        return 65535;
    }
    if (scaled <= 0.0f) {
        return 0;
    }
    return static_cast<uint16_t>(static_cast<int32_t>(scaled));
}
#endif

}  // namespace

bool use_avx2() {
    return !g_force_scalar.load(std::memory_order_relaxed) && cpu_has_avx2();
}

void set_force_scalar(bool force) {
    g_force_scalar.store(force, std::memory_order_relaxed);
}

std::string simd_info() {
    return use_avx2() ? "AVX2" : "Scalar";
}

#if OPENFOCUS_X86_64

void pyr_down_vertical_row_f32_avx2(
    const float* p0, const float* p1, const float* p2, const float* p3,
    float* dst, int64_t elems, float d0, float d1
) {
    const __m256 vd0 = _mm256_set1_ps(d0);
    const __m256 vd1 = _mm256_set1_ps(d1);
    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(p0 + i), vd0);
        v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_loadu_ps(p1 + i), vd1));
        v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_loadu_ps(p2 + i), vd1));
        v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_loadu_ps(p3 + i), vd0));
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < elems; ++i) {
        float v = p0[i] * d0;
        v += p1[i] * d1;
        v += p2[i] * d1;
        dst[i] = v + p3[i] * d0;
    }
}

void pyr_reconstruct_row_f32_avx2(
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    float* tr, int64_t elems,
    float w0, float w1, float w2, float w3, float w4, float point_one
) {
    const __m256 vw0 = _mm256_set1_ps(w0);
    const __m256 vw1 = _mm256_set1_ps(w1);
    const __m256 vw2 = _mm256_set1_ps(w2);
    const __m256 vw3 = _mm256_set1_ps(w3);
    const __m256 vw4 = _mm256_set1_ps(w4);
    const __m256 vp1 = _mm256_set1_ps(point_one);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 lo = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(_mm256_loadu_ps(hr0 + i), vw0),
                _mm256_mul_ps(_mm256_loadu_ps(hr1 + i), vw1)),
            _mm256_mul_ps(_mm256_loadu_ps(hr2 + i), vw2));
        lo = _mm256_add_ps(
            _mm256_add_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(hr3 + i), vw3)),
            _mm256_mul_ps(_mm256_loadu_ps(hr4 + i), vw4));
        const __m256 t = _mm256_loadu_ps(tr + i);
        __m256 val = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(lo, vp1), t), lo);
        const __m256 lt0 = _mm256_cmp_ps(val, vzero, _CMP_LT_OQ);
        val = _mm256_blendv_ps(val, vzero, lt0);
        const __m256 gt1 = _mm256_cmp_ps(val, vone, _CMP_GT_OQ);
        val = _mm256_blendv_ps(val, vone, gt1);
        _mm256_storeu_ps(tr + i, val);
    }
    for (; i < elems; ++i) {
        float lo = (hr0[i] * w0 + hr1[i] * w1) + hr2[i] * w2;
        lo = (lo + hr3[i] * w3) + hr4[i] * w4;
        const float val = (lo + point_one) * tr[i] + lo;
        tr[i] = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
    }
}

void pyr_detail_weight_3c_f32_avx2(
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    float* target_row, uint16_t* w_row, int64_t width,
    float w0, float w1, float w2, float w3, float w4, float point_one,
    float w_scale, bool y_border
) {
    const __m256 vw0 = _mm256_set1_ps(w0);
    const __m256 vw1 = _mm256_set1_ps(w1);
    const __m256 vw2 = _mm256_set1_ps(w2);
    const __m256 vw3 = _mm256_set1_ps(w3);
    const __m256 vw4 = _mm256_set1_ps(w4);
    const __m256 vp1 = _mm256_set1_ps(point_one);
    const __m256 vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    int64_t x = 0;
    alignas(32) float av[24];
    for (; x + 8 <= width; x += 8) {
        const int64_t off = x * 3;
        for (int blk = 0; blk < 3; ++blk) {
            const int64_t b = off + blk * 8;
            __m256 lo = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(hr0 + b), vw0),
                    _mm256_mul_ps(_mm256_loadu_ps(hr1 + b), vw1)),
                _mm256_mul_ps(_mm256_loadu_ps(hr2 + b), vw2));
            lo = _mm256_add_ps(
                _mm256_add_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(hr3 + b), vw3)),
                _mm256_mul_ps(_mm256_loadu_ps(hr4 + b), vw4));
            const __m256 s = _mm256_loadu_ps(target_row + b);
            const __m256 v = _mm256_div_ps(_mm256_sub_ps(s, lo), _mm256_add_ps(lo, vp1));
            _mm256_storeu_ps(target_row + b, v);
            _mm256_store_ps(av + blk * 8, _mm256_and_ps(v, vabs));
        }
        for (int64_t k = 0; k < 8; ++k) {
            const int64_t px = x + k;
            const bool is_border = y_border || (width > 2 && (px == 0 || px == width - 1));
            const float strength = (av[k * 3] + av[k * 3 + 1]) + av[k * 3 + 2];
            w_row[px] = encode_weight_scalar(strength, is_border, w_scale);
        }
    }
    for (; x < width; ++x) {
        const int64_t off3 = x * 3;
        float lo0 = (hr0[off3] * w0 + hr1[off3] * w1) + hr2[off3] * w2;
        float lo1 = (hr0[off3 + 1] * w0 + hr1[off3 + 1] * w1) + hr2[off3 + 1] * w2;
        float lo2 = (hr0[off3 + 2] * w0 + hr1[off3 + 2] * w1) + hr2[off3 + 2] * w2;
        lo0 = (lo0 + hr3[off3] * w3) + hr4[off3] * w4;
        lo1 = (lo1 + hr3[off3 + 1] * w3) + hr4[off3 + 1] * w4;
        lo2 = (lo2 + hr3[off3 + 2] * w3) + hr4[off3 + 2] * w4;
        const float v0 = (target_row[off3] - lo0) / (lo0 + point_one);
        const float v1 = (target_row[off3 + 1] - lo1) / (lo1 + point_one);
        const float v2 = (target_row[off3 + 2] - lo2) / (lo2 + point_one);
        target_row[off3] = v0;
        target_row[off3 + 1] = v1;
        target_row[off3 + 2] = v2;
        const bool is_border = y_border || (width > 2 && (x == 0 || x == width - 1));
        const float strength = (std::abs(v0) + std::abs(v1)) + std::abs(v2);
        w_row[x] = encode_weight_scalar(strength, is_border, w_scale);
    }
}

void pyr_level0_weights_3c_f32_from_raw_avx2(
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    const void* raw_row, bool raw_is_u8, uint16_t* w_row, int64_t width,
    float factor, float w0, float w1, float w2, float w3, float w4, float point_one,
    float w_scale, bool y_border
) {
    const __m256 vw0 = _mm256_set1_ps(w0);
    const __m256 vw1 = _mm256_set1_ps(w1);
    const __m256 vw2 = _mm256_set1_ps(w2);
    const __m256 vw3 = _mm256_set1_ps(w3);
    const __m256 vw4 = _mm256_set1_ps(w4);
    const __m256 vp1 = _mm256_set1_ps(point_one);
    const __m256 vfac = _mm256_set1_ps(factor);
    const __m256 vabs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

    const uint8_t* r8 = static_cast<const uint8_t*>(raw_row);
    const uint16_t* r16 = static_cast<const uint16_t*>(raw_row);

    int64_t x = 0;
    alignas(32) float av[24];
    for (; x + 8 <= width; x += 8) {
        const int64_t off = x * 3;
        for (int blk = 0; blk < 3; ++blk) {
            const int64_t b = off + blk * 8;
            __m256 lo = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_loadu_ps(hr0 + b), vw0),
                    _mm256_mul_ps(_mm256_loadu_ps(hr1 + b), vw1)),
                _mm256_mul_ps(_mm256_loadu_ps(hr2 + b), vw2));
            lo = _mm256_add_ps(
                _mm256_add_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(hr3 + b), vw3)),
                _mm256_mul_ps(_mm256_loadu_ps(hr4 + b), vw4));
            __m256i raw_i;
            if (raw_is_u8) {
                raw_i = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(r8 + b)));
            } else {
                raw_i = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r16 + b)));
            }
            const __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(raw_i), vfac);
            const __m256 v = _mm256_div_ps(_mm256_sub_ps(s, lo), _mm256_add_ps(lo, vp1));
            _mm256_store_ps(av + blk * 8, _mm256_and_ps(v, vabs));
        }
        for (int64_t k = 0; k < 8; ++k) {
            const int64_t px = x + k;
            const bool is_border = y_border || (width > 2 && (px == 0 || px == width - 1));
            const float strength = (av[k * 3] + av[k * 3 + 1]) + av[k * 3 + 2];
            w_row[px] = encode_weight_scalar(strength, is_border, w_scale);
        }
    }
    for (; x < width; ++x) {
        const int64_t off3 = x * 3;
        float lo0 = (hr0[off3] * w0 + hr1[off3] * w1) + hr2[off3] * w2;
        float lo1 = (hr0[off3 + 1] * w0 + hr1[off3 + 1] * w1) + hr2[off3 + 1] * w2;
        float lo2 = (hr0[off3 + 2] * w0 + hr1[off3 + 2] * w1) + hr2[off3 + 2] * w2;
        lo0 = (lo0 + hr3[off3] * w3) + hr4[off3] * w4;
        lo1 = (lo1 + hr3[off3 + 1] * w3) + hr4[off3 + 1] * w4;
        lo2 = (lo2 + hr3[off3 + 2] * w3) + hr4[off3 + 2] * w4;
        float s0, s1, s2;
        if (raw_is_u8) {
            s0 = static_cast<float>(r8[off3]) * factor;
            s1 = static_cast<float>(r8[off3 + 1]) * factor;
            s2 = static_cast<float>(r8[off3 + 2]) * factor;
        } else {
            s0 = static_cast<float>(r16[off3]) * factor;
            s1 = static_cast<float>(r16[off3 + 1]) * factor;
            s2 = static_cast<float>(r16[off3 + 2]) * factor;
        }
        const float v0 = (s0 - lo0) / (lo0 + point_one);
        const float v1 = (s1 - lo1) / (lo1 + point_one);
        const float v2 = (s2 - lo2) / (lo2 + point_one);
        const bool is_border = y_border || (width > 2 && (x == 0 || x == width - 1));
        const float strength = (std::abs(v0) + std::abs(v1)) + std::abs(v2);
        w_row[x] = encode_weight_scalar(strength, is_border, w_scale);
    }
}

void depth_axis_iir_y_unguided_avx2(
    const uint16_t* src, int64_t src_sy,
    uint16_t* out, int64_t out_sy,
    int64_t h, int64_t left, int64_t cols, int64_t cnt,
    uint32_t q, uint32_t t
) {
    const uint32_t t16 = t * 16u;
    const uint32_t t_plus_q = t + q;
    std::vector<uint16_t> first(static_cast<size_t>(h) * cols);
    std::vector<uint32_t> acc(static_cast<size_t>(cols), 0u);
    for (int64_t y = 0; y < cnt; ++y) {
        const uint16_t* sr = src + y * src_sy + left;
        for (int64_t c = 0; c < cols; ++c) {
            acc[c] += static_cast<uint32_t>(sr[c]);
        }
    }
    for (int64_t c = 0; c < cols; ++c) {
        const uint64_t init = ((static_cast<uint64_t>(acc[c]) << 4u) / static_cast<uint64_t>(cnt)) * t_plus_q;
        acc[c] = static_cast<uint32_t>(init & 0xffffffffu);
    }

    const __m256i v_t16 = _mm256_set1_epi32(static_cast<int32_t>(t16));
    const __m256i v_q = _mm256_set1_epi32(static_cast<int32_t>(q));
    const __m256i v_2048 = _mm256_set1_epi32(2048);
    const __m256i v_4 = _mm256_set1_epi32(4);
    const __m256i v_3 = _mm256_set1_epi32(3);
    const __m256i v_1 = _mm256_set1_epi32(1);
    const int64_t vec_cols = cols & ~int64_t{7};

    for (int64_t y = 0; y < h; ++y) {
        const uint16_t* sr = src + y * src_sy + left;
        uint16_t* fr = first.data() + y * cols;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i sv = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i inn = _mm256_mullo_epi32(sv, v_t16);
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + c));
            const __m256i st = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a, v_2048), inn), 12);
            const __m256i fv = _mm256_srli_epi32(_mm256_add_epi32(st, v_4), 4);
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(fv, fv), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(fr + c), _mm256_castsi256_si128(p16));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + c),
                                _mm256_add_epi32(_mm256_mullo_epi32(st, v_q), inn));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t inn = static_cast<uint32_t>(sr[c]) * t16;
            const uint32_t st = (acc[c] + 2048u + inn) >> 12u;
            fr[c] = static_cast<uint16_t>((st + 4u) >> 4u);
            acc[c] = st * q + inn;
        }
    }

    const uint16_t* last_fr = first.data() + (h - 1) * cols;
    for (int64_t c = 0; c < cols; ++c) {
        acc[c] = (static_cast<uint32_t>(last_fr[c]) << 4u) * t_plus_q;
    }

    for (int64_t y = h - 1; y >= 0; --y) {
        const uint16_t* sr = src + y * src_sy + left;
        const uint16_t* fr = first.data() + y * cols;
        uint16_t* orow = out + y * out_sy + left;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i sv = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i f32 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(fr + c)));
            const __m256i inn = _mm256_mullo_epi32(sv, v_t16);
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + c));
            const __m256i st = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a, v_2048), inn), 12);
            const __m256i term = _mm256_srli_epi32(_mm256_add_epi32(st, v_3), 4);
            const __m256i ov = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(term, v_1), f32), 1);
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(ov, ov), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(orow + c), _mm256_castsi256_si128(p16));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + c),
                                _mm256_add_epi32(_mm256_mullo_epi32(st, v_q), inn));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t inn = static_cast<uint32_t>(sr[c]) * t16;
            const uint32_t st = (acc[c] + 2048u + inn) >> 12u;
            orow[c] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + static_cast<uint32_t>(fr[c])) >> 1u);
            acc[c] = st * q + inn;
        }
    }
}

void depth_axis_iir_y_guided_avx2(
    const uint16_t* src, int64_t src_sy,
    const uint16_t* guide, int64_t guide_sy,
    uint16_t* out, int64_t out_sy,
    int64_t h, int64_t left, int64_t cols, int64_t cnt,
    uint32_t tm, uint32_t tl
) {
    std::vector<uint32_t> first(static_cast<size_t>(h) * cols);
    std::vector<uint32_t> acc(static_cast<size_t>(cols), 0u);
    for (int64_t y = 0; y < cnt; ++y) {
        const uint16_t* sr = src + y * src_sy + left;
        for (int64_t c = 0; c < cols; ++c) {
            acc[c] += static_cast<uint32_t>(sr[c]);
        }
    }
    for (int64_t c = 0; c < cols; ++c) {
        acc[c] = static_cast<uint32_t>((static_cast<uint64_t>(acc[c]) << 4u) / static_cast<uint64_t>(cnt));
    }

    const __m256i v_65536 = _mm256_set1_epi32(65536);
    const __m256i v_4096 = _mm256_set1_epi32(4096);
    const __m256i v_2048 = _mm256_set1_epi32(2048);
    const __m256i v_tl = _mm256_set1_epi32(static_cast<int32_t>(tl));
    const __m256i v_tm = _mm256_set1_epi32(static_cast<int32_t>(tm));
    const __m256i v_4 = _mm256_set1_epi32(4);
    const __m256i v_3 = _mm256_set1_epi32(3);
    const __m256i v_1 = _mm256_set1_epi32(1);
    const int64_t vec_cols = cols & ~int64_t{7};

    // Forward y = 0
    {
        const uint16_t* sr0 = src + left;
        const uint16_t* gr0 = guide + left;
        uint32_t* fr0 = first.data();
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr0 + c)));
            const __m256i s = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr0 + c)));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + c));
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i num = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, a), v_2048),
                _mm256_mullo_epi32(_mm256_add_epi32(v, v), _mm256_slli_epi32(amt, 4)));
            const __m256i st = _mm256_srli_epi32(num, 12);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + c), st);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(fr0 + c),
                                _mm256_srli_epi32(_mm256_add_epi32(st, v_4), 4));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t v = sr0[c];
            const uint32_t s = gr0[c];
            const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
            const uint32_t st = ((4096u - 2u * amt) * acc[c] + 2048u + (v + v) * (amt << 4u)) >> 12u;
            acc[c] = st;
            fr0[c] = (st + 4u) >> 4u;
        }
    }

    for (int64_t y = 1; y < h; ++y) {
        const uint16_t* sr = src + y * src_sy + left;
        const uint16_t* pr = src + (y - 1) * src_sy + left;
        const uint16_t* gr = guide + y * guide_sy + left;
        uint32_t* fr = first.data() + y * cols;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i p = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pr + c)));
            const __m256i s = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr + c)));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + c));
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i num = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, a), v_2048),
                _mm256_mullo_epi32(_mm256_add_epi32(v, p), _mm256_slli_epi32(amt, 4)));
            const __m256i st = _mm256_srli_epi32(num, 12);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + c), st);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(fr + c),
                                _mm256_srli_epi32(_mm256_add_epi32(st, v_4), 4));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t v = sr[c];
            const uint32_t p = pr[c];
            const uint32_t s = gr[c];
            const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
            const uint32_t st = ((4096u - 2u * amt) * acc[c] + 2048u + (v + p) * (amt << 4u)) >> 12u;
            acc[c] = st;
            fr[c] = (st + 4u) >> 4u;
        }
    }

    const uint32_t* last_fr = first.data() + (h - 1) * cols;
    for (int64_t c = 0; c < cols; ++c) {
        acc[c] = last_fr[c] << 4u;
    }

    // Backward y = h - 1
    {
        const int64_t yt = h - 1;
        const uint16_t* sr = src + yt * src_sy + left;
        const uint16_t* gr = guide + yt * guide_sy + left;
        const uint32_t* fr = first.data() + yt * cols;
        uint16_t* orow = out + yt * out_sy + left;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fr + c));
            const __m256i s = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr + c)));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + c));
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i num = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, a), v_2048),
                _mm256_mullo_epi32(_mm256_add_epi32(v, p), _mm256_slli_epi32(amt, 4)));
            const __m256i st = _mm256_srli_epi32(num, 12);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + c), st);
            const __m256i ov = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_srli_epi32(_mm256_add_epi32(st, v_3), 4), v_1), p), 1);
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(ov, ov), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(orow + c), _mm256_castsi256_si128(p16));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t v = sr[c];
            const uint32_t p = fr[c];
            const uint32_t s = gr[c];
            const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
            const uint32_t st = ((4096u - 2u * amt) * acc[c] + 2048u + (v + p) * (amt << 4u)) >> 12u;
            acc[c] = st;
            orow[c] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + p) >> 1u);
        }
    }

    for (int64_t y = h - 2; y >= 0; --y) {
        const uint16_t* sr = src + y * src_sy + left;
        const uint16_t* pr = src + (y + 1) * src_sy + left;
        const uint16_t* gr = guide + y * guide_sy + left;
        const uint32_t* fr = first.data() + y * cols;
        uint16_t* orow = out + y * out_sy + left;
        for (int64_t c = 0; c < vec_cols; c += 8) {
            const __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + c)));
            const __m256i p = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pr + c)));
            const __m256i s = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr + c)));
            const __m256i f = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fr + c));
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + c));
            const __m256i amt = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(v_65536, s), v_tl),
                                 _mm256_mullo_epi32(s, v_tm)), 16);
            const __m256i coef = _mm256_sub_epi32(v_4096, _mm256_slli_epi32(amt, 1));
            const __m256i num = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(coef, a), v_2048),
                _mm256_mullo_epi32(_mm256_add_epi32(v, p), _mm256_slli_epi32(amt, 4)));
            const __m256i st = _mm256_srli_epi32(num, 12);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + c), st);
            const __m256i ov = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_add_epi32(_mm256_srli_epi32(_mm256_add_epi32(st, v_3), 4), v_1), f), 1);
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(ov, ov), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(orow + c), _mm256_castsi256_si128(p16));
        }
        for (int64_t c = vec_cols; c < cols; ++c) {
            const uint32_t v = sr[c];
            const uint32_t p = pr[c];
            const uint32_t s = gr[c];
            const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
            const uint32_t st = ((4096u - 2u * amt) * acc[c] + 2048u + (v + p) * (amt << 4u)) >> 12u;
            acc[c] = st;
            orow[c] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + fr[c]) >> 1u);
        }
    }
}

void select_a_combine_m2_to_m5_avx2(
    const uint32_t* da2, const uint32_t* db2, uint32_t* m2, uint32_t wy2, uint32_t fy2,
    const uint32_t* da3, const uint32_t* db3, uint32_t* m3, uint32_t wy3, uint32_t fy3,
    const uint32_t* da4, const uint32_t* db4, uint32_t* m4, uint32_t wy4, uint32_t fy4,
    const uint32_t* da5, const uint32_t* db5, uint32_t* m5, uint32_t wy5, uint32_t fy5,
    int64_t n_out
) {
    const __m256i vwy2 = _mm256_set1_epi32(static_cast<int32_t>(wy2));
    const __m256i vfy2 = _mm256_set1_epi32(static_cast<int32_t>(fy2));
    const __m256i vwy3 = _mm256_set1_epi32(static_cast<int32_t>(wy3));
    const __m256i vfy3 = _mm256_set1_epi32(static_cast<int32_t>(fy3));
    const __m256i vwy4 = _mm256_set1_epi32(static_cast<int32_t>(wy4));
    const __m256i vfy4 = _mm256_set1_epi32(static_cast<int32_t>(fy4));
    const __m256i vwy5 = _mm256_set1_epi32(static_cast<int32_t>(wy5));
    const __m256i vfy5 = _mm256_set1_epi32(static_cast<int32_t>(fy5));
    int64_t j = 0;
    for (; j + 8 <= n_out; j += 8) {
        const auto step = [j](const uint32_t* da, const uint32_t* db, uint32_t* dst, __m256i wy, __m256i fy) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(da + j));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(db + j));
            const __m256i r = _mm256_add_epi32(_mm256_mullo_epi32(a, wy), _mm256_mullo_epi32(b, fy));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + j), r);
        };
        step(da2, db2, m2, vwy2, vfy2);
        step(da3, db3, m3, vwy3, vfy3);
        step(da4, db4, m4, vwy4, vfy4);
        step(da5, db5, m5, vwy5, vfy5);
    }
    for (; j < n_out; ++j) {
        m2[j] = da2[j] * wy2 + db2[j] * fy2;
        m3[j] = da3[j] * wy3 + db3[j] * fy3;
        m4[j] = da4[j] * wy4 + db4[j] * fy4;
        m5[j] = da5[j] * wy5 + db5[j] * fy5;
    }
}

void select_a_response_pairs_avx2(
    const void* g_row, bool g_is_u16,
    const uint32_t* m0, const uint32_t* m1,
    const uint32_t* m2, const uint32_t* m3,
    const uint32_t* m4, const uint32_t* m5,
    int32_t* response, int64_t full_pairs
) {
    const uint16_t* g16 = static_cast<const uint16_t*>(g_row);
    const int32_t* g32 = static_cast<const int32_t*>(g_row);
    const __m256i v3 = _mm256_set1_epi32(3);
    const __m256i v8 = _mm256_set1_epi32(8);
    const __m256i v32 = _mm256_set1_epi32(32);
    const __m256i v128 = _mm256_set1_epi32(128);
    const __m256i v512 = _mm256_set1_epi32(512);
    const __m256i v2048 = _mm256_set1_epi32(2048);
    const __m256i v_mask16 = _mm256_set1_epi32(0x0000FFFF);

    int64_t j = 0;
    for (; j + 8 <= full_pairs; j += 8) {
        const int64_t ev = 2 * j;
        __m256i ge, go;
        if (g_is_u16) {
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(g16 + ev));
            ge = _mm256_and_si256(raw, v_mask16);
            go = _mm256_srli_epi32(raw, 16);
        } else {
            const __m256 p0 = _mm256_loadu_ps(reinterpret_cast<const float*>(g32 + ev));
            const __m256 p1 = _mm256_loadu_ps(reinterpret_cast<const float*>(g32 + ev + 8));
            ge = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(p0, p1, 0x88)), 0xD8);
            go = _mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_shuffle_ps(p0, p1, 0xDD)), 0xD8);
        }
        const __m256i a0l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m0 + j));
        const __m256i a0c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m0 + j + 1));
        const __m256i a0r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m0 + j + 2));
        const __m256i a0c3 = _mm256_mullo_epi32(a0c, v3);
        const __m256i d0 = _mm256_sub_epi32(_mm256_slli_epi32(ge, 4), _mm256_add_epi32(a0l, a0c3));
        const __m256i d1 = _mm256_sub_epi32(_mm256_slli_epi32(go, 4), _mm256_add_epi32(a0c3, a0r));
        __m256i s0 = _mm256_abs_epi32(d0);
        __m256i s1 = _mm256_abs_epi32(d1);

        const __m256i a1l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m1 + j));
        const __m256i a1c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m1 + j + 1));
        const __m256i a1r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m1 + j + 2));
        const __m256i a1c3 = _mm256_mullo_epi32(a1c, v3);
        s0 = _mm256_add_epi32(s0, _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a1l, a1c3), v8), 4));
        s1 = _mm256_add_epi32(s1, _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(a1c3, a1r), v8), 4));

        const auto accum = [&](const uint32_t* m, __m256i bias, int shift) {
            const __m256i l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + j));
            const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(m + j + 1));
            const __m256i l3 = _mm256_mullo_epi32(l, v3);
            const __m256i r3 = _mm256_mullo_epi32(r, v3);
            s0 = _mm256_add_epi32(s0, _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(l3, r), bias), shift));
            s1 = _mm256_add_epi32(s1, _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(l, r3), bias), shift));
        };
        accum(m2, v32, 6);
        accum(m3, v128, 8);
        accum(m4, v512, 10);
        accum(m5, v2048, 12);

        const __m256i u0 = _mm256_unpacklo_epi32(s0, s1);
        const __m256i u1 = _mm256_unpackhi_epi32(s0, s1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(response + ev), _mm256_permute2x128_si256(u0, u1, 0x20));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(response + ev + 8), _mm256_permute2x128_si256(u0, u1, 0x31));
    }
    for (; j < full_pairs; ++j) {
        const int64_t ev = 2 * j;
        const uint32_t gev = g_is_u16 ? static_cast<uint32_t>(g16[ev]) : static_cast<uint32_t>(g32[ev]);
        const uint32_t god = g_is_u16 ? static_cast<uint32_t>(g16[ev + 1]) : static_cast<uint32_t>(g32[ev + 1]);
        const uint32_t a0_l = m0[j], a0_c = m0[j + 1], a0_r = m0[j + 2];
        const uint32_t a1_l = m1[j], a1_c = m1[j + 1], a1_r = m1[j + 2];
        const int32_t d0 = static_cast<int32_t>(gev << 4u) - static_cast<int32_t>(a0_l + a0_c * 3u);
        const int32_t d1 = static_cast<int32_t>(god << 4u) - static_cast<int32_t>(a0_c * 3u + a0_r);
        int32_t s0 = (d0 < 0) ? -d0 : d0;
        int32_t s1 = (d1 < 0) ? -d1 : d1;
        s0 += static_cast<int32_t>((a1_l + a1_c * 3u + 8u) >> 4u);
        s1 += static_cast<int32_t>((a1_c * 3u + a1_r + 8u) >> 4u);
        s0 += static_cast<int32_t>((m2[j] * 3u + m2[j + 1] + 32u) >> 6u);
        s1 += static_cast<int32_t>((m2[j] + m2[j + 1] * 3u + 32u) >> 6u);
        s0 += static_cast<int32_t>((m3[j] * 3u + m3[j + 1] + 128u) >> 8u);
        s1 += static_cast<int32_t>((m3[j] + m3[j + 1] * 3u + 128u) >> 8u);
        s0 += static_cast<int32_t>((m4[j] * 3u + m4[j + 1] + 512u) >> 10u);
        s1 += static_cast<int32_t>((m4[j] + m4[j + 1] * 3u + 512u) >> 10u);
        s0 += static_cast<int32_t>((m5[j] * 3u + m5[j + 1] + 2048u) >> 12u);
        s1 += static_cast<int32_t>((m5[j] + m5[j + 1] * 3u + 2048u) >> 12u);
        response[ev] = s0;
        response[ev + 1] = s1;
    }
}

void quantize_f32_to_u16_row_avx2(
    const float* sr, uint16_t* dr, int64_t elems, float scale_f32
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    const __m256 v_scale = _mm256_set1_ps(scale_f32);
    const __m256i v_32768 = _mm256_set1_epi32(32768);
    const __m256i v_xor = _mm256_set1_epi16(static_cast<short>(-32768));

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_loadu_ps(sr + i);
        v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_255);
        v = _mm256_mul_ps(v, v_scale);
        __m256i qi = _mm256_cvtps_epi32(v);
        qi = _mm256_sub_epi32(qi, v_32768);
        __m256i p16 = _mm256_packs_epi32(qi, qi);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        p16 = _mm256_xor_si256(p16, v_xor);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dr + i), _mm256_castsi256_si128(p16));
    }
    for (; i < elems; ++i) {
        const float s = sr[i];
        const float c = (s > 0.0f) ? (s < 255.0f ? s : 255.0f) : 0.0f;
        dr[i] = static_cast<uint16_t>(std::nearbyintf(c * scale_f32));
    }
}

void quantize_f32_to_u8_row_avx2(
    const float* sr, uint8_t* dr, int64_t elems
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_255 = _mm256_set1_ps(255.0f);

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_loadu_ps(sr + i);
        v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_255);
        __m256i qi = _mm256_cvtps_epi32(v);
        __m256i p16 = _mm256_packs_epi32(qi, qi);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        __m128i lo = _mm256_castsi256_si128(p16);
        __m128i p8 = _mm_packus_epi16(lo, lo);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dr + i), p8);
    }
    for (; i < elems; ++i) {
        const float s = sr[i];
        const float c = (s > 0.0f) ? (s < 255.0f ? s : 255.0f) : 0.0f;
        dr[i] = static_cast<uint8_t>(std::nearbyintf(c));
    }
}

#else

void pyr_down_vertical_row_f32_avx2(const float*, const float*, const float*, const float*, float*, int64_t, float, float) {}
void pyr_reconstruct_row_f32_avx2(const float*, const float*, const float*, const float*, const float*, float*, int64_t, float, float, float, float, float, float) {}
void pyr_detail_weight_3c_f32_avx2(const float*, const float*, const float*, const float*, const float*, float*, uint16_t*, int64_t, float, float, float, float, float, float, float, bool) {}
void pyr_level0_weights_3c_f32_from_raw_avx2(const float*, const float*, const float*, const float*, const float*, const void*, bool, uint16_t*, int64_t, float, float, float, float, float, float, float, float, bool) {}
void depth_axis_iir_y_unguided_avx2(const uint16_t*, int64_t, uint16_t*, int64_t, int64_t, int64_t, int64_t, int64_t, uint32_t, uint32_t) {}
void depth_axis_iir_y_guided_avx2(const uint16_t*, int64_t, const uint16_t*, int64_t, uint16_t*, int64_t, int64_t, int64_t, int64_t, int64_t, uint32_t, uint32_t) {}
void select_a_combine_m2_to_m5_avx2(const uint32_t*, const uint32_t*, uint32_t*, uint32_t, uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t, uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t, uint32_t, const uint32_t*, const uint32_t*, uint32_t*, uint32_t, uint32_t, int64_t) {}
void select_a_response_pairs_avx2(const void*, bool, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, int32_t*, int64_t) {}
void quantize_f32_to_u16_row_avx2(const float*, uint16_t*, int64_t, float) {}
void quantize_f32_to_u8_row_avx2(const float*, uint8_t*, int64_t) {}

#endif

}  // namespace simd
}  // namespace openfocus
