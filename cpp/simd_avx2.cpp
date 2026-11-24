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

void exposure_gain_u16_row_avx2(
    const uint16_t* sr, uint16_t* dr, int64_t elems, float gain, float limit
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_limit = _mm256_set1_ps(limit);
    const __m256 v_gain = _mm256_set1_ps(gain);
    const __m256i v_32768 = _mm256_set1_epi32(32768);
    const __m256i v_xor = _mm256_set1_epi16(static_cast<short>(-32768));

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        const __m256i u = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + i)));
        __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(u), v_gain);
        v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_limit);
        __m256i qi = _mm256_cvttps_epi32(v);
        qi = _mm256_sub_epi32(qi, v_32768);
        __m256i p16 = _mm256_packs_epi32(qi, qi);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        p16 = _mm256_xor_si256(p16, v_xor);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dr + i), _mm256_castsi256_si128(p16));
    }
    for (; i < elems; ++i) {
        const float v = std::clamp(static_cast<float>(sr[i]) * gain, 0.0f, limit);
        dr[i] = static_cast<uint16_t>(static_cast<int32_t>(v));
    }
}

void exposure_gain_u8_row_avx2(
    const uint8_t* sr, uint8_t* dr, int64_t elems, float gain, float limit
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_limit = _mm256_set1_ps(limit);
    const __m256 v_gain = _mm256_set1_ps(gain);

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        const __m256i u = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(sr + i)));
        __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(u), v_gain);
        v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_limit);
        __m256i qi = _mm256_cvttps_epi32(v);
        __m256i p16 = _mm256_packs_epi32(qi, qi);
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);
        __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dr + i), p8);
    }
    for (; i < elems; ++i) {
        const float v = std::clamp(static_cast<float>(sr[i]) * gain, 0.0f, limit);
        dr[i] = static_cast<uint8_t>(static_cast<int32_t>(v));
    }
}

void exposure_gain_f32_row_avx2(
    const float* sr, float* dr, int64_t elems, float gain, float limit
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_limit = _mm256_set1_ps(limit);
    const __m256 v_gain = _mm256_set1_ps(gain);

    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(sr + i), v_gain);
        v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_limit);
        _mm256_storeu_ps(dr + i, v);
    }
    for (; i < elems; ++i) {
        dr[i] = std::clamp(sr[i] * gain, 0.0f, limit);
    }
}

void down2_row_u16_to_i32_avx2(
    const uint16_t* r0, const uint16_t* r1, int32_t* dst, int64_t full_w, int32_t scale
) {
    const __m256i v_mask16 = _mm256_set1_epi32(0x0000FFFF);
    const __m256i v_scale = _mm256_set1_epi32(scale);
    int64_t x = 0;
    for (; x + 8 <= full_w; x += 8) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * x));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 2 * x));
        const __m256i pa = _mm256_add_epi32(_mm256_and_si256(a, v_mask16), _mm256_srli_epi32(a, 16));
        const __m256i pb = _mm256_add_epi32(_mm256_and_si256(b, v_mask16), _mm256_srli_epi32(b, 16));
        const __m256i tot = _mm256_add_epi32(pa, pb);
        const __m256i out = _mm256_srai_epi32(_mm256_mullo_epi32(tot, v_scale), 2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), out);
    }
    for (; x < full_w; ++x) {
        const int32_t left = static_cast<int32_t>(2 * x);
        const int32_t tot = static_cast<int32_t>(r0[left]) + static_cast<int32_t>(r0[left + 1])
                          + static_cast<int32_t>(r1[left]) + static_cast<int32_t>(r1[left + 1]);
        dst[x] = (tot * scale) >> 2;
    }
}

void down2_row_i32_to_i32_avx2(
    const int32_t* r0, const int32_t* r1, int32_t* dst, int64_t full_w, int32_t scale
) {
    const __m256i v_scale = _mm256_set1_epi32(scale);
    int64_t x = 0;
    for (; x + 8 <= full_w; x += 8) {
        const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * x));
        const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * x + 8));
        const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 2 * x));
        const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 2 * x + 8));
        const __m256i s0 = _mm256_add_epi32(a0, b0);
        const __m256i s1 = _mm256_add_epi32(a1, b1);
        const __m256i h = _mm256_permute4x64_epi64(_mm256_hadd_epi32(s0, s1), 0xD8);
        const __m256i out = _mm256_srai_epi32(_mm256_mullo_epi32(h, v_scale), 2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), out);
    }
    for (; x < full_w; ++x) {
        const int64_t left = 2 * x;
        const int32_t tot = r0[left] + r0[left + 1] + r1[left] + r1[left + 1];
        dst[x] = (tot * scale) >> 2;
    }
}

void gray_rgb_u8_row_avx2(
    const uint8_t* sr, uint16_t* dst16, int32_t* dst32, int64_t width,
    int32_t scale, bool depth_color, bool dst_is_u16
) {
    const __m256i v_magic = _mm256_set1_epi32(static_cast<int32_t>(0xAAAAAAABu));
    const __m256i v_scale = _mm256_set1_epi32(scale);
    const __m256i v_s32 = _mm256_set1_epi32(scale * 85);
    int64_t x = 0;
    for (; x + 8 <= width; x += 8) {
        const uint8_t* p = sr + x * 3;
        const __m256i t = _mm256_setr_epi32(
            static_cast<int32_t>(p[0]) + p[1] + p[2],
            static_cast<int32_t>(p[3]) + p[4] + p[5],
            static_cast<int32_t>(p[6]) + p[7] + p[8],
            static_cast<int32_t>(p[9]) + p[10] + p[11],
            static_cast<int32_t>(p[12]) + p[13] + p[14],
            static_cast<int32_t>(p[15]) + p[16] + p[17],
            static_cast<int32_t>(p[18]) + p[19] + p[20],
            static_cast<int32_t>(p[21]) + p[22] + p[23]
        );
        __m256i res;
        if (depth_color) {
            res = _mm256_srli_epi32(_mm256_mullo_epi32(t, v_s32), 8);
        } else {
            const __m256i q_ev = _mm256_srli_epi64(_mm256_mul_epu32(t, v_magic), 33);
            const __m256i t_od = _mm256_srli_epi64(t, 32);
            const __m256i q_od = _mm256_slli_epi64(_mm256_srli_epi64(_mm256_mul_epu32(t_od, v_magic), 33), 32);
            const __m256i q = _mm256_blend_epi32(q_ev, q_od, 0xAA);
            res = _mm256_mullo_epi32(q, v_scale);
        }
        if (dst_is_u16) {
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(res, res), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst16 + x), _mm256_castsi256_si128(p16));
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst32 + x), res);
        }
    }
    for (; x < width; ++x) {
        const uint8_t* p = sr + x * 3;
        const uint32_t tot = static_cast<uint32_t>(p[0]) + p[1] + p[2];
        const uint32_t v = depth_color ? ((tot * static_cast<uint32_t>(scale * 85)) >> 8u)
                                       : ((tot / 3u) * static_cast<uint32_t>(scale));
        if (dst_is_u16) {
            dst16[x] = static_cast<uint16_t>(v);
        } else {
            dst32[x] = static_cast<int32_t>(v);
        }
    }
}

void gray_rgb_u16_row_avx2(
    const uint16_t* sr, uint16_t* dst16, int32_t* dst32, int64_t width,
    int32_t scale, bool depth_color, bool dst_is_u16
) {
    const __m256i v_magic = _mm256_set1_epi32(static_cast<int32_t>(0xAAAAAAABu));
    const __m256i v_scale = _mm256_set1_epi32(scale);
    const __m256i v_s32 = _mm256_set1_epi32(scale * 85);
    int64_t x = 0;
    for (; x + 8 <= width; x += 8) {
        const uint16_t* p = sr + x * 3;
        const __m256i t = _mm256_setr_epi32(
            static_cast<int32_t>(p[0]) + p[1] + p[2],
            static_cast<int32_t>(p[3]) + p[4] + p[5],
            static_cast<int32_t>(p[6]) + p[7] + p[8],
            static_cast<int32_t>(p[9]) + p[10] + p[11],
            static_cast<int32_t>(p[12]) + p[13] + p[14],
            static_cast<int32_t>(p[15]) + p[16] + p[17],
            static_cast<int32_t>(p[18]) + p[19] + p[20],
            static_cast<int32_t>(p[21]) + p[22] + p[23]
        );
        __m256i res;
        if (depth_color) {
            res = _mm256_srli_epi32(_mm256_mullo_epi32(t, v_s32), 8);
        } else {
            const __m256i q_ev = _mm256_srli_epi64(_mm256_mul_epu32(t, v_magic), 33);
            const __m256i t_od = _mm256_srli_epi64(t, 32);
            const __m256i q_od = _mm256_slli_epi64(_mm256_srli_epi64(_mm256_mul_epu32(t_od, v_magic), 33), 32);
            const __m256i q = _mm256_blend_epi32(q_ev, q_od, 0xAA);
            res = _mm256_mullo_epi32(q, v_scale);
        }
        if (dst_is_u16) {
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(res, res), 0xD8);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst16 + x), _mm256_castsi256_si128(p16));
        } else {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst32 + x), res);
        }
    }
    for (; x < width; ++x) {
        const uint16_t* p = sr + x * 3;
        const uint32_t tot = static_cast<uint32_t>(p[0]) + p[1] + p[2];
        const uint32_t v = depth_color ? ((tot * static_cast<uint32_t>(scale * 85)) >> 8u)
                                       : ((tot / 3u) * static_cast<uint32_t>(scale));
        if (dst_is_u16) {
            dst16[x] = static_cast<uint16_t>(v);
        } else {
            dst32[x] = static_cast<int32_t>(v);
        }
    }
}

namespace {
template <typename SrcT, typename DstT>
void merge_recip_row_impl(
    const uint16_t* drow, const SrcT* irow, DstT* arow, int64_t width,
    int64_t is1, int64_t is2, int64_t as1, int64_t as2,
    int32_t base32, int32_t step32, uint64_t recip48, int32_t s32
) {
    const int32_t neg_step = -step32;
    const __m256i v_base = _mm256_set1_epi32(base32);
    const __m256i v_step = _mm256_set1_epi32(step32);
    const __m256i v_nstep = _mm256_set1_epi32(neg_step);
    const bool packed3 = (is1 == 3 && is2 == 1 && as1 == 3 && as2 == 1);

    int64_t x = 0;
    while (x < width) {
        if (x + 16 <= width) {
            const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(drow + x));
            const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(drow + x + 8));
            const __m256i d0 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(r0), v_base);
            const __m256i d1 = _mm256_sub_epi32(_mm256_cvtepu16_epi32(r1), v_base);
            const __m256i a0 = _mm256_andnot_si256(_mm256_cmpgt_epi32(d0, v_step), _mm256_cmpgt_epi32(d0, v_nstep));
            const __m256i a1 = _mm256_andnot_si256(_mm256_cmpgt_epi32(d1, v_step), _mm256_cmpgt_epi32(d1, v_nstep));
            const __m256i any = _mm256_or_si256(a0, a1);
            if (_mm256_testz_si256(any, any)) {
                x += 16;
                continue;
            }
        }
        const int64_t x_end = (x + 16 <= width) ? (x + 16) : width;
        for (; x < x_end; ++x) {
            const int32_t delta = static_cast<int32_t>(drow[x]) - base32;
            if (delta >= 0) {
                if (delta <= step32) {
                    const SrcT* sp = irow + x * is1;
                    DstT* dp = arow + x * as1;
                    if (packed3) {
                        dp[0] = static_cast<DstT>(static_cast<int32_t>(sp[0]) * s32);
                        dp[1] = static_cast<DstT>(static_cast<int32_t>(sp[1]) * s32);
                        dp[2] = static_cast<DstT>(static_cast<int32_t>(sp[2]) * s32);
                    } else {
                        for (int64_t ch = 0; ch < 3; ++ch) {
                            dp[ch * as2] = static_cast<DstT>(static_cast<int32_t>(sp[ch * is2]) * s32);
                        }
                    }
                }
            } else if (delta > neg_step) {
                const int32_t w = 1024 - static_cast<int32_t>(
                    (static_cast<uint64_t>(-delta) << 10u) * recip48 >> 48u);
                if (w > 0) {
                    const int32_t iw = 1024 - w;
                    const SrcT* sp = irow + x * is1;
                    DstT* dp = arow + x * as1;
                    if (packed3) {
                        dp[0] = static_cast<DstT>((static_cast<int32_t>(dp[0]) * iw + static_cast<int32_t>(sp[0]) * s32 * w) >> 10);
                        dp[1] = static_cast<DstT>((static_cast<int32_t>(dp[1]) * iw + static_cast<int32_t>(sp[1]) * s32 * w) >> 10);
                        dp[2] = static_cast<DstT>((static_cast<int32_t>(dp[2]) * iw + static_cast<int32_t>(sp[2]) * s32 * w) >> 10);
                    } else {
                        for (int64_t ch = 0; ch < 3; ++ch) {
                            const int32_t cur = static_cast<int32_t>(dp[ch * as2]);
                            const int32_t val = static_cast<int32_t>(sp[ch * is2]) * s32;
                            dp[ch * as2] = static_cast<DstT>((cur * iw + val * w) >> 10);
                        }
                    }
                }
            }
        }
    }
}

template <typename SrcT, typename DstT>
int select_a_no_bg_row_impl(
    int32_t* b_row, const int32_t* response, int64_t width,
    const SrcT* irow, DstT* orow, int64_t is1, int64_t is2, int64_t os1, int64_t os2,
    int64_t scale, int state, int64_t max_response, int sharp_state, int saturated_state
) {
    int64_t x = 0;
    while (x < width) {
        if (state != saturated_state && x + 8 <= width) {
            const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(response + x));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_row + x));
            if (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(r, b))) == 0) {
                x += 8;
                continue;
            }
        }
        const int64_t x_end = (state != saturated_state && x + 8 <= width) ? (x + 8) : width;
        for (; x < x_end; ++x) {
            const int64_t prev = b_row[x];
            if (prev == max_response) {
                continue;
            }
            int64_t sc = response[x];
            if (sc >= max_response) {
                sc = max_response;
            } else if (state == saturated_state) {
                state = sharp_state;
            }
            if (sc > prev) {
                b_row[x] = static_cast<int32_t>(sc);
                const SrcT* sp = irow + x * is1;
                DstT* dp = orow + x * os1;
                dp[0] = static_cast<DstT>(static_cast<int64_t>(sp[0]) * scale);
                dp[os2] = static_cast<DstT>(static_cast<int64_t>(sp[is2]) * scale);
                dp[2 * os2] = static_cast<DstT>(static_cast<int64_t>(sp[2 * is2]) * scale);
            }
        }
    }
    return state;
}
}  // namespace

void merge_recip_packed_row_avx2(
    const uint16_t* drow, const void* irow, bool src_u8,
    void* arow, bool dst_u16, int64_t width,
    int64_t is1, int64_t is2, int64_t as1, int64_t as2,
    int32_t base32, int32_t step32, uint64_t recip48, int32_t s32
) {
    if (src_u8 && dst_u16) {
        merge_recip_row_impl<uint8_t, uint16_t>(
            drow, static_cast<const uint8_t*>(irow), static_cast<uint16_t*>(arow),
            width, is1, is2, as1, as2, base32, step32, recip48, s32);
    } else if (src_u8 && !dst_u16) {
        merge_recip_row_impl<uint8_t, int32_t>(
            drow, static_cast<const uint8_t*>(irow), static_cast<int32_t*>(arow),
            width, is1, is2, as1, as2, base32, step32, recip48, s32);
    } else if (!src_u8 && dst_u16) {
        merge_recip_row_impl<uint16_t, uint16_t>(
            drow, static_cast<const uint16_t*>(irow), static_cast<uint16_t*>(arow),
            width, is1, is2, as1, as2, base32, step32, recip48, s32);
    } else {
        merge_recip_row_impl<uint16_t, int32_t>(
            drow, static_cast<const uint16_t*>(irow), static_cast<int32_t*>(arow),
            width, is1, is2, as1, as2, base32, step32, recip48, s32);
    }
}

int select_a_no_bg_packed_row_avx2(
    int32_t* b_row, const int32_t* response, int64_t width,
    const void* irow, bool src_u8, void* orow, bool dst_u16,
    int64_t is1, int64_t is2, int64_t os1, int64_t os2,
    int64_t scale, int state, int64_t max_response, int sharp_state, int saturated_state
) {
    if (src_u8 && dst_u16) {
        return select_a_no_bg_row_impl<uint8_t, uint16_t>(
            b_row, response, width, static_cast<const uint8_t*>(irow), static_cast<uint16_t*>(orow),
            is1, is2, os1, os2, scale, state, max_response, sharp_state, saturated_state);
    }
    if (src_u8 && !dst_u16) {
        return select_a_no_bg_row_impl<uint8_t, int32_t>(
            b_row, response, width, static_cast<const uint8_t*>(irow), static_cast<int32_t*>(orow),
            is1, is2, os1, os2, scale, state, max_response, sharp_state, saturated_state);
    }
    if (!src_u8 && dst_u16) {
        return select_a_no_bg_row_impl<uint16_t, uint16_t>(
            b_row, response, width, static_cast<const uint16_t*>(irow), static_cast<uint16_t*>(orow),
            is1, is2, os1, os2, scale, state, max_response, sharp_state, saturated_state);
    }
    return select_a_no_bg_row_impl<uint16_t, int32_t>(
        b_row, response, width, static_cast<const uint16_t*>(irow), static_cast<int32_t*>(orow),
        is1, is2, os1, os2, scale, state, max_response, sharp_state, saturated_state);
}

void tile_accumulate_f32_1c_row_avx2(
    const float* tr, const float* wr, float* ar, float* war, int64_t fw
) {
    int64_t dx = 0;
    for (; dx + 8 <= fw; dx += 8) {
        const __m256 w = _mm256_loadu_ps(wr + dx);
        const __m256 wa = _mm256_loadu_ps(war + dx);
        const __m256 t = _mm256_loadu_ps(tr + dx);
        const __m256 a = _mm256_loadu_ps(ar + dx);
        _mm256_storeu_ps(war + dx, _mm256_add_ps(wa, w));
        _mm256_storeu_ps(ar + dx, _mm256_add_ps(a, _mm256_mul_ps(t, w)));
    }
    for (; dx < fw; ++dx) {
        const float w = wr[dx];
        war[dx] += w;
        ar[dx] += tr[dx] * w;
    }
}

void tile_finalize_3c_row_avx2(
    const float* ar, const float* war, uint8_t* u8_dr, float* f32_dr,
    int64_t width, int64_t channels, bool out_u8
) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_255 = _mm256_set1_ps(255.0f);
    if (channels == 1) {
        int64_t x = 0;
        for (; x + 8 <= width; x += 8) {
            const __m256 w = _mm256_loadu_ps(war + x);
            const __m256 den = _mm256_blendv_ps(w, v_one, _mm256_cmp_ps(w, v_zero, _CMP_EQ_OQ));
            const __m256 raw = _mm256_div_ps(_mm256_loadu_ps(ar + x), den);
            __m256 v = _mm256_blendv_ps(raw, v_zero, _mm256_cmp_ps(raw, v_zero, _CMP_LE_OQ));
            v = _mm256_blendv_ps(v, v_255, _mm256_cmp_ps(v, v_255, _CMP_GE_OQ));
            if (out_u8) {
                const __m256i qi = _mm256_cvtps_epi32(v);
                const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(qi, qi), 0xD8);
                const __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(u8_dr + x), p8);
            } else {
                _mm256_storeu_ps(f32_dr + x, v);
            }
        }
        for (; x < width; ++x) {
            const float w = war[x];
            const float raw = ar[x] / ((w == 0.0f) ? 1.0f : w);
            const float val = (raw > 0.0f) ? (raw < 255.0f ? raw : 255.0f) : 0.0f;
            if (out_u8) {
                u8_dr[x] = static_cast<uint8_t>(std::nearbyintf(val));
            } else {
                f32_dr[x] = val;
            }
        }
        return;
    }

    int64_t x = 0;
    alignas(32) float d_buf[24];
    if (channels == 3) {
        for (; x + 8 <= width; x += 8) {
            for (int k = 0; k < 8; ++k) {
                const float w = war[x + k];
                const float d = (w == 0.0f) ? 1.0f : w;
                d_buf[k * 3] = d;
                d_buf[k * 3 + 1] = d;
                d_buf[k * 3 + 2] = d;
            }
            const int64_t off = x * 3;
            for (int blk = 0; blk < 3; ++blk) {
                const int64_t b = off + blk * 8;
                const __m256 raw = _mm256_div_ps(_mm256_loadu_ps(ar + b), _mm256_load_ps(d_buf + blk * 8));
                __m256 v = _mm256_blendv_ps(raw, v_zero, _mm256_cmp_ps(raw, v_zero, _CMP_LE_OQ));
                v = _mm256_blendv_ps(v, v_255, _mm256_cmp_ps(v, v_255, _CMP_GE_OQ));
                if (out_u8) {
                    const __m256i qi = _mm256_cvtps_epi32(v);
                    const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(qi, qi), 0xD8);
                    const __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(u8_dr + b), p8);
                } else {
                    _mm256_storeu_ps(f32_dr + b, v);
                }
            }
        }
    }
    for (; x < width; ++x) {
        const float w = war[x];
        const float denom = (w == 0.0f) ? 1.0f : w;
        const int64_t off = x * channels;
        for (int64_t c = 0; c < channels; ++c) {
            const float raw = ar[off + c] / denom;
            const float val = (raw > 0.0f) ? (raw < 255.0f ? raw : 255.0f) : 0.0f;
            if (out_u8) {
                u8_dr[off + c] = static_cast<uint8_t>(std::nearbyintf(val));
            } else {
                f32_dr[off + c] = val;
            }
        }
    }
}

bool pyr_any_gt_u16_avx2(const uint16_t* wr, const uint16_t* br, int64_t width) {
    const __m256i v_ones = _mm256_set1_epi32(-1);
    int64_t x = 0;
    for (; x + 16 <= width; x += 16) {
        const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + x));
        const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
        const __m256i mx = _mm256_max_epu16(wv, bv);
        const __m256i le = _mm256_cmpeq_epi16(mx, bv);
        if (!_mm256_testc_si256(le, v_ones)) {
            return true;
        }
    }
    for (; x < width; ++x) {
        if (wr[x] > br[x]) {
            return true;
        }
    }
    return false;
}

void pyr_level0_merge_3c_row_avx2(
    const uint16_t* wr, uint16_t* br, uint16_t* win_row,
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    const void* raw_row, bool raw_is_u8, float* acc_row,
    int64_t width, float factor, uint16_t index, bool first,
    float w0, float w1, float w2, float w3, float w4, float point_one
) {
    const __m256i v_ones = _mm256_set1_epi32(-1);
    const uint8_t* s8 = static_cast<const uint8_t*>(raw_row);
    const uint16_t* s16 = static_cast<const uint16_t*>(raw_row);
    int64_t x = 0;
    while (x < width) {
        if (!first && x + 16 <= width) {
            const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + x));
            const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
            const __m256i mx = _mm256_max_epu16(wv, bv);
            if (_mm256_testc_si256(_mm256_cmpeq_epi16(mx, bv), v_ones)) {
                x += 16;
                continue;
            }
        }
        const int64_t x_end = (!first && x + 16 <= width) ? (x + 16) : width;
        for (; x < x_end; ++x) {
            const uint16_t wval = wr[x];
            if (!first && wval <= br[x]) {
                continue;
            }
            br[x] = wval;
            if (win_row != nullptr) {
                win_row[x] = index;
            }
            const int64_t off3 = x * 3;
            float lo0 = (hr0[off3] * w0 + hr1[off3] * w1) + hr2[off3] * w2;
            float lo1 = (hr0[off3 + 1] * w0 + hr1[off3 + 1] * w1) + hr2[off3 + 1] * w2;
            float lo2 = (hr0[off3 + 2] * w0 + hr1[off3 + 2] * w1) + hr2[off3 + 2] * w2;
            lo0 = (lo0 + hr3[off3] * w3) + hr4[off3] * w4;
            lo1 = (lo1 + hr3[off3 + 1] * w3) + hr4[off3 + 1] * w4;
            lo2 = (lo2 + hr3[off3 + 2] * w3) + hr4[off3 + 2] * w4;
            float s0, s1, s2;
            if (raw_is_u8) {
                s0 = static_cast<float>(s8[off3]) * factor;
                s1 = static_cast<float>(s8[off3 + 1]) * factor;
                s2 = static_cast<float>(s8[off3 + 2]) * factor;
            } else {
                s0 = static_cast<float>(s16[off3]) * factor;
                s1 = static_cast<float>(s16[off3 + 1]) * factor;
                s2 = static_cast<float>(s16[off3 + 2]) * factor;
            }
            acc_row[off3] = (s0 - lo0) / (lo0 + point_one);
            acc_row[off3 + 1] = (s1 - lo1) / (lo1 + point_one);
            acc_row[off3 + 2] = (s2 - lo2) / (lo2 + point_one);
        }
    }
}

void pyr_merge_f32_3c_row_avx2(
    const uint16_t* wr, uint16_t* br, uint16_t* win_row,
    const float* dr, float* ar, int64_t width, uint16_t index
) {
    const __m256i v_ones = _mm256_set1_epi32(-1);
    int64_t x = 0;
    while (x < width) {
        if (x + 16 <= width) {
            const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + x));
            const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
            const __m256i mx = _mm256_max_epu16(wv, bv);
            if (_mm256_testc_si256(_mm256_cmpeq_epi16(mx, bv), v_ones)) {
                x += 16;
                continue;
            }
        }
        const int64_t x_end = (x + 16 <= width) ? (x + 16) : width;
        for (; x < x_end; ++x) {
            const uint16_t wval = wr[x];
            if (wval > br[x]) {
                br[x] = wval;
                if (win_row != nullptr) {
                    win_row[x] = index;
                }
                const int64_t off = x * 3;
                ar[off] = dr[off];
                ar[off + 1] = dr[off + 1];
                ar[off + 2] = dr[off + 2];
            }
        }
    }
}

void pyr_output_f32_to_u8_row_avx2(const float* sr, uint8_t* dr, int64_t elems) {
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_65535 = _mm256_set1_ps(65535.0f);
    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_loadu_ps(sr + i);
        v = _mm256_blendv_ps(v, v_zero, _mm256_cmp_ps(v, v_zero, _CMP_LT_OQ));
        v = _mm256_blendv_ps(v, v_one, _mm256_cmp_ps(v, v_one, _CMP_GT_OQ));
        const __m256i q = _mm256_srli_epi32(_mm256_cvttps_epi32(_mm256_mul_ps(v, v_65535)), 8);
        const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q, q), 0xD8);
        const __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_castsi256_si128(p16));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dr + i), p8);
    }
    for (; i < elems; ++i) {
        const float v = sr[i] < 0.0f ? 0.0f : (sr[i] > 1.0f ? 1.0f : sr[i]);
        dr[i] = static_cast<uint8_t>(static_cast<int32_t>(v * 65535.0f) >> 8);
    }
}

void pyr_output_f32_to_f32_row_avx2(const float* sr, float* dr, int64_t elems) {
    constexpr float kInv257 = static_cast<float>(1.0 / 257.0);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_65535 = _mm256_set1_ps(65535.0f);
    const __m256 v_inv257 = _mm256_set1_ps(kInv257);
    int64_t i = 0;
    for (; i + 8 <= elems; i += 8) {
        __m256 v = _mm256_loadu_ps(sr + i);
        v = _mm256_blendv_ps(v, v_zero, _mm256_cmp_ps(v, v_zero, _CMP_LT_OQ));
        v = _mm256_blendv_ps(v, v_one, _mm256_cmp_ps(v, v_one, _CMP_GT_OQ));
        const __m256i qi = _mm256_cvtps_epi32(_mm256_mul_ps(v, v_65535));
        const __m256 qf = _mm256_mul_ps(_mm256_cvtepi32_ps(qi), v_inv257);
        _mm256_storeu_ps(dr + i, qf);
    }
    for (; i < elems; ++i) {
        const float v = sr[i] < 0.0f ? 0.0f : (sr[i] > 1.0f ? 1.0f : sr[i]);
        dr[i] = std::nearbyintf(v * 65535.0f) * kInv257;
    }
}

void pyr_up_horiz_1c_row_avx2(
    const float* sr, float* dr, int64_t ix_start, int64_t pair_stop,
    float e0, float e1, float e2, float e3
) {
    const __m256 ve0 = _mm256_set1_ps(e0);
    const __m256 ve1 = _mm256_set1_ps(e1);
    const __m256 ve2 = _mm256_set1_ps(e2);
    const __m256 ve3 = _mm256_set1_ps(e3);
    int64_t x = ix_start;
    for (; x + 16 <= pair_stop; x += 16) {
        const int64_t left = x / 2 - 2;
        const float* p = sr + left;
        const __m256 s0 = _mm256_loadu_ps(p);
        const __m256 s1 = _mm256_loadu_ps(p + 1);
        const __m256 s2 = _mm256_loadu_ps(p + 2);
        const __m256 s3 = _mm256_loadu_ps(p + 3);
        const __m256 s4 = _mm256_loadu_ps(p + 4);
        const __m256 ve = _mm256_add_ps(
            _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(s0, ve0), _mm256_mul_ps(s1, ve1)),
                          _mm256_mul_ps(s2, ve2)),
            _mm256_mul_ps(s3, ve3));
        const __m256 vo = _mm256_add_ps(
            _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(s1, ve3), _mm256_mul_ps(s2, ve2)),
                          _mm256_mul_ps(s3, ve1)),
            _mm256_mul_ps(s4, ve0));
        const __m256 lo = _mm256_unpacklo_ps(ve, vo);
        const __m256 hi = _mm256_unpackhi_ps(ve, vo);
        _mm256_storeu_ps(dr + x, _mm256_permute2f128_ps(lo, hi, 0x20));
        _mm256_storeu_ps(dr + x + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
    }
    for (; x < pair_stop; x += 2) {
        const int64_t left = x / 2 - 2;
        const float* p = sr + left;
        const float ve = (p[0] * e0 + p[1] * e1) + p[2] * e2;
        dr[x] = ve + p[3] * e3;
        const float vo = p[1] * e3 + p[2] * e2;
        dr[x + 1] = (vo + p[3] * e1) + p[4] * e0;
    }
}

void pyr_up_horiz_3c_row_avx2(
    const float* sr, float* dr, int64_t ix_start, int64_t pair_stop,
    float e0, float e1, float e2, float e3
) {
    // Keep last pair scalar to guarantee 4-float loads stay strictly inside row bounds.
    const int64_t safe_stop = (pair_stop > ix_start) ? (pair_stop - 2) : ix_start;
    const __m256 vw_a = _mm256_set_m128(_mm_set1_ps(e3), _mm_set1_ps(e0));
    const __m256 vw_b = _mm256_set_m128(_mm_set1_ps(e2), _mm_set1_ps(e1));
    const __m256 vw_c = _mm256_set_m128(_mm_set1_ps(e1), _mm_set1_ps(e2));
    const __m256 vw_d = _mm256_set_m128(_mm_set1_ps(e0), _mm_set1_ps(e3));
    int64_t x = ix_start;
    alignas(16) float tmp[8];
    for (; x < safe_stop; x += 2) {
        const int64_t left = x / 2 - 2;
        const float* p = sr + left * 3;
        const __m128 a = _mm_loadu_ps(p);
        const __m128 b = _mm_loadu_ps(p + 3);
        const __m128 c = _mm_loadu_ps(p + 6);
        const __m128 d = _mm_loadu_ps(p + 9);
        const __m128 e = _mm_loadu_ps(p + 12);
        const __m256 p0 = _mm256_set_m128(b, a);
        const __m256 p1 = _mm256_set_m128(c, b);
        const __m256 p2 = _mm256_set_m128(d, c);
        const __m256 p3 = _mm256_set_m128(e, d);
        __m256 v = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(p0, vw_a), _mm256_mul_ps(p1, vw_b)),
            _mm256_mul_ps(p2, vw_c));
        v = _mm256_add_ps(v, _mm256_mul_ps(p3, vw_d));
        _mm256_storeu_ps(tmp, v);
        float* d0 = dr + x * 3;
        d0[0] = tmp[0]; d0[1] = tmp[1]; d0[2] = tmp[2];
        d0[3] = tmp[4]; d0[4] = tmp[5]; d0[5] = tmp[6];
    }
    for (; x < pair_stop; x += 2) {
        const int64_t left = x / 2 - 2;
        const float* p = sr + left * 3;
        float* d0 = dr + x * 3;
        float* d1 = dr + (x + 1) * 3;
        for (int64_t ch = 0; ch < 3; ++ch) {
            const float s0 = p[ch], s1 = p[3 + ch], s2 = p[6 + ch], s3 = p[9 + ch], s4 = p[12 + ch];
            float ve = (s0 * e0 + s1 * e1) + s2 * e2;
            ve = (ve + s3 * e3) + 0.0f;
            d0[ch] = ve;
            float vo = (0.0f + s1 * e3) + s2 * e2;
            vo = (vo + s3 * e1) + s4 * e0;
            d1[ch] = vo;
        }
    }
}

void pyr_down_horiz_f32_3c_row_avx2(
    const float* sr, float* h_row, int64_t ix_stop, float d0, float d1
) {
    const __m128 vd0 = _mm_set1_ps(d0);
    const __m128 vd1 = _mm_set1_ps(d1);
    const int64_t safe_stop = (ix_stop > 1) ? (ix_stop - 1) : 1;
    int64_t x = 1;
    alignas(16) float out4[4];
    for (; x < safe_stop; ++x) {
        const float* p = sr + (2 * x - 1) * 3;
        __m128 v = _mm_mul_ps(_mm_loadu_ps(p), vd0);
        v = _mm_add_ps(v, _mm_mul_ps(_mm_loadu_ps(p + 3), vd1));
        v = _mm_add_ps(v, _mm_mul_ps(_mm_loadu_ps(p + 6), vd1));
        v = _mm_add_ps(v, _mm_mul_ps(_mm_loadu_ps(p + 9), vd0));
        _mm_store_ps(out4, v);
        float* dst = h_row + x * 3;
        dst[0] = out4[0];
        dst[1] = out4[1];
        dst[2] = out4[2];
    }
    for (; x < ix_stop; ++x) {
        const float* p = sr + (2 * x - 1) * 3;
        float* dst = h_row + x * 3;
        for (int64_t c = 0; c < 3; ++c) {
            float v = p[c] * d0;
            v += p[3 + c] * d1;
            v += p[6 + c] * d1;
            dst[c] = v + p[9 + c] * d0;
        }
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
void exposure_gain_u16_row_avx2(const uint16_t*, uint16_t*, int64_t, float, float) {}
void exposure_gain_u8_row_avx2(const uint8_t*, uint8_t*, int64_t, float, float) {}
void exposure_gain_f32_row_avx2(const float*, float*, int64_t, float, float) {}
void down2_row_u16_to_i32_avx2(const uint16_t*, const uint16_t*, int32_t*, int64_t, int32_t) {}
void down2_row_i32_to_i32_avx2(const int32_t*, const int32_t*, int32_t*, int64_t, int32_t) {}
void gray_rgb_u8_row_avx2(const uint8_t*, uint16_t*, int32_t*, int64_t, int32_t, bool, bool) {}
void gray_rgb_u16_row_avx2(const uint16_t*, uint16_t*, int32_t*, int64_t, int32_t, bool, bool) {}
void merge_recip_packed_row_avx2(const uint16_t*, const void*, bool, void*, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, int32_t, uint64_t, int32_t) {}
int select_a_no_bg_packed_row_avx2(int32_t*, const int32_t*, int64_t, const void*, bool, void*, bool, int64_t, int64_t, int64_t, int64_t, int64_t, int state, int64_t, int, int) { return state; }
void tile_accumulate_f32_1c_row_avx2(const float*, const float*, float*, float*, int64_t) {}
void tile_finalize_3c_row_avx2(const float*, const float*, uint8_t*, float*, int64_t, int64_t, bool) {}
bool pyr_any_gt_u16_avx2(const uint16_t*, const uint16_t*, int64_t) { return false; }
void pyr_level0_merge_3c_row_avx2(const uint16_t*, uint16_t*, uint16_t*, const float*, const float*, const float*, const float*, const float*, const void*, bool, float*, int64_t, float, uint16_t, bool, float, float, float, float, float, float) {}
void pyr_merge_f32_3c_row_avx2(const uint16_t*, uint16_t*, uint16_t*, const float*, float*, int64_t, uint16_t) {}
void pyr_output_f32_to_u8_row_avx2(const float*, uint8_t*, int64_t) {}
void pyr_output_f32_to_f32_row_avx2(const float*, float*, int64_t) {}
void pyr_up_horiz_1c_row_avx2(const float*, float*, int64_t, int64_t, float, float, float, float) {}
void pyr_up_horiz_3c_row_avx2(const float*, float*, int64_t, int64_t, float, float, float, float) {}
void pyr_down_horiz_f32_3c_row_avx2(const float*, float*, int64_t, float, float) {}

#endif

}  // namespace simd
}  // namespace openfocus
