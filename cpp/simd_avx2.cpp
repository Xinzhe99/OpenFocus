#include "simd_dispatch.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
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

struct CpuFeatures {
    bool avx2 = false;
    bool fma = false;
};

CpuFeatures detect_cpu_features() {
#if OPENFOCUS_X86_64
    int info1[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    __cpuid(info1, 0);
    const int max_leaf = info1[0];
    if (max_leaf < 7) {
        return {};
    }
    __cpuidex(info1, 1, 0);
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx) || eax < 7) {
        return {};
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
        return {};
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
        return {};
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
    // Leaf 7 EBX bit 5 = AVX2, Leaf 1 ECX bit 12 = FMA3
    const bool has_avx2 = (ebx7 & (1u << 5)) != 0;
    const bool has_fma = has_avx2 && ((ecx1 & (1u << 12)) != 0);
    return {has_avx2, has_fma};
#else
    return {};
#endif
}

const CpuFeatures& cpu_features() {
    static const CpuFeatures kFeatures = detect_cpu_features();
    return kFeatures;
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
    return !g_force_scalar.load(std::memory_order_relaxed) && cpu_features().avx2;
}

bool use_avx2_fma() {
    return !g_force_scalar.load(std::memory_order_relaxed) && cpu_features().fma;
}

void set_force_scalar(bool force) {
    g_force_scalar.store(force, std::memory_order_relaxed);
}

std::string simd_info() {
    if (use_avx2_fma()) {
        return "AVX2+FMA";
    }
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
        // MINPS and MAXPS return their second operand for ties and NaNs.
        // Put the sample second to match std::clamp for NaNs and signed zero.
        v = _mm256_min_ps(v_limit, _mm256_max_ps(v_zero, v));
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
        const int64_t left = 2 * x;
        const int32_t tot = static_cast<int32_t>(r0[left]) + static_cast<int32_t>(r0[left + 1])
                          + static_cast<int32_t>(r1[left]) + static_cast<int32_t>(r1[left + 1]);
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
            const __m256i low = _mm256_and_si256(res, _mm256_set1_epi32(0xFFFF));
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(low, low), 0xD8);
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
            const __m256i low = _mm256_and_si256(res, _mm256_set1_epi32(0xFFFF));
            const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(low, low), 0xD8);
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
            const __m256 v = _mm256_min_ps(_mm256_max_ps(raw, v_zero), v_255);
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
                const __m256 v = _mm256_min_ps(_mm256_max_ps(raw, v_zero), v_255);
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
        const __m256 rounded = _mm256_round_ps(_mm256_mul_ps(v, v_65535),
            _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
        _mm256_storeu_ps(dr + i, _mm256_mul_ps(rounded, v_inv257));
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
    const __m256 zero = _mm256_setzero_ps();
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
        // The scalar odd sum starts with +0 and the even sum ends with +0.
        const __m256 first = _mm256_mul_ps(p0, vw_a);
        __m256 v = _mm256_add_ps(
            _mm256_add_ps(_mm256_blend_ps(first, _mm256_add_ps(zero, first), 0xF0),
                          _mm256_mul_ps(p1, vw_b)),
            _mm256_mul_ps(p2, vw_c));
        v = _mm256_add_ps(v, _mm256_mul_ps(p3, vw_d));
        _mm256_storeu_ps(tmp, _mm256_blend_ps(v, _mm256_add_ps(v, zero), 0x0F));
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

namespace {

constexpr uint32_t kAlignDownW0 = 12u;
constexpr uint32_t kAlignDownW1 = 64u;
constexpr uint32_t kAlignDownW2 = 37u;
constexpr uint32_t kAlignDownW3 = 2u;
// Exact unsigned division magic constants verified over full input ranges:
// sum / 3 == (sum * 0xAAAAAAABu) >> 33 for 0..196605
// (v + 57) / 115 == ((v + 57) * 0x0239E0D6u) >> 32 for 0..7536582
constexpr uint32_t kDiv3Magic = 0xAAAAAAABu;
constexpr uint32_t kDiv115Magic = 0x0239E0D6u;
constexpr float kAlignClampEdge = 1.0010000467300415f;

inline __m256i div3_epu32_avx2(__m256i sum) {
    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(kDiv3Magic));
    const __m256i q_even = _mm256_srli_epi64(_mm256_mul_epu32(sum, vm), 33);
    const __m256i q_odd = _mm256_srli_epi32(
        _mm256_mul_epu32(_mm256_srli_epi64(sum, 32), vm), 1);
    return _mm256_blend_epi32(q_even, q_odd, 0xAA);
}

inline __m256i div115_round_epu32_avx2(__m256i v) {
    const __m256i v57 = _mm256_add_epi32(v, _mm256_set1_epi32(57));
    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(kDiv115Magic));
    const __m256i q_even = _mm256_srli_epi64(_mm256_mul_epu32(v57, vm), 32);
    const __m256i q_odd = _mm256_mul_epu32(_mm256_srli_epi64(v57, 32), vm);
    return _mm256_blend_epi32(q_even, q_odd, 0xAA);
}

inline void store_u32x8_as_u16(__m256i v, uint16_t* dst) {
    const __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi32(v, v), 0xD8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_castsi256_si128(p));
}

template <typename T, int Channels, bool BgrRev>
inline void warp_chunk_8(
    const T* img, int64_t img_s0, int64_t src_y0,
    T* out_row, int64_t ox,
    const int32_t* ix, const int32_t* jx, const int32_t* iy, const int32_t* jy,
    const uint32_t* fx, const uint32_t* fy, int inv_mask
) {
    for (int k = 0; k < 8; ++k) {
        T* dp = out_row + (ox + k) * Channels;
        if ((inv_mask >> k) & 1) {
            for (int c = 0; c < Channels; ++c) {
                dp[c] = 0;
            }
            continue;
        }
        const uint32_t fxx = fx[k];
        const uint32_t gxx = 256u - fxx;
        const uint32_t fyy = fy[k];
        const uint32_t gyy = 256u - fyy;
        const T* r0 = img + (static_cast<int64_t>(iy[k]) - src_y0) * img_s0;
        const T* r1 = img + (static_cast<int64_t>(jy[k]) - src_y0) * img_s0;
        const int64_t ixx = static_cast<int64_t>(ix[k]) * Channels;
        const int64_t jxx = static_cast<int64_t>(jx[k]) * Channels;
        if constexpr (Channels == 1) {
            const uint32_t top = static_cast<uint32_t>(r0[ixx]) * gxx + static_cast<uint32_t>(r0[jxx]) * fxx;
            const uint32_t bot = static_cast<uint32_t>(r1[ixx]) * gxx + static_cast<uint32_t>(r1[jxx]) * fxx;
            dp[0] = static_cast<T>((top * gyy + bot * fyy) >> 16u);
        } else {
            for (int c = 0; c < 3; ++c) {
                const int sc = BgrRev ? (2 - c) : c;
                const uint32_t top = static_cast<uint32_t>(r0[ixx + sc]) * gxx
                                   + static_cast<uint32_t>(r0[jxx + sc]) * fxx;
                const uint32_t bot = static_cast<uint32_t>(r1[ixx + sc]) * gxx
                                   + static_cast<uint32_t>(r1[jxx + sc]) * fxx;
                dp[c] = static_cast<T>((top * gyy + bot * fyy) >> 16u);
            }
        }
    }
}

}  // namespace

void align_valid_mask_row_avx2(
    uint8_t* mr, int64_t w,
    float cx, float p08, float p18, float tx, float ty, float v_dy, float u_dy,
    float wf, float hf
) {
    const __m256 v_cx = _mm256_set1_ps(cx);
    const __m256 v_p08 = _mm256_set1_ps(p08);
    const __m256 v_p18 = _mm256_set1_ps(p18);
    const __m256 v_tx = _mm256_set1_ps(tx);
    const __m256 v_ty = _mm256_set1_ps(ty);
    const __m256 v_vdy = _mm256_set1_ps(v_dy);
    const __m256 v_udy = _mm256_set1_ps(u_dy);
    const __m256 v_wf = _mm256_set1_ps(wf);
    const __m256 v_hf = _mm256_set1_ps(hf);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_step = _mm256_set1_ps(8.0f);
    const __m256i v_one = _mm256_set1_epi32(1);
    __m256 vx = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);

    int64_t x = 0;
    for (; x + 8 <= w; x += 8) {
        const __m256 dx = _mm256_sub_ps(vx, v_cx);
        const __m256 v = _mm256_add_ps(_mm256_sub_ps(v_vdy, _mm256_mul_ps(dx, v_p08)), v_ty);
        const __m256 u = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, v_p18), v_udy), v_tx);
        const __m256 ok_u = _mm256_and_ps(
            _mm256_cmp_ps(u, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(u, v_wf, _CMP_LT_OQ));
        const __m256 ok_v = _mm256_and_ps(
            _mm256_cmp_ps(v, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(v, v_hf, _CMP_LT_OQ));
        const __m256i bits = _mm256_and_si256(
            _mm256_castps_si256(_mm256_and_ps(ok_u, ok_v)), v_one);
        const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(bits, bits), 0xD8);
        const __m128i lo16 = _mm256_castsi256_si128(p16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(mr + x), _mm_packus_epi16(lo16, lo16));
        vx = _mm256_add_ps(vx, v_step);
    }
    for (; x < w; ++x) {
        const float dx_ = static_cast<float>(x) - cx;
        const float v = (v_dy - (dx_ * p08)) + ty;
        const float u = ((dx_ * p18) + u_dy) + tx;
        mr[x] = (u >= 0.0f && u < wf && v >= 0.0f && v < hf) ? 1 : 0;
    }
}

void align_gray16_rgb_row_avx2(
    const void* src, bool is_u8, uint16_t* dst, int64_t w, int64_t src_px_stride
) {
    int64_t x = 0;
    if (is_u8) {
        const uint8_t* sr = static_cast<const uint8_t*>(src);
        if (src_px_stride == 4) {
            const __m256i m8 = _mm256_set1_epi32(0xFF);
            for (; x + 8 <= w; x += 8) {
                const __m256i px = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sr + x * 4));
                const __m256i r = _mm256_and_si256(px, m8);
                const __m256i g = _mm256_and_si256(_mm256_srli_epi32(px, 8), m8);
                const __m256i b = _mm256_and_si256(_mm256_srli_epi32(px, 16), m8);
                const __m256i q = div3_epu32_avx2(_mm256_add_epi32(_mm256_add_epi32(r, g), b));
                store_u32x8_as_u16(_mm256_slli_epi32(q, 8), dst + x);
            }
        } else {
            const __m128i shuf_r = _mm_setr_epi8(0, -1, -1, -1, 3, -1, -1, -1, 6, -1, -1, -1, 9, -1, -1, -1);
            const __m128i shuf_g = _mm_setr_epi8(1, -1, -1, -1, 4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1);
            const __m128i shuf_b = _mm_setr_epi8(2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1);
            for (; x + 10 <= w; x += 8) {
                const uint8_t* p = sr + x * 3;
                const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
                const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 12));
                const __m256i vr = _mm256_set_m128i(_mm_shuffle_epi8(b, shuf_r), _mm_shuffle_epi8(a, shuf_r));
                const __m256i vg = _mm256_set_m128i(_mm_shuffle_epi8(b, shuf_g), _mm_shuffle_epi8(a, shuf_g));
                const __m256i vb = _mm256_set_m128i(_mm_shuffle_epi8(b, shuf_b), _mm_shuffle_epi8(a, shuf_b));
                const __m256i q = div3_epu32_avx2(_mm256_add_epi32(_mm256_add_epi32(vr, vg), vb));
                store_u32x8_as_u16(_mm256_slli_epi32(q, 8), dst + x);
            }
        }
        for (; x < w; ++x) {
            const uint8_t* p = sr + x * src_px_stride;
            const uint32_t sum = static_cast<uint32_t>(p[0]) + p[1] + p[2];
            dst[x] = static_cast<uint16_t>((sum / 3u) << 8u);
        }
        return;
    }

    const uint16_t* sr = static_cast<const uint16_t*>(src);
    if (src_px_stride == 4) {
        const __m256i m16 = _mm256_set1_epi32(0xFFFF);
        const __m128i shuf_b = _mm_setr_epi8(
            4, 5, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        for (; x + 4 <= w; x += 4) {
            const uint16_t* p = sr + x * 4;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            const __m128i lo = _mm256_castsi256_si128(raw);
            const __m128i hi = _mm256_extracti128_si256(raw, 1);
            // Deinterleave 4 RGBA u16 pixels in lower 128 bits via 64-bit chunks
            const __m128i rg = _mm_castps_si128(_mm_shuffle_ps(
                _mm_castsi128_ps(lo), _mm_castsi128_ps(hi), 0x88));
            const __m128i ba = _mm_castps_si128(_mm_shuffle_ps(
                _mm_castsi128_ps(lo), _mm_castsi128_ps(hi), 0xDD));
            const __m128i r = _mm_and_si128(rg, _mm256_castsi256_si128(m16));
            const __m128i g = _mm_srli_epi32(rg, 16);
            const __m128i b = _mm_and_si128(ba, _mm256_castsi256_si128(m16));
            const __m128i sum = _mm_add_epi32(_mm_add_epi32(r, g), b);
            const __m128i vm = _mm_set1_epi32(static_cast<int32_t>(kDiv3Magic));
            const __m128i qe = _mm_srli_epi64(_mm_mul_epu32(sum, vm), 33);
            const __m128i qo = _mm_srli_epi32(_mm_mul_epu32(_mm_srli_epi64(sum, 32), vm), 1);
            const __m128i q = _mm_blend_epi32(qe, qo, 0xA);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), _mm_packus_epi32(q, q));
        }
        (void)shuf_b;
    } else {
        const __m128i z = _mm_setzero_si128();
        const __m128i sr0 = _mm_setr_epi8(0, 1, -1, -1, 6, 7, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1);
        const __m128i sg0 = _mm_setr_epi8(2, 3, -1, -1, 8, 9, -1, -1, 14, 15, -1, -1, -1, -1, -1, -1);
        const __m128i sb0 = _mm_setr_epi8(4, 5, -1, -1, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i px3_r = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, -1, -1);
        const __m128i px3_g = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, -1, -1);
        const __m128i px23_b = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 6, 7, -1, -1);
        const __m128i vm = _mm_set1_epi32(static_cast<int32_t>(kDiv3Magic));
        (void)z;
        for (; x + 4 <= w; x += 4) {
            const uint16_t* p = sr + x * 3;
            const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
            const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 8));
            const __m128i vr = _mm_or_si128(_mm_shuffle_epi8(a, sr0), _mm_shuffle_epi8(b, px3_r));
            const __m128i vg = _mm_or_si128(_mm_shuffle_epi8(a, sg0), _mm_shuffle_epi8(b, px3_g));
            const __m128i vb = _mm_or_si128(_mm_shuffle_epi8(a, sb0), _mm_shuffle_epi8(b, px23_b));
            const __m128i sum = _mm_add_epi32(_mm_add_epi32(vr, vg), vb);
            const __m128i qe = _mm_srli_epi64(_mm_mul_epu32(sum, vm), 33);
            const __m128i qo = _mm_srli_epi32(_mm_mul_epu32(_mm_srli_epi64(sum, 32), vm), 1);
            const __m128i q = _mm_blend_epi32(qe, qo, 0xA);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), _mm_packus_epi32(q, q));
        }
    }
    for (; x < w; ++x) {
        const uint16_t* p = sr + x * src_px_stride;
        const uint32_t sum = static_cast<uint32_t>(p[0]) + p[1] + p[2];
        dst[x] = static_cast<uint16_t>(sum / 3u);
    }
}

void align_down_half_vert_row_avx2(
    const uint16_t* r0, const uint16_t* r1, const uint16_t* r2, const uint16_t* r3,
    uint32_t* scratch, int64_t w
) {
    const __m256i vw0 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW0));
    const __m256i vw1 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW1));
    const __m256i vw2 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW2));
    const __m256i vw3 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW3));
    int64_t x = 0;
    for (; x + 8 <= w; x += 8) {
        const __m256i u0 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r0 + x)));
        const __m256i u1 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r1 + x)));
        const __m256i u2 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r2 + x)));
        const __m256i u3 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r3 + x)));
        const __m256i t0 = _mm256_mullo_epi32(u0, vw0);
        const __m256i t1 = _mm256_mullo_epi32(u1, vw1);
        const __m256i t2 = _mm256_mullo_epi32(u2, vw2);
        const __m256i t3 = _mm256_mullo_epi32(u3, vw3);
        const __m256i sum = _mm256_add_epi32(_mm256_add_epi32(t0, t1), _mm256_add_epi32(t2, t3));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(scratch + x), div115_round_epu32_avx2(sum));
    }
    for (; x < w; ++x) {
        const uint32_t v = static_cast<uint32_t>(r0[x]) * kAlignDownW0
                         + static_cast<uint32_t>(r1[x]) * kAlignDownW1
                         + static_cast<uint32_t>(r2[x]) * kAlignDownW2
                         + static_cast<uint32_t>(r3[x]) * kAlignDownW3;
        scratch[x] = (v + 57u) / 115u;
    }
}

void align_down_half_horiz_row_avx2(
    const uint32_t* scratch, uint16_t* out, int64_t ow, int64_t w
) {
    if (ow <= 0) {
        return;
    }
    auto clamp_idx = [w](int64_t idx) -> int64_t {
        return idx < 0 ? 0 : (idx >= w ? w - 1 : idx);
    };
    {
        const uint32_t v = scratch[0] * kAlignDownW0
                         + scratch[0] * kAlignDownW1
                         + scratch[clamp_idx(1)] * kAlignDownW2
                         + scratch[clamp_idx(2)] * kAlignDownW3;
        out[0] = static_cast<uint16_t>((v + 57u) / 115u);
    }
    const __m256i vw0 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW0));
    const __m256i vw1 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW1));
    const __m256i vw2 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW2));
    const __m256i vw3 = _mm256_set1_epi32(static_cast<int32_t>(kAlignDownW3));
    int64_t ox = 1;
    for (; ox + 8 <= ow && (2 * ox + 16 < w); ox += 8) {
        const uint32_t* sp = scratch + (2 * ox - 1);
        const __m256 v0 = _mm256_loadu_ps(reinterpret_cast<const float*>(sp));
        const __m256 v1 = _mm256_loadu_ps(reinterpret_cast<const float*>(sp + 8));
        const __m256 v2 = _mm256_loadu_ps(reinterpret_cast<const float*>(sp + 2));
        const __m256 v3 = _mm256_loadu_ps(reinterpret_cast<const float*>(sp + 10));
        const __m256i sx0 = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, 0x88)), 0xD8);
        const __m256i sx1 = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v0, v1, 0xDD)), 0xD8);
        const __m256i sx2 = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v2, v3, 0x88)), 0xD8);
        const __m256i sx3 = _mm256_permute4x64_epi64(
            _mm256_castps_si256(_mm256_shuffle_ps(v2, v3, 0xDD)), 0xD8);
        const __m256i sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(sx0, vw0), _mm256_mullo_epi32(sx1, vw1)),
            _mm256_add_epi32(_mm256_mullo_epi32(sx2, vw2), _mm256_mullo_epi32(sx3, vw3)));
        store_u32x8_as_u16(div115_round_epu32_avx2(sum), out + ox);
    }
    for (; ox < ow; ++ox) {
        const uint32_t v = scratch[clamp_idx(2 * ox - 1)] * kAlignDownW0
                         + scratch[clamp_idx(2 * ox)] * kAlignDownW1
                         + scratch[clamp_idx(2 * ox + 1)] * kAlignDownW2
                         + scratch[clamp_idx(2 * ox + 2)] * kAlignDownW3;
        out[ox] = static_cast<uint16_t>((v + 57u) / 115u);
    }
}

void align_fast_downsize_u16_row_avx2(
    const uint16_t* r0, const uint16_t* r1, uint16_t* dst, int64_t ow, int64_t w
) {
    const __m256i m16 = _mm256_set1_epi32(0xFFFF);
    const __m256i v2 = _mm256_set1_epi32(2);
    int64_t ox = 0;
    for (; ox + 8 <= ow && (2 * ox + 16 <= w); ox += 8) {
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * ox));
        const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 2 * ox));
        const __m256i s0 = _mm256_add_epi32(_mm256_and_si256(p0, m16), _mm256_srli_epi32(p0, 16));
        const __m256i s1 = _mm256_add_epi32(_mm256_and_si256(p1, m16), _mm256_srli_epi32(p1, 16));
        const __m256i avg = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(s0, s1), v2), 2);
        store_u32x8_as_u16(avg, dst + ox);
    }
    for (; ox < ow; ++ox) {
        const int64_t x0 = 2 * ox;
        const int64_t x1 = (x0 + 1 < w) ? (x0 + 1) : (w - 1);
        dst[ox] = static_cast<uint16_t>(
            (static_cast<uint32_t>(r0[x0]) + r0[x1] + r1[x0] + r1[x1] + 2u) >> 2u);
    }
}

void align_warp_alignment_row_avx2(
    const void* img_data, int64_t img_s0, int64_t img_s1, bool is_u8,
    void* out_row, int64_t out_s1,
    int64_t src_h, int64_t src_w, int64_t src_y0, int64_t x_off, int64_t rw,
    int64_t channels, bool bgr_rev,
    float cx, float p08, float p18, float tx, float ty, float v_dy, float u_dy
) {
    (void)img_s1;
    (void)out_s1;
    const float wf = static_cast<float>(static_cast<int32_t>(src_w));
    const float hf = static_cast<float>(static_cast<int32_t>(src_h));
    const float w_m1 = static_cast<float>(static_cast<int32_t>(src_w) - 1);
    const float h_m1 = static_cast<float>(static_cast<int32_t>(src_h) - 1);
    const float u_clamp = wf - kAlignClampEdge;
    const float v_clamp = hf - kAlignClampEdge;

    const __m256 v_cx = _mm256_set1_ps(cx);
    const __m256 v_p08 = _mm256_set1_ps(p08);
    const __m256 v_p18 = _mm256_set1_ps(p18);
    const __m256 v_tx = _mm256_set1_ps(tx);
    const __m256 v_ty = _mm256_set1_ps(ty);
    const __m256 v_vdy = _mm256_set1_ps(v_dy);
    const __m256 v_udy = _mm256_set1_ps(u_dy);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_wf = _mm256_set1_ps(wf);
    const __m256 v_hf = _mm256_set1_ps(hf);
    const __m256 v_wm1 = _mm256_set1_ps(w_m1);
    const __m256 v_hm1 = _mm256_set1_ps(h_m1);
    const __m256 v_uc = _mm256_set1_ps(u_clamp);
    const __m256 v_vc = _mm256_set1_ps(v_clamp);
    const __m256 v_256 = _mm256_set1_ps(256.0f);
    const __m256 v_step = _mm256_set1_ps(8.0f);
    const __m256i vi_zero = _mm256_setzero_si256();
    const __m256i vi_one = _mm256_set1_epi32(1);
    const __m256i vi_wm1 = _mm256_set1_epi32(static_cast<int32_t>(src_w - 1));
    const __m256i vi_hm1 = _mm256_set1_epi32(static_cast<int32_t>(src_h - 1));

    const float x0_f = static_cast<float>(x_off);
    __m256 vx = _mm256_add_ps(
        _mm256_set1_ps(x0_f),
        _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f));

    alignas(32) int32_t ix_buf[8], jx_buf[8], iy_buf[8], jy_buf[8];
    alignas(32) uint32_t fx_buf[8], fy_buf[8];

    int64_t ox = 0;
    for (; ox + 8 <= rw; ox += 8) {
        const __m256 dx = _mm256_sub_ps(vx, v_cx);
        const __m256 v = _mm256_add_ps(_mm256_sub_ps(v_vdy, _mm256_mul_ps(dx, v_p08)), v_ty);
        const __m256 u = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, v_p18), v_udy), v_tx);

        const __m256 inv_u = _mm256_or_ps(_mm256_cmp_ps(u, v_wf, _CMP_GE_OQ), _mm256_cmp_ps(u, v_zero, _CMP_LT_OQ));
        const __m256 inv_v = _mm256_or_ps(_mm256_cmp_ps(v, v_hf, _CMP_GE_OQ), _mm256_cmp_ps(v, v_zero, _CMP_LT_OQ));
        const int inv_mask = _mm256_movemask_ps(_mm256_or_ps(inv_u, inv_v));

        vx = _mm256_add_ps(vx, v_step);
        if (inv_mask == 0xFF) {
            if (is_u8) {
                std::memset(static_cast<uint8_t*>(out_row) + ox * channels, 0, static_cast<size_t>(8 * channels));
            } else {
                std::memset(static_cast<uint16_t*>(out_row) + ox * channels, 0, static_cast<size_t>(8 * channels) * 2u);
            }
            continue;
        }

        __m256 uc = _mm256_blendv_ps(u, v_uc, _mm256_cmp_ps(u, v_wm1, _CMP_GE_OQ));
        uc = _mm256_blendv_ps(uc, v_zero, _mm256_cmp_ps(uc, v_zero, _CMP_LT_OQ));
        const __m256i uidx = _mm256_cvttps_epi32(uc);
        const __m256i ufx = _mm256_cvttps_epi32(
            _mm256_mul_ps(_mm256_sub_ps(uc, _mm256_cvtepi32_ps(uidx)), v_256));
        const __m256i uix = _mm256_min_epi32(_mm256_max_epi32(uidx, vi_zero), vi_wm1);
        const __m256i ujx = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(uidx, vi_one), vi_zero), vi_wm1);

        __m256 vc = _mm256_blendv_ps(v, v_vc, _mm256_cmp_ps(v, v_hm1, _CMP_GE_OQ));
        vc = _mm256_blendv_ps(vc, v_zero, _mm256_cmp_ps(vc, v_zero, _CMP_LT_OQ));
        const __m256i vidx = _mm256_cvttps_epi32(vc);
        const __m256i vfy = _mm256_cvttps_epi32(
            _mm256_mul_ps(_mm256_sub_ps(vc, _mm256_cvtepi32_ps(vidx)), v_256));
        const __m256i viy = _mm256_min_epi32(_mm256_max_epi32(vidx, vi_zero), vi_hm1);
        const __m256i vjy = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vidx, vi_one), vi_zero), vi_hm1);

        _mm256_store_si256(reinterpret_cast<__m256i*>(ix_buf), uix);
        _mm256_store_si256(reinterpret_cast<__m256i*>(jx_buf), ujx);
        _mm256_store_si256(reinterpret_cast<__m256i*>(iy_buf), viy);
        _mm256_store_si256(reinterpret_cast<__m256i*>(jy_buf), vjy);
        _mm256_store_si256(reinterpret_cast<__m256i*>(fx_buf), ufx);
        _mm256_store_si256(reinterpret_cast<__m256i*>(fy_buf), vfy);

        if (is_u8) {
            const auto* src_t = static_cast<const uint8_t*>(img_data);
            auto* dst_t = static_cast<uint8_t*>(out_row);
            if (channels == 1) {
                warp_chunk_8<uint8_t, 1, false>(src_t, img_s0, src_y0, dst_t, ox, ix_buf, jx_buf, iy_buf, jy_buf, fx_buf, fy_buf, inv_mask);
            } else if (bgr_rev) {
                warp_chunk_8<uint8_t, 3, true>(src_t, img_s0, src_y0, dst_t, ox, ix_buf, jx_buf, iy_buf, jy_buf, fx_buf, fy_buf, inv_mask);
            } else {
                warp_chunk_8<uint8_t, 3, false>(src_t, img_s0, src_y0, dst_t, ox, ix_buf, jx_buf, iy_buf, jy_buf, fx_buf, fy_buf, inv_mask);
            }
        } else {
            const auto* src_t = static_cast<const uint16_t*>(img_data);
            auto* dst_t = static_cast<uint16_t*>(out_row);
            if (channels == 1) {
                warp_chunk_8<uint16_t, 1, false>(src_t, img_s0, src_y0, dst_t, ox, ix_buf, jx_buf, iy_buf, jy_buf, fx_buf, fy_buf, inv_mask);
            } else if (bgr_rev) {
                warp_chunk_8<uint16_t, 3, true>(src_t, img_s0, src_y0, dst_t, ox, ix_buf, jx_buf, iy_buf, jy_buf, fx_buf, fy_buf, inv_mask);
            } else {
                warp_chunk_8<uint16_t, 3, false>(src_t, img_s0, src_y0, dst_t, ox, ix_buf, jx_buf, iy_buf, jy_buf, fx_buf, fy_buf, inv_mask);
            }
        }
    }

    auto clamp_i32 = [](int32_t v, int32_t hi) -> int32_t {
        return v < 0 ? 0 : (v > hi ? hi : v);
    };
    const int32_t len_x = static_cast<int32_t>(src_w);
    const int32_t len_y = static_cast<int32_t>(src_h);
    for (; ox < rw; ++ox) {
        const float dx_ = static_cast<float>(x_off + ox) - cx;
        const float v = (v_dy - (dx_ * p08)) + ty;
        const float u = ((dx_ * p18) + u_dy) + tx;
        const bool invalid = (u >= wf || u < 0.0f || v >= hf || v < 0.0f);
        if (invalid) {
            for (int64_t c = 0; c < channels; ++c) {
                if (is_u8) {
                    static_cast<uint8_t*>(out_row)[ox * channels + c] = 0;
                } else {
                    static_cast<uint16_t*>(out_row)[ox * channels + c] = 0;
                }
            }
            continue;
        }
        float uc = (u >= w_m1) ? u_clamp : u;
        if (uc < 0.0f) uc = 0.0f;
        const int32_t uidx = static_cast<int32_t>(uc);
        const uint32_t fxx = static_cast<uint32_t>((uc - static_cast<float>(uidx)) * 256.0f);
        const uint32_t gxx = 256u - fxx;
        const int64_t ixx = static_cast<int64_t>(clamp_i32(uidx, len_x - 1)) * channels;
        const int64_t jxx = static_cast<int64_t>(clamp_i32(uidx + 1, len_x - 1)) * channels;

        float vc = (v >= h_m1) ? v_clamp : v;
        if (vc < 0.0f) vc = 0.0f;
        const int32_t vidx = static_cast<int32_t>(vc);
        const uint32_t fyy = static_cast<uint32_t>((vc - static_cast<float>(vidx)) * 256.0f);
        const uint32_t gyy = 256u - fyy;
        const int64_t iy_b = static_cast<int64_t>(clamp_i32(vidx, len_y - 1)) - src_y0;
        const int64_t jy_b = static_cast<int64_t>(clamp_i32(vidx + 1, len_y - 1)) - src_y0;

        for (int64_t c = 0; c < channels; ++c) {
            const int64_t sc = (bgr_rev && channels == 3) ? (2 - c) : c;
            if (is_u8) {
                const uint8_t* src_t = static_cast<const uint8_t*>(img_data);
                const uint8_t* r0 = src_t + iy_b * img_s0;
                const uint8_t* r1 = src_t + jy_b * img_s0;
                const uint32_t top = static_cast<uint32_t>(r0[ixx + sc]) * gxx + static_cast<uint32_t>(r0[jxx + sc]) * fxx;
                const uint32_t bot = static_cast<uint32_t>(r1[ixx + sc]) * gxx + static_cast<uint32_t>(r1[jxx + sc]) * fxx;
                static_cast<uint8_t*>(out_row)[ox * channels + c] = static_cast<uint8_t>((top * gyy + bot * fyy) >> 16u);
            } else {
                const uint16_t* src_t = static_cast<const uint16_t*>(img_data);
                const uint16_t* r0 = src_t + iy_b * img_s0;
                const uint16_t* r1 = src_t + jy_b * img_s0;
                const uint32_t top = static_cast<uint32_t>(r0[ixx + sc]) * gxx + static_cast<uint32_t>(r0[jxx + sc]) * fxx;
                const uint32_t bot = static_cast<uint32_t>(r1[ixx + sc]) * gxx + static_cast<uint32_t>(r1[jxx + sc]) * fxx;
                static_cast<uint16_t*>(out_row)[ox * channels + c] = static_cast<uint16_t>((top * gyy + bot * fyy) >> 16u);
            }
        }
    }
}

namespace {

inline void score_block_packed_avx2(
    const uint16_t* rb, int64_t ref_sy,
    const uint16_t* bb, int64_t b_sy, int64_t h, int64_t w,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    float tx_k, float ty_k, float cx_k, float cy_k,
    float p08_k, float p0c_k, float p18_k, float p1c_k,
    int64_t& out_sum, int64_t& out_sq, int64_t& out_cnt
) {
    const int32_t len_x = static_cast<int32_t>(w);
    const int32_t len_y = static_cast<int32_t>(h);
    const float wf = static_cast<float>(len_x);
    const float hf = static_cast<float>(len_y);
    const float w_m1 = static_cast<float>(len_x - 1);
    const float h_m1 = static_cast<float>(len_y - 1);
    const float u_clamp = wf - kAlignClampEdge;
    const float v_clamp = hf - kAlignClampEdge;

    const __m256 v_cx = _mm256_set1_ps(cx_k);
    const __m256 v_p08 = _mm256_set1_ps(p08_k);
    const __m256 v_p18 = _mm256_set1_ps(p18_k);
    const __m256 v_tx = _mm256_set1_ps(tx_k);
    const __m256 v_ty = _mm256_set1_ps(ty_k);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_wf = _mm256_set1_ps(wf);
    const __m256 v_hf = _mm256_set1_ps(hf);
    const __m256 v_wm1 = _mm256_set1_ps(w_m1);
    const __m256 v_hm1 = _mm256_set1_ps(h_m1);
    const __m256 v_uc = _mm256_set1_ps(u_clamp);
    const __m256 v_vc = _mm256_set1_ps(v_clamp);
    const __m256 v_256 = _mm256_set1_ps(256.0f);
    const __m256i vi_zero = _mm256_setzero_si256();
    const __m256i vi_one = _mm256_set1_epi32(1);
    const __m256i vi_256 = _mm256_set1_epi32(256);
    const __m256i vi_wm1 = _mm256_set1_epi32(len_x - 1);
    const __m256i vi_hm1 = _mm256_set1_epi32(len_y - 1);

    __m256i v_sum_lo = _mm256_setzero_si256();
    __m256i v_sum_hi = _mm256_setzero_si256();
    __m256i v_sq_lo = _mm256_setzero_si256();
    __m256i v_sq_hi = _mm256_setzero_si256();
    int64_t scalar_sum = 0;
    int64_t scalar_sq = 0;
    int64_t acc_cnt = 0;

    alignas(32) int32_t ix_buf[8], jx_buf[8], iy_buf[8], jy_buf[8];
    alignas(32) int32_t fx_buf[8], fy_buf[8], a0_buf[8], a1_buf[8], b0_buf[8], b1_buf[8];

    auto clamp_i32 = [](int32_t v, int32_t hi) -> int32_t {
        return v < 0 ? 0 : (v > hi ? hi : v);
    };

    for (int64_t iy_i = 0; iy_i < nyy; ++iy_i) {
        const float dy_ = ys[iy_i] - cy_k;
        const float v_dy = dy_ * p1c_k;
        const float u_dy = dy_ * p0c_k;
        const __m256 v_vdy = _mm256_set1_ps(v_dy);
        const __m256 v_udy = _mm256_set1_ps(u_dy);
        const uint16_t* rrow = rb + iy_i * ref_sy;

        int64_t ix_i = 0;
        for (; ix_i + 8 <= nxx; ix_i += 8) {
            const __m256 dx_ = _mm256_sub_ps(_mm256_loadu_ps(xs + ix_i), v_cx);
            const __m256 v = _mm256_add_ps(_mm256_sub_ps(v_vdy, _mm256_mul_ps(dx_, v_p08)), v_ty);
            const __m256 u = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx_, v_p18), v_udy), v_tx);

            const __m256 ok_u = _mm256_and_ps(
                _mm256_cmp_ps(u, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(u, v_wf, _CMP_LT_OQ));
            const __m256 ok_v = _mm256_and_ps(
                _mm256_cmp_ps(v, v_zero, _CMP_GE_OQ), _mm256_cmp_ps(v, v_hf, _CMP_LT_OQ));
            const int valid_mask = _mm256_movemask_ps(_mm256_and_ps(ok_u, ok_v));
            if (valid_mask == 0) {
                continue;
            }

            __m256 uc = _mm256_blendv_ps(u, v_uc, _mm256_cmp_ps(u, v_wm1, _CMP_GE_OQ));
            uc = _mm256_blendv_ps(uc, v_zero, _mm256_cmp_ps(uc, v_zero, _CMP_LT_OQ));
            const __m256i uidx = _mm256_cvttps_epi32(uc);
            const __m256i vfx = _mm256_cvttps_epi32(
                _mm256_mul_ps(_mm256_sub_ps(uc, _mm256_cvtepi32_ps(uidx)), v_256));
            const __m256i vix = _mm256_min_epi32(_mm256_max_epi32(uidx, vi_zero), vi_wm1);
            const __m256i vjx = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(uidx, vi_one), vi_zero), vi_wm1);

            __m256 vc = _mm256_blendv_ps(v, v_vc, _mm256_cmp_ps(v, v_hm1, _CMP_GE_OQ));
            vc = _mm256_blendv_ps(vc, v_zero, _mm256_cmp_ps(vc, v_zero, _CMP_LT_OQ));
            const __m256i vidx = _mm256_cvttps_epi32(vc);
            const __m256i vfy = _mm256_cvttps_epi32(
                _mm256_mul_ps(_mm256_sub_ps(vc, _mm256_cvtepi32_ps(vidx)), v_256));
            const __m256i viy = _mm256_min_epi32(_mm256_max_epi32(vidx, vi_zero), vi_hm1);
            const __m256i vjy = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(vidx, vi_one), vi_zero), vi_hm1);

            _mm256_store_si256(reinterpret_cast<__m256i*>(ix_buf), vix);
            _mm256_store_si256(reinterpret_cast<__m256i*>(jx_buf), vjx);
            _mm256_store_si256(reinterpret_cast<__m256i*>(iy_buf), viy);
            _mm256_store_si256(reinterpret_cast<__m256i*>(jy_buf), vjy);

            if (valid_mask == 0xFF) {
                for (int m = 0; m < 8; ++m) {
                    const uint16_t* br0 = bb + static_cast<int64_t>(iy_buf[m]) * b_sy;
                    const uint16_t* br1 = bb + static_cast<int64_t>(jy_buf[m]) * b_sy;
                    a0_buf[m] = static_cast<int32_t>(br0[ix_buf[m]]);
                    a1_buf[m] = static_cast<int32_t>(br0[jx_buf[m]]);
                    b0_buf[m] = static_cast<int32_t>(br1[ix_buf[m]]);
                    b1_buf[m] = static_cast<int32_t>(br1[jx_buf[m]]);
                }
                const __m256i vgx = _mm256_sub_epi32(vi_256, vfx);
                const __m256i vgy = _mm256_sub_epi32(vi_256, vfy);
                const __m256i top = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(a0_buf)), vgx),
                    _mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(a1_buf)), vfx));
                const __m256i bot = _mm256_add_epi32(
                    _mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(b0_buf)), vgx),
                    _mm256_mullo_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(b1_buf)), vfx));
                const __m256i val = _mm256_srli_epi32(
                    _mm256_add_epi32(_mm256_mullo_epi32(top, vgy), _mm256_mullo_epi32(bot, vfy)), 16);
                const __m256i vref = _mm256_cvtepu16_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(rrow + ix_i)));
                const __m256i diff = _mm256_sub_epi32(vref, val);

                v_sum_lo = _mm256_add_epi64(
                    v_sum_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(diff)));
                v_sum_hi = _mm256_add_epi64(
                    v_sum_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(diff, 1)));

                const __m256i diff_odd = _mm256_srli_epi64(diff, 32);
                v_sq_lo = _mm256_add_epi64(v_sq_lo, _mm256_mul_epi32(diff, diff));
                v_sq_hi = _mm256_add_epi64(v_sq_hi, _mm256_mul_epi32(diff_odd, diff_odd));
                acc_cnt += 8;
            } else {
                _mm256_store_si256(reinterpret_cast<__m256i*>(fx_buf), vfx);
                _mm256_store_si256(reinterpret_cast<__m256i*>(fy_buf), vfy);
                for (int m = 0; m < 8; ++m) {
                    if (!((valid_mask >> m) & 1)) {
                        continue;
                    }
                    const uint32_t fx = static_cast<uint32_t>(fx_buf[m]);
                    const uint32_t fy = static_cast<uint32_t>(fy_buf[m]);
                    const uint32_t gx = 256u - fx;
                    const uint32_t gy = 256u - fy;
                    const uint16_t* br0 = bb + static_cast<int64_t>(iy_buf[m]) * b_sy;
                    const uint16_t* br1 = bb + static_cast<int64_t>(jy_buf[m]) * b_sy;
                    const uint32_t top = static_cast<uint32_t>(br0[ix_buf[m]]) * gx
                                       + static_cast<uint32_t>(br0[jx_buf[m]]) * fx;
                    const uint32_t bot = static_cast<uint32_t>(br1[ix_buf[m]]) * gx
                                       + static_cast<uint32_t>(br1[jx_buf[m]]) * fx;
                    const uint32_t val = (top * gy + bot * fy) >> 16u;
                    const int64_t diff = static_cast<int64_t>(rrow[ix_i + m]) - static_cast<int64_t>(val);
                    scalar_sum += diff;
                    scalar_sq += diff * diff;
                    acc_cnt += 1;
                }
            }
        }

        for (; ix_i < nxx; ++ix_i) {
            const float dx_ = xs[ix_i] - cx_k;
            const float v = (v_dy - (dx_ * p08_k)) + ty_k;
            const float u = ((dx_ * p18_k) + u_dy) + tx_k;
            if (!(u >= 0.0f && u < wf && v >= 0.0f && v < hf)) {
                continue;
            }
            float uc = (u >= w_m1) ? u_clamp : u;
            if (uc < 0.0f) uc = 0.0f;
            const int32_t uidx = static_cast<int32_t>(uc);
            const uint32_t fx = static_cast<uint32_t>((uc - static_cast<float>(uidx)) * 256.0f);
            const int32_t ix = clamp_i32(uidx, len_x - 1);
            const int32_t jx = clamp_i32(uidx + 1, len_x - 1);

            float vc = (v >= h_m1) ? v_clamp : v;
            if (vc < 0.0f) vc = 0.0f;
            const int32_t vidx = static_cast<int32_t>(vc);
            const uint32_t fy = static_cast<uint32_t>((vc - static_cast<float>(vidx)) * 256.0f);
            const int32_t iy = clamp_i32(vidx, len_y - 1);
            const int32_t jy = clamp_i32(vidx + 1, len_y - 1);

            const uint32_t gx = 256u - fx;
            const uint32_t gy = 256u - fy;
            const uint16_t* br0 = bb + static_cast<int64_t>(iy) * b_sy;
            const uint16_t* br1 = bb + static_cast<int64_t>(jy) * b_sy;
            const uint32_t top = static_cast<uint32_t>(br0[ix]) * gx + static_cast<uint32_t>(br0[jx]) * fx;
            const uint32_t bot = static_cast<uint32_t>(br1[ix]) * gx + static_cast<uint32_t>(br1[jx]) * fx;
            const uint32_t val = (top * gy + bot * fy) >> 16u;
            const int64_t diff = static_cast<int64_t>(rrow[ix_i]) - static_cast<int64_t>(val);
            scalar_sum += diff;
            scalar_sq += diff * diff;
            acc_cnt += 1;
        }
    }

    alignas(32) int64_t red_sum[4], red_sq[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(red_sum), _mm256_add_epi64(v_sum_lo, v_sum_hi));
    _mm256_store_si256(reinterpret_cast<__m256i*>(red_sq), _mm256_add_epi64(v_sq_lo, v_sq_hi));
    out_sum = scalar_sum + red_sum[0] + red_sum[1] + red_sum[2] + red_sum[3];
    out_sq = scalar_sq + red_sq[0] + red_sq[1] + red_sq[2] + red_sq[3];
    out_cnt = acc_cnt;
}

}  // namespace

void align_score_chunk_packed_avx2(
    const uint16_t* rb, int64_t ref_sy,
    const uint16_t* bb, int64_t b_sy, int64_t h, int64_t w,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    const float* p08, const float* p0c, const float* p18, const float* p1c,
    const float* tx, const float* ty, const float* cx, const float* cy,
    const int64_t* idx, int64_t k_start, int64_t k_stop,
    int64_t* ssum, int64_t* ssq, int64_t* cnt
) {
    for (int64_t k = k_start; k < k_stop; ++k) {
        const int64_t ci = idx[k];
        int64_t acc_sum = 0, acc_sq = 0, acc_cnt = 0;
        score_block_packed_avx2(
            rb, ref_sy, bb, b_sy, h, w, xs, nxx, ys, nyy,
            tx[ci], ty[ci], cx[ci], cy[ci],
            p08[ci], p0c[ci], p18[ci], p1c[ci],
            acc_sum, acc_sq, acc_cnt);
        ssum[ci] = acc_sum;
        ssq[ci] = acc_sq;
        cnt[ci] = acc_cnt;
    }
}

void align_score_grid_4d_packed_avx2(
    const uint16_t* rb, int64_t ref_sy,
    const uint16_t* bb, int64_t b_sy, int64_t h, int64_t w,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    float tx0, float ty0,
    const float* cx_tab, const float* cy_tab,
    const float* p08b, int64_t p08_sy, int64_t p08_sx,
    const float* p18b, int64_t p18_sy, int64_t p18_sx,
    int64_t n0, int64_t n1, int64_t n2, int64_t n3,
    int64_t k_start, int64_t k_stop,
    int64_t* out_best_k, float* out_best_sc, int64_t slot
) {
    (void)n3;
    const float hf = static_cast<float>(h);
    const float wf = static_cast<float>(w);
    const float x_lo = xs[0], x_hi = xs[nxx - 1];
    const float y_lo = ys[0], y_hi = ys[nyy - 1];
    float best_sc = std::numeric_limits<float>::infinity();
    int64_t best_k = -1;

    for (int64_t k = k_start; k < k_stop; ++k) {
        int64_t rem = k;
        const int64_t t0 = rem % n0;
        rem /= n0;
        const int64_t t1 = rem % n1;
        rem /= n1;
        const int64_t t2 = rem % n2;
        const int64_t t3 = rem / n2;

        const float cx_k = cx_tab[t0];
        const float cy_k = cy_tab[t1];
        const float p08_k = p08b[t3 * p08_sy + t2 * p08_sx];
        const float p18_k = p18b[t3 * p18_sy + t2 * p18_sx];
        const float p0c_k = p08_k;
        const float p1c_k = p18_k;

        const float c_u = (tx0 - (cx_k * p18_k)) - (cy_k * p0c_k);
        const float c_v = (ty0 - (cy_k * p1c_k)) - (cx_k * p08_k);

        const float aux0 = p18_k * x_lo, aux1 = p18_k * x_hi;
        const float buy0 = p0c_k * y_lo, buy1 = p0c_k * y_hi;
        const float u_lo = ((aux0 < aux1 ? aux0 : aux1) + (buy0 < buy1 ? buy0 : buy1)) + c_u;
        const float u_hi = ((aux0 > aux1 ? aux0 : aux1) + (buy0 > buy1 ? buy0 : buy1)) + c_u;

        const float avx0 = p1c_k * x_lo, avx1 = p1c_k * x_hi;
        const float bvy0 = p08_k * y_lo, bvy1 = p08_k * y_hi;
        const float v_lo = ((avx0 < avx1 ? avx0 : avx1) + (bvy0 < bvy1 ? bvy0 : bvy1)) + c_v;
        const float v_hi = ((avx0 > avx1 ? avx0 : avx1) + (bvy0 > bvy1 ? bvy0 : bvy1)) + c_v;

        if (!(u_hi >= 0.0f && u_lo < wf && v_hi >= 0.0f && v_lo < hf)) {
            continue;
        }

        int64_t acc_sum = 0, acc_sq = 0, acc_cnt = 0;
        score_block_packed_avx2(
            rb, ref_sy, bb, b_sy, h, w, xs, nxx, ys, nyy,
            tx0, ty0, cx_k, cy_k,
            p08_k, p0c_k, p18_k, p1c_k,
            acc_sum, acc_sq, acc_cnt);
        if (acc_cnt >= 2) {
            const double denom = static_cast<double>(static_cast<float>(acc_cnt));
            const float msq_f = static_cast<float>(static_cast<double>(acc_sq) / denom);
            const float msu_f = static_cast<float>(static_cast<double>(acc_sum) / denom);
            const float var_f = msq_f - (msu_f * msu_f);
            if (var_f >= 0.0f) {
                const float sc = std::sqrt(var_f);
                if (sc < best_sc) {
                    best_sc = sc;
                    best_k = k;
                }
            }
        }
    }
    out_best_k[slot] = best_k;
    out_best_sc[slot] = best_sc;
}

namespace {

constexpr std::array<uint32_t, 101> make_activity_recip_u32() {
    std::array<uint32_t, 101> arr{};
    for (int i = 0; i < 101; ++i) {
        const uint64_t d = static_cast<uint64_t>(10 + i);
        arr[static_cast<size_t>(i)] = static_cast<uint32_t>(((uint64_t{1} << 32) + d - 1u) / d);
    }
    return arr;
}

constexpr std::array<uint32_t, 101> kActivityRecipU32 = make_activity_recip_u32();

inline __m256i normalized_diff_8_avx2(__m256i v_val, __m256i v_mean) {
    const __m256i vmax = _mm256_max_epu32(v_val, v_mean);
    const __m256i v100 = _mm256_mullo_epi32(vmax, _mm256_set1_epi32(100));
    const __m256i vm = _mm256_set1_epi32(static_cast<int32_t>(0x80008001u));
    const __m256i m_even = _mm256_mul_epu32(v100, vm);
    const __m256i m_odd = _mm256_mul_epu32(_mm256_srli_epi64(v100, 32), vm);
    const __m256i denom = _mm256_blend_epi32(
        _mm256_srli_epi64(m_even, 47), _mm256_srli_epi64(m_odd, 15), 0xAA);
    const __m256i recip = _mm256_i32gather_epi32(
        reinterpret_cast<const int32_t*>(kActivityRecipU32.data()), denom, 4);
    const __m256i vdiff = _mm256_abs_epi32(_mm256_sub_epi32(v_val, v_mean));
    const __m256i vnum = _mm256_mullo_epi32(vdiff, _mm256_set1_epi32(110));
    const __m256i q_even = _mm256_srli_epi64(_mm256_mul_epu32(vnum, recip), 32);
    const __m256i q_odd = _mm256_mul_epu32(
        _mm256_srli_epi64(vnum, 32), _mm256_srli_epi64(recip, 32));
    const __m256i res = _mm256_blend_epi32(q_even, q_odd, 0xAA);
    return _mm256_min_epu32(res, _mm256_set1_epi32(65535));
}

}  // namespace

void resample_u16_vert_acc_row_avx2_fma(
    const uint16_t* sline, float* row_buf, int64_t sw, float wval
) {
    const __m256 vw = _mm256_set1_ps(wval);
    int64_t x = 0;
    for (; x + 8 <= sw; x += 8) {
        const __m256 vs = _mm256_cvtepi32_ps(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sline + x))));
        const __m256 acc = _mm256_loadu_ps(row_buf + x);
        _mm256_storeu_ps(row_buf + x, _mm256_fmadd_ps(vw, vs, acc));
    }
    for (; x < sw; ++x) {
        row_buf[x] = std::fmaf(wval, static_cast<float>(sline[x]), row_buf[x]);
    }
}

void resample_u16_horiz_rep_row_avx2_fma(
    const float* row_buf, float* sums_buf, uint16_t* orow,
    int64_t first, int64_t nrep,
    const int32_t* h_idx_first, const float* h_w_first,
    int64_t h_taps, int64_t hi_sy, int64_t hw_sy
) {
    (void)hi_sy;
    (void)hw_sy;
    std::memset(sums_buf, 0, static_cast<size_t>(nrep) * sizeof(float));
    for (int64_t tap = 0; tap < h_taps; tap += 2) {
        const __m256 v_even = _mm256_set1_ps(h_w_first[tap]);
        const __m256 v_odd = _mm256_set1_ps(h_w_first[tap + 1]);
        const float even_s = h_w_first[tap];
        const float odd_s = h_w_first[tap + 1];
        const float* line = row_buf + h_idx_first[tap];
        int64_t x = 0;
        for (; x + 8 <= nrep; x += 8) {
            const __m256 a = _mm256_loadu_ps(line + 2 * x);
            const __m256 b = _mm256_loadu_ps(line + 2 * x + 8);
            const __m256 even_v = _mm256_castsi256_ps(_mm256_permute4x64_epi64(
                _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0x88)), 0xD8));
            const __m256 odd_v = _mm256_castsi256_ps(_mm256_permute4x64_epi64(
                _mm256_castps_si256(_mm256_shuffle_ps(a, b, 0xDD)), 0xD8));
            const __m256 prev = _mm256_loadu_ps(sums_buf + x);
            const __m256 next = _mm256_fmadd_ps(
                v_odd, odd_v, _mm256_fmadd_ps(v_even, even_v, prev));
            _mm256_storeu_ps(sums_buf + x, next);
        }
        for (; x < nrep; ++x) {
            sums_buf[x] = std::fmaf(
                odd_s, line[2 * x + 1], std::fmaf(even_s, line[2 * x], sums_buf[x]));
        }
    }
    const __m256 v_half = _mm256_set1_ps(0.5f);
    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_max = _mm256_set1_ps(65535.0f);
    uint16_t* dst = orow + first;
    int64_t x = 0;
    for (; x + 8 <= nrep; x += 8) {
        const __m256 r = _mm256_add_ps(_mm256_loadu_ps(sums_buf + x), v_half);
        __m256 c = _mm256_blendv_ps(r, v_zero, _mm256_cmp_ps(r, v_zero, _CMP_LE_OQ));
        c = _mm256_blendv_ps(c, v_max, _mm256_cmp_ps(c, v_max, _CMP_GE_OQ));
        store_u32x8_as_u16(_mm256_cvttps_epi32(c), dst + x);
    }
    for (; x < nrep; ++x) {
        const float rounded = sums_buf[x] + 0.5f;
        if (rounded <= 0.0f) {
            dst[x] = 0;
        } else if (rounded >= 65535.0f) {
            dst[x] = 65535;
        } else {
            dst[x] = static_cast<uint16_t>(static_cast<int32_t>(rounded));
        }
    }
}

void activity_row_u16_avx2(
    const uint16_t* gr, const uint16_t* fr, uint16_t* orow, int64_t w
) {
    int64_t x = 0;
    for (; x + 8 <= w; x += 8) {
        const __m256i vg = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gr + x)));
        const __m256i vf = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(fr + x)));
        store_u32x8_as_u16(normalized_diff_8_avx2(vg, vf), orow + x);
    }
    for (; x < w; ++x) {
        const int64_t value = gr[x];
        const int64_t mean = fr[x];
        const int64_t maximum = value > mean ? value : mean;
        const int64_t denom = (maximum * 100LL * int64_t{0x80008001LL}) >> 47;
        const int64_t num = 110LL * std::abs(value - mean);
        const int64_t res = (num * static_cast<int64_t>(kActivityRecipU32[static_cast<size_t>(denom)])) >> 32;
        orow[x] = static_cast<uint16_t>(res < 65535LL ? res : 65535LL);
    }
}

void detail_row_u16_avx2(
    const uint16_t* r0, const uint16_t* r1, uint16_t* orow, int64_t full_w
) {
    const __m256i m16 = _mm256_set1_epi32(0xFFFF);
    int64_t x = 0;
    for (; x + 8 <= full_w; x += 8) {
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + 2 * x));
        const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + 2 * x));
        const __m256i va = _mm256_and_si256(p0, m16);
        const __m256i vb = _mm256_srli_epi32(p0, 16);
        const __m256i vc = _mm256_and_si256(p1, m16);
        const __m256i vd = _mm256_srli_epi32(p1, 16);
        const __m256i sum = _mm256_add_epi32(_mm256_add_epi32(va, vb), _mm256_add_epi32(vc, vd));
        const __m256i vmean = _mm256_srli_epi32(sum, 2);
        const __m256i ea = normalized_diff_8_avx2(va, vmean);
        const __m256i eb = normalized_diff_8_avx2(vb, vmean);
        const __m256i ec = normalized_diff_8_avx2(vc, vmean);
        const __m256i ed = normalized_diff_8_avx2(vd, vmean);
        const __m256i etot = _mm256_add_epi32(_mm256_add_epi32(ea, eb), _mm256_add_epi32(ec, ed));
        store_u32x8_as_u16(_mm256_srli_epi32(etot, 2), orow + x);
    }
    for (; x < full_w; ++x) {
        const int64_t x0 = 2 * x;
        const int64_t a = r0[x0], b = r0[x0 + 1], c = r1[x0], d = r1[x0 + 1];
        const int64_t mean = (a + b + c + d) >> 2;
        auto ndiff = [mean](int64_t v) -> int64_t {
            const int64_t maximum = v > mean ? v : mean;
            const int64_t denom = (maximum * 100LL * int64_t{0x80008001LL}) >> 47;
            const int64_t num = 110LL * std::abs(v - mean);
            const int64_t res = (num * static_cast<int64_t>(kActivityRecipU32[static_cast<size_t>(denom)])) >> 32;
            return res < 65535LL ? res : 65535LL;
        };
        orow[x] = static_cast<uint16_t>((ndiff(a) + ndiff(b) + ndiff(c) + ndiff(d)) >> 2);
    }
}

void select_activity_row_u16_avx2(
    const uint16_t* ar, uint16_t* br, uint16_t* dr, uint16_t* vr,
    int64_t w, uint16_t code, uint32_t reciprocal_u32
) {
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i v_code = _mm256_set1_epi16(static_cast<int16_t>(code));
    const __m256i v_recip = _mm256_set1_epi32(static_cast<int32_t>(reciprocal_u32));
    int64_t x = 0;
    for (; x + 16 <= w; x += 16) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ar + x));
        if (vr != nullptr) {
            const __m256i lo32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(va));
            const __m256i hi32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(va, 1));
            auto mul_r = [&](__m256i v) -> __m256i {
                const __m256i qe = _mm256_srli_epi64(_mm256_mul_epu32(v, v_recip), 32);
                const __m256i qo = _mm256_mul_epu32(_mm256_srli_epi64(v, 32), v_recip);
                return _mm256_blend_epi32(qe, qo, 0xAA);
            };
            const __m256i add16 = _mm256_permute4x64_epi64(
                _mm256_packus_epi32(mul_r(lo32), mul_r(hi32)), 0xD8);
            const __m256i prev_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vr + x));
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(vr + x), _mm256_add_epi16(prev_v, add16));
        }
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
        const __m256i le_b = _mm256_cmpeq_epi16(_mm256_max_epu16(va, vb), vb);
        if (!_mm256_testc_si256(le_b, all_ones)) {
            const __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dr + x));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(br + x), _mm256_blendv_epi8(va, vb, le_b));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dr + x), _mm256_blendv_epi8(v_code, vd, le_b));
        }
    }
    for (; x < w; ++x) {
        const uint16_t val = ar[x];
        if (vr != nullptr) {
            vr[x] = static_cast<uint16_t>(
                vr[x] + ((static_cast<uint64_t>(val) * reciprocal_u32) >> 32u));
        }
        if (val > br[x]) {
            br[x] = val;
            dr[x] = code;
        }
    }
}

void select_detail_row_u16_avx2(
    const uint16_t* ar, uint16_t* br, uint16_t* sr, uint16_t* wr,
    int64_t w, uint16_t code
) {
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i v_code = _mm256_set1_epi16(static_cast<int16_t>(code));
    int64_t x = 0;
    for (; x + 16 <= w; x += 16) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ar + x));
        const __m256i vs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sr + x));
        const __m256i le_s = _mm256_cmpeq_epi16(_mm256_max_epu16(va, vs), vs);
        if (_mm256_testc_si256(le_s, all_ones)) {
            continue;
        }
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(br + x));
        const __m256i le_b = _mm256_cmpeq_epi16(_mm256_max_epu16(va, vb), vb);
        const __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wr + x));
        const __m256i sr_step1 = _mm256_blendv_epi8(va, vs, le_s);
        const __m256i new_sr = _mm256_blendv_epi8(vb, sr_step1, le_b);
        const __m256i new_br = _mm256_blendv_epi8(va, vb, le_b);
        const __m256i new_wr = _mm256_blendv_epi8(v_code, vw, le_b);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(sr + x), new_sr);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(br + x), new_br);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(wr + x), new_wr);
    }
    for (; x < w; ++x) {
        const uint16_t val = ar[x];
        if (val > br[x]) {
            sr[x] = br[x];
            br[x] = val;
            wr[x] = code;
        } else if (val > sr[x]) {
            sr[x] = val;
        }
    }
}

void output_u16_to_u8_row_avx2(
    const uint16_t* cr, uint8_t* orow, int64_t elems, bool e37
) {
    const __m256i v_one = _mm256_set1_epi16(1);
    int64_t x = 0;
    for (; x + 16 <= elems; x += 16) {
        __m256i v = _mm256_srli_epi16(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cr + x)), 8);
        if (e37) {
            v = _mm256_subs_epu16(v, v_one);
        }
        const __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi16(v, v), 0xD8);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(orow + x), _mm256_castsi256_si128(p));
    }
    for (; x < elems; ++x) {
        uint16_t v = cr[x] >> 8u;
        if (e37 && v > 0) {
            v -= 1;
        }
        orow[x] = static_cast<uint8_t>(v);
    }
}

void output_u16_to_f32_row_avx2(
    const uint16_t* cr, float* orow, int64_t elems, float factor
) {
    const __m256 v_fac = _mm256_set1_ps(factor);
    int64_t x = 0;
    for (; x + 8 <= elems; x += 8) {
        const __m256 vf = _mm256_cvtepi32_ps(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(cr + x))));
        _mm256_storeu_ps(orow + x, _mm256_mul_ps(vf, v_fac));
    }
    for (; x < elems; ++x) {
        orow[x] = static_cast<float>(cr[x]) * factor;
    }
}

void tile_accumulate_3c_row_avx2(
    const void* tr, int src_kind, const float* wr,
    float* ar, float* war, int64_t fw
) {
    const __m256i perm0 = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
    const __m256i perm1 = _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
    const __m256i perm2 = _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);
    int64_t dx = 0;
    for (; dx + 8 <= fw; dx += 8) {
        const __m256 w = _mm256_loadu_ps(wr + dx);
        _mm256_storeu_ps(war + dx, _mm256_add_ps(_mm256_loadu_ps(war + dx), w));
        const __m256 w_blk0 = _mm256_permutevar8x32_ps(w, perm0);
        const __m256 w_blk1 = _mm256_permutevar8x32_ps(w, perm1);
        const __m256 w_blk2 = _mm256_permutevar8x32_ps(w, perm2);

        __m256 t0, t1, t2;
        if (src_kind == 0) {
            const float* src = static_cast<const float*>(tr) + dx * 3;
            t0 = _mm256_loadu_ps(src);
            t1 = _mm256_loadu_ps(src + 8);
            t2 = _mm256_loadu_ps(src + 16);
        } else if (src_kind == 1) {
            const uint8_t* src = static_cast<const uint8_t*>(tr) + dx * 3;
            const __m128i b01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            const __m128i b2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + 16));
            t0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b01));
            t1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b01, 8)));
            t2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b2));
        } else {
            const uint16_t* src = static_cast<const uint16_t*>(tr) + dx * 3;
            t0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src))));
            t1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8))));
            t2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 16))));
        }

        float* dst = ar + dx * 3;
        _mm256_storeu_ps(dst, _mm256_add_ps(_mm256_loadu_ps(dst), _mm256_mul_ps(t0, w_blk0)));
        _mm256_storeu_ps(dst + 8, _mm256_add_ps(_mm256_loadu_ps(dst + 8), _mm256_mul_ps(t1, w_blk1)));
        _mm256_storeu_ps(dst + 16, _mm256_add_ps(_mm256_loadu_ps(dst + 16), _mm256_mul_ps(t2, w_blk2)));
    }
    for (; dx < fw; ++dx) {
        const float w = wr[dx];
        war[dx] += w;
        const int64_t o = dx * 3;
        if (src_kind == 0) {
            const float* src = static_cast<const float*>(tr) + o;
            ar[o] += src[0] * w;
            ar[o + 1] += src[1] * w;
            ar[o + 2] += src[2] * w;
        } else if (src_kind == 1) {
            const uint8_t* src = static_cast<const uint8_t*>(tr) + o;
            ar[o] += static_cast<float>(src[0]) * w;
            ar[o + 1] += static_cast<float>(src[1]) * w;
            ar[o + 2] += static_cast<float>(src[2]) * w;
        } else {
            const uint16_t* src = static_cast<const uint16_t*>(tr) + o;
            ar[o] += static_cast<float>(src[0]) * w;
            ar[o + 1] += static_cast<float>(src[1]) * w;
            ar[o + 2] += static_cast<float>(src[2]) * w;
        }
    }
}

namespace {

inline double hsum_pd4(__m256d v) {
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    const __m128d s = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_hadd_pd(s, s));
}

template <int P>
inline void ecc_prepare_level_row_impl(
    const float* rm1, const float* r0, const float* rp1,
    int x0, int rw, double cx, double inv_s, double mean_t, double g_scale, double v,
    bool sample_h_row, bool stride2,
    float* dst_t0, float* dst_gx, float* dst_gy,
    double& norm_T2, double* js, double* tv, double* H_tri
) {
    const __m256d v_mean = _mm256_set1_pd(mean_t);
    const __m256d v_gs = _mm256_set1_pd(g_scale);
    const __m256d v_two = _mm256_set1_pd(2.0);
    const __m256d v_vv = _mm256_set1_pd(v);
    const __m256d v_cx = _mm256_set1_pd(cx);
    const __m256d v_invs = _mm256_set1_pd(inv_s);
    const __m256d v_step = _mm256_set1_pd(4.0);
    __m256d vx = _mm256_setr_pd(
        static_cast<double>(x0), static_cast<double>(x0 + 1),
        static_cast<double>(x0 + 2), static_cast<double>(x0 + 3));

    __m256d v_norm = _mm256_setzero_pd();
    __m256d v_js[P];
    __m256d v_tv[P];
    for (int r = 0; r < P; ++r) {
        v_js[r] = _mm256_setzero_pd();
        v_tv[r] = _mm256_setzero_pd();
    }
    constexpr int kTri = (P * (P + 1)) / 2;
    __m256d v_h256[kTri];
    __m128d v_h128[kTri];
    if (sample_h_row) {
        for (int k = 0; k < kTri; ++k) {
            v_h256[k] = _mm256_setzero_pd();
            v_h128[k] = _mm_setzero_pd();
        }
    }

    int rx = 0;
    for (; rx + 4 <= rw; rx += 4) {
        const int x = x0 + rx;
        const __m256d u = _mm256_mul_pd(_mm256_sub_pd(vx, v_cx), v_invs);
        vx = _mm256_add_pd(vx, v_step);

        const __m256d tl = _mm256_cvtps_pd(_mm_loadu_ps(rm1 + x - 1));
        const __m256d tc = _mm256_cvtps_pd(_mm_loadu_ps(rm1 + x));
        const __m256d tr = _mm256_cvtps_pd(_mm_loadu_ps(rm1 + x + 1));

        const __m256d ml = _mm256_cvtps_pd(_mm_loadu_ps(r0 + x - 1));
        const __m256d c0 = _mm256_cvtps_pd(_mm_loadu_ps(r0 + x));
        const __m256d mr = _mm256_cvtps_pd(_mm_loadu_ps(r0 + x + 1));

        const __m256d bl = _mm256_cvtps_pd(_mm_loadu_ps(rp1 + x - 1));
        const __m256d bc = _mm256_cvtps_pd(_mm_loadu_ps(rp1 + x));
        const __m256d br = _mm256_cvtps_pd(_mm_loadu_ps(rp1 + x + 1));

        const __m256d t0 = _mm256_sub_pd(c0, v_mean);
        const __m256d gx_pos = _mm256_add_pd(_mm256_add_pd(tr, br), _mm256_mul_pd(v_two, mr));
        const __m256d gx_neg = _mm256_add_pd(_mm256_add_pd(tl, bl), _mm256_mul_pd(v_two, ml));
        const __m256d gx = _mm256_mul_pd(_mm256_sub_pd(gx_pos, gx_neg), v_gs);

        const __m256d gy_pos = _mm256_add_pd(_mm256_add_pd(bl, br), _mm256_mul_pd(v_two, bc));
        const __m256d gy_neg = _mm256_add_pd(_mm256_add_pd(tl, tr), _mm256_mul_pd(v_two, tc));
        const __m256d gy = _mm256_mul_pd(_mm256_sub_pd(gy_pos, gy_neg), v_gs);

        _mm_storeu_ps(dst_t0 + rx, _mm256_cvtpd_ps(t0));
        _mm_storeu_ps(dst_gx + rx, _mm256_cvtpd_ps(gx));
        _mm_storeu_ps(dst_gy + rx, _mm256_cvtpd_ps(gy));

        v_norm = _mm256_add_pd(v_norm, _mm256_mul_pd(t0, t0));

        __m256d J[P];
        if constexpr (P == 6) {
            J[0] = _mm256_mul_pd(gx, u);
            J[1] = _mm256_mul_pd(gy, u);
            J[2] = _mm256_mul_pd(gx, v_vv);
            J[3] = _mm256_mul_pd(gy, v_vv);
            J[4] = gx;
            J[5] = gy;
        } else {
            const __m256d guv = _mm256_add_pd(_mm256_mul_pd(gx, u), _mm256_mul_pd(gy, v_vv));
            const __m256d neg_guv = _mm256_sub_pd(_mm256_setzero_pd(), guv);
            J[0] = _mm256_mul_pd(gx, u);
            J[1] = _mm256_mul_pd(gy, u);
            J[2] = _mm256_mul_pd(u, neg_guv);
            J[3] = _mm256_mul_pd(gx, v_vv);
            J[4] = _mm256_mul_pd(gy, v_vv);
            J[5] = _mm256_mul_pd(v_vv, neg_guv);
            J[6] = gx;
            J[7] = gy;
        }

        for (int r = 0; r < P; ++r) {
            v_js[r] = _mm256_add_pd(v_js[r], J[r]);
            v_tv[r] = _mm256_add_pd(v_tv[r], _mm256_mul_pd(J[r], t0));
        }

        if (sample_h_row) {
            if (!stride2) {
                int idx = 0;
                for (int r = 0; r < P; ++r) {
                    const __m256d jr = J[r];
                    for (int c = r; c < P; ++c) {
                        v_h256[idx] = _mm256_add_pd(v_h256[idx], _mm256_mul_pd(jr, J[c]));
                        ++idx;
                    }
                }
            } else {
                // Sample even rx lanes (0 and 2) when stride2 is active.
                __m128d j_even[P];
                for (int r = 0; r < P; ++r) {
                    j_even[r] = _mm_shuffle_pd(
                        _mm256_castpd256_pd128(J[r]), _mm256_extractf128_pd(J[r], 1), 0);
                }
                int idx = 0;
                for (int r = 0; r < P; ++r) {
                    const __m128d jr = j_even[r];
                    for (int c = r; c < P; ++c) {
                        v_h128[idx] = _mm_add_pd(v_h128[idx], _mm_mul_pd(jr, j_even[c]));
                        ++idx;
                    }
                }
            }
        }
    }

    double s_norm = hsum_pd4(v_norm);
    double s_js[P];
    double s_tv[P];
    for (int r = 0; r < P; ++r) {
        s_js[r] = hsum_pd4(v_js[r]);
        s_tv[r] = hsum_pd4(v_tv[r]);
    }
    double s_h[kTri]{};
    if (sample_h_row) {
        if (!stride2) {
            for (int k = 0; k < kTri; ++k) {
                s_h[k] = hsum_pd4(v_h256[k]);
            }
        } else {
            for (int k = 0; k < kTri; ++k) {
                s_h[k] = _mm_cvtsd_f64(_mm_hadd_pd(v_h128[k], v_h128[k]));
            }
        }
    }

    for (; rx < rw; ++rx) {
        const int x = x0 + rx;
        const double u = (static_cast<double>(x) - cx) * inv_s;
        const double t0 = static_cast<double>(r0[x]) - mean_t;
        const double tl = static_cast<double>(rm1[x - 1]);
        const double tc = static_cast<double>(rm1[x]);
        const double tr = static_cast<double>(rm1[x + 1]);
        const double ml = static_cast<double>(r0[x - 1]);
        const double mr = static_cast<double>(r0[x + 1]);
        const double bl = static_cast<double>(rp1[x - 1]);
        const double bc = static_cast<double>(rp1[x]);
        const double br = static_cast<double>(rp1[x + 1]);
        const double gx = ((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)) * g_scale;
        const double gy = ((bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)) * g_scale;
        dst_t0[rx] = static_cast<float>(t0);
        dst_gx[rx] = static_cast<float>(gx);
        dst_gy[rx] = static_cast<float>(gy);
        s_norm += t0 * t0;

        double J[P];
        if constexpr (P == 6) {
            J[0] = gx * u; J[1] = gy * u; J[2] = gx * v;
            J[3] = gy * v; J[4] = gx;     J[5] = gy;
        } else {
            const double guv = gx * u + gy * v;
            J[0] = gx * u; J[1] = gy * u; J[2] = -u * guv;
            J[3] = gx * v; J[4] = gy * v; J[5] = -v * guv;
            J[6] = gx;     J[7] = gy;
        }
        for (int r = 0; r < P; ++r) {
            s_js[r] += J[r];
            s_tv[r] += J[r] * t0;
        }
        if (sample_h_row && ((rx & 1) == 0 || !stride2)) {
            int idx = 0;
            for (int r = 0; r < P; ++r) {
                const double jr = J[r];
                for (int c = r; c < P; ++c) {
                    s_h[idx++] += jr * J[c];
                }
            }
        }
    }

    norm_T2 += s_norm;
    for (int r = 0; r < P; ++r) {
        js[r] += s_js[r];
        tv[r] += s_tv[r];
    }
    if (sample_h_row) {
        for (int k = 0; k < kTri; ++k) {
            H_tri[k] += s_h[k];
        }
    }
}

inline __m256d ecc_bilinear_4_avx2(
    const float* src_data, int w, int h, __m256d sx, __m256d sy
) {
    const __m128i vi_zero = _mm_setzero_si128();
    const __m128i vi_wm2 = _mm_set1_epi32(std::max(0, w - 2));
    const __m128i vi_hm2 = _mm_set1_epi32(std::max(0, h - 2));
    const __m128i ix_v = _mm_min_epi32(_mm_max_epi32(_mm256_cvttpd_epi32(sx), vi_zero), vi_wm2);
    const __m128i iy_v = _mm_min_epi32(_mm_max_epi32(_mm256_cvttpd_epi32(sy), vi_zero), vi_hm2);
    alignas(16) int32_t ix[4], iy[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(ix), ix_v);
    _mm_store_si128(reinterpret_cast<__m128i*>(iy), iy_v);

    const __m256d fx = _mm256_sub_pd(sx, _mm256_cvtepi32_pd(ix_v));
    const __m256d fy = _mm256_sub_pd(sy, _mm256_cvtepi32_pd(iy_v));

    alignas(16) float q00[4], q01[4], q10[4], q11[4];
    for (int m = 0; m < 4; ++m) {
        const float* p0 = src_data + static_cast<int64_t>(iy[m]) * w + ix[m];
        const float* p1 = p0 + w;
        q00[m] = p0[0];
        q01[m] = p0[1];
        q10[m] = p1[0];
        q11[m] = p1[1];
    }
    const __m256d v00 = _mm256_cvtps_pd(_mm_load_ps(q00));
    const __m256d v01 = _mm256_cvtps_pd(_mm_load_ps(q01));
    const __m256d v10 = _mm256_cvtps_pd(_mm_load_ps(q10));
    const __m256d v11 = _mm256_cvtps_pd(_mm_load_ps(q11));

    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d top = _mm256_add_pd(v00, _mm256_mul_pd(fx, _mm256_sub_pd(v01, v00)));
    const __m256d bot = _mm256_add_pd(v10, _mm256_mul_pd(fx, _mm256_sub_pd(v11, v10)));
    return _mm256_add_pd(
        _mm256_mul_pd(top, _mm256_sub_pd(v_one, fy)),
        _mm256_mul_pd(bot, fy));
}

}  // namespace

void ecc_prepare_level_row_avx2(
    const float* rm1, const float* r0, const float* rp1,
    int x0, int rw, double cx, double inv_s, double mean_t, double g_scale, double v,
    int p_count, bool sample_h_row, bool stride2,
    float* dst_t0, float* dst_gx, float* dst_gy,
    double& norm_T2, double* js, double* tv, double* H_tri
) {
    if (p_count == 6) {
        ecc_prepare_level_row_impl<6>(
            rm1, r0, rp1, x0, rw, cx, inv_s, mean_t, g_scale, v,
            sample_h_row, stride2, dst_t0, dst_gx, dst_gy, norm_T2, js, tv, H_tri);
    } else {
        ecc_prepare_level_row_impl<8>(
            rm1, r0, rp1, x0, rw, cx, inv_s, mean_t, g_scale, v,
            sample_h_row, stride2, dst_t0, dst_gx, dst_gy, norm_T2, js, tv, H_tri);
    }
}

void ecc_ic_pass_all_in_row_p6_avx2(
    const float* src_data, int w, int h,
    const float* t0_row, const float* gx_row, const float* gy_row, int rw,
    double sx0, double sy0, double dsx, double dsy,
    double u0, double inv_s, double v,
    double& rs_iw, double& rs_iw2, double& rs_tiw, double* raw_g
) {
    __m256d v_iw = _mm256_setzero_pd();
    __m256d v_iw2 = _mm256_setzero_pd();
    __m256d v_tiw = _mm256_setzero_pd();
    __m256d v_rg[6];
    for (int k = 0; k < 6; ++k) {
        v_rg[k] = _mm256_setzero_pd();
    }
    const __m256d v_offsets = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
    const __m256d v_dsx = _mm256_set1_pd(dsx);
    const __m256d v_dsy = _mm256_set1_pd(dsy);
    const __m256d v_invs = _mm256_set1_pd(inv_s);
    const __m256d v_vv = _mm256_set1_pd(v);
    __m256d v_sx = _mm256_add_pd(_mm256_set1_pd(sx0), _mm256_mul_pd(v_offsets, v_dsx));
    __m256d v_sy = _mm256_add_pd(_mm256_set1_pd(sy0), _mm256_mul_pd(v_offsets, v_dsy));
    __m256d v_u = _mm256_add_pd(_mm256_set1_pd(u0), _mm256_mul_pd(v_offsets, v_invs));
    const __m256d v_dsx4 = _mm256_set1_pd(dsx * 4.0);
    const __m256d v_dsy4 = _mm256_set1_pd(dsy * 4.0);
    const __m256d v_invs4 = _mm256_set1_pd(inv_s * 4.0);

    int rx = 0;
    for (; rx + 4 <= rw; rx += 4) {
        const __m256d iw = ecc_bilinear_4_avx2(src_data, w, h, v_sx, v_sy);
        const __m256d t0 = _mm256_cvtps_pd(_mm_loadu_ps(t0_row + rx));
        const __m256d gxw = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(gx_row + rx)), iw);
        const __m256d gyw = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(gy_row + rx)), iw);

        v_iw = _mm256_add_pd(v_iw, iw);
        v_iw2 = _mm256_add_pd(v_iw2, _mm256_mul_pd(iw, iw));
        v_tiw = _mm256_add_pd(v_tiw, _mm256_mul_pd(t0, iw));

        v_rg[0] = _mm256_add_pd(v_rg[0], _mm256_mul_pd(gxw, v_u));
        v_rg[1] = _mm256_add_pd(v_rg[1], _mm256_mul_pd(gyw, v_u));
        v_rg[2] = _mm256_add_pd(v_rg[2], _mm256_mul_pd(gxw, v_vv));
        v_rg[3] = _mm256_add_pd(v_rg[3], _mm256_mul_pd(gyw, v_vv));
        v_rg[4] = _mm256_add_pd(v_rg[4], gxw);
        v_rg[5] = _mm256_add_pd(v_rg[5], gyw);

        v_sx = _mm256_add_pd(v_sx, v_dsx4);
        v_sy = _mm256_add_pd(v_sy, v_dsy4);
        v_u = _mm256_add_pd(v_u, v_invs4);
    }

    double s_iw = hsum_pd4(v_iw);
    double s_iw2 = hsum_pd4(v_iw2);
    double s_tiw = hsum_pd4(v_tiw);
    double s_rg[6];
    for (int k = 0; k < 6; ++k) {
        s_rg[k] = hsum_pd4(v_rg[k]);
    }

    double sx = sx0 + static_cast<double>(rx) * dsx;
    double sy = sy0 + static_cast<double>(rx) * dsy;
    double u = u0 + static_cast<double>(rx) * inv_s;
    for (; rx < rw; ++rx) {
        int ix = static_cast<int>(sx);
        int iy = static_cast<int>(sy);
        if (ix >= w - 1) ix = w - 2; else if (ix < 0) ix = 0;
        if (iy >= h - 1) iy = h - 2; else if (iy < 0) iy = 0;
        const double fx = sx - static_cast<double>(ix);
        const double fy = sy - static_cast<double>(iy);
        const float* p0 = src_data + static_cast<int64_t>(iy) * w + ix;
        const float* p1 = p0 + w;
        const double v00 = p0[0], v01 = p0[1], v10 = p1[0], v11 = p1[1];
        const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                        + (v10 + fx * (v11 - v10)) * fy;
        const double t0 = t0_row[rx];
        const double gxw = static_cast<double>(gx_row[rx]) * iw;
        const double gyw = static_cast<double>(gy_row[rx]) * iw;
        s_iw += iw;
        s_iw2 += iw * iw;
        s_tiw += t0 * iw;
        s_rg[0] += gxw * u;
        s_rg[1] += gyw * u;
        s_rg[2] += gxw * v;
        s_rg[3] += gyw * v;
        s_rg[4] += gxw;
        s_rg[5] += gyw;
        sx += dsx;
        sy += dsy;
        u += inv_s;
    }

    rs_iw += s_iw;
    rs_iw2 += s_iw2;
    rs_tiw += s_tiw;
    for (int k = 0; k < 6; ++k) {
        raw_g[k] += s_rg[k];
    }
}

void ecc_ic_pass_all_in_row_p8_avx2(
    const float* src_data, int w, int h,
    const float* t0_row, const float* gx_row, const float* gy_row, int rw,
    double nsx0, double nsy0, double den0,
    double d_nsx, double d_nsy, double d_den,
    double u0, double inv_s, double v,
    double& rs_iw, double& rs_iw2, double& rs_tiw, double* raw_g
) {
    __m256d v_iw = _mm256_setzero_pd();
    __m256d v_iw2 = _mm256_setzero_pd();
    __m256d v_tiw = _mm256_setzero_pd();
    __m256d v_rg[8];
    for (int k = 0; k < 8; ++k) {
        v_rg[k] = _mm256_setzero_pd();
    }
    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d v_offsets = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
    const __m256d v_dnsx = _mm256_set1_pd(d_nsx);
    const __m256d v_dnsy = _mm256_set1_pd(d_nsy);
    const __m256d v_dden = _mm256_set1_pd(d_den);
    const __m256d v_invs = _mm256_set1_pd(inv_s);
    const __m256d v_vv = _mm256_set1_pd(v);
    __m256d v_nsx = _mm256_add_pd(_mm256_set1_pd(nsx0), _mm256_mul_pd(v_offsets, v_dnsx));
    __m256d v_nsy = _mm256_add_pd(_mm256_set1_pd(nsy0), _mm256_mul_pd(v_offsets, v_dnsy));
    __m256d v_den = _mm256_add_pd(_mm256_set1_pd(den0), _mm256_mul_pd(v_offsets, v_dden));
    __m256d v_u = _mm256_add_pd(_mm256_set1_pd(u0), _mm256_mul_pd(v_offsets, v_invs));
    const __m256d v_dnsx4 = _mm256_set1_pd(d_nsx * 4.0);
    const __m256d v_dnsy4 = _mm256_set1_pd(d_nsy * 4.0);
    const __m256d v_dden4 = _mm256_set1_pd(d_den * 4.0);
    const __m256d v_invs4 = _mm256_set1_pd(inv_s * 4.0);

    int rx = 0;
    for (; rx + 4 <= rw; rx += 4) {
        const __m256d inv_d = _mm256_div_pd(v_one, v_den);
        const __m256d sx = _mm256_mul_pd(v_nsx, inv_d);
        const __m256d sy = _mm256_mul_pd(v_nsy, inv_d);
        const __m256d iw = ecc_bilinear_4_avx2(src_data, w, h, sx, sy);

        const __m256d t0 = _mm256_cvtps_pd(_mm_loadu_ps(t0_row + rx));
        const __m256d gxw = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(gx_row + rx)), iw);
        const __m256d gyw = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(gy_row + rx)), iw);
        const __m256d guvw = _mm256_add_pd(_mm256_mul_pd(gxw, v_u), _mm256_mul_pd(gyw, v_vv));

        v_iw = _mm256_add_pd(v_iw, iw);
        v_iw2 = _mm256_add_pd(v_iw2, _mm256_mul_pd(iw, iw));
        v_tiw = _mm256_add_pd(v_tiw, _mm256_mul_pd(t0, iw));

        v_rg[0] = _mm256_add_pd(v_rg[0], _mm256_mul_pd(gxw, v_u));
        v_rg[1] = _mm256_add_pd(v_rg[1], _mm256_mul_pd(gyw, v_u));
        v_rg[2] = _mm256_sub_pd(v_rg[2], _mm256_mul_pd(v_u, guvw));
        v_rg[3] = _mm256_add_pd(v_rg[3], _mm256_mul_pd(gxw, v_vv));
        v_rg[4] = _mm256_add_pd(v_rg[4], _mm256_mul_pd(gyw, v_vv));
        v_rg[5] = _mm256_sub_pd(v_rg[5], _mm256_mul_pd(v_vv, guvw));
        v_rg[6] = _mm256_add_pd(v_rg[6], gxw);
        v_rg[7] = _mm256_add_pd(v_rg[7], gyw);

        v_nsx = _mm256_add_pd(v_nsx, v_dnsx4);
        v_nsy = _mm256_add_pd(v_nsy, v_dnsy4);
        v_den = _mm256_add_pd(v_den, v_dden4);
        v_u = _mm256_add_pd(v_u, v_invs4);
    }

    double s_iw = hsum_pd4(v_iw);
    double s_iw2 = hsum_pd4(v_iw2);
    double s_tiw = hsum_pd4(v_tiw);
    double s_rg[8];
    for (int k = 0; k < 8; ++k) {
        s_rg[k] = hsum_pd4(v_rg[k]);
    }

    double nsx = nsx0 + static_cast<double>(rx) * d_nsx;
    double nsy = nsy0 + static_cast<double>(rx) * d_nsy;
    double den = den0 + static_cast<double>(rx) * d_den;
    double u = u0 + static_cast<double>(rx) * inv_s;
    for (; rx < rw; ++rx) {
        const double inv_d = 1.0 / den;
        const double sx = nsx * inv_d;
        const double sy = nsy * inv_d;
        int ix = static_cast<int>(sx);
        int iy = static_cast<int>(sy);
        if (ix >= w - 1) ix = w - 2; else if (ix < 0) ix = 0;
        if (iy >= h - 1) iy = h - 2; else if (iy < 0) iy = 0;
        const double fx = sx - static_cast<double>(ix);
        const double fy = sy - static_cast<double>(iy);
        const float* p0 = src_data + static_cast<int64_t>(iy) * w + ix;
        const float* p1 = p0 + w;
        const double v00 = p0[0], v01 = p0[1], v10 = p1[0], v11 = p1[1];
        const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                        + (v10 + fx * (v11 - v10)) * fy;
        const double t0 = t0_row[rx];
        const double gxw = static_cast<double>(gx_row[rx]) * iw;
        const double gyw = static_cast<double>(gy_row[rx]) * iw;
        const double guvw = gxw * u + gyw * v;
        s_iw += iw;
        s_iw2 += iw * iw;
        s_tiw += t0 * iw;
        s_rg[0] += gxw * u;
        s_rg[1] += gyw * u;
        s_rg[2] -= u * guvw;
        s_rg[3] += gxw * v;
        s_rg[4] += gyw * v;
        s_rg[5] -= v * guvw;
        s_rg[6] += gxw;
        s_rg[7] += gyw;
        nsx += d_nsx;
        nsy += d_nsy;
        den += d_den;
        u += inv_s;
    }

    rs_iw += s_iw;
    rs_iw2 += s_iw2;
    rs_tiw += s_tiw;
    for (int k = 0; k < 8; ++k) {
        raw_g[k] += s_rg[k];
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
void align_valid_mask_row_avx2(uint8_t*, int64_t, float, float, float, float, float, float, float, float, float) {}
void align_gray16_rgb_row_avx2(const void*, bool, uint16_t*, int64_t, int64_t) {}
void align_down_half_vert_row_avx2(const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*, uint32_t*, int64_t) {}
void align_down_half_horiz_row_avx2(const uint32_t*, uint16_t*, int64_t, int64_t) {}
void align_fast_downsize_u16_row_avx2(const uint16_t*, const uint16_t*, uint16_t*, int64_t, int64_t) {}
void align_warp_alignment_row_avx2(const void*, int64_t, int64_t, bool, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, float, float, float, float, float, float, float) {}
void align_score_chunk_packed_avx2(const uint16_t*, int64_t, const uint16_t*, int64_t, int64_t, int64_t, const float*, int64_t, const float*, int64_t, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const float*, const int64_t*, int64_t, int64_t, int64_t*, int64_t*, int64_t*) {}
void align_score_grid_4d_packed_avx2(const uint16_t*, int64_t, const uint16_t*, int64_t, int64_t, int64_t, const float*, int64_t, const float*, int64_t, float, float, const float*, const float*, const float*, int64_t, int64_t, const float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t*, float*, int64_t) {}
void resample_u16_vert_acc_row_avx2_fma(const uint16_t*, float*, int64_t, float) {}
void resample_u16_horiz_rep_row_avx2_fma(const float*, float*, uint16_t*, int64_t, int64_t, const int32_t*, const float*, int64_t, int64_t, int64_t) {}
void activity_row_u16_avx2(const uint16_t*, const uint16_t*, uint16_t*, int64_t) {}
void detail_row_u16_avx2(const uint16_t*, const uint16_t*, uint16_t*, int64_t) {}
void select_activity_row_u16_avx2(const uint16_t*, uint16_t*, uint16_t*, uint16_t*, int64_t, uint16_t, uint32_t) {}
void select_detail_row_u16_avx2(const uint16_t*, uint16_t*, uint16_t*, uint16_t*, int64_t, uint16_t) {}
void output_u16_to_u8_row_avx2(const uint16_t*, uint8_t*, int64_t, bool) {}
void output_u16_to_f32_row_avx2(const uint16_t*, float*, int64_t, float) {}
void tile_accumulate_3c_row_avx2(const void*, int, const float*, float*, float*, int64_t) {}
void ecc_prepare_level_row_avx2(const float*, const float*, const float*, int, int, double, double, double, double, double, int, bool, bool, float*, float*, float*, double&, double*, double*, double*) {}
void ecc_ic_pass_all_in_row_p6_avx2(const float*, int, int, const float*, const float*, const float*, int, double, double, double, double, double, double, double, double&, double&, double&, double*) {}
void ecc_ic_pass_all_in_row_p8_avx2(const float*, int, int, const float*, const float*, const float*, int, double, double, double, double, double, double, double, double, double, double&, double&, double&, double*) {}

#endif

}  // namespace simd
}  // namespace openfocus
