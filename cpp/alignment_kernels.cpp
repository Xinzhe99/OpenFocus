#include "alignment_kernels.hpp"
#include "simd_dispatch.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-math-errno")
#endif

namespace openfocus {
namespace {

constexpr uint32_t kWeights[4] = {12u, 64u, 37u, 2u};
constexpr float kClampEdge = 1.0010000467300415f;

inline int32_t clamp_i32(int32_t v, int32_t hi) {
    return v < 0 ? 0 : (v > hi ? hi : v);
}

inline void axis_sample(float coord, int64_t length, int32_t& lo, int32_t& hi, uint32_t& frac, bool& invalid) {
    const int32_t len = static_cast<int32_t>(length);
    const float extent = static_cast<float>(len);
    invalid = (coord >= extent) || (coord < 0.0f);
    if (coord >= static_cast<float>(len - 1)) {
        coord = extent - kClampEdge;
    }
    if (coord < 0.0f) {
        coord = 0.0f;
    }
    const int32_t index = static_cast<int32_t>(coord);
    frac = static_cast<uint32_t>((coord - static_cast<float>(index)) * 256.0f);
    lo = clamp_i32(index, len - 1);
    hi = clamp_i32(index + 1, len - 1);
}

inline void axis_pad_sample(float coord, int64_t length, int32_t pad,
                            int32_t& lo, int32_t& hi, uint32_t& frac, bool& invalid) {
    const int32_t len = static_cast<int32_t>(length);
    const int32_t extent = len + 2 * pad;
    invalid = (coord >= static_cast<float>(extent)) || (coord < 0.0f);
    if (coord >= static_cast<float>(extent - 1)) {
        coord = static_cast<float>(extent) - kClampEdge;
    }
    if (coord < 0.0f) {
        coord = 0.0f;
    }
    const int32_t index = static_cast<int32_t>(coord);
    frac = static_cast<uint32_t>((coord - static_cast<float>(index)) * 256.0f);
    lo = clamp_i32(index - pad, len - 1);
    hi = clamp_i32(index + 1 - pad, len - 1);
}

inline void map_uv_row(float y, float cy, float p1c, float p0c, float& v_dy, float& u_dy) {
    const float dy_ = y - cy;
    v_dy = dy_ * p1c;
    u_dy = dy_ * p0c;
}

inline void map_uv_col(float x, float cx, float p08, float p18, float tx, float ty,
                       float v_dy, float u_dy, float& u, float& v) {
    const float dx_ = x - cx;
    v = (v_dy - (dx_ * p08)) + ty;
    u = ((dx_ * p18) + u_dy) + tx;
}

inline uint32_t read_u_val(const ArrayView& v, int64_t y, int64_t x, int64_t c = 0) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 1) {
        return static_cast<const uint8_t*>(v.data)[off];
    }
    if (v.itemsize == 2) {
        return static_cast<const uint16_t*>(v.data)[off];
    }
    return static_cast<const uint32_t*>(v.data)[off];
}

inline void write_u_val(const ArrayView& v, int64_t y, int64_t x, uint32_t val, int64_t c = 0) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 1) {
        static_cast<uint8_t*>(v.data)[off] = static_cast<uint8_t>(val);
    } else if (v.itemsize == 2) {
        static_cast<uint16_t*>(v.data)[off] = static_cast<uint16_t>(val);
    } else {
        static_cast<uint32_t*>(v.data)[off] = val;
    }
}

inline double read_float_val(const ArrayView& v, int64_t y, int64_t x, int64_t c = 0) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 4) {
        return static_cast<double>(static_cast<const float*>(v.data)[off]);
    }
    return static_cast<const double*>(v.data)[off];
}

inline void write_float_val(const ArrayView& v, int64_t y, int64_t x, double val, int64_t c = 0) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 4) {
        static_cast<float*>(v.data)[off] = static_cast<float>(val);
    } else {
        static_cast<double*>(v.data)[off] = val;
    }
}

inline uint32_t bilinear_u32(
    const ArrayView& img, int64_t iy, int64_t jy, int64_t ix, int64_t jx,
    uint32_t gx, uint32_t fx, uint32_t gy, uint32_t fy, int64_t c = 0
) {
    const uint32_t top = read_u_val(img, iy, ix, c) * gx + read_u_val(img, iy, jx, c) * fx;
    const uint32_t bottom = read_u_val(img, jy, ix, c) * gx + read_u_val(img, jy, jx, c) * fx;
    return (top * gy + bottom * fy) >> 16u;
}

template <typename T>
void swap_rgb_to_bgr_typed(const ArrayView& src, const ArrayView& dst, int64_t y_start, int64_t y_stop) {
    const int64_t w = src.shape[1];
    const auto* sb = static_cast<const T*>(src.data);
    auto* db = static_cast<T*>(dst.data);
    for (int64_t y = y_start; y < y_stop; ++y) {
        const T* sr = sb + y * src.stride[0];
        T* dr = db + y * dst.stride[0];
        for (int64_t x = 0; x < w; ++x) {
            const T* sp = sr + x * src.stride[1];
            T* dp = dr + x * dst.stride[1];
            const T c0 = sp[0], c1 = sp[src.stride[2]], c2 = sp[2 * src.stride[2]];
            dp[0] = c2;
            dp[dst.stride[2]] = c1;
            dp[2 * dst.stride[2]] = c0;
        }
    }
}

template <typename T>
void swap_bgr_inplace_typed(const ArrayView& arr, int64_t y_start, int64_t y_stop) {
    const int64_t w = arr.shape[1];
    auto* ab = static_cast<T*>(arr.data);
    for (int64_t y = y_start; y < y_stop; ++y) {
        T* ar = ab + y * arr.stride[0];
        for (int64_t x = 0; x < w; ++x) {
            T* p = ar + x * arr.stride[1];
            const T v0 = p[0];
            p[0] = p[2 * arr.stride[2]];
            p[2 * arr.stride[2]] = v0;
        }
    }
}

template <typename T, int Channels, bool BgrRev>
void warp_translation_row_packed_impl(
    const T* r0, const T* r1, T* out_row, int64_t rw,
    const int32_t* ix, const int32_t* jx, const uint32_t* fx, const uint8_t* inv_x,
    uint32_t gy, uint32_t fy
) {
    for (int64_t ox = 0; ox < rw; ++ox) {
        T* dp = out_row + ox * Channels;
        if (inv_x[ox]) {
            for (int c = 0; c < Channels; ++c) {
                dp[c] = 0;
            }
            continue;
        }
        const uint32_t fxx = fx[ox];
        const uint32_t gxx = 256u - fxx;
        const int64_t ixx = static_cast<int64_t>(ix[ox]) * Channels;
        const int64_t jxx = static_cast<int64_t>(jx[ox]) * Channels;
        if constexpr (Channels == 1) {
            const uint32_t top = static_cast<uint32_t>(r0[ixx]) * gxx + static_cast<uint32_t>(r0[jxx]) * fxx;
            const uint32_t bot = static_cast<uint32_t>(r1[ixx]) * gxx + static_cast<uint32_t>(r1[jxx]) * fxx;
            dp[0] = static_cast<T>((top * gy + bot * fy) >> 16u);
        } else {
            for (int c = 0; c < 3; ++c) {
                const int sc = BgrRev ? (2 - c) : c;
                const uint32_t top = static_cast<uint32_t>(r0[ixx + sc]) * gxx
                                   + static_cast<uint32_t>(r0[jxx + sc]) * fxx;
                const uint32_t bot = static_cast<uint32_t>(r1[ixx + sc]) * gxx
                                   + static_cast<uint32_t>(r1[jxx + sc]) * fxx;
                dp[c] = static_cast<T>((top * gy + bot * fy) >> 16u);
            }
        }
    }
}

void warp_translation_row_packed(
    const ArrayView& image, const ArrayView& out, int64_t iy_b, int64_t jy_b, int64_t oy,
    int64_t rw, int64_t channels, bool bgr_rev,
    const int32_t* ix, const int32_t* jx, const uint32_t* fx, const uint8_t* inv_x,
    uint32_t gy, uint32_t fy
) {
    if (channels == 3 && simd::use_avx2() && image.shape[1] >= 3
            && image.shape[1] <= std::numeric_limits<int32_t>::max()) {
        const int64_t item = image.itemsize;
        const auto* sb = static_cast<const uint8_t*>(image.data);
        const uint8_t* r0 = sb + iy_b * image.stride[0] * item;
        const uint8_t* r1 = sb + jy_b * image.stride[0] * item;
        uint8_t* dst = static_cast<uint8_t*>(out.data) + oy * out.stride[0] * item;
        // The vector kernel reads a whole pixel before it stores one and
        // stores past it; an output row that shares bytes with a source row
        // keeps the per-channel order of the loops below.
        const uint64_t src_bytes = static_cast<uint64_t>(image.shape[1] * 3 * item);
        const uint64_t dst_bytes = static_cast<uint64_t>(rw * 3 * item);
        if (!byte_ranges_overlap(dst, dst_bytes, r0, src_bytes)
                && !byte_ranges_overlap(dst, dst_bytes, r1, src_bytes)) {
            simd::align_warp_translation_rgb_row_avx2(
                r0, r1, item == 1, dst, rw, ix, jx, fx, inv_x, fy, bgr_rev, image.shape[1]);
            return;
        }
    }
    if (channels == 1 && simd::use_avx2()) {
        const int64_t item = image.itemsize;
        const auto* sb = static_cast<const uint8_t*>(image.data);
        const uint8_t* r0 = sb + iy_b * image.stride[0] * item;
        const uint8_t* r1 = sb + jy_b * image.stride[0] * item;
        uint8_t* dst = static_cast<uint8_t*>(out.data) + oy * out.stride[0] * item;
        // As above: the vector kernel reads nine samples before it stores eight.
        const uint64_t src_bytes = static_cast<uint64_t>(image.shape[1] * item);
        const uint64_t dst_bytes = static_cast<uint64_t>(rw * item);
        if (!byte_ranges_overlap(dst, dst_bytes, r0, src_bytes)
                && !byte_ranges_overlap(dst, dst_bytes, r1, src_bytes)) {
            simd::align_warp_translation_gray_row_avx2(r0, r1, item == 1, dst, rw, ix, jx, fx, inv_x, fy);
            return;
        }
    }
    if (image.itemsize == 1) {
        const auto* sb = static_cast<const uint8_t*>(image.data);
        const auto* r0 = sb + iy_b * image.stride[0];
        const auto* r1 = sb + jy_b * image.stride[0];
        auto* dst = static_cast<uint8_t*>(out.data) + oy * out.stride[0];
        if (channels == 1) {
            warp_translation_row_packed_impl<uint8_t, 1, false>(r0, r1, dst, rw, ix, jx, fx, inv_x, gy, fy);
        } else if (bgr_rev) {
            warp_translation_row_packed_impl<uint8_t, 3, true>(r0, r1, dst, rw, ix, jx, fx, inv_x, gy, fy);
        } else {
            warp_translation_row_packed_impl<uint8_t, 3, false>(r0, r1, dst, rw, ix, jx, fx, inv_x, gy, fy);
        }
    } else {
        const auto* sb = static_cast<const uint16_t*>(image.data);
        const auto* r0 = sb + iy_b * image.stride[0];
        const auto* r1 = sb + jy_b * image.stride[0];
        auto* dst = static_cast<uint16_t*>(out.data) + oy * out.stride[0];
        if (channels == 1) {
            warp_translation_row_packed_impl<uint16_t, 1, false>(r0, r1, dst, rw, ix, jx, fx, inv_x, gy, fy);
        } else if (bgr_rev) {
            warp_translation_row_packed_impl<uint16_t, 3, true>(r0, r1, dst, rw, ix, jx, fx, inv_x, gy, fy);
        } else {
            warp_translation_row_packed_impl<uint16_t, 3, false>(r0, r1, dst, rw, ix, jx, fx, inv_x, gy, fy);
        }
    }
}

bool is_warp_packed(const ArrayView& image, const ArrayView& out, int64_t channels) {
    return (image.itemsize == 1 || image.itemsize == 2)
        && out.itemsize == image.itemsize
        && (channels == 1 || channels == 3)
        && image.stride[1] == channels
        && out.stride[1] == channels
        && (channels == 1 || (image.stride[2] == 1 && out.stride[2] == 1));
}

}  // namespace

void of_align_swap_rgb_to_bgr_rows_cpp(
    const ArrayView& src, const ArrayView& dst, int64_t y_start, int64_t y_stop
) {
    if (src.itemsize == 1) {
        swap_rgb_to_bgr_typed<uint8_t>(src, dst, y_start, y_stop);
    } else {
        swap_rgb_to_bgr_typed<uint16_t>(src, dst, y_start, y_stop);
    }
}

void of_align_swap_bgr_inplace_rows_cpp(
    const ArrayView& arr, int64_t y_start, int64_t y_stop
) {
    if (arr.itemsize == 1) {
        swap_bgr_inplace_typed<uint8_t>(arr, y_start, y_stop);
    } else {
        swap_bgr_inplace_typed<uint16_t>(arr, y_start, y_stop);
    }
}

void of_align_gray16_rows_cpp(
    const ArrayView& image, const ArrayView& out, int64_t y_start, int64_t y_stop
) {
    const int64_t w = image.shape[1];
    const uint32_t shift = (image.itemsize == 1) ? 8u : 0u;
    uint16_t* ob = static_cast<uint16_t*>(out.data);
    const int64_t take = (image.ndim == 2) ? 1 : image.shape[2];
    const bool packed_rgb_simd = simd::use_avx2()
        && image.ndim == 3 && take >= 3 && w >= 12
        && (image.itemsize == 1 || image.itemsize == 2)
        && image.stride[2] == 1
        && (image.stride[1] == 3 || (take >= 4 && image.stride[1] == 4))
        && out.stride[1] == 1;

    for (int64_t y = y_start; y < y_stop; ++y) {
        uint16_t* orow = ob + y * out.stride[0];
        if (packed_rgb_simd) {
            const void* src_row = (image.itemsize == 1)
                ? static_cast<const void*>(static_cast<const uint8_t*>(image.data) + y * image.stride[0])
                : static_cast<const void*>(static_cast<const uint16_t*>(image.data) + y * image.stride[0]);
            simd::align_gray16_rgb_row_avx2(src_row, image.itemsize == 1, orow, w, image.stride[1]);
        } else if (image.ndim == 2) {
            for (int64_t x = 0; x < w; ++x) {
                orow[x * out.stride[1]] = static_cast<uint16_t>(read_u_val(image, y, x) << shift);
            }
        } else if (take >= 3) {
            for (int64_t x = 0; x < w; ++x) {
                const uint32_t sum = read_u_val(image, y, x, 0)
                                   + read_u_val(image, y, x, 1)
                                   + read_u_val(image, y, x, 2);
                orow[x * out.stride[1]] = static_cast<uint16_t>((sum / 3u) << shift);
            }
        } else {
            for (int64_t x = 0; x < w; ++x) {
                uint32_t sum = 0;
                for (int64_t c = 0; c < take; ++c) {
                    sum += read_u_val(image, y, x, c);
                }
                orow[x * out.stride[1]] = static_cast<uint16_t>((sum / static_cast<uint32_t>(take)) << shift);
            }
        }
    }
}

void of_align_down_half_rows_cpp(
    const ArrayView& src, const ArrayView& out, int64_t oy_start, int64_t oy_stop
) {
    const int64_t h = src.shape[0];
    const int64_t w = src.shape[1];
    const int64_t ow = out.shape[1];
    std::vector<uint32_t> scratch(static_cast<size_t>(w));
    const uint16_t* sb = static_cast<const uint16_t*>(src.data);
    uint16_t* ob = static_cast<uint16_t*>(out.data);
    const bool packed = (src.itemsize == 2 && out.itemsize == 2
                         && src.stride[1] == 1 && out.stride[1] == 1);
    const bool use_simd_v = packed && simd::use_avx2() && w >= 8;
    const bool use_simd_h = (out.itemsize == 2 && out.stride[1] == 1 && simd::use_avx2() && ow >= 16);

    for (int64_t oy = oy_start; oy < oy_stop; ++oy) {
        const int64_t y = 2 * oy;
        const int64_t sy0 = std::clamp<int64_t>(y - 1, 0, h - 1);
        const int64_t sy1 = std::clamp<int64_t>(y, 0, h - 1);
        const int64_t sy2 = std::clamp<int64_t>(y + 1, 0, h - 1);
        const int64_t sy3 = std::clamp<int64_t>(y + 2, 0, h - 1);
        if (packed) {
            const uint16_t* r0 = sb + sy0 * src.stride[0];
            const uint16_t* r1 = sb + sy1 * src.stride[0];
            const uint16_t* r2 = sb + sy2 * src.stride[0];
            const uint16_t* r3 = sb + sy3 * src.stride[0];
            if (use_simd_v) {
                simd::align_down_half_vert_row_avx2(r0, r1, r2, r3, scratch.data(), w);
            } else {
                for (int64_t x = 0; x < w; ++x) {
                    const uint32_t v = static_cast<uint32_t>(r0[x]) * kWeights[0]
                                     + static_cast<uint32_t>(r1[x]) * kWeights[1]
                                     + static_cast<uint32_t>(r2[x]) * kWeights[2]
                                     + static_cast<uint32_t>(r3[x]) * kWeights[3];
                    scratch[static_cast<size_t>(x)] = (v + 57u) / 115u;
                }
            }
        } else {
            for (int64_t x = 0; x < w; ++x) {
                const uint32_t v = read_u_val(src, sy0, x) * kWeights[0]
                                 + read_u_val(src, sy1, x) * kWeights[1]
                                 + read_u_val(src, sy2, x) * kWeights[2]
                                 + read_u_val(src, sy3, x) * kWeights[3];
                scratch[static_cast<size_t>(x)] = (v + 57u) / 115u;
            }
        }
        uint16_t* orow = ob + oy * out.stride[0];
        if (use_simd_h) {
            simd::align_down_half_horiz_row_avx2(scratch.data(), orow, ow, w);
        } else {
            for (int64_t ox = 0; ox < ow; ++ox) {
                const int64_t sx0 = std::clamp<int64_t>(2 * ox - 1, 0, w - 1);
                const int64_t sx1 = std::clamp<int64_t>(2 * ox, 0, w - 1);
                const int64_t sx2 = std::clamp<int64_t>(2 * ox + 1, 0, w - 1);
                const int64_t sx3 = std::clamp<int64_t>(2 * ox + 2, 0, w - 1);
                const uint32_t v = scratch[static_cast<size_t>(sx0)] * kWeights[0]
                                 + scratch[static_cast<size_t>(sx1)] * kWeights[1]
                                 + scratch[static_cast<size_t>(sx2)] * kWeights[2]
                                 + scratch[static_cast<size_t>(sx3)] * kWeights[3];
                orow[ox * out.stride[1]] = static_cast<uint16_t>((v + 57u) / 115u);
            }
        }
    }
}

void of_align_fast_downsize_u16_rows_cpp(
    const ArrayView& img, const ArrayView& out, int64_t oy_start, int64_t oy_stop
) {
    const int64_t h = img.shape[0];
    const int64_t w = img.shape[1];
    const int64_t ow = out.shape[1];
    const bool packed_simd = simd::use_avx2()
        && img.itemsize == 2 && out.itemsize == 2
        && img.stride[1] == 1 && out.stride[1] == 1 && ow >= 8;
    const uint16_t* ib = static_cast<const uint16_t*>(img.data);
    uint16_t* ob = static_cast<uint16_t*>(out.data);
    for (int64_t oy = oy_start; oy < oy_stop; ++oy) {
        const int64_t y0 = 2 * oy;
        const int64_t y1 = (y0 + 1 < h) ? (y0 + 1) : (h - 1);
        if (packed_simd) {
            simd::align_fast_downsize_u16_row_avx2(
                ib + y0 * img.stride[0], ib + y1 * img.stride[0],
                ob + oy * out.stride[0], ow, w);
        } else {
            for (int64_t ox = 0; ox < ow; ++ox) {
                const int64_t x0 = 2 * ox;
                const int64_t x1 = (x0 + 1 < w) ? (x0 + 1) : (w - 1);
                const uint32_t tot = (read_u_val(img, y0, x0) + read_u_val(img, y0, x1)
                                    + read_u_val(img, y1, x0) + read_u_val(img, y1, x1) + 2u) >> 2u;
                write_u_val(out, oy, ox, tot);
            }
        }
    }
}

void of_align_warp_int_rows_cpp(
    const ArrayView& image, const ArrayView& out, float dx, float dy,
    int64_t y_start, int64_t y_stop, bool bgr_rev
) {
    const int64_t h = image.shape[0];
    const int64_t w = image.shape[1];
    std::vector<int32_t> ix(static_cast<size_t>(w)), jx(static_cast<size_t>(w));
    std::vector<uint32_t> fx(static_cast<size_t>(w));
    std::vector<uint8_t> inv_x(static_cast<size_t>(w));
    for (int64_t x = 0; x < w; ++x) {
        bool inv = false;
        axis_sample(static_cast<float>(x) - dx, w, ix[static_cast<size_t>(x)],
                    jx[static_cast<size_t>(x)], fx[static_cast<size_t>(x)], inv);
        inv_x[static_cast<size_t>(x)] = inv ? 1 : 0;
    }
    const int64_t channels = (image.ndim == 2) ? 1 : image.shape[2];
    const bool packed = is_warp_packed(image, out, channels);
    for (int64_t y = y_start; y < y_stop; ++y) {
        int32_t iy = 0, jy = 0;
        uint32_t fy = 0;
        bool inv_y = false;
        axis_sample(static_cast<float>(y) - dy, h, iy, jy, fy, inv_y);
        if (inv_y) {
            if (packed) {
                std::memset(
                    static_cast<uint8_t*>(out.data) + y * out.stride[0] * out.itemsize,
                    0, static_cast<size_t>(w * channels * out.itemsize));
            } else {
                for (int64_t x = 0; x < w; ++x) {
                    for (int64_t c = 0; c < channels; ++c) {
                        write_u_val(out, y, x, 0, c);
                    }
                }
            }
            continue;
        }
        const uint32_t gy = 256u - fy;
        if (packed) {
            warp_translation_row_packed(
                image, out, iy, jy, y, w, channels, bgr_rev,
                ix.data(), jx.data(), fx.data(), inv_x.data(), gy, fy);
            continue;
        }
        for (int64_t x = 0; x < w; ++x) {
            if (inv_x[static_cast<size_t>(x)]) {
                for (int64_t c = 0; c < channels; ++c) {
                    write_u_val(out, y, x, 0, c);
                }
            } else {
                const uint32_t fxx = fx[static_cast<size_t>(x)];
                const uint32_t gxx = 256u - fxx;
                const int32_t ixx = ix[static_cast<size_t>(x)];
                const int32_t jxx = jx[static_cast<size_t>(x)];
                if (bgr_rev && channels == 3) {
                    write_u_val(out, y, x, bilinear_u32(image, iy, jy, ixx, jxx, gxx, fxx, gy, fy, 2), 0);
                    write_u_val(out, y, x, bilinear_u32(image, iy, jy, ixx, jxx, gxx, fxx, gy, fy, 1), 1);
                    write_u_val(out, y, x, bilinear_u32(image, iy, jy, ixx, jxx, gxx, fxx, gy, fy, 0), 2);
                } else {
                    for (int64_t c = 0; c < channels; ++c) {
                        write_u_val(out, y, x, bilinear_u32(image, iy, jy, ixx, jxx, gxx, fxx, gy, fy, c), c);
                    }
                }
            }
        }
    }
}

void of_align_warp_alignment_int_rows_cpp(
    const ArrayView& image, const ArrayView& out,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t y_start, int64_t y_stop, bool bgr_rev
) {
    const int64_t h = image.shape[0];
    const int64_t w = image.shape[1];
    const int64_t channels = (image.ndim == 2) ? 1 : image.shape[2];
    // The vector kernel reads and stores in a different order than the
    // per-channel loop, so an output that overlaps the image stays scalar.
    const bool use_simd = simd::use_avx2()
        && is_warp_packed(image, out, channels)
        && w >= 8 && w <= 16777216
        && h <= std::numeric_limits<int32_t>::max()
        && !packed_rows_overlap(image, 0, h, w * channels, out, y_start, y_stop, w * channels);
    for (int64_t y = y_start; y < y_stop; ++y) {
        float v_dy = 0.0f, u_dy = 0.0f;
        map_uv_row(static_cast<float>(y), cy, p1c, p0c, v_dy, u_dy);
        if (use_simd) {
            void* out_row = (out.itemsize == 1)
                ? static_cast<void*>(static_cast<uint8_t*>(out.data) + y * out.stride[0])
                : static_cast<void*>(static_cast<uint16_t*>(out.data) + y * out.stride[0]);
            simd::align_warp_alignment_row_avx2(
                image.data, image.stride[0], w, image.itemsize == 1,
                out_row, h, w, 0, 0, w, channels, bgr_rev,
                cx, p08, p18, tx, ty, v_dy, u_dy);
            continue;
        }
        for (int64_t x = 0; x < w; ++x) {
            float u = 0.0f, v = 0.0f;
            map_uv_col(static_cast<float>(x), cx, p08, p18, tx, ty, v_dy, u_dy, u, v);
            int32_t ix = 0, jx = 0, iy = 0, jy = 0;
            uint32_t fx = 0, fy = 0;
            bool inv_x = false, inv_y = false;
            axis_sample(u, w, ix, jx, fx, inv_x);
            axis_sample(v, h, iy, jy, fy, inv_y);
            if (inv_x || inv_y) {
                for (int64_t c = 0; c < channels; ++c) {
                    write_u_val(out, y, x, 0, c);
                }
            } else {
                const uint32_t gx = 256u - fx;
                const uint32_t gy = 256u - fy;
                if (bgr_rev && channels == 3) {
                    write_u_val(out, y, x, bilinear_u32(image, iy, jy, ix, jx, gx, fx, gy, fy, 2), 0);
                    write_u_val(out, y, x, bilinear_u32(image, iy, jy, ix, jx, gx, fx, gy, fy, 1), 1);
                    write_u_val(out, y, x, bilinear_u32(image, iy, jy, ix, jx, gx, fx, gy, fy, 0), 2);
                } else {
                    for (int64_t c = 0; c < channels; ++c) {
                        write_u_val(out, y, x, bilinear_u32(image, iy, jy, ix, jx, gx, fx, gy, fy, c), c);
                    }
                }
            }
        }
    }
}

void of_align_warp_alignment_float_rows_cpp(
    const ArrayView& image, const ArrayView& out,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t y_start, int64_t y_stop
) {
    const int64_t h = image.shape[0];
    const int64_t w = image.shape[1];
    const int64_t channels = (image.ndim == 2) ? 1 : image.shape[2];
    constexpr double kInvScale = 65536.0;
    for (int64_t y = y_start; y < y_stop; ++y) {
        float v_dy = 0.0f, u_dy = 0.0f;
        map_uv_row(static_cast<float>(y), cy, p1c, p0c, v_dy, u_dy);
        for (int64_t x = 0; x < w; ++x) {
            float u = 0.0f, v = 0.0f;
            map_uv_col(static_cast<float>(x), cx, p08, p18, tx, ty, v_dy, u_dy, u, v);
            int32_t ix = 0, jx = 0, iy = 0, jy = 0;
            uint32_t fx = 0, fy = 0;
            bool inv_x = false, inv_y = false;
            axis_sample(u, w, ix, jx, fx, inv_x);
            axis_sample(v, h, iy, jy, fy, inv_y);
            if (inv_x || inv_y) {
                for (int64_t c = 0; c < channels; ++c) {
                    write_float_val(out, y, x, 0.0, c);
                }
            } else {
                const double fgx = static_cast<double>(256u - fx);
                const double ffx = static_cast<double>(fx);
                const double fgy = static_cast<double>(256u - fy);
                const double ffy = static_cast<double>(fy);
                for (int64_t c = 0; c < channels; ++c) {
                    const double top = read_float_val(image, iy, ix, c) * fgx
                                     + read_float_val(image, iy, jx, c) * ffx;
                    const double bot = read_float_val(image, jy, ix, c) * fgx
                                     + read_float_val(image, jy, jx, c) * ffx;
                    write_float_val(out, y, x, (top * fgy + bot * ffy) / kInvScale, c);
                }
            }
        }
    }
}

void of_align_valid_mask_rows_cpp(
    uint8_t* mask, int64_t mask_sy, int64_t mask_sx, int64_t h, int64_t w,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t y_start, int64_t y_stop
) {
    const float hf = static_cast<float>(h);
    const float wf = static_cast<float>(w);
    const bool use_simd = simd::use_avx2() && mask_sx == 1 && w >= 8 && w <= 16777216;
    for (int64_t y = y_start; y < y_stop; ++y) {
        float v_dy = 0.0f, u_dy = 0.0f;
        map_uv_row(static_cast<float>(y), cy, p1c, p0c, v_dy, u_dy);
        uint8_t* mr = mask + y * mask_sy;
        if (use_simd) {
            simd::align_valid_mask_row_avx2(mr, w, cx, p08, p18, tx, ty, v_dy, u_dy, wf, hf);
        } else {
            for (int64_t x = 0; x < w; ++x) {
                float u = 0.0f, v = 0.0f;
                map_uv_col(static_cast<float>(x), cx, p08, p18, tx, ty, v_dy, u_dy, u, v);
                mr[x * mask_sx] = (u >= 0.0f && u < wf && v >= 0.0f && v < hf) ? 1 : 0;
            }
        }
    }
}

std::tuple<float, float, bool> of_align_refine_level_cpp(
    const ArrayView& a, const ArrayView& b,
    float c_dx, float c_dy, double half, double step, int64_t crop
) {
    const int64_t h = a.shape[0];
    const int64_t w = a.shape[1];
    const int32_t pad = static_cast<int32_t>(std::ceil(std::max<double>(
        {std::fabs(static_cast<double>(c_dx)), std::fabs(static_cast<double>(c_dy)), 1.0}) + half)) + 1;
    const int64_t n = std::min({crop, h, w});
    const int64_t cy = (h - n) / 2;
    const int64_t cx = (w - n) / 2;
    const int32_t count = static_cast<int32_t>(std::llround(2.0 * half / step)) | 1;
    const int32_t centre = (count - 1) / 2;
    const int32_t total_cand = count * count;
    const double total_px = static_cast<double>(n * n);
    const uint16_t* ab = static_cast<const uint16_t*>(a.data);
    const uint16_t* bb = static_cast<const uint16_t*>(b.data);
    const int64_t a_sx = a.stride[1];
    const int64_t b_sx = b.stride[1];

    std::vector<double> scores(static_cast<size_t>(total_cand));
    std::vector<float> dxs(static_cast<size_t>(total_cand));
    std::vector<float> dys(static_cast<size_t>(total_cand));

    const int threads = (total_cand >= 4 && n * n >= 4096) ? 4 : 1;
    // Residuals are integers below 2^16, so while n * n * 65535^2 stays under
    // 2^53 the double sums below are exact and integer sums give the same value.
    const bool exact_integer_sums = a_sx == 1 && b_sx == 1 && n >= 8
        && n * n <= (int64_t{1} << 21) && simd::use_avx2();
    parallel_bands(total_cand, 1, threads, [&](int64_t s0, int64_t s1) {
        std::vector<int32_t> ix_arr(static_cast<size_t>(n)), jx_arr(static_cast<size_t>(n));
        std::vector<uint32_t> fx_arr(static_cast<size_t>(n));
        std::vector<uint8_t> inv_x_arr(static_cast<size_t>(n));
        std::vector<int32_t> weights(exact_integer_sums ? static_cast<size_t>(n) : size_t{0});
        for (int64_t idx = s0; idx < s1; ++idx) {
            const int32_t oy = static_cast<int32_t>(idx / count);
            const int32_t ox = static_cast<int32_t>(idx % count);
            const float ody = static_cast<float>(static_cast<double>(oy - centre) * step);
            const float dy = c_dy + ody;
            const float odx = static_cast<float>(static_cast<double>(ox - centre) * step);
            const float dx = c_dx + odx;
            for (int64_t xx = 0; xx < n; ++xx) {
                const float xcoord = (static_cast<float>(cx + xx) + static_cast<float>(pad)) - dx;
                bool inv = false;
                axis_pad_sample(xcoord, w, pad, ix_arr[static_cast<size_t>(xx)],
                                jx_arr[static_cast<size_t>(xx)], fx_arr[static_cast<size_t>(xx)], inv);
                inv_x_arr[static_cast<size_t>(xx)] = inv ? 1 : 0;
            }
            double ssum = 0.0, ssq = 0.0;
            if (exact_integer_sums) {
                // The longest run of valid columns that sample consecutive
                // sources goes through the vector kernel, eight at a time.
                int64_t run_lo = 0;
                while (run_lo < n && inv_x_arr[static_cast<size_t>(run_lo)]) {
                    ++run_lo;
                }
                int64_t run_hi = run_lo;
                while (run_hi < n && !inv_x_arr[static_cast<size_t>(run_hi)]
                       && jx_arr[static_cast<size_t>(run_hi)] == ix_arr[static_cast<size_t>(run_hi)] + 1
                       && ix_arr[static_cast<size_t>(run_hi)]
                           == ix_arr[static_cast<size_t>(run_lo)] + static_cast<int32_t>(run_hi - run_lo)) {
                    weights[static_cast<size_t>(run_hi)] = static_cast<int32_t>(
                        (256u - fx_arr[static_cast<size_t>(run_hi)]) | (fx_arr[static_cast<size_t>(run_hi)] << 16u));
                    ++run_hi;
                }
                run_hi = run_lo + ((run_hi - run_lo) & ~int64_t{7});
                int64_t isum = 0, isq = 0;
                for (int64_t yy = cy; yy < cy + n; ++yy) {
                    const float ycoord = (static_cast<float>(yy) + static_cast<float>(pad)) - dy;
                    int32_t iy = 0, jy = 0;
                    uint32_t fy = 0;
                    bool inv_y = false;
                    axis_pad_sample(ycoord, h, pad, iy, jy, fy, inv_y);
                    const uint32_t gy = 256u - fy;
                    const uint16_t* ar = ab + yy * a.stride[0] + cx;
                    const uint16_t* br0 = bb + iy * b.stride[0];
                    const uint16_t* br1 = bb + jy * b.stride[0];
                    const auto plain_columns = [&](int64_t first, int64_t last) {
                        for (int64_t xx_i = first; xx_i < last; ++xx_i) {
                            uint32_t sample = 0;
                            if (!inv_y && !inv_x_arr[static_cast<size_t>(xx_i)]) {
                                const uint32_t fx = fx_arr[static_cast<size_t>(xx_i)];
                                const uint32_t gx = 256u - fx;
                                const int32_t ix = ix_arr[static_cast<size_t>(xx_i)];
                                const int32_t jx = jx_arr[static_cast<size_t>(xx_i)];
                                const uint32_t top = static_cast<uint32_t>(br0[ix]) * gx + static_cast<uint32_t>(br0[jx]) * fx;
                                const uint32_t bot = static_cast<uint32_t>(br1[ix]) * gx + static_cast<uint32_t>(br1[jx]) * fx;
                                sample = (top * gy + bot * fy) >> 16u;
                            }
                            const int64_t diff = static_cast<int64_t>(ar[xx_i]) - static_cast<int64_t>(sample);
                            isum += diff;
                            isq += diff * diff;
                        }
                    };
                    if (inv_y || run_hi <= run_lo) {
                        plain_columns(0, n);
                        continue;
                    }
                    plain_columns(0, run_lo);
                    const int32_t first_source = ix_arr[static_cast<size_t>(run_lo)];
                    simd::align_refine_dense_row_avx2(
                        ar + run_lo, br0 + first_source, br1 + first_source,
                        weights.data() + run_lo, run_hi - run_lo, fy, &isum, &isq);
                    plain_columns(run_hi, n);
                }
                ssum = static_cast<double>(isum);
                ssq = static_cast<double>(isq);
            }
            for (int64_t yy = cy; !exact_integer_sums && yy < cy + n; ++yy) {
                const float ycoord = (static_cast<float>(yy) + static_cast<float>(pad)) - dy;
                int32_t iy = 0, jy = 0;
                uint32_t fy = 0;
                bool inv_y = false;
                axis_pad_sample(ycoord, h, pad, iy, jy, fy, inv_y);
                const uint32_t gy = 256u - fy;
                const uint16_t* ar = ab + yy * a.stride[0];
                const uint16_t* br0 = bb + iy * b.stride[0];
                const uint16_t* br1 = bb + jy * b.stride[0];
                if (a_sx == 1 && b_sx == 1) {
                    for (int64_t xx_i = 0; xx_i < n; ++xx_i) {
                        uint32_t sample = 0;
                        if (!inv_y && !inv_x_arr[static_cast<size_t>(xx_i)]) {
                            const uint32_t fx = fx_arr[static_cast<size_t>(xx_i)];
                            const uint32_t gx = 256u - fx;
                            const int32_t ix = ix_arr[static_cast<size_t>(xx_i)];
                            const int32_t jx = jx_arr[static_cast<size_t>(xx_i)];
                            const uint32_t top = static_cast<uint32_t>(br0[ix]) * gx + static_cast<uint32_t>(br0[jx]) * fx;
                            const uint32_t bot = static_cast<uint32_t>(br1[ix]) * gx + static_cast<uint32_t>(br1[jx]) * fx;
                            sample = (top * gy + bot * fy) >> 16u;
                        }
                        const double diff = static_cast<double>(ar[cx + xx_i]) - static_cast<double>(sample);
                        ssum += diff;
                        ssq += diff * diff;
                    }
                } else {
                    for (int64_t xx_i = 0; xx_i < n; ++xx_i) {
                        uint32_t sample = 0;
                        if (!inv_y && !inv_x_arr[static_cast<size_t>(xx_i)]) {
                            const uint32_t fx = fx_arr[static_cast<size_t>(xx_i)];
                            const uint32_t gx = 256u - fx;
                            const int64_t ix = static_cast<int64_t>(ix_arr[static_cast<size_t>(xx_i)]) * b_sx;
                            const int64_t jx = static_cast<int64_t>(jx_arr[static_cast<size_t>(xx_i)]) * b_sx;
                            const uint32_t top = static_cast<uint32_t>(br0[ix]) * gx + static_cast<uint32_t>(br0[jx]) * fx;
                            const uint32_t bot = static_cast<uint32_t>(br1[ix]) * gx + static_cast<uint32_t>(br1[jx]) * fx;
                            sample = (top * gy + bot * fy) >> 16u;
                        }
                        const double diff = static_cast<double>(ar[(cx + xx_i) * a_sx]) - static_cast<double>(sample);
                        ssum += diff;
                        ssq += diff * diff;
                    }
                }
            }
            const double mean = ssum / total_px;
            const double var = std::max(ssq / total_px - mean * mean, 0.0);
            scores[static_cast<size_t>(idx)] = std::sqrt(var);
            dxs[static_cast<size_t>(idx)] = dx;
            dys[static_cast<size_t>(idx)] = dy;
        }
    });

    float best_score = std::numeric_limits<float>::max();
    float best_dx = 0.0f, best_dy = 0.0f;
    bool found = false;
    for (int32_t idx = 0; idx < total_cand; ++idx) {
        const double sc = scores[static_cast<size_t>(idx)];
        if (sc < static_cast<double>(best_score)) {
            best_score = static_cast<float>(sc);
            best_dx = dxs[static_cast<size_t>(idx)];
            best_dy = dys[static_cast<size_t>(idx)];
            found = true;
        }
    }
    return {best_dx, best_dy, found};
}

template <bool PackedRefs>
inline void accumulate_grid_score_block(
    const uint16_t* rb, int64_t ref_sy, int64_t ref_sx,
    const uint16_t* bb, int64_t b_sy, int64_t b_sx,
    int64_t h, int64_t w, float hf, float wf,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    float tx_k, float ty_k, float cx_k, float cy_k,
    float p08_k, float p0c_k, float p18_k, float p1c_k,
    int64_t& acc_sum, int64_t& acc_sq, int64_t& acc_cnt
) {
    int64_t sum = 0, sq = 0, count = 0;
    for (int64_t iy_i = 0; iy_i < nyy; ++iy_i) {
        float v_dy = 0.0f, u_dy = 0.0f;
        map_uv_row(ys[iy_i], cy_k, p1c_k, p0c_k, v_dy, u_dy);
        const uint16_t* rrow = rb + iy_i * ref_sy;
        for (int64_t ix_i = 0; ix_i < nxx; ++ix_i) {
            float u = 0.0f, v = 0.0f;
            map_uv_col(xs[ix_i], cx_k, p08_k, p18_k, tx_k, ty_k, v_dy, u_dy, u, v);
            if (!(u >= 0.0f && u < wf && v >= 0.0f && v < hf)) {
                continue;
            }
            int32_t ix = 0, jx = 0, iy = 0, jy = 0;
            uint32_t fx = 0, fy = 0;
            bool inv_x = false, inv_y = false;
            axis_sample(u, w, ix, jx, fx, inv_x);
            axis_sample(v, h, iy, jy, fy, inv_y);
            const uint32_t gx = 256u - fx;
            const uint32_t gy = 256u - fy;
            const uint16_t* br0 = bb + iy * b_sy;
            const uint16_t* br1 = bb + jy * b_sy;
            uint32_t val;
            int64_t ref_val;
            if constexpr (PackedRefs) {
                const uint32_t top = static_cast<uint32_t>(br0[ix]) * gx + static_cast<uint32_t>(br0[jx]) * fx;
                const uint32_t bot = static_cast<uint32_t>(br1[ix]) * gx + static_cast<uint32_t>(br1[jx]) * fx;
                val = (top * gy + bot * fy) >> 16u;
                ref_val = static_cast<int64_t>(rrow[ix_i]);
            } else {
                const int64_t oix = static_cast<int64_t>(ix) * b_sx;
                const int64_t ojx = static_cast<int64_t>(jx) * b_sx;
                const uint32_t top = static_cast<uint32_t>(br0[oix]) * gx + static_cast<uint32_t>(br0[ojx]) * fx;
                const uint32_t bot = static_cast<uint32_t>(br1[oix]) * gx + static_cast<uint32_t>(br1[ojx]) * fx;
                val = (top * gy + bot * fy) >> 16u;
                ref_val = static_cast<int64_t>(rrow[ix_i * ref_sx]);
            }
            const int64_t diff = ref_val - static_cast<int64_t>(val);
            sum += diff;
            sq += diff * diff;
            count += 1;
        }
    }
    acc_sum = sum;
    acc_sq = sq;
    acc_cnt = count;
}

void of_align_score_chunk_u16_slice_cpp(
    const ArrayView& ref, const ArrayView& img_b,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    const float* p08, const float* p0c, const float* p18, const float* p1c,
    const float* tx, const float* ty, const float* cx, const float* cy,
    const int64_t* idx, int64_t k_start, int64_t k_stop,
    int64_t* ssum, int64_t* ssq, int64_t* cnt
) {
    const int64_t h = img_b.shape[0];
    const int64_t w = img_b.shape[1];
    const float hf = static_cast<float>(h);
    const float wf = static_cast<float>(w);
    const uint16_t* rb = static_cast<const uint16_t*>(ref.data);
    const uint16_t* bb = static_cast<const uint16_t*>(img_b.data);
    const bool packed_refs = (ref.stride[1] == 1 && img_b.stride[1] == 1);

    if (packed_refs && simd::use_avx2() && nxx >= 8
            && h <= std::numeric_limits<int32_t>::max()
            && w <= std::numeric_limits<int32_t>::max()) {
        simd::align_score_chunk_packed_avx2(
            rb, ref.stride[0], bb, img_b.stride[0], h, w, xs, nxx, ys, nyy,
            p08, p0c, p18, p1c, tx, ty, cx, cy, idx, k_start, k_stop, ssum, ssq, cnt);
        return;
    }

    for (int64_t k = k_start; k < k_stop; ++k) {
        const int64_t ci = idx[k];
        int64_t acc_sum = 0, acc_sq = 0, acc_cnt = 0;
        if (packed_refs) {
            accumulate_grid_score_block<true>(
                rb, ref.stride[0], 1, bb, img_b.stride[0], 1,
                h, w, hf, wf, xs, nxx, ys, nyy,
                tx[ci], ty[ci], cx[ci], cy[ci],
                p08[ci], p0c[ci], p18[ci], p1c[ci],
                acc_sum, acc_sq, acc_cnt);
        } else {
            accumulate_grid_score_block<false>(
                rb, ref.stride[0], ref.stride[1], bb, img_b.stride[0], img_b.stride[1],
                h, w, hf, wf, xs, nxx, ys, nyy,
                tx[ci], ty[ci], cx[ci], cy[ci],
                p08[ci], p0c[ci], p18[ci], p1c[ci],
                acc_sum, acc_sq, acc_cnt);
        }
        ssum[ci] = acc_sum;
        ssq[ci] = acc_sq;
        cnt[ci] = acc_cnt;
    }
}

void of_align_score_grid_4d_slice_cpp(
    const ArrayView& ref, const ArrayView& img_b,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    float tx0, float ty0,
    const float* cx_tab, const float* cy_tab,
    const ArrayView& p08_tab, const ArrayView& p18_tab,
    int64_t n0, int64_t n1, int64_t n2, int64_t n3,
    int64_t k_start, int64_t k_stop,
    int64_t* out_best_k, float* out_best_sc, int64_t slot
) {
    const int64_t h = img_b.shape[0];
    const int64_t w = img_b.shape[1];
    const float hf = static_cast<float>(h);
    const float wf = static_cast<float>(w);
    const float x_lo = xs[0], x_hi = xs[nxx - 1];
    const float y_lo = ys[0], y_hi = ys[nyy - 1];
    const uint16_t* rb = static_cast<const uint16_t*>(ref.data);
    const uint16_t* bb = static_cast<const uint16_t*>(img_b.data);
    const float* p08b = static_cast<const float*>(p08_tab.data);
    const float* p18b = static_cast<const float*>(p18_tab.data);
    const bool packed_refs = (ref.stride[1] == 1 && img_b.stride[1] == 1);

    if (packed_refs && simd::use_avx2() && nxx >= 8
            && h <= std::numeric_limits<int32_t>::max()
            && w <= std::numeric_limits<int32_t>::max()) {
        simd::align_score_grid_4d_packed_avx2(
            rb, ref.stride[0], bb, img_b.stride[0], h, w, xs, nxx, ys, nyy,
            tx0, ty0, cx_tab, cy_tab,
            p08b, p08_tab.stride[0], p08_tab.stride[1],
            p18b, p18_tab.stride[0], p18_tab.stride[1],
            n0, n1, n2, n3, k_start, k_stop, out_best_k, out_best_sc, slot);
        return;
    }

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
        const float p08_k = p08b[t3 * p08_tab.stride[0] + t2 * p08_tab.stride[1]];
        const float p18_k = p18b[t3 * p18_tab.stride[0] + t2 * p18_tab.stride[1]];
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
        if (packed_refs) {
            accumulate_grid_score_block<true>(
                rb, ref.stride[0], 1, bb, img_b.stride[0], 1,
                h, w, hf, wf, xs, nxx, ys, nyy,
                tx0, ty0, cx_k, cy_k,
                p08_k, p0c_k, p18_k, p1c_k,
                acc_sum, acc_sq, acc_cnt);
        } else {
            accumulate_grid_score_block<false>(
                rb, ref.stride[0], ref.stride[1], bb, img_b.stride[0], img_b.stride[1],
                h, w, hf, wf, xs, nxx, ys, nyy,
                tx0, ty0, cx_k, cy_k,
                p08_k, p0c_k, p18_k, p1c_k,
                acc_sum, acc_sq, acc_cnt);
        }
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

bool of_align_perspective_has_rounding_edge_cpp(
    const double* inverse, int64_t x0, int64_t x1, int64_t y0, int64_t y1,
    double bins, double tolerance
) {
    for (int64_t y = y0; y < y1; ++y) {
        const double yd = static_cast<double>(y);
        const double nx0 = inverse[1] * yd + inverse[2];
        const double ny0 = inverse[4] * yd + inverse[5];
        const double denom0 = inverse[7] * yd + inverse[8];
        for (int64_t x = x0; x < x1; ++x) {
            const double xd = static_cast<double>(x);
            const double scale = bins / (denom0 + inverse[6] * xd);
            const double nx = (nx0 + inverse[0] * xd) * scale;
            const double ny = (ny0 + inverse[3] * xd) * scale;
            if (std::fabs(nx - std::floor(nx) - 0.5) <= tolerance
                    || std::fabs(ny - std::floor(ny) - 0.5) <= tolerance) {
                return true;
            }
        }
    }
    return false;
}

void of_align_warp_alignment_region_int_rows_cpp(
    const ArrayView& image, const ArrayView& out,
    int64_t src_h, int64_t src_w, int64_t src_y0,
    int64_t x_off, int64_t y_off,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t oy_start, int64_t oy_stop, bool bgr_rev
) {
    const int64_t rw = out.shape[1];
    const int64_t channels = (out.ndim == 2) ? 1 : out.shape[2];
    const bool use_simd = simd::use_avx2()
        && is_warp_packed(image, out, channels)
        && rw >= 8
        && src_h <= std::numeric_limits<int32_t>::max()
        && src_w <= std::numeric_limits<int32_t>::max()
        && x_off >= -16777216 && x_off <= 16777216 - rw
        && !packed_rows_overlap(image, 0, image.shape[0], std::max(image.shape[1], src_w) * channels,
                                out, oy_start, oy_stop, rw * channels);
    for (int64_t oy = oy_start; oy < oy_stop; ++oy) {
        float v_dy = 0.0f, u_dy = 0.0f;
        map_uv_row(static_cast<float>(y_off + oy), cy, p1c, p0c, v_dy, u_dy);
        if (use_simd) {
            void* out_row = (out.itemsize == 1)
                ? static_cast<void*>(static_cast<uint8_t*>(out.data) + oy * out.stride[0])
                : static_cast<void*>(static_cast<uint16_t*>(out.data) + oy * out.stride[0]);
            simd::align_warp_alignment_row_avx2(
                image.data, image.stride[0], image.shape[1], image.itemsize == 1,
                out_row, src_h, src_w, src_y0, x_off, rw, channels, bgr_rev,
                cx, p08, p18, tx, ty, v_dy, u_dy);
            continue;
        }
        for (int64_t ox = 0; ox < rw; ++ox) {
            float u = 0.0f, v = 0.0f;
            map_uv_col(static_cast<float>(x_off + ox), cx, p08, p18, tx, ty, v_dy, u_dy, u, v);
            int32_t ix = 0, jx = 0, iy = 0, jy = 0;
            uint32_t fx = 0, fy = 0;
            bool inv_x = false, inv_y = false;
            axis_sample(u, src_w, ix, jx, fx, inv_x);
            axis_sample(v, src_h, iy, jy, fy, inv_y);
            if (inv_x || inv_y) {
                for (int64_t c = 0; c < channels; ++c) {
                    write_u_val(out, oy, ox, 0, c);
                }
            } else {
                const uint32_t gx = 256u - fx;
                const uint32_t gy = 256u - fy;
                const int64_t iy_b = iy - src_y0;
                const int64_t jy_b = jy - src_y0;
                if (bgr_rev && channels == 3) {
                    write_u_val(out, oy, ox, bilinear_u32(image, iy_b, jy_b, ix, jx, gx, fx, gy, fy, 2), 0);
                    write_u_val(out, oy, ox, bilinear_u32(image, iy_b, jy_b, ix, jx, gx, fx, gy, fy, 1), 1);
                    write_u_val(out, oy, ox, bilinear_u32(image, iy_b, jy_b, ix, jx, gx, fx, gy, fy, 0), 2);
                } else {
                    for (int64_t c = 0; c < channels; ++c) {
                        write_u_val(out, oy, ox, bilinear_u32(image, iy_b, jy_b, ix, jx, gx, fx, gy, fy, c), c);
                    }
                }
            }
        }
    }
}

void of_align_warp_alignment_region_float_rows_cpp(
    const ArrayView& image, const ArrayView& out,
    int64_t src_h, int64_t src_w, int64_t src_y0,
    int64_t x_off, int64_t y_off,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t oy_start, int64_t oy_stop
) {
    const int64_t rw = out.shape[1];
    const int64_t channels = (out.ndim == 2) ? 1 : out.shape[2];
    constexpr double kInvScale = 65536.0;
    for (int64_t oy = oy_start; oy < oy_stop; ++oy) {
        float v_dy = 0.0f, u_dy = 0.0f;
        map_uv_row(static_cast<float>(y_off + oy), cy, p1c, p0c, v_dy, u_dy);
        for (int64_t ox = 0; ox < rw; ++ox) {
            float u = 0.0f, v = 0.0f;
            map_uv_col(static_cast<float>(x_off + ox), cx, p08, p18, tx, ty, v_dy, u_dy, u, v);
            int32_t ix = 0, jx = 0, iy = 0, jy = 0;
            uint32_t fx = 0, fy = 0;
            bool inv_x = false, inv_y = false;
            axis_sample(u, src_w, ix, jx, fx, inv_x);
            axis_sample(v, src_h, iy, jy, fy, inv_y);
            if (inv_x || inv_y) {
                for (int64_t c = 0; c < channels; ++c) {
                    write_float_val(out, oy, ox, 0.0, c);
                }
            } else {
                const double fgx = static_cast<double>(256u - fx);
                const double ffx = static_cast<double>(fx);
                const double fgy = static_cast<double>(256u - fy);
                const double ffy = static_cast<double>(fy);
                const int64_t iy_b = iy - src_y0;
                const int64_t jy_b = jy - src_y0;
                for (int64_t c = 0; c < channels; ++c) {
                    const double top = read_float_val(image, iy_b, ix, c) * fgx
                                     + read_float_val(image, iy_b, jx, c) * ffx;
                    const double bot = read_float_val(image, jy_b, ix, c) * fgx
                                     + read_float_val(image, jy_b, jx, c) * ffx;
                    write_float_val(out, oy, ox, (top * fgy + bot * ffy) / kInvScale, c);
                }
            }
        }
    }
}

void of_align_warp_translation_region_int_rows_cpp(
    const ArrayView& image, const ArrayView& out,
    int64_t src_h, int64_t src_w, int64_t src_y0,
    int64_t x_off, int64_t y_off,
    float dx, float dy,
    int64_t oy_start, int64_t oy_stop, bool bgr_rev
) {
    const int64_t rw = out.shape[1];
    std::vector<int32_t> ix(static_cast<size_t>(rw)), jx(static_cast<size_t>(rw));
    std::vector<uint32_t> fx(static_cast<size_t>(rw));
    std::vector<uint8_t> inv_x(static_cast<size_t>(rw));
    for (int64_t ox = 0; ox < rw; ++ox) {
        bool inv = false;
        axis_sample(static_cast<float>(x_off + ox) - dx, src_w, ix[static_cast<size_t>(ox)],
                    jx[static_cast<size_t>(ox)], fx[static_cast<size_t>(ox)], inv);
        inv_x[static_cast<size_t>(ox)] = inv ? 1 : 0;
    }
    const int64_t channels = (out.ndim == 2) ? 1 : out.shape[2];
    const bool packed = is_warp_packed(image, out, channels);
    for (int64_t oy = oy_start; oy < oy_stop; ++oy) {
        int32_t iy = 0, jy = 0;
        uint32_t fy = 0;
        bool inv_y = false;
        axis_sample(static_cast<float>(y_off + oy) - dy, src_h, iy, jy, fy, inv_y);
        if (inv_y) {
            if (packed) {
                std::memset(
                    static_cast<uint8_t*>(out.data) + oy * out.stride[0] * out.itemsize,
                    0, static_cast<size_t>(rw * channels * out.itemsize));
            } else {
                for (int64_t ox = 0; ox < rw; ++ox) {
                    for (int64_t c = 0; c < channels; ++c) {
                        write_u_val(out, oy, ox, 0, c);
                    }
                }
            }
            continue;
        }
        const uint32_t gy = 256u - fy;
        const int64_t iy_b = iy - src_y0;
        const int64_t jy_b = jy - src_y0;
        if (packed) {
            warp_translation_row_packed(
                image, out, iy_b, jy_b, oy, rw, channels, bgr_rev,
                ix.data(), jx.data(), fx.data(), inv_x.data(), gy, fy);
            continue;
        }
        for (int64_t ox = 0; ox < rw; ++ox) {
            if (inv_x[static_cast<size_t>(ox)]) {
                for (int64_t c = 0; c < channels; ++c) {
                    write_u_val(out, oy, ox, 0, c);
                }
            } else {
                const uint32_t fxx = fx[static_cast<size_t>(ox)];
                const uint32_t gxx = 256u - fxx;
                const int32_t ixx = ix[static_cast<size_t>(ox)];
                const int32_t jxx = jx[static_cast<size_t>(ox)];
                if (bgr_rev && channels == 3) {
                    write_u_val(out, oy, ox, bilinear_u32(image, iy_b, jy_b, ixx, jxx, gxx, fxx, gy, fy, 2), 0);
                    write_u_val(out, oy, ox, bilinear_u32(image, iy_b, jy_b, ixx, jxx, gxx, fxx, gy, fy, 1), 1);
                    write_u_val(out, oy, ox, bilinear_u32(image, iy_b, jy_b, ixx, jxx, gxx, fxx, gy, fy, 0), 2);
                } else {
                    for (int64_t c = 0; c < channels; ++c) {
                        write_u_val(out, oy, ox, bilinear_u32(image, iy_b, jy_b, ixx, jxx, gxx, fxx, gy, fy, c), c);
                    }
                }
            }
        }
    }
}

}  // namespace openfocus
