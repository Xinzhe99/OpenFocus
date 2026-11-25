#include "openfocus_ab.hpp"
#include "simd_dispatch.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace openfocus {
namespace {

constexpr int64_t kAMaxResponse = 65535 * 16;
constexpr int64_t kABlendLimit = 65536;
constexpr int kASaturated = 0;
constexpr int kASharp = 1;
constexpr int kABackground = 2;

constexpr std::array<int64_t, 101> make_activity_reciprocals() {
    std::array<int64_t, 101> arr{};
    for (int i = 0; i < 101; ++i) {
        const int64_t d = 10 + i;
        arr[static_cast<size_t>(i)] = ((int64_t{1} << 32) + d - 1) / d;
    }
    return arr;
}

constexpr std::array<int64_t, 101> kActivityReciprocals = make_activity_reciprocals();

inline uint16_t normalized_diff_u16(int64_t value, int64_t mean) {
    const int64_t maximum = value > mean ? value : mean;
    const int64_t denom = (maximum * 100LL * int64_t{0x80008001LL}) >> 47;
    const int64_t num = 110LL * std::abs(value - mean);
    const int64_t res = (num * kActivityReciprocals[static_cast<size_t>(denom)]) >> 32;
    return static_cast<uint16_t>(res < 65535LL ? res : 65535LL);
}

inline int64_t read_int(const ArrayView& v, int64_t y, int64_t x, int64_t c = 0) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 1) {
        return static_cast<const uint8_t*>(v.data)[off];
    }
    if (v.itemsize == 2) {
        return static_cast<const uint16_t*>(v.data)[off];
    }
    return static_cast<const int32_t*>(v.data)[off];
}

inline void write_int(const ArrayView& v, int64_t y, int64_t x, int64_t val, int64_t c = 0) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 1) {
        static_cast<uint8_t*>(v.data)[off] = static_cast<uint8_t>(val);
    } else if (v.itemsize == 2) {
        static_cast<uint16_t*>(v.data)[off] = static_cast<uint16_t>(val);
    } else {
        static_cast<int32_t*>(v.data)[off] = static_cast<int32_t>(val);
    }
}

inline int64_t read_2d(const ArrayView& v, int64_t y, int64_t x) {
    const int64_t off = y * v.stride[0] + x * v.stride[1];
    return (v.itemsize == 2)
        ? static_cast<int64_t>(static_cast<const uint16_t*>(v.data)[off])
        : static_cast<int64_t>(static_cast<const int32_t*>(v.data)[off]);
}

inline void write_2d(const ArrayView& v, int64_t y, int64_t x, int64_t val) {
    const int64_t off = y * v.stride[0] + x * v.stride[1];
    if (v.itemsize == 2) {
        static_cast<uint16_t*>(v.data)[off] = static_cast<uint16_t>(val);
    } else {
        static_cast<int32_t*>(v.data)[off] = static_cast<int32_t>(val);
    }
}

void burt_laplacian_row_impl(const ArrayView& gray, const ArrayView& reduced,
                             const ArrayView& output, int64_t y) {
    const int64_t height = reduced.shape[0];
    const int64_t width = reduced.shape[1];
    const int64_t fine_width = gray.shape[1];
    if (fine_width == 0) {
        return;
    }
    int64_t y0 = (y / 2) - ((y % 2 == 0) ? 1 : 0);
    const int64_t y1 = std::min(y0 + 1, height - 1);
    y0 = std::max<int64_t>(y0, 0);
    const int64_t wy0 = (y % 2 == 0) ? 1 : 3;
    const int64_t wy1 = 4 - wy0;

    if (gray.itemsize == 4 && reduced.itemsize == 4 && output.itemsize == 4
            && gray.stride[1] == 1 && reduced.stride[1] == 1 && output.stride[1] == 1
            && simd::use_avx2() && width >= 10) {
        const int32_t* g_row = static_cast<const int32_t*>(gray.data) + y * gray.stride[0];
        const int32_t* r0 = static_cast<const int32_t*>(reduced.data) + y0 * reduced.stride[0];
        const int32_t* r1 = static_cast<const int32_t*>(reduced.data) + y1 * reduced.stride[0];
        int32_t* o_row = static_cast<int32_t*>(output.data) + y * output.stride[0];
        simd::burt_laplacian_row_i32_avx2(g_row, r0, r1, o_row, width, fine_width,
                                          static_cast<int32_t>(wy0), static_cast<int32_t>(wy1));
        return;
    }

    int64_t left = read_2d(reduced, y0, 0) * wy0 + read_2d(reduced, y1, 0) * wy1;
    write_2d(output, y, 0, std::abs(read_2d(gray, y, 0) * 16 - left * 4));
    for (int64_t x = 1; x < width; ++x) {
        const int64_t right = read_2d(reduced, y0, x) * wy0 + read_2d(reduced, y1, x) * wy1;
        const int64_t odd = 2 * x - 1;
        write_2d(output, y, odd, std::abs(read_2d(gray, y, odd) * 16 - (3 * left + right)));
        write_2d(output, y, odd + 1, std::abs(read_2d(gray, y, odd + 1) * 16 - (left + 3 * right)));
        left = right;
    }
    if (fine_width % 2 == 0) {
        write_2d(output, y, fine_width - 1,
                 std::abs(read_2d(gray, y, fine_width - 1) * 16 - left * 4));
    }
}

inline void double_single_row_u32(
    const int32_t* laps, int64_t off, int64_t width, int level,
    uint32_t* out, uint32_t* tmp0, uint32_t* tmp1, int64_t n_out
) {
    const int steps = level - 1;
    uint32_t* src = tmp0;
    uint32_t* dst = tmp1;
    for (int64_t x = 0; x < width; ++x) {
        src[x] = static_cast<uint32_t>(laps[off + x]);
    }
    int64_t cur_len = width;
    const int64_t max_pairs = n_out / 2 + 1;
    const int64_t cap = n_out + 2;
    for (int s = 0; s < steps; ++s) {
        int64_t limit = cur_len - 1;
        if (limit > max_pairs) {
            limit = max_pairs;
        }
        for (int64_t x = 0; x < limit; ++x) {
            const uint32_t a = src[x];
            const uint32_t b = src[x + 1];
            dst[2 * x] = a + a;
            dst[2 * x + 1] = a + b;
        }
        const int64_t last_idx = 2 * limit;
        if (last_idx < cap) {
            const uint32_t a = src[limit];
            dst[last_idx] = a + a;
            cur_len = last_idx + 1;
        } else {
            cur_len = cap;
        }
        std::swap(src, dst);
    }
    const int64_t pad = int64_t{1} << (level - 2);
    const uint32_t v_first = src[0];
    const uint32_t v_last = src[cur_len - 1];
    const int64_t left_end = std::min(pad, n_out);
    for (int64_t j = 0; j < left_end; ++j) {
        out[j] = v_first;
    }
    const int64_t copy_end = std::min(pad + cur_len, n_out);
    for (int64_t j = left_end; j < copy_end; ++j) {
        out[j] = src[j - pad];
    }
    for (int64_t j = copy_end; j < n_out; ++j) {
        out[j] = v_last;
    }
}

inline void update_level_pair_u32(
    const int32_t* laps, int64_t off, int64_t width, int level,
    int64_t ya, int64_t yb, int64_t& cur_ya, int64_t& cur_yb,
    uint32_t* da, uint32_t* db, uint32_t* tmp0, uint32_t* tmp1, int64_t n_out
) {
    if (ya == cur_ya && yb == cur_yb) {
        return;
    }
    if (ya == cur_yb) {
        std::memcpy(da, db, static_cast<size_t>(n_out) * sizeof(uint32_t));
    } else if (ya != cur_ya) {
        double_single_row_u32(laps, off + ya * width, width, level, da, tmp0, tmp1, n_out);
    }
    if (yb == ya) {
        std::memcpy(db, da, static_cast<size_t>(n_out) * sizeof(uint32_t));
    } else if (yb != cur_yb) {
        double_single_row_u32(laps, off + yb * width, width, level, db, tmp0, tmp1, n_out);
    }
    cur_ya = ya;
    cur_yb = yb;
}

inline void burt_accumulate_row_i64(
    const int32_t* laps, int64_t offset, int64_t height, int64_t width,
    int32_t* response, int64_t fine_width, int level, int64_t y
) {
    if (fine_width == 0) {
        return;
    }
    const int shift = level + 1;
    const int64_t factor = int64_t{1} << level;
    const int64_t den = int64_t{1} << shift;
    const int64_t py = 2 * y + 1 - factor;
    const int64_t y0 = py >> shift;
    const int64_t fy = py - (y0 << shift);
    const int64_t row_a = offset + std::clamp<int64_t>(y0, 0, height - 1) * width;
    const int64_t row_b = offset + std::clamp<int64_t>(y0 + 1, 0, height - 1) * width;
    int64_t left = static_cast<int64_t>(laps[row_a]) * (den - fy)
                 + static_cast<int64_t>(laps[row_b]) * fy;
    if (level == 1) {
        response[0] += static_cast<int32_t>((left + 2) >> 2);
        for (int64_t x = 1; x < width; ++x) {
            const int64_t right = static_cast<int64_t>(laps[row_a + x]) * (den - fy)
                                + static_cast<int64_t>(laps[row_b + x]) * fy;
            const int64_t odd = 2 * x - 1;
            response[odd] += static_cast<int32_t>((3 * left + right + 8) >> 4);
            response[odd + 1] += static_cast<int32_t>((left + 3 * right + 8) >> 4);
            left = right;
        }
        if (fine_width % 2 == 0) {
            response[fine_width - 1] += static_cast<int32_t>((left + 2) >> 2);
        }
        return;
    }
    int64_t end = std::min(factor >> 1, fine_width);
    int64_t value = (left + factor) >> shift;
    for (int64_t x = 0; x < end; ++x) {
        response[x] += static_cast<int32_t>(value);
    }
    const int64_t rounding = int64_t{1} << (2 * shift - 1);
    for (int64_t col = 1; col < width; ++col) {
        const int64_t right = static_cast<int64_t>(laps[row_a + col]) * (den - fy)
                            + static_cast<int64_t>(laps[row_b + col]) * fy;
        const int64_t delta = right - left;
        const int64_t initial = (left << shift) + delta + rounding;
        const int64_t span = std::min(factor, fine_width - end);
        for (int64_t dx = 0; dx < span; ++dx) {
            response[end + dx] += static_cast<int32_t>((initial + 2 * delta * dx) >> (2 * shift));
        }
        end += span;
        left = right;
    }
    value = (left + factor) >> shift;
    for (int64_t x = end; x < fine_width; ++x) {
        response[x] += static_cast<int32_t>(value);
    }
}

inline int64_t reflect_idx(int64_t pos, int64_t len) {
    if (len <= 1) {
        return 0;
    }
    const int64_t period = 2 * (len - 1);
    int64_t p = pos % period;
    if (p < 0) {
        p += period;
    }
    return (p >= len) ? (period - p) : p;
}

inline uint16_t store_f32_code(float acc) {
    const float rounded = acc + 0.5f;
    if (rounded <= 0.0f) {
        return 0;
    }
    if (rounded >= 65535.0f) {
        return 65535;
    }
    return static_cast<uint16_t>(static_cast<int32_t>(rounded));
}

template <typename SrcT, typename DstT>
void gray_rgb3_packed_row(
    const SrcT* sr, DstT* dr, int64_t width, int64_t s1, int64_t s2,
    int64_t scale, bool depth_color
) {
    if (depth_color) {
        const uint32_t s32 = static_cast<uint32_t>(scale * 85);
        for (int64_t x = 0; x < width; ++x) {
            const SrcT* p = sr + x * s1;
            const uint32_t total = static_cast<uint32_t>(p[0])
                                 + static_cast<uint32_t>(p[s2])
                                 + static_cast<uint32_t>(p[2 * s2]);
            dr[x] = static_cast<DstT>((total * s32) >> 8u);
        }
    } else {
        for (int64_t x = 0; x < width; ++x) {
            const SrcT* p = sr + x * s1;
            const uint32_t total = static_cast<uint32_t>(p[0])
                                 + static_cast<uint32_t>(p[s2])
                                 + static_cast<uint32_t>(p[2 * s2]);
            dr[x] = static_cast<DstT>(static_cast<int64_t>(total / 3u) * scale);
        }
    }
}

template <typename SrcT, typename DstT>
void merge_recip_packed_rows(
    const ArrayView& accumulator, const ArrayView& image,
    const uint16_t* depth, int64_t d_sy,
    int64_t base, int64_t step, int64_t start, int64_t stop, int32_t s32,
    bool src_packed, bool dst_packed, bool use_avx2_skip
) {
    const int64_t width = accumulator.shape[1];
    const uint64_t recip48 = ((uint64_t{1} << 48) + static_cast<uint64_t>(step) - 1u) / static_cast<uint64_t>(step);
    const int32_t base32 = static_cast<int32_t>(base);
    const int32_t step32 = static_cast<int32_t>(step);
    const int32_t neg_step = -step32;
    const SrcT* ib = static_cast<const SrcT*>(image.data);
    DstT* ab = static_cast<DstT*>(accumulator.data);
    const int64_t is1 = src_packed ? 3 : image.stride[1];
    const int64_t is2 = src_packed ? 1 : image.stride[2];
    const int64_t as1 = dst_packed ? 3 : accumulator.stride[1];
    const int64_t as2 = dst_packed ? 1 : accumulator.stride[2];

    for (int64_t y = start; y < stop; ++y) {
        const uint16_t* drow = depth + y * d_sy;
        const SrcT* irow = ib + y * image.stride[0];
        DstT* arow = ab + y * accumulator.stride[0];
        if (use_avx2_skip) {
            simd::merge_recip_packed_row_avx2(
                drow, irow, sizeof(SrcT) == 1, arow, sizeof(DstT) == 2,
                width, is1, is2, as1, as2, base32, step32, recip48, s32);
            continue;
        }
        for (int64_t x = 0; x < width; ++x) {
            const int32_t delta = static_cast<int32_t>(drow[x]) - base32;
            if (delta >= 0) {
                if (delta <= step32) {
                    const SrcT* sp = irow + x * is1;
                    DstT* dp = arow + x * as1;
                    if (src_packed && dst_packed) {
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
                    if (src_packed && dst_packed) {
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
int select_a_no_bg_packed_row(
    int32_t* b_row, const int32_t* response, int64_t width,
    const SrcT* irow, DstT* orow, int64_t is1, int64_t is2, int64_t os1, int64_t os2,
    int64_t scale, int state, bool use_avx2
) {
    if (use_avx2) {
        return simd::select_a_no_bg_packed_row_avx2(
            b_row, response, width, irow, sizeof(SrcT) == 1, orow, sizeof(DstT) == 2,
            is1, is2, os1, os2, scale, state, kAMaxResponse, kASharp, kASaturated);
    }
    for (int64_t x = 0; x < width; ++x) {
        const int64_t prev = b_row[x];
        if (prev == kAMaxResponse) {
            continue;
        }
        int64_t sc = response[x];
        if (sc >= kAMaxResponse) {
            sc = kAMaxResponse;
        } else if (state == kASaturated) {
            state = kASharp;
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
    return state;
}

}  // namespace

void of_gray_rows_cpp(const ArrayView& image, const ArrayView& output,
                      int64_t start, int64_t stop, int64_t scale, bool depth_color) {
    if (start >= stop || image.shape[0] == 0 || image.shape[1] == 0) {
        return;
    }
    const int64_t width = image.shape[1];
    const int ndim = image.ndim;
    const int64_t channels = ndim == 2 ? 1 : image.shape[2];

    if (ndim == 2 || channels == 1) {
        for (int64_t y = start; y < stop; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                write_int(output, y, x, read_int(image, y, x, 0) * scale);
            }
        }
        return;
    }
    if (channels >= 3) {
        const bool can_packed = (channels == 3 && !image.is_float && !output.is_float
                                 && output.ndim == 2 && output.stride[1] == 1
                                 && (image.itemsize == 1 || image.itemsize == 2)
                                 && (output.itemsize == 2 || output.itemsize == 4)
                                 && image.stride[1] == 3 && (image.stride[2] == 1 || image.stride[2] == -1)
                                 && (!depth_color || (scale >= 0 && scale <= 256))
                                 && (scale >= 0 && scale <= 65535));
        if (can_packed) {
            const bool contiguous_rgb = (image.stride[2] == 1);
            const bool avx2_ok = contiguous_rgb && simd::use_avx2() && width >= 8;
            const bool out_u16 = (output.itemsize == 2);
            for (int64_t y = start; y < stop; ++y) {
                if (image.itemsize == 1) {
                    const uint8_t* sr = static_cast<const uint8_t*>(image.data) + y * image.stride[0];
                    if (out_u16) {
                        uint16_t* dr = static_cast<uint16_t*>(output.data) + y * output.stride[0];
                        if (avx2_ok) {
                            simd::gray_rgb_u8_row_avx2(sr, dr, nullptr, width, static_cast<int32_t>(scale), depth_color, true);
                        } else {
                            gray_rgb3_packed_row(sr, dr, width, image.stride[1], image.stride[2], scale, depth_color);
                        }
                    } else {
                        int32_t* dr = static_cast<int32_t*>(output.data) + y * output.stride[0];
                        if (avx2_ok) {
                            simd::gray_rgb_u8_row_avx2(sr, nullptr, dr, width, static_cast<int32_t>(scale), depth_color, false);
                        } else {
                            gray_rgb3_packed_row(sr, dr, width, image.stride[1], image.stride[2], scale, depth_color);
                        }
                    }
                } else {
                    const uint16_t* sr = static_cast<const uint16_t*>(image.data) + y * image.stride[0];
                    if (out_u16) {
                        uint16_t* dr = static_cast<uint16_t*>(output.data) + y * output.stride[0];
                        if (avx2_ok) {
                            simd::gray_rgb_u16_row_avx2(sr, dr, nullptr, width, static_cast<int32_t>(scale), depth_color, true);
                        } else {
                            gray_rgb3_packed_row(sr, dr, width, image.stride[1], image.stride[2], scale, depth_color);
                        }
                    } else {
                        int32_t* dr = static_cast<int32_t*>(output.data) + y * output.stride[0];
                        if (avx2_ok) {
                            simd::gray_rgb_u16_row_avx2(sr, nullptr, dr, width, static_cast<int32_t>(scale), depth_color, false);
                        } else {
                            gray_rgb3_packed_row(sr, dr, width, image.stride[1], image.stride[2], scale, depth_color);
                        }
                    }
                }
            }
            return;
        }
        if (depth_color && (image.itemsize == 1 || image.itemsize == 2) && scale >= 0 && scale <= 256) {
            const uint32_t s32 = static_cast<uint32_t>(scale * 85);
            for (int64_t y = start; y < stop; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const uint32_t total = static_cast<uint32_t>(read_int(image, y, x, 0))
                                         + static_cast<uint32_t>(read_int(image, y, x, 1))
                                         + static_cast<uint32_t>(read_int(image, y, x, 2));
                    write_int(output, y, x, (total * s32) >> 8u);
                }
            }
        } else if (depth_color) {
            for (int64_t y = start; y < stop; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const int64_t total = read_int(image, y, x, 0)
                                        + read_int(image, y, x, 1)
                                        + read_int(image, y, x, 2);
                    write_int(output, y, x, (total * scale * 85) >> 8);
                }
            }
        } else {
            for (int64_t y = start; y < stop; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    const uint32_t total = static_cast<uint32_t>(read_int(image, y, x, 0))
                                         + static_cast<uint32_t>(read_int(image, y, x, 1))
                                         + static_cast<uint32_t>(read_int(image, y, x, 2));
                    write_int(output, y, x, static_cast<int64_t>(total / 3u) * scale);
                }
            }
        }
        return;
    }
    for (int64_t y = start; y < stop; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            const int64_t val = (read_int(image, y, x, 0) + read_int(image, y, x, 1)) / 3;
            write_int(output, y, x, val * scale);
        }
    }
}

void of_down2_rows_cpp(const ArrayView& image, const ArrayView& output,
                       int64_t start, int64_t stop, int64_t scale) {
    if (start >= stop || output.shape[0] == 0 || output.shape[1] == 0) {
        return;
    }
    const int64_t height = image.shape[0];
    const int64_t width = image.shape[1];
    const int64_t out_w = output.shape[1];
    const int64_t full_w = width / 2;

    if (image.ndim == 2) {
        if (!image.is_float && !output.is_float && output.ndim == 2
                && image.stride[1] == 1 && output.stride[1] == 1
                && (image.itemsize == 2 || image.itemsize == 4)
                && output.itemsize == 4 && scale >= 0 && scale <= 256) {
            const bool avx2_ok = simd::use_avx2() && full_w >= 8;
            int32_t* ob = static_cast<int32_t*>(output.data);
            for (int64_t y = start; y < stop; ++y) {
                const int64_t top = 2 * y;
                const int64_t bottom = (top + 1 < height) ? (top + 1) : (height - 1);
                int32_t* dst = ob + y * output.stride[0];
                if (image.itemsize == 2) {
                    const uint16_t* ib = static_cast<const uint16_t*>(image.data);
                    const uint16_t* r0 = ib + top * image.stride[0];
                    const uint16_t* r1 = ib + bottom * image.stride[0];
                    if (avx2_ok) {
                        simd::down2_row_u16_to_i32_avx2(r0, r1, dst, full_w, static_cast<int32_t>(scale));
                    } else {
                        for (int64_t x = 0; x < full_w; ++x) {
                            const int64_t left = 2 * x;
                            const int32_t tot = static_cast<int32_t>(r0[left]) + static_cast<int32_t>(r0[left + 1])
                                              + static_cast<int32_t>(r1[left]) + static_cast<int32_t>(r1[left + 1]);
                            dst[x] = (tot * static_cast<int32_t>(scale)) >> 2;
                        }
                    }
                    for (int64_t x = full_w; x < out_w; ++x) {
                        const int64_t left = 2 * x;
                        const int64_t right = (left + 1 < width) ? (left + 1) : (width - 1);
                        const int32_t tot = static_cast<int32_t>(r0[left]) + static_cast<int32_t>(r0[right])
                                          + static_cast<int32_t>(r1[left]) + static_cast<int32_t>(r1[right]);
                        dst[x] = (tot * static_cast<int32_t>(scale)) >> 2;
                    }
                } else {
                    const int32_t* ib = static_cast<const int32_t*>(image.data);
                    const int32_t* r0 = ib + top * image.stride[0];
                    const int32_t* r1 = ib + bottom * image.stride[0];
                    // Keep the sum and scaled product wide for int32 inputs.
                    for (int64_t x = 0; x < full_w; ++x) {
                        const int64_t left = 2 * x;
                        const int64_t tot = static_cast<int64_t>(r0[left]) + r0[left + 1]
                                          + r1[left] + r1[left + 1];
                        dst[x] = static_cast<int32_t>((tot * scale) >> 2);
                    }
                    for (int64_t x = full_w; x < out_w; ++x) {
                        const int64_t left = 2 * x;
                        const int64_t right = (left + 1 < width) ? (left + 1) : (width - 1);
                        const int64_t tot = static_cast<int64_t>(r0[left]) + r0[right]
                                          + r1[left] + r1[right];
                        dst[x] = static_cast<int32_t>((tot * scale) >> 2);
                    }
                }
            }
            return;
        }
        for (int64_t y = start; y < stop; ++y) {
            const int64_t top = 2 * y;
            const int64_t bottom = (top + 1 < height) ? (top + 1) : (height - 1);
            for (int64_t x = 0; x < full_w; ++x) {
                const int64_t left = 2 * x;
                const int64_t tot = read_int(image, top, left) + read_int(image, top, left + 1)
                                  + read_int(image, bottom, left) + read_int(image, bottom, left + 1);
                write_int(output, y, x, (tot * scale) >> 2);
            }
            for (int64_t x = full_w; x < out_w; ++x) {
                const int64_t left = 2 * x;
                const int64_t right = (left + 1 < width) ? (left + 1) : (width - 1);
                const int64_t tot = read_int(image, top, left) + read_int(image, top, right)
                                  + read_int(image, bottom, left) + read_int(image, bottom, right);
                write_int(output, y, x, (tot * scale) >> 2);
            }
        }
        return;
    }
    const int64_t channels = image.shape[2];
    for (int64_t y = start; y < stop; ++y) {
        const int64_t top = 2 * y;
        const int64_t bottom = (top + 1 < height) ? (top + 1) : (height - 1);
        for (int64_t x = 0; x < full_w; ++x) {
            const int64_t left = 2 * x;
            for (int64_t ch = 0; ch < channels; ++ch) {
                const int64_t tot = read_int(image, top, left, ch) + read_int(image, top, left + 1, ch)
                                  + read_int(image, bottom, left, ch) + read_int(image, bottom, left + 1, ch);
                write_int(output, y, x, (tot * scale) >> 2, ch);
            }
        }
        for (int64_t x = full_w; x < out_w; ++x) {
            const int64_t left = 2 * x;
            const int64_t right = (left + 1 < width) ? (left + 1) : (width - 1);
            for (int64_t ch = 0; ch < channels; ++ch) {
                const int64_t tot = read_int(image, top, left, ch) + read_int(image, top, right, ch)
                                  + read_int(image, bottom, left, ch) + read_int(image, bottom, right, ch);
                write_int(output, y, x, (tot * scale) >> 2, ch);
            }
        }
    }
}

void of_burt_laplacian_rows_cpp(const ArrayView& gray, const ArrayView& reduced,
                                const ArrayView& output, int64_t start, int64_t stop) {
    for (int64_t y = start; y < stop; ++y) {
        burt_laplacian_row_impl(gray, reduced, output, y);
    }
}

int of_select_rows_a_cpp(
    const ArrayView& image,
    const ArrayView& gray,
    const ArrayView& half_gray,
    const ArrayView& laplacians,
    int levels,
    const ArrayView& best,
    const ArrayView& output,
    int64_t start,
    int64_t stop,
    int64_t scale,
    bool first,
    const ArrayView& background
) {
    const int64_t width = gray.shape[1];
    if (start >= stop || width == 0) {
        return kASaturated;
    }

    const int64_t h0 = half_gray.shape[0];
    const int64_t w0 = half_gray.shape[1];
    const int32_t* laps = static_cast<const int32_t*>(laplacians.data);
    const int64_t n_out = (width + 1) / 2 + 1;
    const int64_t full_pairs = width / 2;
    const bool odd_width = (width & 1) == 1;

    struct SelectAScratch {
        std::vector<int32_t> response;
        std::vector<uint32_t> m0, m1, m2, m3, m4, m5;
        std::vector<uint32_t> da2, db2, da3, db3, da4, db4, da5, db5, tmp0, tmp1;
    };
    thread_local SelectAScratch scratch;

    auto& response = scratch.response;
    if (response.size() < static_cast<size_t>(width)) {
        response.resize(static_cast<size_t>(width));
    }

    const bool use_fast_pyramid = (levels >= 6 && h0 >= 1 && w0 >= 1 && half_gray.itemsize == 4
                                   && gray.stride[1] == 1 && half_gray.stride[1] == 1);

    int state = kASaturated;
    const int64_t channels = (image.ndim == 2) ? 1 : image.shape[2];
    const int64_t weight_col = channels;
    const int64_t stat_col = channels + 1;
    const bool has_bg = background.shape[0] > 0;
    const bool no_bg = (!first) && !has_bg;

    auto& m0 = scratch.m0;
    auto& m1 = scratch.m1;
    auto& m2 = scratch.m2;
    auto& m3 = scratch.m3;
    auto& m4 = scratch.m4;
    auto& m5 = scratch.m5;
    auto& da2 = scratch.da2;
    auto& db2 = scratch.db2;
    auto& da3 = scratch.da3;
    auto& db3 = scratch.db3;
    auto& da4 = scratch.da4;
    auto& db4 = scratch.db4;
    auto& da5 = scratch.da5;
    auto& db5 = scratch.db5;
    auto& tmp0 = scratch.tmp0;
    auto& tmp1 = scratch.tmp1;
    int64_t off1 = 0, off2 = 0, off3 = 0, off4 = 0, off5 = 0, off6 = 0;
    int64_t h2 = 0, w2 = 0, h3 = 0, w3 = 0, h4 = 0, w4 = 0, h5 = 0, w5 = 0, h6 = 0, w6 = 0;
    int64_t cya2 = -1, cyb2 = -1, cya3 = -1, cyb3 = -1, cya4 = -1, cyb4 = -1, cya5 = -1, cyb5 = -1;

    if (use_fast_pyramid) {
        auto ensure = [](std::vector<uint32_t>& v, size_t n) {
            if (v.size() < n) {
                v.resize(n);
            }
        };
        ensure(m0, static_cast<size_t>(w0 + 2));
        ensure(m1, static_cast<size_t>(w0 + 2));
        ensure(m2, static_cast<size_t>(n_out));
        ensure(m3, static_cast<size_t>(n_out));
        ensure(m4, static_cast<size_t>(n_out));
        ensure(m5, static_cast<size_t>(n_out));
        ensure(da2, static_cast<size_t>(n_out));
        ensure(db2, static_cast<size_t>(n_out));
        ensure(da3, static_cast<size_t>(n_out));
        ensure(db3, static_cast<size_t>(n_out));
        ensure(da4, static_cast<size_t>(n_out));
        ensure(db4, static_cast<size_t>(n_out));
        ensure(da5, static_cast<size_t>(n_out));
        ensure(db5, static_cast<size_t>(n_out));
        ensure(tmp0, static_cast<size_t>(n_out + 4));
        ensure(tmp1, static_cast<size_t>(n_out + 4));

        off1 = 0;
        h2 = (h0 + 1) / 2; w2 = (w0 + 1) / 2;
        off2 = off1 + h0 * w0;
        h3 = (h2 + 1) / 2; w3 = (w2 + 1) / 2;
        off3 = off2 + h2 * w2;
        h4 = (h3 + 1) / 2; w4 = (w3 + 1) / 2;
        off4 = off3 + h3 * w3;
        h5 = (h4 + 1) / 2; w5 = (w4 + 1) / 2;
        off5 = off4 + h4 * w4;
        h6 = (h5 + 1) / 2; w6 = (w5 + 1) / 2;
        off6 = off5 + h5 * w5;
    }

    int32_t* best_data = static_cast<int32_t*>(best.data);
    float* bg_f32 = static_cast<float*>(background.data);
    uint32_t* bg_u32 = static_cast<uint32_t*>(background.data);

    for (int64_t y = start; y < stop; ++y) {
        if (use_fast_pyramid) {
            int64_t y0 = (y / 2) - ((y & 1) == 0 ? 1 : 0);
            int64_t y1 = (y0 + 1 < h0) ? (y0 + 1) : (h0 - 1);
            if (y0 < 0) {
                y0 = 0;
            }
            const uint32_t wy0 = ((y & 1) == 0) ? 1u : 3u;
            const uint32_t wy1 = 4u - wy0;
            const int32_t* r0 = static_cast<const int32_t*>(half_gray.data) + y0 * half_gray.stride[0];
            const int32_t* r1 = static_cast<const int32_t*>(half_gray.data) + y1 * half_gray.stride[0];
            if (simd::use_avx2() && w0 >= 8) {
                simd::select_a_blend_pair_i32_avx2(r0, r1, m0.data() + 1, w0, wy0, wy1);
            } else {
                for (int64_t x = 0; x < w0; ++x) {
                    m0[x + 1] = static_cast<uint32_t>(r0[x]) * wy0 + static_cast<uint32_t>(r1[x]) * wy1;
                }
            }
            m0[0] = m0[1];
            m0[w0 + 1] = m0[w0];

            {
                const int64_t py = 2 * y - 1;
                const int64_t cy0 = py >> 2;
                const uint32_t fy1 = static_cast<uint32_t>(py - (cy0 << 2));
                const uint32_t wy_1 = 4u - fy1;
                const int64_t ya = std::clamp<int64_t>(cy0, 0, h0 - 1);
                const int64_t yb = std::clamp<int64_t>(cy0 + 1, 0, h0 - 1);
                const int32_t* ra = laps + off1 + ya * w0;
                const int32_t* rb = laps + off1 + yb * w0;
                if (simd::use_avx2() && w0 >= 8) {
                    simd::select_a_blend_pair_i32_avx2(ra, rb, m1.data() + 1, w0, wy_1, fy1);
                } else {
                    for (int64_t x = 0; x < w0; ++x) {
                        m1[x + 1] = static_cast<uint32_t>(ra[x]) * wy_1 + static_cast<uint32_t>(rb[x]) * fy1;
                    }
                }
                m1[0] = m1[1];
                m1[w0 + 1] = m1[w0];
            }

            uint32_t fy2, wy2, fy3, wy3, fy4, wy4, fy5, wy5;
            {
                const int64_t py = 2 * y - 3;
                const int64_t cy0 = py >> 3;
                fy2 = static_cast<uint32_t>(py - (cy0 << 3));
                wy2 = 8u - fy2;
                update_level_pair_u32(laps, off2, w2, 2,
                                      std::clamp<int64_t>(cy0, 0, h2 - 1),
                                      std::clamp<int64_t>(cy0 + 1, 0, h2 - 1),
                                      cya2, cyb2, da2.data(), db2.data(), tmp0.data(), tmp1.data(), n_out);
            }
            {
                const int64_t py = 2 * y - 7;
                const int64_t cy0 = py >> 4;
                fy3 = static_cast<uint32_t>(py - (cy0 << 4));
                wy3 = 16u - fy3;
                update_level_pair_u32(laps, off3, w3, 3,
                                      std::clamp<int64_t>(cy0, 0, h3 - 1),
                                      std::clamp<int64_t>(cy0 + 1, 0, h3 - 1),
                                      cya3, cyb3, da3.data(), db3.data(), tmp0.data(), tmp1.data(), n_out);
            }
            {
                const int64_t py = 2 * y - 15;
                const int64_t cy0 = py >> 5;
                fy4 = static_cast<uint32_t>(py - (cy0 << 5));
                wy4 = 32u - fy4;
                update_level_pair_u32(laps, off4, w4, 4,
                                      std::clamp<int64_t>(cy0, 0, h4 - 1),
                                      std::clamp<int64_t>(cy0 + 1, 0, h4 - 1),
                                      cya4, cyb4, da4.data(), db4.data(), tmp0.data(), tmp1.data(), n_out);
            }
            {
                const int64_t py = 2 * y - 31;
                const int64_t cy0 = py >> 6;
                fy5 = static_cast<uint32_t>(py - (cy0 << 6));
                wy5 = 64u - fy5;
                update_level_pair_u32(laps, off5, w5, 5,
                                      std::clamp<int64_t>(cy0, 0, h5 - 1),
                                      std::clamp<int64_t>(cy0 + 1, 0, h5 - 1),
                                      cya5, cyb5, da5.data(), db5.data(), tmp0.data(), tmp1.data(), n_out);
            }

            if (simd::use_avx2() && n_out >= 8) {
                simd::select_a_combine_m2_to_m5_avx2(
                    da2.data(), db2.data(), m2.data(), wy2, fy2,
                    da3.data(), db3.data(), m3.data(), wy3, fy3,
                    da4.data(), db4.data(), m4.data(), wy4, fy4,
                    da5.data(), db5.data(), m5.data(), wy5, fy5,
                    n_out);
            } else {
                for (int64_t j = 0; j < n_out; ++j) {
                    m2[j] = da2[j] * wy2 + db2[j] * fy2;
                    m3[j] = da3[j] * wy3 + db3[j] * fy3;
                    m4[j] = da4[j] * wy4 + db4[j] * fy4;
                    m5[j] = da5[j] * wy5 + db5[j] * fy5;
                }
            }

            const void* g_ptr = (gray.itemsize == 2)
                ? static_cast<const void*>(static_cast<const uint16_t*>(gray.data) + y * gray.stride[0])
                : static_cast<const void*>(static_cast<const int32_t*>(gray.data) + y * gray.stride[0]);
            auto read_g = [&](int64_t x) -> uint32_t {
                return (gray.itemsize == 2)
                    ? static_cast<uint32_t>(static_cast<const uint16_t*>(g_ptr)[x])
                    : static_cast<uint32_t>(static_cast<const int32_t*>(g_ptr)[x]);
            };

            if (simd::use_avx2() && full_pairs >= 8) {
                simd::select_a_response_pairs_avx2(
                    g_ptr, gray.itemsize == 2,
                    m0.data(), m1.data(), m2.data(), m3.data(), m4.data(), m5.data(),
                    response.data(), full_pairs);
            } else {
                for (int64_t j = 0; j < full_pairs; ++j) {
                    const int64_t ev = 2 * j;
                    const uint32_t a0_l = m0[j], a0_c = m0[j + 1], a0_r = m0[j + 2];
                    const uint32_t a1_l = m1[j], a1_c = m1[j + 1], a1_r = m1[j + 2];
                    const int32_t d0 = static_cast<int32_t>(read_g(ev) << 4u)
                                     - static_cast<int32_t>(a0_l + a0_c * 3u);
                    const int32_t d1 = static_cast<int32_t>(read_g(ev + 1) << 4u)
                                     - static_cast<int32_t>(a0_c * 3u + a0_r);
                    int32_t s0 = (d0 < 0) ? -d0 : d0;
                    int32_t s1 = (d1 < 0) ? -d1 : d1;
                    s0 += static_cast<int32_t>((a1_l + a1_c * 3u + 8u) >> 4u);
                    s1 += static_cast<int32_t>((a1_c * 3u + a1_r + 8u) >> 4u);
                    const uint32_t l2 = m2[j], r2 = m2[j + 1];
                    const uint32_t l3 = m3[j], r3 = m3[j + 1];
                    const uint32_t l4 = m4[j], r4 = m4[j + 1];
                    const uint32_t l5 = m5[j], r5 = m5[j + 1];
                    s0 += static_cast<int32_t>((l2 * 3u + r2 + 32u) >> 6u);
                    s1 += static_cast<int32_t>((l2 + r2 * 3u + 32u) >> 6u);
                    s0 += static_cast<int32_t>((l3 * 3u + r3 + 128u) >> 8u);
                    s1 += static_cast<int32_t>((l3 + r3 * 3u + 128u) >> 8u);
                    s0 += static_cast<int32_t>((l4 * 3u + r4 + 512u) >> 10u);
                    s1 += static_cast<int32_t>((l4 + r4 * 3u + 512u) >> 10u);
                    s0 += static_cast<int32_t>((l5 * 3u + r5 + 2048u) >> 12u);
                    s1 += static_cast<int32_t>((l5 + r5 * 3u + 2048u) >> 12u);
                    response[ev] = s0;
                    response[ev + 1] = s1;
                }
            }
            if (odd_width) {
                const int64_t j = full_pairs;
                const int64_t ev = 2 * j;
                const int32_t d0 = static_cast<int32_t>(read_g(ev) << 4u)
                                 - static_cast<int32_t>(m0[j] + m0[j + 1] * 3u);
                int32_t s0 = (d0 < 0) ? -d0 : d0;
                s0 += static_cast<int32_t>((m1[j] + m1[j + 1] * 3u + 8u) >> 4u);
                s0 += static_cast<int32_t>((m2[j] * 3u + m2[j + 1] + 32u) >> 6u);
                s0 += static_cast<int32_t>((m3[j] * 3u + m3[j + 1] + 128u) >> 8u);
                s0 += static_cast<int32_t>((m4[j] * 3u + m4[j + 1] + 512u) >> 10u);
                s0 += static_cast<int32_t>((m5[j] * 3u + m5[j + 1] + 2048u) >> 12u);
                response[ev] = s0;
            }
            if (levels > 6) {
                int64_t hh = h6, ww = w6, off = off6;
                for (int lv = 6; lv < levels; ++lv) {
                    burt_accumulate_row_i64(laps, off, hh, ww, response.data(), width, lv, y);
                    off += hh * ww;
                    hh = (hh + 1) / 2;
                    ww = (ww + 1) / 2;
                }
            }
        } else {
            ArrayView row_view{};
            row_view.data = response.data();
            row_view.shape[0] = 1;
            row_view.shape[1] = width;
            row_view.stride[0] = 0;
            row_view.stride[1] = 1;
            row_view.ndim = 2;
            row_view.itemsize = 4;
            burt_laplacian_row_impl(gray, half_gray, row_view, y);
            int64_t hh = h0, ww = w0, off = 0;
            for (int lv = 1; lv < levels; ++lv) {
                burt_accumulate_row_i64(laps, off, hh, ww, response.data(), width, lv, y);
                off += hh * ww;
                hh = (hh + 1) / 2;
                ww = (ww + 1) / 2;
            }
        }

        int32_t* b_row = best_data + y * best.stride[0];
        const int64_t local_y = y - start;
        if (no_bg) {
            if (channels == 3 && image.ndim == 3 && output.ndim == 3
                    && !image.is_float && !output.is_float
                    && (image.itemsize == 1 || image.itemsize == 2)
                    && (output.itemsize == 2 || output.itemsize == 4)) {
                const bool avx2_skip = simd::use_avx2() && width >= 8;
                const int64_t is1 = image.stride[1], is2 = image.stride[2];
                const int64_t os1 = output.stride[1], os2 = output.stride[2];
                if (image.itemsize == 1 && output.itemsize == 2) {
                    state = select_a_no_bg_packed_row<uint8_t, uint16_t>(
                        b_row, response.data(), width,
                        static_cast<const uint8_t*>(image.data) + y * image.stride[0],
                        static_cast<uint16_t*>(output.data) + y * output.stride[0],
                        is1, is2, os1, os2, scale, state, avx2_skip);
                } else if (image.itemsize == 1 && output.itemsize == 4) {
                    state = select_a_no_bg_packed_row<uint8_t, int32_t>(
                        b_row, response.data(), width,
                        static_cast<const uint8_t*>(image.data) + y * image.stride[0],
                        static_cast<int32_t*>(output.data) + y * output.stride[0],
                        is1, is2, os1, os2, scale, state, avx2_skip);
                } else if (image.itemsize == 2 && output.itemsize == 2) {
                    state = select_a_no_bg_packed_row<uint16_t, uint16_t>(
                        b_row, response.data(), width,
                        static_cast<const uint16_t*>(image.data) + y * image.stride[0],
                        static_cast<uint16_t*>(output.data) + y * output.stride[0],
                        is1, is2, os1, os2, scale, state, avx2_skip);
                } else {
                    state = select_a_no_bg_packed_row<uint16_t, int32_t>(
                        b_row, response.data(), width,
                        static_cast<const uint16_t*>(image.data) + y * image.stride[0],
                        static_cast<int32_t*>(output.data) + y * output.stride[0],
                        is1, is2, os1, os2, scale, state, avx2_skip);
                }
                continue;
            }
            for (int64_t x = 0; x < width; ++x) {
                const int64_t prev = b_row[x];
                if (prev == kAMaxResponse) {
                    continue;
                }
                int64_t sc = response[x];
                if (sc >= kAMaxResponse) {
                    sc = kAMaxResponse;
                } else if (state == kASaturated) {
                    state = kASharp;
                }
                if (sc > prev) {
                    b_row[x] = static_cast<int32_t>(sc);
                    if (channels == 1 && output.ndim == 2) {
                        write_int(output, y, x, read_int(image, y, x, 0) * scale);
                    } else {
                        for (int64_t ch = 0; ch < channels; ++ch) {
                            write_int(output, y, x, read_int(image, y, x, ch) * scale, ch);
                        }
                    }
                }
            }
        } else {
            float* bg_row = bg_f32 + local_y * background.stride[0];
            uint32_t* st_row = bg_u32 + local_y * background.stride[0];
            const int64_t bsx = background.stride[1];
            const int64_t bsc = background.stride[2];
            if (channels == 3 && image.ndim == 3 && output.ndim == 3
                    && !image.is_float && !output.is_float
                    && (image.itemsize == 1 || image.itemsize == 2)
                    && (output.itemsize == 2 || output.itemsize == 4)
                    && bsx == 5 && bsc == 1
                    && simd::use_avx2() && width >= 8) {
                const void* irow = (image.itemsize == 1)
                    ? static_cast<const void*>(static_cast<const uint8_t*>(image.data) + y * image.stride[0])
                    : static_cast<const void*>(static_cast<const uint16_t*>(image.data) + y * image.stride[0]);
                void* orow = (output.itemsize == 2)
                    ? static_cast<void*>(static_cast<uint16_t*>(output.data) + y * output.stride[0])
                    : static_cast<void*>(static_cast<int32_t*>(output.data) + y * output.stride[0]);
                state = simd::select_a_with_bg_packed_row_avx2(
                    b_row, response.data(), width,
                    irow, image.itemsize == 1, orow, output.itemsize == 2,
                    image.stride[1], image.stride[2], output.stride[1], output.stride[2],
                    bg_row, st_row, scale, first, state);
                continue;
            }
            for (int64_t x = 0; x < width; ++x) {
                const int64_t prev = first ? 0 : static_cast<int64_t>(b_row[x]);
                if (!first && prev == kAMaxResponse) {
                    continue;
                }
                int64_t sc = response[x];
                if (sc > kAMaxResponse) {
                    sc = kAMaxResponse;
                }
                const bool take = first || sc > prev;
                const bool collect = prev < kABlendLimit && sc < kABlendLimit;
                if (collect) {
                    state = kABackground;
                } else if (state == kASaturated && sc != kAMaxResponse) {
                    state = kASharp;
                }
                if (!(collect || take)) {
                    continue;
                }
                float factor = 1.0f;
                float weight = 0.0f;
                if (collect) {
                    const uint32_t cur_code = static_cast<uint32_t>(sc >> 4);
                    const int64_t px_off = x * bsx;
                    if (first) {
                        st_row[px_off + stat_col * bsc] = cur_code;
                        weight = (sc != 0) ? 1.0f : 0.0f;
                        bg_row[px_off + weight_col * bsc] = weight;
                    } else {
                        const uint32_t packed = st_row[px_off + stat_col * bsc];
                        uint32_t minimum = packed & 4095u;
                        uint32_t second = packed >> 12u;
                        if (sc == 0) {
                            if (minimum) {
                                st_row[px_off + stat_col * bsc] = second << 12u;
                            }
                            continue;
                        }
                        if (take) {
                            second = static_cast<uint32_t>(prev >> 4);
                            const float ratio = static_cast<float>(prev) / static_cast<float>(sc);
                            factor = ratio * ratio;
                            factor *= factor;
                            weight = 1.0f;
                            bg_row[px_off + weight_col * bsc] = bg_row[px_off + weight_col * bsc] * factor + weight;
                        } else {
                            if (cur_code > second) {
                                second = cur_code;
                            }
                            const float ratio = static_cast<float>(sc) / static_cast<float>(prev);
                            weight = ratio * ratio;
                            weight *= weight;
                            bg_row[px_off + weight_col * bsc] += weight;
                        }
                        if (cur_code < minimum) {
                            minimum = cur_code;
                        }
                        st_row[px_off + stat_col * bsc] = (second << 12u) | minimum;
                    }
                }
                const int64_t px_off = x * bsx;
                for (int64_t ch = 0; ch < channels; ++ch) {
                    const int64_t val = read_int(image, y, x, ch) * scale;
                    if (collect) {
                        float weighted = static_cast<float>(val) * weight;
                        if (!first) {
                            weighted = bg_row[px_off + ch * bsc] * factor + weighted;
                        }
                        bg_row[px_off + ch * bsc] = weighted;
                    }
                    if (take) {
                        if (output.ndim == 2) {
                            write_int(output, y, x, val);
                        } else {
                            write_int(output, y, x, val, ch);
                        }
                    }
                }
                if (take) {
                    b_row[x] = static_cast<int32_t>(sc);
                }
            }
        }
    }
    return state;
}

void of_blend_low_response_rows_cpp(
    const ArrayView& output,
    const ArrayView& best,
    const ArrayView& background,
    int64_t start,
    int64_t stop
) {
    const int64_t width = best.shape[1];
    const int64_t channels = (output.ndim == 2) ? 1 : output.shape[2];
    const int64_t stat_col = channels + 1;
    const int32_t* best_data = static_cast<const int32_t*>(best.data);
    const float* bg_f32 = static_cast<const float*>(background.data);
    const uint32_t* bg_u32 = static_cast<const uint32_t*>(background.data);

    for (int64_t y = start; y < stop; ++y) {
        const int64_t local_y = y - start;
        const int32_t* b_row = best_data + y * best.stride[0];
        const float* bg_row = bg_f32 + local_y * background.stride[0];
        const uint32_t* st_row = bg_u32 + local_y * background.stride[0];
        const int64_t bsx = background.stride[1];
        const int64_t bsc = background.stride[2];

        for (int64_t x = 0; x < width; ++x) {
            const int64_t score = b_row[x];
            if (score >= kABlendLimit) {
                continue;
            }
            const int64_t px_off = x * bsx;
            const int64_t packed = st_row[px_off + stat_col * bsc];
            if ((score >> 4) == (packed & 4095)) {
                continue;
            }
            const int64_t second = (packed >> 12) << 4;
            const int64_t gap = std::max<int64_t>(0, score - second);
            if (gap * 2 >= score) {
                continue;
            }
            const int64_t confidence = std::min(score, 5 * score - 10 * gap);
            const float strength = (static_cast<float>(std::min<int64_t>(32768, kABlendLimit - score)) / 32768.0f)
                                 * static_cast<float>(confidence) / static_cast<float>(score);
            const float denom = bg_row[px_off + channels * bsc];
            if (!(denom > 0.0f)) {
                continue;
            }
            const float inv_s = 1.0f - strength;
            for (int64_t ch = 0; ch < channels; ++ch) {
                const float avg = bg_row[px_off + ch * bsc] / denom;
                const int64_t cur = (output.ndim == 2) ? read_int(output, y, x) : read_int(output, y, x, ch);
                const float val = static_cast<float>(cur) * inv_s + avg * strength;
                const int64_t iv = std::clamp<int64_t>(static_cast<int64_t>(val + 0.5f), 0, 65535);
                if (output.ndim == 2) {
                    write_int(output, y, x, iv);
                } else {
                    write_int(output, y, x, iv, ch);
                }
            }
        }
    }
}

void of_depth_axis_iir_cpp(
    const uint16_t* src,
    int64_t src_sy,
    const uint16_t* guide,
    int64_t guide_sy,
    uint16_t* out,
    int64_t out_sy,
    int64_t h,
    int64_t w,
    uint32_t q,
    uint32_t t,
    uint32_t tm,
    uint32_t tl,
    int64_t count,
    int axis,
    int num_threads
) {
    if (h <= 0 || w <= 0) {
        return;
    }
    const int64_t cnt = std::max<int64_t>(1, std::min<int64_t>(count, axis == 0 ? h : w));

    if (axis == 0) {
        const int64_t band = 64;
        const int threads = (h * w >= 131072) ? num_threads : 1;
        openfocus::parallel_bands(w, band, threads, [&](int64_t left, int64_t right) {
            const int64_t cols = right - left;
            if (guide == nullptr) {
                if (simd::use_avx2() && cols >= 8) {
                    simd::depth_axis_iir_y_unguided_avx2(src, src_sy, out, out_sy, h, left, cols, cnt, q, t);
                    return;
                }
                const uint32_t t16 = t * 16u;
                thread_local std::vector<uint16_t> first;
                thread_local std::vector<uint32_t> acc;
                const size_t first_sz = static_cast<size_t>(h) * cols;
                if (first.size() < first_sz) {
                    first.resize(first_sz);
                }
                acc.assign(static_cast<size_t>(cols), 0u);
                for (int64_t y = 0; y < cnt; ++y) {
                    const uint16_t* sr = src + y * src_sy + left;
                    for (int64_t c = 0; c < cols; ++c) {
                        acc[c] += static_cast<uint32_t>(sr[c]);
                    }
                }
                const uint32_t t_plus_q = t + q;
                for (int64_t c = 0; c < cols; ++c) {
                    const uint64_t init = ((static_cast<uint64_t>(acc[c]) << 4u) / static_cast<uint64_t>(cnt)) * t_plus_q;
                    acc[c] = static_cast<uint32_t>(init & 0xffffffffu);
                }
                for (int64_t y = 0; y < h; ++y) {
                    const uint16_t* sr = src + y * src_sy + left;
                    uint16_t* fr = first.data() + y * cols;
                    for (int64_t c = 0; c < cols; ++c) {
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
                    for (int64_t c = 0; c < cols; ++c) {
                        const uint32_t inn = static_cast<uint32_t>(sr[c]) * t16;
                        const uint32_t st = (acc[c] + 2048u + inn) >> 12u;
                        orow[c] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + static_cast<uint32_t>(fr[c])) >> 1u);
                        acc[c] = st * q + inn;
                    }
                }
            } else {
                if (simd::use_avx2() && cols >= 8) {
                    simd::depth_axis_iir_y_guided_avx2(src, src_sy, guide, guide_sy, out, out_sy, h, left, cols, cnt, tm, tl);
                    return;
                }
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
                // Forward y = 0
                {
                    const uint16_t* sr0 = src + left;
                    const uint16_t* gr0 = guide + left;
                    uint32_t* fr0 = first.data();
                    for (int64_t c = 0; c < cols; ++c) {
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
                    for (int64_t c = 0; c < cols; ++c) {
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
                    for (int64_t c = 0; c < cols; ++c) {
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
                    for (int64_t c = 0; c < cols; ++c) {
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
        });
        return;
    }

    // Horizontal (axis == 1): each row is independent -> keep row history in L1/L2 per-thread buffer, zero full-plane transpose.
    const int64_t row_band = 16;
    const int threads = (h * w >= 131072) ? num_threads : 1;
    openfocus::parallel_bands(h, row_band, threads, [&](int64_t start, int64_t stop) {
        if (guide == nullptr) {
            if (simd::use_avx2() && w >= 8 && (stop - start) >= 8) {
                simd::depth_axis_iir_x_unguided_avx2(src, src_sy, out, out_sy, start, stop, w, cnt, q, t);
                return;
            }
            const uint32_t t16 = t * 16u;
            const uint32_t t_plus_q = t + q;
            thread_local std::vector<uint16_t> first;
            if (first.size() < static_cast<size_t>(w)) {
                first.resize(static_cast<size_t>(w));
            }
            for (int64_t y = start; y < stop; ++y) {
                const uint16_t* sr = src + y * src_sy;
                uint16_t* orow = out + y * out_sy;
                uint64_t sum = 0;
                for (int64_t x = 0; x < cnt; ++x) {
                    sum += sr[x];
                }
                uint32_t acc = static_cast<uint32_t>((((sum << 4u) / static_cast<uint64_t>(cnt)) * t_plus_q) & 0xffffffffu);
                for (int64_t x = 0; x < w; ++x) {
                    const uint32_t inn = static_cast<uint32_t>(sr[x]) * t16;
                    const uint32_t st = (acc + 2048u + inn) >> 12u;
                    first[x] = static_cast<uint16_t>((st + 4u) >> 4u);
                    acc = st * q + inn;
                }
                acc = (static_cast<uint32_t>(first[w - 1]) << 4u) * t_plus_q;
                for (int64_t x = w - 1; x >= 0; --x) {
                    const uint32_t inn = static_cast<uint32_t>(sr[x]) * t16;
                    const uint32_t st = (acc + 2048u + inn) >> 12u;
                    orow[x] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + static_cast<uint32_t>(first[x])) >> 1u);
                    acc = st * q + inn;
                }
            }
        } else {
            if (simd::use_avx2() && w >= 8 && (stop - start) >= 8) {
                simd::depth_axis_iir_x_guided_avx2(src, src_sy, guide, guide_sy, out, out_sy, start, stop, w, cnt, tm, tl);
                return;
            }
            thread_local std::vector<uint32_t> first;
            if (first.size() < static_cast<size_t>(w)) {
                first.resize(static_cast<size_t>(w));
            }
            for (int64_t y = start; y < stop; ++y) {
                const uint16_t* sr = src + y * src_sy;
                const uint16_t* gr = guide + y * guide_sy;
                uint16_t* orow = out + y * out_sy;
                uint64_t sum = 0;
                for (int64_t x = 0; x < cnt; ++x) {
                    sum += sr[x];
                }
                uint32_t acc = static_cast<uint32_t>((sum << 4u) / static_cast<uint64_t>(cnt));
                {
                    const uint32_t v = sr[0];
                    const uint32_t s = gr[0];
                    const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
                    const uint32_t st = ((4096u - 2u * amt) * acc + 2048u + (v + v) * (amt << 4u)) >> 12u;
                    acc = st;
                    first[0] = (st + 4u) >> 4u;
                }
                for (int64_t x = 1; x < w; ++x) {
                    const uint32_t v = sr[x];
                    const uint32_t p = sr[x - 1];
                    const uint32_t s = gr[x];
                    const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
                    const uint32_t st = ((4096u - 2u * amt) * acc + 2048u + (v + p) * (amt << 4u)) >> 12u;
                    acc = st;
                    first[x] = (st + 4u) >> 4u;
                }
                acc = first[w - 1] << 4u;
                {
                    const int64_t x = w - 1;
                    const uint32_t v = sr[x];
                    const uint32_t p = first[x];
                    const uint32_t s = gr[x];
                    const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
                    const uint32_t st = ((4096u - 2u * amt) * acc + 2048u + (v + p) * (amt << 4u)) >> 12u;
                    acc = st;
                    orow[x] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + p) >> 1u);
                }
                for (int64_t x = w - 2; x >= 0; --x) {
                    const uint32_t v = sr[x];
                    const uint32_t p = sr[x + 1];
                    const uint32_t s = gr[x];
                    const uint32_t amt = ((65536u - s) * tl + s * tm) >> 16u;
                    const uint32_t st = ((4096u - 2u * amt) * acc + 2048u + (v + p) * (amt << 4u)) >> 12u;
                    acc = st;
                    orow[x] = static_cast<uint16_t>((((st + 3u) >> 4u) + 1u + first[x]) >> 1u);
                }
            }
        }
    });
}

void of_activity_rows_cpp(const uint16_t* gray, int64_t g_sy,
                          const uint16_t* fine, int64_t f_sy,
                          uint16_t* out, int64_t o_sy,
                          int64_t w, int64_t start, int64_t stop) {
    const bool use_simd = simd::use_avx2() && w >= 8;
    const auto row_bytes = w * sizeof(uint16_t);
    for (int64_t y = start; y < stop; ++y) {
        const uint16_t* gr = gray + y * g_sy;
        const uint16_t* fr = fine + y * f_sy;
        uint16_t* orow = out + y * o_sy;
        if (use_simd && !byte_ranges_overlap(orow, row_bytes, gr, row_bytes)
                     && !byte_ranges_overlap(orow, row_bytes, fr, row_bytes)) {
            simd::activity_row_u16_avx2(gr, fr, orow, w);
        } else {
            for (int64_t x = 0; x < w; ++x) {
                orow[x] = normalized_diff_u16(gr[x], fr[x]);
            }
        }
    }
}

void of_detail_rows_cpp(const uint16_t* gray, int64_t g_sy, int64_t gh, int64_t gw,
                        uint16_t* out, int64_t o_sy, int64_t oh, int64_t ow,
                        int64_t row_offset) {
    if (oh == 0 || ow == 0 || gh == 0 || gw == 0) {
        return;
    }
    const int64_t full_w = std::min(gw / 2, ow);
    const bool use_simd = simd::use_avx2() && full_w >= 8;
    const auto row_bytes = full_w * sizeof(uint16_t);
    for (int64_t y = 0; y < oh; ++y) {
        int64_t y0 = 2 * (y + row_offset);
        if (y0 >= gh) {
            y0 = gh - 1;
        }
        const int64_t y1 = (y0 + 1 < gh) ? (y0 + 1) : (gh - 1);
        const uint16_t* r0 = gray + y0 * g_sy;
        const uint16_t* r1 = gray + y1 * g_sy;
        uint16_t* orow = out + y * o_sy;
        if (use_simd && !byte_ranges_overlap(orow, row_bytes, r0, 2 * row_bytes)
                     && !byte_ranges_overlap(orow, row_bytes, r1, 2 * row_bytes)) {
            simd::detail_row_u16_avx2(r0, r1, orow, full_w);
        } else {
            for (int64_t x = 0; x < full_w; ++x) {
                const int64_t x0 = 2 * x;
                const int64_t a = r0[x0], b = r0[x0 + 1], c = r1[x0], d = r1[x0 + 1];
                const int64_t mean = (a + b + c + d) >> 2;
                const int64_t e = static_cast<int64_t>(normalized_diff_u16(a, mean))
                                + normalized_diff_u16(b, mean)
                                + normalized_diff_u16(c, mean)
                                + normalized_diff_u16(d, mean);
                orow[x] = static_cast<uint16_t>(e >> 2);
            }
        }
        for (int64_t x = full_w; x < ow; ++x) {
            int64_t x0 = 2 * x;
            if (x0 >= gw) {
                x0 = gw - 1;
            }
            const int64_t x1 = (x0 + 1 < gw) ? (x0 + 1) : (gw - 1);
            const int64_t a = r0[x0], b = r0[x1], c = r1[x0], d = r1[x1];
            const int64_t mean = (a + b + c + d) >> 2;
            const int64_t e = static_cast<int64_t>(normalized_diff_u16(a, mean))
                            + normalized_diff_u16(b, mean)
                            + normalized_diff_u16(c, mean)
                            + normalized_diff_u16(d, mean);
            orow[x] = static_cast<uint16_t>(e >> 2);
        }
    }
}

void of_smooth_detail_rows_cpp(const ArrayView& detail, const ArrayView& output,
                               int64_t start, int64_t stop) {
    if (start >= stop || detail.shape[0] == 0 || detail.shape[1] == 0) {
        return;
    }
    const int64_t height = detail.shape[0];
    const int64_t width = detail.shape[1];
    thread_local std::vector<int32_t> columns;
    if (columns.size() < static_cast<size_t>(width)) {
        columns.resize(static_cast<size_t>(width));
    }
    const int64_t y0 = reflect_idx(start - 2, height);
    const int64_t y1 = reflect_idx(start - 1, height);
    const int64_t y2 = start;
    const int64_t y3 = reflect_idx(start + 1, height);
    const int64_t y4 = reflect_idx(start + 2, height);
    const bool packed_u16 = (detail.itemsize == 2 && output.itemsize == 2
                             && detail.stride[1] == 1 && output.stride[1] == 1);
    const uint16_t* db = packed_u16 ? static_cast<const uint16_t*>(detail.data) : nullptr;
    uint16_t* ob = packed_u16 ? static_cast<uint16_t*>(output.data) : nullptr;
    if (packed_u16) {
        const uint16_t* r0 = db + y0 * detail.stride[0];
        const uint16_t* r1 = db + y1 * detail.stride[0];
        const uint16_t* r2 = db + y2 * detail.stride[0];
        const uint16_t* r3 = db + y3 * detail.stride[0];
        const uint16_t* r4 = db + y4 * detail.stride[0];
        for (int64_t x = 0; x < width; ++x) {
            columns[x] = static_cast<int32_t>(r0[x] + r1[x] + r2[x] + r3[x] + r4[x]);
        }
    } else {
        for (int64_t x = 0; x < width; ++x) {
            columns[x] = static_cast<int32_t>(
                read_2d(detail, y0, x) + read_2d(detail, y1, x) + read_2d(detail, y2, x)
                + read_2d(detail, y3, x) + read_2d(detail, y4, x));
        }
    }
    const bool use_u64 = (detail.itemsize == 1 || detail.itemsize == 2);
    constexpr uint64_t recip25 = 0x51EB851FULL;
    for (int64_t y = start; y < stop; ++y) {
        if (packed_u16 && width >= 5) {
            uint16_t* orow = ob + y * output.stride[0];
            int32_t total = columns[2] + columns[1] + columns[0] + columns[1] + columns[2];
            orow[0] = static_cast<uint16_t>((static_cast<uint64_t>(total) * recip25) >> 35u);
            total += columns[3] - columns[2];
            orow[1] = static_cast<uint16_t>((static_cast<uint64_t>(total) * recip25) >> 35u);
            total += columns[4] - columns[1];
            for (int64_t x = 2; x < width - 3; ++x) {
                orow[x] = static_cast<uint16_t>((static_cast<uint64_t>(total) * recip25) >> 35u);
                total += columns[x + 3] - columns[x - 2];
            }
            for (int64_t x = width - 3; x < width; ++x) {
                orow[x] = static_cast<uint16_t>((static_cast<uint64_t>(total) * recip25) >> 35u);
                if (x + 1 < width) {
                    const int64_t ent = (x + 3 < width) ? (x + 3) : reflect_idx(x + 3, width);
                    const int64_t leav = (x >= 2) ? (x - 2) : reflect_idx(x - 2, width);
                    total += columns[ent] - columns[leav];
                }
            }
        } else if (width >= 5 && use_u64) {
            int32_t total = columns[2] + columns[1] + columns[0] + columns[1] + columns[2];
            write_2d(output, y, 0, static_cast<int64_t>((static_cast<uint64_t>(total) * recip25) >> 35u));
            total += columns[3] - columns[2];
            write_2d(output, y, 1, static_cast<int64_t>((static_cast<uint64_t>(total) * recip25) >> 35u));
            total += columns[4] - columns[1];
            for (int64_t x = 2; x < width - 3; ++x) {
                write_2d(output, y, x, static_cast<int64_t>((static_cast<uint64_t>(total) * recip25) >> 35u));
                total += columns[x + 3] - columns[x - 2];
            }
            for (int64_t x = width - 3; x < width; ++x) {
                write_2d(output, y, x, static_cast<int64_t>((static_cast<uint64_t>(total) * recip25) >> 35u));
                if (x + 1 < width) {
                    const int64_t ent = (x + 3 < width) ? (x + 3) : reflect_idx(x + 3, width);
                    const int64_t leav = (x >= 2) ? (x - 2) : reflect_idx(x - 2, width);
                    total += columns[ent] - columns[leav];
                }
            }
        } else {
            int64_t total = 0;
            for (int dx = -2; dx <= 2; ++dx) {
                total += columns[reflect_idx(dx, width)];
            }
            for (int64_t x = 0; x < width; ++x) {
                write_2d(output, y, x, total / 25);
                if (x + 1 < width) {
                    const int64_t ent = (x + 3 < width) ? (x + 3) : reflect_idx(x + 3, width);
                    const int64_t leav = (x >= 2) ? (x - 2) : reflect_idx(x - 2, width);
                    total += columns[ent] - columns[leav];
                }
            }
        }
        if (y + 1 < stop) {
            const int64_t ye = reflect_idx(y + 3, height);
            const int64_t yl = reflect_idx(y - 2, height);
            if (packed_u16) {
                const uint16_t* re = db + ye * detail.stride[0];
                const uint16_t* rl = db + yl * detail.stride[0];
                for (int64_t x = 0; x < width; ++x) {
                    columns[x] += static_cast<int32_t>(re[x]) - static_cast<int32_t>(rl[x]);
                }
            } else {
                for (int64_t x = 0; x < width; ++x) {
                    columns[x] += static_cast<int32_t>(read_2d(detail, ye, x) - read_2d(detail, yl, x));
                }
            }
        }
    }
}

void of_initialize_detail_rows_cpp(uint16_t* second, int64_t s_sy,
                                   uint16_t* winner, int64_t w_sy,
                                   int64_t w, int64_t start, int64_t stop) {
    for (int64_t y = start; y < stop; ++y) {
        std::memset(second + y * s_sy, 0, static_cast<size_t>(w) * sizeof(uint16_t));
        std::memset(winner + y * w_sy, 0, static_cast<size_t>(w) * sizeof(uint16_t));
    }
}

void of_select_detail_rows_cpp(const uint16_t* activity, int64_t a_sy,
                               uint16_t* best, int64_t b_sy,
                               uint16_t* second, int64_t s_sy,
                               uint16_t* winner, int64_t w_sy,
                               int64_t w, uint16_t code, int64_t start, int64_t stop) {
    const bool use_simd = simd::use_avx2() && w >= 16;
    for (int64_t y = start; y < stop; ++y) {
        const uint16_t* ar = activity + y * a_sy;
        uint16_t* br = best + y * b_sy;
        uint16_t* sr = second + y * s_sy;
        uint16_t* wr = winner + y * w_sy;
        if (use_simd) {
            simd::select_detail_row_u16_avx2(ar, br, sr, wr, w, code);
        } else {
            for (int64_t x = 0; x < w; ++x) {
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
    }
}

void of_detail_confidence_rows_cpp(const uint16_t* coarse, int64_t c_sy,
                                   const uint16_t* best, int64_t b_sy,
                                   uint16_t* second, int64_t s_sy,
                                   int64_t w, int64_t start, int64_t stop) {
    for (int64_t y = start; y < stop; ++y) {
        const uint16_t* cr = coarse + y * c_sy;
        const uint16_t* br = best + y * b_sy;
        uint16_t* sr = second + y * s_sy;
        for (int64_t x = 0; x < w; ++x) {
            const int64_t top = br[x];
            const int64_t gap = top - static_cast<int64_t>(sr[x]);
            sr[x] = (top >= 128 && top > 4 * (static_cast<int64_t>(cr[x]) + 1) && 2 * gap > top) ? 1 : 0;
        }
    }
}

void of_apply_detail_rows_cpp(uint16_t* depth, int64_t d_sy,
                              const uint16_t* mask, int64_t m_sy,
                              const uint16_t* winner, int64_t w_sy,
                              int64_t w, int64_t start, int64_t stop) {
    for (int64_t y = start; y < stop; ++y) {
        uint16_t* dr = depth + y * d_sy;
        const uint16_t* mr = mask + y * m_sy;
        const uint16_t* wr = winner + y * w_sy;
        for (int64_t x = 0; x < w; ++x) {
            if (mr[x]) {
                dr[x] = wr[x];
            }
        }
    }
}

void of_initialize_activity_rows_cpp(const uint16_t* activity, int64_t a_sy,
                                     uint16_t* depth, int64_t d_sy,
                                     uint16_t* votes, int64_t v_sy,
                                     int64_t w, int64_t reciprocal,
                                     int64_t start, int64_t stop) {
    for (int64_t y = start; y < stop; ++y) {
        std::memset(depth + y * d_sy, 0, static_cast<size_t>(w) * sizeof(uint16_t));
        if (votes != nullptr) {
            const uint16_t* ar = activity + y * a_sy;
            uint16_t* vr = votes + y * v_sy;
            for (int64_t x = 0; x < w; ++x) {
                vr[x] = static_cast<uint16_t>((static_cast<int64_t>(ar[x]) * reciprocal) >> 32);
            }
        }
    }
}

void of_select_activity_rows_cpp(const uint16_t* activity, int64_t a_sy,
                                 uint16_t* best, int64_t b_sy,
                                 uint16_t* depth, int64_t d_sy,
                                 uint16_t* votes, int64_t v_sy,
                                 int64_t w, uint16_t code, int64_t reciprocal,
                                 int64_t start, int64_t stop) {
    const bool use_simd = simd::use_avx2() && w >= 16
        && (votes == nullptr || (reciprocal >= 0 && reciprocal <= int64_t{0xFFFFFFFFLL}));
    const uint32_t recip_u32 = static_cast<uint32_t>(reciprocal);
    for (int64_t y = start; y < stop; ++y) {
        const uint16_t* ar = activity + y * a_sy;
        uint16_t* br = best + y * b_sy;
        uint16_t* dr = depth + y * d_sy;
        uint16_t* vr = (votes != nullptr) ? (votes + y * v_sy) : nullptr;
        if (use_simd) {
            simd::select_activity_row_u16_avx2(ar, br, dr, vr, w, code, recip_u32);
        } else if (vr != nullptr) {
            for (int64_t x = 0; x < w; ++x) {
                const uint16_t val = ar[x];
                vr[x] = static_cast<uint16_t>(vr[x] + ((static_cast<int64_t>(val) * reciprocal) >> 32));
                if (val > br[x]) {
                    br[x] = val;
                    dr[x] = code;
                }
            }
        } else {
            for (int64_t x = 0; x < w; ++x) {
                const uint16_t val = ar[x];
                if (val > br[x]) {
                    br[x] = val;
                    dr[x] = code;
                }
            }
        }
    }
}

void of_confidence_hist_rows_cpp(uint16_t* best, int64_t b_sy,
                                  const uint16_t* votes, int64_t v_sy,
                                  int64_t* hist, int64_t w,
                                  int64_t start, int64_t stop) {
    for (int64_t y = start; y < stop; ++y) {
        uint16_t* br = best + y * b_sy;
        const uint16_t* vr = votes + y * v_sy;
        for (int64_t x = 0; x < w; ++x) {
            int64_t conf = static_cast<int64_t>(br[x]) - static_cast<int64_t>(vr[x]);
            if (conf < 0) {
                conf = -conf;
            }
            br[x] = static_cast<uint16_t>(conf);
            hist[conf] += 1;
        }
    }
}

void of_guide_rows_cpp(uint16_t* best, int64_t b_sy, const float* lut,
                       int64_t w, int64_t start, int64_t stop) {
    constexpr float ratio = 4095.0f / 65535.0f;
    for (int64_t y = start; y < stop; ++y) {
        uint16_t* br = best + y * b_sy;
        for (int64_t x = 0; x < w; ++x) {
            const float pos = static_cast<float>(br[x]) * ratio;
            const int32_t idx = std::clamp<int32_t>(static_cast<int32_t>(pos), 0, 4094);
            const float frac = pos - static_cast<float>(idx);
            const float val = lut[idx] * (1.0f - frac) + lut[idx + 1] * frac;
            br[x] = static_cast<uint16_t>(std::clamp<int64_t>(static_cast<int64_t>(val), 0, 65535));
        }
    }
}

void of_snap_rows_cpp(uint16_t* depth, int64_t d_sy, const int32_t* lut,
                      int64_t w, int64_t step, int64_t start, int64_t stop) {
    if (start >= stop || w == 0) {
        return;
    }
    const int64_t half = step >> 1;
    if (step >= 2 && step <= 65535 && half >= 1) {
        const uint64_t step_recip = ((uint64_t{1} << 48) + static_cast<uint64_t>(step) - 1u) / static_cast<uint64_t>(step);
        const uint64_t half_recip = ((uint64_t{1} << 48) + static_cast<uint64_t>(half) - 1u) / static_cast<uint64_t>(half);
        for (int64_t y = start; y < stop; ++y) {
            uint16_t* dr = depth + y * d_sy;
            for (int64_t x = 0; x < w; ++x) {
                const int64_t block = dr[x];
                const int64_t total = half + block;
                const int64_t snapped = static_cast<int64_t>((static_cast<uint64_t>(total) * step_recip) >> 48u) * step;
                const int64_t dist = block - snapped;
                if (dist >= 0) {
                    const int64_t idx = std::min<int64_t>(1024, static_cast<int64_t>((static_cast<uint64_t>(dist << 10) * half_recip) >> 48u));
                    const int64_t corr = (half * static_cast<int64_t>(lut[idx])) >> 10;
                    dr[x] = static_cast<uint16_t>((snapped + corr) & 65535);
                } else {
                    const int64_t idx = std::min<int64_t>(1024, static_cast<int64_t>((static_cast<uint64_t>((-dist) << 10) * half_recip) >> 48u));
                    const int64_t corr = (half * static_cast<int64_t>(lut[idx])) >> 10;
                    dr[x] = static_cast<uint16_t>((snapped - corr) & 65535);
                }
            }
        }
        return;
    }
    for (int64_t y = start; y < stop; ++y) {
        uint16_t* dr = depth + y * d_sy;
        for (int64_t x = 0; x < w; ++x) {
            const int64_t block = dr[x];
            const int64_t total = half + block;
            const int64_t snapped = total - total % step;
            const int64_t dist = block - snapped;
            const int64_t sgnd = ((std::abs(dist) << 10) / half) * (dist < 0 ? -1 : 1);
            const int64_t idx = std::min<int64_t>(1024, std::abs(sgnd));
            int64_t corr = half * static_cast<int64_t>(lut[idx]) * (sgnd > 0 ? 1 : -1);
            corr = (corr >= 0) ? (corr >> 10) : -((-corr) >> 10);
            dr[x] = static_cast<uint16_t>((snapped + corr) & 65535);
        }
    }
}

void of_resample_rows_u16_cpp(
    const uint16_t* src, int64_t s_sy, int64_t sw,
    uint16_t* out, int64_t o_sy, int64_t ow,
    const int32_t* v_idx, int64_t vi_sy, int64_t v_taps,
    const float* v_w, int64_t vw_sy,
    const int32_t* v_cnt,
    const int32_t* h_idx, int64_t hi_sy, int64_t h_taps,
    const float* h_w, int64_t hw_sy,
    const int32_t* h_cnt,
    int64_t start, int64_t stop
) {
    if (start >= stop || sw <= 0 || ow <= 0) {
        return;
    }
    // Check if columns repeat (matching _repeated_columns)
    int64_t first = 0, last = 0;
    if (h_taps >= 2 && (h_taps % 2) == 0) {
        auto repeats = [&](int64_t col, int64_t ref) -> bool {
            const int64_t base = h_idx[ref * hi_sy] + 2 * (col - ref);
            if (h_cnt[col] != h_taps || base < 0) {
                return false;
            }
            for (int64_t t = 0; t < h_taps; ++t) {
                if (h_w[col * hw_sy + t] != h_w[ref * hw_sy + t] || h_idx[col * hi_sy + t] != base + t) {
                    return false;
                }
            }
            return true;
        };
        while (first < ow && !repeats(first, first)) {
            ++first;
        }
        last = first;
        while (last < ow && repeats(last, first)) {
            ++last;
        }
    }

    thread_local std::vector<float> row_buf;
    thread_local std::vector<float> sums_buf;
    if (row_buf.size() < static_cast<size_t>(sw)) {
        row_buf.resize(static_cast<size_t>(sw));
    }
    const size_t nrep_sz = static_cast<size_t>(std::max<int64_t>(0, last - first));
    if (sums_buf.size() < nrep_sz) {
        sums_buf.resize(nrep_sz);
    }
    const bool use_fma = simd::use_avx2_fma();

    for (int64_t y = start; y < stop; ++y) {
        std::fill_n(row_buf.data(), static_cast<size_t>(sw), 0.0f);
        const int32_t vc = v_cnt[y];
        const int32_t* vi_row = v_idx + y * vi_sy;
        const float* vw_row = v_w + y * vw_sy;
        for (int32_t tap = 0; tap < vc; ++tap) {
            const float wval = vw_row[tap];
            const uint16_t* sline = src + static_cast<int64_t>(vi_row[tap]) * s_sy;
            if (use_fma && sw >= 8) {
                simd::resample_u16_vert_acc_row_avx2_fma(sline, row_buf.data(), sw, wval);
            } else {
                for (int64_t x = 0; x < sw; ++x) {
                    row_buf[x] = std::fmaf(wval, static_cast<float>(sline[x]), row_buf[x]);
                }
            }
        }
        uint16_t* orow = out + y * o_sy;
        for (int64_t x = 0; x < first; ++x) {
            float acc = 0.0f;
            const int32_t hc = h_cnt[x];
            const int32_t* hi = h_idx + x * hi_sy;
            const float* hw = h_w + x * hw_sy;
            for (int32_t t = 0; t < hc; ++t) {
                acc = std::fmaf(hw[t], row_buf[hi[t]], acc);
            }
            orow[x] = store_f32_code(acc);
        }
        for (int64_t x = last; x < ow; ++x) {
            float acc = 0.0f;
            const int32_t hc = h_cnt[x];
            const int32_t* hi = h_idx + x * hi_sy;
            const float* hw = h_w + x * hw_sy;
            for (int32_t t = 0; t < hc; ++t) {
                acc = std::fmaf(hw[t], row_buf[hi[t]], acc);
            }
            orow[x] = store_f32_code(acc);
        }
        if (last > first) {
            const int64_t nrep = last - first;
            if (use_fma && nrep >= 8) {
                simd::resample_u16_horiz_rep_row_avx2_fma(
                    row_buf.data(), sums_buf.data(), orow, first, nrep,
                    h_idx + first * hi_sy, h_w + first * hw_sy, h_taps, hi_sy, hw_sy);
            } else {
                std::fill_n(sums_buf.data(), static_cast<size_t>(nrep), 0.0f);
                for (int64_t tap = 0; tap < h_taps; tap += 2) {
                    const float even = h_w[first * hw_sy + tap];
                    const float odd = h_w[first * hw_sy + tap + 1];
                    const float* line = row_buf.data() + h_idx[first * hi_sy + tap];
                    for (int64_t x = 0; x < nrep; ++x) {
                        sums_buf[x] = std::fmaf(odd, line[2 * x + 1],
                                                std::fmaf(even, line[2 * x], sums_buf[x]));
                    }
                }
                for (int64_t x = 0; x < nrep; ++x) {
                    orow[first + x] = store_f32_code(sums_buf[x]);
                }
            }
        }
    }
}

void of_merge_rows_cpp(
    const ArrayView& accumulator,
    const ArrayView& image,
    const uint16_t* depth, int64_t d_sy,
    int64_t index, int64_t step,
    int64_t start, int64_t stop,
    int64_t scale, bool first, bool reverse
) {
    const int64_t width = accumulator.shape[1];
    if (first) {
        if (image.ndim == 2) {
            for (int64_t y = start; y < stop; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    write_int(accumulator, y, x, read_int(image, y, x) * scale);
                }
            }
        } else {
            const int64_t channels = image.shape[2];
            for (int64_t y = start; y < stop; ++y) {
                for (int64_t x = 0; x < width; ++x) {
                    for (int64_t ch = 0; ch < channels; ++ch) {
                        write_int(accumulator, y, x, read_int(image, y, x, ch) * scale, ch);
                    }
                }
            }
        }
        return;
    }

    const int64_t base = index * step;
    if (image.ndim == 2) {
        for (int64_t y = start; y < stop; ++y) {
            const uint16_t* drow = depth + y * d_sy;
            for (int64_t x = 0; x < width; ++x) {
                int64_t w = 1024;
                int64_t delta = static_cast<int64_t>(drow[x]) - base;
                if (reverse) {
                    delta = -delta;
                }
                if (delta >= 0) {
                    if (delta > step) {
                        continue;
                    }
                } else {
                    if (delta <= -step) {
                        continue;
                    }
                    w = 1024 - ((-delta << 10) / step);
                    if (w <= 0) {
                        continue;
                    }
                }
                const int64_t val = read_int(image, y, x) * scale;
                if (w == 1024) {
                    write_int(accumulator, y, x, val);
                } else {
                    const int64_t cur = read_int(accumulator, y, x);
                    write_int(accumulator, y, x, (cur * (1024 - w) + val * w) >> 10);
                }
            }
        }
        return;
    }

    const int64_t channels = image.shape[2];
    const bool use_recip = (accumulator.ndim == 3 && (!reverse) && step >= 2 && step <= 65535
                            && channels == 3 && scale >= 1 && scale <= 256
                            && (image.itemsize == 1 || image.itemsize == 2)
                            && !image.is_float && !accumulator.is_float
                            && (accumulator.itemsize == 2 || accumulator.itemsize == 4));
    if (use_recip) {
        const int32_t s32 = static_cast<int32_t>(scale);
        const bool src_packed = (image.stride[1] == 3 && image.stride[2] == 1);
        const bool dst_packed = (accumulator.stride[1] == 3 && accumulator.stride[2] == 1);
        const bool use_avx2_skip = simd::use_avx2() && width >= 16;
        if (image.itemsize == 1 && accumulator.itemsize == 2) {
            merge_recip_packed_rows<uint8_t, uint16_t>(
                accumulator, image, depth, d_sy, base, step, start, stop, s32,
                src_packed, dst_packed, use_avx2_skip);
        } else if (image.itemsize == 1 && accumulator.itemsize == 4) {
            merge_recip_packed_rows<uint8_t, int32_t>(
                accumulator, image, depth, d_sy, base, step, start, stop, s32,
                src_packed, dst_packed, use_avx2_skip);
        } else if (image.itemsize == 2 && accumulator.itemsize == 2) {
            merge_recip_packed_rows<uint16_t, uint16_t>(
                accumulator, image, depth, d_sy, base, step, start, stop, s32,
                src_packed, dst_packed, use_avx2_skip);
        } else {
            merge_recip_packed_rows<uint16_t, int32_t>(
                accumulator, image, depth, d_sy, base, step, start, stop, s32,
                src_packed, dst_packed, use_avx2_skip);
        }
        return;
    }

    for (int64_t y = start; y < stop; ++y) {
        const uint16_t* drow = depth + y * d_sy;
        for (int64_t x = 0; x < width; ++x) {
            int64_t w = 1024;
            int64_t delta = static_cast<int64_t>(drow[x]) - base;
            if (reverse) {
                delta = -delta;
            }
            if (delta >= 0) {
                if (delta > step) {
                    continue;
                }
            } else {
                if (delta <= -step) {
                    continue;
                }
                w = 1024 - ((-delta << 10) / step);
                if (w <= 0) {
                    continue;
                }
            }
            const int64_t iw = 1024 - w;
            for (int64_t ch = 0; ch < channels; ++ch) {
                const int64_t val = read_int(image, y, x, ch) * scale;
                if (w == 1024) {
                    write_int(accumulator, y, x, val, ch);
                } else {
                    const int64_t cur = read_int(accumulator, y, x, ch);
                    write_int(accumulator, y, x, (cur * iw + val * w) >> 10, ch);
                }
            }
        }
    }
}

void of_active_source_rows_cpp(
    const uint16_t* depth, int64_t d_sy, int64_t w,
    int64_t step, uint8_t* active, int64_t active_len,
    int64_t start, int64_t stop
) {
    if (start >= stop || w == 0 || active_len <= 0) {
        return;
    }
    const int64_t last = active_len - 1;
    if (step >= 1 && step <= 65535) {
        const uint64_t recip48 = ((uint64_t{1} << 48) + static_cast<uint64_t>(step) - 1u) / static_cast<uint64_t>(step);
        const uint64_t step64 = static_cast<uint64_t>(step);
        for (int64_t y = start; y < stop; ++y) {
            const uint16_t* drow = depth + y * d_sy;
            for (int64_t x = 0; x < w; ++x) {
                const uint64_t code = drow[x];
                const int64_t raw_lower = static_cast<int64_t>((code * recip48) >> 48u);
                const int64_t lower = std::min(raw_lower, last);
                active[lower] = 1;
                if (code != static_cast<uint64_t>(raw_lower) * step64) {
                    const int64_t nxt = std::min(lower + 1, last);
                    active[nxt] = 1;
                }
            }
        }
        return;
    }
    for (int64_t y = start; y < stop; ++y) {
        const uint16_t* drow = depth + y * d_sy;
        for (int64_t x = 0; x < w; ++x) {
            const int64_t code = drow[x];
            const int64_t lower = std::clamp<int64_t>(code / step, 0, last);
            active[lower] = 1;
            if (code % step != 0) {
                active[std::min(lower + 1, last)] = 1;
            }
        }
    }
}

void of_output_u16_to_u8_rows_cpp(
    const ArrayView& code, const ArrayView& output,
    int64_t start, int64_t stop, bool e37
) {
    const int64_t width = code.shape[1];
    const int64_t channels = (code.ndim == 2) ? 1 : code.shape[2];
    const uint16_t* cd = static_cast<const uint16_t*>(code.data);
    uint8_t* od = static_cast<uint8_t*>(output.data);
    const bool packed = (code.ndim == 2 && code.stride[1] == 1 && output.stride[1] == 1)
                     || (code.ndim == 3 && output.ndim == 3 && code.stride[2] == 1
                         && output.stride[2] == 1 && code.stride[1] == channels
                         && output.stride[1] == channels);
    const int64_t elems = width * channels;
    const bool use_simd = packed && simd::use_avx2() && elems >= 16 && (!e37 || channels <= 3);

    for (int64_t y = start; y < stop; ++y) {
        if (use_simd) {
            simd::output_u16_to_u8_row_avx2(
                cd + y * code.stride[0], od + y * output.stride[0], elems, e37);
        } else if (code.ndim == 2) {
            const uint16_t* cr = cd + y * code.stride[0];
            uint8_t* orow = od + y * output.stride[0];
            for (int64_t x = 0; x < width; ++x) {
                uint16_t v = cr[x * code.stride[1]] >> 8u;
                orow[x * output.stride[1]] = static_cast<uint8_t>((e37 && v > 0) ? (v - 1) : v);
            }
        } else {
            for (int64_t x = 0; x < width; ++x) {
                for (int64_t ch = 0; ch < channels; ++ch) {
                    const uint16_t cval = cd[y * code.stride[0] + x * code.stride[1] + ch * code.stride[2]];
                    uint16_t v = cval >> 8u;
                    if (e37 && ch < 3 && v > 0) {
                        v -= 1;
                    }
                    od[y * output.stride[0] + x * output.stride[1] + ch * output.stride[2]] = static_cast<uint8_t>(v);
                }
            }
        }
    }
}

void of_output_u16_to_f32_rows_cpp(
    const ArrayView& code, const ArrayView& output,
    int64_t start, int64_t stop, float factor
) {
    const int64_t width = code.shape[1];
    const int64_t channels = (code.ndim == 2) ? 1 : code.shape[2];
    const uint16_t* cd = static_cast<const uint16_t*>(code.data);
    float* od = static_cast<float*>(output.data);
    const bool packed = (code.ndim == 2 && code.stride[1] == 1 && output.stride[1] == 1)
                     || (code.ndim == 3 && output.ndim == 3 && code.stride[2] == 1
                         && output.stride[2] == 1 && code.stride[1] == channels
                         && output.stride[1] == channels);
    const int64_t elems = width * channels;
    const bool use_simd = packed && simd::use_avx2() && elems >= 8;

    for (int64_t y = start; y < stop; ++y) {
        if (use_simd) {
            simd::output_u16_to_f32_row_avx2(
                cd + y * code.stride[0], od + y * output.stride[0], elems, factor);
        } else if (code.ndim == 2) {
            const uint16_t* cr = cd + y * code.stride[0];
            float* orow = od + y * output.stride[0];
            for (int64_t x = 0; x < width; ++x) {
                orow[x * output.stride[1]] = static_cast<float>(cr[x * code.stride[1]]) * factor;
            }
        } else {
            for (int64_t x = 0; x < width; ++x) {
                for (int64_t ch = 0; ch < channels; ++ch) {
                    const uint16_t cval = cd[y * code.stride[0] + x * code.stride[1] + ch * code.stride[2]];
                    od[y * output.stride[0] + x * output.stride[1] + ch * output.stride[2]] = static_cast<float>(cval) * factor;
                }
            }
        }
    }
}

}  // namespace openfocus
