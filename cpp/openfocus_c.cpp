#include "openfocus_c.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-math-errno")
#endif

namespace openfocus {
namespace {

constexpr float kD0 = 0.125f;
constexpr float kD1 = 0.375f;
constexpr float kE0 = 0.019027505f;
constexpr float kE1 = 0.318410337f;
constexpr float kE2 = 0.559359729f;
constexpr float kE3 = 0.103202470f;
constexpr float kPointOne = 0.1f;
constexpr float kWScale = 21845.0f;
constexpr float kF65535 = 65535.0f;
constexpr float kInv257 = static_cast<float>(1.0 / 257.0);

constexpr float kEvenWeights[5] = {kE0, kE1, kE2, kE3, 0.0f};
constexpr float kOddWeights[5] = {0.0f, kE3, kE2, kE1, kE0};

inline int64_t clamp_idx(int64_t v, int64_t hi) {
    return v < 0 ? 0 : (v > hi ? hi : v);
}

inline float read_raw_scaled(const ArrayView& v, int64_t y, int64_t x, int64_t c, float factor) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 1) {
        return static_cast<float>(static_cast<const uint8_t*>(v.data)[off]) * factor;
    }
    return static_cast<float>(static_cast<const uint16_t*>(v.data)[off]) * factor;
}

inline float read_f32(const ArrayView& v, int64_t y, int64_t x, int64_t c) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + c * v.stride[2];
    return static_cast<const float*>(v.data)[off];
}

inline void write_f32(const ArrayView& v, int64_t y, int64_t x, int64_t c, float val) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + c * v.stride[2];
    static_cast<float*>(v.data)[off] = val;
}

template <bool ScaledRaw>
void down_horizontal_impl(
    const ArrayView& source,
    const ArrayView& scratch,
    int64_t origin,
    int64_t rows,
    float factor
) {
    const int64_t width = source.shape[1];
    const int64_t channels = (source.ndim == 2) ? 1 : source.shape[2];
    const int64_t out_w = scratch.shape[1];
    const int64_t ix_stop = (width >= 3) ? std::min(out_w, (width - 1) / 2) : 0;

    auto read_s = [&](int64_t y, int64_t x, int64_t c) -> float {
        if constexpr (ScaledRaw) {
            return read_raw_scaled(source, y, x, c, factor);
        } else {
            return read_f32(source, y, x, c);
        }
    };

    const bool fast_src_3c = (channels == 3 && ix_stop > 1 && source.ndim == 3
                              && source.stride[2] == 1 && source.stride[1] == 3
                              && scratch.stride[2] == 1 && scratch.stride[1] == 3);

    for (int64_t row = 0; row < rows; ++row) {
        const int64_t y = row + origin;
        float* h_row = static_cast<float*>(scratch.data) + row * scratch.stride[0];

        if (fast_src_3c) {
            const int64_t sx0 = 0;
            const int64_t sx1 = std::min(width - 1, int64_t{0});
            const int64_t sx2 = std::min(width - 1, int64_t{1});
            const int64_t sx3 = std::min(width - 1, int64_t{2});
            for (int64_t c = 0; c < 3; ++c) {
                float v = read_s(y, sx0, c) * kD0;
                v += read_s(y, sx1, c) * kD1;
                v += read_s(y, sx2, c) * kD1;
                h_row[c] = v + read_s(y, sx3, c) * kD0;
            }
            if constexpr (!ScaledRaw) {
                const float* sr = static_cast<const float*>(source.data) + y * source.stride[0];
                for (int64_t x = 1; x < ix_stop; ++x) {
                    const float* p = sr + (2 * x - 1) * 3;
                    float* dst = h_row + x * 3;
                    for (int64_t c = 0; c < 3; ++c) {
                        float v = p[c] * kD0;
                        v += p[3 + c] * kD1;
                        v += p[6 + c] * kD1;
                        dst[c] = v + p[9 + c] * kD0;
                    }
                }
            } else if (source.itemsize == 1) {
                const uint8_t* sr = static_cast<const uint8_t*>(source.data) + y * source.stride[0];
                for (int64_t x = 1; x < ix_stop; ++x) {
                    const uint8_t* p = sr + (2 * x - 1) * 3;
                    float* dst = h_row + x * 3;
                    for (int64_t c = 0; c < 3; ++c) {
                        float v = (static_cast<float>(p[c]) * factor) * kD0;
                        v += (static_cast<float>(p[3 + c]) * factor) * kD1;
                        v += (static_cast<float>(p[6 + c]) * factor) * kD1;
                        dst[c] = v + (static_cast<float>(p[9 + c]) * factor) * kD0;
                    }
                }
            } else {
                const uint16_t* sr = static_cast<const uint16_t*>(source.data) + y * source.stride[0];
                for (int64_t x = 1; x < ix_stop; ++x) {
                    const uint16_t* p = sr + (2 * x - 1) * 3;
                    float* dst = h_row + x * 3;
                    for (int64_t c = 0; c < 3; ++c) {
                        float v = (static_cast<float>(p[c]) * factor) * kD0;
                        v += (static_cast<float>(p[3 + c]) * factor) * kD1;
                        v += (static_cast<float>(p[6 + c]) * factor) * kD1;
                        dst[c] = v + (static_cast<float>(p[9 + c]) * factor) * kD0;
                    }
                }
            }
            for (int64_t x = ix_stop; x < out_w; ++x) {
                const int64_t b = 2 * x - 1;
                const int64_t s0 = clamp_idx(b, width - 1);
                const int64_t s1 = clamp_idx(b + 1, width - 1);
                const int64_t s2 = clamp_idx(b + 2, width - 1);
                const int64_t s3 = clamp_idx(b + 3, width - 1);
                for (int64_t c = 0; c < 3; ++c) {
                    float v = read_s(y, s0, c) * kD0;
                    v += read_s(y, s1, c) * kD1;
                    v += read_s(y, s2, c) * kD1;
                    write_f32(scratch, row, x, c, v + read_s(y, s3, c) * kD0);
                }
            }
        } else {
            for (int64_t x = 0; x < out_w; ++x) {
                const int64_t b = 2 * x - 1;
                const int64_t s0 = clamp_idx(b, width - 1);
                const int64_t s1 = clamp_idx(b + 1, width - 1);
                const int64_t s2 = clamp_idx(b + 2, width - 1);
                const int64_t s3 = clamp_idx(b + 3, width - 1);
                for (int64_t c = 0; c < channels; ++c) {
                    float v = read_s(y, s0, c) * kD0;
                    v += read_s(y, s1, c) * kD1;
                    v += read_s(y, s2, c) * kD1;
                    write_f32(scratch, row, x, c, v + read_s(y, s3, c) * kD0);
                }
            }
        }
    }
}

void down_vertical_impl(
    const ArrayView& scratch,
    const ArrayView& output,
    int64_t height,
    int64_t origin,
    int64_t start,
    int64_t stop
) {
    const int64_t out_w = output.shape[1];
    const int64_t channels = output.shape[2];
    const bool packed = (scratch.stride[2] == 1 && scratch.stride[1] == channels
                         && output.stride[2] == 1 && output.stride[1] == channels);
    const int64_t elems = out_w * channels;

    for (int64_t y = start; y < stop; ++y) {
        const int64_t r0 = clamp_idx(2 * y - 1, height - 1) - origin;
        const int64_t r1 = clamp_idx(2 * y, height - 1) - origin;
        const int64_t r2 = clamp_idx(2 * y + 1, height - 1) - origin;
        const int64_t r3 = clamp_idx(2 * y + 2, height - 1) - origin;
        if (packed) {
            const float* p0 = static_cast<const float*>(scratch.data) + r0 * scratch.stride[0];
            const float* p1 = static_cast<const float*>(scratch.data) + r1 * scratch.stride[0];
            const float* p2 = static_cast<const float*>(scratch.data) + r2 * scratch.stride[0];
            const float* p3 = static_cast<const float*>(scratch.data) + r3 * scratch.stride[0];
            float* dst = static_cast<float*>(output.data) + y * output.stride[0];
            for (int64_t i = 0; i < elems; ++i) {
                float v = p0[i] * kD0;
                v += p1[i] * kD1;
                v += p2[i] * kD1;
                dst[i] = v + p3[i] * kD0;
            }
        } else {
            for (int64_t x = 0; x < out_w; ++x) {
                for (int64_t c = 0; c < channels; ++c) {
                    float v = read_f32(scratch, r0, x, c) * kD0;
                    v += read_f32(scratch, r1, x, c) * kD1;
                    v += read_f32(scratch, r2, x, c) * kD1;
                    write_f32(output, y, x, c, v + read_f32(scratch, r3, x, c) * kD0);
                }
            }
        }
    }
}

void up_horizontal_impl(
    const ArrayView& coarse,
    const ArrayView& scratch,
    int64_t origin,
    int64_t rows
) {
    const int64_t width = coarse.shape[1];
    const int64_t channels = coarse.shape[2];
    const int64_t out_w = scratch.shape[1];
    const int64_t ix_start = std::min(out_w, int64_t{4});
    const int64_t raw_stop = (width >= 3) ? std::min(out_w, 2 * (width - 2)) : ix_start;
    const int64_t pair_stop = ix_start + (std::max(int64_t{0}, raw_stop - ix_start) & ~int64_t{1});

    const bool fast_3c = (channels == 3 && pair_stop > ix_start
                          && coarse.stride[2] == 1 && coarse.stride[1] == 3
                          && scratch.stride[2] == 1 && scratch.stride[1] == 3);
    const bool fast_1c = (channels == 1 && pair_stop > ix_start
                          && coarse.stride[1] == 1 && scratch.stride[1] == 1);

    for (int64_t row = 0; row < rows; ++row) {
        const int64_t y = row + origin;
        for (int64_t x = 0; x < ix_start; ++x) {
            const float* w = (x & 1) ? kOddWeights : kEvenWeights;
            const int64_t left = x / 2 - 2;
            const int64_t first = std::max(int64_t{0}, -left);
            const int64_t last = std::min(int64_t{5}, width - left);
            float total = 0.0f;
            for (int64_t k = first; k < last; ++k) {
                total += w[k];
            }
            for (int64_t c = 0; c < channels; ++c) {
                float val = 0.0f;
                for (int64_t k = first; k < last; ++k) {
                    val += read_f32(coarse, y, left + k, c) * w[k];
                }
                write_f32(scratch, row, x, c, val / total);
            }
        }
        if (fast_3c) {
            const float* sr = static_cast<const float*>(coarse.data) + y * coarse.stride[0];
            float* dr = static_cast<float*>(scratch.data) + row * scratch.stride[0];
            for (int64_t x = ix_start; x < pair_stop; x += 2) {
                const int64_t left = x / 2 - 2;
                const float* p = sr + left * 3;
                float* d0 = dr + x * 3;
                float* d1 = dr + (x + 1) * 3;
                for (int64_t c = 0; c < 3; ++c) {
                    const float s0 = p[c];
                    const float s1 = p[3 + c];
                    const float s2 = p[6 + c];
                    const float s3 = p[9 + c];
                    const float s4 = p[12 + c];
                    float ve = (s0 * kE0 + s1 * kE1) + s2 * kE2;
                    ve = (ve + s3 * kE3) + 0.0f;
                    d0[c] = ve;
                    float vo = (0.0f + s1 * kE3) + s2 * kE2;
                    vo = (vo + s3 * kE1) + s4 * kE0;
                    d1[c] = vo;
                }
            }
        } else if (fast_1c) {
            const float* sr = static_cast<const float*>(coarse.data) + y * coarse.stride[0];
            float* dr = static_cast<float*>(scratch.data) + row * scratch.stride[0];
            for (int64_t x = ix_start; x < pair_stop; x += 2) {
                const int64_t left = x / 2 - 2;
                const float* p = sr + left;
                float ve = (p[0] * kE0 + p[1] * kE1) + p[2] * kE2;
                ve = (ve + p[3] * kE3);
                dr[x] = ve;
                float vo = (p[1] * kE3 + p[2] * kE2);
                vo = (vo + p[3] * kE1) + p[4] * kE0;
                dr[x + 1] = vo;
            }
        } else if (pair_stop > ix_start) {
            for (int64_t x = ix_start; x < pair_stop; ++x) {
                const float* w = (x & 1) ? kOddWeights : kEvenWeights;
                const int64_t left = x / 2 - 2;
                for (int64_t c = 0; c < channels; ++c) {
                    float val = 0.0f;
                    for (int k = 0; k < 5; ++k) {
                        val += read_f32(coarse, y, left + k, c) * w[k];
                    }
                    write_f32(scratch, row, x, c, val);
                }
            }
        }
        for (int64_t x = pair_stop; x < out_w; ++x) {
            const float* w = (x & 1) ? kOddWeights : kEvenWeights;
            const int64_t left = x / 2 - 2;
            const int64_t first = std::max(int64_t{0}, -left);
            const int64_t last = std::min(int64_t{5}, width - left);
            float total = 0.0f;
            for (int64_t k = first; k < last; ++k) {
                total += w[k];
            }
            for (int64_t c = 0; c < channels; ++c) {
                float val = 0.0f;
                for (int64_t k = first; k < last; ++k) {
                    val += read_f32(coarse, y, left + k, c) * w[k];
                }
                write_f32(scratch, row, x, c, val / total);
            }
        }
    }
}

inline uint16_t encode_weight(float strength, int64_t channels, bool is_border) {
    if (is_border) {
        return 0;
    }
    if (channels == 1) {
        strength *= 3.0f;
    } else if (channels == 2) {
        strength *= 1.5f;
    }
    const float scaled = strength * kWScale;
    if (scaled >= 65535.0f) {
        return 65535;
    }
    if (scaled <= 0.0f) {
        return 0;
    }
    return static_cast<uint16_t>(static_cast<int32_t>(scaled));
}

template <bool ScaledRaw, bool UpdateWeight, bool WriteDetailInPlace>
void synth_detail_band_impl(
    const ArrayView& horizontal,
    const ArrayView& target_or_raw,
    const ArrayView& weight,
    int64_t coarse_height,
    int64_t origin,
    float factor,
    int64_t start,
    int64_t stop
) {
    const int64_t height = target_or_raw.shape[0];
    const int64_t width = target_or_raw.shape[1];
    const int64_t channels = (target_or_raw.ndim == 2) ? 1 : target_or_raw.shape[2];
    const bool h_packed3 = (channels == 3 && horizontal.stride[2] == 1 && horizontal.stride[1] == 3);
    const bool t_packed3 = (channels == 3 && target_or_raw.ndim == 3
                            && target_or_raw.stride[2] == 1 && target_or_raw.stride[1] == 3);
    uint16_t* w_base = static_cast<uint16_t*>(weight.data);
    const int64_t w_sy = weight.stride[0];

    for (int64_t y = start; y < stop; ++y) {
        const int64_t top = y / 2 - 2;
        const int odd = static_cast<int>(y & 1);
        const float* w = odd ? kOddWeights : kEvenWeights;
        const int64_t first = std::max(int64_t{0}, -top);
        const int64_t last = std::min(int64_t{5}, coarse_height - top);
        const bool y_border = (height > 2) && (y == 0 || y == height - 1);
        uint16_t* w_row = w_base + y * w_sy;

        if (first == 0 && last == 5 && channels == 3 && h_packed3 && t_packed3) {
            const float* hr0 = static_cast<const float*>(horizontal.data) + (top - origin) * horizontal.stride[0];
            const float* hr1 = hr0 + horizontal.stride[0];
            const float* hr2 = hr1 + horizontal.stride[0];
            const float* hr3 = hr2 + horizontal.stride[0];
            const float* hr4 = hr3 + horizontal.stride[0];
            const float w0 = odd ? 0.0f : kE0;
            const float w1 = odd ? kE3 : kE1;
            const float w2 = kE2;
            const float w3 = odd ? kE1 : kE3;
            const float w4 = odd ? kE0 : 0.0f;

            for (int64_t x = 0; x < width; ++x) {
                const int64_t off3 = x * 3;
                float lo0 = (hr0[off3] * w0 + hr1[off3] * w1) + hr2[off3] * w2;
                float lo1 = (hr0[off3 + 1] * w0 + hr1[off3 + 1] * w1) + hr2[off3 + 1] * w2;
                float lo2 = (hr0[off3 + 2] * w0 + hr1[off3 + 2] * w1) + hr2[off3 + 2] * w2;
                lo0 = (lo0 + hr3[off3] * w3) + hr4[off3] * w4;
                lo1 = (lo1 + hr3[off3 + 1] * w3) + hr4[off3 + 1] * w4;
                lo2 = (lo2 + hr3[off3 + 2] * w3) + hr4[off3 + 2] * w4;

                float s0, s1, s2;
                if constexpr (ScaledRaw) {
                    if (target_or_raw.itemsize == 1) {
                        const uint8_t* p = static_cast<const uint8_t*>(target_or_raw.data)
                                         + y * target_or_raw.stride[0] + off3;
                        s0 = static_cast<float>(p[0]) * factor;
                        s1 = static_cast<float>(p[1]) * factor;
                        s2 = static_cast<float>(p[2]) * factor;
                    } else {
                        const uint16_t* p = static_cast<const uint16_t*>(target_or_raw.data)
                                          + y * target_or_raw.stride[0] + off3;
                        s0 = static_cast<float>(p[0]) * factor;
                        s1 = static_cast<float>(p[1]) * factor;
                        s2 = static_cast<float>(p[2]) * factor;
                    }
                } else {
                    const float* p = static_cast<const float*>(target_or_raw.data)
                                   + y * target_or_raw.stride[0] + off3;
                    s0 = p[0];
                    s1 = p[1];
                    s2 = p[2];
                }

                const float v0 = (s0 - lo0) / (lo0 + kPointOne);
                const float v1 = (s1 - lo1) / (lo1 + kPointOne);
                const float v2 = (s2 - lo2) / (lo2 + kPointOne);
                if constexpr (WriteDetailInPlace) {
                    float* dst = static_cast<float*>(target_or_raw.data)
                               + y * target_or_raw.stride[0] + off3;
                    dst[0] = v0;
                    dst[1] = v1;
                    dst[2] = v2;
                }
                if constexpr (UpdateWeight) {
                    const bool is_border = y_border || (width > 2 && (x == 0 || x == width - 1));
                    const float strength = (std::abs(v0) + std::abs(v1)) + std::abs(v2);
                    w_row[x] = encode_weight(strength, 3, is_border);
                }
            }
        } else {
            float total = 0.0f;
            for (int64_t k = first; k < last; ++k) {
                total += w[k];
            }
            for (int64_t x = 0; x < width; ++x) {
                float strength = 0.0f;
                for (int64_t c = 0; c < channels; ++c) {
                    float lower = 0.0f;
                    for (int64_t k = first; k < last; ++k) {
                        lower += read_f32(horizontal, top + k - origin, x, c) * w[k];
                    }
                    lower /= total;
                    const float sample = ScaledRaw
                        ? read_raw_scaled(target_or_raw, y, x, c, factor)
                        : read_f32(target_or_raw, y, x, c);
                    const float value = (sample - lower) / (lower + kPointOne);
                    if constexpr (WriteDetailInPlace) {
                        write_f32(target_or_raw, y, x, c, value);
                    }
                    if constexpr (UpdateWeight) {
                        if (c < 3) {
                            strength += std::abs(value);
                        }
                    }
                }
                if constexpr (UpdateWeight) {
                    const bool is_border = y_border || (width > 2 && (x == 0 || x == width - 1));
                    w_row[x] = encode_weight(strength, channels, is_border);
                }
            }
        }
    }
}

}  // namespace

void of_pyr_down_from_raw_cpp(
    const ArrayView& source,
    const ArrayView& output,
    const ArrayView& scratch,
    float factor,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || source.shape[0] == 0 || source.shape[1] == 0) {
        return;
    }
    const int64_t height = source.shape[0];
    const int64_t origin = std::max(int64_t{0}, 2 * start - 1);
    const int64_t end = std::min(height, 2 * stop + 1);
    down_horizontal_impl<true>(source, scratch, origin, end - origin, factor);
    down_vertical_impl(scratch, output, height, origin, start, stop);
}

void of_pyr_down_f32_cpp(
    const ArrayView& source,
    const ArrayView& output,
    const ArrayView& scratch,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || source.shape[0] == 0 || source.shape[1] == 0) {
        return;
    }
    const int64_t height = source.shape[0];
    const int64_t origin = std::max(int64_t{0}, 2 * start - 1);
    const int64_t end = std::min(height, 2 * stop + 1);
    down_horizontal_impl<false>(source, scratch, origin, end - origin, 1.0f);
    down_vertical_impl(scratch, output, height, origin, start, stop);
}

void of_pyr_level0_weights_from_raw_cpp(
    const ArrayView& coarse,
    const ArrayView& source,
    const ArrayView& weight,
    const ArrayView& scratch,
    float factor,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || source.shape[0] == 0 || source.shape[1] == 0) {
        return;
    }
    const int64_t coarse_height = coarse.shape[0];
    const int64_t origin = std::max(int64_t{0}, start / 2 - 2);
    const int64_t end = std::min(coarse_height, (stop - 1) / 2 + 3);
    up_horizontal_impl(coarse, scratch, origin, end - origin);
    synth_detail_band_impl<true, true, false>(
        scratch, source, weight, coarse_height, origin, factor, start, stop);
}

void of_pyr_level0_merge_from_raw_cpp(
    const ArrayView& coarse,
    const ArrayView& source,
    const ArrayView& weight,
    const ArrayView& accumulator,
    const ArrayView& best,
    uint16_t* winner,
    int64_t winner_sy,
    const ArrayView& scratch,
    float factor,
    bool first,
    uint16_t index,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || source.shape[0] == 0 || source.shape[1] == 0) {
        return;
    }
    const int64_t width = weight.shape[1];
    const int64_t channels = accumulator.shape[2];
    const int64_t coarse_height = coarse.shape[0];
    const uint16_t* w_base = static_cast<const uint16_t*>(weight.data);
    const int64_t w_sy = weight.stride[0];
    uint16_t* b_base = static_cast<uint16_t*>(best.data);
    const int64_t b_sy = best.stride[0];

    bool any_active = first;
    if (!first) {
        for (int64_t y = start; y < stop && !any_active; ++y) {
            const uint16_t* wr = w_base + y * w_sy;
            const uint16_t* br = b_base + y * b_sy;
            for (int64_t x = 0; x < width; ++x) {
                if (wr[x] > br[x]) {
                    any_active = true;
                    break;
                }
            }
        }
    }
    if (!any_active) {
        return;
    }

    const int64_t origin = std::max(int64_t{0}, start / 2 - 2);
    const int64_t end = std::min(coarse_height, (stop - 1) / 2 + 3);
    up_horizontal_impl(coarse, scratch, origin, end - origin);

    const bool h_packed3 = (channels == 3 && scratch.stride[2] == 1 && scratch.stride[1] == 3);
    const bool a_packed3 = (channels == 3 && accumulator.stride[2] == 1 && accumulator.stride[1] == 3);
    const bool s_packed3 = (channels == 3 && source.ndim == 3 && source.stride[2] == 1 && source.stride[1] == 3);

    for (int64_t y = start; y < stop; ++y) {
        const uint16_t* wr = w_base + y * w_sy;
        uint16_t* br = b_base + y * b_sy;
        if (!first) {
            bool row_active = false;
            for (int64_t x = 0; x < width; ++x) {
                if (wr[x] > br[x]) {
                    row_active = true;
                    break;
                }
            }
            if (!row_active) {
                continue;
            }
        }

        const int64_t top = y / 2 - 2;
        const int odd = static_cast<int>(y & 1);
        const float* wt = odd ? kOddWeights : kEvenWeights;
        const int64_t first_k = std::max(int64_t{0}, -top);
        const int64_t last_k = std::min(int64_t{5}, coarse_height - top);
        uint16_t* win_row = (winner != nullptr) ? (winner + y * winner_sy) : nullptr;

        if (first_k == 0 && last_k == 5 && channels == 3 && h_packed3 && a_packed3 && s_packed3) {
            const float* hr0 = static_cast<const float*>(scratch.data) + (top - origin) * scratch.stride[0];
            const float* hr1 = hr0 + scratch.stride[0];
            const float* hr2 = hr1 + scratch.stride[0];
            const float* hr3 = hr2 + scratch.stride[0];
            const float* hr4 = hr3 + scratch.stride[0];
            float* acc_row = static_cast<float*>(accumulator.data) + y * accumulator.stride[0];
            const float w0 = odd ? 0.0f : kE0;
            const float w1 = odd ? kE3 : kE1;
            const float w2 = kE2;
            const float w3 = odd ? kE1 : kE3;
            const float w4 = odd ? kE0 : 0.0f;

            for (int64_t x = 0; x < width; ++x) {
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
                if (source.itemsize == 1) {
                    const uint8_t* sp = static_cast<const uint8_t*>(source.data)
                                      + y * source.stride[0] + off3;
                    s0 = static_cast<float>(sp[0]) * factor;
                    s1 = static_cast<float>(sp[1]) * factor;
                    s2 = static_cast<float>(sp[2]) * factor;
                } else {
                    const uint16_t* sp = static_cast<const uint16_t*>(source.data)
                                       + y * source.stride[0] + off3;
                    s0 = static_cast<float>(sp[0]) * factor;
                    s1 = static_cast<float>(sp[1]) * factor;
                    s2 = static_cast<float>(sp[2]) * factor;
                }
                acc_row[off3] = (s0 - lo0) / (lo0 + kPointOne);
                acc_row[off3 + 1] = (s1 - lo1) / (lo1 + kPointOne);
                acc_row[off3 + 2] = (s2 - lo2) / (lo2 + kPointOne);
            }
        } else {
            float total = 0.0f;
            for (int64_t k = first_k; k < last_k; ++k) {
                total += wt[k];
            }
            for (int64_t x = 0; x < width; ++x) {
                const uint16_t wval = wr[x];
                if (!first && wval <= br[x]) {
                    continue;
                }
                br[x] = wval;
                if (win_row != nullptr) {
                    win_row[x] = index;
                }
                for (int64_t c = 0; c < channels; ++c) {
                    float lower = 0.0f;
                    for (int64_t k = first_k; k < last_k; ++k) {
                        lower += read_f32(scratch, top + k - origin, x, c) * wt[k];
                    }
                    lower /= total;
                    const float sample = read_raw_scaled(source, y, x, c, factor);
                    write_f32(accumulator, y, x, c, (sample - lower) / (lower + kPointOne));
                }
            }
        }
    }
}

void of_pyr_detail_weight_f32_cpp(
    const ArrayView& coarse,
    const ArrayView& target,
    const ArrayView& weight,
    const ArrayView& scratch,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || target.shape[0] == 0 || target.shape[1] == 0) {
        return;
    }
    const int64_t coarse_height = coarse.shape[0];
    const int64_t origin = std::max(int64_t{0}, start / 2 - 2);
    const int64_t end = std::min(coarse_height, (stop - 1) / 2 + 3);
    up_horizontal_impl(coarse, scratch, origin, end - origin);
    synth_detail_band_impl<false, true, true>(
        scratch, target, weight, coarse_height, origin, 1.0f, start, stop);
}

void of_pyr_merge_f32_cpp(
    const ArrayView& detail,
    const ArrayView& weight,
    const ArrayView& accumulator,
    const ArrayView& best,
    uint16_t* winner,
    int64_t winner_sy,
    bool first,
    uint16_t index,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || weight.shape[1] == 0) {
        return;
    }
    const int64_t width = weight.shape[1];
    const int64_t channels = detail.shape[2];
    const uint16_t* w_base = static_cast<const uint16_t*>(weight.data);
    const int64_t w_sy = weight.stride[0];
    uint16_t* b_base = static_cast<uint16_t*>(best.data);
    const int64_t b_sy = best.stride[0];
    const bool packed3 = (channels == 3 && detail.stride[2] == 1 && detail.stride[1] == 3
                          && accumulator.stride[2] == 1 && accumulator.stride[1] == 3);

    for (int64_t y = start; y < stop; ++y) {
        const uint16_t* wr = w_base + y * w_sy;
        uint16_t* br = b_base + y * b_sy;
        uint16_t* win_row = (winner != nullptr) ? (winner + y * winner_sy) : nullptr;
        if (first) {
            std::memcpy(br, wr, static_cast<size_t>(width) * sizeof(uint16_t));
            if (win_row != nullptr) {
                std::fill_n(win_row, width, index);
            }
            if (packed3) {
                const float* dr = static_cast<const float*>(detail.data) + y * detail.stride[0];
                float* ar = static_cast<float*>(accumulator.data) + y * accumulator.stride[0];
                std::memcpy(ar, dr, static_cast<size_t>(width * 3) * sizeof(float));
            } else {
                for (int64_t x = 0; x < width; ++x) {
                    for (int64_t c = 0; c < channels; ++c) {
                        write_f32(accumulator, y, x, c, read_f32(detail, y, x, c));
                    }
                }
            }
        } else if (packed3) {
            const float* dr = static_cast<const float*>(detail.data) + y * detail.stride[0];
            float* ar = static_cast<float*>(accumulator.data) + y * accumulator.stride[0];
            for (int64_t x = 0; x < width; ++x) {
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
        } else {
            for (int64_t x = 0; x < width; ++x) {
                const uint16_t wval = wr[x];
                if (wval > br[x]) {
                    br[x] = wval;
                    if (win_row != nullptr) {
                        win_row[x] = index;
                    }
                    for (int64_t c = 0; c < channels; ++c) {
                        write_f32(accumulator, y, x, c, read_f32(detail, y, x, c));
                    }
                }
            }
        }
    }
}

void of_pyr_reconstruct_f32_cpp(
    const ArrayView& coarse,
    const ArrayView& target,
    const ArrayView& scratch,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || target.shape[0] == 0 || target.shape[1] == 0) {
        return;
    }
    const int64_t width = target.shape[1];
    const int64_t channels = target.shape[2];
    const int64_t coarse_height = coarse.shape[0];
    const int64_t origin = std::max(int64_t{0}, start / 2 - 2);
    const int64_t end = std::min(coarse_height, (stop - 1) / 2 + 3);
    up_horizontal_impl(coarse, scratch, origin, end - origin);

    const bool packed = (scratch.stride[2] == 1 && scratch.stride[1] == channels
                         && target.stride[2] == 1 && target.stride[1] == channels);
    const int64_t elems = width * channels;

    for (int64_t y = start; y < stop; ++y) {
        const int64_t top = y / 2 - 2;
        const int odd = static_cast<int>(y & 1);
        const float* wt = odd ? kOddWeights : kEvenWeights;
        const int64_t first = std::max(int64_t{0}, -top);
        const int64_t last = std::min(int64_t{5}, coarse_height - top);

        if (first == 0 && last == 5 && packed) {
            const float* hr0 = static_cast<const float*>(scratch.data) + (top - origin) * scratch.stride[0];
            const float* hr1 = hr0 + scratch.stride[0];
            const float* hr2 = hr1 + scratch.stride[0];
            const float* hr3 = hr2 + scratch.stride[0];
            const float* hr4 = hr3 + scratch.stride[0];
            float* tr = static_cast<float*>(target.data) + y * target.stride[0];
            const float w0 = odd ? 0.0f : kE0;
            const float w1 = odd ? kE3 : kE1;
            const float w2 = kE2;
            const float w3 = odd ? kE1 : kE3;
            const float w4 = odd ? kE0 : 0.0f;
            for (int64_t i = 0; i < elems; ++i) {
                float lo = (hr0[i] * w0 + hr1[i] * w1) + hr2[i] * w2;
                lo = (lo + hr3[i] * w3) + hr4[i] * w4;
                const float val = (lo + kPointOne) * tr[i] + lo;
                tr[i] = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
            }
        } else {
            float total = 0.0f;
            for (int64_t k = first; k < last; ++k) {
                total += wt[k];
            }
            for (int64_t x = 0; x < width; ++x) {
                for (int64_t c = 0; c < channels; ++c) {
                    float lo = 0.0f;
                    for (int64_t k = first; k < last; ++k) {
                        lo += read_f32(scratch, top + k - origin, x, c) * wt[k];
                    }
                    lo /= total;
                    const float val = (lo + kPointOne) * read_f32(target, y, x, c) + lo;
                    write_f32(target, y, x, c, val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val));
                }
            }
        }
    }
}

void of_pyr_output_f32_cpp(
    const ArrayView& source,
    const ArrayView& output,
    bool is_u8,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || source.shape[1] == 0) {
        return;
    }
    const int64_t width = source.shape[1];
    const int64_t channels = source.shape[2];
    const bool packed = (source.stride[2] == 1 && source.stride[1] == channels
                         && output.stride[2] == 1 && output.stride[1] == channels);
    const int64_t elems = width * channels;

    for (int64_t y = start; y < stop; ++y) {
        if (packed) {
            const float* sr = static_cast<const float*>(source.data) + y * source.stride[0];
            if (is_u8) {
                uint8_t* dr = static_cast<uint8_t*>(output.data) + y * output.stride[0];
                for (int64_t i = 0; i < elems; ++i) {
                    const float v = sr[i] < 0.0f ? 0.0f : (sr[i] > 1.0f ? 1.0f : sr[i]);
                    dr[i] = static_cast<uint8_t>(static_cast<int32_t>(v * kF65535) >> 8);
                }
            } else {
                float* dr = static_cast<float*>(output.data) + y * output.stride[0];
                for (int64_t i = 0; i < elems; ++i) {
                    const float v = sr[i] < 0.0f ? 0.0f : (sr[i] > 1.0f ? 1.0f : sr[i]);
                    dr[i] = std::nearbyintf(v * kF65535) * kInv257;
                }
            }
        } else {
            for (int64_t x = 0; x < width; ++x) {
                for (int64_t c = 0; c < channels; ++c) {
                    const float s = read_f32(source, y, x, c);
                    const float v = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
                    const float code = v * kF65535;
                    if (is_u8) {
                        const int64_t off = y * output.stride[0] + x * output.stride[1] + c * output.stride[2];
                        static_cast<uint8_t*>(output.data)[off] =
                            static_cast<uint8_t>(static_cast<int32_t>(code) >> 8);
                    } else {
                        write_f32(output, y, x, c, std::nearbyintf(code) * kInv257);
                    }
                }
            }
        }
    }
}

}  // namespace openfocus
