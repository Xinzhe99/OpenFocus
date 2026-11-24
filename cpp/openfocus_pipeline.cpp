#include "openfocus_pipeline.hpp"
#include "simd_dispatch.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#define OPENFOCUS_HAS_SSE2 1
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-math-errno")
#endif

namespace openfocus {
namespace {

constexpr float kSupport = 1.9989999532699585f;
constexpr float kLUT[64] = {
    1.000000000f,  0.997640193f,  0.990738153f,  0.979559839f,
    0.964371324f,  0.945438623f,  0.923027754f,  0.897404790f,
    0.868835747f,  0.837586582f,  0.803923428f,  0.768112183f,
    0.730419040f,  0.691109836f,  0.650450766f,  0.608707786f,
    0.566146910f,  0.523034155f,  0.479635596f,  0.436217219f,
    0.393045157f,  0.350385249f,  0.308503628f,  0.267666399f,
    0.228139445f,  0.190188855f,  0.154080644f,  0.120080866f,
    0.088455595f,  0.059470784f,  0.033392422f,  0.010486591f,
    -0.009118127f, -0.026175674f, -0.040942304f, -0.053535335f,
    -0.064072095f, -0.072669916f, -0.079446115f, -0.084518000f,
    -0.088002943f, -0.090018235f, -0.090681210f, -0.090109192f,
    -0.088419497f, -0.085729480f, -0.082156427f, -0.077817686f,
    -0.072830573f, -0.067312412f, -0.061380532f, -0.055152260f,
    -0.048744921f, -0.042275835f, -0.035862334f, -0.029621759f,
    -0.023671394f, -0.018128583f, -0.013110672f, -0.008734948f,
    -0.005118754f, -0.002379417f, -0.000634261f, -0.000000612f,
};

inline float lut_weight(float argument) {
    const float position = (std::fabs(argument) / kSupport) * 63.0f;
    if (position >= 63.0f) {
        return kLUT[63];
    }
    const int lower = static_cast<int>(position);
    const float fraction = position - static_cast<float>(lower);
    const float first = (1.0f - fraction) * kLUT[lower];
    const float second = fraction * kLUT[lower + 1];
    return first + second;
}

inline int64_t read_u_idx(const ArrayView& v, int64_t y, int64_t x, int64_t c = 0) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 1) {
        return static_cast<const uint8_t*>(v.data)[off];
    }
    return static_cast<const uint16_t*>(v.data)[off];
}

inline void write_sample(const ArrayView& v, int64_t y, int64_t x, int64_t c, int64_t val) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 1) {
        static_cast<uint8_t*>(v.data)[off] = static_cast<uint8_t>(val);
    } else if (v.itemsize == 2) {
        static_cast<uint16_t*>(v.data)[off] = static_cast<uint16_t>(val);
    } else {
        static_cast<int32_t*>(v.data)[off] = static_cast<int32_t>(val);
    }
}

inline float read_f32_2d3d(const ArrayView& v, int64_t y, int64_t x, int64_t c) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    return static_cast<const float*>(v.data)[off];
}

inline double read_f64_or_f32(const ArrayView& v, int64_t y, int64_t x, int64_t c) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    if (v.itemsize == 4) {
        return static_cast<double>(static_cast<const float*>(v.data)[off]);
    }
    return static_cast<const double*>(v.data)[off];
}

inline void write_f32_2d3d(const ArrayView& v, int64_t y, int64_t x, int64_t c, float val) {
    const int64_t off = y * v.stride[0] + x * v.stride[1] + (v.ndim == 3 ? c * v.stride[2] : 0);
    static_cast<float*>(v.data)[off] = val;
}

template <typename SampleT, typename TableT>
void table_rows_typed(
    const ArrayView& image,
    const ArrayView& table,
    const ArrayView& output,
    int64_t start,
    int64_t stop
) {
    const int64_t width = image.shape[1];
    const auto* img_base = static_cast<const SampleT*>(image.data);
    const auto* tbl_base = static_cast<const TableT*>(table.data);
    auto* out_base = static_cast<TableT*>(output.data);
    const int64_t ts0 = table.stride[0];
    const int64_t ts1 = table.stride[1];

    if (image.ndim == 2) {
        const bool fast = (image.stride[1] == 1 && output.stride[1] == 1 && ts1 == 1 && ts0 == 1);
        for (int64_t y = start; y < stop; ++y) {
            const SampleT* ir = img_base + y * image.stride[0];
            TableT* orow = out_base + y * output.stride[0];
            if (fast) {
                for (int64_t x = 0; x < width; ++x) {
                    orow[x] = tbl_base[ir[x]];
                }
            } else {
                for (int64_t x = 0; x < width; ++x) {
                    orow[x * output.stride[1]] = tbl_base[ir[x * image.stride[1]] * ts0];
                }
            }
        }
        return;
    }

    const int64_t channels = image.shape[2];
    if (channels == 3) {
        const bool fast = (image.stride[1] == 3 && image.stride[2] == 1
                           && output.stride[1] == 3 && output.stride[2] == 1
                           && ts1 == 1 && ts0 == 3);
        for (int64_t y = start; y < stop; ++y) {
            const SampleT* ir = img_base + y * image.stride[0];
            TableT* orow = out_base + y * output.stride[0];
            if (fast) {
                for (int64_t x = 0; x < width; ++x) {
                    const int64_t off = x * 3;
                    orow[off] = tbl_base[ir[off] * 3];
                    orow[off + 1] = tbl_base[ir[off + 1] * 3 + 1];
                    orow[off + 2] = tbl_base[ir[off + 2] * 3 + 2];
                }
            } else {
                for (int64_t x = 0; x < width; ++x) {
                    const SampleT* px = ir + x * image.stride[1];
                    TableT* op = orow + x * output.stride[1];
                    op[0] = tbl_base[px[0] * ts0];
                    op[output.stride[2]] = tbl_base[px[image.stride[2]] * ts0 + ts1];
                    op[2 * output.stride[2]] = tbl_base[px[2 * image.stride[2]] * ts0 + 2 * ts1];
                }
            }
        }
        return;
    }

    for (int64_t y = start; y < stop; ++y) {
        const SampleT* ir = img_base + y * image.stride[0];
        TableT* orow = out_base + y * output.stride[0];
        for (int64_t x = 0; x < width; ++x) {
            const SampleT* px = ir + x * image.stride[1];
            TableT* op = orow + x * output.stride[1];
            for (int64_t c = 0; c < channels; ++c) {
                op[c * output.stride[2]] = tbl_base[px[c * image.stride[2]] * ts0 + c * ts1];
            }
        }
    }
}

}  // namespace

void of_table_rows_cpp(
    const ArrayView& image,
    const ArrayView& table,
    const ArrayView& output,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || image.shape[0] == 0 || image.shape[1] == 0) {
        return;
    }
    if (image.itemsize == 1 && table.itemsize == 1) {
        table_rows_typed<uint8_t, uint8_t>(image, table, output, start, stop);
    } else if (image.itemsize == 2 && table.itemsize == 2) {
        table_rows_typed<uint16_t, uint16_t>(image, table, output, start, stop);
    } else {
        for (int64_t y = start; y < stop; ++y) {
            if (image.ndim == 2) {
                for (int64_t x = 0; x < image.shape[1]; ++x) {
                    const int64_t code = read_u_idx(image, y, x);
                    const int64_t val = (table.itemsize == 1)
                        ? static_cast<const uint8_t*>(table.data)[code * table.stride[0]]
                        : ((table.itemsize == 2)
                            ? static_cast<const uint16_t*>(table.data)[code * table.stride[0]]
                            : static_cast<const int32_t*>(table.data)[code * table.stride[0]]);
                    write_sample(output, y, x, 0, val);
                }
            } else {
                const int64_t channels = image.shape[2];
                for (int64_t x = 0; x < image.shape[1]; ++x) {
                    for (int64_t c = 0; c < channels; ++c) {
                        const int64_t code = read_u_idx(image, y, x, c);
                        const int64_t t_off = code * table.stride[0] + c * table.stride[1];
                        const int64_t val = (table.itemsize == 1)
                            ? static_cast<const uint8_t*>(table.data)[t_off]
                            : ((table.itemsize == 2)
                                ? static_cast<const uint16_t*>(table.data)[t_off]
                                : static_cast<const int32_t*>(table.data)[t_off]);
                        write_sample(output, y, x, c, val);
                    }
                }
            }
        }
    }
}

void of_fill_half_axis_cpp(
    int64_t source_length,
    int64_t target_length,
    int64_t* indices,
    float* weights,
    int64_t* counts
) {
    const float radius = kSupport / 0.5f;
    for (int64_t pos = 0; pos < target_length; ++pos) {
        const float center = static_cast<float>(2 * pos) + 0.5f;
        const int64_t first = static_cast<int64_t>(std::floor(center - radius));
        const int64_t last = static_cast<int64_t>(std::ceil(center + radius));
        int64_t count = 0;
        float total = 0.0f;
        for (int64_t tap = first; tap <= last; ++tap) {
            const float argument = (center - static_cast<float>(tap)) * 0.5f;
            if (tap < 0 || tap >= source_length || std::fabs(argument) >= kSupport) {
                continue;
            }
            const float raw = lut_weight(argument) / 2.0f;
            indices[pos * 8 + count] = tap;
            weights[pos * 8 + count] = raw;
            total += raw;
            count += 1;
        }
        counts[pos] = count;
        const float reciprocal = 1.0f / total;
        for (int64_t tap = 0; tap < count; ++tap) {
            weights[pos * 8 + tap] *= reciprocal;
        }
    }
}

void of_fill_linear_axis_cpp(
    int64_t source_length,
    int64_t target_length,
    int64_t* indices,
    float* weights,
    int64_t* counts
) {
    const float scale = static_cast<float>(target_length) / static_cast<float>(source_length);
    const float offset = (0.5f / scale) - 0.5f;
    for (int64_t pos = 0; pos < target_length; ++pos) {
        const float phase = (static_cast<float>(pos) / scale) + offset;
        const int64_t lower = static_cast<int64_t>(std::floor(phase));
        const float fraction = phase - static_cast<float>(lower);
        indices[pos * 2 + 0] = std::min<int64_t>(source_length - 1, std::max<int64_t>(0, lower));
        indices[pos * 2 + 1] = std::min<int64_t>(source_length - 1, std::max<int64_t>(0, lower + 1));
        weights[pos * 2 + 0] = 1.0f - fraction;
        weights[pos * 2 + 1] = fraction;
        counts[pos] = 2;
    }
}

double of_exposure_mean_cpp(
    const ArrayView& image,
    bool normalize_uint16,
    int num_threads
) {
    const int64_t height = image.shape[0];
    const int64_t width = image.shape[1];
    if (height <= 0 || width <= 0) {
        return std::nan("");
    }
    const int64_t channels = (image.ndim == 2) ? 1 : image.shape[2];
    const int64_t take_c = std::min<int64_t>(3, channels);
    const int64_t count = height * width * take_c;
    const int64_t band = 128;
    const int64_t n_tasks = (height + band - 1) / band;
    const int threads = (count >= 262144) ? std::max(1, num_threads) : 1;

    std::vector<uint64_t> partials(static_cast<size_t>(n_tasks), 0ULL);
    if (image.itemsize == 1) {
        const auto* base = static_cast<const uint8_t*>(image.data);
        parallel_bands(height, band, threads, [&](int64_t start, int64_t stop) {
            uint64_t local = 0ULL;
            for (int64_t y = start; y < stop; ++y) {
                const uint8_t* row = base + y * image.stride[0];
                uint64_t row_sum = 0ULL;
                if (image.ndim == 2 && image.stride[1] == 1) {
                    for (int64_t x = 0; x < width; ++x) {
                        row_sum += row[x];
                    }
                } else if (channels == 3 && image.stride[1] == 3 && (image.stride[2] == 1 || image.stride[2] == -1)) {
                    const uint8_t* rp = (image.stride[2] == 1) ? row : (row - 2);
                    const int64_t elems = width * 3;
                    for (int64_t i = 0; i < elems; ++i) {
                        row_sum += rp[i];
                    }
                } else {
                    for (int64_t x = 0; x < width; ++x) {
                        const uint8_t* px = row + x * image.stride[1];
                        for (int64_t c = 0; c < take_c; ++c) {
                            row_sum += px[(image.ndim == 3 ? c * image.stride[2] : 0)];
                        }
                    }
                }
                local += row_sum;
            }
            partials[static_cast<size_t>(start / band)] = local;
        });
    } else {
        const auto* base = static_cast<const uint16_t*>(image.data);
        parallel_bands(height, band, threads, [&](int64_t start, int64_t stop) {
            uint64_t local = 0ULL;
            for (int64_t y = start; y < stop; ++y) {
                const uint16_t* row = base + y * image.stride[0];
                uint64_t row_sum = 0ULL;
                if (image.ndim == 2 && image.stride[1] == 1) {
                    for (int64_t x = 0; x < width; ++x) {
                        row_sum += row[x];
                    }
                } else if (channels == 3 && image.stride[1] == 3 && (image.stride[2] == 1 || image.stride[2] == -1)) {
                    const uint16_t* rp = (image.stride[2] == 1) ? row : (row - 2);
                    const int64_t elems = width * 3;
                    for (int64_t i = 0; i < elems; ++i) {
                        row_sum += rp[i];
                    }
                } else {
                    for (int64_t x = 0; x < width; ++x) {
                        const uint16_t* px = row + x * image.stride[1];
                        for (int64_t c = 0; c < take_c; ++c) {
                            row_sum += px[(image.ndim == 3 ? c * image.stride[2] : 0)];
                        }
                    }
                }
                local += row_sum;
            }
            partials[static_cast<size_t>(start / band)] = local;
        });
    }
    uint64_t total = 0ULL;
    for (uint64_t p : partials) {
        total += p;
    }
    const double scale = (image.itemsize == 2 && normalize_uint16) ? (255.0 / 65535.0) : 1.0;
    return (static_cast<double>(total) * scale) / static_cast<double>(count);
}

void of_apply_exposure_gain_cpp(
    const ArrayView& image,
    const ArrayView& output,
    float gain,
    int64_t start,
    int64_t stop,
    int num_threads
) {
    if (start >= stop || image.shape[0] == 0 || image.shape[1] == 0) {
        return;
    }
    const int64_t width = image.shape[1];
    const int64_t channels = (image.ndim == 2) ? 1 : image.shape[2];
    const bool keep_alpha = (image.ndim == 3 && channels == 4);
    const int64_t color_c = keep_alpha ? 3 : channels;
    const int64_t rows = stop - start;
    const int64_t total_elems = rows * width * channels;
    const int threads = (total_elems >= 65536) ? std::max(1, num_threads) : 1;

    const bool packed = (image.ndim == 2 && image.stride[1] == 1 && output.stride[1] == 1)
                     || (image.ndim == 3 && image.stride[2] == 1 && image.stride[1] == channels
                         && output.stride[2] == 1 && output.stride[1] == channels);
    const int64_t row_elems = width * channels;
    const bool use_avx2 = packed && (!keep_alpha) && simd::use_avx2() && row_elems >= 8;

    if (image.is_float && output.is_float && image.itemsize == 4 && output.itemsize == 4) {
        const float limit = 255.0f;
        parallel_bands(rows, 128, threads, [&](int64_t b0, int64_t b1) {
            const int64_t y0 = start + b0;
            const int64_t y1 = start + b1;
            for (int64_t y = y0; y < y1; ++y) {
                if (packed) {
                    const float* sr = static_cast<const float*>(image.data) + y * image.stride[0];
                    float* dr = static_cast<float*>(output.data) + y * output.stride[0];
                    if (use_avx2) {
                        simd::exposure_gain_f32_row_avx2(sr, dr, row_elems, gain, limit);
                    } else if (!keep_alpha) {
                        for (int64_t i = 0; i < row_elems; ++i) {
                            dr[i] = std::clamp(sr[i] * gain, 0.0f, limit);
                        }
                    } else {
                        for (int64_t x = 0; x < width; ++x) {
                            const int64_t o = x * 4;
                            dr[o] = std::clamp(sr[o] * gain, 0.0f, limit);
                            dr[o + 1] = std::clamp(sr[o + 1] * gain, 0.0f, limit);
                            dr[o + 2] = std::clamp(sr[o + 2] * gain, 0.0f, limit);
                            dr[o + 3] = sr[o + 3];
                        }
                    }
                } else {
                    for (int64_t x = 0; x < width; ++x) {
                        for (int64_t c = 0; c < color_c; ++c) {
                            const float v = std::clamp(read_f32_2d3d(image, y, x, c) * gain, 0.0f, limit);
                            write_f32_2d3d(output, y, x, c, v);
                        }
                        if (keep_alpha) {
                            write_f32_2d3d(output, y, x, 3, read_f32_2d3d(image, y, x, 3));
                        }
                    }
                }
            }
        });
        return;
    }

    if (image.itemsize == 1 && output.itemsize == 1 && !image.is_float && !output.is_float) {
        const float limit = 255.0f;
        uint8_t lut[256];
        for (int i = 0; i < 256; ++i) {
            const float v = std::clamp(static_cast<float>(i) * gain, 0.0f, limit);
            lut[i] = static_cast<uint8_t>(static_cast<int32_t>(v));
        }
        parallel_bands(rows, 128, threads, [&](int64_t b0, int64_t b1) {
            const int64_t y0 = start + b0;
            const int64_t y1 = start + b1;
            for (int64_t y = y0; y < y1; ++y) {
                if (packed) {
                    const uint8_t* sr = static_cast<const uint8_t*>(image.data) + y * image.stride[0];
                    uint8_t* dr = static_cast<uint8_t*>(output.data) + y * output.stride[0];
                    if (use_avx2) {
                        simd::exposure_gain_u8_row_avx2(sr, dr, row_elems, gain, limit);
                    } else if (!keep_alpha) {
                        for (int64_t i = 0; i < row_elems; ++i) {
                            dr[i] = lut[sr[i]];
                        }
                    } else {
                        for (int64_t x = 0; x < width; ++x) {
                            const int64_t o = x * 4;
                            dr[o] = lut[sr[o]];
                            dr[o + 1] = lut[sr[o + 1]];
                            dr[o + 2] = lut[sr[o + 2]];
                            dr[o + 3] = sr[o + 3];
                        }
                    }
                } else {
                    for (int64_t x = 0; x < width; ++x) {
                        for (int64_t c = 0; c < color_c; ++c) {
                            write_sample(output, y, x, c, lut[read_u_idx(image, y, x, c)]);
                        }
                        if (keep_alpha) {
                            write_sample(output, y, x, 3, read_u_idx(image, y, x, 3));
                        }
                    }
                }
            }
        });
        return;
    }

    if (image.itemsize == 2 && output.itemsize == 2 && !image.is_float && !output.is_float) {
        const float limit = 65535.0f;
        if (packed && !keep_alpha) {
            parallel_bands(rows, 128, threads, [&](int64_t b0, int64_t b1) {
                const int64_t y0 = start + b0;
                const int64_t y1 = start + b1;
                for (int64_t y = y0; y < y1; ++y) {
                    const uint16_t* sr = static_cast<const uint16_t*>(image.data) + y * image.stride[0];
                    uint16_t* dr = static_cast<uint16_t*>(output.data) + y * output.stride[0];
                    if (use_avx2) {
                        simd::exposure_gain_u16_row_avx2(sr, dr, row_elems, gain, limit);
                    } else {
                        for (int64_t i = 0; i < row_elems; ++i) {
                            const float v = std::clamp(static_cast<float>(sr[i]) * gain, 0.0f, limit);
                            dr[i] = static_cast<uint16_t>(static_cast<int32_t>(v));
                        }
                    }
                }
            });
            return;
        }
        std::vector<uint16_t> lut(65536);
        for (int i = 0; i < 65536; ++i) {
            const float v = std::clamp(static_cast<float>(i) * gain, 0.0f, limit);
            lut[static_cast<size_t>(i)] = static_cast<uint16_t>(static_cast<int32_t>(v));
        }
        parallel_bands(rows, 128, threads, [&](int64_t b0, int64_t b1) {
            const int64_t y0 = start + b0;
            const int64_t y1 = start + b1;
            for (int64_t y = y0; y < y1; ++y) {
                if (packed) {
                    const uint16_t* sr = static_cast<const uint16_t*>(image.data) + y * image.stride[0];
                    uint16_t* dr = static_cast<uint16_t*>(output.data) + y * output.stride[0];
                    for (int64_t x = 0; x < width; ++x) {
                        const int64_t o = x * 4;
                        dr[o] = lut[sr[o]];
                        dr[o + 1] = lut[sr[o + 1]];
                        dr[o + 2] = lut[sr[o + 2]];
                        dr[o + 3] = sr[o + 3];
                    }
                } else {
                    for (int64_t x = 0; x < width; ++x) {
                        for (int64_t c = 0; c < color_c; ++c) {
                            write_sample(output, y, x, c, lut[static_cast<size_t>(read_u_idx(image, y, x, c))]);
                        }
                        if (keep_alpha) {
                            write_sample(output, y, x, 3, read_u_idx(image, y, x, 3));
                        }
                    }
                }
            }
        });
        return;
    }

    const float limit = (image.itemsize == 2) ? 65535.0f : 255.0f;
    for (int64_t y = start; y < stop; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            for (int64_t c = 0; c < color_c; ++c) {
                const double raw_v = image.is_float ? read_f64_or_f32(image, y, x, c)
                                                    : static_cast<double>(read_u_idx(image, y, x, c));
                const float v = std::clamp(static_cast<float>(raw_v) * gain, 0.0f, limit);
                if (output.is_float) {
                    write_f32_2d3d(output, y, x, c, v);
                } else {
                    write_sample(output, y, x, c, static_cast<int64_t>(static_cast<int32_t>(v)));
                }
            }
            if (keep_alpha) {
                if (output.is_float) {
                    write_f32_2d3d(output, y, x, 3, read_f32_2d3d(image, y, x, 3));
                } else {
                    write_sample(output, y, x, 3, read_u_idx(image, y, x, 3));
                }
            }
        }
    }
}

void of_quantize_output_cpp(
    const ArrayView& src,
    const ArrayView& dst,
    bool is_16bit,
    int num_threads
) {
    const int64_t height = src.shape[0];
    const int64_t width = src.shape[1];
    if (height <= 0 || width <= 0) {
        return;
    }
    const int64_t channels = (src.ndim == 2) ? 1 : src.shape[2];
    const int64_t count = height * width * channels;
    const int threads = (count >= 65536) ? std::max(1, num_threads) : 1;
    const bool packed_f32 = (src.itemsize == 4
                             && ((src.ndim == 2 && src.stride[1] == 1 && dst.stride[1] == 1)
                                 || (src.ndim == 3 && src.stride[2] == 1 && src.stride[1] == channels
                                     && dst.stride[2] == 1 && dst.stride[1] == channels)));
    const int64_t row_elems = width * channels;
    constexpr float kScaleF32 = static_cast<float>(65535.0 / 255.0);
    constexpr double kScaleF64 = 65535.0 / 255.0;

    const bool use_avx2_q = simd::use_avx2() && row_elems >= 8;

    parallel_bands(height, 128, threads, [&](int64_t start, int64_t stop) {
#ifdef OPENFOCUS_HAS_SSE2
        const __m128 v_zero = _mm_setzero_ps();
        const __m128 v_255 = _mm_set1_ps(255.0f);
        const __m128 v_scale = _mm_set1_ps(kScaleF32);
        const __m128i v_32768 = _mm_set1_epi32(32768);
        const __m128i v_xor = _mm_set1_epi16(static_cast<short>(-32768));
#endif
        for (int64_t y = start; y < stop; ++y) {
            if (packed_f32) {
                const float* sr = static_cast<const float*>(src.data) + y * src.stride[0];
                if (is_16bit) {
                    uint16_t* dr = static_cast<uint16_t*>(dst.data) + y * dst.stride[0];
                    if (use_avx2_q) {
                        simd::quantize_f32_to_u16_row_avx2(sr, dr, row_elems, kScaleF32);
                        continue;
                    }
                    int64_t i = 0;
#ifdef OPENFOCUS_HAS_SSE2
                    for (; i + 4 <= row_elems; i += 4) {
                        __m128 v = _mm_loadu_ps(sr + i);
                        v = _mm_min_ps(_mm_max_ps(v, v_zero), v_255);
                        v = _mm_mul_ps(v, v_scale);
                        __m128i qi = _mm_cvtps_epi32(v);
                        qi = _mm_sub_epi32(qi, v_32768);
                        __m128i packed16 = _mm_packs_epi32(qi, qi);
                        packed16 = _mm_xor_si128(packed16, v_xor);
                        _mm_storel_epi64(reinterpret_cast<__m128i*>(dr + i), packed16);
                    }
#endif
                    for (; i < row_elems; ++i) {
                        const float s = sr[i];
                        const float c = (s > 0.0f) ? (s < 255.0f ? s : 255.0f) : 0.0f;
                        dr[i] = static_cast<uint16_t>(std::nearbyintf(c * kScaleF32));
                    }
                } else {
                    uint8_t* dr = static_cast<uint8_t*>(dst.data) + y * dst.stride[0];
                    if (use_avx2_q) {
                        simd::quantize_f32_to_u8_row_avx2(sr, dr, row_elems);
                        continue;
                    }
                    int64_t i = 0;
#ifdef OPENFOCUS_HAS_SSE2
                    for (; i + 4 <= row_elems; i += 4) {
                        __m128 v = _mm_loadu_ps(sr + i);
                        v = _mm_min_ps(_mm_max_ps(v, v_zero), v_255);
                        __m128i qi = _mm_cvtps_epi32(v);
                        __m128i p16 = _mm_packs_epi32(qi, qi);
                        __m128i p8 = _mm_packus_epi16(p16, p16);
                        const uint32_t out4 = static_cast<uint32_t>(_mm_cvtsi128_si32(p8));
                        std::memcpy(dr + i, &out4, 4);
                    }
#endif
                    for (; i < row_elems; ++i) {
                        const float s = sr[i];
                        const float c = (s > 0.0f) ? (s < 255.0f ? s : 255.0f) : 0.0f;
                        dr[i] = static_cast<uint8_t>(std::nearbyintf(c));
                    }
                }
            } else {
                for (int64_t x = 0; x < width; ++x) {
                    for (int64_t c = 0; c < channels; ++c) {
                        if (src.itemsize == 4) {
                            float v = std::clamp(static_cast<float>(read_f64_or_f32(src, y, x, c)), 0.0f, 255.0f);
                            if (is_16bit) {
                                v *= kScaleF32;
                            }
                            write_sample(dst, y, x, c, static_cast<int64_t>(std::nearbyintf(v)));
                        } else {
                            double v = std::clamp(read_f64_or_f32(src, y, x, c), 0.0, 255.0);
                            if (is_16bit) {
                                v *= kScaleF64;
                            }
                            write_sample(dst, y, x, c, static_cast<int64_t>(std::nearbyint(v)));
                        }
                    }
                }
            }
        }
    });
}

void of_tile_accumulate_cpp(
    const ArrayView& acc,
    const ArrayView& weight_acc,
    const ArrayView& tile,
    const ArrayView& weight2d,
    int64_t x0,
    int64_t y0,
    int num_threads
) {
    const int64_t fh = tile.shape[0];
    const int64_t fw = tile.shape[1];
    if (fh <= 0 || fw <= 0) {
        return;
    }
    const int64_t channels = (tile.ndim == 2) ? 1 : std::min(tile.shape[2], acc.shape[2]);
    const int threads = (fh * fw >= 65536) ? std::max(1, num_threads) : 1;
    const auto* w2_base = static_cast<const float*>(weight2d.data);
    auto* wacc_base = static_cast<float*>(weight_acc.data);
    auto* acc_base = static_cast<float*>(acc.data);

    const bool packed_all = (weight2d.stride[1] == 1 && weight_acc.stride[1] == 1
                             && ((tile.ndim == 2 && acc.ndim == 2 && tile.stride[1] == 1 && acc.stride[1] == 1)
                                 || (tile.ndim == 3 && acc.ndim == 3 && tile.stride[2] == 1 && tile.stride[1] == channels
                                     && acc.stride[2] == 1 && acc.stride[1] == acc.shape[2]
                                     && acc.shape[2] == channels)));
    const bool use_avx2_1c = packed_all && tile.is_float && tile.itemsize == 4 && channels == 1
                          && simd::use_avx2() && fw >= 8;

    parallel_bands(fh, 64, threads, [&](int64_t start, int64_t stop) {
        for (int64_t dy = start; dy < stop; ++dy) {
            const int64_t ay = y0 + dy;
            const float* wr = w2_base + dy * weight2d.stride[0];
            float* war = wacc_base + ay * weight_acc.stride[0] + x0 * weight_acc.stride[1];
            float* ar = acc_base + ay * acc.stride[0] + x0 * acc.stride[1];
            if (packed_all && tile.is_float && tile.itemsize == 4) {
                const float* tr = static_cast<const float*>(tile.data) + dy * tile.stride[0];
                if (use_avx2_1c) {
                    simd::tile_accumulate_f32_1c_row_avx2(tr, wr, ar, war, fw);
                } else if (channels == 3) {
                    for (int64_t dx = 0; dx < fw; ++dx) {
                        const float w = wr[dx];
                        war[dx] += w;
                        const int64_t o = dx * 3;
                        ar[o] += tr[o] * w;
                        ar[o + 1] += tr[o + 1] * w;
                        ar[o + 2] += tr[o + 2] * w;
                    }
                } else {
                    for (int64_t dx = 0; dx < fw; ++dx) {
                        const float w = wr[dx];
                        war[dx] += w;
                        for (int64_t c = 0; c < channels; ++c) {
                            ar[dx * channels + c] += tr[dx * channels + c] * w;
                        }
                    }
                }
            } else if (packed_all && !tile.is_float && (tile.itemsize == 1 || tile.itemsize == 2)) {
                for (int64_t dx = 0; dx < fw; ++dx) {
                    const float w = wr[dx];
                    war[dx] += w;
                    if (tile.itemsize == 1) {
                        const uint8_t* tr = static_cast<const uint8_t*>(tile.data) + dy * tile.stride[0] + dx * channels;
                        for (int64_t c = 0; c < channels; ++c) {
                            ar[dx * channels + c] += static_cast<float>(tr[c]) * w;
                        }
                    } else {
                        const uint16_t* tr = static_cast<const uint16_t*>(tile.data) + dy * tile.stride[0] + dx * channels;
                        for (int64_t c = 0; c < channels; ++c) {
                            ar[dx * channels + c] += static_cast<float>(tr[c]) * w;
                        }
                    }
                }
            } else {
                for (int64_t dx = 0; dx < fw; ++dx) {
                    const float w = wr[dx * weight2d.stride[1]];
                    war[dx * weight_acc.stride[1]] += w;
                    float* ap = ar + dx * acc.stride[1];
                    if (tile.is_float) {
                        for (int64_t c = 0; c < channels; ++c) {
                            ap[c * acc.stride[2]] += read_f32_2d3d(tile, dy, dx, c) * w;
                        }
                    } else {
                        for (int64_t c = 0; c < channels; ++c) {
                            ap[c * acc.stride[2]] += static_cast<float>(read_u_idx(tile, dy, dx, c)) * w;
                        }
                    }
                }
            }
        }
    });
}

void of_tile_finalize_cpp(
    const ArrayView& acc,
    const ArrayView& weight_acc,
    const ArrayView& out,
    int64_t start,
    int64_t stop
) {
    if (start >= stop || acc.shape[1] == 0) {
        return;
    }
    const int64_t width = acc.shape[1];
    const int64_t channels = (acc.ndim == 2) ? 1 : acc.shape[2];
    const bool out_u8 = !out.is_float;
    const auto* acc_base = static_cast<const float*>(acc.data);
    const auto* wacc_base = static_cast<const float*>(weight_acc.data);
    const bool packed = (weight_acc.stride[1] == 1
                         && ((acc.ndim == 2 && out.ndim == 2 && acc.stride[1] == 1 && out.stride[1] == 1)
                             || (acc.ndim == 3 && out.ndim == 3 && acc.stride[2] == 1 && acc.stride[1] == channels
                                 && out.stride[2] == 1 && out.stride[1] == channels))
                         && (out_u8 ? out.itemsize == 1 : out.itemsize == 4));
    const bool use_avx2 = packed && (channels == 1 || channels == 3) && simd::use_avx2() && width >= 8;

    for (int64_t y = start; y < stop; ++y) {
        const float* ar = acc_base + y * acc.stride[0];
        const float* war = wacc_base + y * weight_acc.stride[0];
        if (packed) {
            uint8_t* u8_dr = out_u8 ? (static_cast<uint8_t*>(out.data) + y * out.stride[0]) : nullptr;
            float* f32_dr = out_u8 ? nullptr : (static_cast<float*>(out.data) + y * out.stride[0]);
            if (use_avx2) {
                simd::tile_finalize_3c_row_avx2(ar, war, u8_dr, f32_dr, width, channels, out_u8);
                continue;
            }
            for (int64_t x = 0; x < width; ++x) {
                const float w = war[x];
                const float denom = (w == 0.0f) ? 1.0f : w;
                const int64_t base_o = x * channels;
                for (int64_t c = 0; c < channels; ++c) {
                    const float raw = ar[base_o + c] / denom;
                    const float val = (raw > 0.0f) ? (raw < 255.0f ? raw : 255.0f) : 0.0f;
                    if (out_u8) {
                        u8_dr[base_o + c] = static_cast<uint8_t>(std::nearbyintf(val));
                    } else {
                        f32_dr[base_o + c] = val;
                    }
                }
            }
            continue;
        }
        for (int64_t x = 0; x < width; ++x) {
            const float w = war[x * weight_acc.stride[1]];
            const float denom = (w == 0.0f) ? 1.0f : w;
            const float* ap = ar + x * acc.stride[1];
            for (int64_t c = 0; c < channels; ++c) {
                const float raw = ap[c * acc.stride[2]] / denom;
                const float val = (raw > 0.0f) ? (raw < 255.0f ? raw : 255.0f) : 0.0f;
                if (out_u8) {
                    write_sample(out, y, x, c, static_cast<int64_t>(std::nearbyintf(val)));
                } else {
                    write_f32_2d3d(out, y, x, c, val);
                }
            }
        }
    }
}

}  // namespace openfocus
