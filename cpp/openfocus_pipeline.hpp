#pragma once

#include "openfocus_ab.hpp"

#include <cstdint>

namespace openfocus {

void of_table_rows_cpp(
    const ArrayView& image,
    const ArrayView& table,
    const ArrayView& output,
    int64_t start,
    int64_t stop
);

void of_fill_half_axis_cpp(
    int64_t source_length,
    int64_t target_length,
    int64_t* indices,
    float* weights,
    int64_t* counts
);

void of_fill_linear_axis_cpp(
    int64_t source_length,
    int64_t target_length,
    int64_t* indices,
    float* weights,
    int64_t* counts
);

double of_exposure_mean_cpp(
    const ArrayView& image,
    bool normalize_uint16,
    int num_threads
);

void of_apply_exposure_gain_cpp(
    const ArrayView& image,
    const ArrayView& output,
    float gain,
    int64_t start,
    int64_t stop,
    int num_threads = 4
);

void of_quantize_output_cpp(
    const ArrayView& src,
    const ArrayView& dst,
    bool is_16bit,
    int num_threads
);

void of_tile_accumulate_cpp(
    const ArrayView& acc,
    const ArrayView& weight_acc,
    const ArrayView& tile,
    const ArrayView& weight2d,
    int64_t x0,
    int64_t y0,
    int num_threads
);

void of_tile_finalize_cpp(
    const ArrayView& acc,
    const ArrayView& weight_acc,
    const ArrayView& out,
    int64_t start,
    int64_t stop
);

void pack_png_scanlines_cpp(
    const ArrayView& src,
    const ArrayView& dst,
    int num_threads
);

}  // namespace openfocus
