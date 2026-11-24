#pragma once

#include "openfocus_ab.hpp"

#include <cstdint>

namespace openfocus {

void of_pyr_down_from_raw_cpp(
    const ArrayView& source,
    const ArrayView& output,
    const ArrayView& scratch,
    float factor,
    int64_t start,
    int64_t stop
);

void of_pyr_down_f32_cpp(
    const ArrayView& source,
    const ArrayView& output,
    const ArrayView& scratch,
    int64_t start,
    int64_t stop
);

void of_pyr_level0_weights_from_raw_cpp(
    const ArrayView& coarse,
    const ArrayView& source,
    const ArrayView& weight,
    const ArrayView& scratch,
    float factor,
    int64_t start,
    int64_t stop
);

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
);

void of_pyr_detail_weight_f32_cpp(
    const ArrayView& coarse,
    const ArrayView& target,
    const ArrayView& weight,
    const ArrayView& scratch,
    int64_t start,
    int64_t stop
);

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
);

void of_pyr_reconstruct_f32_cpp(
    const ArrayView& coarse,
    const ArrayView& target,
    const ArrayView& scratch,
    int64_t start,
    int64_t stop
);

void of_pyr_output_f32_cpp(
    const ArrayView& source,
    const ArrayView& output,
    bool is_u8,
    int64_t start,
    int64_t stop
);

}  // namespace openfocus
