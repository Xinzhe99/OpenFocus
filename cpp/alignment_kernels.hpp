#pragma once

#include "openfocus_ab.hpp"

#include <cstdint>
#include <tuple>

namespace openfocus {

void of_align_swap_rgb_to_bgr_rows_cpp(
    const ArrayView& src,
    const ArrayView& dst,
    int64_t y_start,
    int64_t y_stop
);

void of_align_swap_bgr_inplace_rows_cpp(
    const ArrayView& arr,
    int64_t y_start,
    int64_t y_stop
);

void of_align_gray16_rows_cpp(
    const ArrayView& image,
    const ArrayView& out,
    int64_t y_start,
    int64_t y_stop
);

void of_align_down_half_rows_cpp(
    const ArrayView& src,
    const ArrayView& out,
    int64_t oy_start,
    int64_t oy_stop
);

void of_align_fast_downsize_u16_rows_cpp(
    const ArrayView& img,
    const ArrayView& out,
    int64_t oy_start,
    int64_t oy_stop
);

void of_align_warp_int_rows_cpp(
    const ArrayView& image,
    const ArrayView& out,
    float dx,
    float dy,
    int64_t y_start,
    int64_t y_stop,
    bool bgr_rev
);

void of_align_warp_alignment_int_rows_cpp(
    const ArrayView& image,
    const ArrayView& out,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t y_start,
    int64_t y_stop,
    bool bgr_rev
);

void of_align_warp_alignment_float_rows_cpp(
    const ArrayView& image,
    const ArrayView& out,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t y_start,
    int64_t y_stop
);

void of_align_valid_mask_rows_cpp(
    uint8_t* mask,
    int64_t mask_sy,
    int64_t mask_sx,
    int64_t h,
    int64_t w,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t y_start,
    int64_t y_stop
);

std::tuple<float, float, bool> of_align_refine_level_cpp(
    const ArrayView& a,
    const ArrayView& b,
    float c_dx,
    float c_dy,
    double half,
    double step,
    int64_t crop
);

void of_align_score_chunk_u16_slice_cpp(
    const ArrayView& ref,
    const ArrayView& img_b,
    const float* xs, int64_t nxx,
    const float* ys, int64_t nyy,
    const float* p08, const float* p0c, const float* p18, const float* p1c,
    const float* tx, const float* ty, const float* cx, const float* cy,
    const int64_t* idx, int64_t k_start, int64_t k_stop,
    int64_t* ssum, int64_t* ssq, int64_t* cnt
);

void of_align_score_grid_4d_slice_cpp(
    const ArrayView& ref,
    const ArrayView& img_b,
    const float* xs, int64_t nxx,
    const float* ys, int64_t nyy,
    float tx0, float ty0,
    const float* cx_tab, const float* cy_tab,
    const ArrayView& p08_tab, const ArrayView& p18_tab,
    int64_t n0, int64_t n1, int64_t n2, int64_t n3,
    int64_t k_start, int64_t k_stop,
    int64_t* out_best_k, float* out_best_sc, int64_t slot
);

bool of_align_perspective_has_rounding_edge_cpp(
    const double* inverse,
    int64_t x0, int64_t x1,
    int64_t y0, int64_t y1,
    double bins,
    double tolerance
);

void of_align_warp_alignment_region_int_rows_cpp(
    const ArrayView& image,
    const ArrayView& out,
    int64_t src_h, int64_t src_w, int64_t src_y0,
    int64_t x_off, int64_t y_off,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t oy_start, int64_t oy_stop,
    bool bgr_rev
);

void of_align_warp_alignment_region_float_rows_cpp(
    const ArrayView& image,
    const ArrayView& out,
    int64_t src_h, int64_t src_w, int64_t src_y0,
    int64_t x_off, int64_t y_off,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t oy_start, int64_t oy_stop
);

void of_align_warp_translation_region_int_rows_cpp(
    const ArrayView& image,
    const ArrayView& out,
    int64_t src_h, int64_t src_w, int64_t src_y0,
    int64_t x_off, int64_t y_off,
    float dx, float dy,
    int64_t oy_start, int64_t oy_stop,
    bool bgr_rev
);

}  // namespace openfocus
