#pragma once

#include <cstdint>
#include <string>

namespace openfocus {
namespace simd {

bool use_avx2();
void set_force_scalar(bool force);
std::string simd_info();

void pyr_down_vertical_row_f32_avx2(
    const float* p0, const float* p1, const float* p2, const float* p3,
    float* dst, int64_t elems, float d0, float d1);

void pyr_reconstruct_row_f32_avx2(
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    float* tr, int64_t elems,
    float w0, float w1, float w2, float w3, float w4, float point_one);

void pyr_detail_weight_3c_f32_avx2(
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    float* target_row, uint16_t* w_row, int64_t width,
    float w0, float w1, float w2, float w3, float w4, float point_one,
    float w_scale, bool y_border);

void pyr_level0_weights_3c_f32_from_raw_avx2(
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    const void* raw_row, bool raw_is_u8, uint16_t* w_row, int64_t width,
    float factor, float w0, float w1, float w2, float w3, float w4, float point_one,
    float w_scale, bool y_border);

void depth_axis_iir_y_unguided_avx2(
    const uint16_t* src, int64_t src_sy,
    uint16_t* out, int64_t out_sy,
    int64_t h, int64_t left, int64_t cols, int64_t cnt,
    uint32_t q, uint32_t t);

void depth_axis_iir_y_guided_avx2(
    const uint16_t* src, int64_t src_sy,
    const uint16_t* guide, int64_t guide_sy,
    uint16_t* out, int64_t out_sy,
    int64_t h, int64_t left, int64_t cols, int64_t cnt,
    uint32_t tm, uint32_t tl);

void select_a_combine_m2_to_m5_avx2(
    const uint32_t* da2, const uint32_t* db2, uint32_t* m2, uint32_t wy2, uint32_t fy2,
    const uint32_t* da3, const uint32_t* db3, uint32_t* m3, uint32_t wy3, uint32_t fy3,
    const uint32_t* da4, const uint32_t* db4, uint32_t* m4, uint32_t wy4, uint32_t fy4,
    const uint32_t* da5, const uint32_t* db5, uint32_t* m5, uint32_t wy5, uint32_t fy5,
    int64_t n_out);

void select_a_response_pairs_avx2(
    const void* g_row, bool g_is_u16,
    const uint32_t* m0, const uint32_t* m1,
    const uint32_t* m2, const uint32_t* m3,
    const uint32_t* m4, const uint32_t* m5,
    int32_t* response, int64_t full_pairs);

void quantize_f32_to_u16_row_avx2(
    const float* sr, uint16_t* dr, int64_t elems, float scale_f32);

void quantize_f32_to_u8_row_avx2(
    const float* sr, uint8_t* dr, int64_t elems);

}  // namespace simd
}  // namespace openfocus
