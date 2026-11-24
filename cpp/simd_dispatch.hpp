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

void exposure_gain_u16_row_avx2(
    const uint16_t* sr, uint16_t* dr, int64_t elems, float gain, float limit);

void exposure_gain_u8_row_avx2(
    const uint8_t* sr, uint8_t* dr, int64_t elems, float gain, float limit);

void exposure_gain_f32_row_avx2(
    const float* sr, float* dr, int64_t elems, float gain, float limit);

void down2_row_u16_to_i32_avx2(
    const uint16_t* r0, const uint16_t* r1, int32_t* dst, int64_t full_w, int32_t scale);

void down2_row_i32_to_i32_avx2(
    const int32_t* r0, const int32_t* r1, int32_t* dst, int64_t full_w, int32_t scale);

void gray_rgb_u8_row_avx2(
    const uint8_t* sr, uint16_t* dst16, int32_t* dst32, int64_t width,
    int32_t scale, bool depth_color, bool dst_is_u16);

void gray_rgb_u16_row_avx2(
    const uint16_t* sr, uint16_t* dst16, int32_t* dst32, int64_t width,
    int32_t scale, bool depth_color, bool dst_is_u16);

void merge_recip_packed_row_avx2(
    const uint16_t* drow, const void* irow, bool src_u8,
    void* arow, bool dst_u16, int64_t width,
    int64_t is1, int64_t is2, int64_t as1, int64_t as2,
    int32_t base32, int32_t step32, uint64_t recip48, int32_t s32);

int select_a_no_bg_packed_row_avx2(
    int32_t* b_row, const int32_t* response, int64_t width,
    const void* irow, bool src_u8, void* orow, bool dst_u16,
    int64_t is1, int64_t is2, int64_t os1, int64_t os2,
    int64_t scale, int state, int64_t max_response, int sharp_state, int saturated_state);

void tile_accumulate_f32_1c_row_avx2(
    const float* tr, const float* wr, float* ar, float* war, int64_t fw);

void tile_finalize_3c_row_avx2(
    const float* ar, const float* war, uint8_t* u8_dr, float* f32_dr,
    int64_t width, int64_t channels, bool out_u8);

}  // namespace simd
}  // namespace openfocus
