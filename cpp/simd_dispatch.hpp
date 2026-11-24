#pragma once

#include <cstdint>
#include <string>

namespace openfocus {
namespace simd {

bool use_avx2();
bool use_avx2_fma();
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

bool pyr_any_gt_u16_avx2(const uint16_t* wr, const uint16_t* br, int64_t width);

void pyr_level0_merge_3c_row_avx2(
    const uint16_t* wr, uint16_t* br, uint16_t* win_row,
    const float* hr0, const float* hr1, const float* hr2, const float* hr3, const float* hr4,
    const void* raw_row, bool raw_is_u8, float* acc_row,
    int64_t width, float factor, uint16_t index, bool first,
    float w0, float w1, float w2, float w3, float w4, float point_one);

void pyr_merge_f32_3c_row_avx2(
    const uint16_t* wr, uint16_t* br, uint16_t* win_row,
    const float* dr, float* ar, int64_t width, uint16_t index);

void pyr_output_f32_to_u8_row_avx2(const float* sr, uint8_t* dr, int64_t elems);

void pyr_output_f32_to_f32_row_avx2(const float* sr, float* dr, int64_t elems);

void pyr_up_horiz_1c_row_avx2(
    const float* sr, float* dr, int64_t ix_start, int64_t pair_stop,
    float e0, float e1, float e2, float e3);

void pyr_up_horiz_3c_row_avx2(
    const float* sr, float* dr, int64_t ix_start, int64_t pair_stop,
    float e0, float e1, float e2, float e3);

void pyr_down_horiz_f32_3c_row_avx2(
    const float* sr, float* h_row, int64_t ix_stop, float d0, float d1);

void align_valid_mask_row_avx2(
    uint8_t* mr, int64_t w,
    float cx, float p08, float p18, float tx, float ty, float v_dy, float u_dy,
    float wf, float hf);

void align_gray16_rgb_row_avx2(
    const void* src, bool is_u8, uint16_t* dst, int64_t w, int64_t src_px_stride);

void align_down_half_vert_row_avx2(
    const uint16_t* r0, const uint16_t* r1, const uint16_t* r2, const uint16_t* r3,
    uint32_t* scratch, int64_t w);

void align_down_half_horiz_row_avx2(
    const uint32_t* scratch, uint16_t* out, int64_t ow, int64_t w);

void align_fast_downsize_u16_row_avx2(
    const uint16_t* r0, const uint16_t* r1, uint16_t* dst, int64_t ow, int64_t w);

void align_warp_alignment_row_avx2(
    const void* img_data, int64_t img_s0, int64_t img_s1, bool is_u8,
    void* out_row, int64_t out_s1,
    int64_t src_h, int64_t src_w, int64_t src_y0, int64_t x_off, int64_t rw,
    int64_t channels, bool bgr_rev,
    float cx, float p08, float p18, float tx, float ty, float v_dy, float u_dy);

void align_score_chunk_packed_avx2(
    const uint16_t* rb, int64_t ref_sy,
    const uint16_t* bb, int64_t b_sy, int64_t h, int64_t w,
    const float* xs, int64_t nxx, const float* ys, int64_t nyy,
    const float* p08, const float* p0c, const float* p18, const float* p1c,
    const float* tx, const float* ty, const float* cx, const float* cy,
    const int64_t* idx, int64_t k_start, int64_t k_stop,
    int64_t* ssum, int64_t* ssq, int64_t* cnt);

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
    int64_t* out_best_k, float* out_best_sc, int64_t slot);

void resample_u16_vert_acc_row_avx2_fma(
    const uint16_t* sline, float* row_buf, int64_t sw, float wval);

void resample_u16_horiz_rep_row_avx2_fma(
    const float* row_buf, float* sums_buf, uint16_t* orow,
    int64_t first, int64_t nrep,
    const int32_t* h_idx_first, const float* h_w_first,
    int64_t h_taps, int64_t hi_sy, int64_t hw_sy);

void activity_row_u16_avx2(
    const uint16_t* gr, const uint16_t* fr, uint16_t* orow, int64_t w);

void detail_row_u16_avx2(
    const uint16_t* r0, const uint16_t* r1, uint16_t* orow, int64_t full_w);

void select_activity_row_u16_avx2(
    const uint16_t* ar, uint16_t* br, uint16_t* dr, uint16_t* vr,
    int64_t w, uint16_t code, uint32_t reciprocal_u32);

void select_detail_row_u16_avx2(
    const uint16_t* ar, uint16_t* br, uint16_t* sr, uint16_t* wr,
    int64_t w, uint16_t code);

void output_u16_to_u8_row_avx2(
    const uint16_t* cr, uint8_t* orow, int64_t elems, bool e37);

void output_u16_to_f32_row_avx2(
    const uint16_t* cr, float* orow, int64_t elems, float factor);

void tile_accumulate_3c_row_avx2(
    const void* tr, int src_kind, const float* wr,
    float* ar, float* war, int64_t fw);

void ecc_prepare_level_row_avx2(
    const float* rm1, const float* r0, const float* rp1,
    int x0, int rw, double cx, double inv_s, double mean_t, double g_scale, double v,
    int p_count, bool sample_h_row, bool stride2,
    float* dst_t0, float* dst_gx, float* dst_gy,
    double& norm_T2, double* js, double* tv, double* H_tri);

void ecc_ic_pass_all_in_row_p6_avx2(
    const float* src_data, int w, int h,
    const float* t0_row, const float* gx_row, const float* gy_row, int rw,
    double sx0, double sy0, double dsx, double dsy,
    double u0, double inv_s, double v,
    double& rs_iw, double& rs_iw2, double& rs_tiw, double* raw_g);

void ecc_ic_pass_all_in_row_p8_avx2(
    const float* src_data, int w, int h,
    const float* t0_row, const float* gx_row, const float* gy_row, int rw,
    double nsx0, double nsy0, double den0,
    double d_nsx, double d_nsy, double d_den,
    double u0, double inv_s, double v,
    double& rs_iw, double& rs_iw2, double& rs_tiw, double* raw_g);

}  // namespace simd
}  // namespace openfocus
