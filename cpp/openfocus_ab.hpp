#pragma once

#include <cstdint>

namespace openfocus {

struct ArrayView {
    void* data = nullptr;
    int64_t shape[3]{1, 1, 1};
    int64_t stride[3]{0, 0, 1};
    int ndim = 2;
    int itemsize = 1;  // 1=u8, 2=u16, 4=i32/f32
    bool is_signed = false;
    bool is_float = false;
};

inline bool byte_ranges_overlap(const void* lhs, uint64_t lhs_bytes,
                                const void* rhs, uint64_t rhs_bytes) {
    if (lhs_bytes == 0 || rhs_bytes == 0) {
        return false;
    }
    const auto left = reinterpret_cast<uintptr_t>(lhs);
    const auto right = reinterpret_cast<uintptr_t>(rhs);
    return left <= right ? right - left < lhs_bytes : left - right < rhs_bytes;
}

void of_gray_rows_cpp(const ArrayView& image, const ArrayView& output,
                      int64_t start, int64_t stop, int64_t scale, bool depth_color);

void of_down2_rows_cpp(const ArrayView& image, const ArrayView& output,
                       int64_t start, int64_t stop, int64_t scale);

void of_burt_laplacian_rows_cpp(const ArrayView& gray, const ArrayView& reduced,
                                const ArrayView& output, int64_t start, int64_t stop);

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
);

void of_blend_low_response_rows_cpp(
    const ArrayView& output,
    const ArrayView& best,
    const ArrayView& background,
    int64_t start,
    int64_t stop
);

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
);

void of_activity_rows_cpp(const uint16_t* gray, int64_t g_sy,
                          const uint16_t* fine, int64_t f_sy,
                          uint16_t* out, int64_t o_sy,
                          int64_t w, int64_t start, int64_t stop);

void of_detail_rows_cpp(const uint16_t* gray, int64_t g_sy, int64_t gh, int64_t gw,
                        uint16_t* out, int64_t o_sy, int64_t oh, int64_t ow,
                        int64_t row_offset);

void of_smooth_detail_rows_cpp(const ArrayView& detail, const ArrayView& output,
                               int64_t start, int64_t stop);

void of_initialize_detail_rows_cpp(uint16_t* second, int64_t s_sy,
                                   uint16_t* winner, int64_t w_sy,
                                   int64_t w, int64_t start, int64_t stop);

void of_select_detail_rows_cpp(const uint16_t* activity, int64_t a_sy,
                               uint16_t* best, int64_t b_sy,
                               uint16_t* second, int64_t s_sy,
                               uint16_t* winner, int64_t w_sy,
                               int64_t w, uint16_t code, int64_t start, int64_t stop);

void of_detail_confidence_rows_cpp(const uint16_t* coarse, int64_t c_sy,
                                   const uint16_t* best, int64_t b_sy,
                                   uint16_t* second, int64_t s_sy,
                                   int64_t w, int64_t start, int64_t stop);

void of_apply_detail_rows_cpp(uint16_t* depth, int64_t d_sy,
                              const uint16_t* mask, int64_t m_sy,
                              const uint16_t* winner, int64_t w_sy,
                              int64_t w, int64_t start, int64_t stop);

void of_initialize_activity_rows_cpp(const uint16_t* activity, int64_t a_sy,
                                     uint16_t* depth, int64_t d_sy,
                                     uint16_t* votes, int64_t v_sy,
                                     int64_t w, int64_t reciprocal,
                                     int64_t start, int64_t stop);

void of_select_activity_rows_cpp(const uint16_t* activity, int64_t a_sy,
                                 uint16_t* best, int64_t b_sy,
                                 uint16_t* depth, int64_t d_sy,
                                 uint16_t* votes, int64_t v_sy,
                                 int64_t w, uint16_t code, int64_t reciprocal,
                                 int64_t start, int64_t stop);

void of_confidence_hist_rows_cpp(uint16_t* best, int64_t b_sy,
                                  const uint16_t* votes, int64_t v_sy,
                                  int64_t* hist, int64_t w,
                                  int64_t start, int64_t stop);

void of_guide_rows_cpp(uint16_t* best, int64_t b_sy, const float* lut,
                       int64_t w, int64_t start, int64_t stop);

void of_snap_rows_cpp(uint16_t* depth, int64_t d_sy, const int32_t* lut,
                      int64_t w, int64_t step, int64_t start, int64_t stop);

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
);

void of_merge_rows_cpp(
    const ArrayView& accumulator,
    const ArrayView& image,
    const uint16_t* depth, int64_t d_sy,
    int64_t index, int64_t step,
    int64_t start, int64_t stop,
    int64_t scale, bool first, bool reverse
);

void of_active_source_rows_cpp(
    const uint16_t* depth, int64_t d_sy, int64_t w,
    int64_t step, uint8_t* active, int64_t active_len,
    int64_t start, int64_t stop
);

void of_output_u16_to_u8_rows_cpp(
    const ArrayView& code, const ArrayView& output,
    int64_t start, int64_t stop, bool e37
);

void of_output_u16_to_f32_rows_cpp(
    const ArrayView& code, const ArrayView& output,
    int64_t start, int64_t stop, float factor
);

}  // namespace openfocus
