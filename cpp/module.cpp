#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "alignment_kernels.hpp"
#include "ecc_solver.hpp"
#include "openfocus_ab.hpp"
#include "openfocus_c.hpp"
#include "openfocus_pipeline.hpp"
#include "simd_dispatch.hpp"
#include "thread_pool.hpp"

namespace nb = nanobind;

namespace {

constexpr int kAbiVersion = 1;

std::string compiler_info() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__clang__)
    return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#else
    return "Unknown C++17";
#endif
}

double parallel_sum_smoke(
    nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr,
    int num_threads
) {
    const int64_t height = static_cast<int64_t>(arr.shape(0));
    const int64_t width = static_cast<int64_t>(arr.shape(1));
    const float* data = arr.data();

    if (height <= 0 || width <= 0) {
        return 0.0;
    }

    const int64_t band = 64;
    const int64_t num_tasks = (height + band - 1) / band;
    std::vector<double> partials(static_cast<size_t>(num_tasks), 0.0);

    {
        nb::gil_scoped_release release;
        openfocus::parallel_bands(
            height,
            band,
            num_threads,
            [&](int64_t start, int64_t stop) {
                double local_sum = 0.0;
                for (int64_t y = start; y < stop; ++y) {
                    const float* row = data + y * width;
                    double row_sum = 0.0;
                    for (int64_t x = 0; x < width; ++x) {
                        row_sum += static_cast<double>(row[x]);
                    }
                    local_sum += row_sum;
                }
                partials[static_cast<size_t>(start / band)] = local_sum;
            }
        );
    }

    double total = 0.0;
    for (double v : partials) {
        total += v;
    }
    return total;
}

using ConstFloat2D = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ConstDouble2D = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

std::tuple<bool, bool, bool, double> ecc_check_identical_and_stats_py(
    ConstFloat2D tpl,
    ConstFloat2D src,
    int num_threads
) {
    if (tpl.shape(0) != src.shape(0) || tpl.shape(1) != src.shape(1)
            || tpl.shape(0) <= 0 || tpl.shape(1) <= 0) {
        return {false, false, false, -2.0};
    }
    const openfocus::GrayView tv{tpl.data(), static_cast<int>(tpl.shape(0)), static_cast<int>(tpl.shape(1))};
    const openfocus::GrayView sv{src.data(), static_cast<int>(src.shape(0)), static_cast<int>(src.shape(1))};
    openfocus::IdenticalStatsResult res{};
    {
        nb::gil_scoped_release release;
        res = openfocus::ecc_check_identical_and_stats_cpp(tv, sv, num_threads);
    }
    return {res.is_finite, res.is_nonflat, res.is_identical, res.rho_ident};
}

std::tuple<double, int64_t> ecc_eval_rho_warped_py(
    ConstFloat2D tpl,
    ConstFloat2D src,
    ConstDouble2D W,
    int y0,
    int y1,
    int x0,
    int x1,
    bool all_in
) {
    if (W.shape(0) != 3 || W.shape(1) != 3) {
        throw std::invalid_argument("W must have shape (3, 3)");
    }
    double W_copy[9];
    std::memcpy(W_copy, W.data(), sizeof(double) * 9);
    const openfocus::GrayView tv{tpl.data(), static_cast<int>(tpl.shape(0)), static_cast<int>(tpl.shape(1))};
    const openfocus::GrayView sv{src.data(), static_cast<int>(src.shape(0)), static_cast<int>(src.shape(1))};
    std::tuple<double, int64_t> res;
    {
        nb::gil_scoped_release release;
        res = openfocus::ecc_eval_rho_warped_cpp(tv, sv, W_copy, y0, y1, x0, x1, all_in);
    }
    return res;
}

std::tuple<bool, double, nb::ndarray<nb::numpy, float, nb::shape<3, 3>>> ecc_refine_pyramids_py(
    const std::vector<ConstFloat2D>& tpl_pyr,
    const std::vector<ConstFloat2D>& src_pyr,
    ConstDouble2D W_init,
    double rho_ident_full,
    int max_iter,
    double eps,
    int num_threads
) {
    if (W_init.shape(0) != 3 || W_init.shape(1) != 3) {
        throw std::invalid_argument("W_init must have shape (3, 3)");
    }
    if (tpl_pyr.size() != src_pyr.size() || tpl_pyr.empty()) {
        throw std::invalid_argument("Pyramid level lists must be non-empty and equal length");
    }
    std::vector<openfocus::GrayView> tv_pyr;
    std::vector<openfocus::GrayView> sv_pyr;
    tv_pyr.reserve(tpl_pyr.size());
    sv_pyr.reserve(src_pyr.size());
    for (size_t i = 0; i < tpl_pyr.size(); ++i) {
        const auto& t = tpl_pyr[i];
        const auto& s = src_pyr[i];
        if (t.shape(0) != s.shape(0) || t.shape(1) != s.shape(1) || t.shape(0) <= 0 || t.shape(1) <= 0) {
            throw std::invalid_argument("Pyramid level shapes must match");
        }
        tv_pyr.push_back({t.data(), static_cast<int>(t.shape(0)), static_cast<int>(t.shape(1))});
        sv_pyr.push_back({s.data(), static_cast<int>(s.shape(0)), static_cast<int>(s.shape(1))});
    }
    double W_copy[9];
    std::memcpy(W_copy, W_init.data(), sizeof(double) * 9);

    bool ok = false;
    double cc = -2.0;
    std::vector<float> mat_vec;
    {
        nb::gil_scoped_release release;
        std::tie(ok, cc, mat_vec) = openfocus::ecc_refine_pyramids_cpp(
            tv_pyr, sv_pyr, W_copy, rho_ident_full, max_iter, eps, num_threads);
    }

    float* heap_buf = new float[9];
    std::memcpy(heap_buf, mat_vec.data(), sizeof(float) * 9);
    nb::capsule owner(heap_buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    nb::ndarray<nb::numpy, float, nb::shape<3, 3>> out_arr(heap_buf, {3, 3}, owner);
    return {ok, cc, out_arr};
}

template <typename NDArray>
openfocus::ArrayView make_array_view(const NDArray& arr) {
    openfocus::ArrayView view{};
    view.data = const_cast<void*>(static_cast<const void*>(arr.data()));
    view.ndim = static_cast<int>(arr.ndim());
    if (view.ndim < 1 || view.ndim > 3) {
        throw std::invalid_argument("Unsupported array ndim");
    }
    for (int i = 0; i < view.ndim; ++i) {
        view.shape[i] = static_cast<int64_t>(arr.shape(i));
        view.stride[i] = static_cast<int64_t>(arr.stride(i));
    }
    const auto dt = arr.dtype();
    if (dt == nb::dtype<uint8_t>()) {
        view.itemsize = 1;
        view.is_signed = false;
        view.is_float = false;
    } else if (dt == nb::dtype<uint16_t>()) {
        view.itemsize = 2;
        view.is_signed = false;
        view.is_float = false;
    } else if (dt == nb::dtype<int32_t>()) {
        view.itemsize = 4;
        view.is_signed = true;
        view.is_float = false;
    } else if (dt == nb::dtype<uint32_t>()) {
        view.itemsize = 4;
        view.is_signed = false;
        view.is_float = false;
    } else if (dt == nb::dtype<float>()) {
        view.itemsize = 4;
        view.is_signed = true;
        view.is_float = true;
    } else if (dt == nb::dtype<double>()) {
        view.itemsize = 8;
        view.is_signed = true;
        view.is_float = true;
    } else {
        throw std::invalid_argument("Unsupported ndarray dtype");
    }
    return view;
}

using AnyCpuArray = nb::ndarray<nb::device::cpu>;
using ConstAnyCpuArray = nb::ndarray<nb::ro, nb::device::cpu>;
using U16_2D = nb::ndarray<uint16_t, nb::ndim<2>, nb::device::cpu>;
using ConstU16_2D = nb::ndarray<const uint16_t, nb::ndim<2>, nb::device::cpu>;

void of_gray_rows_py(ConstAnyCpuArray image, AnyCpuArray output, int64_t start, int64_t stop,
                     int64_t scale, bool depth_color) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(output);
    nb::gil_scoped_release release;
    openfocus::of_gray_rows_cpp(iv, ov, start, stop, scale, depth_color);
}

void of_down2_rows_py(ConstAnyCpuArray image, AnyCpuArray output, int64_t start, int64_t stop, int64_t scale) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(output);
    nb::gil_scoped_release release;
    openfocus::of_down2_rows_cpp(iv, ov, start, stop, scale);
}

void of_burt_laplacian_rows_py(ConstAnyCpuArray gray, ConstAnyCpuArray reduced, AnyCpuArray output,
                               int64_t start, int64_t stop) {
    const auto gv = make_array_view(gray);
    const auto rv = make_array_view(reduced);
    const auto ov = make_array_view(output);
    nb::gil_scoped_release release;
    openfocus::of_burt_laplacian_rows_cpp(gv, rv, ov, start, stop);
}

int of_select_rows_a_py(
    ConstAnyCpuArray image, ConstAnyCpuArray gray, ConstAnyCpuArray half_gray, ConstAnyCpuArray laplacians,
    int levels, AnyCpuArray best, AnyCpuArray output,
    int64_t start, int64_t stop, int64_t scale, bool first, AnyCpuArray background
) {
    const auto iv = make_array_view(image);
    const auto gv = make_array_view(gray);
    const auto hv = make_array_view(half_gray);
    const auto lv = make_array_view(laplacians);
    const auto bv = make_array_view(best);
    const auto ov = make_array_view(output);
    const auto bgv = make_array_view(background);
    nb::gil_scoped_release release;
    return openfocus::of_select_rows_a_cpp(iv, gv, hv, lv, levels, bv, ov, start, stop, scale, first, bgv);
}

void of_blend_low_response_rows_py(AnyCpuArray output, ConstAnyCpuArray best, ConstAnyCpuArray background,
                                   int64_t start, int64_t stop) {
    const auto ov = make_array_view(output);
    const auto bv = make_array_view(best);
    const auto bgv = make_array_view(background);
    nb::gil_scoped_release release;
    openfocus::of_blend_low_response_rows_cpp(ov, bv, bgv, start, stop);
}

void of_depth_axis_iir_py(
    ConstU16_2D src,
    nb::object guide_obj,
    U16_2D output,
    uint32_t q, uint32_t t, uint32_t tm, uint32_t tl,
    int64_t count, int axis, int num_threads
) {
    if (src.stride(1) != 1 || output.stride(1) != 1) {
        throw std::invalid_argument("of_depth_axis_iir requires unit x-stride");
    }
    const uint16_t* g_ptr = nullptr;
    int64_t g_sy = 0;
    if (!guide_obj.is_none()) {
        auto g_arr = nb::cast<ConstU16_2D>(guide_obj);
        if (g_arr.stride(1) != 1) {
            throw std::invalid_argument("guide must have unit x-stride");
        }
        g_ptr = g_arr.data();
        g_sy = g_arr.stride(0);
    }
    const int64_t h = static_cast<int64_t>(src.shape(0));
    const int64_t w = static_cast<int64_t>(src.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_depth_axis_iir_cpp(
        src.data(), src.stride(0), g_ptr, g_sy,
        output.data(), output.stride(0), h, w,
        q, t, tm, tl, count, axis, num_threads);
}

void of_activity_rows_py(ConstU16_2D gray, ConstU16_2D fine, U16_2D out, int64_t start, int64_t stop) {
    const int64_t w = static_cast<int64_t>(gray.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_activity_rows_cpp(gray.data(), gray.stride(0), fine.data(), fine.stride(0),
                                    out.data(), out.stride(0), w, start, stop);
}

void of_detail_rows_py(ConstU16_2D gray, U16_2D out, int64_t row_offset) {
    const int64_t gh = static_cast<int64_t>(gray.shape(0));
    const int64_t gw = static_cast<int64_t>(gray.shape(1));
    const int64_t oh = static_cast<int64_t>(out.shape(0));
    const int64_t ow = static_cast<int64_t>(out.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_detail_rows_cpp(gray.data(), gray.stride(0), gh, gw,
                                  out.data(), out.stride(0), oh, ow, row_offset);
}

void of_smooth_detail_rows_py(ConstAnyCpuArray detail, AnyCpuArray output, int64_t start, int64_t stop) {
    const auto dv = make_array_view(detail);
    const auto ov = make_array_view(output);
    nb::gil_scoped_release release;
    openfocus::of_smooth_detail_rows_cpp(dv, ov, start, stop);
}

void of_initialize_detail_rows_py(U16_2D second, U16_2D winner, int64_t start, int64_t stop) {
    const int64_t w = static_cast<int64_t>(second.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_initialize_detail_rows_cpp(second.data(), second.stride(0),
                                             winner.data(), winner.stride(0), w, start, stop);
}

void of_select_detail_rows_py(ConstU16_2D activity, U16_2D best, U16_2D second, U16_2D winner,
                              uint16_t code, int64_t start, int64_t stop) {
    const int64_t w = static_cast<int64_t>(activity.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_select_detail_rows_cpp(
        activity.data(), activity.stride(0), best.data(), best.stride(0),
        second.data(), second.stride(0), winner.data(), winner.stride(0),
        w, code, start, stop);
}

void of_detail_confidence_rows_py(ConstU16_2D coarse, ConstU16_2D best, U16_2D second,
                                  int64_t start, int64_t stop) {
    const int64_t w = static_cast<int64_t>(coarse.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_detail_confidence_rows_cpp(
        coarse.data(), coarse.stride(0), best.data(), best.stride(0),
        second.data(), second.stride(0), w, start, stop);
}

void of_apply_detail_rows_py(U16_2D depth, ConstU16_2D mask, ConstU16_2D winner,
                             int64_t start, int64_t stop) {
    const int64_t w = static_cast<int64_t>(depth.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_apply_detail_rows_cpp(
        depth.data(), depth.stride(0), mask.data(), mask.stride(0),
        winner.data(), winner.stride(0), w, start, stop);
}

void of_initialize_activity_rows_py(ConstU16_2D activity, U16_2D depth, nb::object votes_obj,
                                    int64_t reciprocal, int64_t start, int64_t stop) {
    uint16_t* v_ptr = nullptr;
    int64_t v_sy = 0;
    if (!votes_obj.is_none()) {
        auto v_arr = nb::cast<U16_2D>(votes_obj);
        v_ptr = v_arr.data();
        v_sy = v_arr.stride(0);
    }
    const int64_t w = static_cast<int64_t>(activity.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_initialize_activity_rows_cpp(
        activity.data(), activity.stride(0), depth.data(), depth.stride(0),
        v_ptr, v_sy, w, reciprocal, start, stop);
}

void of_select_activity_rows_py(ConstU16_2D activity, U16_2D best, U16_2D depth,
                                nb::object votes_obj, uint16_t code, int64_t reciprocal,
                                int64_t start, int64_t stop) {
    uint16_t* v_ptr = nullptr;
    int64_t v_sy = 0;
    if (!votes_obj.is_none()) {
        auto v_arr = nb::cast<U16_2D>(votes_obj);
        v_ptr = v_arr.data();
        v_sy = v_arr.stride(0);
    }
    const int64_t w = static_cast<int64_t>(activity.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_select_activity_rows_cpp(
        activity.data(), activity.stride(0), best.data(), best.stride(0),
        depth.data(), depth.stride(0), v_ptr, v_sy, w, code, reciprocal, start, stop);
}

void of_confidence_hist_rows_py(U16_2D best, ConstU16_2D votes,
                                    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> hist,
                                    int64_t start, int64_t stop) {
    const int64_t w = static_cast<int64_t>(best.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_confidence_hist_rows_cpp(
        best.data(), best.stride(0), votes.data(), votes.stride(0),
        hist.data(), w, start, stop);
}

void of_guide_rows_py(U16_2D best,
                      nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> lut,
                      int64_t start, int64_t stop) {
    const int64_t w = static_cast<int64_t>(best.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_guide_rows_cpp(best.data(), best.stride(0), lut.data(), w, start, stop);
}

void of_snap_rows_py(U16_2D depth,
                     nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> lut,
                     int64_t step, int64_t start, int64_t stop) {
    const int64_t w = static_cast<int64_t>(depth.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_snap_rows_cpp(depth.data(), depth.stride(0), lut.data(), w, step, start, stop);
}

void of_resample_rows_u16_py(
    ConstU16_2D src, U16_2D out,
    nb::ndarray<const int32_t, nb::ndim<2>, nb::device::cpu> v_idx,
    nb::ndarray<const float, nb::ndim<2>, nb::device::cpu> v_w,
    nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> v_cnt,
    nb::ndarray<const int32_t, nb::ndim<2>, nb::device::cpu> h_idx,
    nb::ndarray<const float, nb::ndim<2>, nb::device::cpu> h_w,
    nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> h_cnt,
    int64_t start, int64_t stop
) {
    const int64_t sw = static_cast<int64_t>(src.shape(1));
    const int64_t ow = static_cast<int64_t>(out.shape(1));
    const int64_t vt = static_cast<int64_t>(v_idx.shape(1));
    const int64_t ht = static_cast<int64_t>(h_idx.shape(1));
    nb::gil_scoped_release release;
    openfocus::of_resample_rows_u16_cpp(
        src.data(), src.stride(0), sw,
        out.data(), out.stride(0), ow,
        v_idx.data(), v_idx.stride(0), vt,
        v_w.data(), v_w.stride(0),
        v_cnt.data(),
        h_idx.data(), h_idx.stride(0), ht,
        h_w.data(), h_w.stride(0),
        h_cnt.data(),
        start, stop);
}

void of_merge_rows_py(
    AnyCpuArray accumulator, ConstAnyCpuArray image, ConstU16_2D depth,
    int64_t index, int64_t step, int64_t start, int64_t stop,
    int64_t scale, bool first, bool reverse
) {
    const auto av = make_array_view(accumulator);
    const auto iv = make_array_view(image);
    nb::gil_scoped_release release;
    openfocus::of_merge_rows_cpp(av, iv, depth.data(), depth.stride(0),
                                 index, step, start, stop, scale, first, reverse);
}

void of_active_source_rows_py(
    ConstU16_2D depth, int64_t step,
    nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> active,
    int64_t start, int64_t stop
) {
    const int64_t w = static_cast<int64_t>(depth.shape(1));
    const int64_t alen = static_cast<int64_t>(active.shape(0));
    nb::gil_scoped_release release;
    openfocus::of_active_source_rows_cpp(depth.data(), depth.stride(0), w, step,
                                         active.data(), alen, start, stop);
}

void of_output_u16_to_u8_rows_py(ConstAnyCpuArray code, AnyCpuArray output,
                                 int64_t start, int64_t stop, bool e37) {
    const auto cv = make_array_view(code);
    const auto ov = make_array_view(output);
    nb::gil_scoped_release release;
    openfocus::of_output_u16_to_u8_rows_cpp(cv, ov, start, stop, e37);
}

void of_output_u16_to_f32_rows_py(ConstAnyCpuArray code, AnyCpuArray output,
                                  int64_t start, int64_t stop, float factor) {
    const auto cv = make_array_view(code);
    const auto ov = make_array_view(output);
    nb::gil_scoped_release release;
    openfocus::of_output_u16_to_f32_rows_cpp(cv, ov, start, stop, factor);
}

void of_pyr_down_from_raw_py(ConstAnyCpuArray source, AnyCpuArray output, AnyCpuArray scratch,
                             float factor, int64_t start, int64_t stop) {
    const auto sv = make_array_view(source);
    const auto ov = make_array_view(output);
    const auto scv = make_array_view(scratch);
    nb::gil_scoped_release release;
    openfocus::of_pyr_down_from_raw_cpp(sv, ov, scv, factor, start, stop);
}

void of_pyr_down_f32_py(ConstAnyCpuArray source, AnyCpuArray output, AnyCpuArray scratch,
                        int64_t start, int64_t stop) {
    const auto sv = make_array_view(source);
    const auto ov = make_array_view(output);
    const auto scv = make_array_view(scratch);
    nb::gil_scoped_release release;
    openfocus::of_pyr_down_f32_cpp(sv, ov, scv, start, stop);
}

void of_pyr_level0_weights_from_raw_py(ConstAnyCpuArray coarse, ConstAnyCpuArray source,
                                       U16_2D weight, AnyCpuArray scratch,
                                       float factor, int64_t start, int64_t stop) {
    if (weight.stride(1) != 1) {
        throw std::invalid_argument("weight must have unit x-stride");
    }
    const auto cv = make_array_view(coarse);
    const auto sv = make_array_view(source);
    const auto wv = make_array_view(weight);
    const auto scv = make_array_view(scratch);
    nb::gil_scoped_release release;
    openfocus::of_pyr_level0_weights_from_raw_cpp(cv, sv, wv, scv, factor, start, stop);
}

void of_pyr_level0_merge_from_raw_py(ConstAnyCpuArray coarse, ConstAnyCpuArray source,
                                     ConstU16_2D weight, AnyCpuArray accumulator, U16_2D best,
                                     nb::object winner_obj, AnyCpuArray scratch,
                                     float factor, bool first, uint16_t index,
                                     int64_t start, int64_t stop) {
    if (weight.stride(1) != 1 || best.stride(1) != 1) {
        throw std::invalid_argument("weight and best must have unit x-stride");
    }
    uint16_t* win_ptr = nullptr;
    int64_t win_sy = 0;
    if (!winner_obj.is_none()) {
        auto win_arr = nb::cast<U16_2D>(winner_obj);
        if (win_arr.stride(1) != 1) {
            throw std::invalid_argument("winner must have unit x-stride");
        }
        win_ptr = win_arr.data();
        win_sy = win_arr.stride(0);
    }
    const auto cv = make_array_view(coarse);
    const auto sv = make_array_view(source);
    const auto wv = make_array_view(weight);
    const auto av = make_array_view(accumulator);
    const auto bv = make_array_view(best);
    const auto scv = make_array_view(scratch);
    nb::gil_scoped_release release;
    openfocus::of_pyr_level0_merge_from_raw_cpp(
        cv, sv, wv, av, bv, win_ptr, win_sy, scv, factor, first, index, start, stop);
}

void of_pyr_detail_weight_f32_py(ConstAnyCpuArray coarse, AnyCpuArray target,
                                 U16_2D weight, AnyCpuArray scratch,
                                 int64_t start, int64_t stop) {
    if (weight.stride(1) != 1) {
        throw std::invalid_argument("weight must have unit x-stride");
    }
    const auto cv = make_array_view(coarse);
    const auto tv = make_array_view(target);
    const auto wv = make_array_view(weight);
    const auto scv = make_array_view(scratch);
    nb::gil_scoped_release release;
    openfocus::of_pyr_detail_weight_f32_cpp(cv, tv, wv, scv, start, stop);
}

void of_pyr_merge_f32_py(ConstAnyCpuArray detail, ConstU16_2D weight, AnyCpuArray accumulator,
                         U16_2D best, nb::object winner_obj, bool first, uint16_t index,
                         int64_t start, int64_t stop) {
    if (weight.stride(1) != 1 || best.stride(1) != 1) {
        throw std::invalid_argument("weight and best must have unit x-stride");
    }
    uint16_t* win_ptr = nullptr;
    int64_t win_sy = 0;
    if (!winner_obj.is_none()) {
        auto win_arr = nb::cast<U16_2D>(winner_obj);
        if (win_arr.stride(1) != 1) {
            throw std::invalid_argument("winner must have unit x-stride");
        }
        win_ptr = win_arr.data();
        win_sy = win_arr.stride(0);
    }
    const auto dv = make_array_view(detail);
    const auto wv = make_array_view(weight);
    const auto av = make_array_view(accumulator);
    const auto bv = make_array_view(best);
    nb::gil_scoped_release release;
    openfocus::of_pyr_merge_f32_cpp(dv, wv, av, bv, win_ptr, win_sy, first, index, start, stop);
}

void of_pyr_reconstruct_f32_py(ConstAnyCpuArray coarse, AnyCpuArray target, AnyCpuArray scratch,
                               int64_t start, int64_t stop) {
    const auto cv = make_array_view(coarse);
    const auto tv = make_array_view(target);
    const auto scv = make_array_view(scratch);
    nb::gil_scoped_release release;
    openfocus::of_pyr_reconstruct_f32_cpp(cv, tv, scv, start, stop);
}

void of_pyr_output_f32_py(ConstAnyCpuArray source, AnyCpuArray output, bool is_u8,
                          int64_t start, int64_t stop) {
    const auto sv = make_array_view(source);
    const auto ov = make_array_view(output);
    nb::gil_scoped_release release;
    openfocus::of_pyr_output_f32_cpp(sv, ov, is_u8, start, stop);
}

void of_table_rows_py(ConstAnyCpuArray image, ConstAnyCpuArray table, AnyCpuArray output,
                      int64_t start, int64_t stop) {
    const auto iv = make_array_view(image);
    const auto tv = make_array_view(table);
    const auto ov = make_array_view(output);
    nb::gil_scoped_release release;
    openfocus::of_table_rows_cpp(iv, tv, ov, start, stop);
}

void of_fill_half_axis_py(
    int64_t source_length,
    nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> indices,
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> weights,
    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> counts
) {
    if (indices.shape(1) != 8 || weights.shape(1) != 8
            || indices.shape(0) != weights.shape(0)
            || indices.shape(0) != counts.shape(0)) {
        throw std::invalid_argument("Invalid half axis plan buffers");
    }
    const int64_t target_length = static_cast<int64_t>(indices.shape(0));
    nb::gil_scoped_release release;
    openfocus::of_fill_half_axis_cpp(
        source_length, target_length, indices.data(), weights.data(), counts.data());
}

void of_fill_linear_axis_py(
    int64_t source_length,
    nb::ndarray<int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> indices,
    nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> weights,
    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> counts
) {
    if (indices.shape(1) != 2 || weights.shape(1) != 2
            || indices.shape(0) != weights.shape(0)
            || indices.shape(0) != counts.shape(0)) {
        throw std::invalid_argument("Invalid linear axis plan buffers");
    }
    const int64_t target_length = static_cast<int64_t>(indices.shape(0));
    nb::gil_scoped_release release;
    openfocus::of_fill_linear_axis_cpp(
        source_length, target_length, indices.data(), weights.data(), counts.data());
}

double of_exposure_mean_py(ConstAnyCpuArray image, bool normalize_uint16, int num_threads) {
    const auto iv = make_array_view(image);
    nb::gil_scoped_release release;
    return openfocus::of_exposure_mean_cpp(iv, normalize_uint16, num_threads);
}

void of_apply_exposure_gain_py(ConstAnyCpuArray image, AnyCpuArray output, float gain,
                               int64_t start, int64_t stop, int num_threads) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(output);
    nb::gil_scoped_release release;
    openfocus::of_apply_exposure_gain_cpp(iv, ov, gain, start, stop, num_threads);
}

void of_quantize_output_py(ConstAnyCpuArray src, AnyCpuArray dst, bool is_16bit, int num_threads) {
    const auto sv = make_array_view(src);
    const auto dv = make_array_view(dst);
    nb::gil_scoped_release release;
    openfocus::of_quantize_output_cpp(sv, dv, is_16bit, num_threads);
}

void of_tile_accumulate_py(AnyCpuArray acc, AnyCpuArray weight_acc,
                           ConstAnyCpuArray tile, ConstAnyCpuArray weight2d,
                           int64_t x0, int64_t y0, int num_threads) {
    const auto av = make_array_view(acc);
    const auto wav = make_array_view(weight_acc);
    const auto tv = make_array_view(tile);
    const auto w2v = make_array_view(weight2d);
    nb::gil_scoped_release release;
    openfocus::of_tile_accumulate_cpp(av, wav, tv, w2v, x0, y0, num_threads);
}

void of_tile_finalize_py(ConstAnyCpuArray acc, ConstAnyCpuArray weight_acc,
                         AnyCpuArray out, int64_t start, int64_t stop) {
    const auto av = make_array_view(acc);
    const auto wav = make_array_view(weight_acc);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_tile_finalize_cpp(av, wav, ov, start, stop);
}

using ConstF32_1D = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ConstI64_1D = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64_1D = nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F32_1D = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ConstF64_2D = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

void of_align_swap_rgb_to_bgr_rows_py(ConstAnyCpuArray src, AnyCpuArray dst,
                                      int64_t y_start, int64_t y_stop) {
    const auto sv = make_array_view(src);
    const auto dv = make_array_view(dst);
    nb::gil_scoped_release release;
    openfocus::of_align_swap_rgb_to_bgr_rows_cpp(sv, dv, y_start, y_stop);
}

void of_align_swap_bgr_inplace_rows_py(AnyCpuArray arr, int64_t y_start, int64_t y_stop) {
    const auto av = make_array_view(arr);
    nb::gil_scoped_release release;
    openfocus::of_align_swap_bgr_inplace_rows_cpp(av, y_start, y_stop);
}

void of_align_gray16_rows_py(ConstAnyCpuArray image, AnyCpuArray out,
                             int64_t y_start, int64_t y_stop) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_align_gray16_rows_cpp(iv, ov, y_start, y_stop);
}

void of_align_down_half_rows_py(ConstAnyCpuArray src, AnyCpuArray out,
                                int64_t oy_start, int64_t oy_stop) {
    const auto sv = make_array_view(src);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_align_down_half_rows_cpp(sv, ov, oy_start, oy_stop);
}

void of_align_fast_downsize_u16_rows_py(ConstAnyCpuArray img, AnyCpuArray out,
                                        int64_t oy_start, int64_t oy_stop) {
    const auto iv = make_array_view(img);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_align_fast_downsize_u16_rows_cpp(iv, ov, oy_start, oy_stop);
}

void of_align_warp_int_rows_py(ConstAnyCpuArray image, AnyCpuArray out,
                               float dx, float dy,
                               int64_t y_start, int64_t y_stop, bool bgr_rev) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_align_warp_int_rows_cpp(iv, ov, dx, dy, y_start, y_stop, bgr_rev);
}

void of_align_warp_alignment_int_rows_py(
    ConstAnyCpuArray image, AnyCpuArray out,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t y_start, int64_t y_stop, bool bgr_rev
) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_align_warp_alignment_int_rows_cpp(
        iv, ov, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop, bgr_rev);
}

void of_align_warp_alignment_float_rows_py(
    ConstAnyCpuArray image, AnyCpuArray out,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t y_start, int64_t y_stop
) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_align_warp_alignment_float_rows_cpp(
        iv, ov, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop);
}

void of_align_valid_mask_rows_py(
    nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cpu> mask,
    int64_t h, int64_t w,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t y_start, int64_t y_stop
) {
    uint8_t* ptr = mask.data();
    const int64_t sy = static_cast<int64_t>(mask.stride(0));
    const int64_t sx = static_cast<int64_t>(mask.stride(1));
    nb::gil_scoped_release release;
    openfocus::of_align_valid_mask_rows_cpp(
        ptr, sy, sx, h, w, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop);
}

std::tuple<float, float, bool> of_align_refine_level_py(
    ConstU16_2D a, ConstU16_2D b,
    float c_dx, float c_dy, double half, double step, int64_t crop
) {
    const auto av = make_array_view(a);
    const auto bv = make_array_view(b);
    nb::gil_scoped_release release;
    return openfocus::of_align_refine_level_cpp(av, bv, c_dx, c_dy, half, step, crop);
}

void of_align_score_chunk_u16_slice_py(
    ConstU16_2D ref, ConstU16_2D img_b,
    ConstF32_1D xs, ConstF32_1D ys,
    ConstF32_1D p08, ConstF32_1D p0c, ConstF32_1D p18, ConstF32_1D p1c,
    ConstF32_1D tx, ConstF32_1D ty, ConstF32_1D cx, ConstF32_1D cy,
    ConstI64_1D idx, int64_t k_start, int64_t k_stop,
    I64_1D ssum, I64_1D ssq, I64_1D cnt
) {
    const auto rv = make_array_view(ref);
    const auto bv = make_array_view(img_b);
    const float* xs_p = xs.data();
    const int64_t nxx = static_cast<int64_t>(xs.shape(0));
    const float* ys_p = ys.data();
    const int64_t nyy = static_cast<int64_t>(ys.shape(0));
    const float* p08_p = p08.data();
    const float* p0c_p = p0c.data();
    const float* p18_p = p18.data();
    const float* p1c_p = p1c.data();
    const float* tx_p = tx.data();
    const float* ty_p = ty.data();
    const float* cx_p = cx.data();
    const float* cy_p = cy.data();
    const int64_t* idx_p = idx.data();
    int64_t* ssum_p = ssum.data();
    int64_t* ssq_p = ssq.data();
    int64_t* cnt_p = cnt.data();
    nb::gil_scoped_release release;
    openfocus::of_align_score_chunk_u16_slice_cpp(
        rv, bv, xs_p, nxx, ys_p, nyy,
        p08_p, p0c_p, p18_p, p1c_p, tx_p, ty_p, cx_p, cy_p,
        idx_p, k_start, k_stop, ssum_p, ssq_p, cnt_p);
}

void of_align_score_grid_4d_slice_py(
    ConstU16_2D ref, ConstU16_2D img_b,
    ConstF32_1D xs, ConstF32_1D ys,
    float tx0, float ty0,
    ConstF32_1D cx_tab, ConstF32_1D cy_tab,
    ConstAnyCpuArray p08_tab, ConstAnyCpuArray p18_tab,
    int64_t n0, int64_t n1, int64_t n2, int64_t n3,
    int64_t k_start, int64_t k_stop,
    I64_1D out_best_k, F32_1D out_best_sc, int64_t slot
) {
    const auto rv = make_array_view(ref);
    const auto bv = make_array_view(img_b);
    const auto p08v = make_array_view(p08_tab);
    const auto p18v = make_array_view(p18_tab);
    const float* xs_p = xs.data();
    const int64_t nxx = static_cast<int64_t>(xs.shape(0));
    const float* ys_p = ys.data();
    const int64_t nyy = static_cast<int64_t>(ys.shape(0));
    const float* cx_p = cx_tab.data();
    const float* cy_p = cy_tab.data();
    int64_t* bk_p = out_best_k.data();
    float* bsc_p = out_best_sc.data();
    nb::gil_scoped_release release;
    openfocus::of_align_score_grid_4d_slice_cpp(
        rv, bv, xs_p, nxx, ys_p, nyy, tx0, ty0, cx_p, cy_p, p08v, p18v,
        n0, n1, n2, n3, k_start, k_stop, bk_p, bsc_p, slot);
}

bool of_align_perspective_has_rounding_edge_py(
    ConstF64_2D inverse, int64_t x0, int64_t x1, int64_t y0, int64_t y1,
    double bins, double tolerance
) {
    if (inverse.shape(0) < 3 || inverse.shape(1) < 3) {
        throw std::invalid_argument("inverse must be at least 3x3");
    }
    const double* ptr = inverse.data();
    nb::gil_scoped_release release;
    return openfocus::of_align_perspective_has_rounding_edge_cpp(
        ptr, x0, x1, y0, y1, bins, tolerance);
}

void of_align_warp_alignment_region_int_rows_py(
    ConstAnyCpuArray image, AnyCpuArray out,
    int64_t src_h, int64_t src_w, int64_t src_y0,
    int64_t x_off, int64_t y_off,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t oy_start, int64_t oy_stop, bool bgr_rev
) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_align_warp_alignment_region_int_rows_cpp(
        iv, ov, src_h, src_w, src_y0, x_off, y_off,
        tx, ty, cx, cy, p08, p0c, p18, p1c, oy_start, oy_stop, bgr_rev);
}

void of_align_warp_alignment_region_float_rows_py(
    ConstAnyCpuArray image, AnyCpuArray out,
    int64_t src_h, int64_t src_w, int64_t src_y0,
    int64_t x_off, int64_t y_off,
    float tx, float ty, float cx, float cy,
    float p08, float p0c, float p18, float p1c,
    int64_t oy_start, int64_t oy_stop
) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_align_warp_alignment_region_float_rows_cpp(
        iv, ov, src_h, src_w, src_y0, x_off, y_off,
        tx, ty, cx, cy, p08, p0c, p18, p1c, oy_start, oy_stop);
}

void of_align_warp_translation_region_int_rows_py(
    ConstAnyCpuArray image, AnyCpuArray out,
    int64_t src_h, int64_t src_w, int64_t src_y0,
    int64_t x_off, int64_t y_off,
    float dx, float dy,
    int64_t oy_start, int64_t oy_stop, bool bgr_rev
) {
    const auto iv = make_array_view(image);
    const auto ov = make_array_view(out);
    nb::gil_scoped_release release;
    openfocus::of_align_warp_translation_region_int_rows_cpp(
        iv, ov, src_h, src_w, src_y0, x_off, y_off,
        dx, dy, oy_start, oy_stop, bgr_rev);
}

void pack_png_scanlines_py(ConstAnyCpuArray src, AnyCpuArray dst, int num_threads) {
    const auto sv = make_array_view(src);
    const auto dv = make_array_view(dst);
    nb::gil_scoped_release release;
    openfocus::pack_png_scanlines_cpp(sv, dv, num_threads);
}

void parallel_copy_bytes_py(ConstAnyCpuArray src, AnyCpuArray dst, int num_threads) {
    const auto sv = make_array_view(src);
    const auto dv = make_array_view(dst);
    if (sv.ndim <= 0 || dv.ndim <= 0) {
        throw std::invalid_argument("Unsupported array dimensions for parallel_copy_bytes");
    }
    int64_t s_elem = 1;
    int64_t expected_stride = 1;
    for (int i = sv.ndim - 1; i >= 0; --i) {
        if (sv.shape[i] > 1 && sv.stride[i] != expected_stride) {
            throw std::invalid_argument("Source array must be C-contiguous");
        }
        expected_stride *= sv.shape[i];
        s_elem *= sv.shape[i];
    }
    int64_t d_elem = 1;
    expected_stride = 1;
    for (int i = dv.ndim - 1; i >= 0; --i) {
        if (dv.shape[i] > 1 && dv.stride[i] != expected_stride) {
            throw std::invalid_argument("Destination array must be C-contiguous");
        }
        expected_stride *= dv.shape[i];
        d_elem *= dv.shape[i];
    }
    const int64_t nbytes = s_elem * sv.itemsize;
    const int64_t dbytes = d_elem * dv.itemsize;
    if (dbytes < nbytes) {
        throw std::invalid_argument("Destination buffer smaller than source");
    }
    const auto* sp = static_cast<const uint8_t*>(sv.data);
    auto* dp = static_cast<uint8_t*>(dv.data);
    nb::gil_scoped_release release;
    openfocus::parallel_copy_bytes_cpp(sp, dp, nbytes, num_threads);
}

}  // namespace

NB_MODULE(_openfocus_native, m) {
    m.doc() = "OpenFocus C++17 native compute kernels";
    m.def("abi_version", []() { return kAbiVersion; });
    m.def("compiler_info", &compiler_info);
    m.def("simd_info", &openfocus::simd::simd_info);
    m.def("set_force_scalar", &openfocus::simd::set_force_scalar, nb::arg("force"));
    m.def(
        "parallel_sum_smoke",
        &parallel_sum_smoke,
        nb::arg("arr"),
        nb::arg("num_threads") = 0
    );
    m.def(
        "ecc_check_identical_and_stats",
        &ecc_check_identical_and_stats_py,
        nb::arg("tpl"),
        nb::arg("src"),
        nb::arg("num_threads") = 1
    );
    m.def(
        "ecc_eval_rho_warped",
        &ecc_eval_rho_warped_py,
        nb::arg("tpl"),
        nb::arg("src"),
        nb::arg("W"),
        nb::arg("y0"),
        nb::arg("y1"),
        nb::arg("x0"),
        nb::arg("x1"),
        nb::arg("all_in")
    );
    m.def(
        "ecc_refine_pyramids",
        &ecc_refine_pyramids_py,
        nb::arg("tpl_pyr"),
        nb::arg("src_pyr"),
        nb::arg("W_init"),
        nb::arg("rho_ident_full"),
        nb::arg("max_iter") = 50,
        nb::arg("eps") = 1e-4,
        nb::arg("num_threads") = 1
    );
    m.def("of_gray_rows", &of_gray_rows_py);
    m.def("of_down2_rows", &of_down2_rows_py);
    m.def("of_burt_laplacian_rows", &of_burt_laplacian_rows_py);
    m.def("of_select_rows_a", &of_select_rows_a_py);
    m.def("of_blend_low_response_rows", &of_blend_low_response_rows_py);
    m.def(
        "of_depth_axis_iir",
        &of_depth_axis_iir_py,
        nb::arg("src"),
        nb::arg("guide").none(),
        nb::arg("output"),
        nb::arg("q"),
        nb::arg("t"),
        nb::arg("tm"),
        nb::arg("tl"),
        nb::arg("count"),
        nb::arg("axis"),
        nb::arg("num_threads")
    );
    m.def("of_activity_rows", &of_activity_rows_py);
    m.def("of_detail_rows", &of_detail_rows_py);
    m.def("of_smooth_detail_rows", &of_smooth_detail_rows_py);
    m.def("of_initialize_detail_rows", &of_initialize_detail_rows_py);
    m.def("of_select_detail_rows", &of_select_detail_rows_py);
    m.def("of_detail_confidence_rows", &of_detail_confidence_rows_py);
    m.def("of_apply_detail_rows", &of_apply_detail_rows_py);
    m.def(
        "of_initialize_activity_rows",
        &of_initialize_activity_rows_py,
        nb::arg("activity"),
        nb::arg("depth"),
        nb::arg("votes").none(),
        nb::arg("reciprocal"),
        nb::arg("start"),
        nb::arg("stop")
    );
    m.def(
        "of_select_activity_rows",
        &of_select_activity_rows_py,
        nb::arg("activity"),
        nb::arg("best"),
        nb::arg("depth"),
        nb::arg("votes").none(),
        nb::arg("code"),
        nb::arg("reciprocal"),
        nb::arg("start"),
        nb::arg("stop")
    );
    m.def("of_confidence_hist_rows", &of_confidence_hist_rows_py);
    m.def("of_guide_rows", &of_guide_rows_py);
    m.def("of_snap_rows", &of_snap_rows_py);
    m.def("of_resample_rows_u16", &of_resample_rows_u16_py);
    m.def("of_merge_rows", &of_merge_rows_py);
    m.def("of_active_source_rows", &of_active_source_rows_py);
    m.def("of_output_u16_to_u8_rows", &of_output_u16_to_u8_rows_py);
    m.def("of_output_u16_to_f32_rows", &of_output_u16_to_f32_rows_py);
    m.def("of_pyr_down_from_raw", &of_pyr_down_from_raw_py);
    m.def("of_pyr_down_f32", &of_pyr_down_f32_py);
    m.def("of_pyr_level0_weights_from_raw", &of_pyr_level0_weights_from_raw_py);
    m.def(
        "of_pyr_level0_merge_from_raw",
        &of_pyr_level0_merge_from_raw_py,
        nb::arg("coarse"),
        nb::arg("source"),
        nb::arg("weight"),
        nb::arg("accumulator"),
        nb::arg("best"),
        nb::arg("winner").none(),
        nb::arg("scratch"),
        nb::arg("factor"),
        nb::arg("first"),
        nb::arg("index"),
        nb::arg("start"),
        nb::arg("stop")
    );
    m.def("of_pyr_detail_weight_f32", &of_pyr_detail_weight_f32_py);
    m.def(
        "of_pyr_merge_f32",
        &of_pyr_merge_f32_py,
        nb::arg("detail"),
        nb::arg("weight"),
        nb::arg("accumulator"),
        nb::arg("best"),
        nb::arg("winner").none(),
        nb::arg("first"),
        nb::arg("index"),
        nb::arg("start"),
        nb::arg("stop")
    );
    m.def("of_pyr_reconstruct_f32", &of_pyr_reconstruct_f32_py);
    m.def("of_pyr_output_f32", &of_pyr_output_f32_py);
    m.def("of_table_rows", &of_table_rows_py);
    m.def("of_fill_half_axis", &of_fill_half_axis_py);
    m.def("of_fill_linear_axis", &of_fill_linear_axis_py);
    m.def("of_exposure_mean", &of_exposure_mean_py,
          nb::arg("image"), nb::arg("normalize_uint16"), nb::arg("num_threads") = 4);
    m.def("of_apply_exposure_gain", &of_apply_exposure_gain_py,
          nb::arg("image"), nb::arg("output"), nb::arg("gain"),
          nb::arg("start"), nb::arg("stop"), nb::arg("num_threads") = 1);
    m.def("of_quantize_output", &of_quantize_output_py,
          nb::arg("src"), nb::arg("dst"), nb::arg("is_16bit"), nb::arg("num_threads") = 4);
    m.def("of_tile_accumulate", &of_tile_accumulate_py,
          nb::arg("acc"), nb::arg("weight_acc"), nb::arg("tile"), nb::arg("weight2d"),
          nb::arg("x0"), nb::arg("y0"), nb::arg("num_threads") = 4);
    m.def("of_tile_finalize", &of_tile_finalize_py);
    m.def("of_align_swap_rgb_to_bgr_rows", &of_align_swap_rgb_to_bgr_rows_py);
    m.def("of_align_swap_bgr_inplace_rows", &of_align_swap_bgr_inplace_rows_py);
    m.def("of_align_gray16_rows", &of_align_gray16_rows_py);
    m.def("of_align_down_half_rows", &of_align_down_half_rows_py);
    m.def("of_align_fast_downsize_u16_rows", &of_align_fast_downsize_u16_rows_py);
    m.def("of_align_warp_int_rows", &of_align_warp_int_rows_py);
    m.def("of_align_warp_alignment_int_rows", &of_align_warp_alignment_int_rows_py);
    m.def("of_align_warp_alignment_float_rows", &of_align_warp_alignment_float_rows_py);
    m.def("of_align_valid_mask_rows", &of_align_valid_mask_rows_py);
    m.def("of_align_refine_level", &of_align_refine_level_py);
    m.def("of_align_score_chunk_u16_slice", &of_align_score_chunk_u16_slice_py);
    m.def("of_align_score_grid_4d_slice", &of_align_score_grid_4d_slice_py);
    m.def("of_align_perspective_has_rounding_edge", &of_align_perspective_has_rounding_edge_py);
    m.def("of_align_warp_alignment_region_int_rows", &of_align_warp_alignment_region_int_rows_py);
    m.def("of_align_warp_alignment_region_float_rows", &of_align_warp_alignment_region_float_rows_py);
    m.def("of_align_warp_translation_region_int_rows", &of_align_warp_translation_region_int_rows_py);
    m.def("pack_png_scanlines", &pack_png_scanlines_py,
          nb::arg("src"), nb::arg("dst"), nb::arg("num_threads") = 8);
    m.def("parallel_copy_bytes", &parallel_copy_bytes_py,
          nb::arg("src"), nb::arg("dst"), nb::arg("num_threads") = 8);
}
