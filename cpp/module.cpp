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

#include "ecc_solver.hpp"
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

}  // namespace

NB_MODULE(_openfocus_native, m) {
    m.doc() = "OpenFocus C++17 native compute kernels";
    m.def("abi_version", []() { return kAbiVersion; });
    m.def("compiler_info", &compiler_info);
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
}
