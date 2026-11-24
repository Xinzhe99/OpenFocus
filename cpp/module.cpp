#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

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
}
