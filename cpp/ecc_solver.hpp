#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

namespace openfocus {

struct GrayView {
    const float* data;
    int height;
    int width;
};

struct IdenticalStatsResult {
    bool is_finite;
    bool is_nonflat;
    bool is_identical;
    double rho_ident;
};

IdenticalStatsResult ecc_check_identical_and_stats_cpp(
    const GrayView& tpl,
    const GrayView& src,
    int num_threads
);

std::tuple<double, int64_t> ecc_eval_rho_warped_cpp(
    const GrayView& tpl,
    const GrayView& src,
    const double W[9],
    int y0,
    int y1,
    int x0,
    int x1,
    bool all_in
);

std::tuple<bool, double, std::vector<float>> ecc_refine_pyramids_cpp(
    const std::vector<GrayView>& tpl_pyr,
    const std::vector<GrayView>& src_pyr,
    const double W_coarsest_init[9],
    double rho_ident_full,
    int max_iter,
    double eps,
    int num_threads
);

}  // namespace openfocus
