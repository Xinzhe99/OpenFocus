#include "ecc_solver.hpp"
#include "simd_dispatch.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace openfocus {
namespace {

inline void mat3_mul(const double A[9], const double B[9], double C[9]) {
    for (int r = 0; r < 3; ++r) {
        const double a0 = A[r * 3 + 0];
        const double a1 = A[r * 3 + 1];
        const double a2 = A[r * 3 + 2];
        C[r * 3 + 0] = a0 * B[0] + a1 * B[3] + a2 * B[6];
        C[r * 3 + 1] = a0 * B[1] + a1 * B[4] + a2 * B[7];
        C[r * 3 + 2] = a0 * B[2] + a1 * B[5] + a2 * B[8];
    }
}

inline bool mat3_inv(const double M[9], double inv_out[9]) {
    const double a = M[0], b = M[1], c = M[2];
    const double d = M[3], e = M[4], f = M[5];
    const double g = M[6], h = M[7], i = M[8];
    const double c00 = e * i - f * h;
    const double c01 = -(d * i - f * g);
    const double c02 = d * h - e * g;
    const double det = a * c00 + b * c01 + c * c02;
    if (!std::isfinite(det) || std::abs(det) <= 1e-15) {
        return false;
    }
    const double inv_det = 1.0 / det;
    inv_out[0] = c00 * inv_det;
    inv_out[1] = (-(b * i - c * h)) * inv_det;
    inv_out[2] = (b * f - c * e) * inv_det;
    inv_out[3] = c01 * inv_det;
    inv_out[4] = (a * i - c * g) * inv_det;
    inv_out[5] = (-(a * f - c * d)) * inv_det;
    inv_out[6] = c02 * inv_det;
    inv_out[7] = (-(a * h - b * g)) * inv_det;
    inv_out[8] = (a * e - b * d) * inv_det;
    for (int k = 0; k < 9; ++k) {
        if (!std::isfinite(inv_out[k])) {
            return false;
        }
    }
    return true;
}

template <int N>
inline bool solve_linear_system(const double A_in[N * N], const double b_in[N], double x_out[N]) {
    double A[N][N + 1];
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            A[r][c] = A_in[r * N + c];
        }
        A[r][N] = b_in[r];
    }
    for (int col = 0; col < N; ++col) {
        int pivot = col;
        double max_val = std::abs(A[col][col]);
        for (int row = col + 1; row < N; ++row) {
            const double val = std::abs(A[row][col]);
            if (val > max_val) {
                max_val = val;
                pivot = row;
            }
        }
        if (!std::isfinite(max_val) || max_val <= 1e-15) {
            return false;
        }
        if (pivot != col) {
            for (int c = col; c <= N; ++c) {
                std::swap(A[col][c], A[pivot][c]);
            }
        }
        const double diag_inv = 1.0 / A[col][col];
        for (int row = col + 1; row < N; ++row) {
            const double factor = A[row][col] * diag_inv;
            A[row][col] = 0.0;
            for (int c = col + 1; c <= N; ++c) {
                A[row][c] -= factor * A[col][c];
            }
        }
    }
    for (int row = N - 1; row >= 0; --row) {
        double sum = A[row][N];
        for (int c = row + 1; c < N; ++c) {
            sum -= A[row][c] * x_out[c];
        }
        x_out[row] = sum / A[row][row];
        if (!std::isfinite(x_out[row])) {
            return false;
        }
    }
    return true;
}

inline void centered_mats(int w, int h, double& cx, double& cy, double& s, double& inv_s,
                          double T[9], double T_inv[9]) {
    cx = (w - 1) * 0.5;
    cy = (h - 1) * 0.5;
    s = std::max(w, h) * 0.5;
    inv_s = 1.0 / s;
    T[0] = inv_s; T[1] = 0.0;   T[2] = -cx * inv_s;
    T[3] = 0.0;   T[4] = inv_s; T[5] = -cy * inv_s;
    T[6] = 0.0;   T[7] = 0.0;   T[8] = 1.0;

    T_inv[0] = s;   T_inv[1] = 0.0; T_inv[2] = cx;
    T_inv[3] = 0.0; T_inv[4] = s;   T_inv[5] = cy;
    T_inv[6] = 0.0; T_inv[7] = 0.0; T_inv[8] = 1.0;
}

inline bool roi_all_in(const double W[9], int w, int h, int y0, int y1, int x0, int x1) {
    const double xs_in[4] = {static_cast<double>(x0), static_cast<double>(x1 - 1),
                             static_cast<double>(x1 - 1), static_cast<double>(x0)};
    const double ys_in[4] = {static_cast<double>(y0), static_cast<double>(y0),
                             static_cast<double>(y1 - 1), static_cast<double>(y1 - 1)};
    const double w_hi = static_cast<double>(w) - 1.002;
    const double h_hi = static_cast<double>(h) - 1.002;
    for (int k = 0; k < 4; ++k) {
        const double px = W[0] * xs_in[k] + W[1] * ys_in[k] + W[2];
        const double py = W[3] * xs_in[k] + W[4] * ys_in[k] + W[5];
        const double den = W[6] * xs_in[k] + W[7] * ys_in[k] + W[8];
        if (!(den > 1e-4)) {
            return false;
        }
        const double ux = px / den;
        const double uy = py / den;
        if (!(ux >= 0.001 && ux <= w_hi && uy >= 0.001 && uy <= h_hi)) {
            return false;
        }
    }
    return true;
}

inline void inscribed_roi(const double W[9], int w, int h,
                          int& y0, int& y1, int& x0, int& x1, bool& all_in) {
    y0 = 1;
    y1 = h - 1;
    x0 = 1;
    x1 = w - 1;
    if (roi_all_in(W, w, h, y0, y1, x0, x1)) {
        all_in = true;
        return;
    }
    if (std::min(w, h) < 64) {
        all_in = false;
        return;
    }
    double W_inv[9];
    if (!mat3_inv(W, W_inv)) {
        all_in = false;
        return;
    }
    const double pad = 1.0;
    const double sx[4] = {pad, w - 1.0 - pad, w - 1.0 - pad, pad};
    const double sy[4] = {pad, pad, h - 1.0 - pad, h - 1.0 - pad};
    double tx[4], ty[4];
    for (int k = 0; k < 4; ++k) {
        const double mx = W_inv[0] * sx[k] + W_inv[1] * sy[k] + W_inv[2];
        const double my = W_inv[3] * sx[k] + W_inv[4] * sy[k] + W_inv[5];
        const double md = W_inv[6] * sx[k] + W_inv[7] * sy[k] + W_inv[8];
        if (!(md > 1e-4)) {
            all_in = false;
            return;
        }
        tx[k] = mx / md;
        ty[k] = my / md;
    }
    const int rx0 = std::max(1, static_cast<int>(std::ceil(std::max(tx[0], tx[3]))));
    const int rx1 = std::min(w - 1, static_cast<int>(std::floor(std::min(tx[1], tx[2]))) + 1);
    const int ry0 = std::max(1, static_cast<int>(std::ceil(std::max(ty[0], ty[1]))));
    const int ry1 = std::min(h - 1, static_cast<int>(std::floor(std::min(ty[2], ty[3]))) + 1);
    const int64_t area = static_cast<int64_t>(std::max(0, rx1 - rx0)) * std::max(0, ry1 - ry0);
    if (rx1 - rx0 < 12 || ry1 - ry0 < 12 || area * 4 < static_cast<int64_t>(w) * h) {
        all_in = false;
        return;
    }
    y0 = ry0;
    y1 = ry1;
    x0 = rx0;
    x1 = rx1;
    all_in = roi_all_in(W, w, h, y0, y1, x0, x1);
}

inline double max_corner_step_px(const double M_new[9], const double M_old[9],
                                 double cx, double cy, double s, int w, int h) {
    const double u_min = (0.0 - cx) / s;
    const double u_max = (w - 1.0 - cx) / s;
    const double v_min = (0.0 - cy) / s;
    const double v_max = (h - 1.0 - cy) / s;
    const double cu[4] = {u_min, u_max, u_max, u_min};
    const double cv[4] = {v_min, v_min, v_max, v_max};
    double max_d2 = 0.0;
    for (int k = 0; k < 4; ++k) {
        const double d1 = M_new[6] * cu[k] + M_new[7] * cv[k] + M_new[8];
        const double d0 = M_old[6] * cu[k] + M_old[7] * cv[k] + M_old[8];
        if (std::abs(d1) < 1e-6 || std::abs(d0) < 1e-6) {
            return std::numeric_limits<double>::infinity();
        }
        const double x1 = (M_new[0] * cu[k] + M_new[1] * cv[k] + M_new[2]) / d1;
        const double y1 = (M_new[3] * cu[k] + M_new[4] * cv[k] + M_new[5]) / d1;
        const double x0 = (M_old[0] * cu[k] + M_old[1] * cv[k] + M_old[2]) / d0;
        const double y0 = (M_old[3] * cu[k] + M_old[4] * cv[k] + M_old[5]) / d0;
        const double dx = (x1 - x0) * s;
        const double dy = (y1 - y0) * s;
        const double d2 = dx * dx + dy * dy;
        if (d2 > max_d2) {
            max_d2 = d2;
        }
    }
    return std::sqrt(max_d2);
}

inline bool homography_plausible(const double W_in[9], int w, int h) {
    for (int k = 0; k < 9; ++k) {
        if (!std::isfinite(W_in[k])) {
            return false;
        }
    }
    if (std::abs(W_in[8]) <= 1e-6) {
        return false;
    }
    double W[9];
    const double inv22 = 1.0 / W_in[8];
    for (int k = 0; k < 9; ++k) {
        W[k] = W_in[k] * inv22;
    }
    const double cx[4] = {0.0, static_cast<double>(w - 1), static_cast<double>(w - 1), 0.0};
    const double cy[4] = {0.0, 0.0, static_cast<double>(h - 1), static_cast<double>(h - 1)};
    double px[4], py[4];
    for (int k = 0; k < 4; ++k) {
        const double den = W[6] * cx[k] + W[7] * cy[k] + W[8];
        if (!(den > 0.2 && den < 5.0)) {
            return false;
        }
        px[k] = (W[0] * cx[k] + W[1] * cy[k] + W[2]) / den;
        py[k] = (W[3] * cx[k] + W[4] * cy[k] + W[5]) / den;
    }
    double area2 = 0.0;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        area2 += px[i] * py[j] - px[j] * py[i];
    }
    const double img_area = static_cast<double>(w) * static_cast<double>(h);
    if (area2 <= 0.1 * img_area || area2 >= 10.0 * img_area) {
        return false;
    }
    const double det_ul = W[0] * W[4] - W[1] * W[3];
    if (det_ul <= 0.1 || det_ul >= 10.0) {
        return false;
    }
    return true;
}

inline void scale_homography(double W[9], double sx, double sy) {
    W[1] *= sx / sy;
    W[2] *= sx;
    W[3] *= sy / sx;
    W[5] *= sy;
    W[6] /= sx;
    W[7] /= sy;
    const double inv22 = 1.0 / W[8];
    for (int k = 0; k < 9; ++k) {
        W[k] *= inv22;
    }
}

struct PreparedLevel {
    int rh = 0;
    int rw = 0;
    std::vector<float> T0;
    std::vector<float> gx;
    std::vector<float> gy;
    double norm_T2 = 0.0;
    double j_sum[8]{};
    double t_vec[8]{};
    double H[64]{};
};

template <int P>
void prepare_level_impl(
    const GrayView& tpl,
    int y0,
    int y1,
    int x0,
    int x1,
    double cx,
    double cy,
    double s,
    double inv_s,
    int num_threads,
    PreparedLevel& out
) {
    const int rh = y1 - y0;
    const int rw = x1 - x0;
    const int w = tpl.width;
    const double inv_n = 1.0 / static_cast<double>(static_cast<int64_t>(rh) * rw);
    out.rh = rh;
    out.rw = rw;
    out.T0.resize(static_cast<size_t>(rh) * rw);
    out.gx.resize(static_cast<size_t>(rh) * rw);
    out.gy.resize(static_cast<size_t>(rh) * rw);

    const int64_t band = 64;
    const int64_t n_tasks = (rh + band - 1) / band;
    const int threads = (static_cast<int64_t>(rh) * rw >= 16384) ? num_threads : 1;

    std::vector<double> mean_parts(static_cast<size_t>(n_tasks), 0.0);
    openfocus::parallel_bands(rh, band, threads, [&](int64_t start, int64_t stop) {
        double acc = 0.0;
        for (int64_t ry = start; ry < stop; ++ry) {
            const float* row = tpl.data + (y0 + ry) * w;
            double rs = 0.0;
            for (int rx = 0; rx < rw; ++rx) {
                rs += static_cast<double>(row[x0 + rx]);
            }
            acc += rs;
        }
        mean_parts[static_cast<size_t>(start / band)] = acc;
    });
    double sum_t = 0.0;
    for (double v : mean_parts) {
        sum_t += v;
    }
    const double mean_t = sum_t * inv_n;

    const double g_scale = s * 0.125;
    const bool stride2 = rh >= 64 && rw >= 64;
    const bool use_avx2 = rw >= 8 && simd::use_avx2();

    struct PrepBand {
        double norm_T2 = 0.0;
        double js[P]{};
        double tv[P]{};
        double H[(P * (P + 1)) / 2]{};
    };
    std::vector<PrepBand> bands(static_cast<size_t>(n_tasks));

    openfocus::parallel_bands(rh, band, threads, [&](int64_t start, int64_t stop) {
        PrepBand local{};
        for (int64_t ry = start; ry < stop; ++ry) {
            const int y = y0 + static_cast<int>(ry);
            const double v = (static_cast<double>(y) - cy) * inv_s;
            const bool even_y = (ry & 1) == 0;
            const bool sample_h_row = even_y || (!stride2);

            const float* rm1 = tpl.data + static_cast<int64_t>(y - 1) * w;
            const float* r0 = tpl.data + static_cast<int64_t>(y) * w;
            const float* rp1 = tpl.data + static_cast<int64_t>(y + 1) * w;

            float* dst_t0 = out.T0.data() + ry * rw;
            float* dst_gx = out.gx.data() + ry * rw;
            float* dst_gy = out.gy.data() + ry * rw;

            if (use_avx2) {
                simd::ecc_prepare_level_row_avx2(
                    rm1, r0, rp1, x0, rw, cx, inv_s, mean_t, g_scale, v,
                    P, sample_h_row, stride2, dst_t0, dst_gx, dst_gy,
                    local.norm_T2, local.js, local.tv, local.H);
                continue;
            }

            for (int rx = 0; rx < rw; ++rx) {
                const int x = x0 + rx;
                const double u = (static_cast<double>(x) - cx) * inv_s;
                const double t0 = static_cast<double>(r0[x]) - mean_t;

                const double tl = static_cast<double>(rm1[x - 1]);
                const double tc = static_cast<double>(rm1[x]);
                const double tr = static_cast<double>(rm1[x + 1]);
                const double ml = static_cast<double>(r0[x - 1]);
                const double mr = static_cast<double>(r0[x + 1]);
                const double bl = static_cast<double>(rp1[x - 1]);
                const double bc = static_cast<double>(rp1[x]);
                const double br = static_cast<double>(rp1[x + 1]);

                const double gx = ((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)) * g_scale;
                const double gy = ((bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)) * g_scale;

                dst_t0[rx] = static_cast<float>(t0);
                dst_gx[rx] = static_cast<float>(gx);
                dst_gy[rx] = static_cast<float>(gy);

                local.norm_T2 += t0 * t0;

                double J[P];
                if constexpr (P == 6) {
                    J[0] = gx * u;
                    J[1] = gy * u;
                    J[2] = gx * v;
                    J[3] = gy * v;
                    J[4] = gx;
                    J[5] = gy;
                } else {
                    const double guv = gx * u + gy * v;
                    J[0] = gx * u;
                    J[1] = gy * u;
                    J[2] = -u * guv;
                    J[3] = gx * v;
                    J[4] = gy * v;
                    J[5] = -v * guv;
                    J[6] = gx;
                    J[7] = gy;
                }

                for (int r = 0; r < P; ++r) {
                    local.js[r] += J[r];
                    local.tv[r] += J[r] * t0;
                }
                if (sample_h_row && ((rx & 1) == 0 || !stride2)) {
                    int idx = 0;
                    for (int r = 0; r < P; ++r) {
                        const double jr = J[r];
                        for (int c = r; c < P; ++c) {
                            local.H[idx++] += jr * J[c];
                        }
                    }
                }
            }
        }
        bands[static_cast<size_t>(start / band)] = local;
    });

    out.norm_T2 = 0.0;
    std::fill(std::begin(out.j_sum), std::end(out.j_sum), 0.0);
    std::fill(std::begin(out.t_vec), std::end(out.t_vec), 0.0);
    double H_tri[(P * (P + 1)) / 2]{};
    for (const auto& b : bands) {
        out.norm_T2 += b.norm_T2;
        for (int r = 0; r < P; ++r) {
            out.j_sum[r] += b.js[r];
            out.t_vec[r] += b.tv[r];
        }
        for (int k = 0; k < (P * (P + 1)) / 2; ++k) {
            H_tri[k] += b.H[k];
        }
    }
    const double h_scale = stride2 ? 4.0 : 1.0;
    int idx = 0;
    for (int r = 0; r < P; ++r) {
        for (int c = r; c < P; ++c) {
            const double val = H_tri[idx++] * h_scale - out.j_sum[r] * out.j_sum[c] * inv_n;
            out.H[r * P + c] = val;
            out.H[c * P + r] = val;
        }
    }
}

template <int P>
struct IcPassAccum {
    int64_t cnt = 0;
    double sum_iw = 0.0;
    double sum_iw2 = 0.0;
    double sum_tiw = 0.0;
    double oob_t = 0.0;
    double oob_t2 = 0.0;
    double raw_g[P]{};
    double oob_j[P]{};
    double oob_jt[P]{};
};

template <int P>
IcPassAccum<P> ic_pass_impl(
    const PreparedLevel& lvl,
    const GrayView& src,
    const double M[9],
    int y0,
    int x0,
    double cx,
    double cy,
    double s,
    double inv_s,
    bool all_in,
    int num_threads
) {
    const int rh = lvl.rh;
    const int rw = lvl.rw;
    const int h = src.height;
    const int w = src.width;
    const double m00 = M[0], m01 = M[1], m02 = M[2];
    const double m10 = M[3], m11 = M[4], m12 = M[5];
    const double m20 = M[6], m21 = M[7], m22 = M[8];
    const bool use_avx2 = all_in && rw >= 8 && simd::use_avx2();

    const double u0 = (static_cast<double>(x0) - cx) * inv_s;
    const double w_max = static_cast<double>(w - 1);
    const double h_max = static_cast<double>(h - 1);

    const int64_t band = 64;
    const int64_t n_tasks = (rh + band - 1) / band;
    const int threads = (static_cast<int64_t>(rh) * rw >= 16384) ? num_threads : 1;
    std::vector<IcPassAccum<P>> bands(static_cast<size_t>(n_tasks));

    openfocus::parallel_bands(rh, band, threads, [&](int64_t start, int64_t stop) {
        IcPassAccum<P> local{};
        for (int64_t ry = start; ry < stop; ++ry) {
            const double v = (static_cast<double>(y0 + ry) - cy) * inv_s;
            const float* t0_row = lvl.T0.data() + ry * rw;
            const float* gx_row = lvl.gx.data() + ry * rw;
            const float* gy_row = lvl.gy.data() + ry * rw;
            double u = u0;
            double rs_iw = 0.0;
            double rs_iw2 = 0.0;
            double rs_tiw = 0.0;

            if constexpr (P == 6) {
                double sx = (m00 * u0 + m01 * v + m02) * s + cx;
                double sy = (m10 * u0 + m11 * v + m12) * s + cy;
                const double dsx = m00;
                const double dsy = m10;
                if (all_in) {
                    if (use_avx2) {
                        simd::ecc_ic_pass_all_in_row_p6_avx2(
                            src.data, w, h, t0_row, gx_row, gy_row, rw,
                            sx, sy, dsx, dsy, u0, inv_s, v,
                            rs_iw, rs_iw2, rs_tiw, local.raw_g);
                    } else {
                        for (int rx = 0; rx < rw; ++rx) {
                        int ix = static_cast<int>(sx);
                        int iy = static_cast<int>(sy);
                        if (ix >= w - 1) ix = w - 2; else if (ix < 0) ix = 0;
                        if (iy >= h - 1) iy = h - 2; else if (iy < 0) iy = 0;
                        const double fx = sx - static_cast<double>(ix);
                        const double fy = sy - static_cast<double>(iy);
                        const float* p0 = src.data + static_cast<int64_t>(iy) * w + ix;
                        const float* p1 = p0 + w;
                        const double v00 = p0[0], v01 = p0[1];
                        const double v10 = p1[0], v11 = p1[1];
                        const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                                        + (v10 + fx * (v11 - v10)) * fy;
                        const double t0 = t0_row[rx];
                        const double gxw = static_cast<double>(gx_row[rx]) * iw;
                        const double gyw = static_cast<double>(gy_row[rx]) * iw;
                        rs_iw += iw;
                        rs_iw2 += iw * iw;
                        rs_tiw += t0 * iw;
                        local.raw_g[0] += gxw * u;
                        local.raw_g[1] += gyw * u;
                        local.raw_g[2] += gxw * v;
                        local.raw_g[3] += gyw * v;
                        local.raw_g[4] += gxw;
                        local.raw_g[5] += gyw;
                        sx += dsx;
                        sy += dsy;
                        u += inv_s;
                        }
                    }
                    local.cnt += rw;
                } else {
                    for (int rx = 0; rx < rw; ++rx) {
                        const double t0 = t0_row[rx];
                        const double gx = gx_row[rx];
                        const double gy = gy_row[rx];
                        if (sx >= 0.0 && sx <= w_max && sy >= 0.0 && sy <= h_max) {
                            int ix = static_cast<int>(sx);
                            int iy = static_cast<int>(sy);
                            if (ix >= w - 1) ix = w - 2;
                            if (iy >= h - 1) iy = h - 2;
                            const double fx = sx - static_cast<double>(ix);
                            const double fy = sy - static_cast<double>(iy);
                            const float* p0 = src.data + static_cast<int64_t>(iy) * w + ix;
                            const float* p1 = p0 + w;
                            const double v00 = p0[0], v01 = p0[1];
                            const double v10 = p1[0], v11 = p1[1];
                            const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                                            + (v10 + fx * (v11 - v10)) * fy;
                            const double gxw = gx * iw;
                            const double gyw = gy * iw;
                            rs_iw += iw;
                            rs_iw2 += iw * iw;
                            rs_tiw += t0 * iw;
                            local.raw_g[0] += gxw * u;
                            local.raw_g[1] += gyw * u;
                            local.raw_g[2] += gxw * v;
                            local.raw_g[3] += gyw * v;
                            local.raw_g[4] += gxw;
                            local.raw_g[5] += gyw;
                            local.cnt += 1;
                        } else {
                            const double j[6] = {gx * u, gy * u, gx * v, gy * v, gx, gy};
                            local.oob_t += t0;
                            local.oob_t2 += t0 * t0;
                            for (int k = 0; k < 6; ++k) {
                                local.oob_j[k] += j[k];
                                local.oob_jt[k] += j[k] * t0;
                            }
                        }
                        sx += dsx;
                        sy += dsy;
                        u += inv_s;
                    }
                }
            } else {
                const double d_den = m20 * inv_s;
                const double d_nsx = m00 + cx * d_den;
                const double d_nsy = m10 + cy * d_den;
                double den = m20 * u0 + m21 * v + m22;
                double nsx = (m00 * u0 + m01 * v + m02) * s + cx * den;
                double nsy = (m10 * u0 + m11 * v + m12) * s + cy * den;
                if (all_in) {
                    if (use_avx2) {
                        simd::ecc_ic_pass_all_in_row_p8_avx2(
                            src.data, w, h, t0_row, gx_row, gy_row, rw,
                            nsx, nsy, den, d_nsx, d_nsy, d_den, u0, inv_s, v,
                            rs_iw, rs_iw2, rs_tiw, local.raw_g);
                    } else {
                        for (int rx = 0; rx < rw; ++rx) {
                        const double inv_d = 1.0 / den;
                        const double sx = nsx * inv_d;
                        const double sy = nsy * inv_d;
                        int ix = static_cast<int>(sx);
                        int iy = static_cast<int>(sy);
                        if (ix >= w - 1) ix = w - 2; else if (ix < 0) ix = 0;
                        if (iy >= h - 1) iy = h - 2; else if (iy < 0) iy = 0;
                        const double fx = sx - static_cast<double>(ix);
                        const double fy = sy - static_cast<double>(iy);
                        const float* p0 = src.data + static_cast<int64_t>(iy) * w + ix;
                        const float* p1 = p0 + w;
                        const double v00 = p0[0], v01 = p0[1];
                        const double v10 = p1[0], v11 = p1[1];
                        const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                                        + (v10 + fx * (v11 - v10)) * fy;
                        const double t0 = t0_row[rx];
                        const double gxw = static_cast<double>(gx_row[rx]) * iw;
                        const double gyw = static_cast<double>(gy_row[rx]) * iw;
                        const double guvw = gxw * u + gyw * v;
                        rs_iw += iw;
                        rs_iw2 += iw * iw;
                        rs_tiw += t0 * iw;
                        local.raw_g[0] += gxw * u;
                        local.raw_g[1] += gyw * u;
                        local.raw_g[2] -= u * guvw;
                        local.raw_g[3] += gxw * v;
                        local.raw_g[4] += gyw * v;
                        local.raw_g[5] -= v * guvw;
                        local.raw_g[6] += gxw;
                        local.raw_g[7] += gyw;
                        nsx += d_nsx;
                        nsy += d_nsy;
                        den += d_den;
                        u += inv_s;
                        }
                    }
                    local.cnt += rw;
                } else {
                    for (int rx = 0; rx < rw; ++rx) {
                        const double t0 = t0_row[rx];
                        const double gx = gx_row[rx];
                        const double gy = gy_row[rx];
                        bool valid = false;
                        double sx = 0.0, sy = 0.0;
                        if (den > 1e-7) {
                            const double inv_d = 1.0 / den;
                            sx = nsx * inv_d;
                            sy = nsy * inv_d;
                            if (sx >= 0.0 && sx <= w_max && sy >= 0.0 && sy <= h_max) {
                                valid = true;
                            }
                        }
                        if (valid) {
                            int ix = static_cast<int>(sx);
                            int iy = static_cast<int>(sy);
                            if (ix >= w - 1) ix = w - 2;
                            if (iy >= h - 1) iy = h - 2;
                            const double fx = sx - static_cast<double>(ix);
                            const double fy = sy - static_cast<double>(iy);
                            const float* p0 = src.data + static_cast<int64_t>(iy) * w + ix;
                            const float* p1 = p0 + w;
                            const double v00 = p0[0], v01 = p0[1];
                            const double v10 = p1[0], v11 = p1[1];
                            const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                                            + (v10 + fx * (v11 - v10)) * fy;
                            const double gxw = gx * iw;
                            const double gyw = gy * iw;
                            const double guvw = gxw * u + gyw * v;
                            rs_iw += iw;
                            rs_iw2 += iw * iw;
                            rs_tiw += t0 * iw;
                            local.raw_g[0] += gxw * u;
                            local.raw_g[1] += gyw * u;
                            local.raw_g[2] -= u * guvw;
                            local.raw_g[3] += gxw * v;
                            local.raw_g[4] += gyw * v;
                            local.raw_g[5] -= v * guvw;
                            local.raw_g[6] += gxw;
                            local.raw_g[7] += gyw;
                            local.cnt += 1;
                        } else {
                            const double guv = gx * u + gy * v;
                            const double j[8] = {
                                gx * u, gy * u, -u * guv,
                                gx * v, gy * v, -v * guv,
                                gx, gy
                            };
                            local.oob_t += t0;
                            local.oob_t2 += t0 * t0;
                            for (int k = 0; k < 8; ++k) {
                                local.oob_j[k] += j[k];
                                local.oob_jt[k] += j[k] * t0;
                            }
                        }
                        nsx += d_nsx;
                        nsy += d_nsy;
                        den += d_den;
                        u += inv_s;
                    }
                }
            }
            local.sum_iw += rs_iw;
            local.sum_iw2 += rs_iw2;
            local.sum_tiw += rs_tiw;
        }
        bands[static_cast<size_t>(start / band)] = local;
    });

    IcPassAccum<P> tot{};
    for (const auto& b : bands) {
        tot.cnt += b.cnt;
        tot.sum_iw += b.sum_iw;
        tot.sum_iw2 += b.sum_iw2;
        tot.sum_tiw += b.sum_tiw;
        tot.oob_t += b.oob_t;
        tot.oob_t2 += b.oob_t2;
        for (int k = 0; k < P; ++k) {
            tot.raw_g[k] += b.raw_g[k];
            tot.oob_j[k] += b.oob_j[k];
            tot.oob_jt[k] += b.oob_jt[k];
        }
    }
    return tot;
}

template <int P>
void solve_level_impl(
    const GrayView& tpl,
    const GrayView& src,
    const double W_init[9],
    int max_iter,
    double eps_px,
    int num_threads,
    double W_out[9],
    double& best_rho_out
) {
    std::memcpy(W_out, W_init, sizeof(double) * 9);
    best_rho_out = -2.0;

    const int h = tpl.height;
    const int w = tpl.width;
    double cx, cy, s, inv_s, T[9], T_inv[9];
    centered_mats(w, h, cx, cy, s, inv_s, T, T_inv);
    int y0, y1, x0, x1;
    bool all_in = false;
    inscribed_roi(W_init, w, h, y0, y1, x0, x1, all_in);
    if (x1 - x0 < 8 || y1 - y0 < 8) {
        return;
    }

    PreparedLevel lvl;
    prepare_level_impl<P>(tpl, y0, y1, x0, x1, cx, cy, s, inv_s, num_threads, lvl);
    if (lvl.norm_T2 <= 1e-12) {
        return;
    }

    double max_diag = 0.0;
    for (int i = 0; i < P; ++i) {
        const double d = lvl.H[i * P + i];
        if (!std::isfinite(d)) {
            return;
        }
        if (d > max_diag) {
            max_diag = d;
        }
    }
    if (!(max_diag > 1e-15)) {
        return;
    }
    double H_reg[P * P];
    std::memcpy(H_reg, lvl.H, sizeof(double) * P * P);
    const double reg = std::max(1e-12, 1e-8 * max_diag);
    for (int i = 0; i < P; ++i) {
        H_reg[i * P + i] += reg;
    }
    double hp_base[P];
    if (!solve_linear_system<P>(H_reg, lvl.t_vec, hp_base)) {
        return;
    }
    double dot_t_hp = 0.0;
    for (int i = 0; i < P; ++i) {
        dot_t_hp += lvl.t_vec[i] * hp_base[i];
    }
    const double A_base = lvl.norm_T2 - dot_t_hp;
    if (!std::isfinite(A_base) || A_base <= 1e-12) {
        return;
    }

    double tmp[9], M[9];
    mat3_mul(T, W_init, tmp);
    mat3_mul(tmp, T_inv, M);
    if (std::abs(M[8]) <= 1e-9) {
        return;
    }
    const double m22_inv = 1.0 / M[8];
    for (int k = 0; k < 9; ++k) {
        M[k] *= m22_inv;
    }
    if constexpr (P == 6) {
        M[6] = 0.0;
        M[7] = 0.0;
        M[8] = 1.0;
    }

    const int64_t n_pix = static_cast<int64_t>(y1 - y0) * (x1 - x0);
    double best_rho = -2.0;
    double best_M[9];
    std::memcpy(best_M, M, sizeof(double) * 9);
    double prev_rho = -2.0;
    bool eval_matches_M = false;
    bool converged = false;

    for (int it = 0; it < max_iter; ++it) {
        bool iter_all_in = all_in;
        if (it > 0) {
            double W_tmp[9];
            mat3_mul(T_inv, M, tmp);
            mat3_mul(tmp, T, W_tmp);
            iter_all_in = roi_all_in(W_tmp, w, h, y0, y1, x0, x1);
        }

        const auto acc = ic_pass_impl<P>(lvl, src, M, y0, x0, cx, cy, s, inv_s, iter_all_in, num_threads);
        if (acc.cnt * 4 < n_pix || acc.cnt <= 0) {
            break;
        }
        const double n_f = static_cast<double>(acc.cnt);
        const double mean_iw = acc.sum_iw / n_f;
        const double var_iw = acc.sum_iw2 - acc.sum_iw * mean_iw;
        if (!std::isfinite(var_iw) || var_iw <= 1e-12) {
            break;
        }

        double norm_T2_v = lvl.norm_T2;
        double cov = acc.sum_tiw;
        double j_sum_v[P];
        double hp[P];
        double A = A_base;
        if (acc.cnt == n_pix) {
            for (int i = 0; i < P; ++i) {
                j_sum_v[i] = lvl.j_sum[i];
                hp[i] = hp_base[i];
            }
        } else {
            const double sum_t_v = -acc.oob_t;
            const double mean_t_v = sum_t_v / n_f;
            norm_T2_v = (lvl.norm_T2 - acc.oob_t2) - sum_t_v * mean_t_v;
            if (!std::isfinite(norm_T2_v) || norm_T2_v <= 1e-12) {
                break;
            }
            cov = acc.sum_tiw - mean_t_v * acc.sum_iw;
            double t_vec_v[P];
            for (int i = 0; i < P; ++i) {
                j_sum_v[i] = lvl.j_sum[i] - acc.oob_j[i];
                t_vec_v[i] = (lvl.t_vec[i] - acc.oob_jt[i]) - mean_t_v * j_sum_v[i];
            }
            if (!solve_linear_system<P>(H_reg, t_vec_v, hp)) {
                break;
            }
            double dot_v = 0.0;
            for (int i = 0; i < P; ++i) {
                dot_v += t_vec_v[i] * hp[i];
            }
            A = norm_T2_v - dot_v;
            if (!std::isfinite(A) || A <= 1e-12) {
                break;
            }
        }

        const double rho = cov / std::sqrt(norm_T2_v * var_iw);
        if (!std::isfinite(rho)) {
            break;
        }
        eval_matches_M = true;
        if (rho > best_rho) {
            best_rho = rho;
            std::memcpy(best_M, M, sizeof(double) * 9);
        }
        if (it >= 3 && rho < prev_rho - 2e-3) {
            break;
        }
        prev_rho = rho;

        double g[P];
        for (int i = 0; i < P; ++i) {
            g[i] = acc.raw_g[i] - mean_iw * j_sum_v[i];
        }
        double hw[P];
        if (!solve_linear_system<P>(H_reg, g, hw)) {
            break;
        }
        double dot_g_hp = 0.0;
        for (int i = 0; i < P; ++i) {
            dot_g_hp += g[i] * hp[i];
        }
        const double lam_d = cov - dot_g_hp;
        if (!std::isfinite(lam_d) || lam_d <= 1e-12 * A) {
            break;
        }
        const double lam = A / lam_d;
        double dp[P];
        for (int i = 0; i < P; ++i) {
            dp[i] = lam * hw[i] - hp[i];
        }

        double M_inc[9];
        if constexpr (P == 6) {
            M_inc[0] = 1.0 + dp[0]; M_inc[1] = dp[2];       M_inc[2] = dp[4];
            M_inc[3] = dp[1];       M_inc[4] = 1.0 + dp[3]; M_inc[5] = dp[5];
            M_inc[6] = 0.0;         M_inc[7] = 0.0;         M_inc[8] = 1.0;
        } else {
            M_inc[0] = 1.0 + dp[0]; M_inc[1] = dp[3];       M_inc[2] = dp[6];
            M_inc[3] = dp[1];       M_inc[4] = 1.0 + dp[4]; M_inc[5] = dp[7];
            M_inc[6] = dp[2];       M_inc[7] = dp[5];       M_inc[8] = 1.0;
        }

        double M_inc_inv[9];
        if (!mat3_inv(M_inc, M_inc_inv)) {
            break;
        }
        double M_cand[9];
        mat3_mul(M, M_inc_inv, M_cand);
        bool finite_cand = true;
        for (int k = 0; k < 9; ++k) {
            if (!std::isfinite(M_cand[k])) {
                finite_cand = false;
                break;
            }
        }
        if (!finite_cand || std::abs(M_cand[8]) <= 1e-7) {
            break;
        }
        const double cand_inv = 1.0 / M_cand[8];
        for (int k = 0; k < 9; ++k) {
            M_cand[k] *= cand_inv;
        }
        if constexpr (P == 6) {
            M_cand[6] = 0.0;
            M_cand[7] = 0.0;
            M_cand[8] = 1.0;
        }

        const double step_px = max_corner_step_px(M_cand, M, cx, cy, s, w, h);
        if (!std::isfinite(step_px) || step_px > 0.5 * std::max(w, h)) {
            break;
        }
        std::memcpy(M, M_cand, sizeof(double) * 9);
        eval_matches_M = false;
        if (step_px < eps_px) {
            converged = true;
            break;
        }
    }

    if (!eval_matches_M) {
        double W_last[9];
        mat3_mul(T_inv, M, tmp);
        mat3_mul(tmp, T, W_last);
        if (std::abs(W_last[8]) > 1e-7) {
            const double inv22 = 1.0 / W_last[8];
            for (int k = 0; k < 9; ++k) {
                W_last[k] *= inv22;
            }
            const bool last_all_in = roi_all_in(W_last, w, h, y0, y1, x0, x1);
            auto [rho_last, valid_last] = ecc_eval_rho_warped_cpp(
                tpl, src, W_last, y0, y1, x0, x1, last_all_in);
            if (valid_last > 0 && (rho_last > best_rho || (converged && rho_last >= best_rho - 5e-3))) {
                best_rho = rho_last;
                std::memcpy(best_M, M, sizeof(double) * 9);
            }
        }
    }

    double W_best[9];
    mat3_mul(T_inv, best_M, tmp);
    mat3_mul(tmp, T, W_best);
    bool finite_best = true;
    for (int k = 0; k < 9; ++k) {
        if (!std::isfinite(W_best[k])) {
            finite_best = false;
            break;
        }
    }
    if (!finite_best || std::abs(W_best[8]) <= 1e-7) {
        return;
    }
    const double inv22 = 1.0 / W_best[8];
    for (int k = 0; k < 9; ++k) {
        W_out[k] = W_best[k] * inv22;
    }
    best_rho_out = best_rho;
}

}  // namespace

IdenticalStatsResult ecc_check_identical_and_stats_cpp(
    const GrayView& tpl,
    const GrayView& src,
    int num_threads
) {
    const int h = tpl.height;
    const int w = tpl.width;
    const double t_first = static_cast<double>(tpl.data[0]);
    const double s_first = static_cast<double>(src.data[0]);
    if (!std::isfinite(t_first) || !std::isfinite(s_first)) {
        return {false, false, false, -2.0};
    }

    const int64_t band = 64;
    const int64_t n_tasks = (h + band - 1) / band;
    const int threads = (static_cast<int64_t>(h) * w >= 32768) ? num_threads : 1;

    struct StatBand {
        bool finite = true;
        double t_min = 0.0;
        double t_max = 0.0;
        double s_min = 0.0;
        double s_max = 0.0;
        double max_diff = 0.0;
        double st = 0.0;
        double st2 = 0.0;
        double sw = 0.0;
        double sw2 = 0.0;
        double stw = 0.0;
    };
    std::vector<StatBand> bands(static_cast<size_t>(n_tasks));

    openfocus::parallel_bands(h, band, threads, [&](int64_t start, int64_t stop) {
        StatBand local{};
        local.t_min = std::numeric_limits<double>::infinity();
        local.t_max = -std::numeric_limits<double>::infinity();
        local.s_min = std::numeric_limits<double>::infinity();
        local.s_max = -std::numeric_limits<double>::infinity();
        for (int64_t y = start; y < stop; ++y) {
            const float* tr = tpl.data + y * w;
            const float* sr = src.data + y * w;
            double rst = 0.0, rst2 = 0.0, rsw = 0.0, rsw2 = 0.0, rstw = 0.0;
            for (int x = 0; x < w; ++x) {
                const double tv = static_cast<double>(tr[x]);
                const double sv = static_cast<double>(sr[x]);
                if (!std::isfinite(tv) || !std::isfinite(sv)) {
                    local.finite = false;
                    bands[static_cast<size_t>(start / band)] = local;
                    return;
                }
                if (tv < local.t_min) local.t_min = tv;
                if (tv > local.t_max) local.t_max = tv;
                if (sv < local.s_min) local.s_min = sv;
                if (sv > local.s_max) local.s_max = sv;
                const double diff = std::abs(tv - sv);
                if (diff > local.max_diff) local.max_diff = diff;
                rst += tv;
                rst2 += tv * tv;
                rsw += sv;
                rsw2 += sv * sv;
                rstw += tv * sv;
            }
            local.st += rst;
            local.st2 += rst2;
            local.sw += rsw;
            local.sw2 += rsw2;
            local.stw += rstw;
        }
        bands[static_cast<size_t>(start / band)] = local;
    });

    double t_min = t_first, t_max = t_first;
    double s_min = s_first, s_max = s_first;
    double max_diff = 0.0;
    double st = 0.0, st2 = 0.0, sw = 0.0, sw2 = 0.0, stw = 0.0;
    for (const auto& b : bands) {
        if (!b.finite) {
            return {false, false, false, -2.0};
        }
        t_min = std::min(t_min, b.t_min);
        t_max = std::max(t_max, b.t_max);
        s_min = std::min(s_min, b.s_min);
        s_max = std::max(s_max, b.s_max);
        max_diff = std::max(max_diff, b.max_diff);
        st += b.st;
        st2 += b.st2;
        sw += b.sw;
        sw2 += b.sw2;
        stw += b.stw;
    }

    const bool is_nonflat = (t_max - t_min) > 1e-6 && (s_max - s_min) > 1e-6;
    if (!is_nonflat) {
        return {true, false, false, -2.0};
    }
    if (max_diff <= 1e-7) {
        return {true, true, true, 1.0};
    }
    const double inv_n = 1.0 / static_cast<double>(static_cast<int64_t>(h) * w);
    const double var_t = st2 - st * st * inv_n;
    const double var_w = sw2 - sw * sw * inv_n;
    double rho_ident = -2.0;
    if (var_t > 1e-12 && var_w > 1e-12) {
        rho_ident = (stw - st * sw * inv_n) / std::sqrt(var_t * var_w);
    }
    return {true, true, false, rho_ident};
}

std::tuple<double, int64_t> ecc_eval_rho_warped_cpp(
    const GrayView& tpl,
    const GrayView& src,
    const double W[9],
    int y0,
    int y1,
    int x0,
    int x1,
    bool all_in
) {
    const int h = src.height;
    const int w = src.width;
    const int tw = tpl.width;
    const double w00 = W[0], w01 = W[1], w02 = W[2];
    const double w10 = W[3], w11 = W[4], w12 = W[5];
    const double w20 = W[6], w21 = W[7], w22 = W[8];
    const double w_max = static_cast<double>(w - 1);
    const double h_max = static_cast<double>(h - 1);
    const bool is_affine = std::abs(w20) < 1e-13 && std::abs(w21) < 1e-13 && std::abs(w22 - 1.0) < 1e-12;

    double st = 0.0, st2 = 0.0, sw = 0.0, sw2 = 0.0, stw = 0.0;
    int64_t cnt = 0;

    for (int y = y0; y < y1; ++y) {
        const double yf = static_cast<double>(y);
        const float* trow = tpl.data + static_cast<int64_t>(y) * tw;
        if (is_affine) {
            double sx = w00 * static_cast<double>(x0) + w01 * yf + w02;
            double sy = w10 * static_cast<double>(x0) + w11 * yf + w12;
            if (all_in) {
                for (int x = x0; x < x1; ++x) {
                    int ix = static_cast<int>(sx);
                    int iy = static_cast<int>(sy);
                    if (ix >= w - 1) ix = w - 2; else if (ix < 0) ix = 0;
                    if (iy >= h - 1) iy = h - 2; else if (iy < 0) iy = 0;
                    const double fx = sx - static_cast<double>(ix);
                    const double fy = sy - static_cast<double>(iy);
                    const float* p0 = src.data + static_cast<int64_t>(iy) * w + ix;
                    const float* p1 = p0 + w;
                    const double v00 = p0[0], v01 = p0[1];
                    const double v10 = p1[0], v11 = p1[1];
                    const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                                    + (v10 + fx * (v11 - v10)) * fy;
                    const double tv = static_cast<double>(trow[x]);
                    st += tv; st2 += tv * tv;
                    sw += iw; sw2 += iw * iw;
                    stw += tv * iw;
                    cnt += 1;
                    sx += w00;
                    sy += w10;
                }
            } else {
                for (int x = x0; x < x1; ++x) {
                    if (sx >= 0.0 && sx <= w_max && sy >= 0.0 && sy <= h_max) {
                        int ix = static_cast<int>(sx);
                        int iy = static_cast<int>(sy);
                        if (ix >= w - 1) ix = w - 2;
                        if (iy >= h - 1) iy = h - 2;
                        const double fx = sx - static_cast<double>(ix);
                        const double fy = sy - static_cast<double>(iy);
                        const float* p0 = src.data + static_cast<int64_t>(iy) * w + ix;
                        const float* p1 = p0 + w;
                        const double v00 = p0[0], v01 = p0[1];
                        const double v10 = p1[0], v11 = p1[1];
                        const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                                        + (v10 + fx * (v11 - v10)) * fy;
                        const double tv = static_cast<double>(trow[x]);
                        st += tv; st2 += tv * tv;
                        sw += iw; sw2 += iw * iw;
                        stw += tv * iw;
                        cnt += 1;
                    }
                    sx += w00;
                    sy += w10;
                }
            }
        } else {
            double nx = w00 * static_cast<double>(x0) + w01 * yf + w02;
            double ny = w10 * static_cast<double>(x0) + w11 * yf + w12;
            double nd = w20 * static_cast<double>(x0) + w21 * yf + w22;
            for (int x = x0; x < x1; ++x) {
                if (nd > 1e-7) {
                    const double inv_d = 1.0 / nd;
                    const double sx = nx * inv_d;
                    const double sy = ny * inv_d;
                    if (all_in || (sx >= 0.0 && sx <= w_max && sy >= 0.0 && sy <= h_max)) {
                        int ix = static_cast<int>(sx);
                        int iy = static_cast<int>(sy);
                        if (ix >= w - 1) ix = w - 2; else if (ix < 0) ix = 0;
                        if (iy >= h - 1) iy = h - 2; else if (iy < 0) iy = 0;
                        const double fx = sx - static_cast<double>(ix);
                        const double fy = sy - static_cast<double>(iy);
                        const float* p0 = src.data + static_cast<int64_t>(iy) * w + ix;
                        const float* p1 = p0 + w;
                        const double v00 = p0[0], v01 = p0[1];
                        const double v10 = p1[0], v11 = p1[1];
                        const double iw = (v00 + fx * (v01 - v00)) * (1.0 - fy)
                                        + (v10 + fx * (v11 - v10)) * fy;
                        const double tv = static_cast<double>(trow[x]);
                        st += tv; st2 += tv * tv;
                        sw += iw; sw2 += iw * iw;
                        stw += tv * iw;
                        cnt += 1;
                    }
                }
                nx += w00;
                ny += w10;
                nd += w20;
            }
        }
    }

    const int64_t total = static_cast<int64_t>(y1 - y0) * (x1 - x0);
    if (cnt == 0 || cnt * 4 < total) {
        return {-2.0, 0};
    }
    const double inv_n = 1.0 / static_cast<double>(cnt);
    const double var_t = st2 - st * st * inv_n;
    const double var_w = sw2 - sw * sw * inv_n;
    if (var_t <= 1e-12 || var_w <= 1e-12) {
        return {-2.0, cnt};
    }
    const double cov = stw - st * sw * inv_n;
    return {cov / std::sqrt(var_t * var_w), cnt};
}

std::tuple<bool, double, std::vector<float>> ecc_refine_pyramids_cpp(
    const std::vector<GrayView>& tpl_pyr,
    const std::vector<GrayView>& src_pyr,
    const double W_coarsest_init[9],
    double rho_ident_full,
    int max_iter,
    double eps,
    int num_threads
) {
    std::vector<float> out_mat(9, 0.0f);
    out_mat[0] = out_mat[4] = out_mat[8] = 1.0f;
    if (tpl_pyr.empty() || tpl_pyr.size() != src_pyr.size()) {
        return {false, -2.0, out_mat};
    }

    double W[9];
    std::memcpy(W, W_coarsest_init, sizeof(double) * 9);
    const int n_levels = static_cast<int>(tpl_pyr.size());

    for (int lvl = n_levels - 1; lvl >= 0; --lvl) {
        const GrayView& tl = tpl_pyr[static_cast<size_t>(lvl)];
        const GrayView& sl = src_pyr[static_cast<size_t>(lvl)];
        const int hl = tl.height;
        const int wl = tl.width;
        const bool is_affine = (lvl > 0);
        const int lvl_iter = is_affine ? std::min(max_iter, (lvl == n_levels - 1) ? 16 : 12) : max_iter;
        const double lvl_eps = is_affine
            ? std::max(eps * 2.0, 1e-3)
            : std::min(std::max(eps * 2.0, 2e-4), 3e-4);

        double W_next[9];
        double rho_lvl = -2.0;
        if (is_affine) {
            solve_level_impl<6>(tl, sl, W, lvl_iter, lvl_eps, num_threads, W_next, rho_lvl);
        } else {
            solve_level_impl<8>(tl, sl, W, lvl_iter, lvl_eps, num_threads, W_next, rho_lvl);
        }
        if (homography_plausible(W_next, wl, hl)) {
            std::memcpy(W, W_next, sizeof(double) * 9);
        }
        if (lvl > 0) {
            const GrayView& tf = tpl_pyr[static_cast<size_t>(lvl - 1)];
            scale_homography(W, tf.width / static_cast<double>(wl),
                             tf.height / static_cast<double>(hl));
        }
    }

    const GrayView& t0 = tpl_pyr[0];
    const GrayView& s0 = src_pyr[0];
    const int h = t0.height;
    const int w = t0.width;
    if (!homography_plausible(W, w, h)) {
        return {false, -2.0, out_mat};
    }
    const bool final_all_in = roi_all_in(W, w, h, 0, h, 0, w);
    auto [final_cc, valid_final] = ecc_eval_rho_warped_cpp(t0, s0, W, 0, h, 0, w, final_all_in);
    if (valid_final == 0 || !std::isfinite(final_cc)) {
        return {false, -2.0, out_mat};
    }
    if (final_cc < 0.25 || final_cc < rho_ident_full - 1e-3) {
        if (rho_ident_full > final_cc && rho_ident_full > 0.9999) {
            return {true, rho_ident_full, out_mat};
        }
        return {false, -2.0, out_mat};
    }
    for (int k = 0; k < 9; ++k) {
        out_mat[static_cast<size_t>(k)] = static_cast<float>(W[k]);
    }
    return {true, final_cc, out_mat};
}

}  // namespace openfocus
