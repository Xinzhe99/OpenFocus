"""OpenCL GPU fusion for GFG-FGF (opt-in via use_gpu=True on gfgfgf_impl).

Ports channel selection, optional Scharr frame focus scoring, fused GFG gradient
and mean-difference focus mapping (IA), fast guided filter (s=4, r=5, eps=0.3),
Sobel-tie-broken streaming GfgDecision selection, weighted refinement/accumulation,
and normalized output conversion onto OpenCL 1.2 device kernels with single-frame
streaming and automatic CPU fallback.
"""
from __future__ import annotations

import math
from threading import Lock

import cv2
import numpy as np

from fusion_methods.dct import _ensure_color_image
from fusion_methods.openfocus_ab import _check_cancel
from utils.image_utils import fuse_output_dtype

_GFG_THRESHOLD = 0.005
_GF_RADIUS = 5
_GF_EPS = 0.3
_GF_SUB = 4
_TIE_EPS = 1e-6
_CHUNK = 64

_GFG_K = r'''
#pragma OPENCL FP_CONTRACT OFF
#define CHUNK 64

static inline int reflect_idx(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        if (i >= n) i = 2 * n - 1 - i;
    }
    return i;
}

static inline int reflect101_idx(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

#define DEFINE_CHAN_SUM(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global float4* wg_sums, \
                   const int n, const int C) { \
    __local float4 lsum[256]; \
    int lid = get_local_id(0); \
    int gid = get_global_id(0); \
    float3 acc = (float3)(0.0f, 0.0f, 0.0f); \
    int stride = get_global_size(0); \
    for (int i = gid; i < n; i += stride) { \
        if (C == 1) { \
            SRC_T raw0 = src[i]; \
            float v = (CONV_EXPR); \
            acc = acc + (float3)(v, v, v); \
        } else { \
            size_t b = (size_t)i * C; \
            SRC_T raw0 = src[b]; float c0 = (CONV_EXPR); \
            raw0 = src[b + 1]; float c1 = (CONV_EXPR); \
            raw0 = src[b + 2]; float c2 = (CONV_EXPR); \
            acc = acc + (float3)(c0, c1, c2); \
        } \
    } \
    lsum[lid] = (float4)(acc, 0.0f); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int s = 128; s > 0; s >>= 1) { \
        if (lid < s) lsum[lid] = lsum[lid] + lsum[lid + s]; \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (lid == 0) wg_sums[get_group_id(0)] = lsum[0]; \
}

DEFINE_CHAN_SUM(gfg_chan_sum_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_CHAN_SUM(gfg_chan_sum_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_CHAN_SUM(gfg_chan_sum_f32, float, (raw0 / 255.0f))

#define DEFINE_EXTRACT(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global float* gray, \
                   const int n, const int C, const int ch) { \
    int i = get_global_id(0); \
    if (i >= n) return; \
    SRC_T raw0 = (C == 1) ? src[i] : src[(size_t)i * C + ch]; \
    gray[i] = (CONV_EXPR); \
}

DEFINE_EXTRACT(gfg_extract_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_EXTRACT(gfg_extract_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_EXTRACT(gfg_extract_f32, float, (raw0 / 255.0f))

__kernel void gfg_scharr_sum(__global const float* gray, __global float* wg_sums,
                             const int H, const int W) {
    __local float lsum[256];
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int ih = H - 2;
    int iw = W - 2;
    int inner = (ih > 0 && iw > 0) ? (ih * iw) : 0;
    float acc = 0.0f;
    int stride = get_global_size(0);
    for (int idx = gid; idx < inner; idx += stride) {
        int y = 1 + idx / iw;
        int x = 1 + (idx - (y - 1) * iw);
        size_t rm = (size_t)(y - 1) * W;
        size_t r0 = (size_t)y * W;
        size_t rp = (size_t)(y + 1) * W;
        float a00 = gray[rm + x - 1], a01 = gray[rm + x], a02 = gray[rm + x + 1];
        float a10 = gray[r0 + x - 1],                     a12 = gray[r0 + x + 1];
        float a20 = gray[rp + x - 1], a21 = gray[rp + x], a22 = gray[rp + x + 1];
        float gx = (-3.0f * a00 - 10.0f * a01 - 3.0f * a02)
                 + (3.0f * a20 + 10.0f * a21 + 3.0f * a22);
        float gy = (-3.0f * a00 + 3.0f * a02)
                 + (-10.0f * a10 + 10.0f * a12)
                 + (-3.0f * a20 + 3.0f * a22);
        acc = acc + (gx * gx + gy * gy);
    }
    lsum[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (lid < s) lsum[lid] = lsum[lid] + lsum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) wg_sums[get_group_id(0)] = lsum[0];
}

__kernel void transpose_f1(__global const float* src, __global float* dst,
                           const int H, const int W) {
    __local float tile[16][17];
    int x = get_group_id(0) * 16 + get_local_id(0);
    int y = get_group_id(1) * 16 + get_local_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    if (x < W && y < H) tile[ly][lx] = src[(size_t)y * W + x];
    barrier(CLK_LOCAL_MEM_FENCE);
    int ox = get_group_id(1) * 16 + lx;
    int oy = get_group_id(0) * 16 + ly;
    if (ox < H && oy < W) dst[(size_t)oy * H + ox] = tile[lx][ly];
}

__kernel void box1_col_chunk(__global const float* src, __global float* dst,
                             const int N_row, const int N_col, const int rad) {
    int col = get_global_id(0);
    int r0 = get_global_id(1) * CHUNK;
    if (col >= N_col || r0 >= N_row) return;
    int r1 = min(r0 + CHUNK, N_row);
    float inv = 1.0f / (float)(2 * rad + 1);
    float ref = src[(size_t)r0 * N_col + col];
    float s = 0.0f;
    for (int d = -rad; d <= rad; ++d) {
        int rr = reflect101_idx(r0 + d, N_row);
        s = s + (src[(size_t)rr * N_col + col] - ref);
    }
    dst[(size_t)r0 * N_col + col] = ref + s * inv;
    for (int r = r0 + 1; r < r1; ++r) {
        int rr = reflect101_idx(r + rad, N_row);
        int rl = reflect101_idx(r - rad - 1, N_row);
        s = (s + (src[(size_t)rr * N_col + col] - ref))
          - (src[(size_t)rl * N_col + col] - ref);
        dst[(size_t)r * N_col + col] = ref + s * inv;
    }
}

__kernel void box1_col_sum(__global const float* src, __global float* dst,
                           const int H, const int W, const int rad) {
    int x = get_global_id(0);
    int y0 = get_global_id(1) * CHUNK;
    if (x >= W || y0 >= H) return;
    int y1 = min(y0 + CHUNK, H);
    float inv = 1.0f / (float)(2 * rad + 1);
    float ref = src[(size_t)y0 * W + x];
    float s = 0.0f;
    for (int d = -rad; d <= rad; ++d) {
        int yy = reflect101_idx(y0 + d, H);
        s = s + (src[(size_t)yy * W + x] - ref);
    }
    dst[(size_t)y0 * W + x] = ref + s * inv;
    for (int y = y0 + 1; y < y1; ++y) {
        int yr = reflect101_idx(y + rad, H);
        int yl = reflect101_idx(y - rad - 1, H);
        s = (s + (src[(size_t)yr * W + x] - ref))
          - (src[(size_t)yl * W + x] - ref);
        dst[(size_t)y * W + x] = ref + s * inv;
    }
}

__kernel void gfg_ia_combine(__global const float* gray, __global const float* blur,
                             __global float* ia, const int H, const int W,
                             const float threshold, const int use_blur) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int ym = reflect_idx(y - 1, H);
    int yp = reflect_idx(y + 1, H);
    int xm = reflect_idx(x - 1, W);
    int xp = reflect_idx(x + 1, W);

    const float t1 = 1.0f / 16.0f;
    const float t2 = 2.0f / 16.0f;
    const float t4 = 4.0f / 16.0f;

    float c = gray[(size_t)y * W + x];
    float g01 = gray[(size_t)ym * W + x];
    float g10 = gray[(size_t)y * W + xm];
    float g12 = gray[(size_t)y * W + xp];
    float g21 = gray[(size_t)yp * W + x];

    float p = (gray[(size_t)ym * W + xm] * t1 + g01 * t2) + gray[(size_t)ym * W + xp] * t1;
    p = ((p + g10 * t2) + c * t4) + g12 * t2;
    p = ((p + gray[(size_t)yp * W + xm] * t1) + g21 * t2) + gray[(size_t)yp * W + xp] * t1;

    float g = fabs(g12 - p) + fabs(g10 - p);
    g = (g + fabs(g21 - p)) + fabs(g01 - p);
    g = g * 0.5f;

    size_t pos = (size_t)y * W + x;
    if (g >= threshold) {
        ia[pos] = g;
    } else if (use_blur) {
        ia[pos] = fabs(c - blur[pos]);
    } else {
        ia[pos] = 0.0f;
    }
}

__kernel void gfg_area_down_pair(__global const float* gray, __global const float* pbuf,
                                 __global float4* sub4,
                                 const int H, const int W,
                                 const int hs, const int ws) {
    int xs = get_global_id(0);
    int ys = get_global_id(1);
    if (xs >= ws || ys >= hs) return;
    float sy_scale = (float)H / (float)hs;
    float sx_scale = (float)W / (float)ws;
    float fy0 = (float)ys * sy_scale;
    float fy1 = (float)(ys + 1) * sy_scale;
    float fx0 = (float)xs * sx_scale;
    float fx1 = (float)(xs + 1) * sx_scale;
    int y0 = max(0, (int)floor(fy0));
    int y1 = min(H, (int)ceil(fy1));
    int x0 = max(0, (int)floor(fx0));
    int x1 = min(W, (int)ceil(fx1));
    size_t ref_pos = (size_t)y0 * W + x0;
    float refI = gray[ref_pos];
    float refP = pbuf[ref_pos];
    float sI = 0.0f, sP = 0.0f, area = 0.0f;
    for (int sy = y0; sy < y1; ++sy) {
        float wy = fmin(fy1, (float)(sy + 1)) - fmax(fy0, (float)sy);
        if (wy <= 0.0f) continue;
        size_t row = (size_t)sy * W;
        for (int sx = x0; sx < x1; ++sx) {
            float wx = fmin(fx1, (float)(sx + 1)) - fmax(fx0, (float)sx);
            if (wx <= 0.0f) continue;
            float wt = wy * wx;
            area = area + wt;
            size_t pos = row + sx;
            sI = sI + (gray[pos] - refI) * wt;
            sP = sP + (pbuf[pos] - refP) * wt;
        }
    }
    float iv = refI + sI / area;
    float pv = refP + sP / area;
    sub4[(size_t)ys * ws + xs] = (float4)(iv, pv, iv * pv, iv * iv);
}

__kernel void gfg_area_down_mask(__global const float* gray, __global const int* idx,
                                 __global float4* sub4,
                                 const int H, const int W,
                                 const int hs, const int ws, const int frame_k) {
    int xs = get_global_id(0);
    int ys = get_global_id(1);
    if (xs >= ws || ys >= hs) return;
    float sy_scale = (float)H / (float)hs;
    float sx_scale = (float)W / (float)ws;
    float fy0 = (float)ys * sy_scale;
    float fy1 = (float)(ys + 1) * sy_scale;
    float fx0 = (float)xs * sx_scale;
    float fx1 = (float)(xs + 1) * sx_scale;
    int y0 = max(0, (int)floor(fy0));
    int y1 = min(H, (int)ceil(fy1));
    int x0 = max(0, (int)floor(fx0));
    int x1 = min(W, (int)ceil(fx1));
    float refI = gray[(size_t)y0 * W + x0];
    float sI = 0.0f, sP = 0.0f, area = 0.0f;
    for (int sy = y0; sy < y1; ++sy) {
        float wy = fmin(fy1, (float)(sy + 1)) - fmax(fy0, (float)sy);
        if (wy <= 0.0f) continue;
        size_t row = (size_t)sy * W;
        for (int sx = x0; sx < x1; ++sx) {
            float wx = fmin(fx1, (float)(sx + 1)) - fmax(fx0, (float)sx);
            if (wx <= 0.0f) continue;
            float wt = wy * wx;
            area = area + wt;
            size_t pos = row + sx;
            sI = sI + (gray[pos] - refI) * wt;
            if (idx[pos] == frame_k) sP = sP + wt;
        }
    }
    float iv = refI + sI / area;
    float pv = sP / area;
    sub4[(size_t)ys * ws + xs] = (float4)(iv, pv, iv * pv, iv * iv);
}

__kernel void gfg_fgf_sub_ab(__global const float4* sub4, __global float2* ab,
                           const int hs, const int ws, const int rad, const float eps) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= ws || y >= hs) return;
    int k = 2 * rad + 1;
    float inv = 1.0f / (float)(k * k);
    float4 c = sub4[(size_t)y * ws + x];
    float refI = c.x;
    float refP = c.y;
    float sI = 0.0f, sP = 0.0f, sIP = 0.0f, sII = 0.0f;
    for (int dy = -rad; dy <= rad; ++dy) {
        size_t row = (size_t)reflect101_idx(y + dy, hs) * ws;
        for (int dx = -rad; dx <= rad; ++dx) {
            float4 v = sub4[row + reflect101_idx(x + dx, ws)];
            float dI = v.x - refI;
            float dP = v.y - refP;
            sI = sI + dI;
            sP = sP + dP;
            sIP = sIP + dI * dP;
            sII = sII + dI * dI;
        }
    }
    float mdI = sI * inv;
    float mdP = sP * inv;
    float cov = sIP * inv - mdI * mdP;
    float var = sII * inv - mdI * mdI;
    float a = cov / (var + eps);
    float b = (refP + mdP) - a * (refI + mdI);
    ab[(size_t)y * ws + x] = (float2)(a, b);
}

__kernel void gfg_fgf_sub_mean_ab(__global const float2* ab, __global float2* mab,
                                const int hs, const int ws, const int rad) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= ws || y >= hs) return;
    int k = 2 * rad + 1;
    float inv = 1.0f / (float)(k * k);
    float2 ref = ab[(size_t)y * ws + x];
    float2 s = (float2)(0.0f, 0.0f);
    for (int dy = -rad; dy <= rad; ++dy) {
        size_t row = (size_t)reflect101_idx(y + dy, hs) * ws;
        for (int dx = -rad; dx <= rad; ++dx) {
            s = s + (ab[row + reflect101_idx(x + dx, ws)] - ref);
        }
    }
    mab[(size_t)y * ws + x] = ref + s * inv;
}

static inline float2 sample_bilinear_f2(__global const float2* mab,
                                        const int hs, const int ws,
                                        const int H, const int W,
                                        const int x, const int y) {
    float fy = ((float)y + 0.5f) * ((float)hs / (float)H) - 0.5f;
    float fx = ((float)x + 0.5f) * ((float)ws / (float)W) - 0.5f;
    int y0 = (int)floor(fy);
    int x0 = (int)floor(fx);
    float dy = fy - (float)y0;
    float dx = fx - (float)x0;
    if (y0 < 0) { y0 = 0; dy = 0.0f; }
    if (x0 < 0) { x0 = 0; dx = 0.0f; }
    int y1 = min(y0 + 1, hs - 1);
    int x1 = min(x0 + 1, ws - 1);
    if (y0 >= hs - 1) { y0 = hs - 1; y1 = hs - 1; dy = 0.0f; }
    if (x0 >= ws - 1) { x0 = ws - 1; x1 = ws - 1; dx = 0.0f; }
    float2 c00 = mab[(size_t)y0 * ws + x0];
    float2 c01 = mab[(size_t)y0 * ws + x1];
    float2 c10 = mab[(size_t)y1 * ws + x0];
    float2 c11 = mab[(size_t)y1 * ws + x1];
    float2 r0 = c00 + (c01 - c00) * dx;
    float2 r1 = c10 + (c11 - c10) * dx;
    return r0 + (r1 - r0) * dy;
}

__kernel void gfg_fgf_up_q(__global const float2* mab, __global const float* gray,
                           __global float* q_out,
                           const int H, const int W, const int hs, const int ws) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    float2 ab = sample_bilinear_f2(mab, hs, ws, H, W, x, y);
    size_t pos = (size_t)y * W + x;
    q_out[pos] = ab.x * gray[pos] + ab.y;
}

__kernel void gfg_sobel_decision(__global const float* afm, __global float* val,
                                 __global float* sob_best, __global int* idx,
                                 const int H, const int W, const int frame_k,
                                 const float tie_eps, const int first_valid) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int ym = reflect101_idx(y - 1, H);
    int yp = reflect101_idx(y + 1, H);
    int xm = reflect101_idx(x - 1, W);
    int xp = reflect101_idx(x + 1, W);

    float a00 = afm[(size_t)ym * W + xm], a01 = afm[(size_t)ym * W + x], a02 = afm[(size_t)ym * W + xp];
    float a10 = afm[(size_t)y  * W + xm], a11 = afm[(size_t)y  * W + x], a12 = afm[(size_t)y  * W + xp];
    float a20 = afm[(size_t)yp * W + xm], a21 = afm[(size_t)yp * W + x], a22 = afm[(size_t)yp * W + xp];

    float gx = ((a02 - a00) + (a22 - a20)) + 2.0f * (a12 - a10);
    float gy = ((a20 + a22) + 2.0f * a21) - ((a00 + a02) + 2.0f * a01);
    float s = fabs(gx) + fabs(gy);

    size_t pos = (size_t)y * W + x;
    if (first_valid) {
        val[pos] = a11;
        sob_best[pos] = s;
        idx[pos] = frame_k;
    } else {
        float prev_v = val[pos];
        float diff = a11 - prev_v;
        if (diff > tie_eps || (fabs(diff) <= tie_eps && s > sob_best[pos])) {
            val[pos] = a11;
            sob_best[pos] = s;
            idx[pos] = frame_k;
        }
    }
}

__kernel void gfg_bincount(__global const int* idx, __global uint* wins, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    atomic_inc(&wins[idx[i]]);
}

#define DEFINE_UP_ACCUM(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const float2* mab, __global const float* gray, \
                   __global const SRC_T* src, __global float4* acc, \
                   const int H, const int W, const int hs, const int ws, const int C) { \
    int x = get_global_id(0); \
    int y = get_global_id(1); \
    if (x >= W || y >= H) return; \
    size_t pos = (size_t)y * W + x; \
    float2 ab = sample_bilinear_f2(mab, hs, ws, H, W, x, y); \
    float w = ab.x * gray[pos] + ab.y; \
    float c0, c1, c2; \
    if (C == 1) { \
        SRC_T raw0 = src[pos]; \
        c0 = c1 = c2 = (CONV_EXPR); \
    } else { \
        size_t b = pos * C; \
        SRC_T raw0 = src[b]; c0 = (CONV_EXPR); \
        raw0 = src[b + 1]; c1 = (CONV_EXPR); \
        raw0 = src[b + 2]; c2 = (CONV_EXPR); \
    } \
    float4 prev = acc[pos]; \
    acc[pos] = (float4)(prev.x + c0 * w, \
                        prev.y + c1 * w, \
                        prev.z + c2 * w, \
                        prev.w + w); \
}

DEFINE_UP_ACCUM(gfg_up_accum_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_UP_ACCUM(gfg_up_accum_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_UP_ACCUM(gfg_up_accum_f32, float, (raw0 / 255.0f))

__kernel void gfg_count_zero(__global const float4* acc, __global uint* zero_cnt, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    if (acc[i].w <= 1e-6f) atomic_inc(zero_cnt);
}

#define DEFINE_FILL_ZERO(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global const int* idx, \
                   __global float4* acc, const int n, const int C, const int frame_k) { \
    int i = get_global_id(0); \
    if (i >= n) return; \
    float4 prev = acc[i]; \
    if (prev.w <= 1e-6f && idx[i] == frame_k) { \
        float c0, c1, c2; \
        if (C == 1) { \
            SRC_T raw0 = src[i]; \
            c0 = c1 = c2 = (CONV_EXPR); \
        } else { \
            size_t b = (size_t)i * C; \
            SRC_T raw0 = src[b]; c0 = (CONV_EXPR); \
            raw0 = src[b + 1]; c1 = (CONV_EXPR); \
            raw0 = src[b + 2]; c2 = (CONV_EXPR); \
        } \
        acc[i] = (float4)(c0, c1, c2, 1.0f); \
    } \
}

DEFINE_FILL_ZERO(gfg_fill_zero_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_FILL_ZERO(gfg_fill_zero_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_FILL_ZERO(gfg_fill_zero_f32, float, (raw0 / 255.0f))

__kernel void gfg_normalize_u8(__global const float4* acc, __global uchar* out, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 v = acc[i];
    float3 rgb = (v.w > 1e-6f) ? (v.xyz / v.w) : v.xyz;
    rgb = clamp(rgb * 255.0f, 0.0f, 255.0f);
    size_t o = (size_t)i * 3;
    out[o]     = (uchar)rgb.x;
    out[o + 1] = (uchar)rgb.y;
    out[o + 2] = (uchar)rgb.z;
}

__kernel void gfg_normalize_f32(__global const float4* acc, __global float* out, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 v = acc[i];
    float3 rgb = (v.w > 1e-6f) ? (v.xyz / v.w) : v.xyz;
    rgb = clamp(rgb * 255.0f, 0.0f, 255.0f);
    size_t o = (size_t)i * 3;
    out[o]     = rgb.x;
    out[o + 1] = rgb.y;
    out[o + 2] = rgb.z;
}
'''


def _cdiv(value, divisor):
    return (value + divisor - 1) // divisor


def _round16(value):
    return ((value + 15) // 16) * 16


def _source_staging_bytes(frame, shape, img_resize):
    from core.source_frames import frame_has_mapped_pixels, frame_resolution_bytes

    resolution = frame_resolution_bytes(frame)
    if frame_has_mapped_pixels(frame):
        resolution = max(resolution, frame.nbytes)
    samples = math.prod(shape)
    if img_resize is not None:
        return resolution + frame.nbytes * 4 + samples * 12
    packed = 0 if isinstance(frame, np.ndarray) and frame.flags.c_contiguous else frame.nbytes
    itemsize = min(4, np.dtype(frame.dtype).itemsize)
    upload = frame.size * itemsize
    return resolution + packed + upload


def _frame_upload_bytes(frame, shape, img_resize):
    height, width = shape[:2]
    if img_resize is not None:
        return height * width * 3 * 4
    channels = 1 if len(frame.shape) == 2 else int(frame.shape[2])
    itemsize = min(4, np.dtype(frame.dtype).itemsize)
    return height * width * channels * itemsize


def _upload_kind(dtype):
    if dtype == np.uint8:
        return 'u8'
    if dtype == np.uint16:
        return 'u16'
    return 'f32'


_KERNEL_NAMES = (
    'gfg_chan_sum_u8', 'gfg_chan_sum_u16', 'gfg_chan_sum_f32',
    'gfg_extract_u8', 'gfg_extract_u16', 'gfg_extract_f32',
    'gfg_scharr_sum',
    'transpose_f1', 'box1_col_chunk', 'box1_col_sum',
    'gfg_ia_combine',
    'gfg_area_down_pair', 'gfg_area_down_mask',
    'gfg_fgf_sub_ab', 'gfg_fgf_sub_mean_ab', 'gfg_fgf_up_q',
    'gfg_sobel_decision',
    'gfg_bincount',
    'gfg_up_accum_u8', 'gfg_up_accum_u16', 'gfg_up_accum_f32',
    'gfg_count_zero',
    'gfg_fill_zero_u8', 'gfg_fill_zero_u16', 'gfg_fill_zero_f32',
    'gfg_normalize_u8', 'gfg_normalize_f32',
)
_cache_lock = Lock()
_PROGRAM_CACHE = None


def _device_cache_key(device):
    return (
        str(getattr(device, 'name', '')),
        str(getattr(device, 'vendor', '')),
        str(getattr(device, 'driver_version', '')),
        int(getattr(device, 'global_mem_size', 0)),
    )


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _build_cached_program, _get_shared_context

    ctx, q = _get_shared_context(cl, device)
    key = _device_cache_key(device)
    with _cache_lock:
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key and _PROGRAM_CACHE[1] is ctx:
            return _PROGRAM_CACHE[1:]
        prog = _build_cached_program(
            cl, device, ctx, _GFG_K,
            options=['-cl-fp32-correctly-rounded-divide-sqrt'],
        )
        kernels = {name: cl.Kernel(prog, name) for name in _KERNEL_NAMES}
        _PROGRAM_CACHE = (key, ctx, q, kernels)
        return ctx, q, kernels


def _invalidate_program_cache():
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _invalidate_shared_context

    with _cache_lock:
        _PROGRAM_CACHE = None
    _invalidate_shared_context()


class GpuGfg:
    """Device-resident OpenCL GFG-FGF fusion with single-frame streaming."""

    def __init__(self):
        from fusion_methods._openfocus_gpu import GpuError, _device, opencl_device_name
        try:
            import pyopencl as cl
            device = _device(cl)
            if device is None:
                raise GpuError('No available OpenCL GPU')
            if int(device.max_work_group_size) < 256 or int(device.local_mem_size) < 8192:
                raise GpuError('OpenCL GPU lacks required 256-thread workgroups or local memory')
            self.cl = cl
            self.device = device
            self.device_name = opencl_device_name(device)
            self.mf = cl.mem_flags
            self.ctx, self.q, self.k = _get_or_build_program(cl, device)
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU GFG-FGF initialization failed') from exc

    def _run(self, kernel, gsize, *args, lsize=None):
        g = gsize if isinstance(gsize, tuple) else (gsize,)
        l = (lsize,) if isinstance(lsize, int) else lsize
        kernel(self.q, g, l, *args)

    def _drain_queue(self):
        q = getattr(self, 'q', None)
        if q is not None:
            try:
                q.finish()
            except Exception:
                _invalidate_program_cache()

    def fuse(self, frames, img_resize, kernel_size, min_frame_focus,
             should_cancel=None, progress_callback=None):
        from fusion_methods._openfocus_gpu import GpuError, _device_lock

        class CallbackFailure(BaseException):
            pass

        def progress(index, total):
            try:
                progress_callback(index, total)
            except BaseException as exc:
                raise CallbackFailure() from exc

        while not _device_lock.acquire(timeout=0.1):
            _check_cancel(should_cancel)
        try:
            try:
                return self._fuse_locked(
                    frames, img_resize, kernel_size, min_frame_focus, should_cancel,
                    progress if progress_callback is not None else None)
            except CallbackFailure as exc:
                self._drain_queue()
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                _invalidate_program_cache()
                self._drain_queue()
                raise GpuError('OpenCL GPU GFG-FGF execution failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.k = None
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, img_resize, kernel_size, min_frame_focus,
                     should_cancel, progress_callback):
        from dataclasses import replace
        from core import resource_policy
        from core.source_frames import resolve_frame, without_source_cache_growth
        from fusion_methods._openfocus_gpu import GpuError, _free_vram

        if not frames:
            raise ValueError('Input stack is empty')
        mean_ksize = int(kernel_size)
        if mean_ksize < 1 or (mean_ksize > 1 and mean_ksize % 2 == 0):
            raise GpuError('GFG-FGF GPU requires kernel_size == 1 or positive odd integer')
        first_shape = frames[0].shape
        if img_resize is not None:
            height, width = int(img_resize[1]), int(img_resize[0])
        else:
            height, width = int(first_shape[0]), int(first_shape[1])
        h_sub = height // _GF_SUB
        w_sub = width // _GF_SUB
        if h_sub < 1 or w_sub < 1:
            raise GpuError('GFG-FGF GPU requires images at least 4x4')

        shape = (height, width, 3)
        n = height * width
        n_sub = h_sub * w_sub
        count = len(frames)
        total_steps = count * 2
        if progress_callback is not None:
            progress_callback(0, total_steps)
        _check_cancel(should_cancel)

        out_dtype = fuse_output_dtype(frames)
        is_u8_out = out_dtype == np.uint8
        out_bytes = n * 3 * (1 if is_u8_out else 4)

        snapshot = resource_policy.probe_memory()
        host_budget = resource_policy.ram_ceiling(snapshot)
        source_peak = 0
        src_bytes = 0
        for frame in frames:
            _check_cancel(should_cancel)
            source_peak = max(source_peak, _source_staging_bytes(frame, shape, img_resize))
            src_bytes = max(src_bytes, _frame_upload_bytes(frame, shape, img_resize))
        staging = max(source_peak, out_bytes * 2) + 16 * 1024 * 1024
        if staging > host_budget:
            raise GpuError(f'GFG-FGF GPU staging exceeds the host memory budget '
                           f'({staging >> 20} MiB needed, {host_budget >> 20} MiB available)')

        need = (36 * n
                + 32 * n_sub
                + max(src_bytes, out_bytes)
                + count * 4
                + 16 * 1024 * 1024)
        limit = int(self.device.global_mem_size) // 2
        if self.device.host_unified_memory:
            limit = min(limit, host_budget - staging)
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError('GFG-FGF GPU free memory is unknown')
            snapshot = replace(snapshot, free_vram=memory[0], total_vram=memory[1])
            limit = min(limit, resource_policy.vram_budget(snapshot))
        single = int(self.device.max_mem_alloc_size)
        largest = max(16 * n, src_bytes, out_bytes)
        if need > limit or largest > single:
            raise GpuError('GFG-FGF GPU workspace exceeds the device budget')

        resize = (img_resize is not None and (int(first_shape[1]), int(first_shape[0])) != img_resize)

        def prepare_upload(index):
            _check_cancel(should_cancel)
            with without_source_cache_growth():
                raw = resolve_frame(frames[index], should_cancel=should_cancel)
            _check_cancel(should_cancel)
            if raw.dtype.kind == 'f':
                if not np.isfinite(raw).all():
                    raise GpuError('Non-finite floating-point frame uses CPU fallback')
                mn, mx = float(np.min(raw)), float(np.max(raw))
                if mn < -255.0 or mx > 255.0:
                    raise GpuError('Unbounded floating-point frame uses CPU fallback')
            if resize:
                image = _ensure_color_image(np.ascontiguousarray(raw))
                del raw
                if image.dtype == np.uint16:
                    image = image.astype(np.float32) * (255.0 / 65535.0)
                elif image.dtype == np.float64:
                    image = image.astype(np.float32)
                host_src = np.ascontiguousarray(cv2.resize(image, img_resize))
                del image
                return host_src, 3, _upload_kind(host_src.dtype)
            if raw.ndim == 2:
                channels = 1
            elif raw.ndim == 3 and raw.shape[2] in (3, 4):
                channels = int(raw.shape[2])
            else:
                raise ValueError(f'Unsupported image shape: {raw.shape}')
            if raw.shape[:2] != (height, width):
                raise ValueError('GFG-FGF requires frames with matching shapes')
            dtype = np.float32 if raw.dtype == np.float64 else raw.dtype
            if dtype not in (np.uint8, np.uint16, np.float32):
                raise GpuError(f'Unsupported GPU pixel dtype: {raw.dtype}')
            host_src = np.ascontiguousarray(raw, dtype=dtype)
            del raw
            return host_src, channels, _upload_kind(dtype)

        src_buf = self.cl.Buffer(self.ctx, self.mf.READ_ONLY, max(1, src_bytes))
        gray_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        ia_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        t1_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        afm_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        val_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        sob_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        idx_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        sub4_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n_sub))
        ab_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 8 * n_sub))
        mab_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 8 * n_sub))

        red_groups = max(1, min(256, _cdiv(n, 256)))
        wg4_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, red_groups * 16)

        in_flight = False
        try:
            # Frame 0 upload + dominant channel selection (same argmax over channel sums).
            host0, ch0, kind0 = prepare_upload(0)
            in_flight = True
            self.cl.enqueue_copy(self.q, src_buf, host0, is_blocking=False)
            self._run(self.k[f'gfg_chan_sum_{kind0}'], red_groups * 256,
                      src_buf, wg4_buf, np.int32(n), np.int32(ch0), lsize=256)
            wg4_host = np.empty((red_groups, 4), dtype=np.float32)
            self.cl.enqueue_copy(self.q, wg4_host, wg4_buf, is_blocking=True)
            in_flight = False
            channel_idx = int(np.argmax(wg4_host[:, :3].sum(axis=0, dtype=np.float64)))
            del wg4_buf, wg4_host

            scale = float(min_frame_focus)
            focus_vals = None
            max_focus = 1.0
            if scale > 0.0:
                inner_n = max(1, (height - 2) * (width - 2))
                sc_groups = max(1, min(256, _cdiv(inner_n, 256)))
                sc_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, sc_groups * 4)
                sc_host = np.empty(sc_groups, dtype=np.float32)
                focus_vals = np.zeros(count, dtype=np.float32)
                for index in range(count):
                    h_src = None
                    if index == 0:
                        cur_ch, cur_kind = ch0, kind0
                    else:
                        h_src, cur_ch, cur_kind = prepare_upload(index)
                        in_flight = True
                        self.cl.enqueue_copy(self.q, src_buf, h_src, is_blocking=False)
                    in_flight = True
                    self._run(self.k[f'gfg_extract_{cur_kind}'], n,
                              src_buf, gray_buf, np.int32(n), np.int32(cur_ch), np.int32(channel_idx))
                    if height > 2 and width > 2:
                        self._run(self.k['gfg_scharr_sum'], sc_groups * 256,
                                  gray_buf, sc_buf, np.int32(height), np.int32(width), lsize=256)
                        self.cl.enqueue_copy(self.q, sc_host, sc_buf, is_blocking=True)
                        focus_vals[index] = float(sc_host.sum(dtype=np.float64)) / ((height - 2) * (width - 2))
                    else:
                        self.q.finish()
                    in_flight = False
                    if h_src is not None:
                        del h_src
                del sc_buf, sc_host
                max_focus = float(np.max(focus_vals)) or 1.0

            grid_hw = (_round16(width), _round16(height))
            grid_wh = (_round16(height), _round16(width))
            local_16 = (16, 16)
            rad_mean = mean_ksize // 2
            r_sub = max(1, _GF_RADIUS // _GF_SUB)

            first_valid = True
            for index in range(count):
                _check_cancel(should_cancel)
                if scale > 0.0 and focus_vals[index] < scale * max_focus:
                    if progress_callback is not None:
                        progress_callback(index + 1, total_steps)
                    continue

                h_src = None
                if index == 0 and host0 is not None:
                    # Re-use frame 0 already resident in src_buf when scale <= 0
                    if scale > 0.0:
                        in_flight = True
                        self.cl.enqueue_copy(self.q, src_buf, host0, is_blocking=False)
                    cur_ch, cur_kind = ch0, kind0
                else:
                    if host0 is not None:
                        del host0
                        host0 = None
                    h_src, cur_ch, cur_kind = prepare_upload(index)
                    in_flight = True
                    self.cl.enqueue_copy(self.q, src_buf, h_src, is_blocking=False)

                in_flight = True
                self._run(self.k[f'gfg_extract_{cur_kind}'], n,
                          src_buf, gray_buf, np.int32(n), np.int32(cur_ch), np.int32(channel_idx))

                if rad_mean > 0:
                    self._run(self.k['transpose_f1'], grid_hw, gray_buf, t1_buf,
                              np.int32(height), np.int32(width), lsize=local_16)
                    self._run(self.k['box1_col_chunk'], (height, _cdiv(width, _CHUNK)),
                              t1_buf, ia_buf, np.int32(width), np.int32(height), np.int32(rad_mean))
                    self._run(self.k['transpose_f1'], grid_wh, ia_buf, t1_buf,
                              np.int32(width), np.int32(height), lsize=local_16)
                    self._run(self.k['box1_col_sum'], (width, _cdiv(height, _CHUNK)),
                              t1_buf, afm_buf, np.int32(height), np.int32(width), np.int32(rad_mean))
                    blur_buf = afm_buf
                else:
                    blur_buf = gray_buf

                self._run(self.k['gfg_ia_combine'], (width, height),
                          gray_buf, blur_buf, ia_buf,
                          np.int32(height), np.int32(width),
                          np.float32(_GFG_THRESHOLD), np.int32(1 if rad_mean > 0 else 0))

                self._run(self.k['gfg_area_down_pair'], (w_sub, h_sub),
                          gray_buf, ia_buf, sub4_buf,
                          np.int32(height), np.int32(width), np.int32(h_sub), np.int32(w_sub))
                self._run(self.k['gfg_fgf_sub_ab'], (w_sub, h_sub),
                          sub4_buf, ab_buf,
                          np.int32(h_sub), np.int32(w_sub), np.int32(r_sub), np.float32(_GF_EPS))
                self._run(self.k['gfg_fgf_sub_mean_ab'], (w_sub, h_sub),
                          ab_buf, mab_buf,
                          np.int32(h_sub), np.int32(w_sub), np.int32(r_sub))
                self._run(self.k['gfg_fgf_up_q'], (width, height),
                          mab_buf, gray_buf, afm_buf,
                          np.int32(height), np.int32(width), np.int32(h_sub), np.int32(w_sub))
                self._run(self.k['gfg_sobel_decision'], (width, height),
                          afm_buf, val_buf, sob_buf, idx_buf,
                          np.int32(height), np.int32(width), np.int32(index),
                          np.float32(_TIE_EPS), np.int32(1 if first_valid else 0))
                first_valid = False
                self.q.finish()
                in_flight = False
                if index == 0 and host0 is not None:
                    del host0
                    host0 = None
                if h_src is not None:
                    del h_src
                if progress_callback is not None:
                    progress_callback(index + 1, total_steps)

            if host0 is not None:
                del host0
                host0 = None

            if first_valid:
                # All frames filtered by min_frame_focus: degrade to frame 0.
                in_flight = True
                self.cl.enqueue_fill_buffer(self.q, idx_buf, np.int32(0), 0, 4 * n)

            del val_buf, sob_buf, ia_buf, t1_buf, afm_buf

            wins_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, count * 4))
            in_flight = True
            self.cl.enqueue_fill_buffer(self.q, wins_buf, np.uint32(0), 0, count * 4)
            self._run(self.k['gfg_bincount'], n, idx_buf, wins_buf, np.int32(n))
            wins = np.empty(count, dtype=np.uint32)
            self.cl.enqueue_copy(self.q, wins, wins_buf, is_blocking=True)
            in_flight = False
            del wins_buf

            # Pass 3: Fast guided filter refinement on active decision masks + fused RGB accumulation.
            acc_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            in_flight = True
            self.cl.enqueue_fill_buffer(self.q, acc_buf, np.float32(0.0), 0, 16 * n)

            for index in range(count):
                _check_cancel(should_cancel)
                if not wins[index]:
                    if progress_callback is not None:
                        progress_callback(count + index + 1, total_steps)
                    continue
                h_src, cur_ch, cur_kind = prepare_upload(index)
                in_flight = True
                self.cl.enqueue_copy(self.q, src_buf, h_src, is_blocking=False)

                self._run(self.k[f'gfg_extract_{cur_kind}'], n,
                          src_buf, gray_buf, np.int32(n), np.int32(cur_ch), np.int32(channel_idx))
                self._run(self.k['gfg_area_down_mask'], (w_sub, h_sub),
                          gray_buf, idx_buf, sub4_buf,
                          np.int32(height), np.int32(width),
                          np.int32(h_sub), np.int32(w_sub), np.int32(index))
                self._run(self.k['gfg_fgf_sub_ab'], (w_sub, h_sub),
                          sub4_buf, ab_buf,
                          np.int32(h_sub), np.int32(w_sub), np.int32(r_sub), np.float32(_GF_EPS))
                self._run(self.k['gfg_fgf_sub_mean_ab'], (w_sub, h_sub),
                          ab_buf, mab_buf,
                          np.int32(h_sub), np.int32(w_sub), np.int32(r_sub))
                self._run(self.k[f'gfg_up_accum_{cur_kind}'], (width, height),
                          mab_buf, gray_buf, src_buf, acc_buf,
                          np.int32(height), np.int32(width),
                          np.int32(h_sub), np.int32(w_sub), np.int32(cur_ch))
                self.q.finish()
                in_flight = False
                del h_src
                if progress_callback is not None:
                    progress_callback(count + index + 1, total_steps)

            del gray_buf, sub4_buf, ab_buf, mab_buf

            # Zero-weight fallback: only resolve frames referenced by sum_w <= 1e-6 pixels.
            zcnt_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, 4)
            in_flight = True
            self.cl.enqueue_fill_buffer(self.q, zcnt_buf, np.uint32(0), 0, 4)
            self._run(self.k['gfg_count_zero'], n, acc_buf, zcnt_buf, np.int32(n))
            zcnt = np.empty(1, dtype=np.uint32)
            self.cl.enqueue_copy(self.q, zcnt, zcnt_buf, is_blocking=True)
            in_flight = False
            del zcnt_buf
            if int(zcnt[0]) > 0:
                host_acc = np.empty((height, width, 4), dtype=np.float32)
                host_idx = np.empty((height, width), dtype=np.int32)
                in_flight = True
                self.cl.enqueue_copy(self.q, host_acc, acc_buf, is_blocking=False)
                self.cl.enqueue_copy(self.q, host_idx, idx_buf, is_blocking=True)
                in_flight = False
                zero_winners = np.unique(host_idx[host_acc[:, :, 3] <= 1e-6])
                del host_acc, host_idx
                for k in zero_winners:
                    _check_cancel(should_cancel)
                    h_src, cur_ch, cur_kind = prepare_upload(int(k))
                    in_flight = True
                    self.cl.enqueue_copy(self.q, src_buf, h_src, is_blocking=True)
                    del h_src
                    self._run(self.k[f'gfg_fill_zero_{cur_kind}'], n,
                              src_buf, idx_buf, acc_buf,
                              np.int32(n), np.int32(cur_ch), np.int32(int(k)))
                self.q.finish()
                in_flight = False

            del src_buf, idx_buf

            _check_cancel(should_cancel)
            out = np.empty(shape, dtype=out_dtype)
            dst_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, out_bytes)
            norm_kern = self.k['gfg_normalize_u8' if is_u8_out else 'gfg_normalize_f32']
            in_flight = True
            self._run(norm_kern, n, acc_buf, dst_buf, np.int32(n))
            self.cl.enqueue_copy(self.q, out, dst_buf, is_blocking=True)
            in_flight = False
            if progress_callback is not None:
                progress_callback(total_steps, total_steps)
            _check_cancel(should_cancel)
            return out
        except BaseException:
            if in_flight:
                self._drain_queue()
            raise
