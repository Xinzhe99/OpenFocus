"""OpenCL GPU fusion for GFG-FGF (opt-in via use_gpu=True on gfgfgf_impl).

Full-frame streaming OpenCL 1.2 implementation using:
- Fused local-memory channel extraction + mean blur + GFG gradient -> IA focus map
  (zero-transpose shared-memory scan fallback for large kernel_size > 17).
- Fast exact 4x4 area downsampling when dimensions are multiples of 4.
- Fused bilinear upsample + Sobel 3x3 tie-broken streaming GfgDecision in local memory
  (eliminating the full-frame afm intermediate buffer).
- GPU active bounding-box extraction to restrict Pass 3 strictly to each frame's
  non-zero guided-filter support, with zero-ab early skip and direct src_buf reads
  (eliminating Pass 3 gfg_extract).
- Bounded background frame prefetch and ping-pong async H2D DMA transfers.
"""
from __future__ import annotations

import math
from threading import Lock

import cv2
import numpy as np

from core.frame_storage import async_upload_view
from fusion_methods.dct import _ensure_color_image
from fusion_methods.openfocus_ab import _check_cancel
from utils.image_utils import fuse_output_dtype

_GFG_THRESHOLD = 0.005
_GF_RADIUS = 5
_GF_EPS = 0.3
_GF_SUB = 4
_TIE_EPS = 1e-6
_CHUNK = 64
_LOCAL_IA_MAX_RAD = 8
_READBACK_BUFFER_BYTES = 8 * 1024 * 1024

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
                   const int n, const int C, const int bgr_rev) { \
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
            SRC_T raw0 = src[b + (bgr_rev ? 2 : 0)]; float c0 = (CONV_EXPR); \
            raw0 = src[b + 1]; float c1 = (CONV_EXPR); \
            raw0 = src[b + (bgr_rev ? 0 : 2)]; float c2 = (CONV_EXPR); \
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
                   const int n, const int C, const int ch, const int bgr_rev) { \
    int i = get_global_id(0); \
    if (i >= n) return; \
    int sch = (C > 1 && bgr_rev && ch < 3) ? (2 - ch) : ch; \
    SRC_T raw0 = (C == 1) ? src[i] : src[(size_t)i * C + sch]; \
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

// Fused channel extraction + (2*rad+1)x(2*rad+1) box blur + 3x3 GFG gradient -> IA
// for rad <= 8 (kernel_size <= 17, including default kernel_size == 7).
// Uses reflect101 for box blur and exact reflect for 3x3 GFG gradient.
#define DEFINE_EXTRACT_IA_LOCAL(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global float* gray, __global float* ia, \
                   const int H, const int W, const int C, const int ch, const int bgr_rev, \
                   const int rad, const float threshold) { \
    __local float tile[32][32]; \
    __local float hsum[32][16]; \
    int lx = get_local_id(0); \
    int ly = get_local_id(1); \
    int tid = ly * 16 + lx; \
    int gx0 = get_group_id(0) * 16; \
    int gy0 = get_group_id(1) * 16; \
    int sch = (C > 1 && bgr_rev && ch < 3) ? (2 - ch) : ch; \
    int halo = (rad > 0) ? rad : 1; \
    int dim = 16 + 2 * halo; \
    int total = dim * dim; \
    for (int i = tid; i < total; i += 256) { \
        int dy = i / dim; \
        int dx = i - dy * dim; \
        int yy = (rad > 0) ? reflect101_idx(gy0 + dy - halo, H) : reflect_idx(gy0 + dy - halo, H); \
        int xx = (rad > 0) ? reflect101_idx(gx0 + dx - halo, W) : reflect_idx(gx0 + dx - halo, W); \
        size_t pos = (size_t)yy * W + xx; \
        SRC_T raw0 = (C == 1) ? src[pos] : src[pos * C + sch]; \
        tile[dy][dx] = (CONV_EXPR); \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int k = 2 * rad + 1; \
    if (rad > 0) { \
        int htotal = dim * 16; \
        for (int i = tid; i < htotal; i += 256) { \
            int hy = i >> 4; \
            int hx = i & 15; \
            float s = 0.0f; \
            for (int d = 0; d < k; ++d) { \
                s = s + tile[hy][hx + d]; \
            } \
            hsum[hy][hx] = s; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    int x = gx0 + lx; \
    int y = gy0 + ly; \
    if (x >= W || y >= H) return; \
    int ty = ly + halo; \
    int tx = lx + halo; \
    float c = tile[ty][tx]; \
    size_t pos = (size_t)y * W + x; \
    gray[pos] = c; \
    const float t1 = 1.0f / 16.0f; \
    const float t2 = 2.0f / 16.0f; \
    const float t4 = 4.0f / 16.0f; \
    float g00, g01, g02, g10, g12, g20, g21, g22; \
    if (rad == 0 || (x > 0 && x + 1 < W && y > 0 && y + 1 < H)) { \
        g00 = tile[ty - 1][tx - 1]; g01 = tile[ty - 1][tx]; g02 = tile[ty - 1][tx + 1]; \
        g10 = tile[ty][tx - 1];                             g12 = tile[ty][tx + 1]; \
        g20 = tile[ty + 1][tx - 1]; g21 = tile[ty + 1][tx]; g22 = tile[ty + 1][tx + 1]; \
    } else { \
        int ym = reflect_idx(y - 1, H); \
        int yp = reflect_idx(y + 1, H); \
        int xm = reflect_idx(x - 1, W); \
        int xp = reflect_idx(x + 1, W); \
        size_t p00 = (size_t)ym * W + xm; SRC_T raw0 = (C == 1) ? src[p00] : src[p00 * C + sch]; g00 = (CONV_EXPR); \
        size_t p01 = (size_t)ym * W + x;  raw0 = (C == 1) ? src[p01] : src[p01 * C + sch]; g01 = (CONV_EXPR); \
        size_t p02 = (size_t)ym * W + xp; raw0 = (C == 1) ? src[p02] : src[p02 * C + sch]; g02 = (CONV_EXPR); \
        size_t p10 = (size_t)y  * W + xm; raw0 = (C == 1) ? src[p10] : src[p10 * C + sch]; g10 = (CONV_EXPR); \
        size_t p12 = (size_t)y  * W + xp; raw0 = (C == 1) ? src[p12] : src[p12 * C + sch]; g12 = (CONV_EXPR); \
        size_t p20 = (size_t)yp * W + xm; raw0 = (C == 1) ? src[p20] : src[p20 * C + sch]; g20 = (CONV_EXPR); \
        size_t p21 = (size_t)yp * W + x;  raw0 = (C == 1) ? src[p21] : src[p21 * C + sch]; g21 = (CONV_EXPR); \
        size_t p22 = (size_t)yp * W + xp; raw0 = (C == 1) ? src[p22] : src[p22 * C + sch]; g22 = (CONV_EXPR); \
    } \
    float p = (g00 * t1 + g01 * t2) + g02 * t1; \
    p = ((p + g10 * t2) + c * t4) + g12 * t2; \
    p = ((p + g20 * t1) + g21 * t2) + g22 * t1; \
    float g = fabs(g12 - p) + fabs(g10 - p); \
    g = (g + fabs(g21 - p)) + fabs(g01 - p); \
    g = g * 0.5f; \
    if (g >= threshold) { \
        ia[pos] = g; \
    } else if (rad > 0) { \
        float bs = 0.0f; \
        for (int d = 0; d < k; ++d) { \
            bs = bs + hsum[ly + d][lx]; \
        } \
        float blur = bs / (float)(k * k); \
        ia[pos] = fabs(c - blur); \
    } else { \
        ia[pos] = 0.0f; \
    } \
}

DEFINE_EXTRACT_IA_LOCAL(gfg_extract_ia_local_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_EXTRACT_IA_LOCAL(gfg_extract_ia_local_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_EXTRACT_IA_LOCAL(gfg_extract_ia_local_f32, float, (raw0 / 255.0f))

// Zero-transpose shared-memory horizontal box sum for rad > 8 (rad <= 64).
__kernel void box1_h_scan(__global const float* gray, __global float* dst,
                          const int H, const int W, const int rad) {
    __local float scan[256];
    __local float blk[128];
    int lx = get_local_id(0);
    int tile_x0 = get_group_id(0) * 128;
    int y = get_group_id(1);
    if (y >= H) return;
    int need = 128 + 2 * rad;
    size_t row = (size_t)y * W;
    for (int i = lx; i < 256; i += 128) {
        if (i < need) {
            int gx = reflect101_idx(tile_x0 + i - rad, W);
            scan[i] = gray[row + gx];
        } else {
            scan[i] = 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float v0 = scan[2 * lx];
    float v1 = scan[2 * lx + 1];
    float psum = v0 + v1;
    blk[lx] = psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 1; off < 128; off <<= 1) {
        float add = (lx >= off) ? blk[lx - off] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        blk[lx] = blk[lx] + add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float pref = (lx > 0) ? blk[lx - 1] : 0.0f;
    scan[2 * lx]     = pref + v0;
    scan[2 * lx + 1] = pref + psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    int col = tile_x0 + lx;
    if (col < W) {
        float hi = scan[lx + 2 * rad];
        float lo = (lx > 0) ? scan[lx - 1] : 0.0f;
        dst[row + col] = hi - lo;
    }
}

__kernel void box1_v_ia_combine(__global const float* gray, __global const float* hsum_buf,
                                __global float* ia, const int H, const int W,
                                const int rad, const float threshold) {
    int x = get_global_id(0);
    int y0 = get_global_id(1) * CHUNK;
    if (x >= W || y0 >= H) return;
    int y1 = min(y0 + CHUNK, H);
    float inv = 1.0f / (float)((2 * rad + 1) * (2 * rad + 1));
    float s = 0.0f;
    for (int d = -rad; d <= rad; ++d) {
        int yy = reflect101_idx(y0 + d, H);
        s = s + hsum_buf[(size_t)yy * W + x];
    }
    const float t1 = 1.0f / 16.0f;
    const float t2 = 2.0f / 16.0f;
    const float t4 = 4.0f / 16.0f;
    int xm = reflect_idx(x - 1, W);
    int xp = reflect_idx(x + 1, W);
    for (int y = y0; y < y1; ++y) {
        if (y > y0) {
            int yr = reflect101_idx(y + rad, H);
            int yl = reflect101_idx(y - rad - 1, H);
            s = (s + hsum_buf[(size_t)yr * W + x]) - hsum_buf[(size_t)yl * W + x];
        }
        int ym = reflect_idx(y - 1, H);
        int yp = reflect_idx(y + 1, H);
        size_t rm = (size_t)ym * W;
        size_t r0 = (size_t)y * W;
        size_t rp = (size_t)yp * W;
        float c = gray[r0 + x];
        float g01 = gray[rm + x], g10 = gray[r0 + xm], g12 = gray[r0 + xp], g21 = gray[rp + x];
        float p = (gray[rm + xm] * t1 + g01 * t2) + gray[rm + xp] * t1;
        p = ((p + g10 * t2) + c * t4) + g12 * t2;
        p = ((p + gray[rp + xm] * t1) + g21 * t2) + gray[rp + xp] * t1;
        float g = fabs(g12 - p) + fabs(g10 - p);
        g = (g + fabs(g21 - p)) + fabs(g01 - p);
        g = g * 0.5f;
        ia[r0 + x] = (g >= threshold) ? g : fabs(c - s * inv);
    }
}

__kernel void gfg_area4_down_pair(__global const float* gray, __global const float* pbuf,
                                  __global float4* sub4,
                                  const int H, const int W, const int hs, const int ws) {
    int xs = get_global_id(0);
    int ys = get_global_id(1);
    if (xs >= ws || ys >= hs) return;
    int y0 = ys * 4;
    int x0 = xs * 4;
    float sI = 0.0f, sP = 0.0f;
    for (int dy = 0; dy < 4; ++dy) {
        size_t row = (size_t)(y0 + dy) * W + x0;
        sI = sI + ((gray[row] + gray[row + 1]) + (gray[row + 2] + gray[row + 3]));
        sP = sP + ((pbuf[row] + pbuf[row + 1]) + (pbuf[row + 2] + pbuf[row + 3]));
    }
    float iv = sI * (1.0f / 16.0f);
    float pv = sP * (1.0f / 16.0f);
    sub4[(size_t)ys * ws + xs] = (float4)(iv, pv, iv * pv, iv * iv);
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

#define DEFINE_AREA4_MASK_ROI(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global const int* idx, \
                   __global float4* sub4, \
                   const int H, const int W, const int C, const int ch, const int bgr_rev, \
                   const int sy0, const int sx0, const int sh, const int sw, const int frame_k) { \
    int rx = get_global_id(0); \
    int ry = get_global_id(1); \
    if (rx >= sw || ry >= sh) return; \
    int y0 = (sy0 + ry) * 4; \
    int x0 = (sx0 + rx) * 4; \
    int sch = (C > 1 && bgr_rev && ch < 3) ? (2 - ch) : ch; \
    float sI = 0.0f; \
    int cnt = 0; \
    for (int dy = 0; dy < 4; ++dy) { \
        size_t row = (size_t)(y0 + dy) * W + x0; \
        for (int dx = 0; dx < 4; ++dx) { \
            size_t p = row + dx; \
            SRC_T raw0 = (C == 1) ? src[p] : src[p * C + sch]; \
            sI = sI + (CONV_EXPR); \
            if (idx[p] == frame_k) ++cnt; \
        } \
    } \
    float iv = sI * (1.0f / 16.0f); \
    float pv = (float)cnt * (1.0f / 16.0f); \
    sub4[(size_t)ry * sw + rx] = (float4)(iv, pv, iv * pv, iv * iv); \
}

DEFINE_AREA4_MASK_ROI(gfg_area4_mask_roi_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_AREA4_MASK_ROI(gfg_area4_mask_roi_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_AREA4_MASK_ROI(gfg_area4_mask_roi_f32, float, (raw0 / 255.0f))

#define DEFINE_AREA_MASK_ROI(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global const int* idx, \
                   __global float4* sub4, \
                   const int H, const int W, const int C, const int ch, const int bgr_rev, \
                   const int hs, const int ws, \
                   const int sy0, const int sx0, const int sh, const int sw, const int frame_k) { \
    int rx = get_global_id(0); \
    int ry = get_global_id(1); \
    if (rx >= sw || ry >= sh) return; \
    int xs = sx0 + rx; \
    int ys = sy0 + ry; \
    int sch = (C > 1 && bgr_rev && ch < 3) ? (2 - ch) : ch; \
    float sy_scale = (float)H / (float)hs; \
    float sx_scale = (float)W / (float)ws; \
    float fy0 = (float)ys * sy_scale; \
    float fy1 = (float)(ys + 1) * sy_scale; \
    float fx0 = (float)xs * sx_scale; \
    float fx1 = (float)(xs + 1) * sx_scale; \
    int y0 = max(0, (int)floor(fy0)); \
    int y1 = min(H, (int)ceil(fy1)); \
    int x0 = max(0, (int)floor(fx0)); \
    int x1 = min(W, (int)ceil(fx1)); \
    size_t refp = (size_t)y0 * W + x0; \
    SRC_T raw0 = (C == 1) ? src[refp] : src[refp * C + sch]; \
    float refI = (CONV_EXPR); \
    float sI = 0.0f, sP = 0.0f, area = 0.0f; \
    for (int sy = y0; sy < y1; ++sy) { \
        float wy = fmin(fy1, (float)(sy + 1)) - fmax(fy0, (float)sy); \
        if (wy <= 0.0f) continue; \
        size_t row = (size_t)sy * W; \
        for (int sx = x0; sx < x1; ++sx) { \
            float wx = fmin(fx1, (float)(sx + 1)) - fmax(fx0, (float)sx); \
            if (wx <= 0.0f) continue; \
            float wt = wy * wx; \
            area = area + wt; \
            size_t pos = row + sx; \
            raw0 = (C == 1) ? src[pos] : src[pos * C + sch]; \
            sI = sI + ((CONV_EXPR) - refI) * wt; \
            if (idx[pos] == frame_k) sP = sP + wt; \
        } \
    } \
    float iv = refI + sI / area; \
    float pv = sP / area; \
    sub4[(size_t)ry * sw + rx] = (float4)(iv, pv, iv * pv, iv * iv); \
}

DEFINE_AREA_MASK_ROI(gfg_area_mask_roi_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_AREA_MASK_ROI(gfg_area_mask_roi_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_AREA_MASK_ROI(gfg_area_mask_roi_f32, float, (raw0 / 255.0f))

__kernel void gfg_fgf_sub_ab_roi(__global const float4* sub4, __global float2* ab,
                                 const int hs, const int ws,
                                 const int sub_y0, const int sub_x0, const int sub_w,
                                 const int ay0, const int ax0, const int ah, const int aw,
                                 const int rad, const float eps) {
    int rx = get_global_id(0);
    int ry = get_global_id(1);
    if (rx >= aw || ry >= ah) return;
    int x = ax0 + rx;
    int y = ay0 + ry;
    int k = 2 * rad + 1;
    float inv = 1.0f / (float)(k * k);
    float4 c = sub4[(size_t)(y - sub_y0) * sub_w + (x - sub_x0)];
    float refI = c.x;
    float refP = c.y;
    float sI = 0.0f, sP = 0.0f, sIP = 0.0f, sII = 0.0f, sumP_raw = 0.0f;
    for (int dy = -rad; dy <= rad; ++dy) {
        size_t row = (size_t)(reflect101_idx(y + dy, hs) - sub_y0) * sub_w;
        for (int dx = -rad; dx <= rad; ++dx) {
            float4 v = sub4[row + (reflect101_idx(x + dx, ws) - sub_x0)];
            sumP_raw = sumP_raw + v.y;
            float dI = v.x - refI;
            float dP = v.y - refP;
            sI = sI + dI;
            sP = sP + dP;
            sIP = sIP + dI * dP;
            sII = sII + dI * dI;
        }
    }
    if (sumP_raw == 0.0f) {
        ab[(size_t)ry * aw + rx] = (float2)(0.0f, 0.0f);
        return;
    }
    float mdI = sI * inv;
    float mdP = sP * inv;
    float cov = sIP * inv - mdI * mdP;
    float var = sII * inv - mdI * mdI;
    float a = cov / (var + eps);
    float b = (refP + mdP) - a * (refI + mdI);
    ab[(size_t)ry * aw + rx] = (float2)(a, b);
}

__kernel void gfg_fgf_sub_mean_ab_roi(__global const float2* ab, __global float2* mab,
                                      const int hs, const int ws,
                                      const int ay0, const int ay1, const int ax0, const int ax1, const int aw,
                                      const int my0, const int mx0, const int mh, const int mw,
                                      const int rad) {
    int rx = get_global_id(0);
    int ry = get_global_id(1);
    if (rx >= mw || ry >= mh) return;
    int x = mx0 + rx;
    int y = my0 + ry;
    int k = 2 * rad + 1;
    float inv = 1.0f / (float)(k * k);
    float2 ref = (y >= ay0 && y < ay1 && x >= ax0 && x < ax1)
                 ? ab[(size_t)(y - ay0) * aw + (x - ax0)]
                 : (float2)(0.0f, 0.0f);
    float2 s = (float2)(0.0f, 0.0f);
    for (int dy = -rad; dy <= rad; ++dy) {
        int yy = reflect101_idx(y + dy, hs);
        int row_ok = (yy >= ay0 && yy < ay1);
        size_t row = row_ok ? (size_t)(yy - ay0) * aw : 0;
        for (int dx = -rad; dx <= rad; ++dx) {
            int xx = reflect101_idx(x + dx, ws);
            float2 v = (row_ok && xx >= ax0 && xx < ax1)
                       ? ab[row + (xx - ax0)]
                       : (float2)(0.0f, 0.0f);
            s = s + (v - ref);
        }
    }
    mab[(size_t)ry * mw + rx] = ref + s * inv;
}

static inline float2 sample_bilinear_roi(__global const float2* mab,
                                         const int hs, const int ws,
                                         const int my0, const int my1,
                                         const int mx0, const int mx1, const int mw,
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
    float2 z = (float2)(0.0f, 0.0f);
    float2 c00 = (y0 >= my0 && y0 < my1 && x0 >= mx0 && x0 < mx1) ? mab[(size_t)(y0 - my0) * mw + (x0 - mx0)] : z;
    float2 c01 = (y0 >= my0 && y0 < my1 && x1 >= mx0 && x1 < mx1) ? mab[(size_t)(y0 - my0) * mw + (x1 - mx0)] : z;
    float2 c10 = (y1 >= my0 && y1 < my1 && x0 >= mx0 && x0 < mx1) ? mab[(size_t)(y1 - my0) * mw + (x0 - mx0)] : z;
    float2 c11 = (y1 >= my0 && y1 < my1 && x1 >= mx0 && x1 < mx1) ? mab[(size_t)(y1 - my0) * mw + (x1 - mx0)] : z;
    float2 r0 = c00 + (c01 - c00) * dx;
    float2 r1 = c10 + (c11 - c10) * dx;
    return r0 + (r1 - r0) * dy;
}

// Fused bilinear upsample + 3x3 Sobel + streaming GfgDecision update in local memory.
__kernel void gfg_up_sobel_decision_local(__global const float2* mab,
                                          __global const float* gray,
                                          __global float* val,
                                          __global float* sob_best,
                                          __global int* idx,
                                          const int H, const int W,
                                          const int hs, const int ws,
                                          const int frame_k, const float tie_eps,
                                          const int first_valid) {
    __local float tile[18][18];
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int tid = ly * 16 + lx;
    int gx0 = get_group_id(0) * 16;
    int gy0 = get_group_id(1) * 16;

    for (int i = tid; i < 324; i += 256) {
        int dy = i / 18;
        int dx = i - dy * 18;
        int yy = reflect101_idx(gy0 + dy - 1, H);
        int xx = reflect101_idx(gx0 + dx - 1, W);
        float2 ab = sample_bilinear_roi(mab, hs, ws, 0, hs, 0, ws, ws, H, W, xx, yy);
        tile[dy][dx] = ab.x * gray[(size_t)yy * W + xx] + ab.y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = gx0 + lx;
    int y = gy0 + ly;
    if (x >= W || y >= H) return;

    float a00 = tile[ly][lx],     a01 = tile[ly][lx + 1],     a02 = tile[ly][lx + 2];
    float a10 = tile[ly + 1][lx], a11 = tile[ly + 1][lx + 1], a12 = tile[ly + 1][lx + 2];
    float a20 = tile[ly + 2][lx], a21 = tile[ly + 2][lx + 1], a22 = tile[ly + 2][lx + 2];

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

__kernel void gfg_active_bboxes(__global const int* idx,
                                __global uint* wins,
                                __global uint* ymin, __global uint* ymax,
                                __global uint* xmin, __global uint* xmax,
                                const int H, const int W) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int k = idx[(size_t)y * W + x];
    if (wins[k] == 0u) wins[k] = 1u;
    uint uy = (uint)y;
    uint ux = (uint)x;
    if (uy < ymin[k]) atomic_min(&ymin[k], uy);
    if (uy > ymax[k]) atomic_max(&ymax[k], uy);
    if (ux < xmin[k]) atomic_min(&xmin[k], ux);
    if (ux > xmax[k]) atomic_max(&xmax[k], ux);
}

#define DEFINE_UP_ACCUM_ROI(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const float2* mab, \
                   __global const SRC_T* src, \
                   __global float4* acc, \
                   const int H, const int W, const int C, const int ch, const int bgr_rev, \
                   const int hs, const int ws, \
                   const int my0, const int my1, const int mx0, const int mx1, const int mw, \
                   const int oy0, const int ox0, const int oh, const int ow) { \
    int rx = get_global_id(0); \
    int ry = get_global_id(1); \
    if (rx >= ow || ry >= oh) return; \
    int x = ox0 + rx; \
    int y = oy0 + ry; \
    float2 ab = sample_bilinear_roi(mab, hs, ws, my0, my1, mx0, mx1, mw, H, W, x, y); \
    if (ab.x == 0.0f && ab.y == 0.0f) return; \
    size_t pos = (size_t)y * W + x; \
    float c0, c1, c2, g; \
    if (C == 1) { \
        SRC_T raw0 = src[pos]; \
        c0 = c1 = c2 = g = (CONV_EXPR); \
    } else { \
        size_t b = pos * C; \
        SRC_T raw0 = src[b + (bgr_rev ? 2 : 0)]; c0 = (CONV_EXPR); \
        raw0 = src[b + 1]; c1 = (CONV_EXPR); \
        raw0 = src[b + (bgr_rev ? 0 : 2)]; c2 = (CONV_EXPR); \
        g = (ch == 0) ? c0 : ((ch == 1) ? c1 : c2); \
    } \
    float w = ab.x * g + ab.y; \
    float4 prev = acc[pos]; \
    acc[pos] = (float4)(prev.x + c0 * w, \
                        prev.y + c1 * w, \
                        prev.z + c2 * w, \
                        prev.w + w); \
}

DEFINE_UP_ACCUM_ROI(gfg_up_accum_roi_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_UP_ACCUM_ROI(gfg_up_accum_roi_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_UP_ACCUM_ROI(gfg_up_accum_roi_f32, float, (raw0 / 255.0f))

__kernel void gfg_count_zero(__global const float4* acc, __global uint* zero_cnt, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    if (acc[i].w <= 1e-6f) atomic_inc(zero_cnt);
}

#define DEFINE_FILL_ZERO(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global const int* idx, \
                   __global float4* acc, const int n, const int C, \
                   const int bgr_rev, const int frame_k) { \
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
            SRC_T raw0 = src[b + (bgr_rev ? 2 : 0)]; c0 = (CONV_EXPR); \
            raw0 = src[b + 1]; c1 = (CONV_EXPR); \
            raw0 = src[b + (bgr_rev ? 0 : 2)]; c2 = (CONV_EXPR); \
        } \
        acc[i] = (float4)(c0, c1, c2, 1.0f); \
    } \
}

DEFINE_FILL_ZERO(gfg_fill_zero_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_FILL_ZERO(gfg_fill_zero_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_FILL_ZERO(gfg_fill_zero_f32, float, (raw0 / 255.0f))

__kernel void gfg_normalize_u8(__global const float4* acc, __global uchar* out,
                               const int start, const int count) {
    int i = get_global_id(0);
    if (i >= count) return;
    float4 v = acc[start + i];
    float3 rgb = (v.w > 1e-6f) ? (v.xyz / v.w) : v.xyz;
    rgb = clamp(rgb * 255.0f, 0.0f, 255.0f);
    size_t o = (size_t)i * 3;
    out[o]     = (uchar)rgb.x;
    out[o + 1] = (uchar)rgb.y;
    out[o + 2] = (uchar)rgb.z;
}

__kernel void gfg_normalize_f32(__global const float4* acc, __global float* out,
                                const int start, const int count) {
    int i = get_global_id(0);
    if (i >= count) return;
    float4 v = acc[start + i];
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


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _reflect101_span(lo, hi, rad, n):
    if rad <= 0 or n <= 1:
        return lo, hi
    if n <= 2 * rad + 1:
        return 0, n
    r_lo = max(0, lo - rad)
    r_hi = min(n, hi + rad)
    if lo < rad:
        r_hi = max(r_hi, min(n, rad - lo + 1))
    if hi > n - rad:
        r_lo = min(r_lo, max(0, 2 * (n - 1) - (hi - 1 + rad)))
    return r_lo, r_hi


def _gf_active_bounds(lo_win, hi_win, rad, n):
    k = 2 * rad + 1
    if n <= k:
        return 0, n, 0, n
    a0 = max(0, lo_win - rad)
    a1 = min(n, hi_win + rad + 1)
    q0 = max(0, a0 - rad)
    q1 = min(n, a1 + rad)
    return a0, a1, q0, q1


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
    'gfg_extract_ia_local_u8', 'gfg_extract_ia_local_u16', 'gfg_extract_ia_local_f32',
    'box1_h_scan', 'box1_v_ia_combine',
    'gfg_area4_down_pair', 'gfg_area_down_pair',
    'gfg_area4_mask_roi_u8', 'gfg_area4_mask_roi_u16', 'gfg_area4_mask_roi_f32',
    'gfg_area_mask_roi_u8', 'gfg_area_mask_roi_u16', 'gfg_area_mask_roi_f32',
    'gfg_fgf_sub_ab_roi', 'gfg_fgf_sub_mean_ab_roi',
    'gfg_up_sobel_decision_local',
    'gfg_active_bboxes',
    'gfg_up_accum_roi_u8', 'gfg_up_accum_roi_u16', 'gfg_up_accum_roi_f32',
    'gfg_count_zero',
    'gfg_fill_zero_u8', 'gfg_fill_zero_u16', 'gfg_fill_zero_f32',
    'gfg_normalize_u8', 'gfg_normalize_f32',
)
_cache_lock = Lock()
_PROGRAM_CACHE = None


def _device_cache_key(device):
    return (
        getattr(device, 'int_ptr', None),
        str(getattr(device, 'name', '')),
        str(getattr(device, 'vendor', '')),
        str(getattr(device, 'driver_version', '')),
        int(getattr(device, 'global_mem_size', 0)),
    )


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import (
        _build_cached_program, _fp32_build_options, _get_shared_context,
    )

    key = _device_cache_key(device)
    with _cache_lock:
        ctx = _get_shared_context(cl, device)
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key and _PROGRAM_CACHE[1] is ctx:
            return _PROGRAM_CACHE[1:]
        q = cl.CommandQueue(ctx, device=device)
        prog = _build_cached_program(
            cl, device, ctx, _GFG_K,
            options=_fp32_build_options(cl, device),
        )
        kernels = {name: cl.Kernel(prog, name) for name in _KERNEL_NAMES}
        _PROGRAM_CACHE = (key, ctx, q, kernels)
        return ctx, q, kernels


def _invalidate_program_cache(context=None):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _invalidate_shared_context

    with _cache_lock:
        _PROGRAM_CACHE = None
        _invalidate_shared_context(context)


class GpuGfg:
    """Device-resident OpenCL GFG-FGF fusion with fused local kernels & ROI Pass 3."""

    def __init__(self):
        from fusion_methods._openfocus_gpu import (
            GpuError, _device, _get_transfer_queue, opencl_device_name,
        )
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
            self.tq = _get_transfer_queue(cl, device, self.ctx)
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU GFG-FGF initialization failed') from exc

    def _run(self, kernel, gsize, *args, lsize=None, wait_for=None):
        from fusion_methods._openfocus_gpu import _enqueue_kernel
        g = gsize if isinstance(gsize, tuple) else (gsize,)
        l = (lsize,) if isinstance(lsize, int) else lsize
        return _enqueue_kernel(kernel, self.q, g, l, *args, wait_for=wait_for)

    def _drain_queue(self):
        for q in (getattr(self, 'tq', None), getattr(self, 'q', None)):
            if q is not None:
                try:
                    q.finish()
                except Exception:
                    _invalidate_program_cache(getattr(self, 'ctx', None))

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
                _invalidate_program_cache(getattr(self, 'ctx', None))
                self._drain_queue()
                raise GpuError('OpenCL GPU GFG-FGF execution failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.k = None
            self.tq = None
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, img_resize, kernel_size, min_frame_focus,
                     should_cancel, progress_callback):
        from dataclasses import replace
        from core import resource_policy
        from core.source_frames import resolve_frame, without_source_cache_growth
        from fusion_methods._openfocus_gpu import (
            GpuError, _PinnedUploader, _event_wait_list, _free_vram,
            _iter_prefetched, _prefetch_window, _prepare_gpu_upload,
            _release_buffers, _wait_event,
        )

        if not frames:
            raise ValueError('Input stack is empty')
        mean_ksize = int(kernel_size)
        if mean_ksize < 1 or (mean_ksize > 1 and mean_ksize % 2 == 0):
            raise GpuError('GFG-FGF GPU requires kernel_size == 1 or positive odd integer')
        rad_mean = mean_ksize // 2
        if rad_mean > 64:
            raise GpuError('GFG-FGF GPU requires kernel_size <= 129')

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
        out_itemsize = 1 if is_u8_out else 4
        out_bytes = n * 3 * out_itemsize
        readback_bytes = min(max(3 * out_itemsize, out_bytes), _READBACK_BUFFER_BYTES)

        snapshot = resource_policy.probe_memory()
        host_budget = resource_policy.ram_ceiling(snapshot)
        source_peak = 0
        src_bytes = 0
        for frame in frames:
            _check_cancel(should_cancel)
            source_peak = max(source_peak, _source_staging_bytes(frame, shape, img_resize))
            src_bytes = max(src_bytes, _frame_upload_bytes(frame, shape, img_resize))
        staging = max(source_peak, out_bytes + readback_bytes) + 16 * 1024 * 1024
        if staging > host_budget:
            raise GpuError(f'GFG-FGF GPU staging exceeds the host memory budget '
                           f'({staging >> 20} MiB needed, {host_budget >> 20} MiB available)')

        need_t1 = rad_mean > _LOCAL_IA_MAX_RAD
        # Pass 2 peak: gray(4n) + ia(4n) + [t1(4n)] + val(4n) + sob(4n) + idx(4n) + sub(32*n_sub) + src
        # Pass 3 peak: idx(4n) + acc(16n) + sub(32*n_sub) + src
        n_mult = 24 if need_t1 else 20
        need = (n_mult * n
                + 32 * n_sub
                + max(src_bytes, readback_bytes)
                + 5 * count * 4
                + 16 * 1024 * 1024)
        unified_memory = self.device.host_unified_memory
        limit = int(self.device.global_mem_size) // 2
        if unified_memory:
            limit = min(limit, host_budget - staging)
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError('GFG-FGF GPU free memory is unknown')
            snapshot = replace(snapshot, free_vram=memory[0], total_vram=memory[1])
            limit = min(limit, resource_policy.vram_budget(snapshot))
        single = int(self.device.max_mem_alloc_size)
        largest = max(16 * n, src_bytes, readback_bytes)
        if need > limit or largest > single:
            raise GpuError('GFG-FGF GPU workspace exceeds the device budget')

        prefetch_budget = host_budget - need if unified_memory else host_budget
        prefetch_slots = _prefetch_window(
            prefetch_budget, staging, source_peak, count,
        )
        ping_pong = (
            count >= 2
            and (need + src_bytes) <= limit
            and (staging + prefetch_slots * source_peak + 2 * src_bytes
                 + (src_bytes if unified_memory else 0)) <= prefetch_budget
        )

        resize = (img_resize is not None and (int(first_shape[1]), int(first_shape[0])) != img_resize)
        exact_div4 = (height % _GF_SUB == 0) and (width % _GF_SUB == 0)

        def prepare_upload(_index, source, cancelled):
            _check_cancel(cancelled)
            with without_source_cache_growth():
                raw = resolve_frame(source, should_cancel=cancelled)
            _check_cancel(cancelled)
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
                return host_src, 3, _upload_kind(host_src.dtype), 0
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
            host_src, bgr_rev = _prepare_gpu_upload(raw, dtype)
            del raw
            return host_src, channels, _upload_kind(dtype), (1 if bgr_rev else 0)

        src_bufs = ()
        gray_buf = ia_buf = t1_buf = val_buf = sob_buf = idx_buf = None
        sub4_buf = ab_buf = mab_buf = wg4_buf = sc_buf = None
        wins_buf = ymin_buf = ymax_buf = xmin_buf = xmax_buf = None
        acc_buf = zcnt_buf = dst_buf = None
        uploader = prepared_iter = None
        in_flight = False
        cancel_fn = lambda: _check_cancel(should_cancel)

        try:
            src_bufs = tuple(
                self.cl.Buffer(self.ctx, self.mf.READ_ONLY, max(1, src_bytes))
                for _ in range(2 if ping_pong else 1)
            )
            if ping_pong:
                uploader = _PinnedUploader(
                    self.cl, self.ctx, self.q, getattr(self, 'tq', None),
                    2, max(1, src_bytes),
                )
            gray_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            ia_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            t1_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n)) if need_t1 else None
            val_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            sob_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            idx_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            sub4_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n_sub))
            ab_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 8 * n_sub))
            mab_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 8 * n_sub))

            red_groups = max(1, min(256, _cdiv(n, 256)))
            wg4_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, red_groups * 16)

            # Frame 0 upload + dominant channel selection.
            host0, ch0, kind0, rev0 = prepare_upload(0, frames[0], should_cancel)
            in_flight = True
            self.cl.enqueue_copy(self.q, src_bufs[0], async_upload_view(host0), is_blocking=False)
            self._run(
                self.k[f'gfg_chan_sum_{kind0}'], red_groups * 256,
                src_bufs[0], wg4_buf, np.int32(n), np.int32(ch0), np.int32(rev0), lsize=256,
            )
            wg4_host = np.empty((red_groups, 4), dtype=np.float32)
            self.cl.enqueue_copy(self.q, wg4_host, wg4_buf, is_blocking=True)
            in_flight = False
            channel_idx = int(np.argmax(wg4_host[:, :3].sum(axis=0, dtype=np.float64)))
            _release_buffers(wg4_buf)
            wg4_buf = None
            del wg4_host

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
                        cur_ch, cur_kind, cur_rev = ch0, kind0, rev0
                    else:
                        h_src, cur_ch, cur_kind, cur_rev = prepare_upload(index, frames[index], should_cancel)
                        in_flight = True
                        self.cl.enqueue_copy(self.q, src_bufs[0], async_upload_view(h_src), is_blocking=False)
                    in_flight = True
                    self._run(
                        self.k[f'gfg_extract_{cur_kind}'], n,
                        src_bufs[0], gray_buf, np.int32(n), np.int32(cur_ch),
                        np.int32(channel_idx), np.int32(cur_rev),
                    )
                    if height > 2 and width > 2:
                        self._run(
                            self.k['gfg_scharr_sum'], sc_groups * 256,
                            gray_buf, sc_buf, np.int32(height), np.int32(width), lsize=256,
                        )
                        self.cl.enqueue_copy(self.q, sc_host, sc_buf, is_blocking=True)
                        focus_vals[index] = float(sc_host.sum(dtype=np.float64)) / ((height - 2) * (width - 2))
                    else:
                        self.q.finish()
                    in_flight = False
                    if h_src is not None:
                        del h_src
                _release_buffers(sc_buf)
                sc_buf = None
                del sc_host
                max_focus = float(np.max(focus_vals)) or 1.0

            grid_16 = (_round_up(width, 16), _round_up(height, 16))
            local_16 = (16, 16)
            local_128 = (128, 1)
            r_sub = max(1, _GF_RADIUS // _GF_SUB)

            # Pass 2: Focus map + Sobel-tie-broken streaming decision.
            valid_indices = [
                idx for idx in range(count)
                if not (scale > 0.0 and focus_vals[idx] < scale * max_focus)
            ]
            # Prefetch frames after index 0 (frame 0 is already loaded in host0).
            rest_indices = [idx for idx in valid_indices if idx > 0]
            rest_frames = [frames[idx] for idx in rest_indices]
            rest_slots = min(prefetch_slots, max(0, len(rest_indices) - 1)) if rest_indices else 0
            prepared_iter = _iter_prefetched(
                rest_frames, prepare_upload, enabled=rest_slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, rest_slots),
            )

            def _iter_pass2():
                if valid_indices and valid_indices[0] == 0:
                    yield 0, (host0, ch0, kind0, rev0), (scale <= 0.0)
                for pos, item in prepared_iter:
                    yield rest_indices[pos], item, False

            slot_events = [None, None]
            slot_hosts = [None, None]
            first_valid = True
            next_progress = 0
            step_pos = 0

            for index, (h_src, cur_ch, cur_kind, cur_rev), already_on_gpu in _iter_pass2():
                while next_progress < index:
                    if progress_callback is not None:
                        progress_callback(next_progress + 1, total_steps)
                    next_progress += 1

                slot = 0 if already_on_gpu else (step_pos & 1 if ping_pong else 0)
                step_pos += 1
                prev_ev = slot_events[slot]
                if prev_ev is not None and (uploader is None or not uploader.separate_queue):
                    _wait_event(prev_ev, cancel_fn)
                    slot_events[slot] = None
                    slot_hosts[slot] = None
                    prev_ev = None
                _check_cancel(should_cancel)
                src_buf = src_bufs[slot]
                in_flight = True
                if already_on_gpu:
                    upload = None
                    wait_upload = None
                elif uploader is not None:
                    upload = uploader.upload(
                        slot, src_buf, h_src,
                        wait_for=_event_wait_list(prev_ev), cancel=cancel_fn,
                    )
                    wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                else:
                    upload = self.cl.enqueue_copy(self.q, src_buf, async_upload_view(h_src), is_blocking=False)
                    wait_upload = None

                if rad_mean <= _LOCAL_IA_MAX_RAD:
                    src_done = self._run(
                        self.k[f'gfg_extract_ia_local_{cur_kind}'], grid_16,
                        src_buf, gray_buf, ia_buf,
                        np.int32(height), np.int32(width), np.int32(cur_ch),
                        np.int32(channel_idx), np.int32(cur_rev),
                        np.int32(rad_mean), np.float32(_GFG_THRESHOLD),
                        lsize=local_16, wait_for=wait_upload,
                    )
                else:
                    src_done = self._run(
                        self.k[f'gfg_extract_{cur_kind}'], n,
                        src_buf, gray_buf, np.int32(n), np.int32(cur_ch),
                        np.int32(channel_idx), np.int32(cur_rev),
                        wait_for=wait_upload,
                    )
                    self._run(
                        self.k['box1_h_scan'], (_round_up(width, 128), height),
                        gray_buf, t1_buf, np.int32(height), np.int32(width),
                        np.int32(rad_mean), lsize=local_128,
                    )
                    self._run(
                        self.k['box1_v_ia_combine'], (width, _cdiv(height, _CHUNK)),
                        gray_buf, t1_buf, ia_buf,
                        np.int32(height), np.int32(width),
                        np.int32(rad_mean), np.float32(_GFG_THRESHOLD),
                    )

                down_kern = 'gfg_area4_down_pair' if exact_div4 else 'gfg_area_down_pair'
                self._run(
                    self.k[down_kern], (w_sub, h_sub),
                    gray_buf, ia_buf, sub4_buf,
                    np.int32(height), np.int32(width), np.int32(h_sub), np.int32(w_sub),
                )
                self._run(
                    self.k['gfg_fgf_sub_ab_roi'], (w_sub, h_sub),
                    sub4_buf, ab_buf, np.int32(h_sub), np.int32(w_sub),
                    np.int32(0), np.int32(0), np.int32(w_sub),
                    np.int32(0), np.int32(0), np.int32(h_sub), np.int32(w_sub),
                    np.int32(r_sub), np.float32(_GF_EPS),
                )
                self._run(
                    self.k['gfg_fgf_sub_mean_ab_roi'], (w_sub, h_sub),
                    ab_buf, mab_buf, np.int32(h_sub), np.int32(w_sub),
                    np.int32(0), np.int32(h_sub), np.int32(0), np.int32(w_sub), np.int32(w_sub),
                    np.int32(0), np.int32(0), np.int32(h_sub), np.int32(w_sub),
                    np.int32(r_sub),
                )
                self._run(
                    self.k['gfg_up_sobel_decision_local'], grid_16,
                    mab_buf, gray_buf, val_buf, sob_buf, idx_buf,
                    np.int32(height), np.int32(width), np.int32(h_sub), np.int32(w_sub),
                    np.int32(index), np.float32(_TIE_EPS), np.int32(1 if first_valid else 0),
                    lsize=local_16,
                )
                first_valid = False

                if ping_pong and src_done is not None and hasattr(src_done, 'wait'):
                    if hasattr(self.q, 'flush'):
                        self.q.flush()
                    slot_events[slot] = src_done
                    slot_hosts[slot] = (h_src, upload)
                else:
                    self.q.finish()
                    in_flight = False
                    del h_src, upload

                if index == 0:
                    host0 = None
                if progress_callback is not None:
                    progress_callback(index + 1, total_steps)
                next_progress = index + 1

            host0 = None
            while next_progress < count:
                if progress_callback is not None:
                    progress_callback(next_progress + 1, total_steps)
                next_progress += 1

            if ping_pong:
                if uploader is not None:
                    uploader.finish(cancel_fn)
                for slot in (0, 1):
                    if slot_events[slot] is not None:
                        _wait_event(slot_events[slot], cancel_fn)
                        slot_events[slot] = None
                    slot_hosts[slot] = None
            prepared_iter.close()
            prepared_iter = None

            if first_valid:
                in_flight = True
                self.cl.enqueue_fill_buffer(self.q, idx_buf, np.int32(0), 0, 4 * n)

            _release_buffers(val_buf, sob_buf, ia_buf, t1_buf, gray_buf)
            val_buf = sob_buf = ia_buf = t1_buf = gray_buf = None

            # Compute active flags and exact per-frame bounding boxes on GPU.
            wins_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, count * 4))
            ymin_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, count * 4))
            ymax_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, count * 4))
            xmin_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, count * 4))
            xmax_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, count * 4))
            in_flight = True
            self.cl.enqueue_fill_buffer(self.q, wins_buf, np.uint32(0), 0, count * 4)
            self.cl.enqueue_fill_buffer(self.q, ymin_buf, np.uint32(height), 0, count * 4)
            self.cl.enqueue_fill_buffer(self.q, ymax_buf, np.uint32(0), 0, count * 4)
            self.cl.enqueue_fill_buffer(self.q, xmin_buf, np.uint32(width), 0, count * 4)
            self.cl.enqueue_fill_buffer(self.q, xmax_buf, np.uint32(0), 0, count * 4)
            self._run(
                self.k['gfg_active_bboxes'], (width, height),
                idx_buf, wins_buf, ymin_buf, ymax_buf, xmin_buf, xmax_buf,
                np.int32(height), np.int32(width),
            )
            wins = np.empty(count, dtype=np.uint32)
            ymin = np.empty(count, dtype=np.uint32)
            ymax = np.empty(count, dtype=np.uint32)
            xmin = np.empty(count, dtype=np.uint32)
            xmax = np.empty(count, dtype=np.uint32)
            self.cl.enqueue_copy(self.q, wins, wins_buf, is_blocking=False)
            self.cl.enqueue_copy(self.q, ymin, ymin_buf, is_blocking=False)
            self.cl.enqueue_copy(self.q, ymax, ymax_buf, is_blocking=False)
            self.cl.enqueue_copy(self.q, xmin, xmin_buf, is_blocking=False)
            self.cl.enqueue_copy(self.q, xmax, xmax_buf, is_blocking=True)
            in_flight = False
            _release_buffers(wins_buf, ymin_buf, ymax_buf, xmin_buf, xmax_buf)
            wins_buf = ymin_buf = ymax_buf = xmin_buf = xmax_buf = None

            # Pass 3: ROI-restricted guided-filter mask refinement + zero-ab-skip RGB accumulation.
            acc_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            in_flight = True
            self.cl.enqueue_fill_buffer(self.q, acc_buf, np.float32(0.0), 0, 16 * n)

            active_indices = [idx for idx in range(count) if wins[idx]]
            active_frames = [frames[idx] for idx in active_indices]
            act_slots = min(prefetch_slots, max(0, len(active_indices) - 1)) if active_indices else 0
            prepared_iter = _iter_prefetched(
                active_frames, prepare_upload, enabled=act_slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, act_slots),
            )
            slot_events = [None, None]
            slot_hosts = [None, None]
            next_progress = 0

            for pos, (h_src, cur_ch, cur_kind, cur_rev) in prepared_iter:
                index = active_indices[pos]
                while next_progress < index:
                    if progress_callback is not None:
                        progress_callback(count + next_progress + 1, total_steps)
                    next_progress += 1

                slot = pos & 1 if ping_pong else 0
                prev_ev = slot_events[slot]
                if prev_ev is not None and (uploader is None or not uploader.separate_queue):
                    _wait_event(prev_ev, cancel_fn)
                    slot_events[slot] = None
                    slot_hosts[slot] = None
                    prev_ev = None
                _check_cancel(should_cancel)
                src_buf = src_bufs[slot]
                in_flight = True
                if uploader is not None:
                    upload = uploader.upload(
                        slot, src_buf, h_src,
                        wait_for=_event_wait_list(prev_ev), cancel=cancel_fn,
                    )
                    wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                else:
                    upload = self.cl.enqueue_copy(self.q, src_buf, async_upload_view(h_src), is_blocking=False)
                    wait_upload = None

                y0_k, y1_k = int(ymin[index]), int(ymax[index])
                x0_k, x1_k = int(xmin[index]), int(xmax[index])

                # Map active pixel bbox to sub-grid bbox + exact two-pass r_sub support.
                win_sy0 = max(0, (y0_k * h_sub) // height)
                win_sy1 = min(h_sub, ((y1_k + 1) * h_sub + height - 1) // height)
                win_sx0 = max(0, (x0_k * w_sub) // width)
                win_sx1 = min(w_sub, ((x1_k + 1) * w_sub + width - 1) // width)

                say0, say1, smy0, smy1 = _gf_active_bounds(win_sy0, max(win_sy0, win_sy1 - 1), r_sub, h_sub)
                sax0, sax1, smx0, smx1 = _gf_active_bounds(win_sx0, max(win_sx0, win_sx1 - 1), r_sub, w_sub)
                sub_y0, sub_y1 = _reflect101_span(say0, say1, r_sub, h_sub)
                sub_x0, sub_x1 = _reflect101_span(sax0, sax1, r_sub, w_sub)
                sub_h, sub_w = sub_y1 - sub_y0, sub_x1 - sub_x0
                sah, saw = say1 - say0, sax1 - sax0
                smh, smw = smy1 - smy0, smx1 - smx0

                out_y0 = max(0, int(math.floor((smy0 - 0.5) * height / h_sub)) - 1)
                out_y1 = min(height, int(math.ceil((smy1 + 0.5) * height / h_sub)) + 1)
                out_x0 = max(0, int(math.floor((smx0 - 0.5) * width / w_sub)) - 1)
                out_x1 = min(width, int(math.ceil((smx1 + 0.5) * width / w_sub)) + 1)

                if exact_div4:
                    self._run(
                        self.k[f'gfg_area4_mask_roi_{cur_kind}'], (sub_w, sub_h),
                        src_buf, idx_buf, sub4_buf,
                        np.int32(height), np.int32(width), np.int32(cur_ch),
                        np.int32(channel_idx), np.int32(cur_rev),
                        np.int32(sub_y0), np.int32(sub_x0), np.int32(sub_h), np.int32(sub_w),
                        np.int32(index), wait_for=wait_upload,
                    )
                else:
                    self._run(
                        self.k[f'gfg_area_mask_roi_{cur_kind}'], (sub_w, sub_h),
                        src_buf, idx_buf, sub4_buf,
                        np.int32(height), np.int32(width), np.int32(cur_ch),
                        np.int32(channel_idx), np.int32(cur_rev),
                        np.int32(h_sub), np.int32(w_sub),
                        np.int32(sub_y0), np.int32(sub_x0), np.int32(sub_h), np.int32(sub_w),
                        np.int32(index), wait_for=wait_upload,
                    )
                self._run(
                    self.k['gfg_fgf_sub_ab_roi'], (saw, sah),
                    sub4_buf, ab_buf, np.int32(h_sub), np.int32(w_sub),
                    np.int32(sub_y0), np.int32(sub_x0), np.int32(sub_w),
                    np.int32(say0), np.int32(sax0), np.int32(sah), np.int32(saw),
                    np.int32(r_sub), np.float32(_GF_EPS),
                )
                self._run(
                    self.k['gfg_fgf_sub_mean_ab_roi'], (smw, smh),
                    ab_buf, mab_buf, np.int32(h_sub), np.int32(w_sub),
                    np.int32(say0), np.int32(say1), np.int32(sax0), np.int32(sax1), np.int32(saw),
                    np.int32(smy0), np.int32(smx0), np.int32(smh), np.int32(smw),
                    np.int32(r_sub),
                )
                last_ev = self._run(
                    self.k[f'gfg_up_accum_roi_{cur_kind}'], (out_x1 - out_x0, out_y1 - out_y0),
                    mab_buf, src_buf, acc_buf,
                    np.int32(height), np.int32(width), np.int32(cur_ch),
                    np.int32(channel_idx), np.int32(cur_rev),
                    np.int32(h_sub), np.int32(w_sub),
                    np.int32(smy0), np.int32(smy1), np.int32(smx0), np.int32(smx1), np.int32(smw),
                    np.int32(out_y0), np.int32(out_x0), np.int32(out_y1 - out_y0), np.int32(out_x1 - out_x0),
                )

                if ping_pong and last_ev is not None and hasattr(last_ev, 'wait'):
                    if hasattr(self.q, 'flush'):
                        self.q.flush()
                    slot_events[slot] = last_ev
                    slot_hosts[slot] = (h_src, upload)
                else:
                    self.q.finish()
                    in_flight = False
                    del h_src, upload
                if progress_callback is not None:
                    progress_callback(count + index + 1, total_steps)
                next_progress = index + 1

            while next_progress < count:
                if progress_callback is not None:
                    progress_callback(count + next_progress + 1, total_steps)
                next_progress += 1

            if ping_pong:
                if uploader is not None:
                    uploader.finish(cancel_fn)
                for slot in (0, 1):
                    if slot_events[slot] is not None:
                        _wait_event(slot_events[slot], cancel_fn)
                        slot_events[slot] = None
                    slot_hosts[slot] = None
                self.q.finish()
                in_flight = False
            elif in_flight:
                self.q.finish()
                in_flight = False

            if uploader is not None:
                uploader.close()
                uploader = None
            prepared_iter.close()
            prepared_iter = None
            _release_buffers(sub4_buf, ab_buf, mab_buf)
            sub4_buf = ab_buf = mab_buf = None

            # Zero-weight fallback: only resolve frames referenced by sum_w <= 1e-6 pixels.
            zcnt_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, 4)
            in_flight = True
            self.cl.enqueue_fill_buffer(self.q, zcnt_buf, np.uint32(0), 0, 4)
            self._run(self.k['gfg_count_zero'], n, acc_buf, zcnt_buf, np.int32(n))
            zcnt = np.empty(1, dtype=np.uint32)
            self.cl.enqueue_copy(self.q, zcnt, zcnt_buf, is_blocking=True)
            in_flight = False
            _release_buffers(zcnt_buf)
            zcnt_buf = None
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
                    h_src, cur_ch, cur_kind, cur_rev = prepare_upload(int(k), frames[int(k)], should_cancel)
                    in_flight = True
                    self.cl.enqueue_copy(self.q, src_bufs[0], async_upload_view(h_src), is_blocking=True)
                    del h_src
                    self._run(
                        self.k[f'gfg_fill_zero_{cur_kind}'], n,
                        src_bufs[0], idx_buf, acc_buf,
                        np.int32(n), np.int32(cur_ch), np.int32(cur_rev), np.int32(int(k)),
                    )
                self.q.finish()
                in_flight = False

            _release_buffers(src_bufs, idx_buf)
            src_bufs = ()
            idx_buf = None

            _check_cancel(should_cancel)
            out = np.empty(shape, dtype=out_dtype)
            out_pixels = out.reshape(n, 3)
            dst_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, readback_bytes)
            chunk_pixels = max(1, readback_bytes // (3 * out_itemsize))
            norm_kern = self.k['gfg_normalize_u8' if is_u8_out else 'gfg_normalize_f32']
            in_flight = True
            for start in range(0, n, chunk_pixels):
                _check_cancel(should_cancel)
                cnt = min(chunk_pixels, n - start)
                self._run(norm_kern, cnt, acc_buf, dst_buf, np.int32(start), np.int32(cnt))
                self.cl.enqueue_copy(self.q, out_pixels[start:start + cnt], dst_buf, is_blocking=True)
            in_flight = False
            if progress_callback is not None:
                progress_callback(total_steps, total_steps)
            _check_cancel(should_cancel)
            return out
        except BaseException:
            if in_flight:
                self._drain_queue()
            raise
        finally:
            if uploader is not None:
                uploader.close()
            if prepared_iter is not None:
                prepared_iter.close()
            _release_buffers(
                src_bufs, gray_buf, ia_buf, t1_buf, val_buf, sob_buf, idx_buf,
                sub4_buf, ab_buf, mab_buf, wg4_buf, sc_buf,
                wins_buf, ymin_buf, ymax_buf, xmin_buf, xmax_buf,
                acc_buf, zcnt_buf, dst_buf,
            )
