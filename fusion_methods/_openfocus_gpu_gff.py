"""OpenCL GPU fusion for Guided Filter (opt-in via use_gpu=True on gff_impl).

Full-frame streaming OpenCL 1.2 implementation using:
- Fused channel-sum + |Laplacian| and tiled local-memory 41-tap separable Gaussian
  with online strict-`>` winner selection in Pass 1.
- GPU active bounding-box extraction to restrict Pass 2 work strictly to each
  frame's non-zero guided-filter support.
- Band-streamed, zero-transpose Pass 2 (shared-memory parallel prefix-sum horizontal
  box filters + coalesced vertical sliding sums, plus tile-skipping local-memory
  r=7 detail guided filter) that keeps full-frame mathematical equivalence while
  reducing VRAM footprint from ~120*N to ~45*N bytes.
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

_DEFAULT_R1 = 45
_DEFAULT_R2 = 7
_DEFAULT_EPS1 = 0.3
_DEFAULT_EPS2 = 10e-6
_DEFAULT_SIGMA_R = 5.0
_GAUSS_KSIZE = 41
_GAUSS_RAD = _GAUSS_KSIZE // 2
_CHUNK = 64
_BAND_H = 512
_SCAN_MAX_RAD = 64
_READBACK_BUFFER_BYTES = 8 * 1024 * 1024

_GFF_K = r'''
#pragma OPENCL FP_CONTRACT OFF
#define CHUNK 64
#define GAUSS_RAD 20

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

#define DEFINE_SUM_LAP(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global float* dst, \
                   const int H, const int W, const int C, const int bgr_rev) { \
    __local float tile[18][18]; \
    int lx = get_local_id(0); \
    int ly = get_local_id(1); \
    int tid = ly * 16 + lx; \
    int gx0 = get_group_id(0) * 16; \
    int gy0 = get_group_id(1) * 16; \
    for (int i = tid; i < 324; i += 256) { \
        int dy = i / 18; \
        int dx = i - dy * 18; \
        int yy = reflect_idx(gy0 + dy - 1, H); \
        int xx = reflect_idx(gx0 + dx - 1, W); \
        size_t p = (size_t)yy * W + xx; \
        float s; \
        if (C == 1) { \
            SRC_T raw0 = src[p]; \
            float v = (CONV_EXPR); \
            s = (v + v) + v; \
        } else { \
            size_t b = p * C; \
            SRC_T raw0 = src[b + (bgr_rev ? 2 : 0)]; float c0 = (CONV_EXPR); \
            raw0 = src[b + 1]; float c1 = (CONV_EXPR); \
            raw0 = src[b + (bgr_rev ? 0 : 2)]; float c2 = (CONV_EXPR); \
            s = (c0 + c1) + c2; \
        } \
        tile[dy][dx] = s; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int x = gx0 + lx; \
    int y = gy0 + ly; \
    if (x >= W || y >= H) return; \
    float c = tile[ly + 1][lx + 1]; \
    float nb = (tile[ly][lx + 1] + tile[ly + 2][lx + 1]) \
             + (tile[ly + 1][lx] + tile[ly + 1][lx + 2]); \
    dst[(size_t)y * W + x] = fabs(nb - 4.0f * c); \
}

DEFINE_SUM_LAP(gff_sum_lap_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_SUM_LAP(gff_sum_lap_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_SUM_LAP(gff_sum_lap_f32, float, (raw0 / 255.0f))

__kernel void gff_gauss_h_local(__global const float* src, __global float* dst,
                                __constant float* kern, const int H, const int W) {
    __local float row[256 + 2 * GAUSS_RAD];
    int lx = get_local_id(0);
    int gx0 = get_group_id(0) * 256;
    int y = get_group_id(1);
    if (y >= H) return;
    size_t rbase = (size_t)y * W;
    const int need = 256 + 2 * GAUSS_RAD;
    for (int i = lx; i < need; i += 256) {
        int xx = reflect_idx(gx0 + i - GAUSS_RAD, W);
        row[i] = src[rbase + xx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int x = gx0 + lx;
    if (x >= W) return;
    float acc = 0.0f;
    for (int d = -GAUSS_RAD; d <= GAUSS_RAD; ++d) {
        float p = row[lx + GAUSS_RAD + d] * kern[d + GAUSS_RAD];
        acc = acc + p;
    }
    dst[rbase + x] = acc;
}

__kernel void gff_gauss_v_argmax_local(__global const float* src,
                                       __global float* best, __global int* idx,
                                       __constant float* kern, const int H, const int W,
                                       const int frame_k) {
    __local float tile[16 + 2 * GAUSS_RAD][17];
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx0 = get_group_id(0) * 16;
    int gy0 = get_group_id(1) * 16;
    int x = gx0 + lx;
    int cx = min(x, W - 1);
    const int rows_load = 16 + 2 * GAUSS_RAD;
    for (int dy = ly; dy < rows_load; dy += 16) {
        int yy = reflect_idx(gy0 + dy - GAUSS_RAD, H);
        tile[dy][lx] = src[(size_t)yy * W + cx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int y = gy0 + ly;
    if (x >= W || y >= H) return;
    float acc = 0.0f;
    for (int d = -GAUSS_RAD; d <= GAUSS_RAD; ++d) {
        float p = tile[ly + GAUSS_RAD + d][lx] * kern[d + GAUSS_RAD];
        acc = acc + p;
    }
    size_t pos = (size_t)y * W + x;
    if (frame_k == 0) {
        best[pos] = acc;
        idx[pos] = 0;
    } else if (acc > best[pos]) {
        best[pos] = acc;
        idx[pos] = frame_k;
    }
}

__kernel void gff_active_bboxes(__global const int* idx,
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

#define DEFINE_PREP_GRAY_ROI(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global float* gray, \
                   const int H, const int W, const int C, const int bgr_rev, \
                   const int y0, const int x0, const int rh, const int rw) { \
    int rx = get_global_id(0); \
    int ry = get_global_id(1); \
    if (rx >= rw || ry >= rh) return; \
    int y = y0 + ry; \
    int x = x0 + rx; \
    size_t p = (size_t)y * W + x; \
    float c0, c1, c2; \
    if (C == 1) { \
        SRC_T raw0 = src[p]; \
        c0 = c1 = c2 = (CONV_EXPR); \
    } else { \
        size_t b = p * C; \
        SRC_T raw0 = src[b + (bgr_rev ? 2 : 0)]; c0 = (CONV_EXPR); \
        raw0 = src[b + 1]; c1 = (CONV_EXPR); \
        raw0 = src[b + (bgr_rev ? 0 : 2)]; c2 = (CONV_EXPR); \
    } \
    gray[p] = (0.114f * c0 + 0.587f * c1) + 0.299f * c2; \
}

DEFINE_PREP_GRAY_ROI(prep_gray_roi_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_PREP_GRAY_ROI(prep_gray_roi_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_PREP_GRAY_ROI(prep_gray_roi_f32, float, (raw0 / 255.0f))

#define DEFINE_COPY_BASE_ROI(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global float4* dst, \
                   const int H, const int W, const int C, const int bgr_rev, \
                   const int y0, const int x0, const int rh, const int rw) { \
    int rx = get_global_id(0); \
    int ry = get_global_id(1); \
    if (rx >= rw || ry >= rh) return; \
    size_t p = (size_t)(y0 + ry) * W + (x0 + rx); \
    float c0, c1, c2; \
    if (C == 1) { \
        SRC_T raw0 = src[p]; \
        c0 = c1 = c2 = (CONV_EXPR); \
    } else { \
        size_t b = p * C; \
        SRC_T raw0 = src[b + (bgr_rev ? 2 : 0)]; c0 = (CONV_EXPR); \
        raw0 = src[b + 1]; c1 = (CONV_EXPR); \
        raw0 = src[b + (bgr_rev ? 0 : 2)]; c2 = (CONV_EXPR); \
    } \
    dst[(size_t)ry * rw + rx] = (float4)(c0, c1, c2, 0.0f); \
}

DEFINE_COPY_BASE_ROI(copy_base_roi_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_COPY_BASE_ROI(copy_base_roi_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_COPY_BASE_ROI(copy_base_roi_f32, float, (raw0 / 255.0f))

#define DEFINE_BOX3_H_SCAN(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global float4* dst, \
                   const int H, const int W, const int C, const int bgr_rev, \
                   const int y0, const int x0, const int rh, const int rw, const int rad) { \
    __local float4 scan[256]; \
    __local float4 blk[128]; \
    int lx = get_local_id(0); \
    int tile_x0 = get_group_id(0) * 128; \
    int ry = get_group_id(1); \
    if (ry >= rh) return; \
    int gy = y0 + ry; \
    int need = 128 + 2 * rad; \
    for (int i = lx; i < 256; i += 128) { \
        if (i < need) { \
            int gx = reflect_idx(x0 + tile_x0 + i - rad, W); \
            size_t p = (size_t)gy * W + gx; \
            float c0, c1, c2; \
            if (C == 1) { \
                SRC_T raw0 = src[p]; \
                c0 = c1 = c2 = (CONV_EXPR); \
            } else { \
                size_t b = p * C; \
                SRC_T raw0 = src[b + (bgr_rev ? 2 : 0)]; c0 = (CONV_EXPR); \
                raw0 = src[b + 1]; c1 = (CONV_EXPR); \
                raw0 = src[b + (bgr_rev ? 0 : 2)]; c2 = (CONV_EXPR); \
            } \
            scan[i] = (float4)(c0, c1, c2, 0.0f); \
        } else { \
            scan[i] = (float4)(0.0f); \
        } \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    float4 v0 = scan[2 * lx]; \
    float4 v1 = scan[2 * lx + 1]; \
    float4 psum = v0 + v1; \
    blk[lx] = psum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int off = 1; off < 128; off <<= 1) { \
        float4 add = (lx >= off) ? blk[lx - off] : (float4)(0.0f); \
        barrier(CLK_LOCAL_MEM_FENCE); \
        blk[lx] = blk[lx] + add; \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    float4 pref = (lx > 0) ? blk[lx - 1] : (float4)(0.0f); \
    scan[2 * lx]     = pref + v0; \
    scan[2 * lx + 1] = pref + psum; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int col = tile_x0 + lx; \
    if (col < rw) { \
        float4 hi = scan[lx + 2 * rad]; \
        float4 lo = (lx > 0) ? scan[lx - 1] : (float4)(0.0f); \
        dst[(size_t)ry * rw + col] = hi - lo; \
    } \
}

DEFINE_BOX3_H_SCAN(box3_h_scan_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_BOX3_H_SCAN(box3_h_scan_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_BOX3_H_SCAN(box3_h_scan_f32, float, (raw0 / 255.0f))

#define DEFINE_BOX3_H_SLIDE(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global float4* dst, \
                   const int H, const int W, const int C, const int bgr_rev, \
                   const int y0, const int x0, const int rh, const int rw, const int rad) { \
    int ry = get_global_id(0); \
    int c0_idx = get_global_id(1) * CHUNK; \
    if (ry >= rh || c0_idx >= rw) return; \
    int c1_idx = min(c0_idx + CHUNK, rw); \
    int gy = y0 + ry; \
    float3 s = (float3)(0.0f); \
    for (int d = -rad; d <= rad; ++d) { \
        int gx = reflect_idx(x0 + c0_idx + d, W); \
        size_t p = (size_t)gy * W + gx; \
        float a0, a1, a2; \
        if (C == 1) { SRC_T raw0 = src[p]; a0 = a1 = a2 = (CONV_EXPR); } \
        else { \
            size_t b = p * C; \
            SRC_T raw0 = src[b + (bgr_rev ? 2 : 0)]; a0 = (CONV_EXPR); \
            raw0 = src[b + 1]; a1 = (CONV_EXPR); \
            raw0 = src[b + (bgr_rev ? 0 : 2)]; a2 = (CONV_EXPR); \
        } \
        s = s + (float3)(a0, a1, a2); \
    } \
    dst[(size_t)ry * rw + c0_idx] = (float4)(s, 0.0f); \
    for (int col = c0_idx + 1; col < c1_idx; ++col) { \
        int gxr = reflect_idx(x0 + col + rad, W); \
        int gxl = reflect_idx(x0 + col - rad - 1, W); \
        size_t pr = (size_t)gy * W + gxr; \
        size_t pl = (size_t)gy * W + gxl; \
        float r0, r1, r2, l0, l1, l2; \
        if (C == 1) { \
            SRC_T raw0 = src[pr]; r0 = r1 = r2 = (CONV_EXPR); \
            raw0 = src[pl]; l0 = l1 = l2 = (CONV_EXPR); \
        } else { \
            size_t br = pr * C; \
            SRC_T raw0 = src[br + (bgr_rev ? 2 : 0)]; r0 = (CONV_EXPR); \
            raw0 = src[br + 1]; r1 = (CONV_EXPR); \
            raw0 = src[br + (bgr_rev ? 0 : 2)]; r2 = (CONV_EXPR); \
            size_t bl = pl * C; \
            raw0 = src[bl + (bgr_rev ? 2 : 0)]; l0 = (CONV_EXPR); \
            raw0 = src[bl + 1]; l1 = (CONV_EXPR); \
            raw0 = src[bl + (bgr_rev ? 0 : 2)]; l2 = (CONV_EXPR); \
        } \
        s = (s + (float3)(r0, r1, r2)) - (float3)(l0, l1, l2); \
        dst[(size_t)ry * rw + col] = (float4)(s, 0.0f); \
    } \
}

DEFINE_BOX3_H_SLIDE(box3_h_slide_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_BOX3_H_SLIDE(box3_h_slide_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_BOX3_H_SLIDE(box3_h_slide_f32, float, (raw0 / 255.0f))

__kernel void box3_v_band(__global const float4* hsrc, __global float4* dst,
                          const int H, const int hy0,
                          const int by0, const int bh, const int bw,
                          const int rad, const float inv) {
    int col = get_global_id(0);
    int r0 = get_global_id(1) * CHUNK;
    if (col >= bw || r0 >= bh) return;
    int r1 = min(r0 + CHUNK, bh);
    float3 s = (float3)(0.0f);
    for (int d = -rad; d <= rad; ++d) {
        int yy = reflect_idx(by0 + r0 + d, H) - hy0;
        s = s + hsrc[(size_t)yy * bw + col].xyz;
    }
    dst[(size_t)r0 * bw + col] = (float4)(s * inv, 0.0f);
    for (int r = r0 + 1; r < r1; ++r) {
        int yr = reflect_idx(by0 + r + rad, H) - hy0;
        int yl = reflect_idx(by0 + r - rad - 1, H) - hy0;
        s = (s + hsrc[(size_t)yr * bw + col].xyz) - hsrc[(size_t)yl * bw + col].xyz;
        dst[(size_t)r * bw + col] = (float4)(s * inv, 0.0f);
    }
}

__kernel void gf4_h_scan(__global const float* gray, __global const int* idx,
                         __global float4* dst,
                         const int H, const int W,
                         const int y0, const int x0, const int rh, const int rw,
                         const int rad, const int frame_k) {
    __local float4 scan[256];
    __local float4 blk[128];
    int lx = get_local_id(0);
    int tile_x0 = get_group_id(0) * 128;
    int ry = get_group_id(1);
    if (ry >= rh) return;
    int gy = y0 + ry;
    int need = 128 + 2 * rad;
    for (int i = lx; i < 256; i += 128) {
        if (i < need) {
            int gx = reflect101_idx(x0 + tile_x0 + i - rad, W);
            size_t pos = (size_t)gy * W + gx;
            float iv = gray[pos];
            float pv = (idx[pos] == frame_k) ? 1.0f : 0.0f;
            scan[i] = (float4)(iv, pv, iv * pv, iv * iv);
        } else {
            scan[i] = (float4)(0.0f);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 v0 = scan[2 * lx];
    float4 v1 = scan[2 * lx + 1];
    float4 psum = v0 + v1;
    blk[lx] = psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 1; off < 128; off <<= 1) {
        float4 add = (lx >= off) ? blk[lx - off] : (float4)(0.0f);
        barrier(CLK_LOCAL_MEM_FENCE);
        blk[lx] = blk[lx] + add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float4 pref = (lx > 0) ? blk[lx - 1] : (float4)(0.0f);
    scan[2 * lx]     = pref + v0;
    scan[2 * lx + 1] = pref + psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    int col = tile_x0 + lx;
    if (col < rw) {
        float4 hi = scan[lx + 2 * rad];
        float4 lo = (lx > 0) ? scan[lx - 1] : (float4)(0.0f);
        dst[(size_t)ry * rw + col] = hi - lo;
    }
}

__kernel void gf4_v_ab_band(__global const float4* hsrc, __global float2* ab,
                            const int H, const int hy0,
                            const int ay0, const int ah, const int aw,
                            const int rad, const float eps) {
    int col = get_global_id(0);
    int r0 = get_global_id(1) * CHUNK;
    if (col >= aw || r0 >= ah) return;
    int r1 = min(r0 + CHUNK, ah);
    float inv = 1.0f / (float)((2 * rad + 1) * (2 * rad + 1));
    float4 s = (float4)(0.0f);
    for (int d = -rad; d <= rad; ++d) {
        int yy = reflect101_idx(ay0 + r0 + d, H) - hy0;
        s = s + hsrc[(size_t)yy * aw + col];
    }
    for (int r = r0; r < r1; ++r) {
        if (r > r0) {
            int yr = reflect101_idx(ay0 + r + rad, H) - hy0;
            int yl = reflect101_idx(ay0 + r - rad - 1, H) - hy0;
            s = (s + hsrc[(size_t)yr * aw + col]) - hsrc[(size_t)yl * aw + col];
        }
        if (s.y <= 0.0f) {
            ab[(size_t)r * aw + col] = (float2)(0.0f, 0.0f);
        } else {
            float4 m = s * inv;
            float cov = m.z - m.x * m.y;
            float var = m.w - m.x * m.x;
            float a = cov / (var + eps);
            float b = m.y - a * m.x;
            ab[(size_t)r * aw + col] = (float2)(a, b);
        }
    }
}

__kernel void gf2_h_scan(__global const float2* ab, __global float2* dst,
                         const int W,
                         const int ax0, const int ax1, const int aw,
                         const int bx0, const int bw, const int ah, const int rad) {
    __local float2 scan[256];
    __local float2 blk[128];
    int lx = get_local_id(0);
    int tile_x0 = get_group_id(0) * 128;
    int ry = get_group_id(1);
    if (ry >= ah) return;
    int need = 128 + 2 * rad;
    size_t arow = (size_t)ry * aw;
    for (int i = lx; i < 256; i += 128) {
        if (i < need) {
            int gx = reflect101_idx(bx0 + tile_x0 + i - rad, W);
            scan[i] = (gx >= ax0 && gx < ax1) ? ab[arow + (gx - ax0)] : (float2)(0.0f, 0.0f);
        } else {
            scan[i] = (float2)(0.0f, 0.0f);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float2 v0 = scan[2 * lx];
    float2 v1 = scan[2 * lx + 1];
    float2 psum = v0 + v1;
    blk[lx] = psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = 1; off < 128; off <<= 1) {
        float2 add = (lx >= off) ? blk[lx - off] : (float2)(0.0f, 0.0f);
        barrier(CLK_LOCAL_MEM_FENCE);
        blk[lx] = blk[lx] + add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float2 pref = (lx > 0) ? blk[lx - 1] : (float2)(0.0f, 0.0f);
    scan[2 * lx]     = pref + v0;
    scan[2 * lx + 1] = pref + psum;
    barrier(CLK_LOCAL_MEM_FENCE);
    int col = tile_x0 + lx;
    if (col < bw) {
        float2 hi = scan[lx + 2 * rad];
        float2 lo = (lx > 0) ? scan[lx - 1] : (float2)(0.0f, 0.0f);
        dst[(size_t)ry * bw + col] = hi - lo;
    }
}

__kernel void gf2_v_accum_band(__global const float2* h2,
                               __global const float* gray,
                               __global const float4* base_band,
                               __global float4* acc_b,
                               const int H, const int W,
                               const int ay0, const int ay1,
                               const int by0, const int bx0, const int bh, const int bw,
                               const int rad) {
    int col = get_global_id(0);
    int r0 = get_global_id(1) * CHUNK;
    if (col >= bw || r0 >= bh) return;
    int r1 = min(r0 + CHUNK, bh);
    float inv = 1.0f / (float)((2 * rad + 1) * (2 * rad + 1));
    float2 s = (float2)(0.0f, 0.0f);
    for (int d = -rad; d <= rad; ++d) {
        int yy = reflect101_idx(by0 + r0 + d, H);
        if (yy >= ay0 && yy < ay1) {
            s = s + h2[(size_t)(yy - ay0) * bw + col];
        }
    }
    int gx = bx0 + col;
    for (int r = r0; r < r1; ++r) {
        if (r > r0) {
            int yr = reflect101_idx(by0 + r + rad, H);
            int yl = reflect101_idx(by0 + r - rad - 1, H);
            float2 vr = (yr >= ay0 && yr < ay1) ? h2[(size_t)(yr - ay0) * bw + col] : (float2)(0.0f, 0.0f);
            float2 vl = (yl >= ay0 && yl < ay1) ? h2[(size_t)(yl - ay0) * bw + col] : (float2)(0.0f, 0.0f);
            s = (s + vr) - vl;
        }
        if (s.x != 0.0f || s.y != 0.0f) {
            size_t gp = (size_t)(by0 + r) * W + gx;
            float w = (s.x * inv) * gray[gp] + (s.y * inv);
            float4 ba = base_band[(size_t)r * bw + col];
            float4 prev = acc_b[gp];
            acc_b[gp] = (float4)(prev.x + ba.x * w,
                                 prev.y + ba.y * w,
                                 prev.z + ba.z * w,
                                 prev.w + w);
        }
    }
}

// Fused local-memory r=7 (15x15) stage-1 guided filter on ROI with per-tile mean
// shift and zero-p early skip.
__kernel void gf4_r7_local_roi(__global const float* gray, __global const int* idx,
                               __global float2* ab,
                               const int H, const int W,
                               const int ay0, const int ax0, const int ah, const int aw,
                               const float eps, const int frame_k) {
    __local float  tI[30][30];
    __local uchar  tP[30][30];
    __local float4 hsum[30][16];
    __local uint   tile_has_p;

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int tid = ly * 16 + lx;
    int rx0 = get_group_id(0) * 16;
    int ry0 = get_group_id(1) * 16;
    if (tid == 0) tile_has_p = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint local_any = 0u;
    for (int i = tid; i < 900; i += 256) {
        int dy = i / 30;
        int dx = i - dy * 30;
        int yy = reflect101_idx(ay0 + ry0 + dy - 7, H);
        int xx = reflect101_idx(ax0 + rx0 + dx - 7, W);
        size_t pos = (size_t)yy * W + xx;
        tI[dy][dx] = gray[pos];
        uchar p = (idx[pos] == frame_k) ? (uchar)1 : (uchar)0;
        tP[dy][dx] = p;
        local_any |= (uint)p;
    }
    if (local_any) tile_has_p = 1u;
    barrier(CLK_LOCAL_MEM_FENCE);

    int rx = rx0 + lx;
    int ry = ry0 + ly;
    if (tile_has_p == 0u) {
        if (rx < aw && ry < ah) {
            ab[(size_t)ry * aw + rx] = (float2)(0.0f, 0.0f);
        }
        return;
    }

    float c = tI[15][15];

    for (int i = tid; i < 480; i += 256) {
        int hy = i >> 4;
        int hx = i & 15;
        float sd = 0.0f, sp = 0.0f, sdp = 0.0f, sd2 = 0.0f;
        for (int k = 0; k < 15; ++k) {
            float d = tI[hy][hx + k] - c;
            float p = (float)tP[hy][hx + k];
            sd  = sd  + d;
            sp  = sp  + p;
            sdp = sdp + d * p;
            sd2 = sd2 + d * d;
        }
        hsum[hy][hx] = (float4)(sd, sp, sdp, sd2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (rx >= aw || ry >= ah) return;

    float4 s = (float4)(0.0f);
    for (int k = 0; k < 15; ++k) {
        s = s + hsum[ly + k][lx];
    }
    if (s.y <= 0.0f) {
        ab[(size_t)ry * aw + rx] = (float2)(0.0f, 0.0f);
        return;
    }
    const float inv = 1.0f / 225.0f;
    float md  = s.x * inv;
    float mp  = s.y * inv;
    float mdp = s.z * inv;
    float md2 = s.w * inv;
    float cov = mdp - md * mp;
    float var = md2 - md * md;
    float a = cov / (var + eps);
    float mI = c + md;
    float b = mp - a * mI;
    ab[(size_t)ry * aw + rx] = (float2)(a, b);
}

#define DEFINE_GF2_R7_ACCUM_ROI(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const float2* ab, \
                   __global const SRC_T* src, \
                   __global const float* gray, \
                   __global const float4* base_band, \
                   __global float4* acc_d, \
                   const int H, const int W, const int C, const int bgr_rev, \
                   const int ay0, const int ay1, const int ax0, const int ax1, const int aw, \
                   const int by0, const int bx0, const int bw, \
                   const int dy0, const int dx0, const int dh, const int dw) { \
    __local float2 tAB[30][30]; \
    __local float2 hsum[30][16]; \
    __local uint   tile_has_ab; \
    int lx = get_local_id(0); \
    int ly = get_local_id(1); \
    int tid = ly * 16 + lx; \
    int rx0 = get_group_id(0) * 16; \
    int ry0 = get_group_id(1) * 16; \
    if (tid == 0) tile_has_ab = 0u; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    uint local_any = 0u; \
    for (int i = tid; i < 900; i += 256) { \
        int dy = i / 30; \
        int dx = i - dy * 30; \
        int yy = reflect101_idx(dy0 + ry0 + dy - 7, H); \
        int xx = reflect101_idx(dx0 + rx0 + dx - 7, W); \
        float2 v = (yy >= ay0 && yy < ay1 && xx >= ax0 && xx < ax1) \
                   ? ab[(size_t)(yy - ay0) * aw + (xx - ax0)] \
                   : (float2)(0.0f, 0.0f); \
        tAB[dy][dx] = v; \
        if (v.x != 0.0f || v.y != 0.0f) local_any = 1u; \
    } \
    if (local_any) tile_has_ab = 1u; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    if (tile_has_ab == 0u) return; \
    for (int i = tid; i < 480; i += 256) { \
        int hy = i >> 4; \
        int hx = i & 15; \
        float2 s = (float2)(0.0f, 0.0f); \
        for (int k = 0; k < 15; ++k) { \
            s = s + tAB[hy][hx + k]; \
        } \
        hsum[hy][hx] = s; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int rx = rx0 + lx; \
    int ry = ry0 + ly; \
    if (rx >= dw || ry >= dh) return; \
    float2 s = (float2)(0.0f, 0.0f); \
    for (int k = 0; k < 15; ++k) { \
        s = s + hsum[ly + k][lx]; \
    } \
    if (s.x == 0.0f && s.y == 0.0f) return; \
    const float inv = 1.0f / 225.0f; \
    int gy = dy0 + ry; \
    int gx = dx0 + rx; \
    size_t gp = (size_t)gy * W + gx; \
    float w = (s.x * inv) * gray[gp] + (s.y * inv); \
    float c0, c1, c2; \
    if (C == 1) { \
        SRC_T raw0 = src[gp]; \
        c0 = c1 = c2 = (CONV_EXPR); \
    } else { \
        size_t b = gp * C; \
        SRC_T raw0 = src[b + (bgr_rev ? 2 : 0)]; c0 = (CONV_EXPR); \
        raw0 = src[b + 1]; c1 = (CONV_EXPR); \
        raw0 = src[b + (bgr_rev ? 0 : 2)]; c2 = (CONV_EXPR); \
    } \
    float4 ba = base_band[(size_t)(gy - by0) * bw + (gx - bx0)]; \
    float4 prev = acc_d[gp]; \
    acc_d[gp] = (float4)(prev.x + (c0 - ba.x) * w, \
                         prev.y + (c1 - ba.y) * w, \
                         prev.z + (c2 - ba.z) * w, \
                         prev.w + w); \
}

DEFINE_GF2_R7_ACCUM_ROI(gf2_r7_accum_u8, uchar, ((float)raw0 / 255.0f))
DEFINE_GF2_R7_ACCUM_ROI(gf2_r7_accum_u16, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_GF2_R7_ACCUM_ROI(gf2_r7_accum_f32, float, (raw0 / 255.0f))

__kernel void gff_reconstruct_u8(__global const float4* acc_b,
                                 __global const float4* acc_d,
                                 __global uchar* out, const int start, const int count) {
    int i = get_global_id(0);
    if (i >= count) return;
    int p = start + i;
    float4 ab = acc_b[p];
    float4 ad = acc_d[p];
    float bden = ab.w < 1e-6f ? 1e-6f : ab.w;
    float dden = ad.w < 1e-6f ? 1e-6f : ad.w;
    float3 rgb = ((ab.xyz / bden) + (ad.xyz / dden)) * 255.0f;
    rgb = clamp(rgb, 0.0f, 255.0f);
    size_t o = (size_t)i * 3;
    out[o]     = (uchar)rgb.x;
    out[o + 1] = (uchar)rgb.y;
    out[o + 2] = (uchar)rgb.z;
}

__kernel void gff_reconstruct_f32(__global const float4* acc_b,
                                  __global const float4* acc_d,
                                  __global float* out, const int start, const int count) {
    int i = get_global_id(0);
    if (i >= count) return;
    int p = start + i;
    float4 ab = acc_b[p];
    float4 ad = acc_d[p];
    float bden = ab.w < 1e-6f ? 1e-6f : ab.w;
    float dden = ad.w < 1e-6f ? 1e-6f : ad.w;
    float3 rgb = ((ab.xyz / bden) + (ad.xyz / dden)) * 255.0f;
    rgb = clamp(rgb, 0.0f, 255.0f);
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


def _reflect_span(lo, hi, rad, n):
    """Exact [r_lo, r_hi) covering reflect_idx(y + d, n) for y in [lo, hi), |d| <= rad."""
    if rad <= 0 or n <= 1:
        return lo, hi
    if n <= 2 * rad + 1:
        return 0, n
    r_lo = max(0, lo - rad)
    r_hi = min(n, hi + rad)
    if lo < rad:
        # reflect(-1) = 0, reflect(lo - rad) = rad - lo - 1
        r_hi = max(r_hi, min(n, rad - lo))
    if hi > n - rad:
        # reflect(hi - 1 + rad) = 2*n - 1 - (hi - 1 + rad) = 2*n - hi - rad
        r_lo = min(r_lo, max(0, 2 * n - hi - rad))
    return r_lo, r_hi


def _reflect101_span(lo, hi, rad, n):
    """Exact [r_lo, r_hi) covering reflect101_idx(y + d, n) for y in [lo, hi), |d| <= rad."""
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
    """Return (a0, a1, q0, q1) matching exact two-pass guided-filter support."""
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
    'gff_sum_lap_u8', 'gff_sum_lap_u16', 'gff_sum_lap_f32',
    'gff_gauss_h_local', 'gff_gauss_v_argmax_local',
    'gff_active_bboxes',
    'prep_gray_roi_u8', 'prep_gray_roi_u16', 'prep_gray_roi_f32',
    'copy_base_roi_u8', 'copy_base_roi_u16', 'copy_base_roi_f32',
    'box3_h_scan_u8', 'box3_h_scan_u16', 'box3_h_scan_f32',
    'box3_h_slide_u8', 'box3_h_slide_u16', 'box3_h_slide_f32',
    'box3_v_band',
    'gf4_h_scan', 'gf4_v_ab_band', 'gf2_h_scan', 'gf2_v_accum_band',
    'gf4_r7_local_roi',
    'gf2_r7_accum_u8', 'gf2_r7_accum_u16', 'gf2_r7_accum_f32',
    'gff_reconstruct_u8', 'gff_reconstruct_f32',
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
            cl, device, ctx, _GFF_K,
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


class GpuGff:
    """Device-resident OpenCL Guided Filter fusion with band-streamed full-frame Pass 2."""

    def __init__(self):
        from fusion_methods._openfocus_gpu import (
            GpuError, _device, _get_transfer_queue, opencl_device_name,
        )
        try:
            import pyopencl as cl
            device = _device(cl)
            if device is None:
                raise GpuError('No available OpenCL GPU')
            if int(device.max_work_group_size) < 256 or int(device.local_mem_size) < 16384:
                raise GpuError('OpenCL GPU lacks required 256-thread workgroups or 16 KiB local memory')
            self.cl = cl
            self.device = device
            self.device_name = opencl_device_name(device)
            self.mf = cl.mem_flags
            self.ctx, self.q, self.k = _get_or_build_program(cl, device)
            self.tq = _get_transfer_queue(cl, device, self.ctx)
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU Guided Filter initialization failed') from exc

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

    def fuse(self, frames, img_resize, average_filter_size,
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
                    frames, img_resize, average_filter_size, should_cancel,
                    progress if progress_callback is not None else None)
            except CallbackFailure as exc:
                self._drain_queue()
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                _invalidate_program_cache(getattr(self, 'ctx', None))
                self._drain_queue()
                raise GpuError('OpenCL GPU Guided Filter execution failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.k = None
            self.tq = None
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, img_resize, average_filter_size,
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
            raise ValueError('No image data was loaded')
        first_shape = frames[0].shape
        if img_resize is not None:
            height, width = int(img_resize[1]), int(img_resize[0])
        else:
            height, width = int(first_shape[0]), int(first_shape[1])
        if min(height, width) < 1:
            raise GpuError('Guided Filter GPU requires non-empty frames')

        shape = (height, width, 3)
        n = height * width
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
            raise GpuError(f'Guided Filter GPU staging exceeds the host memory budget '
                           f'({staging >> 20} MiB needed, {host_budget >> 20} MiB available)')

        rad_avg = int(average_filter_size) // 2
        band_h = min(height, _BAND_H)
        h4_rows = min(height, band_h + 2 * max(2 * _DEFAULT_R1, rad_avg))
        s1_rows = min(height, band_h + 2 * _DEFAULT_R1)
        band_scratch = (
            h4_rows * width * 16   # h4_buf (float4)
            + s1_rows * width * 8  # ab_buf (float2)
            + s1_rows * width * 8  # h2_buf (float2)
            + band_h * width * 16  # base_band (float4)
        )
        # Pass 1 peak: idx(4n) + best(4n) + t1(4n) + t2(4n) + src_bytes
        # Pass 2 peak: idx(4n) + acc_b(16n) + acc_d(16n) + gray(4n) + band_scratch + src_bytes
        pass1_need = 16 * n + src_bytes + 5 * count * 4 + 16 * 1024 * 1024
        pass2_need = 40 * n + band_scratch + max(src_bytes, readback_bytes) + 16 * 1024 * 1024
        need = max(pass1_need, pass2_need)

        unified_memory = self.device.host_unified_memory
        limit = int(self.device.global_mem_size) // 2
        if unified_memory:
            limit = min(limit, host_budget - staging)
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError('Guided Filter GPU free memory is unknown')
            snapshot = replace(snapshot, free_vram=memory[0], total_vram=memory[1])
            limit = min(limit, resource_policy.vram_budget(snapshot))
        single = int(self.device.max_mem_alloc_size)
        largest = max(16 * n, h4_rows * width * 16, src_bytes, readback_bytes)
        if need > limit or largest > single:
            raise GpuError('Guided Filter GPU workspace exceeds the device budget')

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

        src_bufs = ()
        idx_buf = best_buf = t1_buf = t2_buf = gkern_buf = None
        wins_buf = ymin_buf = ymax_buf = xmin_buf = xmax_buf = None
        acc_b = acc_d = gray_buf = h4_buf = ab_buf = h2_buf = base_band = dst_buf = None
        uploader = prepared_iter = None
        in_flight = False
        cancel_fn = lambda: _check_cancel(should_cancel)

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
            if img_resize is not None:
                image = _ensure_color_image(np.ascontiguousarray(raw))
                del raw
                if image.dtype == np.uint16:
                    image = image.astype(np.float32) * (255.0 / 65535.0)
                elif image.dtype == np.float64:
                    image = image.astype(np.float32)
                host_src = np.ascontiguousarray(cv2.resize(image, img_resize), dtype=np.float32)
                del image
                return host_src, 3, 'f32', 0
            if raw.ndim == 2:
                channels = 1
            elif raw.ndim == 3 and raw.shape[2] in (3, 4):
                channels = int(raw.shape[2])
            else:
                raise ValueError(f'Unsupported image shape: {raw.shape}')
            if raw.shape[:2] != (height, width):
                raise ValueError('Guided Filter requires frames with matching shapes')
            dtype = np.float32 if raw.dtype == np.float64 else raw.dtype
            if dtype not in (np.uint8, np.uint16, np.float32):
                raise GpuError(f'Unsupported GPU pixel dtype: {raw.dtype}')
            host_src, bgr_rev = _prepare_gpu_upload(raw, dtype)
            del raw
            return host_src, channels, _upload_kind(dtype), (1 if bgr_rev else 0)

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
            idx_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            best_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            t1_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            t2_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            gkern = cv2.getGaussianKernel(_GAUSS_KSIZE, _DEFAULT_SIGMA_R, cv2.CV_32F).ravel().astype(np.float32)
            gkern_buf = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=gkern)

            grid_16 = (_round_up(width, 16), _round_up(height, 16))
            grid_gh = (_round_up(width, 256), height)
            local_16 = (16, 16)
            local_256 = (256, 1)

            # Pass 1: Fused sum+Laplacian, tiled local-mem separable Gaussian + online argmax.
            prepared_iter = _iter_prefetched(
                frames, prepare_upload, enabled=prefetch_slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, prefetch_slots),
            )
            slot_events = [None, None]
            slot_hosts = [None, None]
            for index, (host_src, channels, kind, bgr_rev) in prepared_iter:
                slot = index & 1 if ping_pong else 0
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
                        slot, src_buf, host_src,
                        wait_for=_event_wait_list(prev_ev), cancel=cancel_fn,
                    )
                    wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                else:
                    upload = self.cl.enqueue_copy(self.q, src_buf, async_upload_view(host_src), is_blocking=False)
                    wait_upload = None

                src_done = self._run(
                    self.k[f'gff_sum_lap_{kind}'], grid_16,
                    src_buf, t1_buf, np.int32(height), np.int32(width),
                    np.int32(channels), np.int32(bgr_rev),
                    lsize=local_16, wait_for=wait_upload,
                )
                self._run(
                    self.k['gff_gauss_h_local'], grid_gh,
                    t1_buf, t2_buf, gkern_buf, np.int32(height), np.int32(width),
                    lsize=local_256,
                )
                self._run(
                    self.k['gff_gauss_v_argmax_local'], grid_16,
                    t2_buf, best_buf, idx_buf, gkern_buf,
                    np.int32(height), np.int32(width), np.int32(index),
                    lsize=local_16,
                )
                if ping_pong and src_done is not None and hasattr(src_done, 'wait'):
                    if hasattr(self.q, 'flush'):
                        self.q.flush()
                    slot_events[slot] = src_done
                    slot_hosts[slot] = (host_src, upload)
                else:
                    self.q.finish()
                    in_flight = False
                    del host_src, upload
                if progress_callback is not None:
                    progress_callback(index + 1, total_steps)

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
                self.k['gff_active_bboxes'], (width, height),
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
            _release_buffers(best_buf, t1_buf, t2_buf, gkern_buf,
                             wins_buf, ymin_buf, ymax_buf, xmin_buf, xmax_buf)
            best_buf = t1_buf = t2_buf = gkern_buf = None
            wins_buf = ymin_buf = ymax_buf = xmin_buf = xmax_buf = None

            # Pass 2: Band-streamed, zero-transpose, ROI-restricted guided-filter accumulation.
            acc_b = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            acc_d = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            gray_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            h4_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, h4_rows * width * 16))
            ab_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, s1_rows * width * 8))
            h2_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, s1_rows * width * 8))
            base_band = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, band_h * width * 16))

            zero_f32 = np.float32(0.0)
            in_flight = True
            self.cl.enqueue_fill_buffer(self.q, acc_b, zero_f32, 0, 16 * n)
            self.cl.enqueue_fill_buffer(self.q, acc_d, zero_f32, 0, 16 * n)

            inv_avg = np.float32(1.0 / ((2 * rad_avg + 1) * (2 * rad_avg + 1)))
            local_128 = (128, 1)

            active_indices = [idx for idx in range(count) if wins[idx]]
            active_frames = [frames[idx] for idx in active_indices]
            active_slots = min(prefetch_slots, max(0, len(active_indices) - 1)) if active_indices else 0
            prepared_iter = _iter_prefetched(
                active_frames, prepare_upload, enabled=active_slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, active_slots),
            )
            slot_events = [None, None]
            slot_hosts = [None, None]
            next_progress_idx = 0

            for pos, (host_src, channels, kind, bgr_rev) in prepared_iter:
                index = active_indices[pos]
                while next_progress_idx < index:
                    if progress_callback is not None:
                        progress_callback(count + next_progress_idx + 1, total_steps)
                    next_progress_idx += 1

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
                        slot, src_buf, host_src,
                        wait_for=_event_wait_list(prev_ev), cancel=cancel_fn,
                    )
                    wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                else:
                    upload = self.cl.enqueue_copy(self.q, src_buf, async_upload_view(host_src), is_blocking=False)
                    wait_upload = None

                y0_k, y1_k = int(ymin[index]), int(ymax[index])
                x0_k, x1_k = int(xmin[index]), int(xmax[index])

                ay0, ay1, by0, by1 = _gf_active_bounds(y0_k, y1_k, _DEFAULT_R1, height)
                ax0, ax1, bx0, bx1 = _gf_active_bounds(x0_k, x1_k, _DEFAULT_R1, width)
                day0, day1, dy0, dy1 = _gf_active_bounds(y0_k, y1_k, _DEFAULT_R2, height)
                dax0, dax1, dx0, dx1 = _gf_active_bounds(x0_k, x1_k, _DEFAULT_R2, width)

                # Gray buffer only needed across the Stage-1 r=45 input span.
                gy0, gy1 = _reflect101_span(ay0, ay1, _DEFAULT_R1, height)
                gx0, gx1 = _reflect101_span(ax0, ax1, _DEFAULT_R1, width)
                self._run(
                    self.k[f'prep_gray_roi_{kind}'], (gx1 - gx0, gy1 - gy0),
                    src_buf, gray_buf, np.int32(height), np.int32(width),
                    np.int32(channels), np.int32(bgr_rev),
                    np.int32(gy0), np.int32(gx0), np.int32(gy1 - gy0), np.int32(gx1 - gx0),
                    wait_for=wait_upload,
                )

                aw = ax1 - ax0
                bw = bx1 - bx0
                daw = dax1 - dax0
                dw = dx1 - dx0
                last_src_ev = None

                for b_y0 in range(by0, by1, band_h):
                    b_y1 = min(by1, b_y0 + band_h)
                    bh = b_y1 - b_y0

                    # 1. Base layer box blur on [b_y0:b_y1, bx0:bx1] -> base_band.
                    if rad_avg > 0:
                        hy0_b, hy1_b = _reflect_span(b_y0, b_y1, rad_avg, height)
                        hh_b = hy1_b - hy0_b
                        if rad_avg <= _SCAN_MAX_RAD:
                            last_src_ev = self._run(
                                self.k[f'box3_h_scan_{kind}'], (_round_up(bw, 128), hh_b),
                                src_buf, h4_buf, np.int32(height), np.int32(width),
                                np.int32(channels), np.int32(bgr_rev),
                                np.int32(hy0_b), np.int32(bx0), np.int32(hh_b), np.int32(bw),
                                np.int32(rad_avg), lsize=local_128,
                            )
                        else:
                            last_src_ev = self._run(
                                self.k[f'box3_h_slide_{kind}'], (hh_b, _cdiv(bw, _CHUNK)),
                                src_buf, h4_buf, np.int32(height), np.int32(width),
                                np.int32(channels), np.int32(bgr_rev),
                                np.int32(hy0_b), np.int32(bx0), np.int32(hh_b), np.int32(bw),
                                np.int32(rad_avg),
                            )
                        self._run(
                            self.k['box3_v_band'], (bw, _cdiv(bh, _CHUNK)),
                            h4_buf, base_band, np.int32(height), np.int32(hy0_b),
                            np.int32(b_y0), np.int32(bh), np.int32(bw),
                            np.int32(rad_avg), inv_avg,
                        )
                    else:
                        last_src_ev = self._run(
                            self.k[f'copy_base_roi_{kind}'], (bw, bh),
                            src_buf, base_band, np.int32(height), np.int32(width),
                            np.int32(channels), np.int32(bgr_rev),
                            np.int32(b_y0), np.int32(bx0), np.int32(bh), np.int32(bw),
                        )

                    # 2. Base layer guided filter (r=45, eps=0.3) -> accumulate acc_b.
                    s1_y0, s1_y1 = _reflect101_span(b_y0, b_y1, _DEFAULT_R1, height)
                    ab_y0 = max(ay0, s1_y0)
                    ab_y1 = min(ay1, s1_y1)
                    if ab_y0 < ab_y1:
                        ab_h = ab_y1 - ab_y0
                        h4_y0, h4_y1 = _reflect101_span(ab_y0, ab_y1, _DEFAULT_R1, height)
                        h4_h = h4_y1 - h4_y0
                        self._run(
                            self.k['gf4_h_scan'], (_round_up(aw, 128), h4_h),
                            gray_buf, idx_buf, h4_buf,
                            np.int32(height), np.int32(width),
                            np.int32(h4_y0), np.int32(ax0), np.int32(h4_h), np.int32(aw),
                            np.int32(_DEFAULT_R1), np.int32(index), lsize=local_128,
                        )
                        self._run(
                            self.k['gf4_v_ab_band'], (aw, _cdiv(ab_h, _CHUNK)),
                            h4_buf, ab_buf, np.int32(height), np.int32(h4_y0),
                            np.int32(ab_y0), np.int32(ab_h), np.int32(aw),
                            np.int32(_DEFAULT_R1), np.float32(_DEFAULT_EPS1),
                        )
                        self._run(
                            self.k['gf2_h_scan'], (_round_up(bw, 128), ab_h),
                            ab_buf, h2_buf, np.int32(width),
                            np.int32(ax0), np.int32(ax1), np.int32(aw),
                            np.int32(bx0), np.int32(bw), np.int32(ab_h),
                            np.int32(_DEFAULT_R1), lsize=local_128,
                        )
                        self._run(
                            self.k['gf2_v_accum_band'], (bw, _cdiv(bh, _CHUNK)),
                            h2_buf, gray_buf, base_band, acc_b,
                            np.int32(height), np.int32(width),
                            np.int32(ab_y0), np.int32(ab_y1),
                            np.int32(b_y0), np.int32(bx0), np.int32(bh), np.int32(bw),
                            np.int32(_DEFAULT_R1),
                        )

                    # 3. Detail layer guided filter (r=7, eps=1e-5) -> accumulate acc_d.
                    db_y0 = max(b_y0, dy0)
                    db_y1 = min(b_y1, dy1)
                    if db_y0 < db_y1:
                        dh = db_y1 - db_y0
                        d1_y0, d1_y1 = _reflect101_span(db_y0, db_y1, _DEFAULT_R2, height)
                        dab_y0 = max(day0, d1_y0)
                        dab_y1 = min(day1, d1_y1)
                        if dab_y0 < dab_y1:
                            dah = dab_y1 - dab_y0
                            self._run(
                                self.k['gf4_r7_local_roi'],
                                (_round_up(daw, 16), _round_up(dah, 16)),
                                gray_buf, idx_buf, ab_buf,
                                np.int32(height), np.int32(width),
                                np.int32(dab_y0), np.int32(dax0), np.int32(dah), np.int32(daw),
                                np.float32(_DEFAULT_EPS2), np.int32(index), lsize=local_16,
                            )
                            last_src_ev = self._run(
                                self.k[f'gf2_r7_accum_{kind}'],
                                (_round_up(dw, 16), _round_up(dh, 16)),
                                ab_buf, src_buf, gray_buf, base_band, acc_d,
                                np.int32(height), np.int32(width),
                                np.int32(channels), np.int32(bgr_rev),
                                np.int32(dab_y0), np.int32(dab_y1),
                                np.int32(dax0), np.int32(dax1), np.int32(daw),
                                np.int32(b_y0), np.int32(bx0), np.int32(bw),
                                np.int32(db_y0), np.int32(dx0), np.int32(dh), np.int32(dw),
                                lsize=local_16,
                            )

                if ping_pong and last_src_ev is not None and hasattr(last_src_ev, 'wait'):
                    if hasattr(self.q, 'flush'):
                        self.q.flush()
                    slot_events[slot] = last_src_ev
                    slot_hosts[slot] = (host_src, upload)
                else:
                    self.q.finish()
                    in_flight = False
                    del host_src, upload
                if progress_callback is not None:
                    progress_callback(count + index + 1, total_steps)
                next_progress_idx = index + 1

            while next_progress_idx < count:
                if progress_callback is not None:
                    progress_callback(count + next_progress_idx + 1, total_steps)
                next_progress_idx += 1

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
            _release_buffers(src_bufs, idx_buf, gray_buf, h4_buf, ab_buf, h2_buf, base_band)
            src_bufs = ()
            idx_buf = gray_buf = h4_buf = ab_buf = h2_buf = base_band = None

            _check_cancel(should_cancel)
            out = np.empty(shape, dtype=out_dtype)
            out_pixels = out.reshape(n, 3)
            dst_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, readback_bytes)
            chunk_pixels = max(1, readback_bytes // (3 * out_itemsize))
            recon_kern = self.k['gff_reconstruct_u8' if is_u8_out else 'gff_reconstruct_f32']
            in_flight = True
            for start in range(0, n, chunk_pixels):
                _check_cancel(should_cancel)
                cnt = min(chunk_pixels, n - start)
                self._run(recon_kern, cnt, acc_b, acc_d, dst_buf, np.int32(start), np.int32(cnt))
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
                src_bufs, idx_buf, best_buf, t1_buf, t2_buf, gkern_buf,
                wins_buf, ymin_buf, ymax_buf, xmin_buf, xmax_buf,
                acc_b, acc_d, gray_buf, h4_buf, ab_buf, h2_buf, base_band, dst_buf,
            )
