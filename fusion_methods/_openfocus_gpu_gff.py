"""OpenCL GPU fusion for Guided Filter (opt-in via use_gpu=True on gff_impl).

Ports saliency (channel sum, Laplacian, separable Gaussian + online argmax),
box-blur base/detail decomposition, two-scale guided-filter weight refinement
(r=45, eps=0.3 base; r=7, eps=1e-5 detail with local tile-mean shift), and
weighted reconstruction onto OpenCL 1.2 device kernels with single-frame
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

_DEFAULT_R1 = 45
_DEFAULT_R2 = 7
_DEFAULT_EPS1 = 0.3
_DEFAULT_EPS2 = 10e-6
_DEFAULT_SIGMA_R = 5.0
_GAUSS_KSIZE = 41
_GAUSS_RAD = _GAUSS_KSIZE // 2
_CHUNK = 64

_GFF_K = r'''
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

__kernel void gff_sum_u8(__global const uchar* src, __global float* out,
                         const int n, const int C) {
    int i = get_global_id(0);
    if (i >= n) return;
    if (C == 1) {
        float v = (float)src[i] / 255.0f;
        out[i] = (v + v) + v;
    } else {
        size_t b = (size_t)i * C;
        float c0 = (float)src[b] / 255.0f;
        float c1 = (float)src[b + 1] / 255.0f;
        float c2 = (float)src[b + 2] / 255.0f;
        out[i] = (c0 + c1) + c2;
    }
}

__kernel void gff_sum_u16(__global const ushort* src, __global float* out,
                          const int n, const int C) {
    int i = get_global_id(0);
    if (i >= n) return;
    const float scale = 255.0f / 65535.0f;
    if (C == 1) {
        float v = ((float)src[i] * scale) / 255.0f;
        out[i] = (v + v) + v;
    } else {
        size_t b = (size_t)i * C;
        float c0 = ((float)src[b] * scale) / 255.0f;
        float c1 = ((float)src[b + 1] * scale) / 255.0f;
        float c2 = ((float)src[b + 2] * scale) / 255.0f;
        out[i] = (c0 + c1) + c2;
    }
}

__kernel void gff_sum_f32(__global const float* src, __global float* out,
                          const int n, const int C) {
    int i = get_global_id(0);
    if (i >= n) return;
    if (C == 1) {
        float v = src[i] / 255.0f;
        out[i] = (v + v) + v;
    } else {
        size_t b = (size_t)i * C;
        float c0 = src[b] / 255.0f;
        float c1 = src[b + 1] / 255.0f;
        float c2 = src[b + 2] / 255.0f;
        out[i] = (c0 + c1) + c2;
    }
}

__kernel void gff_laplacian_abs(__global const float* src, __global float* dst,
                                const int H, const int W) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    int ym = (y > 0) ? (y - 1) : 0;
    int yp = (y + 1 < H) ? (y + 1) : (H - 1);
    int xm = (x > 0) ? (x - 1) : 0;
    int xp = (x + 1 < W) ? (x + 1) : (W - 1);
    float c = src[(size_t)y * W + x];
    float s = (src[(size_t)ym * W + x] + src[(size_t)yp * W + x])
            + (src[(size_t)y * W + xm] + src[(size_t)y * W + xp]);
    dst[(size_t)y * W + x] = fabs(s - 4.0f * c);
}

__kernel void gff_gauss_h(__global const float* src, __global float* dst,
                          __constant float* kern, const int rad,
                          const int H, const int W) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    size_t row = (size_t)y * W;
    float acc = 0.0f;
    if (x >= rad && x + rad < W) {
        for (int d = -rad; d <= rad; ++d) {
            float p = src[row + x + d] * kern[d + rad];
            acc = acc + p;
        }
    } else {
        for (int d = -rad; d <= rad; ++d) {
            int xx = reflect_idx(x + d, W);
            float p = src[row + xx] * kern[d + rad];
            acc = acc + p;
        }
    }
    dst[row + x] = acc;
}

__kernel void gff_gauss_v_argmax(__global const float* src,
                                 __global float* best, __global int* idx,
                                 __constant float* kern, const int rad,
                                 const int H, const int W, const int frame_k) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= W || y >= H) return;
    float acc = 0.0f;
    if (y >= rad && y + rad < H) {
        for (int d = -rad; d <= rad; ++d) {
            float p = src[(size_t)(y + d) * W + x] * kern[d + rad];
            acc = acc + p;
        }
    } else {
        for (int d = -rad; d <= rad; ++d) {
            int yy = reflect_idx(y + d, H);
            float p = src[(size_t)yy * W + x] * kern[d + rad];
            acc = acc + p;
        }
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

__kernel void gff_bincount(__global const int* idx, __global uint* wins,
                         const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    atomic_inc(&wins[idx[i]]);
}

__kernel void transpose_i32(__global const int* src, __global int* dst,
                            const int H, const int W) {
    __local int tile[16][17];
    int x = get_group_id(0) * 16 + get_local_id(0);
    int y = get_group_id(1) * 16 + get_local_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    if (x < W && y < H) tile[ly][lx] = src[(size_t)y * W + x];
    barrier(CLK_LOCAL_MEM_FENCE);
    int ox = get_group_id(1) * 16 + lx;
    int oy = get_group_id(0) * 16 + ly;
    if (ox < H && oy < W) dst[(size_t)oy * H + ox] = tile[lx][ly];
}

__kernel void transpose_f4(__global const float4* src, __global float4* dst,
                           const int H, const int W) {
    __local float4 tile[16][17];
    int x = get_group_id(0) * 16 + get_local_id(0);
    int y = get_group_id(1) * 16 + get_local_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    if (x < W && y < H) tile[ly][lx] = src[(size_t)y * W + x];
    barrier(CLK_LOCAL_MEM_FENCE);
    int ox = get_group_id(1) * 16 + lx;
    int oy = get_group_id(0) * 16 + ly;
    if (ox < H && oy < W) dst[(size_t)oy * H + ox] = tile[lx][ly];
}

__kernel void transpose_f2(__global const float2* src, __global float2* dst,
                           const int H, const int W) {
    __local float2 tile[16][17];
    int x = get_group_id(0) * 16 + get_local_id(0);
    int y = get_group_id(1) * 16 + get_local_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    if (x < W && y < H) tile[ly][lx] = src[(size_t)y * W + x];
    barrier(CLK_LOCAL_MEM_FENCE);
    int ox = get_group_id(1) * 16 + lx;
    int oy = get_group_id(0) * 16 + ly;
    if (ox < H && oy < W) dst[(size_t)oy * H + ox] = tile[lx][ly];
}

#define DEFINE_PREP(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME(__global const SRC_T* src, __global float4* bgrg, \
                   __global float4* bgrg_T, const int H, const int W, const int C) { \
    __local float4 tile[16][17]; \
    int x = get_group_id(0) * 16 + get_local_id(0); \
    int y = get_group_id(1) * 16 + get_local_id(1); \
    int lx = get_local_id(0), ly = get_local_id(1); \
    if (x < W && y < H) { \
        size_t p = (size_t)y * W + x; \
        float c0, c1, c2; \
        if (C == 1) { \
            SRC_T raw0 = src[p]; \
            c0 = c1 = c2 = (CONV_EXPR); \
        } else { \
            size_t b = p * C; \
            SRC_T raw0 = src[b]; \
            c0 = (CONV_EXPR); \
            raw0 = src[b + 1]; \
            c1 = (CONV_EXPR); \
            raw0 = src[b + 2]; \
            c2 = (CONV_EXPR); \
        } \
        float gray = (0.114f * c0 + 0.587f * c1) + 0.299f * c2; \
        float4 v = (float4)(c0, c1, c2, gray); \
        bgrg[p] = v; \
        tile[ly][lx] = v; \
    } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int ox = get_group_id(1) * 16 + lx; \
    int oy = get_group_id(0) * 16 + ly; \
    if (ox < H && oy < W) bgrg_T[(size_t)oy * H + ox] = tile[lx][ly]; \
}

DEFINE_PREP(prep_u8_f4, uchar, ((float)raw0 / 255.0f))
DEFINE_PREP(prep_u16_f4, ushort, (((float)raw0 * (255.0f / 65535.0f)) / 255.0f))
DEFINE_PREP(prep_f32_f4, float, (raw0 / 255.0f))

__kernel void box3_col_chunk(__global const float4* src, __global float4* dst,
                             const int N_row, const int N_col, const int rad,
                             const float scale) {
    int col = get_global_id(0);
    int r0 = get_global_id(1) * CHUNK;
    if (col >= N_col || r0 >= N_row) return;
    int r1 = min(r0 + CHUNK, N_row);
    float3 s = (float3)(0.0f, 0.0f, 0.0f);
    for (int d = -rad; d <= rad; ++d) {
        int rr = reflect_idx(r0 + d, N_row);
        s = s + src[(size_t)rr * N_col + col].xyz;
    }
    dst[(size_t)r0 * N_col + col] = (float4)(s * scale, 0.0f);
    for (int r = r0 + 1; r < r1; ++r) {
        int rr = reflect_idx(r + rad, N_row);
        int rl = reflect_idx(r - rad - 1, N_row);
        s = (s + src[(size_t)rr * N_col + col].xyz) - src[(size_t)rl * N_col + col].xyz;
        dst[(size_t)r * N_col + col] = (float4)(s * scale, 0.0f);
    }
}

__kernel void gf4_h_chunk(__global const float4* bgrg_T, __global const int* idx_T,
                          __global float4* dst_T, const int H, const int W,
                          const int rad, const int frame_k) {
    int y = get_global_id(0);
    int x0 = get_global_id(1) * CHUNK;
    if (y >= H || x0 >= W) return;
    int x1 = min(x0 + CHUNK, W);
    float4 s = (float4)(0.0f);
    for (int d = -rad; d <= rad; ++d) {
        int xx = reflect101_idx(x0 + d, W);
        size_t pos = (size_t)xx * H + y;
        float iv = bgrg_T[pos].w;
        float pv = (idx_T[pos] == frame_k) ? 1.0f : 0.0f;
        s = s + (float4)(iv, pv, iv * pv, iv * iv);
    }
    dst_T[(size_t)x0 * H + y] = s;
    for (int x = x0 + 1; x < x1; ++x) {
        int xr = reflect101_idx(x + rad, W);
        int xl = reflect101_idx(x - rad - 1, W);
        size_t pr = (size_t)xr * H + y;
        size_t pl = (size_t)xl * H + y;
        float ivr = bgrg_T[pr].w, ivl = bgrg_T[pl].w;
        float pvr = (idx_T[pr] == frame_k) ? 1.0f : 0.0f;
        float pvl = (idx_T[pl] == frame_k) ? 1.0f : 0.0f;
        s = (s + (float4)(ivr, pvr, ivr * pvr, ivr * ivr))
               - (float4)(ivl, pvl, ivl * pvl, ivl * ivl);
        dst_T[(size_t)x * H + y] = s;
    }
}

__kernel void gf4_v_ab_chunk(__global const float4* src, __global float2* ab,
                             const int H, const int W, const int rad, const float eps) {
    int x = get_global_id(0);
    int y0 = get_global_id(1) * CHUNK;
    if (x >= W || y0 >= H) return;
    int y1 = min(y0 + CHUNK, H);
    float inv = 1.0f / (float)((2 * rad + 1) * (2 * rad + 1));
    float4 s = (float4)(0.0f);
    for (int d = -rad; d <= rad; ++d) {
        int yy = reflect101_idx(y0 + d, H);
        s = s + src[(size_t)yy * W + x];
    }
    for (int y = y0; y < y1; ++y) {
        if (y > y0) {
            int yr = reflect101_idx(y + rad, H);
            int yl = reflect101_idx(y - rad - 1, H);
            s = (s + src[(size_t)yr * W + x]) - src[(size_t)yl * W + x];
        }
        float4 m = s * inv;
        float cov = m.z - m.x * m.y;
        float var = m.w - m.x * m.x;
        float a = cov / (var + eps);
        float b = m.y - a * m.x;
        ab[(size_t)y * W + x] = (float2)(a, b);
    }
}

__kernel void gf2_h_chunk(__global const float2* ab_T, __global float2* dst_T,
                          const int H, const int W, const int rad) {
    int y = get_global_id(0);
    int x0 = get_global_id(1) * CHUNK;
    if (y >= H || x0 >= W) return;
    int x1 = min(x0 + CHUNK, W);
    float2 s = (float2)(0.0f);
    for (int d = -rad; d <= rad; ++d) {
        int xx = reflect101_idx(x0 + d, W);
        s = s + ab_T[(size_t)xx * H + y];
    }
    dst_T[(size_t)x0 * H + y] = s;
    for (int x = x0 + 1; x < x1; ++x) {
        int xr = reflect101_idx(x + rad, W);
        int xl = reflect101_idx(x - rad - 1, W);
        s = (s + ab_T[(size_t)xr * H + y]) - ab_T[(size_t)xl * H + y];
        dst_T[(size_t)x * H + y] = s;
    }
}

__kernel void gf2_v_accum_chunk(__global const float2* src, __global const float4* bgrg,
                                __global const float4* base, __global float4* acc,
                                const int H, const int W, const int rad) {
    int x = get_global_id(0);
    int y0 = get_global_id(1) * CHUNK;
    if (x >= W || y0 >= H) return;
    int y1 = min(y0 + CHUNK, H);
    float inv = 1.0f / (float)((2 * rad + 1) * (2 * rad + 1));
    float2 s = (float2)(0.0f);
    for (int d = -rad; d <= rad; ++d) {
        int yy = reflect101_idx(y0 + d, H);
        s = s + src[(size_t)yy * W + x];
    }
    for (int y = y0; y < y1; ++y) {
        if (y > y0) {
            int yr = reflect101_idx(y + rad, H);
            int yl = reflect101_idx(y - rad - 1, H);
            s = (s + src[(size_t)yr * W + x]) - src[(size_t)yl * W + x];
        }
        size_t p = (size_t)y * W + x;
        float ma = s.x * inv;
        float mb = s.y * inv;
        float w = ma * bgrg[p].w + mb;
        float4 ba = base[p];
        float4 prev = acc[p];
        acc[p] = (float4)(prev.x + ba.x * w,
                          prev.y + ba.y * w,
                          prev.z + ba.z * w,
                          prev.w + w);
    }
}

// Fused local-memory r=7 (15x15) stage-1 guided filter with per-tile mean shift
// to avoid catastrophic cancellation in pure OpenCL 1.2 FP32 arithmetic.
__kernel void gf4_r7_local(__global const float4* bgrg, __global const int* idx,
                           __global float2* ab, const int H, const int W,
                           const float eps, const int frame_k) {
    __local float  tI[30][30];
    __local uchar  tP[30][30];
    __local float4 hsum[30][16];

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int tid = ly * 16 + lx;
    int gx0 = get_group_id(0) * 16;
    int gy0 = get_group_id(1) * 16;

    for (int i = tid; i < 900; i += 256) {
        int dy = i / 30;
        int dx = i - dy * 30;
        int yy = reflect101_idx(gy0 + dy - 7, H);
        int xx = reflect101_idx(gx0 + dx - 7, W);
        size_t pos = (size_t)yy * W + xx;
        tI[dy][dx] = bgrg[pos].w;
        tP[dy][dx] = (idx[pos] == frame_k) ? (uchar)1 : (uchar)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float c = tI[15][15];

    for (int i = tid; i < 480; i += 256) {
        int ry = i >> 4;
        int rx = i & 15;
        float sd = 0.0f, sp = 0.0f, sdp = 0.0f, sd2 = 0.0f;
        for (int k = 0; k < 15; ++k) {
            float d = tI[ry][rx + k] - c;
            float p = (float)tP[ry][rx + k];
            sd  = sd  + d;
            sp  = sp  + p;
            sdp = sdp + d * p;
            sd2 = sd2 + d * d;
        }
        hsum[ry][rx] = (float4)(sd, sp, sdp, sd2);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = gx0 + lx;
    int y = gy0 + ly;
    if (x >= W || y >= H) return;

    float4 s = (float4)(0.0f);
    for (int k = 0; k < 15; ++k) {
        s = s + hsum[ly + k][lx];
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
    ab[(size_t)y * W + x] = (float2)(a, b);
}

__kernel void gf2_r7_accum_local(__global const float2* ab, __global const float4* bgrg,
                                 __global const float4* base, __global float4* acc_d,
                                 const int H, const int W) {
    __local float2 tAB[30][30];
    __local float2 hsum[30][16];

    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int tid = ly * 16 + lx;
    int gx0 = get_group_id(0) * 16;
    int gy0 = get_group_id(1) * 16;

    for (int i = tid; i < 900; i += 256) {
        int dy = i / 30;
        int dx = i - dy * 30;
        int yy = reflect101_idx(gy0 + dy - 7, H);
        int xx = reflect101_idx(gx0 + dx - 7, W);
        tAB[dy][dx] = ab[(size_t)yy * W + xx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = tid; i < 480; i += 256) {
        int ry = i >> 4;
        int rx = i & 15;
        float2 s = (float2)(0.0f, 0.0f);
        for (int k = 0; k < 15; ++k) {
            s = s + tAB[ry][rx + k];
        }
        hsum[ry][rx] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = gx0 + lx;
    int y = gy0 + ly;
    if (x >= W || y >= H) return;

    float2 s = (float2)(0.0f, 0.0f);
    for (int k = 0; k < 15; ++k) {
        s = s + hsum[ly + k][lx];
    }
    const float inv = 1.0f / 225.0f;
    size_t pos = (size_t)y * W + x;
    float4 bg = bgrg[pos];
    float ma = s.x * inv;
    float mb = s.y * inv;
    float w = ma * bg.w + mb;
    float4 det = bg - base[pos];
    float4 prev = acc_d[pos];
    acc_d[pos] = (float4)(prev.x + det.x * w,
                          prev.y + det.y * w,
                          prev.z + det.z * w,
                          prev.w + w);
}

__kernel void gff_reconstruct_u8(__global const float4* acc_b,
                                 __global const float4* acc_d,
                                 __global uchar* out, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 ab = acc_b[i];
    float4 ad = acc_d[i];
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
                                  __global float* out, const int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 ab = acc_b[i];
    float4 ad = acc_d[i];
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
    'gff_sum_u8', 'gff_sum_u16', 'gff_sum_f32',
    'gff_laplacian_abs', 'gff_gauss_h', 'gff_gauss_v_argmax',
    'gff_bincount', 'transpose_i32', 'transpose_f4', 'transpose_f2',
    'prep_u8_f4', 'prep_u16_f4', 'prep_f32_f4',
    'box3_col_chunk', 'gf4_h_chunk', 'gf4_v_ab_chunk',
    'gf2_h_chunk', 'gf2_v_accum_chunk',
    'gf4_r7_local', 'gf2_r7_accum_local',
    'gff_reconstruct_u8', 'gff_reconstruct_f32',
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
    key = _device_cache_key(device)
    with _cache_lock:
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key:
            return _PROGRAM_CACHE[1:]
        ctx = cl.Context(devices=[device])
        q = cl.CommandQueue(ctx, device=device)
        prog = cl.Program(ctx, _GFF_K).build(
            options=['-cl-fp32-correctly-rounded-divide-sqrt'])
        kernels = {name: cl.Kernel(prog, name) for name in _KERNEL_NAMES}
        _PROGRAM_CACHE = (key, ctx, q, kernels)
        return ctx, q, kernels


def _invalidate_program_cache():
    global _PROGRAM_CACHE
    with _cache_lock:
        _PROGRAM_CACHE = None


class GpuGff:
    """Device-resident OpenCL Guided Filter fusion with single-frame streaming."""

    def __init__(self):
        from fusion_methods._openfocus_gpu import GpuError, _device, opencl_device_name
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
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU Guided Filter initialization failed') from exc

    def _run(self, kernel, gsize, *args, lsize=None):
        kernel(self.q, gsize if isinstance(gsize, tuple) else (gsize,), lsize, *args)

    def _drain_queue(self):
        q = getattr(self, 'q', None)
        if q is not None:
            try:
                q.finish()
            except Exception:
                _invalidate_program_cache()

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
                _invalidate_program_cache()
                self._drain_queue()
                raise GpuError('OpenCL GPU Guided Filter execution failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.k = None
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, img_resize, average_filter_size,
                     should_cancel, progress_callback):
        from dataclasses import replace
        from core import resource_policy
        from core.source_frames import resolve_frame, without_source_cache_growth
        from fusion_methods._openfocus_gpu import GpuError, _free_vram

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
            raise GpuError(f'Guided Filter GPU staging exceeds the host memory budget '
                           f'({staging >> 20} MiB needed, {host_budget >> 20} MiB available)')

        need = (120 * n
                + max(src_bytes, out_bytes)
                + count * 4
                + 16 * 1024 * 1024)
        limit = int(self.device.global_mem_size) // 2
        if self.device.host_unified_memory:
            limit = min(limit, host_budget - staging)
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError('Guided Filter GPU free memory is unknown')
            snapshot = replace(snapshot, free_vram=memory[0], total_vram=memory[1])
            limit = min(limit, resource_policy.vram_budget(snapshot))
        single = int(self.device.max_mem_alloc_size)
        largest = max(16 * n, src_bytes, out_bytes)
        if need > limit or largest > single:
            raise GpuError('Guided Filter GPU workspace exceeds the device budget')

        src_buf = self.cl.Buffer(self.ctx, self.mf.READ_ONLY, max(1, src_bytes))
        idx_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        best_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        t1_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        t2_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
        gkern = cv2.getGaussianKernel(_GAUSS_KSIZE, _DEFAULT_SIGMA_R, cv2.CV_32F).ravel().astype(np.float32)
        gkern_buf = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=gkern)

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
            if img_resize is not None:
                image = _ensure_color_image(np.ascontiguousarray(raw))
                del raw
                if image.dtype == np.uint16:
                    image = image.astype(np.float32) * (255.0 / 65535.0)
                elif image.dtype == np.float64:
                    image = image.astype(np.float32)
                host_src = np.ascontiguousarray(cv2.resize(image, img_resize), dtype=np.float32)
                del image
                return host_src, 3, 'f32'
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
            host_src = np.ascontiguousarray(raw, dtype=dtype)
            del raw
            return host_src, channels, _upload_kind(dtype)

        in_flight = False
        try:
            # Pass 1: Saliency map and streaming strict-`>` winner selection.
            for index in range(count):
                host_src, channels, kind = prepare_upload(index)
                in_flight = True
                self.cl.enqueue_copy(self.q, src_buf, host_src, is_blocking=False)
                self._run(self.k[f'gff_sum_{kind}'], n, src_buf, t1_buf,
                          np.int32(n), np.int32(channels))
                self._run(self.k['gff_laplacian_abs'], (width, height),
                          t1_buf, t2_buf, np.int32(height), np.int32(width))
                self._run(self.k['gff_gauss_h'], (width, height),
                          t2_buf, t1_buf, gkern_buf, np.int32(_GAUSS_RAD),
                          np.int32(height), np.int32(width))
                self._run(self.k['gff_gauss_v_argmax'], (width, height),
                          t1_buf, best_buf, idx_buf, gkern_buf, np.int32(_GAUSS_RAD),
                          np.int32(height), np.int32(width), np.int32(index))
                self.q.finish()
                in_flight = False
                del host_src
                if progress_callback is not None:
                    progress_callback(index + 1, total_steps)

            wins_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, count * 4))
            in_flight = True
            self.cl.enqueue_fill_buffer(self.q, wins_buf, np.uint32(0), 0, count * 4)
            self._run(self.k['gff_bincount'], n, idx_buf, wins_buf, np.int32(n))
            wins = np.empty(count, dtype=np.uint32)
            self.cl.enqueue_copy(self.q, wins, wins_buf, is_blocking=True)
            in_flight = False
            del best_buf, t1_buf, t2_buf, gkern_buf, wins_buf

            # Pass 2: Base/detail guided-filter weight refinement and accumulation.
            idx_T = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n))
            acc_b = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            acc_d = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            bgrg = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            bgrg_T = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            base_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            tmp4_T = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))
            tmp4 = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 16 * n))

            zero_f32 = np.float32(0.0)
            in_flight = True
            self.cl.enqueue_fill_buffer(self.q, acc_b, zero_f32, 0, 16 * n)
            self.cl.enqueue_fill_buffer(self.q, acc_d, zero_f32, 0, 16 * n)

            grid_hw = (_round16(width), _round16(height))
            grid_wh = (_round16(height), _round16(width))
            local_16 = (16, 16)
            rad_avg = int(average_filter_size) // 2
            inv_avg = np.float32(1.0 / ((2 * rad_avg + 1) * (2 * rad_avg + 1)))

            self._run(self.k['transpose_i32'], grid_hw, idx_buf, idx_T,
                      np.int32(height), np.int32(width), lsize=local_16)

            for index in range(count):
                _check_cancel(should_cancel)
                if not wins[index]:
                    if progress_callback is not None:
                        progress_callback(count + index + 1, total_steps)
                    continue
                host_src, channels, kind = prepare_upload(index)
                in_flight = True
                self.cl.enqueue_copy(self.q, src_buf, host_src, is_blocking=False)
                self._run(self.k[f'prep_{kind}_f4'], grid_hw, src_buf, bgrg, bgrg_T,
                          np.int32(height), np.int32(width), np.int32(channels), lsize=local_16)
                if rad_avg > 0:
                    self._run(self.k['box3_col_chunk'], (height, _cdiv(width, _CHUNK)),
                              bgrg_T, tmp4_T, np.int32(width), np.int32(height),
                              np.int32(rad_avg), np.float32(1.0))
                    self._run(self.k['transpose_f4'], grid_wh, tmp4_T, tmp4,
                              np.int32(width), np.int32(height), lsize=local_16)
                    self._run(self.k['box3_col_chunk'], (width, _cdiv(height, _CHUNK)),
                              tmp4, base_buf, np.int32(height), np.int32(width),
                              np.int32(rad_avg), inv_avg)
                    layer_base = base_buf
                else:
                    layer_base = bgrg

                # Base layer guided filter (r=45, eps=0.3) via coalesced chunked passes.
                self._run(self.k['gf4_h_chunk'], (height, _cdiv(width, _CHUNK)),
                          bgrg_T, idx_T, tmp4_T, np.int32(height), np.int32(width),
                          np.int32(_DEFAULT_R1), np.int32(index))
                self._run(self.k['transpose_f4'], grid_wh, tmp4_T, tmp4,
                          np.int32(width), np.int32(height), lsize=local_16)
                self._run(self.k['gf4_v_ab_chunk'], (width, _cdiv(height, _CHUNK)),
                          tmp4, tmp4_T, np.int32(height), np.int32(width),
                          np.int32(_DEFAULT_R1), np.float32(_DEFAULT_EPS1))
                self._run(self.k['transpose_f2'], grid_hw, tmp4_T, tmp4,
                          np.int32(height), np.int32(width), lsize=local_16)
                self._run(self.k['gf2_h_chunk'], (height, _cdiv(width, _CHUNK)),
                          tmp4, tmp4_T, np.int32(height), np.int32(width),
                          np.int32(_DEFAULT_R1))
                self._run(self.k['transpose_f2'], grid_wh, tmp4_T, tmp4,
                          np.int32(width), np.int32(height), lsize=local_16)
                self._run(self.k['gf2_v_accum_chunk'], (width, _cdiv(height, _CHUNK)),
                          tmp4, bgrg, layer_base, acc_b, np.int32(height), np.int32(width),
                          np.int32(_DEFAULT_R1))

                # Detail layer guided filter (r=7, eps=1e-5) via 16x16 local-memory tiles.
                self._run(self.k['gf4_r7_local'], grid_hw, bgrg, idx_buf, tmp4_T,
                          np.int32(height), np.int32(width), np.float32(_DEFAULT_EPS2),
                          np.int32(index), lsize=local_16)
                self._run(self.k['gf2_r7_accum_local'], grid_hw, tmp4_T, bgrg, layer_base,
                          acc_d, np.int32(height), np.int32(width), lsize=local_16)

                self.q.finish()
                in_flight = False
                del host_src
                if progress_callback is not None:
                    progress_callback(count + index + 1, total_steps)

            if in_flight:
                self.q.finish()
                in_flight = False
            del src_buf, idx_buf, idx_T, bgrg, bgrg_T, base_buf, tmp4_T, tmp4

            _check_cancel(should_cancel)
            out = np.empty(shape, dtype=out_dtype)
            dst_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, out_bytes)
            recon_kern = self.k['gff_reconstruct_u8' if is_u8_out else 'gff_reconstruct_f32']
            in_flight = True
            self._run(recon_kern, n, acc_b, acc_d, dst_buf, np.int32(n))
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
