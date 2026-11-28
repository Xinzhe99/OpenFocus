"""OpenCL GPU fusion for DTCWT (Dual-Tree Complex Wavelet Transform) fusion.

Ports luminance/channel extraction, odd-size even-extension, near_sym_a level-0
and qshift_a deeper-level complex wavelet forward/inverse transforms, fused
horizontal-filter + q2c + activity/selection kernels, fused inverse c2q
kernels, 3x3 reflect max-activity streaming argmax, O(1) 3x3 nearest-mode
consistency filtering, O(1)-VRAM bilinear+Gaussian low-pass weighting, and
output packing to OpenCL 1.2 FP32 kernels with bounded background frame
prefetch, ping-pong async DMA, and automatic CPU fallback.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import contextvars
from dataclasses import dataclass
import os
from threading import Event, Lock

import numpy as np

from core.frame_storage import async_upload_view
from fusion_methods.openfocus_ab import _check_cancel
from utils.image_utils import fuse_output_dtype

_DTCWT_K = r'''
#pragma OPENCL FP_CONTRACT OFF

#define INV_SQRT2 0.7071067811865476f
#define SQRT_HALF 0.7071067811865476f

__constant float c_h0o[5] = {
    -0.05f, 0.25f, 0.6f, 0.25f, -0.05f
};
__constant float c_h1o[7] = {
    0.010714285714285713f, -0.05357142857142857f, -0.26071428571428573f,
    0.6071428571428571f, -0.26071428571428573f, -0.05357142857142857f,
    0.010714285714285713f
};
__constant float c_g0o[7] = {
    -0.010714285714285713f, -0.05357142857142857f, 0.26071428571428573f,
    0.6071428571428571f, 0.26071428571428573f, -0.05357142857142857f,
    -0.010714285714285713f
};
__constant float c_g1o[5] = {
    -0.05f, -0.25f, 0.6f, -0.25f, -0.05f
};

__constant float c_h0a[10] = {
    0.051130405283831656f, -0.013975370246888838f, -0.10983605166597087f,
    0.26383956105893763f, 0.7666284677930372f, 0.5636557101270515f,
    0.0008736226952170968f, -0.1002312195074762f, -0.0016896812725281543f,
    -0.006181881892116438f
};
__constant float c_h0b[10] = {
    -0.006181881892116438f, -0.0016896812725281543f, -0.1002312195074762f,
    0.0008736226952170968f, 0.5636557101270515f, 0.7666284677930372f,
    0.26383956105893763f, -0.10983605166597087f, -0.013975370246888838f,
    0.051130405283831656f
};
__constant float c_h1a[10] = {
    -0.006181881892116438f, 0.0016896812725281543f, -0.1002312195074762f,
    -0.0008736226952170968f, 0.5636557101270515f, -0.7666284677930372f,
    0.26383956105893763f, 0.10983605166597087f, -0.013975370246888838f,
    -0.051130405283831656f
};
__constant float c_h1b[10] = {
    -0.051130405283831656f, -0.013975370246888838f, 0.10983605166597087f,
    0.26383956105893763f, -0.7666284677930372f, 0.5636557101270515f,
    -0.0008736226952170968f, -0.1002312195074762f, 0.0016896812725281543f,
    -0.006181881892116438f
};

__constant float c_gauss13[13] = {
    8.922106644604355e-05f, 0.0010281965369358659f, 0.007597402203828096f,
    0.03599434718489647f, 0.10934117436408997f, 0.21296752989292145f,
    0.2659642696380615f, 0.21296752989292145f, 0.10934117436408997f,
    0.03599434718489647f, 0.007597402203828096f, 0.0010281965369358659f,
    8.922106644604355e-05f
};

static inline int reflect_idx(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i - 1;
        if (i >= n) i = 2 * n - 1 - i;
    }
    return i;
}

static inline int reflect_pad4_idx(int i, int orig_n, int pad_n, int extra) {
    int j = reflect_idx(i, pad_n);
    return extra ? clamp(j - 1, 0, orig_n - 1) : j;
}

static inline int reflect101_idx(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

__kernel void dtcwt_extract_luma_u8(
    __global const uchar* src,
    __global float* dst,
    const int orig_h,
    const int orig_w,
    const int r0,
    const int c0,
    const int C)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c0 || y >= r0) return;
    int sy = y < orig_h ? y : orig_h - 1;
    int sx = x < orig_w ? x : orig_w - 1;
    size_t pos = (size_t)sy * (size_t)orig_w + (size_t)sx;
    float luma;
    if (C == 1) {
        float v = (float)src[pos] / 255.0f;
        luma = 0.299f * v + 0.587f * v + 0.114f * v;
    } else {
        size_t base = pos * (size_t)C;
        float b = (float)src[base] / 255.0f;
        float g = (float)src[base + 1] / 255.0f;
        float r = (float)src[base + 2] / 255.0f;
        luma = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    dst[(size_t)y * (size_t)c0 + (size_t)x] = luma;
}

__kernel void dtcwt_extract_luma_u16(
    __global const ushort* src,
    __global float* dst,
    const int orig_h,
    const int orig_w,
    const int r0,
    const int c0,
    const int C)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c0 || y >= r0) return;
    int sy = y < orig_h ? y : orig_h - 1;
    int sx = x < orig_w ? x : orig_w - 1;
    size_t pos = (size_t)sy * (size_t)orig_w + (size_t)sx;
    const float to_f32 = 255.0f / 65535.0f;
    float luma;
    if (C == 1) {
        float v = ((float)src[pos] * to_f32) / 255.0f;
        luma = 0.299f * v + 0.587f * v + 0.114f * v;
    } else {
        size_t base = pos * (size_t)C;
        float b = ((float)src[base] * to_f32) / 255.0f;
        float g = ((float)src[base + 1] * to_f32) / 255.0f;
        float r = ((float)src[base + 2] * to_f32) / 255.0f;
        luma = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    dst[(size_t)y * (size_t)c0 + (size_t)x] = luma;
}

__kernel void dtcwt_extract_luma_f32(
    __global const float* src,
    __global float* dst,
    const int orig_h,
    const int orig_w,
    const int r0,
    const int c0,
    const int C)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c0 || y >= r0) return;
    int sy = y < orig_h ? y : orig_h - 1;
    int sx = x < orig_w ? x : orig_w - 1;
    size_t pos = (size_t)sy * (size_t)orig_w + (size_t)sx;
    float luma;
    if (C == 1) {
        float v = src[pos] / 255.0f;
        luma = 0.299f * v + 0.587f * v + 0.114f * v;
    } else {
        size_t base = pos * (size_t)C;
        float b = src[base] / 255.0f;
        float g = src[base + 1] / 255.0f;
        float r = src[base + 2] / 255.0f;
        luma = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    dst[(size_t)y * (size_t)c0 + (size_t)x] = luma;
}

__kernel void dtcwt_extract_chan_u8(
    __global const uchar* src,
    __global float* dst,
    const int orig_h,
    const int orig_w,
    const int r0,
    const int c0,
    const int C,
    const int src_ch)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c0 || y >= r0) return;
    int sy = y < orig_h ? y : orig_h - 1;
    int sx = x < orig_w ? x : orig_w - 1;
    size_t pos = (size_t)sy * (size_t)orig_w + (size_t)sx;
    float v = (float)src[pos * (size_t)C + (size_t)src_ch] / 255.0f;
    dst[(size_t)y * (size_t)c0 + (size_t)x] = v;
}

__kernel void dtcwt_extract_chan_u16(
    __global const ushort* src,
    __global float* dst,
    const int orig_h,
    const int orig_w,
    const int r0,
    const int c0,
    const int C,
    const int src_ch)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c0 || y >= r0) return;
    int sy = y < orig_h ? y : orig_h - 1;
    int sx = x < orig_w ? x : orig_w - 1;
    size_t pos = (size_t)sy * (size_t)orig_w + (size_t)sx;
    const float to_f32 = 255.0f / 65535.0f;
    float v = ((float)src[pos * (size_t)C + (size_t)src_ch] * to_f32) / 255.0f;
    dst[(size_t)y * (size_t)c0 + (size_t)x] = v;
}

__kernel void dtcwt_extract_chan_f32(
    __global const float* src,
    __global float* dst,
    const int orig_h,
    const int orig_w,
    const int r0,
    const int c0,
    const int C,
    const int src_ch)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c0 || y >= r0) return;
    int sy = y < orig_h ? y : orig_h - 1;
    int sx = x < orig_w ? x : orig_w - 1;
    size_t pos = (size_t)sy * (size_t)orig_w + (size_t)sx;
    float v = src[pos * (size_t)C + (size_t)src_ch] / 255.0f;
    dst[(size_t)y * (size_t)c0 + (size_t)x] = v;
}

__kernel void dtcwt_colfilter_v_pair(
    __global const float* src,
    __global float* lo,
    __global float* hi,
    const int r,
    const int c)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c || y >= r) return;

    float s0, s1, s2, s3, s4, s5, s6;
    if (y >= 3 && y < r - 3) {
        size_t cs = (size_t)c;
        size_t p3 = (size_t)y * cs + (size_t)x;
        s0 = src[p3 - 3 * cs];
        s1 = src[p3 - 2 * cs];
        s2 = src[p3 - cs];
        s3 = src[p3];
        s4 = src[p3 + cs];
        s5 = src[p3 + 2 * cs];
        s6 = src[p3 + 3 * cs];
    } else {
        s0 = src[(size_t)reflect_idx(y - 3, r) * (size_t)c + (size_t)x];
        s1 = src[(size_t)reflect_idx(y - 2, r) * (size_t)c + (size_t)x];
        s2 = src[(size_t)reflect_idx(y - 1, r) * (size_t)c + (size_t)x];
        s3 = src[(size_t)y * (size_t)c + (size_t)x];
        s4 = src[(size_t)reflect_idx(y + 1, r) * (size_t)c + (size_t)x];
        s5 = src[(size_t)reflect_idx(y + 2, r) * (size_t)c + (size_t)x];
        s6 = src[(size_t)reflect_idx(y + 3, r) * (size_t)c + (size_t)x];
    }

    size_t pos = (size_t)y * (size_t)c + (size_t)x;
    lo[pos] = s5 * c_h0o[0] + s4 * c_h0o[1] + s3 * c_h0o[2] + s2 * c_h0o[3] + s1 * c_h0o[4];
    hi[pos] = (s6 * c_h1o[0] + s5 * c_h1o[1] + s4 * c_h1o[2] + s3 * c_h1o[3]
               + s2 * c_h1o[4] + s1 * c_h1o[5] + s0 * c_h1o[6]);
}

__kernel void dtcwt_colfilter_h5(
    __global const float* src,
    __global float* dst,
    const int r,
    const int c,
    const int use_h0)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c || y >= r) return;
    __constant const float* h = use_h0 ? c_h0o : c_g1o;
    size_t row = (size_t)y * (size_t)c;
    if (x >= 2 && x < c - 2) {
        dst[row + (size_t)x] = (
            src[row + (size_t)(x + 2)] * h[0]
            + src[row + (size_t)(x + 1)] * h[1]
            + src[row + (size_t)x]       * h[2]
            + src[row + (size_t)(x - 1)] * h[3]
            + src[row + (size_t)(x - 2)] * h[4]);
    } else {
        dst[row + (size_t)x] = (
            src[row + (size_t)reflect_idx(x + 2, c)] * h[0]
            + src[row + (size_t)reflect_idx(x + 1, c)] * h[1]
            + src[row + (size_t)x]                     * h[2]
            + src[row + (size_t)reflect_idx(x - 1, c)] * h[3]
            + src[row + (size_t)reflect_idx(x - 2, c)] * h[4]);
    }
}

static inline float conv5_row(__global const float* row, int x, int c) {
    if (x >= 2 && x < c - 2) {
        return row[x + 2] * c_h0o[0] + row[x + 1] * c_h0o[1] + row[x] * c_h0o[2]
             + row[x - 1] * c_h0o[3] + row[x - 2] * c_h0o[4];
    }
    return row[reflect_idx(x + 2, c)] * c_h0o[0]
         + row[reflect_idx(x + 1, c)] * c_h0o[1]
         + row[x]                     * c_h0o[2]
         + row[reflect_idx(x - 1, c)] * c_h0o[3]
         + row[reflect_idx(x - 2, c)] * c_h0o[4];
}

static inline float conv7_row(__global const float* row, int x, int c) {
    if (x >= 3 && x < c - 3) {
        return row[x + 3] * c_h1o[0] + row[x + 2] * c_h1o[1] + row[x + 1] * c_h1o[2]
             + row[x] * c_h1o[3] + row[x - 1] * c_h1o[4] + row[x - 2] * c_h1o[5]
             + row[x - 3] * c_h1o[6];
    }
    return row[reflect_idx(x + 3, c)] * c_h1o[0]
         + row[reflect_idx(x + 2, c)] * c_h1o[1]
         + row[reflect_idx(x + 1, c)] * c_h1o[2]
         + row[x]                     * c_h1o[3]
         + row[reflect_idx(x - 1, c)] * c_h1o[4]
         + row[reflect_idx(x - 2, c)] * c_h1o[5]
         + row[reflect_idx(x - 3, c)] * c_h1o[6];
}

static inline void eval_level0_6orient(
    __global const float* lo,
    __global const float* hi,
    int qx, int qy, int c0,
    float2 out_z[6])
{
    size_t row0 = (size_t)(2 * qy) * (size_t)c0;
    size_t row1 = row0 + (size_t)c0;
    int cx = 2 * qx;
    float a, b, cc, d;

    if (cx >= 3 && cx + 4 < c0) {
        __global const float* lt = lo + row0 + (size_t)(cx - 3);
        __global const float* lb = lo + row1 + (size_t)(cx - 3);
        __global const float* ht = hi + row0 + (size_t)(cx - 3);
        __global const float* hb = hi + row1 + (size_t)(cx - 3);
        float lt0 = lt[0], lt1 = lt[1], lt2 = lt[2], lt3 = lt[3], lt4 = lt[4], lt5 = lt[5], lt6 = lt[6], lt7 = lt[7];
        float lb0 = lb[0], lb1 = lb[1], lb2 = lb[2], lb3 = lb[3], lb4 = lb[4], lb5 = lb[5], lb6 = lb[6], lb7 = lb[7];
        float ht0 = ht[0], ht1 = ht[1], ht2 = ht[2], ht3 = ht[3], ht4 = ht[4], ht5 = ht[5], ht6 = ht[6], ht7 = ht[7];
        float hb0 = hb[0], hb1 = hb[1], hb2 = hb[2], hb3 = hb[3], hb4 = hb[4], hb5 = hb[5], hb6 = hb[6], hb7 = hb[7];

        // Orientations 0, 5: h5(hi)
        a  = (ht5 * c_h0o[0] + ht4 * c_h0o[1] + ht3 * c_h0o[2] + ht2 * c_h0o[3] + ht1 * c_h0o[4]) * INV_SQRT2;
        b  = (ht6 * c_h0o[0] + ht5 * c_h0o[1] + ht4 * c_h0o[2] + ht3 * c_h0o[3] + ht2 * c_h0o[4]) * INV_SQRT2;
        cc = (hb5 * c_h0o[0] + hb4 * c_h0o[1] + hb3 * c_h0o[2] + hb2 * c_h0o[3] + hb1 * c_h0o[4]) * INV_SQRT2;
        d  = (hb6 * c_h0o[0] + hb5 * c_h0o[1] + hb4 * c_h0o[2] + hb3 * c_h0o[3] + hb2 * c_h0o[4]) * INV_SQRT2;
        out_z[0] = (float2)(a - d, b + cc);
        out_z[5] = (float2)(a + d, b - cc);

        // Orientations 2, 3: h7(lo)
        a  = (lt6 * c_h1o[0] + lt5 * c_h1o[1] + lt4 * c_h1o[2] + lt3 * c_h1o[3] + lt2 * c_h1o[4] + lt1 * c_h1o[5] + lt0 * c_h1o[6]) * INV_SQRT2;
        b  = (lt7 * c_h1o[0] + lt6 * c_h1o[1] + lt5 * c_h1o[2] + lt4 * c_h1o[3] + lt3 * c_h1o[4] + lt2 * c_h1o[5] + lt1 * c_h1o[6]) * INV_SQRT2;
        cc = (lb6 * c_h1o[0] + lb5 * c_h1o[1] + lb4 * c_h1o[2] + lb3 * c_h1o[3] + lb2 * c_h1o[4] + lb1 * c_h1o[5] + lb0 * c_h1o[6]) * INV_SQRT2;
        d  = (lb7 * c_h1o[0] + lb6 * c_h1o[1] + lb5 * c_h1o[2] + lb4 * c_h1o[3] + lb3 * c_h1o[4] + lb2 * c_h1o[5] + lb1 * c_h1o[6]) * INV_SQRT2;
        out_z[2] = (float2)(a - d, b + cc);
        out_z[3] = (float2)(a + d, b - cc);

        // Orientations 1, 4: h7(hi)
        a  = (ht6 * c_h1o[0] + ht5 * c_h1o[1] + ht4 * c_h1o[2] + ht3 * c_h1o[3] + ht2 * c_h1o[4] + ht1 * c_h1o[5] + ht0 * c_h1o[6]) * INV_SQRT2;
        b  = (ht7 * c_h1o[0] + ht6 * c_h1o[1] + ht5 * c_h1o[2] + ht4 * c_h1o[3] + ht3 * c_h1o[4] + ht2 * c_h1o[5] + ht1 * c_h1o[6]) * INV_SQRT2;
        cc = (hb6 * c_h1o[0] + hb5 * c_h1o[1] + hb4 * c_h1o[2] + hb3 * c_h1o[3] + hb2 * c_h1o[4] + hb1 * c_h1o[5] + hb0 * c_h1o[6]) * INV_SQRT2;
        d  = (hb7 * c_h1o[0] + hb6 * c_h1o[1] + hb5 * c_h1o[2] + hb4 * c_h1o[3] + hb3 * c_h1o[4] + hb2 * c_h1o[5] + hb1 * c_h1o[6]) * INV_SQRT2;
        out_z[1] = (float2)(a - d, b + cc);
        out_z[4] = (float2)(a + d, b - cc);
    } else {
        __global const float* lt = lo + row0;
        __global const float* lb = lo + row1;
        __global const float* ht = hi + row0;
        __global const float* hb = hi + row1;

        a  = conv5_row(ht, cx, c0) * INV_SQRT2;
        b  = conv5_row(ht, cx + 1, c0) * INV_SQRT2;
        cc = conv5_row(hb, cx, c0) * INV_SQRT2;
        d  = conv5_row(hb, cx + 1, c0) * INV_SQRT2;
        out_z[0] = (float2)(a - d, b + cc);
        out_z[5] = (float2)(a + d, b - cc);

        a  = conv7_row(lt, cx, c0) * INV_SQRT2;
        b  = conv7_row(lt, cx + 1, c0) * INV_SQRT2;
        cc = conv7_row(lb, cx, c0) * INV_SQRT2;
        d  = conv7_row(lb, cx + 1, c0) * INV_SQRT2;
        out_z[2] = (float2)(a - d, b + cc);
        out_z[3] = (float2)(a + d, b - cc);

        a  = conv7_row(ht, cx, c0) * INV_SQRT2;
        b  = conv7_row(ht, cx + 1, c0) * INV_SQRT2;
        cc = conv7_row(hb, cx, c0) * INV_SQRT2;
        d  = conv7_row(hb, cx + 1, c0) * INV_SQRT2;
        out_z[1] = (float2)(a - d, b + cc);
        out_z[4] = (float2)(a + d, b - cc);
    }
}

__kernel void dtcwt_level0_h_mag(
    __global const float* lo,
    __global const float* hi,
    __global float* mag,
    const int hp_h,
    const int hp_w,
    const int c0)
{
    int qx = get_global_id(0);
    int qy = get_global_id(1);
    if (qx >= hp_w || qy >= hp_h) return;

    float2 z[6];
    eval_level0_6orient(lo, hi, qx, qy, c0, z);
    float s = 0.0f;
    for (int o = 0; o < 6; ++o) {
        s += hypot(z[o].x, z[o].y);
    }
    mag[(size_t)qy * (size_t)hp_w + (size_t)qx] = s;
}

__kernel void dtcwt_level0_h_select(
    __global const float* lo,
    __global const float* hi,
    __global const int* decision,
    __global float2* hp_dst,
    const int hp_h,
    const int hp_w,
    const int c0,
    const int frame_idx)
{
    int qx = get_global_id(0);
    int qy = get_global_id(1);
    if (qx >= hp_w || qy >= hp_h) return;

    size_t pos = (size_t)qy * (size_t)hp_w + (size_t)qx;
    if (decision[pos] != frame_idx) return;

    float2 z[6];
    eval_level0_6orient(lo, hi, qx, qy, c0, z);
    size_t plane = (size_t)hp_h * (size_t)hp_w;
    for (int o = 0; o < 6; ++o) {
        hp_dst[(size_t)o * plane + pos] = z[o];
    }
}

__kernel void dtcwt_coldfilt_v_pair(
    __global const float* src,
    __global float* lo,
    __global float* hi,
    const int in_h,
    const int in_w,
    const int pad_h,
    const int pr)
{
    int x = get_global_id(0);
    int j = get_global_id(1);
    int groups = pad_h >> 2;
    if (x >= in_w || j >= groups) return;

    int base = 4 * j;
    float s_e[10];
    float s_o[10];
    if (base >= 8 + pr && base + 11 < in_h + pr) {
        int b_src = base - pr;
        size_t cs = (size_t)in_w;
        size_t bx = (size_t)x;
        s_e[0] = src[(size_t)(b_src + 10) * cs + bx];
        s_e[1] = src[(size_t)(b_src +  8) * cs + bx];
        s_e[2] = src[(size_t)(b_src +  6) * cs + bx];
        s_e[3] = src[(size_t)(b_src +  4) * cs + bx];
        s_e[4] = src[(size_t)(b_src +  2) * cs + bx];
        s_e[5] = src[(size_t)(b_src     ) * cs + bx];
        s_e[6] = src[(size_t)(b_src -  2) * cs + bx];
        s_e[7] = src[(size_t)(b_src -  4) * cs + bx];
        s_e[8] = src[(size_t)(b_src -  6) * cs + bx];
        s_e[9] = src[(size_t)(b_src -  8) * cs + bx];

        s_o[0] = src[(size_t)(b_src + 11) * cs + bx];
        s_o[1] = src[(size_t)(b_src +  9) * cs + bx];
        s_o[2] = src[(size_t)(b_src +  7) * cs + bx];
        s_o[3] = src[(size_t)(b_src +  5) * cs + bx];
        s_o[4] = src[(size_t)(b_src +  3) * cs + bx];
        s_o[5] = src[(size_t)(b_src +  1) * cs + bx];
        s_o[6] = src[(size_t)(b_src -  1) * cs + bx];
        s_o[7] = src[(size_t)(b_src -  3) * cs + bx];
        s_o[8] = src[(size_t)(b_src -  5) * cs + bx];
        s_o[9] = src[(size_t)(b_src -  7) * cs + bx];
    } else {
        size_t cs = (size_t)in_w;
        size_t bx = (size_t)x;
        s_e[0] = src[(size_t)reflect_pad4_idx(base + 10, in_h, pad_h, pr) * cs + bx];
        s_e[1] = src[(size_t)reflect_pad4_idx(base +  8, in_h, pad_h, pr) * cs + bx];
        s_e[2] = src[(size_t)reflect_pad4_idx(base +  6, in_h, pad_h, pr) * cs + bx];
        s_e[3] = src[(size_t)reflect_pad4_idx(base +  4, in_h, pad_h, pr) * cs + bx];
        s_e[4] = src[(size_t)reflect_pad4_idx(base +  2, in_h, pad_h, pr) * cs + bx];
        s_e[5] = src[(size_t)reflect_pad4_idx(base     , in_h, pad_h, pr) * cs + bx];
        s_e[6] = src[(size_t)reflect_pad4_idx(base -  2, in_h, pad_h, pr) * cs + bx];
        s_e[7] = src[(size_t)reflect_pad4_idx(base -  4, in_h, pad_h, pr) * cs + bx];
        s_e[8] = src[(size_t)reflect_pad4_idx(base -  6, in_h, pad_h, pr) * cs + bx];
        s_e[9] = src[(size_t)reflect_pad4_idx(base -  8, in_h, pad_h, pr) * cs + bx];

        s_o[0] = src[(size_t)reflect_pad4_idx(base + 11, in_h, pad_h, pr) * cs + bx];
        s_o[1] = src[(size_t)reflect_pad4_idx(base +  9, in_h, pad_h, pr) * cs + bx];
        s_o[2] = src[(size_t)reflect_pad4_idx(base +  7, in_h, pad_h, pr) * cs + bx];
        s_o[3] = src[(size_t)reflect_pad4_idx(base +  5, in_h, pad_h, pr) * cs + bx];
        s_o[4] = src[(size_t)reflect_pad4_idx(base +  3, in_h, pad_h, pr) * cs + bx];
        s_o[5] = src[(size_t)reflect_pad4_idx(base +  1, in_h, pad_h, pr) * cs + bx];
        s_o[6] = src[(size_t)reflect_pad4_idx(base -  1, in_h, pad_h, pr) * cs + bx];
        s_o[7] = src[(size_t)reflect_pad4_idx(base -  3, in_h, pad_h, pr) * cs + bx];
        s_o[8] = src[(size_t)reflect_pad4_idx(base -  5, in_h, pad_h, pr) * cs + bx];
        s_o[9] = src[(size_t)reflect_pad4_idx(base -  7, in_h, pad_h, pr) * cs + bx];
    }

    float4 even = (float4)(0.0f), odd = (float4)(0.0f);
    for (int t = 0; t < 10; t += 2) {
        even += (float4)(s_e[t] * c_h0b[t], s_o[t] * c_h0a[t],
                         s_o[t] * c_h1a[t], s_e[t] * c_h1b[t]);
        int u = t + 1;
        odd += (float4)(s_e[u] * c_h0b[u], s_o[u] * c_h0a[u],
                        s_o[u] * c_h1a[u], s_e[u] * c_h1b[u]);
    }
    float4 v = even + odd;

    size_t row0 = (size_t)(2 * j) * (size_t)in_w + (size_t)x;
    size_t row1 = row0 + (size_t)in_w;
    lo[row0] = v.x;
    lo[row1] = v.y;
    hi[row0] = v.z;
    hi[row1] = v.w;
}

__kernel void dtcwt_coldfilt_h_lo(
    __global const float* src,
    __global float* dst,
    const int r,
    const int in_w,
    const int pad_w,
    const int pc)
{
    int j = get_global_id(0);
    int y = get_global_id(1);
    int groups = pad_w >> 2;
    int dst_c = pad_w >> 1;
    if (j >= groups || y >= r) return;

    __global const float* s_row = src + (size_t)y * (size_t)in_w;
    int base = 4 * j;
    float2 even = (float2)(0.0f), odd = (float2)(0.0f);

    if (base >= 8 + pc && base + 11 < in_w + pc) {
        int b_src = base - pc;
        for (int t = 0; t < 10; t += 2) {
            int i = b_src + 10 - 2 * t;
            even += (float2)(s_row[i] * c_h0b[t], s_row[i + 1] * c_h0a[t]);
            odd  += (float2)(s_row[i - 2] * c_h0b[t + 1], s_row[i - 1] * c_h0a[t + 1]);
        }
    } else {
        for (int t = 0; t < 10; t += 2) {
            int i = base + 10 - 2 * t;
            even += (float2)(
                s_row[reflect_pad4_idx(i, in_w, pad_w, pc)] * c_h0b[t],
                s_row[reflect_pad4_idx(i + 1, in_w, pad_w, pc)] * c_h0a[t]);
            odd += (float2)(
                s_row[reflect_pad4_idx(i - 2, in_w, pad_w, pc)] * c_h0b[t + 1],
                s_row[reflect_pad4_idx(i - 1, in_w, pad_w, pc)] * c_h0a[t + 1]);
        }
    }
    float2 v = even + odd;
    size_t dst_row = (size_t)y * (size_t)dst_c;
    dst[dst_row + (size_t)(2 * j)]     = v.x;
    dst[dst_row + (size_t)(2 * j + 1)] = v.y;
}

static inline float4 coldfilt_h_both_row(
    __global const float* s_row, int base, int in_w, int pad_w, int pc)
{
    float4 even = (float4)(0.0f), odd = (float4)(0.0f);
    if (base >= 8 + pc && base + 11 < in_w + pc) {
        int b_src = base - pc;
        for (int t = 0; t < 10; t += 2) {
            int i = b_src + 10 - 2 * t;
            float e0 = s_row[i];
            float o0 = s_row[i + 1];
            float e1 = s_row[i - 2];
            float o1 = s_row[i - 1];
            even += (float4)(e0 * c_h0b[t], o0 * c_h0a[t], o0 * c_h1a[t], e0 * c_h1b[t]);
            odd  += (float4)(e1 * c_h0b[t + 1], o1 * c_h0a[t + 1], o1 * c_h1a[t + 1], e1 * c_h1b[t + 1]);
        }
    } else {
        for (int t = 0; t < 10; t += 2) {
            int i = base + 10 - 2 * t;
            float e0 = s_row[reflect_pad4_idx(i, in_w, pad_w, pc)];
            float o0 = s_row[reflect_pad4_idx(i + 1, in_w, pad_w, pc)];
            float e1 = s_row[reflect_pad4_idx(i - 2, in_w, pad_w, pc)];
            float o1 = s_row[reflect_pad4_idx(i - 1, in_w, pad_w, pc)];
            even += (float4)(e0 * c_h0b[t], o0 * c_h0a[t], o0 * c_h1a[t], e0 * c_h1b[t]);
            odd  += (float4)(e1 * c_h0b[t + 1], o1 * c_h0a[t + 1], o1 * c_h1a[t + 1], e1 * c_h1b[t + 1]);
        }
    }
    // .xy = (pos_sign=1: out0, out1), .zw = (pos_sign=0: out0, out1)
    return even + odd;
}

static inline float2 coldfilt_h_hi_row(
    __global const float* s_row, int base, int in_w, int pad_w, int pc)
{
    float2 even = (float2)(0.0f), odd = (float2)(0.0f);
    if (base >= 8 + pc && base + 11 < in_w + pc) {
        int b_src = base - pc;
        for (int t = 0; t < 10; t += 2) {
            int i = b_src + 10 - 2 * t;
            even += (float2)(s_row[i + 1] * c_h1a[t], s_row[i] * c_h1b[t]);
            odd  += (float2)(s_row[i - 1] * c_h1a[t + 1], s_row[i - 2] * c_h1b[t + 1]);
        }
    } else {
        for (int t = 0; t < 10; t += 2) {
            int i = base + 10 - 2 * t;
            even += (float2)(
                s_row[reflect_pad4_idx(i + 1, in_w, pad_w, pc)] * c_h1a[t],
                s_row[reflect_pad4_idx(i, in_w, pad_w, pc)] * c_h1b[t]);
            odd += (float2)(
                s_row[reflect_pad4_idx(i - 1, in_w, pad_w, pc)] * c_h1a[t + 1],
                s_row[reflect_pad4_idx(i - 2, in_w, pad_w, pc)] * c_h1b[t + 1]);
        }
    }
    // pos_sign=0: out0 = h1a (v.x), out1 = h1b (v.y)
    return even + odd;
}

static inline void eval_levelk_6orient(
    __global const float* lo2,
    __global const float* hi2,
    int qx, int qy, int in_w, int pad_w, int pc,
    float2 out_z[6])
{
    size_t row0 = (size_t)(2 * qy) * (size_t)in_w;
    size_t row1 = row0 + (size_t)in_w;
    int base = 4 * qx;

    float4 ht = coldfilt_h_both_row(hi2 + row0, base, in_w, pad_w, pc);
    float4 hb = coldfilt_h_both_row(hi2 + row1, base, in_w, pad_w, pc);
    float2 lt = coldfilt_h_hi_row(lo2 + row0, base, in_w, pad_w, pc);
    float2 lb = coldfilt_h_hi_row(lo2 + row1, base, in_w, pad_w, pc);

    float a, b, cc, d;
    // Orientations 0, 5: h0(hi2)
    a = ht.x * INV_SQRT2; b = ht.y * INV_SQRT2; cc = hb.x * INV_SQRT2; d = hb.y * INV_SQRT2;
    out_z[0] = (float2)(a - d, b + cc);
    out_z[5] = (float2)(a + d, b - cc);

    // Orientations 2, 3: h1(lo2)
    a = lt.x * INV_SQRT2; b = lt.y * INV_SQRT2; cc = lb.x * INV_SQRT2; d = lb.y * INV_SQRT2;
    out_z[2] = (float2)(a - d, b + cc);
    out_z[3] = (float2)(a + d, b - cc);

    // Orientations 1, 4: h1(hi2)
    a = ht.z * INV_SQRT2; b = ht.w * INV_SQRT2; cc = hb.z * INV_SQRT2; d = hb.w * INV_SQRT2;
    out_z[1] = (float2)(a - d, b + cc);
    out_z[4] = (float2)(a + d, b - cc);
}

__kernel void dtcwt_levelk_h_mag(
    __global const float* lo2,
    __global const float* hi2,
    __global float* mag,
    const int hp_h,
    const int hp_w,
    const int in_w,
    const int pad_w,
    const int pc)
{
    int qx = get_global_id(0);
    int qy = get_global_id(1);
    if (qx >= hp_w || qy >= hp_h) return;

    float2 z[6];
    eval_levelk_6orient(lo2, hi2, qx, qy, in_w, pad_w, pc, z);
    float s = 0.0f;
    for (int o = 0; o < 6; ++o) {
        s += hypot(z[o].x, z[o].y);
    }
    mag[(size_t)qy * (size_t)hp_w + (size_t)qx] = s;
}

__kernel void dtcwt_levelk_h_select(
    __global const float* lo2,
    __global const float* hi2,
    __global const int* decision,
    __global float2* hp_dst,
    const int hp_h,
    const int hp_w,
    const int in_w,
    const int pad_w,
    const int pc,
    const int frame_idx)
{
    int qx = get_global_id(0);
    int qy = get_global_id(1);
    if (qx >= hp_w || qy >= hp_h) return;

    size_t pos = (size_t)qy * (size_t)hp_w + (size_t)qx;
    if (decision[pos] != frame_idx) return;

    float2 z[6];
    eval_levelk_6orient(lo2, hi2, qx, qy, in_w, pad_w, pc, z);
    size_t plane = (size_t)hp_h * (size_t)hp_w;
    for (int o = 0; o < 6; ++o) {
        hp_dst[(size_t)o * plane + pos] = z[o];
    }
}

static inline float c2q_sample(
    __global const float2* hp, int y, int x, int h2, int w2, int c0, int c1)
{
    size_t plane = (size_t)h2 * (size_t)w2;
    size_t pos = (size_t)(y >> 1) * (size_t)w2 + (size_t)(x >> 1);
    float2 z0 = hp[(size_t)c0 * plane + pos];
    float2 z1 = hp[(size_t)c1 * plane + pos];
    int oy = y & 1;
    int ox = x & 1;
    if (oy == 0) {
        return (ox == 0 ? (z0.x + z1.x) : (z0.y + z1.y)) * SQRT_HALF;
    }
    return (ox == 0 ? (z0.y - z1.y) : (z1.x - z0.x)) * SQRT_HALF;
}

__kernel void dtcwt_colfilter_v_add0_z_hp(
    __global const float* src7,
    __global const float2* hp,
    __global float* dst,
    const int r,
    const int c,
    const int c0,
    const int c1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c || y >= r) return;
    int h2 = r >> 1;
    int w2 = c >> 1;

    int ym3 = reflect_idx(y - 3, r);
    int ym2 = reflect_idx(y - 2, r);
    int ym1 = reflect_idx(y - 1, r);
    int yp1 = reflect_idx(y + 1, r);
    int yp2 = reflect_idx(y + 2, r);
    int yp3 = reflect_idx(y + 3, r);

    float v7 = (
        src7[(size_t)ym3 * (size_t)c + (size_t)x] * c_g0o[0]
        + src7[(size_t)ym2 * (size_t)c + (size_t)x] * c_g0o[1]
        + src7[(size_t)ym1 * (size_t)c + (size_t)x] * c_g0o[2]
        + src7[(size_t)y   * (size_t)c + (size_t)x] * c_g0o[3]
        + src7[(size_t)yp1 * (size_t)c + (size_t)x] * c_g0o[4]
        + src7[(size_t)yp2 * (size_t)c + (size_t)x] * c_g0o[5]
        + src7[(size_t)yp3 * (size_t)c + (size_t)x] * c_g0o[6]);

    float v5 = (
        c2q_sample(hp, ym2, x, h2, w2, c0, c1) * c_g1o[0]
        + c2q_sample(hp, ym1, x, h2, w2, c0, c1) * c_g1o[1]
        + c2q_sample(hp, y,   x, h2, w2, c0, c1) * c_g1o[2]
        + c2q_sample(hp, yp1, x, h2, w2, c0, c1) * c_g1o[3]
        + c2q_sample(hp, yp2, x, h2, w2, c0, c1) * c_g1o[4]);

    dst[(size_t)y * (size_t)c + (size_t)x] = v7 + v5;
}

__kernel void dtcwt_colfilter_v_add0_hp_hp(
    __global const float2* hp,
    __global float* dst,
    const int r,
    const int c,
    const int a0,
    const int a1,
    const int b0,
    const int b1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c || y >= r) return;
    int h2 = r >> 1;
    int w2 = c >> 1;

    int ym3 = reflect_idx(y - 3, r);
    int ym2 = reflect_idx(y - 2, r);
    int ym1 = reflect_idx(y - 1, r);
    int yp1 = reflect_idx(y + 1, r);
    int yp2 = reflect_idx(y + 2, r);
    int yp3 = reflect_idx(y + 3, r);

    float v7 = (
        c2q_sample(hp, ym3, x, h2, w2, a0, a1) * c_g0o[0]
        + c2q_sample(hp, ym2, x, h2, w2, a0, a1) * c_g0o[1]
        + c2q_sample(hp, ym1, x, h2, w2, a0, a1) * c_g0o[2]
        + c2q_sample(hp, y,   x, h2, w2, a0, a1) * c_g0o[3]
        + c2q_sample(hp, yp1, x, h2, w2, a0, a1) * c_g0o[4]
        + c2q_sample(hp, yp2, x, h2, w2, a0, a1) * c_g0o[5]
        + c2q_sample(hp, yp3, x, h2, w2, a0, a1) * c_g0o[6]);

    float v5 = (
        c2q_sample(hp, ym2, x, h2, w2, b0, b1) * c_g1o[0]
        + c2q_sample(hp, ym1, x, h2, w2, b0, b1) * c_g1o[1]
        + c2q_sample(hp, y,   x, h2, w2, b0, b1) * c_g1o[2]
        + c2q_sample(hp, yp1, x, h2, w2, b0, b1) * c_g1o[3]
        + c2q_sample(hp, yp2, x, h2, w2, b0, b1) * c_g1o[4]);

    dst[(size_t)y * (size_t)c + (size_t)x] = v7 + v5;
}

__kernel void dtcwt_colfilter_h_add0(
    __global const float* src7,
    __global const float* src5,
    __global float* dst,
    const int r,
    const int c)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c || y >= r) return;

    size_t row = (size_t)y * (size_t)c;
    int xm3 = reflect_idx(x - 3, c);
    int xm2 = reflect_idx(x - 2, c);
    int xm1 = reflect_idx(x - 1, c);
    int xp1 = reflect_idx(x + 1, c);
    int xp2 = reflect_idx(x + 2, c);
    int xp3 = reflect_idx(x + 3, c);

    float v7 = (
        src7[row + (size_t)xm3] * c_g0o[0]
        + src7[row + (size_t)xm2] * c_g0o[1]
        + src7[row + (size_t)xm1] * c_g0o[2]
        + src7[row + (size_t)x]   * c_g0o[3]
        + src7[row + (size_t)xp1] * c_g0o[4]
        + src7[row + (size_t)xp2] * c_g0o[5]
        + src7[row + (size_t)xp3] * c_g0o[6]);

    float v5 = (
        src5[row + (size_t)xm2] * c_g1o[0]
        + src5[row + (size_t)xm1] * c_g1o[1]
        + src5[row + (size_t)x]   * c_g1o[2]
        + src5[row + (size_t)xp1] * c_g1o[3]
        + src5[row + (size_t)xp2] * c_g1o[4]);

    dst[row + (size_t)x] = v7 + v5;
}

__kernel void dtcwt_colifilt_v_add_z_hp(
    __global const float* src0,
    __global const float2* hp,
    __global float* dst,
    const int r,
    const int c,
    const int c0,
    const int c1)
{
    int x = get_global_id(0);
    int j = get_global_id(1);
    int groups = r >> 1;
    if (x >= c || j >= groups) return;
    int h2 = r >> 1;
    int w2 = c >> 1;

    int im3 = reflect_idx(2 * j - 3, r);
    int im1 = reflect_idx(2 * j - 1, r);
    int ip1 = reflect_idx(2 * j + 1, r);
    int ip3 = reflect_idx(2 * j + 3, r);
    int ip5 = reflect_idx(2 * j + 5, r);

    int im4 = reflect_idx(2 * j - 4, r);
    int im2 = reflect_idx(2 * j - 2, r);
    int i0  = reflect_idx(2 * j, r);
    int ip2 = reflect_idx(2 * j + 2, r);
    int ip4 = reflect_idx(2 * j + 4, r);

    float a0_4 = src0[(size_t)im4 * (size_t)c + (size_t)x];
    float a0_3 = src0[(size_t)im2 * (size_t)c + (size_t)x];
    float a0_2 = src0[(size_t)i0  * (size_t)c + (size_t)x];
    float a0_1 = src0[(size_t)ip2 * (size_t)c + (size_t)x];
    float a0_0 = src0[(size_t)ip4 * (size_t)c + (size_t)x];

    float b0_4 = src0[(size_t)im3 * (size_t)c + (size_t)x];
    float b0_3 = src0[(size_t)im1 * (size_t)c + (size_t)x];
    float b0_2 = src0[(size_t)ip1 * (size_t)c + (size_t)x];
    float b0_1 = src0[(size_t)ip3 * (size_t)c + (size_t)x];
    float b0_0 = src0[(size_t)ip5 * (size_t)c + (size_t)x];

    float z_y0 = a0_0 * c_h0a[0] + a0_1 * c_h0a[2] + a0_2 * c_h0a[4] + a0_3 * c_h0a[6] + a0_4 * c_h0a[8];
    float z_y1 = b0_0 * c_h0b[0] + b0_1 * c_h0b[2] + b0_2 * c_h0b[4] + b0_3 * c_h0b[6] + b0_4 * c_h0b[8];
    float z_y2 = a0_0 * c_h0a[1] + a0_1 * c_h0a[3] + a0_2 * c_h0a[5] + a0_3 * c_h0a[7] + a0_4 * c_h0a[9];
    float z_y3 = b0_0 * c_h0b[1] + b0_1 * c_h0b[3] + b0_2 * c_h0b[5] + b0_3 * c_h0b[7] + b0_4 * c_h0b[9];

    float a1_4 = c2q_sample(hp, im3, x, h2, w2, c0, c1);
    float a1_3 = c2q_sample(hp, im1, x, h2, w2, c0, c1);
    float a1_2 = c2q_sample(hp, ip1, x, h2, w2, c0, c1);
    float a1_1 = c2q_sample(hp, ip3, x, h2, w2, c0, c1);
    float a1_0 = c2q_sample(hp, ip5, x, h2, w2, c0, c1);

    float b1_4 = c2q_sample(hp, im4, x, h2, w2, c0, c1);
    float b1_3 = c2q_sample(hp, im2, x, h2, w2, c0, c1);
    float b1_2 = c2q_sample(hp, i0,  x, h2, w2, c0, c1);
    float b1_1 = c2q_sample(hp, ip2, x, h2, w2, c0, c1);
    float b1_0 = c2q_sample(hp, ip4, x, h2, w2, c0, c1);

    float h_y0 = a1_0 * c_h1a[0] + a1_1 * c_h1a[2] + a1_2 * c_h1a[4] + a1_3 * c_h1a[6] + a1_4 * c_h1a[8];
    float h_y1 = b1_0 * c_h1b[0] + b1_1 * c_h1b[2] + b1_2 * c_h1b[4] + b1_3 * c_h1b[6] + b1_4 * c_h1b[8];
    float h_y2 = a1_0 * c_h1a[1] + a1_1 * c_h1a[3] + a1_2 * c_h1a[5] + a1_3 * c_h1a[7] + a1_4 * c_h1a[9];
    float h_y3 = b1_0 * c_h1b[1] + b1_1 * c_h1b[3] + b1_2 * c_h1b[5] + b1_3 * c_h1b[7] + b1_4 * c_h1b[9];

    size_t y0 = (size_t)(4 * j);
    dst[y0       * (size_t)c + (size_t)x] = z_y0 + h_y0;
    dst[(y0 + 1) * (size_t)c + (size_t)x] = z_y1 + h_y1;
    dst[(y0 + 2) * (size_t)c + (size_t)x] = z_y2 + h_y2;
    dst[(y0 + 3) * (size_t)c + (size_t)x] = z_y3 + h_y3;
}

__kernel void dtcwt_colifilt_v_add_hp_hp(
    __global const float2* hp,
    __global float* dst,
    const int r,
    const int c,
    const int a0,
    const int a1,
    const int b0,
    const int b1)
{
    int x = get_global_id(0);
    int j = get_global_id(1);
    int groups = r >> 1;
    if (x >= c || j >= groups) return;
    int h2 = r >> 1;
    int w2 = c >> 1;

    int im3 = reflect_idx(2 * j - 3, r);
    int im1 = reflect_idx(2 * j - 1, r);
    int ip1 = reflect_idx(2 * j + 1, r);
    int ip3 = reflect_idx(2 * j + 3, r);
    int ip5 = reflect_idx(2 * j + 5, r);

    int im4 = reflect_idx(2 * j - 4, r);
    int im2 = reflect_idx(2 * j - 2, r);
    int i0  = reflect_idx(2 * j, r);
    int ip2 = reflect_idx(2 * j + 2, r);
    int ip4 = reflect_idx(2 * j + 4, r);

    float a0_4 = c2q_sample(hp, im4, x, h2, w2, a0, a1);
    float a0_3 = c2q_sample(hp, im2, x, h2, w2, a0, a1);
    float a0_2 = c2q_sample(hp, i0,  x, h2, w2, a0, a1);
    float a0_1 = c2q_sample(hp, ip2, x, h2, w2, a0, a1);
    float a0_0 = c2q_sample(hp, ip4, x, h2, w2, a0, a1);

    float b0_4 = c2q_sample(hp, im3, x, h2, w2, a0, a1);
    float b0_3 = c2q_sample(hp, im1, x, h2, w2, a0, a1);
    float b0_2 = c2q_sample(hp, ip1, x, h2, w2, a0, a1);
    float b0_1 = c2q_sample(hp, ip3, x, h2, w2, a0, a1);
    float b0_0 = c2q_sample(hp, ip5, x, h2, w2, a0, a1);

    float z_y0 = a0_0 * c_h0a[0] + a0_1 * c_h0a[2] + a0_2 * c_h0a[4] + a0_3 * c_h0a[6] + a0_4 * c_h0a[8];
    float z_y1 = b0_0 * c_h0b[0] + b0_1 * c_h0b[2] + b0_2 * c_h0b[4] + b0_3 * c_h0b[6] + b0_4 * c_h0b[8];
    float z_y2 = a0_0 * c_h0a[1] + a0_1 * c_h0a[3] + a0_2 * c_h0a[5] + a0_3 * c_h0a[7] + a0_4 * c_h0a[9];
    float z_y3 = b0_0 * c_h0b[1] + b0_1 * c_h0b[3] + b0_2 * c_h0b[5] + b0_3 * c_h0b[7] + b0_4 * c_h0b[9];

    float a1_4 = c2q_sample(hp, im3, x, h2, w2, b0, b1);
    float a1_3 = c2q_sample(hp, im1, x, h2, w2, b0, b1);
    float a1_2 = c2q_sample(hp, ip1, x, h2, w2, b0, b1);
    float a1_1 = c2q_sample(hp, ip3, x, h2, w2, b0, b1);
    float a1_0 = c2q_sample(hp, ip5, x, h2, w2, b0, b1);

    float b1_4 = c2q_sample(hp, im4, x, h2, w2, b0, b1);
    float b1_3 = c2q_sample(hp, im2, x, h2, w2, b0, b1);
    float b1_2 = c2q_sample(hp, i0,  x, h2, w2, b0, b1);
    float b1_1 = c2q_sample(hp, ip2, x, h2, w2, b0, b1);
    float b1_0 = c2q_sample(hp, ip4, x, h2, w2, b0, b1);

    float h_y0 = a1_0 * c_h1a[0] + a1_1 * c_h1a[2] + a1_2 * c_h1a[4] + a1_3 * c_h1a[6] + a1_4 * c_h1a[8];
    float h_y1 = b1_0 * c_h1b[0] + b1_1 * c_h1b[2] + b1_2 * c_h1b[4] + b1_3 * c_h1b[6] + b1_4 * c_h1b[8];
    float h_y2 = a1_0 * c_h1a[1] + a1_1 * c_h1a[3] + a1_2 * c_h1a[5] + a1_3 * c_h1a[7] + a1_4 * c_h1a[9];
    float h_y3 = b1_0 * c_h1b[1] + b1_1 * c_h1b[3] + b1_2 * c_h1b[5] + b1_3 * c_h1b[7] + b1_4 * c_h1b[9];

    size_t y0 = (size_t)(4 * j);
    dst[y0       * (size_t)c + (size_t)x] = z_y0 + h_y0;
    dst[(y0 + 1) * (size_t)c + (size_t)x] = z_y1 + h_y1;
    dst[(y0 + 2) * (size_t)c + (size_t)x] = z_y2 + h_y2;
    dst[(y0 + 3) * (size_t)c + (size_t)x] = z_y3 + h_y3;
}

__kernel void dtcwt_colifilt_h_add_trim(
    __global const float* src0,
    __global const float* src1,
    __global float* dst,
    const int r,
    const int c,
    const int exp_r,
    const int exp_c,
    const int off_y,
    const int off_x)
{
    int j = get_global_id(0);
    int y = get_global_id(1);
    int groups = c >> 1;
    if (j >= groups || y >= r) return;
    if (y < off_y || y >= off_y + exp_r) return;

    int im4 = reflect_idx(2 * j - 4, c);
    int im2 = reflect_idx(2 * j - 2, c);
    int i0  = reflect_idx(2 * j, c);
    int ip2 = reflect_idx(2 * j + 2, c);
    int ip4 = reflect_idx(2 * j + 4, c);

    int im3 = reflect_idx(2 * j - 3, c);
    int im1 = reflect_idx(2 * j - 1, c);
    int ip1 = reflect_idx(2 * j + 1, c);
    int ip3 = reflect_idx(2 * j + 3, c);
    int ip5 = reflect_idx(2 * j + 5, c);

    size_t row = (size_t)y * (size_t)c;

    float a0_4 = src0[row + (size_t)im4];
    float a0_3 = src0[row + (size_t)im2];
    float a0_2 = src0[row + (size_t)i0 ];
    float a0_1 = src0[row + (size_t)ip2];
    float a0_0 = src0[row + (size_t)ip4];

    float b0_4 = src0[row + (size_t)im3];
    float b0_3 = src0[row + (size_t)im1];
    float b0_2 = src0[row + (size_t)ip1];
    float b0_1 = src0[row + (size_t)ip3];
    float b0_0 = src0[row + (size_t)ip5];

    float a1_4 = src1[row + (size_t)im3];
    float a1_3 = src1[row + (size_t)im1];
    float a1_2 = src1[row + (size_t)ip1];
    float a1_1 = src1[row + (size_t)ip3];
    float a1_0 = src1[row + (size_t)ip5];

    float b1_4 = src1[row + (size_t)im4];
    float b1_3 = src1[row + (size_t)im2];
    float b1_2 = src1[row + (size_t)i0 ];
    float b1_1 = src1[row + (size_t)ip2];
    float b1_0 = src1[row + (size_t)ip4];

    float vals[4];
    vals[0] = (a0_0 * c_h0a[0] + a0_1 * c_h0a[2] + a0_2 * c_h0a[4] + a0_3 * c_h0a[6] + a0_4 * c_h0a[8])
            + (a1_0 * c_h1a[0] + a1_1 * c_h1a[2] + a1_2 * c_h1a[4] + a1_3 * c_h1a[6] + a1_4 * c_h1a[8]);
    vals[1] = (b0_0 * c_h0b[0] + b0_1 * c_h0b[2] + b0_2 * c_h0b[4] + b0_3 * c_h0b[6] + b0_4 * c_h0b[8])
            + (b1_0 * c_h1b[0] + b1_1 * c_h1b[2] + b1_2 * c_h1b[4] + b1_3 * c_h1b[6] + b1_4 * c_h1b[8]);
    vals[2] = (a0_0 * c_h0a[1] + a0_1 * c_h0a[3] + a0_2 * c_h0a[5] + a0_3 * c_h0a[7] + a0_4 * c_h0a[9])
            + (a1_0 * c_h1a[1] + a1_1 * c_h1a[3] + a1_2 * c_h1a[5] + a1_3 * c_h1a[7] + a1_4 * c_h1a[9]);
    vals[3] = (b0_0 * c_h0b[1] + b0_1 * c_h0b[3] + b0_2 * c_h0b[5] + b0_3 * c_h0b[7] + b0_4 * c_h0b[9])
            + (b1_0 * c_h1b[1] + b1_1 * c_h1b[3] + b1_2 * c_h1b[5] + b1_3 * c_h1b[7] + b1_4 * c_h1b[9]);

    size_t dst_row = (size_t)(y - off_y) * (size_t)exp_c;
    int x0 = 4 * j;
    for (int d = 0; d < 4; ++d) {
        int col = x0 + d;
        if (col >= off_x && col < off_x + exp_c) {
            dst[dst_row + (size_t)(col - off_x)] = vals[d];
        }
    }
}

__kernel void dtcwt_max3x3_argmax(
    __global const float* mag,
    __global float* best_act,
    __global int* best_idx,
    const int h,
    const int w,
    const int frame_idx,
    const int first_frame)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    int ym = h == 1 ? 0 : (y == 0 ? 1 : y - 1);
    int yp = h == 1 ? 0 : (y == h - 1 ? h - 2 : y + 1);
    int xm = w == 1 ? 0 : (x == 0 ? 1 : x - 1);
    int xp = w == 1 ? 0 : (x == w - 1 ? w - 2 : x + 1);

    size_t rm = (size_t)ym * (size_t)w;
    size_t r0 = (size_t)y  * (size_t)w;
    size_t rp = (size_t)yp * (size_t)w;

    float m = mag[rm + (size_t)xm];
    m = fmax(m, mag[rm + (size_t)x]);
    m = fmax(m, mag[rm + (size_t)xp]);
    m = fmax(m, mag[r0 + (size_t)xm]);
    m = fmax(m, mag[r0 + (size_t)x]);
    m = fmax(m, mag[r0 + (size_t)xp]);
    m = fmax(m, mag[rp + (size_t)xm]);
    m = fmax(m, mag[rp + (size_t)x]);
    m = fmax(m, mag[rp + (size_t)xp]);

    size_t pos = r0 + (size_t)x;
    if (first_frame || m > best_act[pos]) {
        best_act[pos] = m;
        best_idx[pos] = frame_idx;
    }
}

__kernel void dtcwt_consistent_argmax_3x3(
    __global const int* idx,
    __global int* out,
    const int h,
    const int w)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    int ym = y == 0 ? 0 : y - 1;
    int yp = y == h - 1 ? h - 1 : y + 1;
    int xm = x == 0 ? 0 : x - 1;
    int xp = x == w - 1 ? w - 1 : x + 1;

    size_t rm = (size_t)ym * (size_t)w;
    size_t r0 = (size_t)y  * (size_t)w;
    size_t rp = (size_t)yp * (size_t)w;

    int vals[9];
    vals[0] = idx[rm + (size_t)xm];
    vals[1] = idx[rm + (size_t)x];
    vals[2] = idx[rm + (size_t)xp];
    vals[3] = idx[r0 + (size_t)xm];
    vals[4] = idx[r0 + (size_t)x];
    vals[5] = idx[r0 + (size_t)xp];
    vals[6] = idx[rp + (size_t)xm];
    vals[7] = idx[rp + (size_t)x];
    vals[8] = idx[rp + (size_t)xp];

    int all_eq = 1;
    for (int a = 1; a < 9; ++a) {
        if (vals[a] != vals[0]) {
            all_eq = 0;
            break;
        }
    }
    if (all_eq) {
        out[r0 + (size_t)x] = vals[4];
        return;
    }

    int best_c = 0;
    int best_l = vals[0];
    for (int a = 0; a < 9; ++a) {
        int la = vals[a];
        int cnt = 1;
        for (int b = a + 1; b < 9; ++b) {
            if (vals[b] == la) cnt++;
        }
        if (cnt > best_c || (cnt == best_c && la < best_l)) {
            best_c = cnt;
            best_l = la;
        }
    }
    out[r0 + (size_t)x] = best_l;
}

__kernel void dtcwt_mark_active(
    __global const int* decision,
    volatile __global uint* active_flags,
    const int n_pixels,
    const int lvl,
    const int n_levels)
{
    int gid = get_global_id(0);
    if (gid >= n_pixels) return;
    int f = decision[gid];
    if (gid > 0 && decision[gid - 1] == f) return;
    atomic_or(&active_flags[(size_t)f * (size_t)n_levels + (size_t)lvl], 1u);
}

__kernel void dtcwt_lp_mask_resize(
    __global const int* coarse_idx,
    __global float* dst,
    const int src_h,
    const int src_w,
    const int dst_h,
    const int dst_w,
    const int frame_idx)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= dst_w || y >= dst_h) return;

    float fy = ((float)y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    float fx = ((float)x + 0.5f) * ((float)src_w / (float)dst_w) - 0.5f;

    int y0 = (int)floor(fy);
    float dy = fy - (float)y0;
    if (y0 < 0) { y0 = 0; dy = 0.0f; }
    else if (y0 >= src_h - 1) { y0 = src_h - 1; dy = 0.0f; }
    int y1 = (y0 + 1 < src_h) ? (y0 + 1) : y0;

    int x0 = (int)floor(fx);
    float dx = fx - (float)x0;
    if (x0 < 0) { x0 = 0; dx = 0.0f; }
    else if (x0 >= src_w - 1) { x0 = src_w - 1; dx = 0.0f; }
    int x1 = (x0 + 1 < src_w) ? (x0 + 1) : x0;

    float v00 = (coarse_idx[(size_t)y0 * (size_t)src_w + (size_t)x0] == frame_idx) ? 1.0f : 0.0f;
    float v01 = (coarse_idx[(size_t)y0 * (size_t)src_w + (size_t)x1] == frame_idx) ? 1.0f : 0.0f;
    float v10 = (coarse_idx[(size_t)y1 * (size_t)src_w + (size_t)x0] == frame_idx) ? 1.0f : 0.0f;
    float v11 = (coarse_idx[(size_t)y1 * (size_t)src_w + (size_t)x1] == frame_idx) ? 1.0f : 0.0f;

    float top = v00 + dx * (v01 - v00);
    float bot = v10 + dx * (v11 - v10);
    dst[(size_t)y * (size_t)dst_w + (size_t)x] = top + dy * (bot - top);
}

__kernel void dtcwt_gauss13_h(
    __global const float* src,
    __global float* dst,
    const int h,
    const int w)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;
    size_t row = (size_t)y * (size_t)w;
    float acc = 0.0f;
    for (int k = -6; k <= 6; ++k) {
        acc += src[row + (size_t)reflect101_idx(x + k, w)] * c_gauss13[k + 6];
    }
    dst[row + (size_t)x] = acc;
}

__kernel void dtcwt_gauss13_v_to_w(
    __global const float* src,
    __global float* w_out,
    __global float* w_sum,
    const int h,
    const int w)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;
    float acc = 0.0f;
    for (int k = -6; k <= 6; ++k) {
        acc += src[(size_t)reflect101_idx(y + k, h) * (size_t)w + (size_t)x] * c_gauss13[k + 6];
    }
    size_t pos = (size_t)y * (size_t)w + (size_t)x;
    w_out[pos] = acc;
    w_sum[pos] += acc;
}

__kernel void dtcwt_accum_lp_1ch(
    __global const float* lp_src,
    __global const float* weight,
    __global float* lp_dst,
    const int n_pixels)
{
    int pos = get_global_id(0);
    if (pos >= n_pixels) return;
    lp_dst[pos] += weight[pos] * lp_src[pos];
}

__kernel void dtcwt_div_lp_weight(
    __global float* lp_dst,
    __global const float* w_sum,
    const int n_pixels)
{
    int pos = get_global_id(0);
    if (pos >= n_pixels) return;
    float d = w_sum[pos];
    lp_dst[pos] /= (d > 1e-6f ? d : 1e-6f);
}

__kernel void dtcwt_pack_channel_u8(
    __global const float* rec,
    __global uchar* out,
    const int h,
    const int w,
    const int rec_w,
    const int out_ch)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;
    float v = rec[(size_t)y * (size_t)rec_w + (size_t)x] * 255.0f;
    v = clamp(v, 0.0f, 255.0f);
    out[((size_t)y * (size_t)w + (size_t)x) * 3 + (size_t)out_ch] = (uchar)v;
}

__kernel void dtcwt_pack_channel_f32(
    __global const float* rec,
    __global float* out,
    const int h,
    const int w,
    const int rec_w,
    const int out_ch)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;
    float v = rec[(size_t)y * (size_t)rec_w + (size_t)x] * 255.0f;
    v = clamp(v, 0.0f, 255.0f);
    out[((size_t)y * (size_t)w + (size_t)x) * 3 + (size_t)out_ch] = v;
}

__kernel void dtcwt_pack_gray_u8(
    __global const float* rec,
    __global uchar* out,
    const int h,
    const int w,
    const int rec_w)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;
    float v = rec[(size_t)y * (size_t)rec_w + (size_t)x] * 255.0f;
    uchar q = (uchar)clamp(v, 0.0f, 255.0f);
    size_t ob = ((size_t)y * (size_t)w + (size_t)x) * 3;
    out[ob] = q;
    out[ob + 1] = q;
    out[ob + 2] = q;
}

__kernel void dtcwt_pack_gray_f32(
    __global const float* rec,
    __global float* out,
    const int h,
    const int w,
    const int rec_w)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;
    float v = clamp(rec[(size_t)y * (size_t)rec_w + (size_t)x] * 255.0f, 0.0f, 255.0f);
    size_t ob = ((size_t)y * (size_t)w + (size_t)x) * 3;
    out[ob] = v;
    out[ob + 1] = v;
    out[ob + 2] = v;
}
'''

_KERNEL_NAMES = (
    'dtcwt_extract_luma_u8',
    'dtcwt_extract_luma_u16',
    'dtcwt_extract_luma_f32',
    'dtcwt_extract_chan_u8',
    'dtcwt_extract_chan_u16',
    'dtcwt_extract_chan_f32',
    'dtcwt_colfilter_v_pair',
    'dtcwt_colfilter_h5',
    'dtcwt_level0_h_mag',
    'dtcwt_level0_h_select',
    'dtcwt_coldfilt_v_pair',
    'dtcwt_coldfilt_h_lo',
    'dtcwt_levelk_h_mag',
    'dtcwt_levelk_h_select',
    'dtcwt_colfilter_v_add0_z_hp',
    'dtcwt_colfilter_v_add0_hp_hp',
    'dtcwt_colfilter_h_add0',
    'dtcwt_colifilt_v_add_z_hp',
    'dtcwt_colifilt_v_add_hp_hp',
    'dtcwt_colifilt_h_add_trim',
    'dtcwt_max3x3_argmax',
    'dtcwt_consistent_argmax_3x3',
    'dtcwt_mark_active',
    'dtcwt_lp_mask_resize',
    'dtcwt_gauss13_h',
    'dtcwt_gauss13_v_to_w',
    'dtcwt_accum_lp_1ch',
    'dtcwt_div_lp_weight',
    'dtcwt_pack_channel_u8',
    'dtcwt_pack_channel_f32',
    'dtcwt_pack_gray_u8',
    'dtcwt_pack_gray_f32',
)

_cache_lock = Lock()
_PROGRAM_CACHE = None


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import (
        _build_cached_program, _fp32_build_options, _get_shared_context,
    )

    key = getattr(device, 'int_ptr', None)
    with _cache_lock:
        ctx = _get_shared_context(cl, device)
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key and _PROGRAM_CACHE[1] is ctx:
            return _PROGRAM_CACHE[1:]
        queue = cl.CommandQueue(ctx, device=device)
        options = _fp32_build_options(cl, device)
        program = _build_cached_program(cl, device, ctx, _DTCWT_K, options=options)
        kernels = {name: cl.Kernel(program, name) for name in _KERNEL_NAMES}
        _PROGRAM_CACHE = (key, ctx, queue, kernels)
        return ctx, queue, kernels


def _invalidate_program_cache(context=None):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _invalidate_shared_context

    with _cache_lock:
        _PROGRAM_CACHE = None
        _invalidate_shared_context(context)


@dataclass(frozen=True)
class _LevelSpec:
    in_h: int
    in_w: int
    pad_h: int
    pad_w: int
    pr: int
    pc: int
    lolo_h: int
    lolo_w: int
    hp_h: int
    hp_w: int
    exp_r: int
    exp_c: int


def _plan_levels(orig_h: int, orig_w: int, nlevels: int) -> list[_LevelSpec]:
    r0 = orig_h + (orig_h & 1)
    c0 = orig_w + (orig_w & 1)
    specs: list[_LevelSpec] = [
        _LevelSpec(
            in_h=r0, in_w=c0, pad_h=r0, pad_w=c0, pr=0, pc=0,
            lolo_h=r0, lolo_w=c0, hp_h=r0 >> 1, hp_w=c0 >> 1,
            exp_r=r0, exp_c=c0,
        )
    ]
    prev_h, prev_w = r0, c0
    for _ in range(1, nlevels):
        pr = 1 if (prev_h % 4 != 0) else 0
        pc = 1 if (prev_w % 4 != 0) else 0
        pad_h = prev_h + 2 * pr
        pad_w = prev_w + 2 * pc
        lolo_h = pad_h >> 1
        lolo_w = pad_w >> 1
        specs.append(
            _LevelSpec(
                in_h=prev_h, in_w=prev_w, pad_h=pad_h, pad_w=pad_w, pr=pr, pc=pc,
                lolo_h=lolo_h, lolo_w=lolo_w,
                hp_h=lolo_h >> 1, hp_w=lolo_w >> 1,
                exp_r=prev_h, exp_c=prev_w,
            )
        )
        prev_h, prev_w = lolo_h, lolo_w
    return specs


def _estimate_vram_bytes(specs: list[_LevelSpec], count: int, src_bytes: int, out_bytes: int):
    """Return (scratch_base, per_ch_accum_bytes, need_1ch, need_3ch, largest_buf)."""
    s0 = specs[0]
    r0, c0 = s0.in_h, s0.in_w
    hp_level_bytes = [sp.hp_h * sp.hp_w * 6 * 8 for sp in specs]
    pyr_bytes = sum(hp_level_bytes)
    dec_bytes = sum(sp.hp_h * sp.hp_w * 4 for sp in specs)
    lp_n = specs[-1].lolo_h * specs[-1].lolo_w
    quad_bytes = r0 * c0 * 4
    lolo_b_elems = max(
        s0.hp_h * s0.hp_w,
        specs[1].lolo_h * specs[1].lolo_w if len(specs) > 1 else 0,
    )
    lolo_b_bytes = max(1, lolo_b_elems * 4)
    active_bytes = count * len(specs) * np.dtype(np.uint32).itemsize

    # Pass 1 live: src + 3*quad (plane/lolo_a, lo, hi) + lolo_b + 2*dec + active
    pass1_bytes = (
        src_bytes + 3 * quad_bytes + lolo_b_bytes + 2 * dec_bytes
        + active_bytes + 16 * 1024 * 1024
    )
    # Pass 2 live: src + 3*quad + lolo_b + dec + 3*lp + K*per_ch
    # Inverse live (after freeing src, dec, w_tmp, w_buf): out + 3*quad + lolo_b + lp + K*per_ch
    pass2_inv_fixed = (
        max(src_bytes + dec_bytes + 3 * lp_n * 4, out_bytes + lp_n * 4)
        + 3 * quad_bytes + lolo_b_bytes + 16 * 1024 * 1024
    )
    per_ch_accum_bytes = pyr_bytes + lp_n * 4
    need_1ch = max(pass1_bytes, pass2_inv_fixed + per_ch_accum_bytes)
    need_3ch = max(pass1_bytes, pass2_inv_fixed + 3 * per_ch_accum_bytes)
    largest = max(src_bytes, out_bytes, max(hp_level_bytes), quad_bytes, lolo_b_bytes, active_bytes)
    return pass2_inv_fixed, per_ch_accum_bytes, need_1ch, need_3ch, largest


def _upload_kind(dtype):
    if dtype == np.uint8:
        return 'u8'
    if dtype == np.uint16:
        return 'u16'
    return 'f32'


def _source_staging_bytes(frame):
    from core.source_frames import frame_has_mapped_pixels, frame_resolution_bytes

    resolution = frame_resolution_bytes(frame)
    if frame_has_mapped_pixels(frame):
        resolution = max(resolution, frame.nbytes)
    packed = 0 if isinstance(frame, np.ndarray) and frame.flags.c_contiguous else frame.nbytes
    return resolution + packed + frame.nbytes


def _dtcwt_prefetch_slots(host_budget: int, host_staging: int, frame_bytes: int, count: int) -> int:
    from fusion_methods._openfocus_execution import _current

    job = _current.get()
    if job is not None and getattr(job, 'thread_count', None) == 1:
        return 0
    if count < 2 or frame_bytes <= 0 or host_staging + frame_bytes > host_budget:
        return 0
    headroom = max(0, int(host_budget) - int(host_staging))
    cpu_cap = max(1, min(6, (os.cpu_count() or 2) // 2))
    return max(0, min(cpu_cap, int(headroom // int(frame_bytes)), int(count) - 1))


def _iter_prefetched_indices(indices, prepare_fn, *, slots: int, should_cancel=None):
    total = len(indices)
    if total == 0:
        return
    if slots <= 0 or total < 2:
        for frame_index in indices:
            _check_cancel(should_cancel)
            result = prepare_fn(frame_index, should_cancel)
            _check_cancel(should_cancel)
            yield frame_index, result
            del result
        return

    from fusion_methods._openfocus_execution import _worker_native_limit

    stop = Event()

    def cancelled():
        return stop.is_set() or (should_cancel is not None and should_cancel())

    def submit_item(pool, frame_index):
        ctx = contextvars.copy_context()
        return pool.submit(ctx.run, prepare_fn, frame_index, cancelled)

    _check_cancel(should_cancel)
    workers = min(slots, 6)
    with ThreadPoolExecutor(max_workers=workers, initializer=_worker_native_limit) as pool:
        pending = deque()
        next_pos = 0
        try:
            while next_pos < min(slots, total):
                pending.append((indices[next_pos], submit_item(pool, indices[next_pos])))
                next_pos += 1
            while pending:
                frame_index, future = pending[0]
                while not future.done():
                    _check_cancel(should_cancel)
                    wait((future,), timeout=0.05)
                _check_cancel(should_cancel)
                pending.popleft()
                result = future.result()
                while next_pos < total and len(pending) < slots:
                    pending.append((indices[next_pos], submit_item(pool, indices[next_pos])))
                    next_pos += 1
                yield frame_index, result
                del result
        finally:
            stop.set()
            for _, future in pending:
                future.cancel()


class GpuDtcwt:
    """OpenCL 1.2 device workspace and kernels for DTCWT focus fusion."""

    def __init__(self):
        from fusion_methods._openfocus_gpu import GpuError, _device, opencl_device_name
        try:
            import pyopencl as cl
        except (ImportError, OSError) as exc:
            raise GpuError('OpenCL loader unavailable') from exc
        try:
            device = _device(cl)
            if device is None:
                raise GpuError('No available OpenCL GPU')
            self.device_name = opencl_device_name(device)
        except cl.Error as exc:
            raise GpuError('OpenCL GPU discovery failed') from exc
        self.cl = cl
        self.device = device
        self.mf = cl.mem_flags

    def _run(self, kernel, gsize, *args, wait_for=None):
        from fusion_methods._openfocus_gpu import _enqueue_kernel

        g = gsize if isinstance(gsize, tuple) else (gsize,)
        return _enqueue_kernel(kernel, self.q, g, None, *args, wait_for=wait_for)

    def fuse(self, images, nlevels, should_cancel=None, progress_callback=None):
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
            _check_cancel(should_cancel)
            self.q = None
            self.tq = None
            try:
                return self._fuse_locked(
                    images, nlevels, should_cancel,
                    progress if progress_callback is not None else None)
            except CallbackFailure as exc:
                self._drain_queue()
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                _invalidate_program_cache(getattr(self, 'ctx', None))
                self._drain_queue()
                raise GpuError('OpenCL GPU DTCWT execution failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.k = None
            self.tq = None
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _drain_queue(self):
        from fusion_methods._openfocus_gpu import _drain_command_queue

        ctx = getattr(self, 'ctx', None)
        tq = getattr(self, 'tq', None)
        q = getattr(self, 'q', None)
        if tq is not None and tq is not q:
            _drain_command_queue(tq, ctx)
        if q is not None:
            try:
                q.finish()
            except self.cl.Error:
                _invalidate_program_cache(ctx)

    def _forward_activity(self, plane_buf, specs: list[_LevelSpec],
                          best_act: list, best_idx: list,
                          frame_idx: int, first_frame: int, scratch: dict):
        """Stream all forward levels for Pass 1 directly into best_act/best_idx."""
        nlevels = len(specs)
        s0 = specs[0]
        r0, c0 = s0.in_h, s0.in_w
        lo = scratch['lo']
        hi = scratch['hi']
        lolo_b = scratch['lolo_b']

        self._run(self.k['dtcwt_colfilter_v_pair'], (c0, r0),
                  plane_buf, lo, hi, np.int32(r0), np.int32(c0))
        self._run(self.k['dtcwt_level0_h_mag'], (s0.hp_w, s0.hp_h),
                  lo, hi, lolo_b, np.int32(s0.hp_h), np.int32(s0.hp_w), np.int32(c0))
        last_ev = self._run(
            self.k['dtcwt_max3x3_argmax'], (s0.hp_w, s0.hp_h),
            lolo_b, best_act[0], best_idx[0],
            np.int32(s0.hp_h), np.int32(s0.hp_w),
            np.int32(frame_idx), np.int32(first_frame),
        )
        if nlevels == 1:
            return last_ev

        # Reuse plane_buf as lolo_a since plane_buf is no longer read after colfilter_v_pair.
        self._run(self.k['dtcwt_colfilter_h5'], (c0, r0),
                  lo, plane_buf, np.int32(r0), np.int32(c0), np.int32(1))
        cur_lolo = plane_buf
        alt_lolo = lolo_b

        for lvl in range(1, nlevels):
            sk = specs[lvl]
            groups_r = sk.pad_h >> 2
            groups_c = sk.pad_w >> 2
            r2 = sk.lolo_h

            self._run(self.k['dtcwt_coldfilt_v_pair'], (sk.in_w, groups_r),
                      cur_lolo, lo, hi,
                      np.int32(sk.in_h), np.int32(sk.in_w),
                      np.int32(sk.pad_h), np.int32(sk.pr))
            # Safe to write mag into cur_lolo since coldfilt_v_pair finished reading it.
            self._run(self.k['dtcwt_levelk_h_mag'], (sk.hp_w, sk.hp_h),
                      lo, hi, cur_lolo,
                      np.int32(sk.hp_h), np.int32(sk.hp_w),
                      np.int32(sk.in_w), np.int32(sk.pad_w), np.int32(sk.pc))
            last_ev = self._run(
                self.k['dtcwt_max3x3_argmax'], (sk.hp_w, sk.hp_h),
                cur_lolo, best_act[lvl], best_idx[lvl],
                np.int32(sk.hp_h), np.int32(sk.hp_w),
                np.int32(frame_idx), np.int32(first_frame),
            )
            if lvl + 1 < nlevels:
                self._run(self.k['dtcwt_coldfilt_h_lo'], (groups_c, r2),
                          lo, alt_lolo,
                          np.int32(r2), np.int32(sk.in_w),
                          np.int32(sk.pad_w), np.int32(sk.pc))
                cur_lolo, alt_lolo = alt_lolo, cur_lolo

        return last_ev

    def _forward_select(self, plane_buf, specs: list[_LevelSpec], max_lvl: int,
                        need_lp: bool, active_row, decisions: list,
                        fused_hp: list, frame_idx: int, scratch: dict):
        """Stream forward levels for Pass 2, writing only winning coefficients."""
        s0 = specs[0]
        r0, c0 = s0.in_h, s0.in_w
        lo = scratch['lo']
        hi = scratch['hi']
        lolo_b = scratch['lolo_b']
        need_lolo0 = (max_lvl >= 1) or (need_lp and len(specs) == 1)

        self._run(self.k['dtcwt_colfilter_v_pair'], (c0, r0),
                  plane_buf, lo, hi, np.int32(r0), np.int32(c0))
        last_ev = None
        if active_row[0]:
            last_ev = self._run(
                self.k['dtcwt_level0_h_select'], (s0.hp_w, s0.hp_h),
                lo, hi, decisions[0], fused_hp[0],
                np.int32(s0.hp_h), np.int32(s0.hp_w),
                np.int32(c0), np.int32(frame_idx),
            )
        if not need_lolo0:
            return None, last_ev

        last_ev = self._run(
            self.k['dtcwt_colfilter_h5'], (c0, r0),
            lo, plane_buf, np.int32(r0), np.int32(c0), np.int32(1),
        )
        cur_lolo = plane_buf
        alt_lolo = lolo_b

        for lvl in range(1, max_lvl + 1):
            sk = specs[lvl]
            groups_r = sk.pad_h >> 2
            groups_c = sk.pad_w >> 2
            r2 = sk.lolo_h
            need_next_lolo = (lvl < max_lvl) or need_lp

            self._run(self.k['dtcwt_coldfilt_v_pair'], (sk.in_w, groups_r),
                      cur_lolo, lo, hi,
                      np.int32(sk.in_h), np.int32(sk.in_w),
                      np.int32(sk.pad_h), np.int32(sk.pr))
            if active_row[lvl]:
                last_ev = self._run(
                    self.k['dtcwt_levelk_h_select'], (sk.hp_w, sk.hp_h),
                    lo, hi, decisions[lvl], fused_hp[lvl],
                    np.int32(sk.hp_h), np.int32(sk.hp_w),
                    np.int32(sk.in_w), np.int32(sk.pad_w),
                    np.int32(sk.pc), np.int32(frame_idx),
                )
            if need_next_lolo:
                last_ev = self._run(
                    self.k['dtcwt_coldfilt_h_lo'], (groups_c, r2),
                    lo, alt_lolo,
                    np.int32(r2), np.int32(sk.in_w),
                    np.int32(sk.pad_w), np.int32(sk.pc),
                )
                cur_lolo, alt_lolo = alt_lolo, cur_lolo

        return (cur_lolo if need_lp else None), last_ev

    def _inverse_plane(self, fused_lp, fused_hp: list, specs: list[_LevelSpec],
                       scratch: dict):
        """Reconstruct fused image plane with fused c2q inverse kernels."""
        nlevels = len(specs)
        z = fused_lp
        z_a = scratch['plane']
        z_b = scratch['lolo_b']
        y1 = scratch['lo']
        y2 = scratch['hi']

        for lvl in range(nlevels, 1, -1):
            sk = specs[lvl - 1]
            hp = fused_hp[lvl - 1]
            rq, cq = sk.hp_h << 1, sk.hp_w << 1
            r2, c2 = rq << 1, cq << 1

            self._run(self.k['dtcwt_colifilt_v_add_z_hp'], (cq, rq >> 1),
                      z, hp, y1, np.int32(rq), np.int32(cq),
                      np.int32(0), np.int32(5))
            self._run(self.k['dtcwt_colifilt_v_add_hp_hp'], (cq, rq >> 1),
                      hp, y2, np.int32(rq), np.int32(cq),
                      np.int32(2), np.int32(3), np.int32(1), np.int32(4))

            off_y = (r2 - sk.exp_r) >> 1
            off_x = (c2 - sk.exp_c) >> 1
            # Final level-1 output has size (r0, c0) which always requires z_a (full quad_bytes).
            if lvl == 2:
                dst = z_a
            else:
                dst = z_b if z is z_a else z_a
            self._run(self.k['dtcwt_colifilt_h_add_trim'], (cq >> 1, r2),
                      y1, y2, dst, np.int32(r2), np.int32(cq),
                      np.int32(sk.exp_r), np.int32(sk.exp_c),
                      np.int32(off_y), np.int32(off_x))
            z = dst

        # Final level 0 synthesis using near_sym_a
        s0 = specs[0]
        hp0 = fused_hp[0]
        rq, cq = s0.in_h, s0.in_w
        self._run(self.k['dtcwt_colfilter_v_add0_z_hp'], (cq, rq),
                  z, hp0, y1, np.int32(rq), np.int32(cq),
                  np.int32(0), np.int32(5))
        self._run(self.k['dtcwt_colfilter_v_add0_hp_hp'], (cq, rq),
                  hp0, y2, np.int32(rq), np.int32(cq),
                  np.int32(2), np.int32(3), np.int32(1), np.int32(4))

        rec = z_a
        self._run(self.k['dtcwt_colfilter_h_add0'], (cq, rq),
                  y1, y2, rec, np.int32(rq), np.int32(cq))
        return rec

    def _fuse_locked(self, images, nlevels, should_cancel, progress_callback):
        from dataclasses import replace
        from core import resource_policy
        from core.source_frames import resolve_frame, without_source_cache_growth
        from fusion_methods._openfocus_gpu import (
            GpuError, _PinnedUploader, _event_wait_list, _free_vram,
            _get_transfer_queue, _release_buffers, _wait_event,
        )

        if not images or len(images) < 2:
            raise ValueError('At least two images are required for fusion.')

        nlevels = int(nlevels)
        if nlevels < 1:
            raise GpuError(f'Unsupported DTCWT level count for GPU: {nlevels}')

        orig_h, orig_w = int(images[0].shape[0]), int(images[0].shape[1])
        if orig_h < 4 or orig_w < 4:
            raise GpuError(f'Image too small for GPU DTCWT: {(orig_h, orig_w)}')

        specs = _plan_levels(orig_h, orig_w, nlevels)
        for idx, sp in enumerate(specs):
            if sp.hp_h < 1 or sp.hp_w < 1:
                raise GpuError(f'Empty subband at level {idx} for GPU DTCWT')
            if idx >= 1 and (sp.pad_h < 4 or sp.pad_w < 4):
                raise GpuError(f'Subband too small at level {idx} for GPU DTCWT')

        shape = (orig_h, orig_w, 3)
        count = len(images)
        s0 = specs[0]
        r0, c0 = s0.in_h, s0.in_w
        lp_spec = specs[-1]
        lp_h, lp_w = lp_spec.lolo_h, lp_spec.lolo_w
        lp_n = lp_h * lp_w

        out_dtype = fuse_output_dtype(images)
        is_u8_out = out_dtype == np.uint8
        out_bytes = orig_h * orig_w * 3 * (1 if is_u8_out else 4)

        snapshot = resource_policy.probe_memory()
        host_budget = resource_policy.ram_ceiling(snapshot)
        source_peak = 0
        src_bytes = 0
        all_gray = True
        for frame in images:
            _check_cancel(should_cancel)
            if frame.shape[:2] != shape[:2] or frame.dtype not in (np.uint8, np.uint16, np.float32):
                raise GpuError('GPU DTCWT requires matching dimensions and uint8, uint16, or float32 samples')
            if not (frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1)):
                all_gray = False
            source_peak = max(source_peak, _source_staging_bytes(frame))
            src_bytes = max(src_bytes, frame.nbytes)
        active_bytes = count * nlevels * np.dtype(np.uint32).itemsize
        staging = max(source_peak, out_bytes * 2) + active_bytes + 16 * 1024 * 1024
        if staging > host_budget:
            raise GpuError(
                f'DTCWT GPU staging exceeds the host memory budget '
                f'({staging >> 20} MiB needed, {host_budget >> 20} MiB available)'
            )

        hp_level_bytes = [sp.hp_h * sp.hp_w * 6 * 8 for sp in specs]
        quad_bytes = r0 * c0 * 4
        lolo_b_elems = max(
            s0.hp_h * s0.hp_w,
            specs[1].lolo_h * specs[1].lolo_w if nlevels > 1 else 0,
        )
        lolo_b_bytes = max(1, lolo_b_elems * 4)

        _, _, need_1ch, need_3ch, largest = _estimate_vram_bytes(
            specs, count, src_bytes, out_bytes,
        )

        limit = int(self.device.global_mem_size) // 2
        unified_memory = self.device.host_unified_memory
        if unified_memory:
            limit = min(limit, host_budget - staging)
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError('DTCWT GPU free memory is unknown')
            snapshot = replace(snapshot, free_vram=memory[0], total_vram=memory[1])
            limit = min(limit, resource_policy.vram_budget(snapshot))

        single = int(self.device.max_mem_alloc_size)
        if largest > single:
            raise GpuError('DTCWT GPU buffer exceeds max_mem_alloc_size')

        if all_gray:
            if need_1ch > limit:
                raise GpuError('DTCWT GPU workspace exceeds the device budget')
            ch_mode_3 = False
            need_chosen = need_1ch
        elif need_3ch <= limit:
            ch_mode_3 = True
            need_chosen = need_3ch
        elif need_1ch <= limit:
            ch_mode_3 = False
            need_chosen = need_1ch
        else:
            raise GpuError('DTCWT GPU workspace exceeds the device budget')

        total_steps = 2 * count + 1
        if progress_callback is not None:
            progress_callback(0, total_steps)
        _check_cancel(should_cancel)
        self.ctx, self.q, self.k = _get_or_build_program(self.cl, self.device)
        self.tq = _get_transfer_queue(self.cl, self.device, self.ctx)

        def prepare_upload(index, cancelled):
            _check_cancel(cancelled)
            with without_source_cache_growth():
                raw = resolve_frame(images[index], should_cancel=cancelled)
            _check_cancel(cancelled)
            if raw.dtype.kind == 'f':
                if not np.isfinite(raw).all():
                    raise GpuError('Non-finite floating-point frame uses CPU fallback')
                mn, mx = float(np.min(raw)), float(np.max(raw))
                if mn < -255.0 or mx > 255.0:
                    raise GpuError('Unbounded floating-point frame uses CPU fallback')
            if raw.ndim == 2:
                channels = 1
            elif raw.ndim == 3 and raw.shape[2] == 1:
                channels = 1
            elif raw.ndim == 3 and raw.shape[2] in (3, 4):
                channels = int(raw.shape[2])
            else:
                raise ValueError(f'Unsupported image shape: {raw.shape}')
            return np.ascontiguousarray(raw), channels, _upload_kind(raw.dtype)

        prefetch_budget = host_budget - need_chosen if unified_memory else host_budget
        prefetch_slots = _dtcwt_prefetch_slots(prefetch_budget, staging, source_peak, count)
        ping_pong = (
            count >= 2
            and (need_chosen + src_bytes) <= limit
            and (staging + prefetch_slots * source_peak + 2 * src_bytes
                 + (src_bytes if unified_memory else 0)) <= prefetch_budget
        )

        src_bufs = ()
        scratch = {}
        best_act = []
        best_idx = []
        decisions = []
        active_buf = None
        w_tmp_h = w_buf = w_sum_buf = out_buf = None
        fused_hps = None
        fused_lps = None
        fused_hp = None
        fused_lp = None
        uploader = pass1_iter = pass2_iter = None
        cancel_fn = lambda: _check_cancel(should_cancel)

        try:
            src_bufs = tuple(
                self.cl.Buffer(self.ctx, self.mf.READ_ONLY, max(1, src_bytes))
                for _ in range(2 if ping_pong else 1)
            )
            scratch = {
                'plane': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, quad_bytes)),
                'lo': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, quad_bytes)),
                'hi': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, quad_bytes)),
                'lolo_b': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, lolo_b_bytes),
            }
            best_act = [
                self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, sp.hp_h * sp.hp_w * 4))
                for sp in specs
            ]
            best_idx = [
                self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, sp.hp_h * sp.hp_w * 4))
                for sp in specs
            ]
            if ping_pong:
                uploader = _PinnedUploader(
                    self.cl, self.ctx, self.q, getattr(self, 'tq', None),
                    2, src_bytes,
                )

            # Pass 1: pipelined luminance activity + strict '>' argmax
            pass1_iter = _iter_prefetched_indices(
                range(count), prepare_upload, slots=prefetch_slots,
                should_cancel=should_cancel,
            )
            slot_events = [None, None]
            slot_hosts = [None, None]
            for index, (h_src, ch, kind) in pass1_iter:
                slot = index & 1 if ping_pong else 0
                prev_slot_ev = slot_events[slot]
                if prev_slot_ev is not None and (uploader is None or not uploader.separate_queue):
                    _wait_event(prev_slot_ev, cancel_fn)
                    slot_events[slot] = None
                    slot_hosts[slot] = None
                    prev_slot_ev = None
                _check_cancel(should_cancel)
                src_buf = src_bufs[slot]
                if uploader is not None:
                    upload = uploader.upload(
                        slot, src_buf, h_src,
                        wait_for=_event_wait_list(prev_slot_ev), cancel=cancel_fn,
                    )
                    wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                else:
                    upload = self.cl.enqueue_copy(
                        self.q, src_buf, async_upload_view(h_src), is_blocking=False,
                    )
                    wait_upload = None

                extract_ev = self._run(
                    self.k[f'dtcwt_extract_luma_{kind}'], (c0, r0),
                    src_buf, scratch['plane'],
                    np.int32(orig_h), np.int32(orig_w),
                    np.int32(r0), np.int32(c0), np.int32(ch),
                    wait_for=wait_upload,
                )
                self._forward_activity(
                    scratch['plane'], specs, best_act, best_idx,
                    index, 1 if index == 0 else 0, scratch,
                )
                # src_buf is free as soon as extract_luma completes!
                slot_ev = extract_ev
                if ping_pong and slot_ev is not None and hasattr(slot_ev, 'wait'):
                    if hasattr(self.q, 'flush'):
                        self.q.flush()
                    slot_events[slot] = slot_ev
                    slot_hosts[slot] = (h_src, upload)
                else:
                    self.q.finish()
                    del h_src, upload
                if progress_callback is not None:
                    progress_callback(index + 1, total_steps)

            pass1_iter.close()
            pass1_iter = None
            if ping_pong:
                if uploader is not None:
                    uploader.finish(cancel_fn)
                for slot in (0, 1):
                    if slot_events[slot] is not None:
                        _wait_event(slot_events[slot], cancel_fn)
                        slot_events[slot] = None
                    slot_hosts[slot] = None
                self.q.finish()

            # Consistency 3x3 nearest-mode filter on decision maps (reusing best_act buffers)
            decisions = best_act
            best_act = []
            active_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, active_bytes)
            self.cl.enqueue_fill_buffer(self.q, active_buf, np.uint32(0), 0, active_bytes)
            for lvl, sp in enumerate(specs):
                _check_cancel(should_cancel)
                self._run(
                    self.k['dtcwt_consistent_argmax_3x3'], (sp.hp_w, sp.hp_h),
                    best_idx[lvl], decisions[lvl],
                    np.int32(sp.hp_h), np.int32(sp.hp_w),
                )
                self._run(
                    self.k['dtcwt_mark_active'], (sp.hp_h * sp.hp_w,),
                    decisions[lvl], active_buf,
                    np.int32(sp.hp_h * sp.hp_w), np.int32(lvl), np.int32(nlevels),
                )
            _release_buffers(best_idx)
            best_idx = []

            active_flags = np.empty((count, nlevels), dtype=np.uint32)
            self.cl.enqueue_copy(self.q, active_flags, active_buf, is_blocking=True)
            _release_buffers(active_buf)
            active_buf = None
            _check_cancel(should_cancel)

            coarse_active = [bool(active_flags[i, nlevels - 1]) for i in range(count)]
            max_needed = []
            for i in range(count):
                highest = -1
                for lvl in range(nlevels):
                    if active_flags[i, lvl]:
                        highest = lvl
                max_needed.append(highest)
            active_indices = [
                i for i in range(count) if max_needed[i] >= 0 or coarse_active[i]
            ]

            coarse_spec = specs[-1]
            lp_buf_bytes = lp_n * 4
            w_tmp_h = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
            w_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
            w_sum_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
            zero_f32 = np.float32(0.0)

            def _compute_coarse_weight(index):
                self._run(
                    self.k['dtcwt_lp_mask_resize'], (lp_w, lp_h),
                    decisions[-1], scratch['plane'],
                    np.int32(coarse_spec.hp_h), np.int32(coarse_spec.hp_w),
                    np.int32(lp_h), np.int32(lp_w),
                    np.int32(index),
                )
                self._run(
                    self.k['dtcwt_gauss13_h'], (lp_w, lp_h),
                    scratch['plane'], w_tmp_h, np.int32(lp_h), np.int32(lp_w),
                )
                self._run(
                    self.k['dtcwt_gauss13_v_to_w'], (lp_w, lp_h),
                    w_tmp_h, w_buf, w_sum_buf, np.int32(lp_h), np.int32(lp_w),
                )

            p2_slots = _dtcwt_prefetch_slots(
                prefetch_budget, staging, source_peak, len(active_indices),
            )

            if all_gray:
                fused_hp = [
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, b))
                    for b in hp_level_bytes
                ]
                fused_lp = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
                self.cl.enqueue_fill_buffer(self.q, fused_lp, zero_f32, 0, lp_buf_bytes)
                for lvl, b in enumerate(hp_level_bytes):
                    self.cl.enqueue_fill_buffer(self.q, fused_hp[lvl], zero_f32, 0, b)
                self.cl.enqueue_fill_buffer(self.q, w_sum_buf, zero_f32, 0, lp_buf_bytes)

                pass2_iter = _iter_prefetched_indices(
                    active_indices, prepare_upload, slots=p2_slots,
                    should_cancel=should_cancel,
                )
                slot_events = [None, None]
                slot_hosts = [None, None]
                next_progress = 0
                for step_pos, (index, (h_src, ch, kind)) in enumerate(pass2_iter):
                    slot = step_pos & 1 if ping_pong else 0
                    prev_slot_ev = slot_events[slot]
                    if prev_slot_ev is not None and (uploader is None or not uploader.separate_queue):
                        _wait_event(prev_slot_ev, cancel_fn)
                        slot_events[slot] = None
                        slot_hosts[slot] = None
                        prev_slot_ev = None
                    _check_cancel(should_cancel)
                    src_buf = src_bufs[slot]
                    if uploader is not None:
                        upload = uploader.upload(
                            slot, src_buf, h_src,
                            wait_for=_event_wait_list(prev_slot_ev), cancel=cancel_fn,
                        )
                        wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                    else:
                        upload = self.cl.enqueue_copy(
                            self.q, src_buf, async_upload_view(h_src), is_blocking=False,
                        )
                        wait_upload = None

                    need_lp_i = coarse_active[index]
                    if need_lp_i:
                        _compute_coarse_weight(index)
                        run_max_lvl = nlevels - 1
                    else:
                        run_max_lvl = max_needed[index]

                    extract_ev = self._run(
                        self.k[f'dtcwt_extract_chan_{kind}'], (c0, r0),
                        src_buf, scratch['plane'],
                        np.int32(orig_h), np.int32(orig_w),
                        np.int32(r0), np.int32(c0),
                        np.int32(ch), np.int32(0),
                        wait_for=wait_upload,
                    )
                    lp_cur, _ = self._forward_select(
                        scratch['plane'], specs, run_max_lvl, need_lp_i,
                        active_flags[index], decisions, fused_hp, index, scratch,
                    )
                    if need_lp_i and lp_cur is not None:
                        self._run(
                            self.k['dtcwt_accum_lp_1ch'], (lp_n,),
                            lp_cur, w_buf, fused_lp, np.int32(lp_n),
                        )
                    if ping_pong and extract_ev is not None and hasattr(extract_ev, 'wait'):
                        if hasattr(self.q, 'flush'):
                            self.q.flush()
                        slot_events[slot] = extract_ev
                        slot_hosts[slot] = (h_src, upload)
                    else:
                        self.q.finish()
                        del h_src, upload
                    if progress_callback is not None:
                        while next_progress <= index:
                            progress_callback(count + next_progress + 1, total_steps)
                            next_progress += 1

                pass2_iter.close()
                pass2_iter = None
                if ping_pong:
                    if uploader is not None:
                        uploader.finish(cancel_fn)
                    for slot in (0, 1):
                        if slot_events[slot] is not None:
                            _wait_event(slot_events[slot], cancel_fn)
                            slot_events[slot] = None
                        slot_hosts[slot] = None
                    self.q.finish()
                if progress_callback is not None:
                    while next_progress < count:
                        progress_callback(count + next_progress + 1, total_steps)
                        next_progress += 1

                _release_buffers(src_bufs, decisions, w_tmp_h, w_buf)
                src_bufs = ()
                decisions = []
                w_tmp_h = w_buf = None
                out_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, max(1, out_bytes))

                _check_cancel(should_cancel)
                self._run(self.k['dtcwt_div_lp_weight'], (lp_n,),
                          fused_lp, w_sum_buf, np.int32(lp_n))
                rec = self._inverse_plane(fused_lp, fused_hp, specs, scratch)
                pack_gray = 'dtcwt_pack_gray_u8' if is_u8_out else 'dtcwt_pack_gray_f32'
                self._run(self.k[pack_gray], (orig_w, orig_h),
                          rec, out_buf,
                          np.int32(orig_h), np.int32(orig_w), np.int32(s0.in_w))
                self.q.finish()
                _release_buffers(fused_lp, fused_hp)
                fused_lp = fused_hp = None

            elif ch_mode_3:
                fused_hps = [
                    [self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, b)) for b in hp_level_bytes]
                    for _ in range(3)
                ]
                fused_lps = [
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
                    for _ in range(3)
                ]
                for c in range(3):
                    self.cl.enqueue_fill_buffer(self.q, fused_lps[c], zero_f32, 0, lp_buf_bytes)
                    for lvl, b in enumerate(hp_level_bytes):
                        self.cl.enqueue_fill_buffer(self.q, fused_hps[c][lvl], zero_f32, 0, b)
                self.cl.enqueue_fill_buffer(self.q, w_sum_buf, zero_f32, 0, lp_buf_bytes)

                pass2_iter = _iter_prefetched_indices(
                    active_indices, prepare_upload, slots=p2_slots,
                    should_cancel=should_cancel,
                )
                slot_events = [None, None]
                slot_hosts = [None, None]
                next_progress = 0
                for step_pos, (index, (h_src, ch, kind)) in enumerate(pass2_iter):
                    slot = step_pos & 1 if ping_pong else 0
                    prev_slot_ev = slot_events[slot]
                    if prev_slot_ev is not None and (uploader is None or not uploader.separate_queue):
                        _wait_event(prev_slot_ev, cancel_fn)
                        slot_events[slot] = None
                        slot_hosts[slot] = None
                        prev_slot_ev = None
                    _check_cancel(should_cancel)
                    src_buf = src_bufs[slot]
                    if uploader is not None:
                        upload = uploader.upload(
                            slot, src_buf, h_src,
                            wait_for=_event_wait_list(prev_slot_ev), cancel=cancel_fn,
                        )
                        wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                    else:
                        upload = self.cl.enqueue_copy(
                            self.q, src_buf, async_upload_view(h_src), is_blocking=False,
                        )
                        wait_upload = None

                    need_lp_i = coarse_active[index]
                    if need_lp_i:
                        _compute_coarse_weight(index)
                        run_max_lvl = nlevels - 1
                    else:
                        run_max_lvl = max_needed[index]

                    last_extract_ev = None
                    for c in range(3):
                        src_ch = 0 if ch == 1 else (2 - c)
                        last_extract_ev = self._run(
                            self.k[f'dtcwt_extract_chan_{kind}'], (c0, r0),
                            src_buf, scratch['plane'],
                            np.int32(orig_h), np.int32(orig_w),
                            np.int32(r0), np.int32(c0),
                            np.int32(ch), np.int32(src_ch),
                            wait_for=wait_upload if c == 0 else None,
                        )
                        lp_cur, _ = self._forward_select(
                            scratch['plane'], specs, run_max_lvl, need_lp_i,
                            active_flags[index], decisions, fused_hps[c], index, scratch,
                        )
                        if need_lp_i and lp_cur is not None:
                            self._run(
                                self.k['dtcwt_accum_lp_1ch'], (lp_n,),
                                lp_cur, w_buf, fused_lps[c], np.int32(lp_n),
                            )

                    if ping_pong and last_extract_ev is not None and hasattr(last_extract_ev, 'wait'):
                        if hasattr(self.q, 'flush'):
                            self.q.flush()
                        slot_events[slot] = last_extract_ev
                        slot_hosts[slot] = (h_src, upload)
                    else:
                        self.q.finish()
                        del h_src, upload
                    if progress_callback is not None:
                        while next_progress <= index:
                            progress_callback(count + next_progress + 1, total_steps)
                            next_progress += 1

                pass2_iter.close()
                pass2_iter = None
                if ping_pong:
                    if uploader is not None:
                        uploader.finish(cancel_fn)
                    for slot in (0, 1):
                        if slot_events[slot] is not None:
                            _wait_event(slot_events[slot], cancel_fn)
                            slot_events[slot] = None
                        slot_hosts[slot] = None
                    self.q.finish()
                if progress_callback is not None:
                    while next_progress < count:
                        progress_callback(count + next_progress + 1, total_steps)
                        next_progress += 1

                _release_buffers(src_bufs, decisions, w_tmp_h, w_buf)
                src_bufs = ()
                decisions = []
                w_tmp_h = w_buf = None
                out_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, max(1, out_bytes))

                pack_name = 'dtcwt_pack_channel_u8' if is_u8_out else 'dtcwt_pack_channel_f32'
                for c in range(3):
                    _check_cancel(should_cancel)
                    self._run(self.k['dtcwt_div_lp_weight'], (lp_n,),
                              fused_lps[c], w_sum_buf, np.int32(lp_n))
                    rec = self._inverse_plane(fused_lps[c], fused_hps[c], specs, scratch)
                    self._run(self.k[pack_name], (orig_w, orig_h),
                              rec, out_buf,
                              np.int32(orig_h), np.int32(orig_w),
                              np.int32(s0.in_w), np.int32(2 - c))
                    self.q.finish()
                    _release_buffers(fused_lps[c], fused_hps[c])
                    fused_lps[c] = None
                    fused_hps[c] = None
            else:
                fused_hp = [
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, b))
                    for b in hp_level_bytes
                ]
                fused_lp = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
                out_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, max(1, out_bytes))
                pack_name = 'dtcwt_pack_channel_u8' if is_u8_out else 'dtcwt_pack_channel_f32'

                for c in range(3):
                    self.cl.enqueue_fill_buffer(self.q, fused_lp, zero_f32, 0, lp_buf_bytes)
                    for lvl, b in enumerate(hp_level_bytes):
                        self.cl.enqueue_fill_buffer(self.q, fused_hp[lvl], zero_f32, 0, b)
                    self.cl.enqueue_fill_buffer(self.q, w_sum_buf, zero_f32, 0, lp_buf_bytes)

                    pass2_iter = _iter_prefetched_indices(
                        active_indices, prepare_upload, slots=p2_slots,
                        should_cancel=should_cancel,
                    )
                    slot_events = [None, None]
                    slot_hosts = [None, None]
                    next_progress = 0
                    for step_pos, (index, (h_src, ch, kind)) in enumerate(pass2_iter):
                        slot = step_pos & 1 if ping_pong else 0
                        prev_slot_ev = slot_events[slot]
                        if prev_slot_ev is not None and (uploader is None or not uploader.separate_queue):
                            _wait_event(prev_slot_ev, cancel_fn)
                            slot_events[slot] = None
                            slot_hosts[slot] = None
                            prev_slot_ev = None
                        _check_cancel(should_cancel)
                        src_buf = src_bufs[slot]
                        if uploader is not None:
                            upload = uploader.upload(
                                slot, src_buf, h_src,
                                wait_for=_event_wait_list(prev_slot_ev), cancel=cancel_fn,
                            )
                            wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                        else:
                            upload = self.cl.enqueue_copy(
                                self.q, src_buf, async_upload_view(h_src), is_blocking=False,
                            )
                            wait_upload = None

                        need_lp_i = coarse_active[index]
                        if need_lp_i:
                            _compute_coarse_weight(index)
                            run_max_lvl = nlevels - 1
                        else:
                            run_max_lvl = max_needed[index]

                        src_ch = 0 if ch == 1 else (2 - c)
                        extract_ev = self._run(
                            self.k[f'dtcwt_extract_chan_{kind}'], (c0, r0),
                            src_buf, scratch['plane'],
                            np.int32(orig_h), np.int32(orig_w),
                            np.int32(r0), np.int32(c0),
                            np.int32(ch), np.int32(src_ch),
                            wait_for=wait_upload,
                        )
                        lp_cur, _ = self._forward_select(
                            scratch['plane'], specs, run_max_lvl, need_lp_i,
                            active_flags[index], decisions, fused_hp, index, scratch,
                        )
                        if need_lp_i and lp_cur is not None:
                            self._run(
                                self.k['dtcwt_accum_lp_1ch'], (lp_n,),
                                lp_cur, w_buf, fused_lp, np.int32(lp_n),
                            )
                        if ping_pong and extract_ev is not None and hasattr(extract_ev, 'wait'):
                            if hasattr(self.q, 'flush'):
                                self.q.flush()
                            slot_events[slot] = extract_ev
                            slot_hosts[slot] = (h_src, upload)
                        else:
                            self.q.finish()
                            del h_src, upload
                        if progress_callback is not None:
                            while next_progress <= index:
                                done = count + min(count, ((c * count + next_progress + 1) * count) // (3 * count))
                                progress_callback(done, total_steps)
                                next_progress += 1

                    pass2_iter.close()
                    pass2_iter = None
                    if ping_pong:
                        if uploader is not None:
                            uploader.finish(cancel_fn)
                        for slot in (0, 1):
                            if slot_events[slot] is not None:
                                _wait_event(slot_events[slot], cancel_fn)
                                slot_events[slot] = None
                            slot_hosts[slot] = None
                        self.q.finish()
                    if progress_callback is not None:
                        while next_progress < count:
                            done = count + min(count, ((c * count + next_progress + 1) * count) // (3 * count))
                            progress_callback(done, total_steps)
                            next_progress += 1

                    _check_cancel(should_cancel)
                    self._run(self.k['dtcwt_div_lp_weight'], (lp_n,),
                              fused_lp, w_sum_buf, np.int32(lp_n))
                    rec = self._inverse_plane(fused_lp, fused_hp, specs, scratch)
                    self._run(self.k[pack_name], (orig_w, orig_h),
                              rec, out_buf,
                              np.int32(orig_h), np.int32(orig_w),
                              np.int32(s0.in_w), np.int32(2 - c))
                    self.q.finish()

                _release_buffers(src_bufs, decisions, w_tmp_h, w_buf, fused_lp, fused_hp)
                src_bufs = ()
                decisions = []
                w_tmp_h = w_buf = fused_lp = fused_hp = None

            if uploader is not None:
                uploader.close()
                uploader = None
            _release_buffers(w_sum_buf, scratch)
            w_sum_buf = None
            scratch = {}

            _check_cancel(should_cancel)
            out = np.empty(shape, dtype=out_dtype)
            self.cl.enqueue_copy(self.q, out, out_buf, is_blocking=True)
            if progress_callback is not None:
                progress_callback(total_steps, total_steps)
            _check_cancel(should_cancel)
            return out
        finally:
            if uploader is not None:
                uploader.close()
            if pass1_iter is not None:
                pass1_iter.close()
            if pass2_iter is not None:
                pass2_iter.close()
            _release_buffers(
                src_bufs, scratch, best_act, best_idx, decisions,
                active_buf, w_tmp_h, w_buf, w_sum_buf, out_buf,
                fused_hps, fused_lps, fused_hp, fused_lp,
            )
