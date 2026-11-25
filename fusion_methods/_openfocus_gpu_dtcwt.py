"""OpenCL GPU fusion for DTCWT (Dual-Tree Complex Wavelet Transform) fusion.

Ports luminance/channel extraction, odd-size even-extension, near_sym_a level-0
and qshift_a deeper-level complex wavelet forward/inverse transforms, 3x3
reflect max-activity streaming argmax, O(1) 3x3 nearest-mode consistency
filtering, O(1)-VRAM bilinear+Gaussian low-pass weighting, and output packing
to OpenCL 1.2 FP32 kernels with single-frame streaming and automatic CPU
fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

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

    int ym3 = reflect_idx(y - 3, r);
    int ym2 = reflect_idx(y - 2, r);
    int ym1 = reflect_idx(y - 1, r);
    int yp1 = reflect_idx(y + 1, r);
    int yp2 = reflect_idx(y + 2, r);
    int yp3 = reflect_idx(y + 3, r);

    float s0 = src[(size_t)ym3 * (size_t)c + (size_t)x];
    float s1 = src[(size_t)ym2 * (size_t)c + (size_t)x];
    float s2 = src[(size_t)ym1 * (size_t)c + (size_t)x];
    float s3 = src[(size_t)y   * (size_t)c + (size_t)x];
    float s4 = src[(size_t)yp1 * (size_t)c + (size_t)x];
    float s5 = src[(size_t)yp2 * (size_t)c + (size_t)x];
    float s6 = src[(size_t)yp3 * (size_t)c + (size_t)x];

    size_t pos = (size_t)y * (size_t)c + (size_t)x;
    // NumPy's column convolution adds the taps in this order.
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

__kernel void dtcwt_colfilter_h7(
    __global const float* src,
    __global float* dst,
    const int r,
    const int c,
    const int use_h1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c || y >= r) return;
    __constant const float* h = use_h1 ? c_h1o : c_g0o;
    size_t row = (size_t)y * (size_t)c;
    if (x >= 3 && x < c - 3) {
        dst[row + (size_t)x] = (
            src[row + (size_t)(x + 3)] * h[0]
            + src[row + (size_t)(x + 2)] * h[1]
            + src[row + (size_t)(x + 1)] * h[2]
            + src[row + (size_t)x]       * h[3]
            + src[row + (size_t)(x - 1)] * h[4]
            + src[row + (size_t)(x - 2)] * h[5]
            + src[row + (size_t)(x - 3)] * h[6]);
    } else {
        dst[row + (size_t)x] = (
            src[row + (size_t)reflect_idx(x + 3, c)] * h[0]
            + src[row + (size_t)reflect_idx(x + 2, c)] * h[1]
            + src[row + (size_t)reflect_idx(x + 1, c)] * h[2]
            + src[row + (size_t)x]                     * h[3]
            + src[row + (size_t)reflect_idx(x - 1, c)] * h[4]
            + src[row + (size_t)reflect_idx(x - 2, c)] * h[5]
            + src[row + (size_t)reflect_idx(x - 3, c)] * h[6]);
    }
}

__kernel void dtcwt_colfilter_v_add0(
    __global const float* src7,
    __global const float* src5,
    __global float* dst,
    const int r,
    const int c)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= c || y >= r) return;

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
        src5[(size_t)ym2 * (size_t)c + (size_t)x] * c_g1o[0]
        + src5[(size_t)ym1 * (size_t)c + (size_t)x] * c_g1o[1]
        + src5[(size_t)y   * (size_t)c + (size_t)x] * c_g1o[2]
        + src5[(size_t)yp1 * (size_t)c + (size_t)x] * c_g1o[3]
        + src5[(size_t)yp2 * (size_t)c + (size_t)x] * c_g1o[4]);

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

__kernel void dtcwt_pad4(
    __global const float* src,
    __global float* dst,
    const int r,
    const int c,
    const int rr,
    const int cc,
    const int pr,
    const int pc)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= cc || y >= rr) return;
    int sy = clamp(y - pr, 0, r - 1);
    int sx = clamp(x - pc, 0, c - 1);
    dst[(size_t)y * (size_t)cc + (size_t)x] = src[(size_t)sy * (size_t)c + (size_t)sx];
}

__kernel void dtcwt_coldfilt_v_pair(
    __global const float* src,
    __global float* lo,
    __global float* hi,
    const int r,
    const int c)
{
    int x = get_global_id(0);
    int j = get_global_id(1);
    int groups = r >> 2;
    if (x >= c || j >= groups) return;

    int base = 4 * j;
    float s_e[10];
    s_e[0] = src[(size_t)reflect_idx(base + 10, r) * (size_t)c + (size_t)x];
    s_e[1] = src[(size_t)reflect_idx(base +  8, r) * (size_t)c + (size_t)x];
    s_e[2] = src[(size_t)reflect_idx(base +  6, r) * (size_t)c + (size_t)x];
    s_e[3] = src[(size_t)reflect_idx(base +  4, r) * (size_t)c + (size_t)x];
    s_e[4] = src[(size_t)reflect_idx(base +  2, r) * (size_t)c + (size_t)x];
    s_e[5] = src[(size_t)reflect_idx(base     , r) * (size_t)c + (size_t)x];
    s_e[6] = src[(size_t)reflect_idx(base -  2, r) * (size_t)c + (size_t)x];
    s_e[7] = src[(size_t)reflect_idx(base -  4, r) * (size_t)c + (size_t)x];
    s_e[8] = src[(size_t)reflect_idx(base -  6, r) * (size_t)c + (size_t)x];
    s_e[9] = src[(size_t)reflect_idx(base -  8, r) * (size_t)c + (size_t)x];

    float s_o[10];
    s_o[0] = src[(size_t)reflect_idx(base + 11, r) * (size_t)c + (size_t)x];
    s_o[1] = src[(size_t)reflect_idx(base +  9, r) * (size_t)c + (size_t)x];
    s_o[2] = src[(size_t)reflect_idx(base +  7, r) * (size_t)c + (size_t)x];
    s_o[3] = src[(size_t)reflect_idx(base +  5, r) * (size_t)c + (size_t)x];
    s_o[4] = src[(size_t)reflect_idx(base +  3, r) * (size_t)c + (size_t)x];
    s_o[5] = src[(size_t)reflect_idx(base +  1, r) * (size_t)c + (size_t)x];
    s_o[6] = src[(size_t)reflect_idx(base -  1, r) * (size_t)c + (size_t)x];
    s_o[7] = src[(size_t)reflect_idx(base -  3, r) * (size_t)c + (size_t)x];
    s_o[8] = src[(size_t)reflect_idx(base -  5, r) * (size_t)c + (size_t)x];
    s_o[9] = src[(size_t)reflect_idx(base -  7, r) * (size_t)c + (size_t)x];

    // Upstream sums the even and odd taps separately before adding them.
    float4 even = (float4)(0.0f), odd = (float4)(0.0f);
    for (int t = 0; t < 10; t += 2) {
        even += (float4)(s_e[t] * c_h0b[t], s_o[t] * c_h0a[t],
                         s_o[t] * c_h1a[t], s_e[t] * c_h1b[t]);
        int u = t + 1;
        odd += (float4)(s_e[u] * c_h0b[u], s_o[u] * c_h0a[u],
                        s_o[u] * c_h1a[u], s_e[u] * c_h1b[u]);
    }
    float4 v = even + odd;

    size_t row0 = (size_t)(2 * j) * (size_t)c + (size_t)x;
    size_t row1 = (size_t)(2 * j + 1) * (size_t)c + (size_t)x;
    lo[row0] = v.x;
    lo[row1] = v.y;
    hi[row0] = v.z;
    hi[row1] = v.w;
}

__kernel void dtcwt_coldfilt_h(
    __global const float* src,
    __global float* dst,
    const int r,
    const int c,
    const int pos_sign)
{
    int j = get_global_id(0);
    int y = get_global_id(1);
    int groups = c >> 2;
    int dst_c = c >> 1;
    if (j >= groups || y >= r) return;

    __constant const float* ha = pos_sign ? c_h0b : c_h1b;
    __constant const float* hb = pos_sign ? c_h0a : c_h1a;
    size_t src_row = (size_t)y * (size_t)c;
    size_t dst_row = (size_t)y * (size_t)dst_c;
    int base = 4 * j;

    float2 even = (float2)(0.0f), odd = (float2)(0.0f);
    for (int t = 0; t < 10; t += 2) {
        int i = base + 10 - 2 * t;
        even += (float2)(src[src_row + (size_t)reflect_idx(i, c)] * ha[t],
                         src[src_row + (size_t)reflect_idx(i + 1, c)] * hb[t]);
        odd += (float2)(src[src_row + (size_t)reflect_idx(i - 2, c)] * ha[t + 1],
                        src[src_row + (size_t)reflect_idx(i - 1, c)] * hb[t + 1]);
    }
    float2 v = even + odd;

    int c1 = pos_sign ? (2 * j) : (2 * j + 1);
    int c2 = pos_sign ? (2 * j + 1) : (2 * j);
    dst[dst_row + (size_t)c1] = v.x;
    dst[dst_row + (size_t)c2] = v.y;
}

__kernel void dtcwt_q2c_pair(
    __global const float* quad,
    __global float2* hp,
    const int h2,
    const int w2,
    const int c0,
    const int c1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w2 || y >= h2) return;
    int qw = w2 << 1;
    size_t row0 = (size_t)(2 * y) * (size_t)qw + (size_t)(2 * x);
    size_t row1 = row0 + (size_t)qw;
    // Scale each component before combining it, as in dtcwt's q2c.
    float a = quad[row0] * INV_SQRT2;
    float b = quad[row0 + 1] * INV_SQRT2;
    float c = quad[row1] * INV_SQRT2;
    float d = quad[row1 + 1] * INV_SQRT2;
    float2 z0 = (float2)(a - d, b + c);
    float2 z1 = (float2)(a + d, b - c);
    size_t plane = (size_t)h2 * (size_t)w2;
    size_t pos = (size_t)y * (size_t)w2 + (size_t)x;
    hp[(size_t)c0 * plane + pos] = z0;
    hp[(size_t)c1 * plane + pos] = z1;
}

__kernel void dtcwt_c2q_pair(
    __global const float2* hp,
    __global float* quad,
    const int h2,
    const int w2,
    const int c0,
    const int c1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w2 || y >= h2) return;
    size_t plane = (size_t)h2 * (size_t)w2;
    size_t pos = (size_t)y * (size_t)w2 + (size_t)x;
    float2 z0 = hp[(size_t)c0 * plane + pos];
    float2 z1 = hp[(size_t)c1 * plane + pos];
    int qw = w2 << 1;
    size_t row0 = (size_t)(2 * y) * (size_t)qw + (size_t)(2 * x);
    size_t row1 = row0 + (size_t)qw;
    quad[row0]     = (z0.x + z1.x) * SQRT_HALF;
    quad[row0 + 1] = (z0.y + z1.y) * SQRT_HALF;
    quad[row1]     = (z0.y - z1.y) * SQRT_HALF;
    quad[row1 + 1] = (z1.x - z0.x) * SQRT_HALF;
}

__kernel void dtcwt_colifilt_v_add(
    __global const float* src0,
    __global const float* src1,
    __global float* dst,
    const int r,
    const int c)
{
    int x = get_global_id(0);
    int j = get_global_id(1);
    int groups = r >> 1;
    if (x >= c || j >= groups) return;

    int im1 = reflect_idx(2 * j - 1, r);
    int ip1 = reflect_idx(2 * j + 1, r);
    int ip3 = reflect_idx(2 * j + 3, r);
    int ip5 = reflect_idx(2 * j + 5, r);
    int im3 = reflect_idx(2 * j - 3, r);

    int im4 = reflect_idx(2 * j - 4, r);
    int im2 = reflect_idx(2 * j - 2, r);
    int i0  = reflect_idx(2 * j, r);
    int ip2 = reflect_idx(2 * j + 2, r);
    int ip4 = reflect_idx(2 * j + 4, r);

    // Branch 0 (pos=1): ha = g0b (= h0a), hb = g0a (= h0b); off_a = 0, off_b = -1
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

    // Branch 1 (pos=0): ha = g1b (= h1a), hb = g1a (= h1b); off_a = -1, off_b = 0
    float a1_4 = src1[(size_t)im3 * (size_t)c + (size_t)x];
    float a1_3 = src1[(size_t)im1 * (size_t)c + (size_t)x];
    float a1_2 = src1[(size_t)ip1 * (size_t)c + (size_t)x];
    float a1_1 = src1[(size_t)ip3 * (size_t)c + (size_t)x];
    float a1_0 = src1[(size_t)ip5 * (size_t)c + (size_t)x];

    float b1_4 = src1[(size_t)im4 * (size_t)c + (size_t)x];
    float b1_3 = src1[(size_t)im2 * (size_t)c + (size_t)x];
    float b1_2 = src1[(size_t)i0  * (size_t)c + (size_t)x];
    float b1_1 = src1[(size_t)ip2 * (size_t)c + (size_t)x];
    float b1_0 = src1[(size_t)ip4 * (size_t)c + (size_t)x];

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

__kernel void dtcwt_mag_sum6(
    __global const float2* hp,
    __global float* mag,
    const int h,
    const int w)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;
    size_t plane = (size_t)h * (size_t)w;
    size_t pos = (size_t)y * (size_t)w + (size_t)x;
    float s = 0.0f;
    for (int o = 0; o < 6; ++o) {
        float2 z = hp[(size_t)o * plane + pos];
        s += hypot(z.x, z.y);
    }
    mag[pos] = s;
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

    // Reflect border matching scipy.ndimage.maximum_filter(..., size=3, mode='reflect')
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
    // Mark each run once to avoid contending on an atomic for every pixel.
    if (gid > 0 && decision[gid - 1] == f) return;
    atomic_or(&active_flags[(size_t)f * (size_t)n_levels + (size_t)lvl], 1u);
}

__kernel void dtcwt_select_hp(
    __global const float2* hp_src,
    __global const int* decision,
    __global float2* hp_dst,
    const int n_pixels,
    const int frame_idx)
{
    int pos = get_global_id(0);
    if (pos >= n_pixels) return;
    if (decision[pos] != frame_idx) return;
    for (int o = 0; o < 6; ++o) {
        size_t off = (size_t)o * (size_t)n_pixels + (size_t)pos;
        hp_dst[off] = hp_src[off];
    }
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
    'dtcwt_colfilter_h7',
    'dtcwt_colfilter_v_add0',
    'dtcwt_colfilter_h_add0',
    'dtcwt_pad4',
    'dtcwt_coldfilt_v_pair',
    'dtcwt_coldfilt_h',
    'dtcwt_q2c_pair',
    'dtcwt_c2q_pair',
    'dtcwt_colifilt_v_add',
    'dtcwt_colifilt_h_add_trim',
    'dtcwt_mag_sum6',
    'dtcwt_max3x3_argmax',
    'dtcwt_consistent_argmax_3x3',
    'dtcwt_mark_active',
    'dtcwt_select_hp',
    'dtcwt_lp_mask_resize',
    'dtcwt_gauss13_h',
    'dtcwt_gauss13_v_to_w',
    'dtcwt_accum_lp_1ch',
    'dtcwt_div_lp_weight',
    'dtcwt_pack_channel_u8',
    'dtcwt_pack_channel_f32',
)

_cache_lock = Lock()
_PROGRAM_CACHE = None


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _build_cached_program, _get_shared_context

    key = getattr(device, 'int_ptr', None)
    with _cache_lock:
        ctx = _get_shared_context(cl, device)
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key and _PROGRAM_CACHE[1] is ctx:
            return _PROGRAM_CACHE[1:]
        queue = cl.CommandQueue(ctx, device=device)
        # OpenCL 1.2 makes correctly rounded FP32 divide/sqrt optional.
        rounded = cl.device_fp_config.CORRECTLY_ROUNDED_DIVIDE_SQRT
        options = ['-cl-fp32-correctly-rounded-divide-sqrt'] if device.single_fp_config & rounded else []
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

    def _run(self, kernel, gsize, *args):
        g = gsize if isinstance(gsize, tuple) else (gsize,)
        kernel(self.q, g, None, *args)

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
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _drain_queue(self):
        # Keep the device lease until pending kernels finish. A cleanup
        # error must not replace the original cancellation or device error.
        if self.q is not None:
            try:
                self.q.finish()
            except self.cl.Error:
                _invalidate_program_cache(getattr(self, 'ctx', None))

    def _forward_plane(self, plane_buf, specs: list[_LevelSpec], max_lvl: int,
                       hp_bufs: list, scratch: dict):
        """Run 0..max_lvl DTCWT forward levels on GPU, leaving final lolo in scratch['lolo_cur']."""
        s0 = specs[0]
        r0, c0 = s0.in_h, s0.in_w
        lo = scratch['lo']
        hi = scratch['hi']
        lolo_a = scratch['lolo_a']
        qtmp0 = scratch['qtmp0']

        self._run(self.k['dtcwt_colfilter_v_pair'], (c0, r0),
                  plane_buf, lo, hi, np.int32(r0), np.int32(c0))
        self._run(self.k['dtcwt_colfilter_h5'], (c0, r0),
                  lo, lolo_a, np.int32(r0), np.int32(c0), np.int32(1))

        if hp_bufs[0] is not None:
            self._run(self.k['dtcwt_colfilter_h5'], (c0, r0),
                      hi, qtmp0, np.int32(r0), np.int32(c0), np.int32(1))
            self._run(self.k['dtcwt_q2c_pair'], (s0.hp_w, s0.hp_h),
                      qtmp0, hp_bufs[0], np.int32(s0.hp_h), np.int32(s0.hp_w),
                      np.int32(0), np.int32(5))

            self._run(self.k['dtcwt_colfilter_h7'], (c0, r0),
                      lo, qtmp0, np.int32(r0), np.int32(c0), np.int32(1))
            self._run(self.k['dtcwt_q2c_pair'], (s0.hp_w, s0.hp_h),
                      qtmp0, hp_bufs[0], np.int32(s0.hp_h), np.int32(s0.hp_w),
                      np.int32(2), np.int32(3))

            self._run(self.k['dtcwt_colfilter_h7'], (c0, r0),
                      hi, qtmp0, np.int32(r0), np.int32(c0), np.int32(1))
            self._run(self.k['dtcwt_q2c_pair'], (s0.hp_w, s0.hp_h),
                      qtmp0, hp_bufs[0], np.int32(s0.hp_h), np.int32(s0.hp_w),
                      np.int32(1), np.int32(4))

        cur_lolo = lolo_a
        alt_lolo = scratch['lolo_b']

        for lvl in range(1, max_lvl + 1):
            sk = specs[lvl]
            if sk.pr or sk.pc:
                pad_in = scratch['pad']
                self._run(self.k['dtcwt_pad4'], (sk.pad_w, sk.pad_h),
                          cur_lolo, pad_in,
                          np.int32(sk.in_h), np.int32(sk.in_w),
                          np.int32(sk.pad_h), np.int32(sk.pad_w),
                          np.int32(sk.pr), np.int32(sk.pc))
            else:
                pad_in = cur_lolo

            lo2 = scratch['lo']
            hi2 = scratch['hi']
            groups_r = sk.pad_h >> 2
            groups_c = sk.pad_w >> 2
            r2, c2 = sk.lolo_h, sk.lolo_w

            self._run(self.k['dtcwt_coldfilt_v_pair'], (sk.pad_w, groups_r),
                      pad_in, lo2, hi2, np.int32(sk.pad_h), np.int32(sk.pad_w))

            self._run(self.k['dtcwt_coldfilt_h'], (groups_c, r2),
                      lo2, alt_lolo, np.int32(r2), np.int32(sk.pad_w), np.int32(1))

            if hp_bufs[lvl] is not None:
                self._run(self.k['dtcwt_coldfilt_h'], (groups_c, r2),
                          hi2, qtmp0, np.int32(r2), np.int32(sk.pad_w), np.int32(1))
                self._run(self.k['dtcwt_q2c_pair'], (sk.hp_w, sk.hp_h),
                          qtmp0, hp_bufs[lvl], np.int32(sk.hp_h), np.int32(sk.hp_w),
                          np.int32(0), np.int32(5))

                self._run(self.k['dtcwt_coldfilt_h'], (groups_c, r2),
                          lo2, qtmp0, np.int32(r2), np.int32(sk.pad_w), np.int32(0))
                self._run(self.k['dtcwt_q2c_pair'], (sk.hp_w, sk.hp_h),
                          qtmp0, hp_bufs[lvl], np.int32(sk.hp_h), np.int32(sk.hp_w),
                          np.int32(2), np.int32(3))

                self._run(self.k['dtcwt_coldfilt_h'], (groups_c, r2),
                          hi2, qtmp0, np.int32(r2), np.int32(sk.pad_w), np.int32(0))
                self._run(self.k['dtcwt_q2c_pair'], (sk.hp_w, sk.hp_h),
                          qtmp0, hp_bufs[lvl], np.int32(sk.hp_h), np.int32(sk.hp_w),
                          np.int32(1), np.int32(4))

            cur_lolo, alt_lolo = alt_lolo, cur_lolo

        return cur_lolo

    def _inverse_plane(self, fused_lp, fused_hp: list, specs: list[_LevelSpec],
                       scratch: dict):
        """Reconstruct fused image plane from GPU low-pass & high-pass pyramids."""
        nlevels = len(specs)
        z = fused_lp
        z_out = scratch['lolo_a']
        q_lh = scratch['q_lh']
        q_hl = scratch['q_hl']
        q_hh = scratch['q_hh']
        y1 = scratch['lo']
        y2 = scratch['hi']

        for lvl in range(nlevels, 1, -1):
            sk = specs[lvl - 1]
            hp = fused_hp[lvl - 1]
            rq, cq = sk.hp_h << 1, sk.hp_w << 1
            r2, c2 = rq << 1, cq << 1

            self._run(self.k['dtcwt_c2q_pair'], (sk.hp_w, sk.hp_h),
                      hp, q_lh, np.int32(sk.hp_h), np.int32(sk.hp_w),
                      np.int32(0), np.int32(5))
            self._run(self.k['dtcwt_c2q_pair'], (sk.hp_w, sk.hp_h),
                      hp, q_hl, np.int32(sk.hp_h), np.int32(sk.hp_w),
                      np.int32(2), np.int32(3))
            self._run(self.k['dtcwt_c2q_pair'], (sk.hp_w, sk.hp_h),
                      hp, q_hh, np.int32(sk.hp_h), np.int32(sk.hp_w),
                      np.int32(1), np.int32(4))

            self._run(self.k['dtcwt_colifilt_v_add'], (cq, rq >> 1),
                      z, q_lh, y1, np.int32(rq), np.int32(cq))
            self._run(self.k['dtcwt_colifilt_v_add'], (cq, rq >> 1),
                      q_hl, q_hh, y2, np.int32(rq), np.int32(cq))

            off_y = (r2 - sk.exp_r) >> 1
            off_x = (c2 - sk.exp_c) >> 1
            dst = scratch['lolo_b'] if z is z_out else z_out
            self._run(self.k['dtcwt_colifilt_h_add_trim'], (cq >> 1, r2),
                      y1, y2, dst, np.int32(r2), np.int32(cq),
                      np.int32(sk.exp_r), np.int32(sk.exp_c),
                      np.int32(off_y), np.int32(off_x))
            z = dst

        # Final level 0 synthesis using near_sym_a
        s0 = specs[0]
        hp0 = fused_hp[0]
        self._run(self.k['dtcwt_c2q_pair'], (s0.hp_w, s0.hp_h),
                  hp0, q_lh, np.int32(s0.hp_h), np.int32(s0.hp_w),
                  np.int32(0), np.int32(5))
        self._run(self.k['dtcwt_c2q_pair'], (s0.hp_w, s0.hp_h),
                  hp0, q_hl, np.int32(s0.hp_h), np.int32(s0.hp_w),
                  np.int32(2), np.int32(3))
        self._run(self.k['dtcwt_c2q_pair'], (s0.hp_w, s0.hp_h),
                  hp0, q_hh, np.int32(s0.hp_h), np.int32(s0.hp_w),
                  np.int32(1), np.int32(4))

        rq, cq = s0.in_h, s0.in_w
        self._run(self.k['dtcwt_colfilter_v_add0'], (cq, rq),
                  z, q_lh, y1, np.int32(rq), np.int32(cq))
        self._run(self.k['dtcwt_colfilter_v_add0'], (cq, rq),
                  q_hl, q_hh, y2, np.int32(rq), np.int32(cq))

        rec = scratch['pad']
        self._run(self.k['dtcwt_colfilter_h_add0'], (cq, rq),
                  y1, y2, rec, np.int32(rq), np.int32(cq))
        return rec

    def _fuse_locked(self, images, nlevels, should_cancel, progress_callback):
        from dataclasses import replace
        from core import resource_policy
        from core.source_frames import resolve_frame, without_source_cache_growth
        from fusion_methods._openfocus_gpu import GpuError, _free_vram

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
        for frame in images:
            _check_cancel(should_cancel)
            if frame.shape[:2] != shape[:2] or frame.dtype not in (np.uint8, np.uint16, np.float32):
                raise GpuError('GPU DTCWT requires matching dimensions and uint8, uint16, or float32 samples')
            source_peak = max(source_peak, _source_staging_bytes(frame))
            src_bytes = max(src_bytes, frame.nbytes)
        active_bytes = count * nlevels * np.dtype(np.uint32).itemsize
        staging = max(source_peak, out_bytes * 2) + active_bytes + 16 * 1024 * 1024
        if staging > host_budget:
            raise GpuError(
                f'DTCWT GPU staging exceeds the host memory budget '
                f'({staging >> 20} MiB needed, {host_budget >> 20} MiB available)'
            )

        # Calculate exact device buffer footprint
        hp_level_bytes = [sp.hp_h * sp.hp_w * 6 * 8 for sp in specs]  # float2 * 6
        pyr_bytes = sum(hp_level_bytes)
        dec_bytes = sum(sp.hp_h * sp.hp_w * 4 for sp in specs)  # best_act / decision
        quad_bytes = r0 * c0 * 4
        pad_bytes = max(sp.pad_h * sp.pad_w * 4 for sp in specs)
        lo_bytes = max(quad_bytes, pad_bytes // 2)

        # Shared fixed scratch across passes
        scratch_bytes = (
            src_bytes
            + out_bytes
            + 2 * dec_bytes          # best_act + best_idx -> decision
            + pyr_bytes              # tmp hp_bufs[0..N-1]
            + 6 * quad_bytes         # plane, lolo_a/b, qtmp0/q_lh, q_hl/q_hh
            + 2 * lo_bytes + pad_bytes
            + 3 * lp_n * 4           # lp_tmp_h, lp_weight, w_sum
            + active_bytes + 16 * 1024 * 1024
        )
        per_ch_accum_bytes = pyr_bytes + lp_n * 4
        need_3ch = scratch_bytes + 3 * per_ch_accum_bytes
        need_1ch = scratch_bytes + per_ch_accum_bytes

        limit = int(self.device.global_mem_size) // 2
        if self.device.host_unified_memory:
            limit = min(limit, host_budget - staging)
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError('DTCWT GPU free memory is unknown')
            snapshot = replace(snapshot, free_vram=memory[0], total_vram=memory[1])
            limit = min(limit, resource_policy.vram_budget(snapshot))

        single = int(self.device.max_mem_alloc_size)
        largest = max(src_bytes, out_bytes, max(hp_level_bytes), pad_bytes, active_bytes)
        if largest > single:
            raise GpuError('DTCWT GPU buffer exceeds max_mem_alloc_size')

        if need_3ch <= limit:
            ch_mode_3 = True
        elif need_1ch <= limit:
            ch_mode_3 = False
        else:
            raise GpuError('DTCWT GPU workspace exceeds the device budget')

        total_steps = 2 * count + 1  # final inverse transform and readback
        if progress_callback is not None:
            progress_callback(0, total_steps)
        _check_cancel(should_cancel)
        self.ctx, self.q, self.k = _get_or_build_program(self.cl, self.device)

        def prepare_upload(index):
            _check_cancel(should_cancel)
            with without_source_cache_growth():
                raw = resolve_frame(images[index], should_cancel=should_cancel)
            _check_cancel(should_cancel)
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

        src_buf = self.cl.Buffer(self.ctx, self.mf.READ_ONLY, max(1, src_bytes))
        plane_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, r0 * c0 * 4))
        scratch = {
            'lo': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lo_bytes)),
            'hi': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lo_bytes)),
            'lolo_a': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, quad_bytes)),
            'lolo_b': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, quad_bytes)),
            'qtmp0': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, quad_bytes)),
            'pad': self.cl.Buffer(self.ctx, self.mf.READ_WRITE, pad_bytes),
        }
        tmp_hp = [
            self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, b))
            for b in hp_level_bytes
        ]
        best_act = [
            self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, sp.hp_h * sp.hp_w * 4))
            for sp in specs
        ]
        best_idx = [
            self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, sp.hp_h * sp.hp_w * 4))
            for sp in specs
        ]

        # Pass 1: streaming luminance activity + strict '>' argmax
        for index in range(count):
            _check_cancel(should_cancel)
            h_src, ch, kind = prepare_upload(index)
            self.cl.enqueue_copy(self.q, src_buf, h_src, is_blocking=True)
            del h_src
            self._run(
                self.k[f'dtcwt_extract_luma_{kind}'], (c0, r0),
                src_buf, plane_buf,
                np.int32(orig_h), np.int32(orig_w),
                np.int32(r0), np.int32(c0), np.int32(ch),
            )
            self._forward_plane(plane_buf, specs, nlevels - 1, tmp_hp, scratch)
            first_f = 1 if index == 0 else 0
            for lvl, sp in enumerate(specs):
                self._run(
                    self.k['dtcwt_mag_sum6'], (sp.hp_w, sp.hp_h),
                    tmp_hp[lvl], scratch['qtmp0'],
                    np.int32(sp.hp_h), np.int32(sp.hp_w),
                )
                self._run(
                    self.k['dtcwt_max3x3_argmax'], (sp.hp_w, sp.hp_h),
                    scratch['qtmp0'], best_act[lvl], best_idx[lvl],
                    np.int32(sp.hp_h), np.int32(sp.hp_w),
                    np.int32(index), np.int32(first_f),
                )
            self.q.finish()
            if progress_callback is not None:
                progress_callback(index + 1, total_steps)

        # Consistency 3x3 nearest-mode filter on decision maps (reusing best_act mem)
        decisions = best_act  # rename buffer handles: overwrite float best_act with int decision
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
        del best_idx

        active_flags = np.empty((count, nlevels), dtype=np.uint32)
        self.cl.enqueue_copy(self.q, active_flags, active_buf, is_blocking=True)
        del active_buf
        _check_cancel(should_cancel)

        # For each frame: max_needed_lvl (-1 if never used) and coarse activity
        coarse_active = [bool(active_flags[i, nlevels - 1]) for i in range(count)]
        max_needed = []
        for i in range(count):
            highest = -1
            for lvl in range(nlevels):
                if active_flags[i, lvl]:
                    highest = lvl
            max_needed.append(highest)

        coarse_spec = specs[-1]
        lp_buf_bytes = lp_n * 4
        w_tmp_h = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
        w_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
        w_sum_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
        out_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, max(1, out_bytes))
        zero_hp = np.float32(0.0)
        zero_lp = np.float32(0.0)

        def _compute_coarse_weight(index):
            self._run(
                self.k['dtcwt_lp_mask_resize'], (lp_w, lp_h),
                decisions[-1], scratch['qtmp0'],
                np.int32(coarse_spec.hp_h), np.int32(coarse_spec.hp_w),
                np.int32(lp_h), np.int32(lp_w),
                np.int32(index),
            )
            self._run(
                self.k['dtcwt_gauss13_h'], (lp_w, lp_h),
                scratch['qtmp0'], w_tmp_h, np.int32(lp_h), np.int32(lp_w),
            )
            self._run(
                self.k['dtcwt_gauss13_v_to_w'], (lp_w, lp_h),
                w_tmp_h, w_buf, w_sum_buf, np.int32(lp_h), np.int32(lp_w),
            )

        # Prepare inverse scratch buffers cleanly (aliased aliases match _inverse_plane)
        scratch['q_lh'] = scratch['qtmp0']
        scratch['q_hl'] = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, quad_bytes))
        scratch['q_hh'] = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, quad_bytes))

        if ch_mode_3:
            fused_hps = [
                [self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, b)) for b in hp_level_bytes]
                for _ in range(3)
            ]
            fused_lps = [
                self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
                for _ in range(3)
            ]
            for c in range(3):
                self.cl.enqueue_fill_buffer(self.q, fused_lps[c], zero_lp, 0, lp_buf_bytes)
                for lvl, b in enumerate(hp_level_bytes):
                    self.cl.enqueue_fill_buffer(self.q, fused_hps[c][lvl], zero_hp, 0, b)
            self.cl.enqueue_fill_buffer(self.q, w_sum_buf, zero_lp, 0, lp_buf_bytes)

            for index in range(count):
                _check_cancel(should_cancel)
                m_lvl = max_needed[index]
                if m_lvl < 0 and not coarse_active[index]:
                    if progress_callback is not None:
                        progress_callback(count + index + 1, total_steps)
                    continue

                if coarse_active[index]:
                    _compute_coarse_weight(index)
                    run_max_lvl = nlevels - 1
                else:
                    run_max_lvl = m_lvl

                h_src, ch, kind = prepare_upload(index)
                self.cl.enqueue_copy(self.q, src_buf, h_src, is_blocking=True)
                del h_src

                # Only extract hp levels where this frame has winning coefficients
                hp_targets = [
                    tmp_hp[lvl] if active_flags[index, lvl] else None
                    for lvl in range(nlevels)
                ]

                for c in range(3):
                    src_ch = 0 if ch == 1 else (2 - c)
                    self._run(
                        self.k[f'dtcwt_extract_chan_{kind}'], (c0, r0),
                        src_buf, plane_buf,
                        np.int32(orig_h), np.int32(orig_w),
                        np.int32(r0), np.int32(c0),
                        np.int32(ch), np.int32(src_ch),
                    )
                    lp_cur = self._forward_plane(plane_buf, specs, run_max_lvl, hp_targets, scratch)
                    for lvl in range(run_max_lvl + 1):
                        if hp_targets[lvl] is None:
                            continue
                        sp = specs[lvl]
                        self._run(
                            self.k['dtcwt_select_hp'], (sp.hp_h * sp.hp_w,),
                            hp_targets[lvl], decisions[lvl], fused_hps[c][lvl],
                            np.int32(sp.hp_h * sp.hp_w), np.int32(index),
                        )
                    if coarse_active[index]:
                        self._run(
                            self.k['dtcwt_accum_lp_1ch'], (lp_n,),
                            lp_cur, w_buf, fused_lps[c], np.int32(lp_n),
                        )
                self.q.finish()
                if progress_callback is not None:
                    progress_callback(count + index + 1, total_steps)

            del src_buf, plane_buf, tmp_hp, decisions, w_tmp_h, w_buf
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
                fused_lps[c] = None
                fused_hps[c] = None
        else:
            fused_hp = [
                self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, b))
                for b in hp_level_bytes
            ]
            fused_lp = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, lp_buf_bytes))
            pack_name = 'dtcwt_pack_channel_u8' if is_u8_out else 'dtcwt_pack_channel_f32'

            for c in range(3):
                self.cl.enqueue_fill_buffer(self.q, fused_lp, zero_lp, 0, lp_buf_bytes)
                for lvl, b in enumerate(hp_level_bytes):
                    self.cl.enqueue_fill_buffer(self.q, fused_hp[lvl], zero_hp, 0, b)
                self.cl.enqueue_fill_buffer(self.q, w_sum_buf, zero_lp, 0, lp_buf_bytes)

                for index in range(count):
                    _check_cancel(should_cancel)
                    m_lvl = max_needed[index]
                    if m_lvl < 0 and not coarse_active[index]:
                        if progress_callback is not None:
                            done = count + min(count, ((c * count + index + 1) * count) // (3 * count))
                            progress_callback(done, total_steps)
                        continue
                    if coarse_active[index]:
                        _compute_coarse_weight(index)
                        run_max_lvl = nlevels - 1
                    else:
                        run_max_lvl = m_lvl

                    h_src, ch, kind = prepare_upload(index)
                    self.cl.enqueue_copy(self.q, src_buf, h_src, is_blocking=True)
                    del h_src

                    hp_targets = [
                        tmp_hp[lvl] if active_flags[index, lvl] else None
                        for lvl in range(nlevels)
                    ]
                    src_ch = 0 if ch == 1 else (2 - c)
                    self._run(
                        self.k[f'dtcwt_extract_chan_{kind}'], (c0, r0),
                        src_buf, plane_buf,
                        np.int32(orig_h), np.int32(orig_w),
                        np.int32(r0), np.int32(c0),
                        np.int32(ch), np.int32(src_ch),
                    )
                    lp_cur = self._forward_plane(plane_buf, specs, run_max_lvl, hp_targets, scratch)
                    for lvl in range(run_max_lvl + 1):
                        if hp_targets[lvl] is None:
                            continue
                        sp = specs[lvl]
                        self._run(
                            self.k['dtcwt_select_hp'], (sp.hp_h * sp.hp_w,),
                            hp_targets[lvl], decisions[lvl], fused_hp[lvl],
                            np.int32(sp.hp_h * sp.hp_w), np.int32(index),
                        )
                    if coarse_active[index]:
                        self._run(
                            self.k['dtcwt_accum_lp_1ch'], (lp_n,),
                            lp_cur, w_buf, fused_lp, np.int32(lp_n),
                        )
                    self.q.finish()
                    if progress_callback is not None:
                        done = count + min(count, ((c * count + index + 1) * count) // (3 * count))
                        progress_callback(done, total_steps)

                _check_cancel(should_cancel)
                self._run(self.k['dtcwt_div_lp_weight'], (lp_n,),
                          fused_lp, w_sum_buf, np.int32(lp_n))
                rec = self._inverse_plane(fused_lp, fused_hp, specs, scratch)
                self._run(self.k[pack_name], (orig_w, orig_h),
                          rec, out_buf,
                          np.int32(orig_h), np.int32(orig_w),
                          np.int32(s0.in_w), np.int32(2 - c))
                self.q.finish()
            del src_buf, plane_buf, tmp_hp, decisions, w_tmp_h, w_buf

        del w_sum_buf, scratch

        _check_cancel(should_cancel)
        out = np.empty(shape, dtype=out_dtype)
        self.cl.enqueue_copy(self.q, out, out_buf, is_blocking=True)
        if progress_callback is not None:
            progress_callback(total_steps, total_steps)
        _check_cancel(should_cancel)
        return out
