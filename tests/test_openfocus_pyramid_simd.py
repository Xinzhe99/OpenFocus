"""Bit-exact parity tests for vectorized OpenFocus C pyramid and shared B/C depth kernels."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import tifffile

from fusion_methods._depth_kernel import axis_block, _pack_rows_T, _unpack_rows_T
from fusion_methods import _openfocus_pyramid as pyr
from fusion_methods.openfocus_depth import _filter


_EVEN_REF = np.array([0.019027505, 0.318410337, 0.559359729, 0.103202470, 0], dtype=np.float32)
_ODD_REF = _EVEN_REF[::-1].copy()
_DOWN_REF = np.array([0.125, 0.375, 0.375, 0.125], dtype=np.float32)


def _ref_down(fine: np.ndarray) -> np.ndarray:
    h, w, ch = fine.shape
    ch_h, ch_w = (h + 1) // 2, (w + 1) // 2
    horiz = np.empty((h, ch_w, ch), dtype=np.float32)
    for y in range(h):
        for x in range(ch_w):
            for c in range(ch):
                v = np.float32(0)
                for k in range(4):
                    sx = min(w - 1, max(0, 2 * x + k - 1))
                    v = np.float32(v + np.float32(fine[y, sx, c] * _DOWN_REF[k]))
                horiz[y, x, c] = v
    out = np.empty((ch_h, ch_w, ch), dtype=np.float32)
    for y in range(ch_h):
        for x in range(ch_w):
            for c in range(ch):
                v = np.float32(0)
                for k in range(4):
                    sy = min(h - 1, max(0, 2 * y + k - 1))
                    v = np.float32(v + np.float32(horiz[sy, x, c] * _DOWN_REF[k]))
                out[y, x, c] = v
    return out


def _ref_synth(coarse: np.ndarray, fine: np.ndarray, with_weight: bool):
    h, w, ch = fine.shape
    ch_h, ch_w, _ = coarse.shape
    horiz = np.empty((ch_h, w, ch), dtype=np.float32)
    for y in range(ch_h):
        for x in range(w):
            weights = _ODD_REF if x % 2 else _EVEN_REF
            left = x // 2 - 2
            first, last = max(0, -left), min(5, ch_w - left)
            total = np.float32(0)
            for k in range(first, last):
                total = np.float32(total + weights[k])
            for c in range(ch):
                val = np.float32(0)
                for k in range(first, last):
                    val = np.float32(val + np.float32(coarse[y, left + k, c] * weights[k]))
                horiz[y, x, c] = np.float32(val / total)

    target = fine.copy()
    weight = np.empty((h, w), dtype=np.uint16) if with_weight else None
    for y in range(h):
        weights = _ODD_REF if y % 2 else _EVEN_REF
        top = y // 2 - 2
        first, last = max(0, -top), min(5, ch_h - top)
        total = np.float32(0)
        for k in range(first, last):
            total = np.float32(total + weights[k])
        for x in range(w):
            strength = np.float32(0)
            for c in range(ch):
                lo = np.float32(0)
                for k in range(first, last):
                    lo = np.float32(lo + np.float32(horiz[top + k, x, c] * weights[k]))
                lo = np.float32(lo / total)
                denom = np.float32(lo + np.float32(0.1))
                if not with_weight:
                    v = np.float32(np.float32(denom * target[y, x, c]) + lo)
                    target[y, x, c] = min(np.float32(1), max(np.float32(0), v))
                else:
                    v = np.float32(np.float32(target[y, x, c] - lo) / denom)
                    target[y, x, c] = v
                    if c < 3:
                        strength = np.float32(strength + abs(v))
            if with_weight:
                if ch == 1:
                    strength = np.float32(strength * np.float32(3))
                elif ch == 2:
                    strength = np.float32(strength * np.float32(1.5))
                if (height_b := (h > 2 and (y == 0 or y == h - 1))) or (w > 2 and (x == 0 or x == w - 1)):
                    weight[y, x] = 0
                else:
                    weight[y, x] = min(65535, int(np.float32(strength * np.float32(21845))))
    return target, weight


def _ref_axis_block(src, guide, first, output, acc, q, t, tm, tl, start, stop, left, right, backward):
    height = src.shape[0]
    mask = 0xFFFFFFFF
    for step in range(stop - start):
        y = stop - 1 - step if backward else start + step
        for x in range(left, right):
            col = x - left
            val = int(src[y, x])
            if guide is None:
                inn = (val * 16 * t) & mask
                st = ((int(acc[col]) + 2048 + inn) & mask) >> 12
                if backward:
                    output[y, x] = (((st + 3) >> 4) + 1 + int(first[y, col])) >> 1 & 65535
                else:
                    first[y, col] = ((st + 4) >> 4) & 65535
                acc[col] = (st * q + inn) & mask
            else:
                strength = int(guide[y, x])
                amount = (((65536 - strength) * tl + strength * tm) & mask) >> 16
                prev = (int(first[height - 1, col]) if y == height - 1 else int(src[y + 1, x])) if backward else int(src[max(0, y - 1), x])
                st = (((4096 - 2 * amount) * int(acc[col]) + 2048 + (val + prev) * (amount << 4)) & mask) >> 12
                acc[col] = st
                if backward:
                    output[y, x] = (((st + 3) >> 4) + 1 + int(first[y, col])) >> 1 & 65535
                else:
                    first[y, col] = (st + 4) >> 4


@pytest.mark.parametrize('shape', [
    (1, 1), (1, 5), (5, 1), (2, 3), (3, 7), (7, 9), (17, 31), (63, 127), (128, 193)
])
@pytest.mark.parametrize('channels', [1, 2, 3, 4])
def test_pyramid_down_and_synthesize_bit_exact(shape, channels):
    h, w = shape
    rng = np.random.default_rng(h * 1009 + w * 37 + channels)
    fine = rng.random((h, w, channels), dtype=np.float32)

    coarse = np.empty(((h + 1) // 2, (w + 1) // 2, channels), dtype=np.float32)
    pyr._down(fine, coarse, lambda: None)
    expected_coarse = _ref_down(fine)
    np.testing.assert_array_equal(coarse.view(np.uint32), expected_coarse.view(np.uint32))

    # Detail + weight pass
    detail_actual = fine.copy()
    w_actual = np.empty((h, w), dtype=np.uint16)
    pyr._synthesize(coarse, detail_actual, w_actual, lambda: None)
    expected_detail, expected_w = _ref_synth(coarse, fine, with_weight=True)
    np.testing.assert_array_equal(detail_actual.view(np.uint32), expected_detail.view(np.uint32))
    np.testing.assert_array_equal(w_actual, expected_w)

    # Reconstruct pass (weight=None)
    recon_actual = detail_actual.copy()
    pyr._synthesize(coarse, recon_actual, None, lambda: None)
    expected_recon, _ = _ref_synth(coarse, expected_detail, with_weight=False)
    np.testing.assert_array_equal(recon_actual.view(np.uint32), expected_recon.view(np.uint32))


def test_pyramid_normalize_merge_and_output_bit_exact():
    rng = np.random.default_rng(2026)
    src_base = rng.integers(0, 65536, (37, 53, 3), dtype=np.uint16)
    src_rev = src_base[:, :, ::-1]
    out_rev = np.empty_like(src_rev, dtype=np.float32)
    factor = np.float32(1.0 / 65535.0)
    pyr._normalize(src_rev, out_rev, factor, lambda: None)
    np.testing.assert_array_equal(
        out_rev.view(np.uint32),
        (src_rev.astype(np.float32) * factor).view(np.uint32),
    )

    acc_a = np.zeros((37, 53, 3), dtype=np.float32)
    best_a = np.zeros((37, 53), dtype=np.uint16)
    win_a = np.zeros((37, 53), dtype=np.uint16)
    d0 = rng.normal(0.0, 0.2, (37, 53, 3)).astype(np.float32)
    w0 = rng.integers(0, 65536, (37, 53), dtype=np.uint16)
    d1 = rng.normal(0.0, 0.2, (37, 53, 3)).astype(np.float32)
    w1 = rng.integers(0, 65536, (37, 53), dtype=np.uint16)
    pyr._merge_rows(d0, w0, acc_a, best_a, True, 0, 37, win_a, 0)
    pyr._merge_rows(d1, w1, acc_a, best_a, False, 0, 37, win_a, 1)
    mask = w1 > w0
    np.testing.assert_array_equal(best_a, np.maximum(w0, w1))
    np.testing.assert_array_equal(win_a, mask.astype(np.uint16))
    np.testing.assert_array_equal(acc_a, np.where(mask[:, :, None], d1, d0))

    # _output_rows vs _output_rows_flat parity
    out_f1 = np.empty((37, 53, 3), dtype=np.float32)
    out_f2 = np.empty((37, 53, 3), dtype=np.float32)
    in_f = rng.uniform(-0.1, 1.1, (37, 53, 3)).astype(np.float32)
    pyr._output_rows(in_f, out_f1, False, 0, 37)
    pyr._output_rows_flat(in_f.reshape(-1), out_f2.reshape(-1), False, 53 * 3, 0, 37)
    np.testing.assert_array_equal(out_f1.view(np.uint32), out_f2.view(np.uint32))


@pytest.mark.parametrize('guided', [False, True])
@pytest.mark.parametrize('backward', [False, True])
def test_axis_block_uint32_matches_64bit_reference_on_extremes(guided, backward):
    rng = np.random.default_rng(77 + 10 * int(guided) + int(backward))
    h, w = 41, 19
    src = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    src[0, :] = 65535
    src[-1, :] = 0
    src[5, :5] = [0, 1, 32768, 65534, 65535]
    guide = rng.integers(0, 65536, (h, w), dtype=np.uint16) if guided else None

    first_dtype = np.uint32 if guided else np.uint16
    first_a = rng.integers(0, 65536, (h, w), dtype=first_dtype)
    first_r = first_a.copy()
    out_a = np.zeros_like(src)
    out_r = np.zeros_like(src)
    acc_a = rng.integers(0, 2**32, (w,), dtype=np.uint32)
    acc_r = acc_a.copy()

    q, t, tm, tl = 3812, 142, 1890, 412
    axis_block(src, guide, first_a, out_a, acc_a, q, t, tm, tl, 0, h, 2, w - 3, backward)
    _ref_axis_block(src, guide, first_r, out_r, acc_r, q, t, tm, tl, 0, h, 2, w - 3, backward)

    np.testing.assert_array_equal(first_a, first_r)
    np.testing.assert_array_equal(out_a, out_r)
    np.testing.assert_array_equal(acc_a, acc_r)


def test_pack_unpack_transpose_parity():
    rng = np.random.default_rng(999)
    orig = rng.integers(0, 65536, (100, 137), dtype=np.uint16)
    buf = np.empty((137, 33), dtype=np.uint16)
    _pack_rows_T(orig, buf, 19, 52)
    np.testing.assert_array_equal(buf, orig[19:52, :].T)
    out = np.zeros_like(orig)
    _unpack_rows_T(buf, out, 19, 52)
    np.testing.assert_array_equal(out[19:52, :], orig[19:52, :])


def test_openfocus_abc_method_switching_determinism(tmp_path):
    from core.image_loader import ImageStackLoader
    from core.multi_focus_fusion import MultiFocusFusion

    rng = np.random.default_rng(515)
    paths = []
    for i in range(4):
        arr = rng.integers(1000, 60000, (96, 128, 3), dtype=np.uint16)
        p = tmp_path / f"img_{i}.tif"
        tifffile.imwrite(str(p), arr, photometric="rgb")
        paths.append(str(p))

    ok, msg, frames, _ = ImageStackLoader().load_from_filepaths(paths)
    assert ok, msg

    def run(alg):
        out = MultiFocusFusion(algorithm=alg, use_gpu=False, tile_enabled=False, exposure_match=False).fuse(frames, thread_count=2)
        return hashlib.sha256(np.ascontiguousarray(out).view(np.uint8)).hexdigest()

    h_c1 = run('openfocus_c')
    h_b1 = run('openfocus_b')
    h_a1 = run('openfocus_a')
    assert h_b1 == run('openfocus_b')
    assert h_c1 == run('openfocus_c')
    assert h_a1 == run('openfocus_a')


def _ref_filter_axis(src, guide, main, low, count, axis):
    from fusion_methods.openfocus_depth import _weight

    s = np.asarray(src)
    g = None if guide is None else np.asarray(guide)
    if axis:
        s = s.T
        if g is not None:
            g = g.T
    h, w = s.shape
    cnt = max(1, min(int(count), h))
    out = np.empty((h, w), dtype=np.uint16)
    first = np.empty((h, w), dtype=np.uint16 if g is None else np.uint32)
    q = t = tm = tl = 0
    if g is None:
        q = _weight(main)
        t = (4096 - q) >> 1
    else:
        tm = (4096 - _weight(main)) >> 1
        tl = (4096 - _weight(low)) >> 1
    init = (s[:cnt, :].sum(axis=0, dtype=np.uint64) << 4) // cnt
    if g is None:
        init *= t + q
    acc = (init & 0xFFFFFFFF).astype(np.uint32)
    _ref_axis_block(s, g, first, out, acc, q, t, tm, tl, 0, h, 0, w, False)
    acc = first[-1, :].astype(np.uint32) << 4
    if g is None:
        acc *= t + q
    _ref_axis_block(s, g, first, out, acc, q, t, tm, tl, 0, h, 0, w, True)
    return out.T if axis else out


@pytest.mark.parametrize('shape', [(97, 256), (73, 131)])
@pytest.mark.parametrize('guided', [False, True])
def test_depth_filter_bit_exact_on_power_of_two_odd_and_strided_views(shape, guided):
    from fusion_methods.openfocus_depth import _AXIS_BLOCK_ROWS, _filter

    rng = np.random.default_rng(910 + shape[0] * 17 + int(guided))
    h, w = shape
    src_base = rng.integers(0, 65536, (h, w * 2), dtype=np.uint16)
    guide_base = rng.integers(0, 65536, (h, w * 2), dtype=np.uint16) if guided else None

    for src, guide in (
        (np.ascontiguousarray(src_base[:, :w]), None if guide_base is None else np.ascontiguousarray(guide_base[:, :w])),
        (src_base[:, ::-2], None if guide_base is None else guide_base[:, ::-2]),
    ):
        actual = _filter(src, 7.5, lambda: None, guide=guide, low=3.25 if guided else None, allocate=np.empty)
        ref = src.copy()
        if guided:
            ref = _ref_filter_axis(ref, guide, 7.5, 3.25, 7.5, 1)
            ref = _ref_filter_axis(ref, guide, 7.5, 3.25, 7.5, 0)
        else:
            ref = _ref_filter_axis(ref, None, 7.5, 7.5, 7.5, 1)
            ref = _ref_filter_axis(ref, None, 7.5, 7.5, 7.5, 0)
        np.testing.assert_array_equal(actual, ref)

    assert _AXIS_BLOCK_ROWS == 2048


@pytest.mark.parametrize('guided', [False, True])
def test_axis_block_c_and_f_layout_parity(guided):
    rng = np.random.default_rng(420 + int(guided))
    h, w = 53, 37
    src_c = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    src_f = np.asfortranarray(src_c)
    guide_c = rng.integers(0, 65536, (h, w), dtype=np.uint16) if guided else None
    guide_f = np.asfortranarray(guide_c) if guided else None
    fdtype = np.uint32 if guided else np.uint16

    for bwd in (False, True):
        fc = np.zeros((h, w), dtype=fdtype)
        ff = np.zeros((h, w), dtype=fdtype)
        oc = np.zeros((h, w), dtype=np.uint16)
        of = np.zeros((h, w), dtype=np.uint16)
        ac = (src_c[0, :] << 4).astype(np.uint32)
        af = ac.copy()
        axis_block(src_c, guide_c, fc, oc, ac, 3812, 142, 1890, 412, 0, h, 3, w - 4, bwd)
        axis_block(src_f, guide_f, ff, of, af, 3812, 142, 1890, 412, 0, h, 3, w - 4, bwd)
        np.testing.assert_array_equal(fc, ff)
        np.testing.assert_array_equal(oc, of)
        np.testing.assert_array_equal(ac, af)
