"""Bit-exact parity tests for OpenFocus A/B optimized integer kernels and output conversion."""
import numpy as np
import pytest

from fusion_methods import _openfocus_depth_rows as depth_rows
from fusion_methods import _openfocus_kernels as kernels
from fusion_methods import _openfocus_pixels as pixels
from fusion_methods import openfocus_a, openfocus_bc
from fusion_methods.openfocus_depth import _snap_lut


def _ref_gray_rows(image, scale, depth_color=False):
    h, w = image.shape[:2]
    out = np.empty((h, w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            if image.ndim == 2:
                val = int(image[y, x])
            elif image.shape[2] >= 3:
                tot = int(image[y, x, 0]) + int(image[y, x, 1]) + int(image[y, x, 2])
                if depth_color:
                    out[y, x] = (tot * scale * 85) >> 8
                    continue
                val = tot // 3
            elif image.shape[2] == 1:
                val = int(image[y, x, 0])
            else:
                val = (int(image[y, x, 0]) + int(image[y, x, 1])) // 3
            out[y, x] = val * scale
    return out


def _ref_down2_rows(image, scale=1):
    h, w = image.shape[:2]
    oh, ow = (h + 1) // 2, (w + 1) // 2
    out_shape = (oh, ow) if image.ndim == 2 else (oh, ow, image.shape[2])
    out = np.empty(out_shape, dtype=np.int64)
    for y in range(oh):
        top = 2 * y
        bot = min(top + 1, h - 1)
        for x in range(ow):
            left = 2 * x
            right = min(left + 1, w - 1)
            if image.ndim == 2:
                tot = int(image[top, left]) + int(image[top, right]) + int(image[bot, left]) + int(image[bot, right])
                out[y, x] = (tot * scale) >> 2
            else:
                for ch in range(image.shape[2]):
                    tot = (int(image[top, left, ch]) + int(image[top, right, ch])
                           + int(image[bot, left, ch]) + int(image[bot, right, ch]))
                    out[y, x, ch] = (tot * scale) >> 2
    return out


def _ref_merge_rows(acc_init, image, depth, index, step, scale, first, reverse):
    acc = acc_init.copy()
    h, w = depth.shape
    for y in range(h):
        for x in range(w):
            weight = 1024
            if not first:
                delta = int(depth[y, x]) - step * index
                if reverse:
                    delta = -delta
                if delta >= 0:
                    if delta > step:
                        continue
                else:
                    if delta <= -step:
                        continue
                    weight = 1024 - ((-delta << 10) // step)
                    if weight <= 0:
                        continue
            if image.ndim == 2:
                val = int(image[y, x]) * scale
                acc[y, x] = val if weight == 1024 else (int(acc[y, x]) * (1024 - weight) + val * weight) >> 10
            else:
                for ch in range(image.shape[2]):
                    val = int(image[y, x, ch]) * scale
                    acc[y, x, ch] = (
                        val if weight == 1024
                        else (int(acc[y, x, ch]) * (1024 - weight) + val * weight) >> 10
                    )
    return acc


@pytest.mark.parametrize('dtype,scale', [(np.uint8, 256), (np.uint16, 1), (np.uint16, 7), (np.int32, 1)])
@pytest.mark.parametrize('shape', [(15, 17), (15, 17, 1), (15, 17, 2), (15, 17, 3), (15, 17, 4)])
@pytest.mark.parametrize('depth_color', [False, True])
@pytest.mark.parametrize('reverse_channels', [False, True])
def test_gray_rows_matches_reference(dtype, scale, shape, depth_color, reverse_channels):
    if reverse_channels and len(shape) != 3:
        pytest.skip('channel reversal applies to 3D images')
    rng = np.random.default_rng(101 + len(shape) * 10 + shape[-1])
    hi = 256 if dtype == np.uint8 else 65536
    img = rng.integers(0, hi, size=shape, dtype=dtype)
    img.flat[::7] = hi - 1
    img.flat[::11] = 0
    view = img[..., ::-1] if reverse_channels else img
    out_dtype = np.int32 if dtype == np.int32 or scale == 7 else np.uint16
    out = np.empty(shape[:2], dtype=out_dtype)
    kernels.gray_rows(view, out, 0, shape[0], scale, depth_color)
    expected = _ref_gray_rows(view, scale, depth_color).astype(out_dtype)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize('shape', [(13, 19), (13, 19, 3), (13, 19, 4)])
def test_table_rows_matches_reference(shape):
    rng = np.random.default_rng(202 + len(shape))
    img = rng.integers(0, 256, size=shape, dtype=np.uint8)
    cols = 1 if len(shape) == 2 else shape[2]
    table = rng.integers(0, 65536, size=(256, cols), dtype=np.uint16)
    view = img[..., ::-1] if len(shape) == 3 else img
    out = np.empty(shape, dtype=np.uint16)
    kernels.table_rows(view, table, out, 0, shape[0])
    if len(shape) == 2:
        expected = table[view, 0]
    else:
        expected = np.stack([table[view[..., c], c] for c in range(cols)], axis=-1)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize('shape', [(1, 1), (1, 17), (23, 1), (31, 33), (32, 34), (17, 19, 3)])
@pytest.mark.parametrize('dtype,scale', [(np.uint8, 1), (np.uint8, 256), (np.uint16, 1), (np.int32, 1)])
def test_down2_rows_and_burt_laplacian_rows_match_reference(shape, dtype, scale):
    rng = np.random.default_rng(303 + shape[0] * 13 + shape[1])
    hi = 256 if dtype == np.uint8 else 65536
    img = rng.integers(0, hi, size=shape, dtype=dtype)
    img.flat[::5] = hi - 1
    out_shape = ((shape[0] + 1) // 2, (shape[1] + 1) // 2) + shape[2:]
    out = np.empty(out_shape, dtype=np.int32 if dtype == np.int32 or scale != 1 else np.uint16)
    kernels.down2_rows(img, out, 0, out_shape[0], scale)
    np.testing.assert_array_equal(out, _ref_down2_rows(img, scale).astype(out.dtype))

    if len(shape) == 2 and dtype == np.uint16:
        red = _ref_down2_rows(img, 1).astype(np.int32)
        lap_new = np.empty(shape, dtype=np.int32)
        lap_ref = np.empty(shape, dtype=np.int32)
        kernels.burt_laplacian_rows(img, red, lap_new, 0, shape[0])
        for y in range(shape[0]):
            kernels._burt_laplacian_row(img, red, lap_ref[y], y)
        np.testing.assert_array_equal(lap_new, lap_ref)


@pytest.mark.parametrize('shape', [(1, 1), (1, 17), (23, 1), (32, 34), (41, 39), (37, 65), (130, 129), (16, 257)])
@pytest.mark.parametrize('radius', [1, 2, 4, 8, 16, 32, 64])
def test_burt_fused_accumulation_matches_exact_reference(shape, radius):
    rng = np.random.default_rng(404 + shape[0] * 31 + shape[1] + radius)
    gray = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    gray.flat[::9] = 65535
    gray.flat[::13] = 0
    img = np.stack([gray, gray, gray], axis=-1)
    half_gray, laplacians, levels = openfocus_a._hard_pyramid(gray, radius, None, lambda *_: None)

    resp = np.empty(shape[1], dtype=np.int32)
    best_new = np.empty(shape, dtype=np.int32)
    out_new = np.empty(img.shape, dtype=np.uint16)
    bg_new = np.zeros(shape + (5,), dtype=np.float32)

    # Compare response accumulation via select_rows first=True across banded calls
    for start in range(0, shape[0], 11):
        stop = min(shape[0], start + 11)
        state = kernels.select_rows(
            img, gray, half_gray, laplacians, levels,
            resp, best_new, out_new, start, stop, 1, True, bg_new[start:stop],
        )
        assert state in (kernels.A_SATURATED, kernels.A_SHARP, kernels.A_BACKGROUND)

    expected = np.zeros(shape, dtype=np.int64)
    r_row = np.empty(shape[1], dtype=np.int32)
    for y in range(shape[0]):
        kernels._burt_laplacian_row(gray, half_gray, r_row, y)
        h, w = half_gray.shape
        off = 0
        for lv in range(1, levels):
            kernels._burt_accumulate_row(laplacians, off, h, w, r_row, lv, y)
            off += h * w
            h, w = (h + 1) // 2, (w + 1) // 2
        expected[y] = np.clip(r_row, 0, kernels.A_MAX_RESPONSE)
    np.testing.assert_array_equal(best_new, expected)
    np.testing.assert_array_equal(out_new, img)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [0, 3, 4])
@pytest.mark.parametrize('reverse_view', [False, True])
def test_select_and_blend_rows_parity_across_states_and_layouts(dtype, channels, reverse_view):
    if reverse_view and channels == 0:
        pytest.skip('channel flip only applies to 3D')
    rng = np.random.default_rng(505 + channels + (1 if dtype == np.uint16 else 0))
    shape = (140, 145) if channels == 0 else (140, 145, channels)
    hi = 256 if dtype == np.uint8 else 65536
    scale = 256 if dtype == np.uint8 else 1

    # Frame 0: flat left half (low score / zero on border), noisy right half (sharp), top edge saturated
    frames = []
    for k in range(4):
        f = np.full(shape, (k * 37 + 10) % hi, dtype=dtype)
        # Mild gradient on left half so response < A_BLEND_LIMIT (65536) and nonzero
        f[:, 70:, ...] = rng.integers(0, hi, size=f[:, 70:, ...].shape, dtype=dtype)
        # Checkerboard high response on rows 128..140 in frame 1 to force A_SHARP transition
        if k == 1:
            cb = np.where((np.indices((12, 145)).sum(axis=0)) % 2, hi - 1, 0).astype(dtype)
            if channels == 0:
                f[128:140, :] = cb
            else:
                f[128:140, :, :3] = cb[:, :, None]
        if k == 2:
            # Flat band in frame 2 so zero-score minimum logic is exercised
            f[10:25, :60, ...] = 100
        frames.append(f[..., ::-1] if reverse_view else f)

    def run_all(force_int32_gray):
        out = np.empty(shape, dtype=np.uint16)
        best = np.empty((140, 145), dtype=np.int32)
        ch = 1 if channels == 0 else channels
        bg = np.zeros((140, 145, ch + 2), dtype=np.float32)
        resp = np.empty(145, dtype=np.int32)
        band = 128
        states = [kernels.A_BACKGROUND, kernels.A_BACKGROUND]
        for idx, f in enumerate(frames):
            gray = pixels.frame_gray_codes(f, None, None)
            if force_int32_gray:
                gray = gray.astype(np.int32)
            half_gray, laps, levels = openfocus_a._hard_pyramid(gray, 8, None, lambda *_: None)
            for s in (0, 128):
                e = min(140, s + band)
                slot = s // band
                if states[slot] == kernels.A_SATURATED:
                    continue
                if states[slot] == kernels.A_BACKGROUND:
                    states[slot] = kernels.select_rows(
                        f, gray, half_gray, laps, levels, resp, best, out,
                        s, e, scale, idx == 0, bg[s:e],
                    )
                else:
                    states[slot] = kernels.select_rows(
                        f, gray, half_gray, laps, levels, resp, best, out,
                        s, e, scale, idx == 0, bg[:0],
                    )
        for s in (0, 128):
            e = min(140, s + band)
            if states[s // band] == kernels.A_BACKGROUND:
                kernels.blend_low_response_rows(out, best, bg[s:e], s, e)
        return out, best

    out_fast, best_fast = run_all(force_int32_gray=False)
    out_ref, best_ref = run_all(force_int32_gray=True)
    np.testing.assert_array_equal(best_fast, best_ref)
    np.testing.assert_array_equal(out_fast, out_ref)


@pytest.mark.parametrize('shape', [(17, 29), (17, 29, 3), (17, 29, 4)])
@pytest.mark.parametrize('dtype,scale', [(np.uint8, 256), (np.uint16, 1)])
@pytest.mark.parametrize('step', [2, 3, 661, 65534, 65535])
@pytest.mark.parametrize('reverse', [False, True])
def test_merge_rows_and_active_source_rows_match_reference(shape, dtype, scale, step, reverse):
    rng = np.random.default_rng(606 + len(shape) + step)
    hi = 256 if dtype == np.uint8 else 65536
    img = rng.integers(0, hi, size=shape, dtype=dtype)
    view = img[..., ::-1] if len(shape) == 3 else img
    depth = rng.integers(0, 65536, size=shape[:2], dtype=np.uint16)
    acc0 = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    acc = acc0.copy()
    index = min(1, 65535 // step)
    kernels.merge_rows(acc, view, depth, index, step, 0, shape[0], scale, False, reverse)
    expected = _ref_merge_rows(acc0, view, depth, index, step, scale, False, reverse)
    np.testing.assert_array_equal(acc, expected)

    n_src = max(2, 65535 // step + 1)
    active = np.zeros(n_src, dtype=bool)
    kernels.active_source_rows(depth, step, active, 0, shape[0])
    ref_active = np.zeros(n_src, dtype=bool)
    for code in depth.ravel():
        lower = min(max(int(code) // step, 0), n_src - 1)
        ref_active[lower] = True
        if int(code) % step != 0:
            ref_active[min(lower + 1, n_src - 1)] = True
    np.testing.assert_array_equal(active, ref_active)


@pytest.mark.parametrize('shape', [(1, 1), (1, 4), (4, 1), (5, 5), (19, 23), (64, 129)])
def test_depth_rows_kernels_match_reference_and_extremes(shape):
    rng = np.random.default_rng(707 + shape[0] * 17 + shape[1])
    gray = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    gray.flat[::5] = 65535
    gray.flat[::9] = 0
    fine = rng.integers(0, 65536, size=shape, dtype=np.uint16)

    act_u16 = np.empty(shape, dtype=np.uint16)
    act_i32 = np.empty(shape, dtype=np.uint16)
    depth_rows.activity_rows(gray, fine, act_u16, 0, shape[0])
    depth_rows.activity_rows(gray.astype(np.int32), fine, act_i32, 0, shape[0])
    np.testing.assert_array_equal(act_u16, act_i32)

    dh, dw = max(1, shape[0] // 2), max(1, shape[1] // 2)
    det_u16 = np.empty((dh, dw), dtype=np.uint16)
    det_i32 = np.empty((dh, dw), dtype=np.uint16)
    depth_rows.detail_rows(gray, det_u16, 0)
    depth_rows.detail_rows(gray.astype(np.int32), det_i32, 0)
    np.testing.assert_array_equal(det_u16, det_i32)

    sm_u16 = np.empty((dh, dw), dtype=np.uint16)
    sm_i32 = np.empty((dh, dw), dtype=np.uint16)
    depth_rows.smooth_detail_rows(det_u16, sm_u16, 0, dh)
    depth_rows.smooth_detail_rows(det_u16.astype(np.int32), sm_i32, 0, dh)
    np.testing.assert_array_equal(sm_u16, sm_i32)


@pytest.mark.parametrize('step', [2, 3, 661, 32767, 65534, 65535])
@pytest.mark.parametrize('feather', [0, 3, 10])
def test_snap_rows_matches_exact_int64_modulo(step, feather):
    lut = _snap_lut(feather)
    depth_vals = np.arange(0, 65536, dtype=np.uint16).reshape(256, 256)
    fast = depth_vals.copy()
    ref = depth_vals.astype(np.int32)
    depth_rows.snap_rows(fast, lut, step, 0, 256)
    depth_rows.snap_rows(ref, lut, step, 0, 256)
    np.testing.assert_array_equal(fast, ref.astype(np.uint16))


@pytest.mark.parametrize('e37', [False, True])
def test_output_codes_matches_legacy_u16_and_u8_conversions(e37):
    vals = np.arange(0, 65536, dtype=np.uint16).reshape(256, 256)
    rgb = np.stack([vals, 65535 - vals, vals], axis=-1)
    rgba = np.stack([vals, vals >> 1, 65535 - vals, vals], axis=-1)

    f_u16 = [np.empty((1, 1, 3), dtype=np.uint16)]
    out_f32 = pixels.output_codes(rgb, f_u16, None, e37=e37)
    ref_f32 = pixels.output_codes(rgb.astype(np.int32), f_u16, None, e37=e37)
    np.testing.assert_array_equal(out_f32, ref_f32)

    out_bc = openfocus_bc._output_codes(rgb, f_u16, None, source_scale=1)
    ref_bc = (rgb.astype(np.float32) * np.float32(255.0 / 65535.0))
    np.testing.assert_array_equal(out_bc, ref_bc)

    f_u8 = [np.empty((1, 1, 4), dtype=np.uint8)]
    out_u8 = pixels.output_codes(rgba, f_u8, None, e37=e37)
    ref_u8 = pixels.output_codes(rgba.astype(np.int32), f_u8, None, e37=e37)
    np.testing.assert_array_equal(out_u8, ref_u8)
