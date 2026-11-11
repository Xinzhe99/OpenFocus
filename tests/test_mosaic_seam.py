"""Consumer-visible global seams and phase-stable, true Laplacian blending."""
import numpy as np
import pytest

from core.mosaic_seam import LowSample, SeamSpec, build_labels, label_weight
from core.mosaic_blend import blend_tiles, expanded_tile


def test_content_seam_follows_a_known_winding_agreement_corridor():
    height, width = 96, 120
    expected = np.rint(48 + 17 * np.sin(np.arange(height) * np.pi / 40)).astype(int)
    first = np.zeros((height, width, 3), np.float32)
    second = np.ones_like(first)
    for y, target in enumerate(expected):
        second[y, target - 24 - 2:target - 24 + 3] = 0
    alpha = np.ones((height, width), np.float32)
    samples = [LowSample(0, first, alpha, (0, 0)),
               LowSample(1, second, alpha, (24, 0))]
    labels = build_labels(samples, (height, 144), SeamSpec(precision=1, cost='color'))
    path = np.array([24 + np.flatnonzero(row[24:120] == 2)[0] for row in labels])
    assert np.ptp(path) > 25
    assert np.max(np.abs(path - expected)) <= 5
    assert np.all(labels[:, :24] == 1)
    assert np.all(labels[:, 120:] == 2)


def test_mask_holes_and_exclusive_third_source_survive_global_seams():
    colors = np.full((40, 80, 3), 0.4, np.float32)
    alpha = np.ones((40, 80), np.float32)
    hole = alpha.copy()
    hole[10:20, 20:30] = 0
    samples = [LowSample(0, colors, alpha, (0, 0)),
               LowSample(1, colors, hole, (40, 0)),
               LowSample(2, colors, alpha, (100, 0))]
    labels = build_labels(samples, (40, 180), SeamSpec(precision=1))
    assert not np.any(labels[10:20, 60:70] == 2)
    assert np.all(labels[:, 120:] == 3)
    assert np.all(labels[:, :40] == 1)


def _render(tile_size, band=12, identical=False):
    height, width = 81, 177
    # Deliberately negative, non-grid-aligned canvas origin.
    origin = (-13, 7)
    labels = np.ones((21, 45), np.uint16)
    labels[:, 22:] = 2
    output = np.empty((height, width, 3), np.uint16)
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w, h = min(tile_size, width - x), min(tile_size, height - y)
            (x0, y0, x1, y1), crop = expanded_tile(
                x, y, w, h, width, height, levels=4, band=band)
            shape = (y1 - y0, x1 - x0)

            def samples():
                for index, value in enumerate((12001, 12001 if identical else 43003)):
                    colors = np.full((*shape, 3), value / 65535, np.float32)
                    alpha = np.ones(shape, np.float32)
                    weight = label_weight(labels, index, origin=origin, scale=4,
                        x=origin[0] + x0, y=origin[1] + y0,
                        width=shape[1], height=shape[0], band=band)
                    yield colors, alpha, weight
            colors, _ = blend_tiles(samples(), shape=shape, levels=4)
            output[y:y + h, x:x + w] = np.rint(np.clip(colors[crop], 0, 1) * 65535)
    return output


def test_arbitrary_tile_boundaries_preserve_global_phase_and_native_codes():
    expected = _render(177)
    for tile in (32, 37, 64, 128):
        np.testing.assert_array_equal(_render(tile), expected)
    assert expected[40, 0, 0] == 12001
    assert expected[40, -1, 0] == 43003


def test_band_changes_real_transition_but_not_identical_overlap():
    narrow, wide = _render(32, 4), _render(32, 24)
    assert np.any(narrow != wide)
    identical = _render(32, 4, True)
    np.testing.assert_array_equal(identical, _render(64, 24, True))
    np.testing.assert_array_equal(identical, np.full(identical.shape, 12001, np.uint16))


def test_single_source_keeps_unassociated_colors_and_actual_opacity():
    colors = np.full((29, 41, 3), (1001, 2003, 3007), np.float32) / 65535
    alpha = np.full((29, 41), 200 / 255, np.float32)
    weight = np.full(alpha.shape, 0.15, np.float32)
    actual, opacity = blend_tiles([(colors, alpha, weight)], shape=alpha.shape)
    np.testing.assert_array_equal(np.rint(actual * 65535),
                                  np.broadcast_to([1001, 2003, 3007], colors.shape))
    np.testing.assert_array_equal(opacity, alpha)


def test_laplacian_blend_smooths_low_frequency_step_without_double_edges():
    shape = (64, 128)
    a = np.full((*shape, 3), 0.2, np.float32)
    b = np.full_like(a, 0.8)
    wa = np.zeros(shape, np.float32)
    wa[:, :64] = 1
    alpha = np.ones(shape, np.float32)
    items = [(a, alpha, wa), (b, alpha, 1 - wa)]
    direct, _ = blend_tiles(iter(items), shape=shape, mode='feather')
    blended, _ = blend_tiles(iter(items), shape=shape, levels=5)
    assert direct[32, 64, 0] - direct[32, 63, 0] > 0.5
    assert np.max(np.abs(np.diff(blended[32, :, 0]))) < 0.1
    assert blended[32, 10, 0] < 0.25 and blended[32, -10, 0] > 0.75


def test_empty_coverage_is_transparent_and_cancel_stops_before_sampling():
    shape = (32, 32)
    colors = np.ones((*shape, 3), np.float32)
    zero, one = np.zeros(shape, np.float32), np.ones(shape, np.float32)
    image, alpha = blend_tiles([(colors, zero, one)], shape=shape)
    np.testing.assert_array_equal(image, np.zeros_like(colors))
    np.testing.assert_array_equal(alpha, zero)
    with pytest.raises(InterruptedError):
        build_labels([], shape, should_cancel=lambda: True)
    with pytest.raises(InterruptedError):
        blend_tiles([], shape=shape, should_cancel=lambda: True)


def test_masked_seam_owner_preserves_the_remaining_source_pixels():
    shape = (64, 80)
    first = np.full((*shape, 3), 0.8, np.float32)
    second = np.full((*shape, 3), 0.2, np.float32)
    owner_alpha = np.ones(shape, np.float32)
    owner_alpha[16:48, 20:60] = 0
    alpha = np.ones(shape, np.float32)
    output, coverage = blend_tiles(
        [(first, owner_alpha, alpha), (second, alpha, np.zeros(shape, np.float32))],
        shape=shape, levels=4)
    np.testing.assert_array_equal(output[16:48, 20:60], second[16:48, 20:60])
    np.testing.assert_array_equal(coverage, alpha)


def test_label_clear_uses_bounded_blocks_and_stops_on_cancel(monkeypatch):
    from contextlib import contextmanager
    from core import mosaic_seam as seam

    monkeypatch.setattr(seam, '_LABEL_BLOCK_PIXELS', 11)
    original = seam.frame_region
    writes = []

    @contextmanager
    def tracking(array, key=Ellipsis, *, write=False):
        with original(array, key, write=write) as region:
            if write:
                writes.append(region.size)
            yield region

    monkeypatch.setattr(seam, 'frame_region', tracking)
    labels = np.full((5, 17), 9, np.uint16)
    seam._clear_labels(labels, None)
    assert np.all(labels == 0)
    assert sum(writes) == labels.size
    assert max(writes) <= 11
    labels.fill(9)
    writes.clear()
    checks = iter((False, True))
    with pytest.raises(InterruptedError):
        seam._clear_labels(labels, lambda: next(checks))
    assert len(writes) == 1
    assert np.count_nonzero(labels == 0) == writes[0]


def test_graphcut_clears_labels_again_before_writing(monkeypatch):
    from core import mosaic_seam as seam
    calls = []
    original = seam._clear_labels

    def clear(labels, cancel):
        calls.append(labels.copy())
        original(labels, cancel)

    class Finder:
        def find(self, colors, origins, masks):
            return masks

    monkeypatch.setattr(seam, '_clear_labels', clear)
    monkeypatch.setattr(seam.cv2.detail, 'GraphCutSeamFinder', lambda *_: Finder(), raising=False)
    colors = np.zeros((3, 4, 3), np.float32)
    alpha = np.ones((3, 4), np.float32)
    result = build_labels([LowSample(0, colors, alpha, (0, 0)),
                           LowSample(1, colors, alpha, (2, 0))], (3, 6),
                          SeamSpec(mode='graphcut'))
    assert len(calls) == 2
    assert np.all(calls[1] != 0)
    assert np.all(result[:, :2] == 1)
    assert np.all(result[:, 2:] == 2)
