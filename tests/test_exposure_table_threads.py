"""Serial frame preparation lends its warp threads to exposure matching, exactly."""
import threading

import cv2
import numpy as np
import pytest

from core import image_loader, registration, resource_policy
from core import source_frames as sources
from fusion_methods import _openfocus_kernels


@pytest.fixture
def lent_threads(monkeypatch):
    # Lend three threads whatever the runner's core count and OpenCV backend.
    monkeypatch.setattr(registration, '_active_backend_bounds_threads', lambda: True)
    monkeypatch.setattr(resource_policy, 'native_thread_ceiling', lambda *args, **kwargs: 3)


@pytest.fixture
def table_threads(monkeypatch):
    threads = set()
    table_rows = _openfocus_kernels.table_rows

    def spy(*args):
        threads.add(threading.get_ident())
        return table_rows(*args)

    monkeypatch.setattr(_openfocus_kernels, 'table_rows', spy)
    return threads


def _png16(path, pixels):
    assert cv2.imwrite(str(path), pixels)
    return sources.scan_source(str(path), header_only=True)[0]


def test_serial_snapshot_matches_serial_warp_and_exposure_bit_for_bit(
        tmp_path, lent_threads, table_threads):
    from core import frame_resolution_cache, frame_storage

    rng = np.random.default_rng(5207)
    bright = rng.integers(0, 65536, (520, 660, 3), dtype=np.uint16)
    dark = (bright * 0.81).astype(np.uint16)
    frames = [_png16(tmp_path / 'bright.png', bright), _png16(tmp_path / 'dark.png', dark)]
    identities = tuple(registration._perspective_source_file(frame, None) for frame in frames)
    matrices = ((1., 0., -4.375, 0., 1., -2.25, 0., 0., 1.),
                (1.001, -.002, -3.6, .0015, .999, -1.9, 2e-6, -1e-6, 1.))
    stage = registration.PerspectiveStage('ecc', identities, matrices, ('float64', 'float64'),
                                          (2, 514, 4, 652), (648, 512))
    warped = stage.apply(frames, defer_warps=True)
    assert sources._uses_exposure_table(warped[1].shape, warped[1].dtype)

    # Reference: OpenCV's warp of each decoded source, then serial matching.
    expected = sources.match_exposure_frames(
        [frame.processor(image_loader.decode_image(frame.source.path)) for frame in warped],
        reference_index=0)
    expected = [np.array(sources.resolve_frame(frame)) for frame in expected]
    assert not np.array_equal(expected[1], warped[1].processor(image_loader.decode_image(
        frames[1].path)))
    table_threads.clear()

    with (frame_storage.scratch_directory(explicit_directory=str(tmp_path)),
          frame_resolution_cache.render_scope(frames) as scope):
        with scope.reuse_only(), sources.without_source_cache_growth():
            matched = sources.match_exposure_frames(warped, reference_index=0)
            # Large stacks stage processed frames one at a time on this thread.
            staged = sources.materialize_for_repeated_access(matched, parallel_processed=False)
        for snapshot, pixels in zip(staged, expected):
            assert isinstance(snapshot, sources.PreparedFrame)
            np.testing.assert_array_equal(sources.resolve_frame(snapshot), pixels)
    assert len(table_threads) > 1


@pytest.mark.parametrize('channels', [1, 3, 4])
def test_lent_exposure_table_bands_equal_serial_bands(lent_threads, table_threads, channels):
    rng = np.random.default_rng(81 + channels)
    shape = (900, 1100) if channels == 1 else (600, 700, channels)
    image = rng.integers(0, 65536, shape, dtype=np.uint16)
    assert sources._uses_exposure_table(image.shape, image.dtype)
    serial = np.empty_like(image)
    sources._match_through_table(image, 1.13, serial, None)
    assert len(table_threads) == 1

    lent = np.empty_like(image)
    with registration.deferred_warp_threads(3):
        sources._match_through_table(image, 1.13, lent, lambda: False)
    np.testing.assert_array_equal(lent, serial)
    assert len(table_threads) > 1


def test_lent_exposure_table_stops_when_cancelled(lent_threads):
    image = np.zeros((2000, 700, 3), np.uint16)
    output = np.empty_like(image)
    checks = []

    def cancelled():
        checks.append(None)
        return len(checks) > 1

    with registration.deferred_warp_threads(3), pytest.raises(registration.RegistrationCancelled):
        sources._match_through_table(image, 1.2, output, cancelled)
