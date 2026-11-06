"""Native source sampling and bounded asynchronous display consumers."""
import threading

import numpy as np
import pytest
from PyQt6.QtTest import QSignalSpy

from core.source_previews import DISPLAY_REGION_MAX_SIDE, SourceRegionWorker, render_source_region
from utils.image_utils import to_display_uint8


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [None, 3])
def test_native_region_owns_only_visible_pixels(dtype, channels):
    # A 100MP domain with one broadcast pixel uses no 100MP physical buffer.
    shape = (10000, 10000) + (() if channels is None else (channels,))
    pixel = np.array(91 if dtype == np.uint8 else 23456, dtype=dtype)
    image = np.broadcast_to(pixel, shape)
    tile, rect = render_source_region(image, (7123.25, 8467.75, 51.5, 37.5), (640, 480))
    assert rect == (7123, 8467, 52, 39)
    expected = to_display_uint8(np.full(tile.shape, pixel, dtype=dtype))
    np.testing.assert_array_equal(tile, expected)
    assert tile.flags.owndata
    assert tile.nbytes <= 52 * 39 * (channels or 1)


def test_visible_crop_preserves_native_edges_and_fractional_domain():
    image = np.arange(71 * 97, dtype=np.uint16).reshape(71, 97)
    tile, rect = render_source_region(image, (-12.2, 65.25, 40, 20), (200, 200))
    assert rect == (0, 65, 28, 6)
    np.testing.assert_array_equal(tile, to_display_uint8(image[65:71, :28]))
    empty, rect = render_source_region(image, (-50, -40, 10, 10), (100, 100))
    assert empty is None
    assert rect[2:] == (0, 0)


def test_region_display_allocation_is_capped_independently_of_requested_size():
    image = np.full((48, 64, 3), 170, np.uint8)
    tile, rect = render_source_region(image, (0, 0, 64, 48), (100000, 100000))
    # Enlarging a tile is the painter's job; never allocate zoomed source images.
    assert tile.shape == image.shape
    assert max(tile.shape[:2]) <= DISPLAY_REGION_MAX_SIDE
    assert rect == (0, 0, 64, 48)
    tall = np.full((5000, 16, 3), 170, np.uint8)
    capped, _ = render_source_region(tall, (0, 0, 16, 5000), (100000, 100000))
    assert capped.shape[0] == DISPLAY_REGION_MAX_SIDE
    assert max(capped.shape[:2]) <= DISPLAY_REGION_MAX_SIDE
    assert capped.nbytes <= DISPLAY_REGION_MAX_SIDE * 16 * 3


def test_selected_source_decode_coalesces_requests_and_never_runs_on_gui(qapp, monkeypatch):
    from core import source_frames
    entered, release = threading.Event(), threading.Event()
    owner = threading.get_ident()
    decoded_threads = []
    image = np.full((128, 192, 3), 83, np.uint16)

    def resolve(frame):
        decoded_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        return image

    monkeypatch.setattr(source_frames, 'resolve_frame', resolve)
    worker = SourceRegionWorker(object(), 'generation')
    received = QSignalSpy(worker.ready)
    failed = QSignalSpy(worker.failed)
    worker.submit(('loupe', 0, (10, 10, 32, 24), (100, 100)))
    worker.start()
    try:
        assert entered.wait(2)
        for serial in range(1, 40):
            worker.submit(('loupe', serial, (serial, 10, 32, 24), (100, 100)))
        latest = ('loupe', 40, (40, 10, 32, 24), (100, 100))
        worker.submit(latest)
        release.set()
        if not received:
            received.wait(2000)
        assert received, list(failed)
        assert len(received) == 1
        assert received[0][0] == 'generation'
        assert received[0][1] == latest
        assert received[0][2].shape == (24, 32, 3)
        assert len(decoded_threads) == 1
        assert decoded_threads[0] != owner
    finally:
        release.set()
        worker.request_cancel()
        assert worker.wait(2000)


def test_cancelled_original_cannot_publish_a_late_tile(qapp, monkeypatch):
    from core import source_frames
    entered, release = threading.Event(), threading.Event()

    def resolve(frame):
        entered.set()
        assert release.wait(2)
        return np.full((80, 100, 3), 128, np.uint8)

    monkeypatch.setattr(source_frames, 'resolve_frame', resolve)
    worker = SourceRegionWorker(object(), 'old-generation')
    received = QSignalSpy(worker.ready)
    worker.submit(('view', 1, (10, 10, 20, 20), (20, 20)))
    worker.start()
    try:
        assert entered.wait(2)
        worker.request_cancel()
        release.set()
        assert worker.wait(2000)
        qapp.processEvents()
        assert len(received) == 0
    finally:
        release.set()
        worker.request_cancel()
        assert worker.wait(2000)


def test_native_camera_linear_region_matches_full_display_development():
    from utils.linear_dng import LinearDNGMetadata, develop_linear
    metadata = LinearDNGMetadata(((0.8293, -0.1611, -0.1132),
                                   (-0.4759, 1.2711, 0.2275),
                                   (-0.0307, 0.1334, 0.8147)), (0.5, 1.0, 0.75))
    image = np.random.default_rng(73).integers(0, 50000, (96, 128, 3), dtype=np.uint16)
    full = to_display_uint8(develop_linear(image, metadata))
    tile, rect = render_source_region(image, (29, 17, 43, 51), (300, 300), metadata)
    assert rect == (29, 17, 43, 51)
    np.testing.assert_array_equal(tile, full[17:68, 29:72])


def test_native_transparent_crop_keeps_global_checkerboard_phase():
    from utils.image_utils import composite_transparent_for_display
    image = np.full((83, 101, 4), (22, 74, 138, 97), np.uint8)
    full = composite_transparent_for_display(image)
    tile, _ = render_source_region(image, (19, 29, 41, 37), (200, 200))
    np.testing.assert_array_equal(tile, full[29:66, 19:60])


def test_fractional_native_crop_does_not_resample_individual_pixels():
    image = np.random.default_rng(81).integers(0, 256, (90, 120, 3), dtype=np.uint8)
    tile, rect = render_source_region(image, (19.25, 29.75, 41, 37), (41, 37))
    assert rect == (19, 29, 42, 38)
    np.testing.assert_array_equal(tile, image[29:67, 19:61])


def test_uncacheable_selected_original_is_resolved_once_across_requests(qapp, tmp_path, monkeypatch):
    import cv2
    from core import source_frames
    from PyQt6.QtCore import Qt
    image = np.full((96, 128, 3), 173, np.uint8)
    path = tmp_path / 'uncached.png'
    assert cv2.imwrite(str(path), image)
    frame = source_frames.scan_source(str(path), header_only=True)[0]
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(max_bytes=0))
    original_resolve = source_frames.resolve_frame
    calls = []
    def resolve(item):
        calls.append(item)
        return original_resolve(item)
    monkeypatch.setattr(source_frames, 'resolve_frame', resolve)
    ready = threading.Event()
    results = []
    worker = SourceRegionWorker(frame, 'uncached')
    def receive(*args):
        results.append(args)
        ready.set()
    worker.ready.connect(receive, Qt.ConnectionType.DirectConnection)
    worker.submit(('loupe', 1, (10, 10, 24, 24), (24, 24)))
    worker.start()
    try:
        assert ready.wait(2)
        ready.clear()
        worker.submit(('loupe', 2, (40, 40, 24, 24), (24, 24)))
        assert ready.wait(2)
        assert len(calls) == 1
        assert [result[1][1] for result in results] == [1, 2]
        for result in results:
            np.testing.assert_array_equal(result[2], np.full((24, 24, 3), 173, np.uint8))
    finally:
        worker.request_cancel()
        assert worker.wait(2000)


def test_region_memory_rejection_preserves_original_pixels(monkeypatch):
    from core import resource_policy
    image = np.full((96, 128, 3), 43000, np.uint16)
    before = image.copy()
    def reject(_required, **_kwargs):
        raise MemoryError('display budget')
    monkeypatch.setattr(resource_policy, 'require_memory', reject)
    with pytest.raises(MemoryError, match='display budget'):
        render_source_region(image, (0, 0, 128, 96), (128, 96))
    np.testing.assert_array_equal(image, before)


def test_active_region_waits_for_render_and_coalesces_queued_view(qapp):
    from PyQt6.QtCore import Qt
    from core.resource_policy import heavy_job
    image = np.arange(80 * 100, dtype=np.uint16).reshape(80, 100)
    worker = SourceRegionWorker(image, 'lease')
    received, held, release = threading.Event(), threading.Event(), threading.Event()
    results = []
    def receive(_key, request, pixels, _rect):
        results.append((request, pixels, held.is_set()))
        received.set()
    worker.ready.connect(receive, Qt.ConnectionType.DirectConnection)
    worker.submit(('view', 1, (0, 0, 20, 20), (20, 20)))
    worker.start()
    owner = None
    try:
        assert received.wait(2)
        received.clear()
        def render():
            with heavy_job():
                held.set()
                release.wait(3)
                held.clear()
        owner = threading.Thread(target=render)
        owner.start()
        assert held.wait(2)
        worker.submit(('view', 2, (10, 10, 20, 20), (20, 20)))
        assert not received.wait(.1)
        latest = ('view', 3, (30, 30, 20, 20), (20, 20))
        worker.submit(latest)
        release.set()
        assert received.wait(2)
        assert results[-1][0] == latest
        np.testing.assert_array_equal(results[-1][1], to_display_uint8(image[30:50, 30:50]))
        assert not any(result[2] for result in results)
    finally:
        release.set()
        worker.request_cancel()
        assert worker.wait(2000)
        if owner is not None:
            owner.join(2)
