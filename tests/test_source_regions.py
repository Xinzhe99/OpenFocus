"""Native source sampling and bounded asynchronous display consumers."""
import threading

import numpy as np
import pytest
from PyQt6.QtTest import QSignalSpy

from core.source_previews import DISPLAY_REGION_MAX_SIDE, SourceRegionWorker, render_source_region
from utils.image_utils import to_display_uint8


def test_source_region_requests_wait_for_retiring_decode_and_keep_latest(qapp, monkeypatch):
    from types import SimpleNamespace
    from PyQt6.QtCore import QObject, pyqtSignal
    import core.source_previews
    from main import OpenFocus
    from utils import scalebar, watermark

    created = []

    class Worker(QObject):
        ready = pyqtSignal(object, object, object, object)
        failed = pyqtSignal(object, str)
        finished = pyqtSignal()

        def __init__(self, frame, key, **kwargs):
            super().__init__()
            self.key, self.cancelled, self.requests = key, False, []
            created.append(self)

        def start(self):
            pass

        def submit(self, request):
            self.requests.append(request)

        def request_cancel(self):
            self.cancelled = True

    monkeypatch.setattr(core.source_previews, 'SourceRegionWorker', Worker)
    monkeypatch.setattr(scalebar, 'export_cfg', lambda *args, **kwargs: None)
    monkeypatch.setattr(watermark, 'export_cfg', lambda *args, **kwargs: None)
    frames = [np.zeros((10, 10, 3), np.uint8) for _ in range(3)]
    window = SimpleNamespace(raw_images=frames, linear_dng_metadata=None,
                             label_manager=SimpleNamespace(snapshot=lambda: None))
    window._request_source_region = lambda key, request: OpenFocus._request_source_region(window, key, request)
    request = ('view', 1, (0, 0, 10, 10), (10, 10))
    keys = [(0, id(frame), index, index) for index, frame in enumerate(frames)]
    window._source_display_key = keys[0]
    window._request_source_region(keys[0], request)
    for key in keys[1:]:
        window._source_display_key = key
        window._request_source_region(key, request)
    assert len(created) == 1 and created[0].cancelled
    created[0].finished.emit()
    assert len(created) == 2 and created[1].key == keys[2]
    assert created[1].requests == [request]
    window._shutting_down = True
    created[1].finished.emit()


def test_native_region_replaces_thumbnail_on_actual_widget_surface(qapp):
    from widgets.magnifier_label import MagnifierLabel
    from utils.image_utils import cv2_to_pixmap

    image = np.full((600, 800, 3), (20, 100, 220), np.uint8)
    thumbnail = np.zeros((60, 80, 3), np.uint8)
    label = MagnifierLabel()
    label.resize(400, 300)
    requests = []
    label.sourceRegionRequested.connect(lambda key, request: requests.append((key, request)))
    label.show()
    try:
        label.set_display_pixmap(cv2_to_pixmap(thumbnail), source_size=(800, 600), source_key='frame')
        assert label.grab().toImage().pixelColor(200, 150).red() == 0
        key, request = requests[-1]
        tile, rect = render_source_region(image, request[2], request[3])
        assert label.set_source_region(cv2_to_pixmap(tile), rect, source_key=key, request=request)
        color = label.grab().toImage().pixelColor(200, 150)
        assert (color.red(), color.green(), color.blue()) == (220, 100, 20)
    finally:
        label.close()


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

    def resolve(frame, *, should_cancel=None):
        assert callable(should_cancel) and not should_cancel()
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
    cancellation_checks = []

    def resolve(frame, *, should_cancel=None):
        cancellation_checks.append(should_cancel)
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
        assert callable(cancellation_checks[0]) and cancellation_checks[0]()
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
    def resolve(item, *, should_cancel=None):
        calls.append(item)
        return original_resolve(item, should_cancel=should_cancel)
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


def test_previews_publish_each_completed_thumbnail_without_waiting_for_batch(qapp, monkeypatch):
    from core import source_previews
    slow_started, release = threading.Event(), threading.Event()
    frames = [object(), object()]
    owner = threading.get_ident()
    threads = []

    def preview(frame, should_cancel=None):
        threads.append(threading.get_ident())
        if frame is frames[0]:
            slow_started.set()
            assert release.wait(5)
        return np.full((80, 120, 3), 32 if frame is frames[0] else 160, np.uint16)

    monkeypatch.setattr(source_previews, 'ensure_preview', preview)
    monkeypatch.setattr(source_previews, '_preview_parallelism', lambda: 2)
    worker = source_previews.SourcePreviewWorker(frames, 'stack')
    received = QSignalSpy(worker.ready)
    worker.start()
    try:
        assert slow_started.wait(2)
        if not received:
            assert received.wait(2000)
        assert len(received) == 1
        assert received[0][:3] == ['stack', 1, id(frames[1])]
        assert received[0][3].shape == (27, 40, 3)
        assert all(thread != owner for thread in threads)
        release.set()
        assert worker.wait(2000)
        qapp.processEvents()
        assert sorted(item[1] for item in received) == [0, 1]
    finally:
        release.set()
        worker.request_cancel()
        assert worker.wait(2000)


def test_preview_switch_cancels_old_work_and_serializes_same_frame(qapp, monkeypatch):
    from core import source_previews
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    frame = object()
    active, calls = [], []

    def preview(source, should_cancel=None):
        with lock:
            active.append(source)
            calls.append(source)
            assert len(active) == 1
            first = len(calls) == 1
        try:
            if first:
                entered.set()
                assert release.wait(5)
            return np.full((12, 16, 3), 42, np.uint8)
        finally:
            with lock:
                active.remove(source)

    monkeypatch.setattr(source_previews, 'ensure_preview', preview)
    monkeypatch.setattr(source_previews, '_preview_parallelism', lambda: 2)
    old = source_previews.SourcePreviewWorker([frame], 'old', [0])
    new = source_previews.SourcePreviewWorker([frame], 'new', [0])
    old_ready, new_ready = QSignalSpy(old.ready), QSignalSpy(new.ready)
    old.start()
    try:
        assert entered.wait(2)
        old.request_cancel()
        new.start()
        release.set()
        assert old.wait(2000) and new.wait(2000)
        qapp.processEvents()
        assert not old_ready
        assert len(new_ready) == 1
        assert new_ready[0][:3] == ['new', 0, id(frame)]
        assert calls == [frame, frame]
    finally:
        release.set()
        old.request_cancel()
        new.request_cancel()
        assert old.wait(2000) and new.wait(2000)


def test_preview_failure_does_not_block_other_thumbnails(qapp, monkeypatch):
    from core import source_previews
    frames = [object(), object(), object()]

    def preview(frame, should_cancel=None):
        if frame is frames[1]:
            raise OSError('preview failed')
        return np.full((10, 20, 3), 80, np.uint8)

    monkeypatch.setattr(source_previews, 'ensure_preview', preview)
    monkeypatch.setattr(source_previews, '_preview_parallelism', lambda: 2)
    worker = source_previews.SourcePreviewWorker(frames, 'stack')
    received, failed = QSignalSpy(worker.ready), QSignalSpy(worker.failed)
    worker.start()
    try:
        assert worker.wait(2000)
        qapp.processEvents()
        assert sorted(row[1] for row in received) == [0, 2]
        assert list(failed[0]) == ['stack', 'preview failed']
        assert len(failed) == 1
    finally:
        worker.request_cancel()
        assert worker.wait(2000)


def test_preview_parallelism_respects_memory_and_cpu_budget(monkeypatch):
    from core import resource_policy, source_previews
    snapshot = resource_policy.ResourceSnapshot(physical_cores=8, logical_cpus=8)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda: snapshot)
    for budget, expected in [(0, 1), (source_previews._PREVIEW_WORKING_BYTES, 1),
                             (2 * source_previews._PREVIEW_WORKING_BYTES, 2),
                             (20 * source_previews._PREVIEW_WORKING_BYTES, 2)]:
        monkeypatch.setattr(resource_policy, 'ram_budget', lambda unused, value=budget: value)
        assert source_previews._preview_parallelism() == expected
    monkeypatch.setattr(resource_policy, 'cpu_capacity', lambda unused, preference: 1)
    assert source_previews._preview_parallelism() == 1


def test_selected_preview_takes_turn_before_next_batch(qapp, monkeypatch):
    from core import source_previews
    first_started, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    frames = [object() for _ in range(4)]
    order = []

    def preview(frame, should_cancel=None):
        with lock:
            order.append(frame)
        if frame is frames[0]:
            first_started.set()
            assert release.wait(5)
        return np.zeros((10, 10, 3), np.uint8)

    monkeypatch.setattr(source_previews, 'ensure_preview', preview)
    monkeypatch.setattr(source_previews, '_preview_parallelism', lambda: 1)
    bulk = source_previews.SourcePreviewWorker(frames, 'bulk', [0, 1, 2])
    selected = source_previews.SourcePreviewWorker(frames, 'selected', [3])
    bulk.start()
    try:
        assert first_started.wait(2)
        selected.start()
        # Observe the selected worker entering the admission queue before
        # releasing the currently executing preview.
        with source_previews._PREVIEW_CONDITION:
            assert source_previews._PREVIEW_CONDITION.wait_for(
                lambda: source_previews._PREVIEW_PRIORITY_WAITERS == 1, timeout=2)
        release.set()
        assert bulk.wait(2000) and selected.wait(2000)
        assert order == [frames[0], frames[3], frames[1], frames[2]]
    finally:
        release.set()
        bulk.request_cancel()
        selected.request_cancel()
        assert bulk.wait(2000) and selected.wait(2000)


def test_cancelled_stack_does_not_start_remaining_frames_or_exceed_limit(qapp, monkeypatch):
    from core import source_previews
    both_started, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    frames = [object() for _ in range(8)]
    active = 0
    peak = 0
    calls = []

    def preview(frame, should_cancel=None):
        nonlocal active, peak
        with lock:
            calls.append(frame)
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_started.set()
        try:
            assert release.wait(5)
            return np.zeros((10, 10, 3), np.uint8)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(source_previews, 'ensure_preview', preview)
    monkeypatch.setattr(source_previews, '_preview_parallelism', lambda: 2)
    old = source_previews.SourcePreviewWorker(frames, 'deleted')
    new = source_previews.SourcePreviewWorker(frames, 'replacement', [5])
    old_ready, new_ready = QSignalSpy(old.ready), QSignalSpy(new.ready)
    old.start()
    try:
        assert both_started.wait(2)
        old.request_cancel()
        new.start()
        release.set()
        assert old.wait(2000) and new.wait(2000)
        qapp.processEvents()
        assert peak == 2
        assert set(calls) == {frames[0], frames[1], frames[5]}
        assert not old_ready
        assert len(new_ready) == 1 and new_ready[0][2] == id(frames[5])
    finally:
        release.set()
        old.request_cancel()
        new.request_cancel()
        assert old.wait(2000) and new.wait(2000)


def test_preview_waits_for_active_render_and_can_be_cancelled(qapp, monkeypatch):
    from core import resource_policy, source_previews
    render_started, release = threading.Event(), threading.Event()
    decoded = threading.Event()

    def render():
        with resource_policy.heavy_job():
            render_started.set()
            assert release.wait(5)

    def preview(frame, should_cancel=None):
        decoded.set()
        return np.zeros((10, 10, 3), np.uint8)

    monkeypatch.setattr(source_previews, 'ensure_preview', preview)
    owner = threading.Thread(target=render)
    owner.start()
    worker = source_previews.SourcePreviewWorker([object()], 'pending')
    received = QSignalSpy(worker.ready)
    try:
        assert render_started.wait(2)
        worker.start()
        worker.request_cancel()
        assert worker.wait(2000)
        qapp.processEvents()
        assert not decoded.is_set() and not received
    finally:
        release.set()
        owner.join(2)
        worker.request_cancel()
        assert worker.wait(2000)


def test_selected_preview_is_available_during_heavy_work(qapp, tmp_path):
    import cv2
    from core import resource_policy, source_previews, source_frames

    image = np.full((64, 96, 3), (20, 80, 140), np.uint8)
    path = tmp_path / 'selected.png'
    assert cv2.imwrite(str(path), image)
    frame = source_frames.scan_source(str(path), header_only=True)[0]
    started, release = threading.Event(), threading.Event()

    def active_job():
        with resource_policy.heavy_job():
            started.set()
            assert release.wait(5)

    owner = threading.Thread(target=active_job)
    owner.start()
    worker = source_previews.SourcePreviewWorker([frame], 'selected', [0], selected_display=True)
    received = QSignalSpy(worker.ready)
    try:
        assert started.wait(2)
        worker.start()
        if not received:
            assert received.wait(2000)
        assert not release.is_set()
        np.testing.assert_array_equal(received[0][3][0, 0], image[0, 0])
    finally:
        release.set()
        owner.join(2)
        worker.request_cancel()
        assert worker.wait(2000)


def test_selected_preview_bypasses_bulk_preview_waiting_for_render(qapp, tmp_path, monkeypatch):
    import cv2
    from contextlib import contextmanager
    from core import resource_policy, source_previews, source_frames

    image = np.full((64, 96, 3), (20, 80, 140), np.uint8)
    path = tmp_path / 'selected-during-bulk.png'
    assert cv2.imwrite(str(path), image)
    frame = source_frames.scan_source(str(path), header_only=True)[0]
    started, release, bulk_waiting = threading.Event(), threading.Event(), threading.Event()
    original_heavy_job = resource_policy.heavy_job

    @contextmanager
    def observed_heavy_job(*args, **kwargs):
        bulk_waiting.set()
        with original_heavy_job(*args, **kwargs):
            yield

    monkeypatch.setattr(resource_policy, 'heavy_job', observed_heavy_job)

    def active_job():
        with original_heavy_job():
            started.set()
            release.wait(5)

    owner = threading.Thread(target=active_job)
    bulk = source_previews.SourcePreviewWorker([frame, frame], 'bulk')
    selected = source_previews.SourcePreviewWorker([frame], 'selected', [0], selected_display=True)
    received = QSignalSpy(selected.ready)
    owner.start()
    try:
        assert started.wait(2)
        bulk.start()
        assert bulk_waiting.wait(2)
        selected.start()
        if not received:
            assert received.wait(1000)
        assert not release.is_set()
        np.testing.assert_array_equal(received[0][3][0, 0], image[0, 0])
    finally:
        release.set()
        owner.join(2)
        bulk.request_cancel()
        selected.request_cancel()
        assert bulk.wait(2000) and selected.wait(2000)


@pytest.mark.parametrize('dtype,channels', [(np.uint16, 3), (np.float32, 3), (np.uint8, 4)])
def test_preview_thumbnail_preserves_original_pixels(qapp, dtype, channels):
    import cv2
    from core import source_previews
    image = np.arange(12 * 20 * channels, dtype=dtype).reshape(12, 20, channels)
    before = image.copy()
    worker = source_previews.SourcePreviewWorker([image], 'pixels')
    received = QSignalSpy(worker.ready)
    worker.start()
    try:
        assert worker.wait(2000)
        qapp.processEvents()
        assert len(received) == 1
        expected = to_display_uint8(cv2.resize(before, (20, 12), interpolation=cv2.INTER_AREA))
        np.testing.assert_array_equal(received[0][3], expected)
        np.testing.assert_array_equal(image, before)
    finally:
        worker.request_cancel()
        assert worker.wait(2000)
