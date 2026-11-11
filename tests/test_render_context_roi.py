"""Cropped and trimmed outputs own pixels without pinning the full frames."""
import gc
import os
import weakref

import numpy as np
import pytest

from core.frame_storage import create_frame
from core.render_context import build_render_context


def test_roi_render_context_does_not_keep_in_memory_frames_alive():
    """An ROI output's provenance holds copies of the crops, not views that
    would pin every full-size aligned frame while the output exists."""
    full = [np.full((200, 300, 3), 40 * (i + 1), np.uint16) for i in range(3)]
    crop = full[0][50:150, 60:260]
    ctx = build_render_context(full, crop.copy(), filenames=["a", "b", "c"], algorithm="test",
                               roi_rect=(60, 50, 200, 100), roi_mode="crop")
    assert ctx is not None
    for frame, source in zip(ctx.frames, full):
        assert frame.image.shape == (100, 200, 3)
        assert frame.image.dtype == np.uint16
        assert not np.shares_memory(frame.image, source)
        np.testing.assert_array_equal(frame.image, source[50:150, 60:260])


def test_roi_render_context_owns_mapped_crops_and_releases_the_full_frames():
    """A mapped aligned frame's crop is stored as its own mapping; dropping
    the full frames really unlinks their temporary files."""
    full = []
    paths = []
    for i in range(3):
        frame = create_frame((200, 300, 3), np.uint8)
        frame[...] = 40 * (i + 1)
        full.append(frame)
        paths.append(frame.filename)
    crop = np.array(full[0][50:150, 60:260])
    ctx = build_render_context(full, crop, filenames=["a", "b", "c"], algorithm="test",
                               roi_rect=(60, 50, 200, 100), roi_mode="crop")
    assert ctx is not None
    for frame, source in zip(ctx.frames, full):
        assert frame.image.shape == (100, 200, 3)
        assert frame.image.dtype == np.uint8
        assert not np.shares_memory(frame.image, source)
        np.testing.assert_array_equal(frame.image, source[50:150, 60:260])
    del full, frame, source
    gc.collect()
    assert not any(os.path.exists(path) for path in paths)


@pytest.mark.parametrize('mapped', [False, True])
def test_trimmed_context_releases_full_source_storage(mapped):
    shape = (200, 300, 3)
    source = create_frame(shape, np.uint16) if mapped else np.empty(shape, np.uint16)
    source[...] = np.arange(shape[1], dtype=np.uint16)[None, :, None] + 12001
    expected = source[:192, :288].copy()
    path = source.filename if mapped else None
    reference = weakref.ref(source)
    context = build_render_context([source], expected.copy(), algorithm='dct')
    assert context is not None
    np.testing.assert_array_equal(context.frames[0].image, expected)
    del source
    gc.collect()
    assert reference() is None
    if mapped:
        assert not os.path.exists(path)
    np.testing.assert_array_equal(context.frames[0].image, expected)


def test_reopened_strided_roi_owns_an_admitted_crop_after_sources_release(tmp_path, monkeypatch):
    from core import frame_storage
    from core import resource_policy

    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    frame_storage.configure_storage_directory(tmp_path)
    monkeypatch.setattr(resource_policy, 'frame_ram_budget', lambda snapshot: 0)
    source = frame_storage.store_frame(
        np.arange(48 * 64 * 3, dtype=np.uint16).reshape(48, 64, 3), prefer_ram=False)
    path = source.filename
    reopened = frame_storage.open_frame(
        frame_storage.describe_frame(source[::-1]), owned=True)
    expected = np.array(reopened[8:24, 12:36])
    context = build_render_context([reopened], expected.copy(), algorithm='test',
                                   roi_rect=(12, 8, 24, 16), roi_mode='crop')
    assert context is not None
    description = frame_storage.describe_frame(context.frames[0].image)
    assert description is not None and description['path'] != path
    del source, reopened
    gc.collect()
    assert not os.path.exists(path)
    np.testing.assert_array_equal(context.frames[0].image, expected)
    crop_path = description['path']
    del context
    gc.collect()
    assert not os.path.exists(crop_path)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('pattern', ['nonzero', 'border', 'interior', 'black'])
def test_aligned_grayscale_processed_mask_matches_array(dtype, pattern):
    from core.source_frames import ProcessedFrame

    image = np.ones((5, 6), dtype=dtype)
    if pattern == 'border':
        image[:, 0] = 0
    elif pattern == 'interior':
        image[2, 3] = 0
    elif pattern == 'black':
        image[:] = 0
    deferred = ProcessedFrame(image, lambda pixels: pixels.copy(), image.shape, image.dtype)
    expected = build_render_context([image], image, aligned=True).frames[0].valid_mask()
    actual = build_render_context([deferred], image, aligned=True).frames[0].valid_mask()
    if expected is None:
        assert actual is None
    else:
        assert actual.dtype == bool
        np.testing.assert_array_equal(actual, expected)


def test_aligned_mask_cancellation_after_connected_components(monkeypatch):
    import core.render_context as render_context
    from core.registration import RegistrationCancelled

    image = np.ones((5, 6), np.uint8)
    image[:, 0] = 0
    cancelled = [False]
    original = render_context.cv2.connectedComponents

    def components(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled[0] = True
        return result

    monkeypatch.setattr(render_context.cv2, 'connectedComponents', components)
    context = build_render_context([image], image, aligned=True)
    with pytest.raises(RegistrationCancelled):
        context.frames[0].valid_mask(should_cancel=lambda: cancelled[0])


@pytest.mark.parametrize('mapped', [False, True])
def test_context_worker_owns_roi_crop(mapped, qapp):
    from PyQt6.QtCore import Qt
    from core.workers import RenderContextWorker

    source = create_frame((12, 16), np.uint16) if mapped else np.empty((12, 16), np.uint16)
    source[:] = np.arange(16, dtype=np.uint16)
    expected = source[2:8, 3:11].copy()
    worker = RenderContextWorker([source], expected, roi_rect=(3, 2, 8, 6), roi_mode='crop')
    ready, failed = [], []
    worker.ready.connect(ready.append, Qt.ConnectionType.DirectConnection)
    worker.failed.connect(failed.append, Qt.ConnectionType.DirectConnection)
    worker.run()
    assert not failed and len(ready) == 1
    np.testing.assert_array_equal(ready[0].frames[0].image, expected)
    assert not np.shares_memory(ready[0].frames[0].image, source)


def test_context_worker_cancelled_before_build(qapp):
    from PyQt6.QtCore import Qt
    from core.workers import RenderContextWorker

    image = np.ones((4, 4), np.uint8)
    worker = RenderContextWorker([image], image)
    ready, failed = [], []
    worker.ready.connect(ready.append, Qt.ConnectionType.DirectConnection)
    worker.failed.connect(failed.append, Qt.ConnectionType.DirectConnection)
    worker.cancel()
    worker.run()
    assert not ready and failed == ['CANCELLED: render context cancelled']
