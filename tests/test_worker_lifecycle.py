"""Real worker lifecycle regressions: admission, cancellation and disk cache."""
from threading import Event, get_ident

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt

from core.compare_worker import PreviewWorker
from core.workers import RenderWorker
from utils import align_cache


@pytest.fixture(autouse=True)
def _application(qapp):
    return qapp


def _frames(dtype=np.uint8):
    image = np.random.default_rng(518).integers(20, 220, (64, 80, 3), dtype=np.uint8)
    frames = [image, cv2.GaussianBlur(image, (5, 5), 1)]
    return [frame.astype(dtype) * 257 for frame in frames] if dtype == np.uint16 else frames


def _render(frames, request=None):
    return RenderWorker(
        frames, algorithm='openfocus_a', alignment='openfocus' if request is not None else 'none',
        fusion_parameters={'kernel_size': 11}, tile_enabled=False,
        thread_count=1, use_gpu=False, exposure_match=False, disk_cache=request,
    )


def _capture(worker):
    results, errors = [], []
    worker.finished_signal.connect(lambda *args: results.append(args), Qt.ConnectionType.DirectConnection)
    worker.error_signal.connect(errors.append, Qt.ConnectionType.DirectConnection)
    return results, errors


@pytest.mark.parametrize("cancel_waiter", [False, True])
def test_render_lease_covers_normalization_and_cancellable_wait(monkeypatch, cancel_waiter):
    import utils.image_utils as image_utils

    first = _render(_frames(np.uint16))
    second = PreviewWorker(_frames(np.uint16), ["openfocus_a"], {},
                           {"thread_count": 1, "tile_enabled": False}, None, roi_only=False)
    entered, release, second_entered = Event(), Event(), Event()
    original = image_utils.normalize_fuse_input
    owner = []

    def normalize(images):
        output = original(images)
        if not owner:
            owner.append(get_ident())
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release normalization")
        elif get_ident() != owner[0]:
            second_entered.set()
        return output

    monkeypatch.setattr(image_utils, "normalize_fuse_input", normalize)
    results, errors = _capture(first)
    previews = []
    second.result_ready.connect(lambda *args: previews.append(args), Qt.ConnectionType.DirectConnection)
    first.start()
    try:
        assert entered.wait(10)
        second.start()
        first.cancel()
        if cancel_waiter:
            second.cancel()
            assert second.wait(2000), "a cancelled waiter must finish before the owner"
        assert not second_entered.wait(0.15), "normalization escaped the heavy-job lease"
        assert first.isRunning(), "cancellation released unfinished work"
    finally:
        release.set()
        assert first.wait(10000)
        if second.isRunning():
            assert second.wait(10000)
    assert results == []
    assert errors and errors[0].startswith("CANCELLED")
    assert bool(previews) is not cancel_waiter
    assert second_entered.is_set() is not cancel_waiter


def _request(tmp_path, frames):
    folder = tmp_path / "stack"
    folder.mkdir()
    names = []
    for index, image in enumerate(frames):
        name = f"frame{index}.png"
        assert cv2.imwrite(str(folder / name), image)
        names.append(name)
    return {"mode": "load", "save": True, "folder": str(folder),
            "filenames": names, "align_options": ('openfocus',),
            "downscale_width": 1024, "frame_shape": frames[0].shape[:2],
            "identity": {"brightness_reference": "first", "exposure_match": False,
                         "dtype": frames[0].dtype.str, "backend": "opencv"}}


def test_fresh_alignment_writes_in_background_then_hit_is_not_rewritten(tmp_path, monkeypatch):
    frames = _frames()
    request = _request(tmp_path, frames)
    original = align_cache.save_aligned
    writers = []

    def save(*args, **kwargs):
        writers.append(get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(align_cache, "save_aligned", save)
    first = _render(frames, request)
    first_results, first_errors = _capture(first)
    first.start()
    assert first.wait(30000)
    assert first_errors == [] and len(first_results) == 1
    assert first.alignment_computed and not first.alignment_reused
    assert writers and writers[0] != get_ident()
    second = _render(frames, request)
    second_results, second_errors = _capture(second)
    second.start()
    assert second.wait(30000)
    assert second_errors == [] and len(second_results) == 1
    assert second.alignment_reused and not second.alignment_computed
    assert len(writers) == 1
    np.testing.assert_array_equal(first_results[0][1], second_results[0][1])


def test_cache_write_failure_keeps_real_fusion_result(tmp_path, monkeypatch):
    frames = _frames()
    request = _request(tmp_path, frames)
    reference = _render(frames, {**request, "save": False})
    expected, reference_errors = _capture(reference)
    reference.run()
    assert reference_errors == [] and len(expected) == 1
    worker = _render(frames, request)
    writes = []

    def fail(*args, **kwargs):
        writes.append(True)
        raise OSError("disk full")

    monkeypatch.setattr(align_cache, "save_aligned", fail)
    results, errors = _capture(worker)
    worker.run()
    assert errors == [] and len(results) == 1
    assert results[0][1].dtype == frames[0].dtype
    assert writes == [True]
    np.testing.assert_array_equal(results[0][1], expected[0][1])


def test_cancel_after_native_fusion_does_not_publish_or_cache(tmp_path, monkeypatch):
    from core.multi_focus_fusion import MultiFocusFusion

    frames = _frames()
    worker = _render(frames, _request(tmp_path, frames))
    original = MultiFocusFusion.fuse
    writes = []

    def fuse(instance, *args, **kwargs):
        result = original(instance, *args, **kwargs)
        worker.cancel()
        return result

    monkeypatch.setattr(MultiFocusFusion, "fuse", fuse)
    monkeypatch.setattr(align_cache, "save_aligned", lambda *args, **kwargs: writes.append(True))
    results, errors = _capture(worker)
    worker.run()
    assert results == [] and writes == []
    assert errors and errors[0].startswith("CANCELLED")


def test_cache_identity_tracks_decoded_pixels_even_when_source_files_do_not_change(tmp_path):
    frames = _frames()
    request = _request(tmp_path, frames)
    first = _render(frames, request)
    first_results, first_errors = _capture(first)
    first.run()
    assert first_errors == [] and first_results
    changed = [np.clip(frame.astype(np.int16) + 10, 0, 255).astype(np.uint8) for frame in frames]
    second = _render(changed, request)
    second_results, second_errors = _capture(second)
    second.run()
    assert second_errors == [] and second_results
    assert second.alignment_computed and not second.alignment_reused
    assert not np.array_equal(first_results[0][1], second_results[0][1])


def test_preview_resize_runs_under_worker_lifetime(monkeypatch):
    import core.workers as workers
    image = np.tile(_frames()[0], (8, 8, 1))
    worker = _render([image, image.copy()])
    worker.preview = True
    original = workers._downscale_for_preview
    resizing_threads = []

    def resize(frame):
        resizing_threads.append(get_ident())
        return original(frame)

    monkeypatch.setattr(workers, "_downscale_for_preview", resize)
    results, errors = _capture(worker)
    worker.start()
    assert worker.wait(30000)
    assert errors == [] and results
    assert all(thread != get_ident() for thread in resizing_threads)
    assert len(resizing_threads) == 2
    assert results[0][1].shape[:2] == (256, 320)
