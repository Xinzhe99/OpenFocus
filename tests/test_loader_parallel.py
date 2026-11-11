"""Bounded decoding must preserve order, identity and the strict error rules.

Parallel decodes finish out of order, but a stack is an ordered sequence of
frames: names, real paths, TIFF page identities and metadata all belong to
the input position, not to whichever thread returned first.
"""
import os
import threading

import cv2
import numpy as np
import pytest

from core import image_loader as loader_module
from core.image_loader import (ImageStackLoader, iter_decoded_sources,
                               _DecodeCancelled)


def _write(folder, name, value, size=(16, 20), dtype=np.uint8):
    path = os.path.join(str(folder), name)
    high = 65535 if dtype == np.uint16 else 255
    data = np.full((*size, 3), min(value, high), dtype=dtype)
    assert cv2.imwrite(path, data)
    return path


def test_parallel_decode_preserves_input_order_and_pixels(tmp_path):
    paths = [_write(tmp_path, f"img{i}.png", i) for i in range(9)]
    loaded = list(iter_decoded_sources(paths, thread_count=4))
    assert [item.path for item in loaded] == paths
    assert [int(item.pages[0][0, 0, 0]) for item in loaded] == list(range(9))
    assert all(item.error is None for item in loaded)


def test_slow_first_file_does_not_reorder_results(tmp_path, monkeypatch):
    paths = [_write(tmp_path, f"img{i}.png", i) for i in range(6)]
    real = loader_module.decode_image
    first_done = threading.Event()

    def slow_first(path):
        if path == paths[0]:
            first_done.wait(2)
        return real(path)

    monkeypatch.setattr(loader_module, "decode_image", slow_first)
    results = []
    for item in iter_decoded_sources(paths, thread_count=4):
        results.append(os.path.basename(item.path))
        first_done.set()
    assert results == [os.path.basename(p) for p in paths]


def test_multipage_tiff_keeps_page_identity_and_count(tmp_path):
    folder = str(tmp_path)
    pages = [np.full((8, 8, 3), value, np.uint8) for value in (10, 20, 30)]
    tiff = os.path.join(folder, "zstack.tif")
    assert cv2.imwritemulti(tiff, pages)
    _write(tmp_path, "single.png", 5)

    ok, message, images, names = ImageStackLoader().load_from_folder(
        folder, thread_count=4)
    assert ok, message
    assert names == ["single.png", "zstack_page_001.tif",
                     "zstack_page_002.tif", "zstack_page_003.tif"]
    assert [int(image[0, 0, 0]) for image in images] == [5, 10, 20, 30]
    assert ImageStackLoader().failed_count == 0


def test_repeated_container_paths_are_not_squared(tmp_path):
    tiff = os.path.join(str(tmp_path), "pages.tif")
    assert cv2.imwritemulti(tiff, [np.full((8, 8, 3), v, np.uint8)
                                   for v in (1, 2)])
    ok, _message, images, names = ImageStackLoader().load_from_filepaths(
        [tiff] * 3, thread_count=4)
    assert ok
    assert len(images) == 2 and len(names) == 2


def test_failed_file_is_counted_without_shifting_later_frames(tmp_path):
    paths = [_write(tmp_path, "a.png", 1), _write(tmp_path, "b.png", 2)]
    broken = os.path.join(str(tmp_path), "broken.png")
    with open(broken, "wb") as handle:
        handle.write(b"not a png")
    loader = ImageStackLoader()
    ok, message, images, names = loader.load_from_filepaths(
        [paths[0], broken, paths[1]], thread_count=4)
    assert ok and loader.failed_count == 1
    assert names == ["a.png", "b.png"]
    assert [int(image[0, 0, 0]) for image in images] == [1, 2]


def test_16bit_values_survive_parallel_decode(tmp_path):
    path = _write(tmp_path, "deep.png", 40000, dtype=np.uint16)
    ok, _message, images, _names = ImageStackLoader().load_from_filepaths(
        [path], thread_count=4)
    assert ok and images[0].dtype == np.uint16
    assert int(images[0][0, 0, 0]) == 40000


def test_scale_factor_is_applied_once(tmp_path):
    path = _write(tmp_path, "big.png", 3, size=(40, 60))
    ok, _message, images, _names = ImageStackLoader().load_from_filepaths(
        [path], scale_factor=0.5, thread_count=4)
    assert ok and images[0].shape[:2] == (20, 30)


def test_file_load_progress_counts_failed_and_missing_sources(tmp_path):
    paths = [_write(tmp_path, 'a.png', 1), _write(tmp_path, 'b.png', 2)]
    broken = tmp_path / 'broken.png'
    broken.write_bytes(b'not a png')
    missing = str(tmp_path / 'missing.png')
    loader = ImageStackLoader()
    progress = []
    loader.progress_callback = lambda done, total: progress.append((done, total))

    ok, _, images, names = loader.load_from_filepaths(
        [paths[0], str(broken), missing, paths[1]], thread_count=1)

    assert ok and loader.failed_count == 2
    assert names == ['a.png', 'b.png']
    assert [int(image[0, 0, 0]) for image in images] == [1, 2]
    assert progress and progress[-1] == (4, 4)
    assert all(total == 4 and 0 <= done <= total for done, total in progress)
    assert all(left[0] <= right[0] for left, right in zip(progress, progress[1:]))


def test_folder_load_progress_stops_when_cancelled(tmp_path):
    for index in range(3):
        _write(tmp_path, f'{index}.png', index)
    loader = ImageStackLoader()
    progress = []
    loader.progress_callback = lambda done, total: progress.append((done, total))
    loader.should_cancel = lambda: len(progress) >= 2

    ok, message, images, _ = loader.load_from_folder(str(tmp_path), thread_count=1)

    assert not ok and message == 'CANCELLED' and images == []
    assert progress and progress[-1][0] < 3
    assert all(total == 3 and 0 <= done < total for done, total in progress)
    assert all(left[0] <= right[0] for left, right in zip(progress, progress[1:]))


def test_cancel_before_start_never_decodes(tmp_path, monkeypatch):
    paths = [_write(tmp_path, f"img{i}.png", i) for i in range(4)]
    calls = []
    real = loader_module.decode_image

    def observed(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(loader_module, "decode_image", observed)
    with pytest.raises(_DecodeCancelled):
        list(iter_decoded_sources(paths, thread_count=4,
                                  should_cancel=lambda: True))
    assert calls == []


def test_cancel_midway_stops_and_joins_workers(tmp_path, monkeypatch):
    paths = [_write(tmp_path, f"img{i}.png", i) for i in range(8)]
    real = loader_module.decode_image
    seen = []
    cancel = threading.Event()

    def observed(path):
        result = real(path)
        seen.append(path)
        if len(seen) >= 2:
            cancel.set()
        return result

    monkeypatch.setattr(loader_module, "decode_image", observed)
    with pytest.raises(_DecodeCancelled):
        for _item in iter_decoded_sources(paths, thread_count=2,
                                          should_cancel=cancel.is_set):
            pass
    # Every worker call returned before the exception propagated.
    assert threading.active_count() >= 1


def test_gui_loader_reports_cancelled_without_partial_stack(tmp_path):
    for i in range(5):
        _write(tmp_path, f"img{i}.png", i)
    loader = ImageStackLoader()
    loader.images = [np.zeros((2, 2, 3), np.uint8)]
    loader.image_paths = ["previous.png"]
    state = {"calls": 0}

    def should_cancel():
        state["calls"] += 1
        return state["calls"] > 2

    loader.should_cancel = should_cancel
    ok, message, images, names = loader.load_from_folder(
        str(tmp_path), thread_count=4)
    assert not ok and message == "CANCELLED"
    assert images == [] and names == []
    # The previously loaded stack is still intact.
    assert loader.image_paths == ["previous.png"]


def test_worker_count_never_exceeds_the_window(monkeypatch, tmp_path):
    paths = [_write(tmp_path, f"img{i}.png", i) for i in range(12)]
    real = loader_module.decode_image
    lock = threading.Lock()
    active = peak = 0

    def observed(path):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return real(path)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(loader_module, "decode_image", observed)
    list(iter_decoded_sources(paths, thread_count=2))
    assert 1 <= peak <= 2


def test_unknown_ram_does_not_assume_a_budget(monkeypatch, tmp_path):
    from core import resource_policy

    paths = [_write(tmp_path, f"img{i}.png", i) for i in range(4)]

    def no_ram():
        raise resource_policy.ResourceLimitError("no budget")

    monkeypatch.setattr(resource_policy, "plan_execution",
                        lambda *a, **k: no_ram())
    with pytest.raises(resource_policy.ResourceLimitError):
        list(iter_decoded_sources(paths, thread_count=4))


@pytest.mark.parametrize('outcome', ['missing', 'empty', 'cancelled', 'error', 'success'])
def test_video_load_commits_pixels_and_provenance_together(tmp_path, monkeypatch, outcome):
    previous_path = _write(tmp_path, 'previous.png', 41)
    loader = ImageStackLoader()
    assert loader.load_from_filepaths([previous_path])[0]
    loader.source_exif = b'previous EXIF'
    loader.source_icc_profile = b'previous ICC'
    loader.source_icc_profiles = (b'previous ICC',)
    loader.px_size_um = 2.5
    loader.linear_dng_metadata = object()
    loader.failed_count = 3
    fields = ('images', 'image_paths', 'source_exif', 'source_icc_profile',
              'source_icc_profiles', 'px_size_um', 'linear_dng_metadata', 'failed_count')
    previous = {name: getattr(loader, name) for name in fields}
    cancelled = False
    loader.should_cancel = lambda: cancelled
    path = tmp_path / 'clip.mp4'
    pixels = np.full((8, 10, 3), 173, np.uint8)
    released = []

    class Capture:
        def __init__(self, _path):
            self.reads = 0
        def isOpened(self):
            return True
        def get(self, _prop):
            return 1
        def read(self):
            nonlocal cancelled
            if outcome == 'error':
                raise OSError('video decode failed')
            if outcome == 'empty':
                return False, None
            self.reads += 1
            if self.reads == 1:
                return True, pixels
            cancelled = outcome == 'cancelled'
            return False, None
        def release(self):
            released.append(True)

    if outcome != 'missing':
        path.touch()
        monkeypatch.setattr(cv2, 'VideoCapture', Capture)
    if outcome == 'error':
        with pytest.raises(OSError, match='video decode failed'):
            loader.load_from_video(str(path))
    else:
        ok, message, images, names = loader.load_from_video(str(path))
        assert ok == (outcome == 'success')
        if outcome == 'success':
            assert len(images) == len(names) == 1
            np.testing.assert_array_equal(images[0], pixels)
            assert loader.images is images and loader.image_paths == [str(path)]
            assert loader.source_exif == loader.source_icc_profile == b''
            assert loader.source_icc_profiles == (b'',) and loader.px_size_um is None
            assert loader.linear_dng_metadata is None and loader.failed_count == 0
        else:
            assert images == names == []
            if outcome == 'cancelled':
                assert message == 'CANCELLED'
    if outcome != 'success':
        assert all(getattr(loader, name) is value for name, value in previous.items())
    assert bool(released) == (outcome != 'missing')


@pytest.mark.parametrize('shape', [(1, 1000), (1000, 1)])
@pytest.mark.parametrize('extension', ['.png', '.tif'])
def test_scaled_thin_sources_keep_a_nonempty_short_edge(tmp_path, shape, extension):
    path = _write(tmp_path, 'thin' + extension, 40000, size=shape, dtype=np.uint16)
    decoded = list(iter_decoded_sources([path], scale_factor=.5, thread_count=1))
    assert len(decoded) == 1 and decoded[0].error is None
    expected_size = (max(1, int(shape[1] * .5)), max(1, int(shape[0] * .5)))
    expected = cv2.resize(loader_module.decode_image(path), expected_size, interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(decoded[0].pages[0], expected)


@pytest.mark.parametrize('shape', [(1, 1000), (1000, 1)])
def test_thin_sources_have_valid_thumbnails_and_pixmaps(qapp, shape):
    image = np.full(shape + (3,), 40000, np.uint16)
    loader = ImageStackLoader()
    thumbnail = loader.create_thumbnails([image], thumb_size=40)[0]
    assert not thumbnail.isNull()
    assert max(thumbnail.width(), thumbnail.height()) == 40
    assert min(thumbnail.width(), thumbnail.height()) == 1
    pixmap = loader.create_pixmaps([image], max_size=(100, 100))[0]
    assert not pixmap.isNull()
    assert max(pixmap.width(), pixmap.height()) == 100
    assert min(pixmap.width(), pixmap.height()) == 1
