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
