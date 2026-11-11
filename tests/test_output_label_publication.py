"""Regressions for label scope and atomic automatic export publication."""
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from controllers.label_manager import LabelManager
from utils.export_paths import publish_unique
from utils.label_model import Label
from utils.validators import LabelConfig


def _publish_concurrently(path, barrier, queue, data):
    def write(staging):
        Path(staging).write_bytes(data)
        barrier.wait(timeout=15)
    try:
        queue.put((publish_unique(path, write), data))
    except BaseException as exc:
        queue.put((None, repr(exc)))


def test_concurrent_publications_preserve_both_files(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    destination = str(tmp_path / 'result.png')
    processes = [ctx.Process(target=_publish_concurrently,
                             args=(destination, barrier, queue, value))
                 for value in (b'first', b'second')]
    for process in processes:
        process.start()
    results = [queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    assert len({path for path, _ in results}) == 2
    for path, expected in results:
        assert path is not None
        assert Path(path).read_bytes() == expected
    assert not list(tmp_path.glob('.openfocus-staging-*'))


def test_scoped_transparent_export_and_frozen_frame_order():
    window = SimpleNamespace(registration_frame_ids=['B', 'A'])
    manager = LabelManager(window)
    config = LabelConfig()
    config.target = 'registered'
    config.frame_ids = ['A']
    config.range = '3'
    manager.labels.add(Label(config))
    frozen = manager.snapshot(frame_ids_override={'registered': ['A', 'B']})
    window.registration_frame_ids = ['C']
    assert len(frozen.applicable_labels('registered', 0)) == 1
    assert frozen.applicable_labels('registered', 1) == []
    pixels = np.ones((3, 4, 4), dtype=np.uint16)
    manager._draw_errors['registered'] = 'old error'
    assert manager.prepare_bgr_image('registered', pixels, 0) is pixels
    assert manager._draw_errors['registered'] == ''
    with pytest.raises(ValueError):
        frozen.prepare_bgr_image('registered', pixels, 0)


def test_invalid_applicable_range_is_an_error():
    manager = LabelManager(SimpleNamespace())
    config = LabelConfig()
    config.target = 'output'
    config.range = 'bad'
    manager.labels.add(Label(config))
    with pytest.raises(ValueError, match='range'):
        manager.prepare_bgr_image('output', np.zeros((2, 2, 4), np.uint8), 0)
