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


@pytest.mark.parametrize('cancel_waiter', [False, True])
def test_render_reports_waiting_until_lease_and_never_prepares_after_cancel(monkeypatch, cancel_waiter):
    from core.resource_policy import heavy_job

    worker = _render(_frames())
    waiting, admitted, finished = Event(), Event(), Event()
    stages = []

    def stage(value):
        stages.append(value)
        if value == 'waiting':
            waiting.set()

    worker.stage_signal.connect(stage, Qt.ConnectionType.DirectConnection)
    worker.finished.connect(finished.set, Qt.ConnectionType.DirectConnection)
    monkeypatch.setattr(worker, '_run_admitted', admitted.set)
    try:
        with heavy_job():
            worker.start()
            assert waiting.wait(5)
            assert stages == ['waiting'] and not admitted.is_set()
            if cancel_waiter:
                worker.cancel()
                assert finished.wait(5)
        assert finished.wait(5)
        assert worker.wait(5000)
    finally:
        worker.cancel()
        worker.wait(5000)
    assert stages == (['waiting'] if cancel_waiter else ['waiting', 'prepare'])
    assert admitted.is_set() is (not cancel_waiter)


@pytest.mark.parametrize("cancel_waiter", [False, True])
def test_stack_load_waits_for_active_render_and_can_cancel(tmp_path, cancel_waiter):
    from core.image_loader import ImageStackLoader
    from core.resource_policy import heavy_job
    from core.workers import StackLoadWorker
    from utils.image_utils import imwrite_auto

    source = tmp_path / "frame.png"
    assert imwrite_auto(str(source), _frames()[0])
    worker = StackLoadWorker(ImageStackLoader(), filepaths=[str(source)], thread_count=1)
    started, finished = Event(), Event()
    results = []
    worker.started.connect(started.set, Qt.ConnectionType.DirectConnection)

    def record(*args):
        results.append(args)
        finished.set()

    worker.finished_load.connect(record, Qt.ConnectionType.DirectConnection)
    try:
        with heavy_job():
            worker.start()
            assert started.wait(5)
            assert not finished.wait(0.25), "stack decoding overlapped an active heavy job"
            if cancel_waiter:
                worker.request_cancel()
                assert finished.wait(2), "cancelled load did not stop while waiting"
        assert finished.wait(10)
        assert worker.wait(10000)
    finally:
        worker.request_cancel()
        worker.wait(10000)
    if cancel_waiter:
        assert results == [(False, 'CANCELLED', [], [])]
    else:
        assert results[0][0]
        np.testing.assert_array_equal(results[0][2][0], _frames()[0])


@pytest.mark.parametrize("mode", ['stack', 'folders'])
@pytest.mark.parametrize("cancel_waiter", [False, True])
def test_batch_import_waits_for_active_render_and_can_cancel(tmp_path, mode, cancel_waiter):
    from core.resource_policy import heavy_job
    from core.workers import BatchImportWorker
    from utils.image_utils import imwrite_auto

    assert imwrite_auto(str(tmp_path / 'frame.png'), _frames()[0])
    worker = BatchImportWorker(mode, folder_path=str(tmp_path),
                               folder_paths=[str(tmp_path)], thread_count=1)
    started, finished = Event(), Event()
    results = []
    worker.started.connect(started.set, Qt.ConnectionType.DirectConnection)
    worker.finished.connect(finished.set, Qt.ConnectionType.DirectConnection)
    signal = worker.stack_loaded if mode == 'stack' else worker.folders_loaded
    signal.connect(lambda *args: results.append(args), Qt.ConnectionType.DirectConnection)
    try:
        with heavy_job():
            worker.start()
            assert started.wait(5)
            assert not finished.wait(0.25), "batch import overlapped an active heavy job"
            if cancel_waiter:
                worker.cancel()
                assert finished.wait(2), "cancelled batch import did not stop while waiting"
        assert finished.wait(10)
        assert worker.wait(10000)
    finally:
        worker.cancel()
        worker.wait(10000)
    assert bool(results) is not cancel_waiter




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
    original = align_cache.save_alignment_plan
    writers = []

    def save(*args, **kwargs):
        writers.append(get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(align_cache, "save_alignment_plan", save)
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


@pytest.mark.parametrize('alignment', ['openfocus', 'ecc'])
def test_cached_alignment_remains_resolvable_after_stack_revision(tmp_path, alignment):
    from core.source_frames import resolve_frame

    frames = _frames(np.uint16)
    frames[1] = np.roll(frames[0], 4, axis=1)
    current = [True]
    request = {**_request(tmp_path, frames), 'is_current': lambda: current[0]}

    def render():
        parameters = {'max_shift_ratio': .2, 'magnification': 0} if alignment == 'openfocus' else {}
        worker = RenderWorker(frames, algorithm='openfocus_a', alignment=alignment,
            registration_parameters=parameters,
            fusion_parameters={'kernel_size': 17}, need_fusion=False,
            thread_count=1, use_gpu=False, exposure_match=False, disk_cache=request)
        results, errors = _capture(worker)
        worker.run()
        assert not errors and len(results) == 1
        return worker, results[0][0]

    _, first_frames = render()
    second, warm_frames = render()
    assert second.alignment_reused
    expected = resolve_frame(first_frames[1]).copy()
    current[0] = False
    np.testing.assert_array_equal(resolve_frame(warm_frames[1]), expected)


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

    monkeypatch.setattr(align_cache, "save_alignment_plan", fail)
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


@pytest.mark.parametrize('cached', ['memory', 'disk'])
def test_requested_report_runs_registration_in_worker_even_with_cached_frames(monkeypatch, cached):
    from core.registration import alignment_report

    frames = _frames(np.uint16)
    request = {'mode': 'load'} if cached == 'disk' else None
    worker = RenderWorker(frames, algorithm=None, need_fusion=False,
                          alignment='openfocus', use_gpu=False,
                          disk_cache=request, collect_alignment_report=True)
    if cached == 'memory':
        worker.aligned_images = [np.zeros_like(frame) for frame in frames]
        worker.is_images_aligned = True
        worker.last_alignment_options = (worker.alignment, worker.registration_parameters)
    observed = []

    def register(images, should_cancel=None):
        observed.append((get_ident(), alignment_report.get(), images[0].dtype))
        alignment_report.get().append({'frame_index': 1, 'status': 'aligned',
                                       'method': 'openfocus'})
        return images, 0.1

    def unexpected_cache_load():
        raise AssertionError('reporting must not load cached alignment')

    monkeypatch.setattr(worker, '_run_registration', register)
    monkeypatch.setattr(worker, '_load_cached_alignment', unexpected_cache_load)
    results, errors = _capture(worker)
    parent_report = []
    token = alignment_report.set(parent_report)
    try:
        worker.start()
        assert worker.wait(10000)
        assert alignment_report.get() is parent_report
    finally:
        alignment_report.reset(token)
    assert not errors and len(results) == 1
    assert len(observed) == 1
    assert observed[0][1] is worker.alignment_report
    assert observed[0][2] == np.dtype('uint16')
    assert observed[0][0] != get_ident()
    assert parent_report == []
    assert worker.alignment_report == [{'frame_index': 1, 'status': 'aligned',
                                        'method': 'openfocus'}]
    assert worker.alignment_report_provenance == 'fresh'
    assert worker.alignment_computed and not worker.alignment_reused
    np.testing.assert_array_equal(results[0][0][0], frames[0])


def test_report_context_restored_after_registration_error(monkeypatch):
    from core.registration import alignment_report

    worker = RenderWorker(_frames(), algorithm=None, need_fusion=False,
                          alignment='openfocus', use_gpu=False,
                          collect_alignment_report=True)

    def fail(images, should_cancel=None):
        assert alignment_report.get() is worker.alignment_report
        raise ValueError('registration failed')

    monkeypatch.setattr(worker, '_run_registration', fail)
    results, errors = _capture(worker)
    outer = []
    token = alignment_report.set(outer)
    try:
        worker.run()
        assert alignment_report.get() is outer
    finally:
        alignment_report.reset(token)
    assert results == [] and errors == ['registration failed']
    assert worker.alignment_report_provenance is None




@pytest.mark.parametrize('shape', [(1, 1000), (1000, 1)])
def test_quick_preview_retains_thin_frame_pixels(shape):
    pixels = np.full(shape + (3,), 40000, np.uint16)
    worker = RenderWorker([pixels], algorithm=None, need_fusion=False, preview=True)
    results, errors = _capture(worker)
    worker.run()
    assert not errors and len(results) == 1
    actual = results[0][0][0]
    expected_size = (max(1, round(shape[1] * .5)), max(1, round(shape[0] * .5)))
    expected = cv2.resize(pixels, expected_size, interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('calibrations', [(0.5, 2.0), (0.5, None)])
def test_split_batch_exports_each_stacks_camera_and_calibration(tmp_path, calibrations):
    import tifffile
    from PIL import Image
    from core.image_loader import ImageStackLoader
    from core.workers import BatchWorker
    from utils.scalebar import burn

    source = tmp_path / 'split'
    source.mkdir()
    pixels = np.full((96, 128, 3), 90, np.uint8)
    for index, px_um in enumerate(calibrations):
        desc = '' if px_um is None else f'<OME><Pixels PhysicalSizeX="{px_um}"/></OME>'
        tifffile.imwrite(source / f'{index}.tif', pixels, photometric='rgb',
                         description=desc, metadata=None,
                         extratags=[(271, 's', 0, 'Camera maker', False),
                                    (272, 's', 0, f'Model {index}', False)])
    loader = ImageStackLoader()
    ok, message, entries, _ = loader.load_images_with_timestamps(str(source))
    assert ok, message
    style = {'position': 'bottom-right', 'color': 'white',
             'font_family': None, 'font_size': 0.0, 'font_bold': False,
             'text_align': 'auto', 'custom_x': None, 'custom_y': None}
    settings = {'format': 'tiff', 'fusion_method': 'none',
                'scale_bar': {'target': 'registered', **style}}
    worker = BatchWorker([], 'custom', str(tmp_path / 'out'), settings,
                         import_mode='single_folder', split_method='count', split_param=1,
                         single_folder_images_with_times=entries, thread_count=1)
    results, errors = {}, []
    worker.finished.connect(results.update)
    worker.error.connect(errors.append)
    worker.run()
    assert not errors and results['counts']['success'] == 2
    for index, (stack, px_um) in enumerate(zip(results['aligned_stacks'], calibrations)):
        assert len(stack['files']) == 1
        path = stack['files'][0]
        with Image.open(path) as exported:
            assert exported.getexif()[272] == f'Model {index}'
        actual = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        expected = burn(pixels, {'px_um': px_um, **style}) if px_um else pixels
        np.testing.assert_array_equal(actual, expected)
        assert stack['annotations_baked'] == (px_um is not None)


@pytest.mark.parametrize('calibrations', [(0.5, 0.5, 0.5), (0.5, 2.0, 0.5), (0.5, None, 0.5)])
def test_grouped_batch_only_draws_a_shared_physical_calibration(tmp_path, calibrations):
    import tifffile
    from core.workers import BatchWorker
    from utils.scalebar import burn

    source = tmp_path / 'grouped'
    source.mkdir()
    pixels = np.full((96, 128, 3), 90, np.uint8)
    for index, px_um in enumerate(calibrations):
        desc = '' if px_um is None else f'<OME><Pixels PhysicalSizeX="{px_um}"/></OME>'
        tifffile.imwrite(source / f'{index}.tif', pixels, photometric='rgb',
                         description=desc, metadata=None)
    style = {'position': 'bottom-right', 'color': 'white',
             'font_family': None, 'font_size': 0.0, 'font_bold': False,
             'text_align': 'auto', 'custom_x': None, 'custom_y': None}
    outputs = []
    for annotated in (False, True):
        settings = {'format': 'png', 'fusion_method': 'openfocus_a', 'group_size': 2,
                    'fusion_params': {'kernel_size': 3}}
        if annotated:
            settings['scale_bar'] = {'target': 'output', **style}
        worker = BatchWorker([str(source)], 'custom', str(tmp_path / str(annotated)),
                             settings, thread_count=1, use_gpu=False, tile_enabled=False)
        results, errors = {}, []
        worker.finished.connect(results.update)
        worker.error.connect(errors.append)
        worker.run()
        assert not errors and results['counts']['success'] == 1, results
        outputs.append(cv2.imdecode(np.fromfile(results['outputs'][0], np.uint8),
                                    cv2.IMREAD_UNCHANGED))
    reference, actual = outputs
    expected = (burn(reference, {'px_um': .5, **style})
                if all(value == .5 for value in calibrations) else reference)
    np.testing.assert_array_equal(actual, expected)
