"""Behavioral coverage for asynchronous comparison registration and fusion."""

from threading import get_ident

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt

from core.compare_worker import AlignmentWorker, PreviewWorker
from core.multi_focus_fusion import MultiFocusFusion
from core.registration import ImageRegistration, RegistrationCancelled
from utils.image_utils import normalize_fuse_input, quantize_fuse_output


@pytest.fixture(autouse=True)
def _application(qapp):
    return qapp


def _stack(height=80, width=112, dtype=np.uint8):
    rng = np.random.default_rng(319)
    sharp = rng.integers(15, 240, (height, width, 3), dtype=np.uint8)
    blur = cv2.GaussianBlur(sharp, (9, 9), 2)
    left, right = sharp.copy(), sharp.copy()
    left[:, width // 2:] = blur[:, width // 2:]
    right[:, :width // 2] = blur[:, :width // 2]
    if dtype == np.uint16:
        # Lower bits carry image information; a uint8 round trip loses it.
        low = rng.integers(0, 257, sharp.shape, dtype=np.uint16)
        return [image.astype(np.uint16) * 256 + low for image in (left, right)]
    return [left, right]


def _run(worker):
    results, errors, progress, finished = [], [], [], []
    direct = Qt.ConnectionType.DirectConnection
    if isinstance(worker, PreviewWorker):
        worker.result_ready.connect(lambda *args: results.append(args), direct)
        worker.method_failed.connect(lambda *args: errors.append(args), direct)
        worker.progress.connect(lambda *args: progress.append(args), direct)
    else:
        worker.ready.connect(lambda frames: results.append(frames), direct)
        worker.failed.connect(errors.append, direct)
    worker.finished.connect(lambda: finished.append(True), direct)
    worker.start()
    try:
        assert worker.wait(30000), "comparison worker did not finish"
    finally:
        if worker.isRunning():
            worker.cancel()
            worker.wait()
    assert finished == [True]
    return results, errors, progress


def _expected(frames, method, **params):
    inputs, is_16bit = normalize_fuse_input(frames)
    params.pop('block_size', None)
    model_path = params.pop('model_path', None)
    result = MultiFocusFusion(algorithm=method, tile_enabled=False,
        exposure_match=False, fusion_parameters=params).fuse(
            inputs, thread_count=1, model_path=model_path)
    return quantize_fuse_output(result, is_16bit)


@pytest.mark.parametrize("method", ["guided_filter", "dct", "gfgfgf",
                                   "openfocus_a", "openfocus_b", "openfocus_c"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_full_render_matches_real_fusion_and_keeps_dtype(method, dtype):
    frames = _stack(dtype=dtype)
    params = {} if method == "openfocus_c" else {"kernel_size": 5}
    worker = PreviewWorker(frames, [method], {method: params},
                           {"tile_enabled": False, "thread_count": 0},
                           (10, 11, 9, 9), roi_only=False)
    results, errors, progress = _run(worker)
    assert errors == []
    assert len(results) == 1
    name, result, bounds = results[0]
    assert name == method
    assert bounds == (0, 0, 112, 80)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, _expected(frames, method, **params))
    assert progress == [(0, 1, method), (1, 1, "")]
    if dtype == np.uint16:
        assert np.any(result % 257), "16-bit details were quantized to 8 bits"


@pytest.mark.parametrize("height,width,region", [
    (281, 703, (0, 0, 31, 31)),
    (281, 703, (672, 0, 31, 31)),
    (281, 703, (0, 250, 31, 31)),
    (281, 703, (672, 250, 31, 31)),
    (703, 281, (250, 672, 31, 31)),
    (23, 701, (676, 0, 23, 23)),
    (701, 23, (0, 676, 23, 23)),
    (285, 707, (301, 123, 39, 39)),
])
def test_dct_roi_preserves_entire_selection_and_global_block_grid(height, width, region):
    frames = _stack(height, width)
    worker = PreviewWorker(frames, ["dct"], {"dct": {"kernel_size": 5}},
                           {"tile_enabled": False, "thread_count": 1}, region)
    results, errors, _ = _run(worker)
    assert errors == []
    x, y, w, h = region
    assert results[0][2] == region
    assert results[0][1].shape == (h, w, 3)
    full = _expected(frames, "dct", kernel_size=5, block_size=8)
    np.testing.assert_array_equal(results[0][1], full[y:y + h, x:x + w])


@pytest.mark.parametrize("height,width,region", [
    (1, 311, (310, 0, 1, 1)), (311, 1, (0, 310, 1, 1)),
    (5, 311, (306, 0, 5, 5)), (311, 5, (0, 306, 5, 5)),
])
def test_dct_handles_frames_thinner_than_one_transform_block(height, width, region):
    frames = _stack(height, width, np.uint16)
    results, errors, _ = _run(PreviewWorker(
        frames, ["dct"], {"dct": {"kernel_size": 3}},
        {"tile_enabled": False}, region))
    assert errors == []
    assert results[0][2] == region
    assert results[0][1].shape[:2] == (region[3], region[2])
    assert results[0][1].dtype == np.uint16
    assert np.any(results[0][1])


def test_roi_uses_context_views_and_does_not_copy_full_stack(monkeypatch):
    frames = _stack(641, 803)
    observations = []
    original = MultiFocusFusion.fuse

    def observe(self, input_source, *args, **kwargs):
        observations.append(input_source)
        return original(self, input_source, *args, **kwargs)

    monkeypatch.setattr(MultiFocusFusion, "fuse", observe)
    region = (301, 257, 41, 41)
    results, errors, _ = _run(PreviewWorker(
        frames, ["openfocus_a"], {}, {"tile_enabled": False}, region))
    assert errors == []
    assert len(observations) == 1
    for source, crop in zip(frames, observations[0]):
        assert np.shares_memory(source, crop)
        assert crop.shape[:2] == (41 + 192, 41 + 192)
    assert results[0][1].shape == (41, 41, 3)
    assert results[0][1].base is None  # Do not retain the context output buffer.


def test_own_method_settings_and_configuration_snapshot():
    frames = _stack()
    methods = ["openfocus_a", "openfocus_b", "openfocus_c"]
    own = {
        "openfocus_a": {"kernel_size": 3, "blend_smoothing": 0.0},
        "openfocus_b": {"kernel_size": 5, "depth_smoothing": 0.0},
        'openfocus_c': {'blend_smoothing': 0.0},
    }
    shared = {"kernel_size": 31, "blend_smoothing": 8.0,
              "depth_smoothing": 8.0, "tile_enabled": False}
    worker = PreviewWorker(frames, methods, own, shared, None, roi_only=False)
    own["openfocus_a"]["blend_smoothing"] = 8.0
    shared["kernel_size"] = 41
    methods.clear()
    results, errors, _ = _run(worker)
    assert errors == []
    assert [row[0] for row in results] == ["openfocus_a", "openfocus_b", "openfocus_c"]
    for (method, result, _), settings in zip(results, [
        {"kernel_size": 3, "blend_smoothing": 0.0},
        {"kernel_size": 5, "depth_smoothing": 0.0},
        {"blend_smoothing": 0.0},
    ]):
        np.testing.assert_array_equal(result, _expected(frames, method, **settings))


def test_shared_brightness_settings_are_separate_from_algorithm_parameters():
    frames = _stack()
    frames[0] //= 2
    own = {'openfocus_a': {'kernel_size': 5}}
    results, errors, _ = _run(PreviewWorker(
        frames, ['openfocus_a'], own,
        {'exposure_match': False, 'base_frame': 'none', 'tile_enabled': False},
        None, roi_only=False))
    assert errors == []
    np.testing.assert_array_equal(results[0][1],
        _expected(frames, 'openfocus_a', kernel_size=5))


def test_dtcwt_uses_own_decomposition_depth():
    pytest.importorskip("dtcwt")
    frames = _stack(65, 99, np.uint16)
    results, errors, _ = _run(PreviewWorker(
        frames, ["dtcwt"], {"dtcwt": {"N": 2}},
        {"N": 4, "tile_enabled": False}, None, roi_only=False))
    assert errors == []
    np.testing.assert_array_equal(results[0][1], _expected(frames, "dtcwt", N=2))


def test_error_is_reported_per_method_and_later_methods_continue():
    results, errors, progress = _run(PreviewWorker(
        _stack(), ["unknown", "openfocus_a"], {},
        {"tile_enabled": False}, None, roi_only=False))
    assert errors[0][0] == "unknown"
    assert 'unknown' in errors[0][1]
    assert [row[0] for row in results] == ["openfocus_a"]
    assert progress[-1] == (2, 2, "")


def test_invalid_region_reports_error_instead_of_partial_result():
    results, errors, _ = _run(PreviewWorker(
        _stack(), ["openfocus_a"], {}, {}, (100, 70, 20, 20)))
    assert results == []
    assert len(errors) == 1
    assert "outside" in errors[0][1]


@pytest.mark.parametrize("kind", ["alignment", "preview"])
def test_cancel_before_start_still_finishes_without_publishing(kind):
    frames = _stack()
    worker = (AlignmentWorker(frames, "openfocus", {}) if kind == "alignment"
              else PreviewWorker(frames, ["openfocus_a"], {}, {}, None, False))
    worker.cancel()
    results, errors, progress = _run(worker)
    assert results == errors == progress == []


def test_cancellation_during_real_tiled_fusion_stops_emission(monkeypatch):
    worker = PreviewWorker(
        _stack(192, 256), ["openfocus_a", "openfocus_b"], {},
        {"tile_enabled": True, "tile_threshold": 64, "tile_block_size": 96,
         "tile_overlap": 16, "thread_count": 1}, None, roi_only=False)
    original = MultiFocusFusion.fuse
    completed_tiles = []

    def cancellable(self, *args, **kwargs):
        def on_progress(done, total):
            completed_tiles.append((done, total))
            worker.cancel()
        kwargs["progress_callback"] = on_progress
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MultiFocusFusion, "fuse", cancellable)
    results, errors, progress = _run(worker)
    assert completed_tiles and completed_tiles[0][1] > 1
    assert results == errors == []
    assert progress == [(0, 2, "openfocus_a")]


def test_cancellation_after_native_fusion_discards_result(monkeypatch):
    worker = PreviewWorker(_stack(), ["openfocus_a"], {},
                            {"tile_enabled": False}, None, roi_only=False)
    original = MultiFocusFusion.fuse
    completed = []

    def cancel_on_return(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        completed.append(result.shape)
        worker.cancel()
        return result

    monkeypatch.setattr(MultiFocusFusion, "fuse", cancel_on_return)
    results, errors, _ = _run(worker)
    assert completed == [(80, 112, 3)]
    assert results == errors == []


def test_unrequested_cancellation_exception_is_not_silently_swallowed(monkeypatch):
    def failed_fusion(self, *args, **kwargs):
        raise RegistrationCancelled()

    monkeypatch.setattr(MultiFocusFusion, "fuse", failed_fusion)
    results, errors, _ = _run(PreviewWorker(
        _stack(), ["openfocus_a"], {}, {}, None, roi_only=False))
    assert results == []
    assert errors == [("openfocus_a", "RegistrationCancelled")]


def test_alignment_runs_real_registration_on_background_thread(monkeypatch):
    frames = _stack(96, 128, np.uint16)
    frames[1] = np.roll(frames[0], (2, -3), axis=(0, 1))
    shared = {'reg_downscale_width': 96, 'registration_parameters': {'max_shift_ratio': .1},
              'thread_count': 0, 'exposure_match': False}
    worker = AlignmentWorker(frames, "openfocus", shared)
    shared["reg_downscale_width"] = 8
    original = ImageRegistration.process
    observations = []
    main_thread = get_ident()

    def observe(self, input_source, *args, **kwargs):
        observations.append((get_ident(), self.downscale_width,
                             self.parameters['max_shift_ratio'], self.brightness_reference,
                             kwargs["thread_count"], callable(kwargs["should_cancel"])))
        return original(self, input_source, *args, **kwargs)

    monkeypatch.setattr(ImageRegistration, "process", observe)
    results, errors, _ = _run(worker)
    assert errors == []
    assert len(results) == 1
    aligned = results[0]
    assert all(frame.dtype == np.uint16 for frame in aligned)
    assert observations[0][0] != main_thread
    assert observations[0][1:] == (96, 0.1, "first", 0, True)
    before = np.mean(np.abs(frames[0].astype(float) - frames[1]))
    after = np.mean(np.abs(aligned[0].astype(float) - aligned[1]))
    assert after < before * 0.5


def test_alignment_none_keeps_frame_storage():
    frames = _stack(dtype=np.uint16)
    results, errors, _ = _run(AlignmentWorker(frames, "none", {}))
    assert errors == []
    assert isinstance(results[0], list)
    assert all(left is right for left, right in zip(results[0], frames))


def test_alignment_failure_is_reported():
    results, errors, _ = _run(AlignmentWorker(_stack(), "unknown", {}))
    assert results == []
    assert len(errors) == 1 and "unknown" in errors[0]


def test_cancellation_is_polled_inside_real_registration(monkeypatch):
    worker = AlignmentWorker(_stack(), "openfocus", {"thread_count": 1})
    original = ImageRegistration.process
    polls = []

    def cancellable(self, *args, **kwargs):
        should_cancel = kwargs["should_cancel"]
        def cancel_during_registration():
            polls.append(True)
            worker.cancel()
            return should_cancel()
        kwargs["should_cancel"] = cancel_during_registration
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ImageRegistration, "process", cancellable)
    results, errors, _ = _run(worker)
    assert polls
    assert results == errors == []


def test_neural_method_uses_own_mode_and_preserves_uint16():
    pytest.importorskip("torch")
    from pathlib import Path
    from utils import resource_path

    model_path = resource_path("weights", "stackmff_v3_final.pth")
    if not Path(model_path).is_file():
        pytest.skip("StackMFF-V3 weights are not installed")
    frames = _stack(32, 48, np.uint16)
    results, errors, _ = _run(PreviewWorker(
        frames, ["stackmffv3"], {"stackmffv3": {"mode": "hard"}},
        {'tile_enabled': False, 'thread_count': 1},
        None, roi_only=False))
    assert errors == []
    assert results[0][1].dtype == np.uint16
    np.testing.assert_array_equal(
        results[0][1], _expected(frames, "stackmffv3", mode="hard", model_path=model_path))
    # Hard selection retains an actual source sample at every channel/pixel.
    assert np.all((results[0][1] == frames[0]) | (results[0][1] == frames[1]))


def _global_average_fusion(fusion, input_source, img_resize=None, offset=0, source='first', **execution):
    frame = input_source[0 if source == 'first' else -1]
    value = np.clip(round(float(frame.mean())) + offset, 0, 255)
    return np.full_like(frame, value)


def test_missing_roi_capability_fuses_full_frame_before_crop(monkeypatch):
    from core import algorithms
    monkeypatch.setattr(algorithms, '_FUSION', dict(algorithms._FUSION))
    algorithms.register_fusion(algorithms.FusionAlgorithm(
        'global_average', 'radio_dct', 'Global Average', 'Average',
        __name__ + ':_global_average_fusion', (
            algorithms.ParameterSpec('offset', 'compare_levels', 0, 0, 30),
            algorithms.ParameterSpec('source', 'label_ai_mode', 'first', choices=('first', 'last')),
        ), default_exposure_match=False))
    gradient = np.broadcast_to(np.arange(240, dtype=np.uint8), (80, 240)).copy()
    frames = [gradient, gradient // 2]
    region = (0, 0, 20, 20)
    outputs = []
    for parameters in ({'offset': 0, 'source': 'first'},
                       {'offset': 7, 'source': 'first'},
                       {'offset': 7, 'source': 'last'}):
        results, errors, _ = _run(PreviewWorker(
            frames, ['global_average'], {'global_average': parameters},
            {'tile_enabled': False, 'exposure_match': False}, region))
        assert errors == []
        image = results[0][1]
        assert image.shape == (20, 20)
        full = _expected(frames, 'global_average', **parameters)
        np.testing.assert_array_equal(image, full[:20, :20])
        outputs.append(image)
    assert not np.array_equal(outputs[0], outputs[1])
    assert not np.array_equal(outputs[1], outputs[2])
