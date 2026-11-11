"""A calibrated flat field must reach batch registration and project restore."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from utils.dust_map import apply_flat_field, compute_flat_field, load_flat_field_reference


def _reference(tmp_path):
    white = np.full((48, 64, 3), 48000, np.uint16)
    white[:, :24] = 30000
    path = tmp_path / 'white.png'
    assert cv2.imwrite(str(path), white)
    return path, white


def test_flat_field_keeps_grayscale_and_alpha(tmp_path):
    _, white = _reference(tmp_path)
    gain = compute_flat_field(white)
    gray = np.full((48, 64), 20000, np.uint16)
    rgba = np.dstack((np.repeat(gray[..., None], 3, axis=2), gray // 3))
    corrected_gray = apply_flat_field(gray, gain)
    corrected_rgba = apply_flat_field(rgba, gain)
    assert corrected_gray.shape == gray.shape and corrected_gray.dtype == gray.dtype
    assert corrected_rgba.dtype == rgba.dtype
    np.testing.assert_array_equal(corrected_rgba[..., 3], rgba[..., 3])
    assert corrected_gray[20, 10] > corrected_gray[20, 50]


def test_project_replays_correction_once_and_rejects_changed_reference(tmp_path):
    from tests.test_project_file import _make_stack, _window
    from utils import project_file
    white_path, white = _reference(tmp_path)
    source_dir = str(tmp_path / 'stack')
    paths = _make_stack(source_dir, n=2)
    # Use frames with the same sensor dimensions as the reference.
    original = np.full(white.shape, 18000, np.uint16)
    for path in paths:
        assert cv2.imwrite(path, original)
    _, digest = load_flat_field_reference(str(white_path))
    gain = compute_flat_field(white)
    window = _window(paths, source_dir)
    window.raw_images = [apply_flat_field(original, gain) for _ in paths]
    window.source_transforms = [{'operation': 'flat_field', 'path': str(white_path),
                                 'sha256': digest, 'sigma': 4.0, 'scale': 1.0}]
    project = str(tmp_path / 'project.ofproj')
    ok, message = project_file.save_project(window, project)
    assert ok, message
    ok, message, state = project_file.validate_project(project)
    assert ok, message
    loader = SimpleNamespace(image_paths=list(paths))
    host = SimpleNamespace(image_loader=loader)
    options = SimpleNamespace(filenames=[__import__('os').path.basename(p) for p in paths],
                              base_images=[original.copy() for _ in paths])
    project_file._prepare_project_load(host, state, options)
    np.testing.assert_array_equal(options.base_images[0], window.raw_images[0])
    assert cv2.imwrite(str(white_path), np.full_like(white, 40000))
    ok, error, _ = project_file.validate_project(project)
    assert not ok and 'changed' in error
    ok, error = project_file.save_project(window, project)
    assert not ok and 'changed' in error


def test_project_rejects_changed_scale(tmp_path):
    from utils.project_file import _validate_source_transforms
    path, _ = _reference(tmp_path)
    _, digest = load_flat_field_reference(str(path))
    with pytest.raises(ValueError, match='scale'):
        _validate_source_transforms([{'operation': 'flat_field', 'path': str(path),
                                      'sha256': digest, 'sigma': 4.0, 'scale': 1.0}], .5)


def test_batch_corrects_before_registration_and_reports_missing(qapp, tmp_path, monkeypatch):
    from core.registration import ImageRegistration
    from core.workers import BatchWorker
    white_path, white = _reference(tmp_path)
    source = tmp_path / 'stack'
    source.mkdir()
    original = np.full_like(white, 20000)
    for i in range(2):
        assert cv2.imwrite(str(source / f'{i}.png'), original)
    seen = []
    def register(self, images, **kwargs):
        seen.append(images[0].copy())
        return images
    monkeypatch.setattr(ImageRegistration, 'process', register)
    settings = {'format': 'png', 'fusion_method': 'none', 'alignment': 'ecc',
                'flat_field_path': str(white_path), 'runtime_settings': {}}
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'), settings,
                         use_gpu=False, thread_count=1)
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert outcome['success'] == 1
    np.testing.assert_array_equal(seen[0], apply_flat_field(original, compute_flat_field(white)))
    settings['flat_field_path'] = str(tmp_path / 'missing.png')
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'missing_out'), settings,
                         use_gpu=False, thread_count=1)
    failed = {}
    worker.finished.connect(failed.update)
    worker.run()
    assert failed['counts']['failed'] == 1
    assert 'missing.png' in str(failed)


def test_batch_dialog_includes_selected_reference(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog
    path, _ = _reference(tmp_path)
    dialog = BatchProcessingDialog()
    try:
        assert dialog.get_processing_settings()['flat_field_path'] == ''
        dialog.flat_field_path.setText(str(path))
        assert dialog.get_processing_settings()['flat_field_path'] == str(path)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_split_batch_uses_same_flat_field(qapp, tmp_path, monkeypatch):
    from core.registration import ImageRegistration
    from core.workers import BatchWorker
    path, white = _reference(tmp_path)
    source = tmp_path / 'stack'
    source.mkdir()
    original = np.full_like(white, 23000)
    names = [source / f'{i}.png' for i in range(2)]
    for name in names:
        assert cv2.imwrite(str(name), original)
    entries = [(str(name), original.copy(), None) for name in names]
    seen = []
    def register(self, images, **kwargs):
        seen.append(images[0].copy())
        return images
    monkeypatch.setattr(ImageRegistration, 'process', register)
    settings = {'format': 'png', 'fusion_method': 'none', 'alignment': 'ecc',
                'flat_field_path': str(path)}
    worker = BatchWorker([], 'custom', str(tmp_path / 'out'), settings,
                         import_mode='single_folder', split_method='count', split_param=2,
                         single_folder_images_with_times=entries, use_gpu=False)
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert outcome['success'] == 1
    np.testing.assert_array_equal(seen[0], apply_flat_field(original, compute_flat_field(white)))
    np.testing.assert_array_equal(entries[0][1], original)


def test_batch_cancels_between_corrected_frames(qapp, tmp_path, monkeypatch):
    import utils.dust_map as dust_map
    from core.workers import BatchWorker
    path, white = _reference(tmp_path)
    source = tmp_path / 'stack'
    source.mkdir()
    for i in range(2):
        assert cv2.imwrite(str(source / f'{i}.png'), white)
    settings = {'format': 'png', 'fusion_method': 'none', 'alignment': 'none',
                'flat_field_path': str(path)}
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'), settings,
                         use_gpu=False)
    original_apply = dust_map.apply_flat_field
    calls = []
    def apply_and_cancel(frame, gain):
        calls.append(1)
        result = original_apply(frame, gain)
        worker.cancel()
        return result
    monkeypatch.setattr(dust_map, 'apply_flat_field', apply_and_cancel)
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert len(calls) == 1
    assert outcome['counts']['cancelled'] == 1
    assert not outcome['outputs']


@pytest.mark.parametrize('grouped', [False, True])
def test_batch_excludes_white_reference_from_folder(qapp, tmp_path, monkeypatch, grouped):
    from core.workers import BatchWorker
    source = tmp_path / 'stack'
    source.mkdir()
    white_path, white = _reference(source)
    for i in range(2):
        assert cv2.imwrite(str(source / f'{i}.png'), np.full_like(white, 20000 + i * 1000))
    settings = {'format': 'png', 'fusion_method': 'dct', 'alignment': 'none',
                'flat_field_path': str(white_path), 'group_size': 2 if grouped else 0}
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'), settings, use_gpu=False)
    counts = []
    def fuse(fusion, method, images, params):
        counts.append(len(images))
        return images[0]
    monkeypatch.setattr(worker, '_fuse_one_stack', fuse)
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert outcome['success'] == 1
    assert counts == [2]
    (source / '1.png').unlink()
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out2'), settings, use_gpu=False)
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert outcome['counts']['failed'] == 1


def test_batch_rejects_transparency_before_flat_field(qapp, tmp_path, monkeypatch):
    from core.workers import BatchWorker
    import utils.dust_map as dust_map
    white, _ = _reference(tmp_path)
    source = tmp_path / 'stack'
    source.mkdir()
    frame = np.full((48, 64, 4), 200, np.uint8)
    frame[:5, :, 3] = 0
    for i in range(2):
        assert cv2.imwrite(str(source / f'{i}.png'), frame)
    def forbidden(*args, **kwargs):
        pytest.fail('flat-field reference decoded before rejecting transparency')
    monkeypatch.setattr(dust_map, 'load_flat_field_reference', forbidden)
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'),
                         {'format': 'png', 'fusion_method': 'dct', 'alignment': 'none',
                          'flat_field_path': str(white)}, use_gpu=False)
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert outcome['counts']['failed'] == 1
    assert 'transparent' in str(outcome).lower()


def test_grouped_reference_admits_before_decode_and_cancels_after(tmp_path, monkeypatch):
    import core.resource_policy as policy
    import utils.dust_map as dust_map
    from core.registration import RegistrationCancelled
    path, white = _reference(tmp_path)
    events = []
    monkeypatch.setattr(policy, 'require_memory', lambda amount: events.append(('admit', amount)))
    original = dust_map.load_flat_field_reference
    def decode(*args, **kwargs):
        result = original(*args, **kwargs)
        events.append(('decode', None))
        return result
    monkeypatch.setattr(dust_map, 'load_flat_field_reference', decode)
    monkeypatch.setattr(dust_map, 'compute_flat_field', lambda image: pytest.fail('gain computed after cancellation'))
    with pytest.raises(RegistrationCancelled):
        dust_map.grouped_flat_field_gain(path, should_cancel=lambda: len(events) >= 2)
    assert events[0][0] == 'admit' and events[0][1] >= path.stat().st_size + white.nbytes
    assert events[1][0] == 'decode'


def test_flat_admission_precedes_decode_and_float_allocation(monkeypatch):
    import core.resource_policy as policy
    import utils.dust_map as dust_map
    amounts = []
    def reject(amount):
        amounts.append(amount)
        raise policy.ResourceLimitError('resource limit')
    monkeypatch.setattr(policy, 'require_memory', reject)
    monkeypatch.setattr(dust_map, '_reference_header', lambda path: (12000, 8000, 4, 100))
    monkeypatch.setattr(dust_map.np, 'fromfile', lambda *a, **k: pytest.fail('decoded before admission'))
    with pytest.raises(policy.ResourceLimitError):
        dust_map.load_flat_field_reference('large.png')
    assert amounts.pop() == 100 + 2 * 12000 * 8000 * 3 * 4
    for shape, expected in [((8, 8, 3), 4608), ((8, 8), 1536)]:
        with pytest.raises(policy.ResourceLimitError):
            dust_map.compute_flat_field(np.ones(shape, np.uint8))
        assert amounts.pop() == expected


@pytest.mark.parametrize('shape,dtype,gain_shape,gain_dtype,expected', [
    ((8, 8, 3), np.uint8, (8, 8, 3), np.float32, 2304),
    ((8, 8), np.uint16, (8, 8, 3), np.float32, 1024),
    ((8, 8, 4), np.uint8, (8, 8, 3), np.float32, 2624),
    ((8, 8, 3), np.float32, (8, 8, 3), np.float64, 4608),
])
def test_flat_apply_admission_formula(monkeypatch, shape, dtype, gain_shape, gain_dtype, expected):
    import core.resource_policy as policy
    amounts = []
    monkeypatch.setattr(policy, 'require_memory', amounts.append)
    apply_flat_field(np.full(shape, 20, dtype), np.ones(gain_shape, gain_dtype))
    assert amounts == [expected]


def test_preview_arrays_match_previous_pipeline():
    from dialogs.dustmap import _prepare_preview
    from utils.image_utils import to_display_uint8
    for shape in [(11, 21), (11, 21, 3), (220, 350, 3)]:
        for dtype in (np.uint8, np.uint16):
            img = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape).astype(dtype)
            scale = min(340 / shape[1], 180 / shape[0])
            expected = cv2.resize(to_display_uint8(img),
                                  (max(1, round(shape[1] * scale)), max(1, round(shape[0] * scale))),
                                  interpolation=cv2.INTER_AREA)
            if expected.ndim == 2:
                expected = cv2.cvtColor(expected, cv2.COLOR_GRAY2BGR)
            actual = _prepare_preview(img)
            np.testing.assert_array_equal(actual, expected)
            assert actual.flags.c_contiguous and actual.flags.owndata and actual.dtype == np.uint8


def test_flat_preview_preparation_runs_in_worker_thread(qapp, monkeypatch):
    from PyQt6.QtCore import QEventLoop, QTimer, QThread
    import core.resource_policy as policy
    import dialogs.dustmap as dialog
    import utils.image_utils as image_utils
    monkeypatch.setattr(policy, 'require_memory', lambda amount: None)
    calls = []
    for module, name in ((dialog.cv2, 'applyColorMap'), (dialog.cv2, 'resize'),
                         (image_utils, 'to_display_uint8')):
        original = getattr(module, name)
        def wrapped(*args, _original=original, _name=name, **kwargs):
            calls.append((_name, QThread.currentThread()))
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapped)
    frame = np.full((16, 24, 3), 40000, np.uint16)
    worker = dialog._FlatFieldWorker('preview', 1, [frame], [], white=frame)
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    worker.start()
    loop.exec()
    assert not worker.isRunning() and not worker.error
    assert {name for name, _ in calls} == {'applyColorMap', 'resize', 'to_display_uint8'}
    assert all(thread != qapp.thread() for _, thread in calls)
    for array in worker.result[2:]:
        assert array.dtype == np.uint8 and array.flags.owndata and array.flags.c_contiguous
        assert array.shape[0] <= 180 and array.shape[1] <= 340 and array.shape[2] == 3


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [1, 3, 4])
@pytest.mark.parametrize('sigma', [4.0, 12.0])
def test_flat_math_preserves_previous_results(dtype, channels, sigma):
    rng = np.random.RandomState(42)
    high = 256 if dtype == np.uint8 else 65536
    shape = (24, 31) if channels == 1 else (24, 31, channels)
    white = rng.randint(high // 4, high, size=shape).astype(dtype)
    image = rng.randint(0, high, size=shape).astype(dtype)
    f = white[:, :, None] if white.ndim == 2 else white
    low = cv2.GaussianBlur(f.astype(np.float32), (0, 0), sigma)
    if low.ndim == 2:
        low = low[:, :, None]
    expected_gain = np.clip(low / float(low.mean()), 0.5, 2.0).astype(np.float32)
    gain = compute_flat_field(white, sigma=sigma)
    np.testing.assert_array_equal(gain, expected_gain)
    frame = image[:, :, None] if image.ndim == 2 else image
    color = frame[:, :, :3] if channels == 4 else frame
    effective_gain = gain[:, :, :3] if channels == 4 else gain
    result = np.clip(np.rint(color.astype(np.float32) / effective_gain), 0, high - 1).astype(dtype)
    if channels == 4:
        result = np.dstack((result, frame[:, :, 3].copy()))
    if image.ndim == 2:
        result = result[:, :, 0]
    np.testing.assert_array_equal(apply_flat_field(image, gain), result)
