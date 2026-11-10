"""Low temporary storage: assessment, GUI prompt, control replies, child binding, CLI."""
import os
import shutil
import time

import numpy as np
import pytest

from core import frame_storage

GIB = 1024 ** 3


def _fake_space(free_by_directory, total=100 * GIB):
    """Replace _space with fixed free space per directory (never touches a disk)."""
    def space(directory):
        key = os.path.abspath(directory)
        if key not in free_by_directory:
            raise OSError(f'not mocked: {key}')
        free = free_by_directory[key]
        if isinstance(free, Exception):
            raise free
        usage = shutil._ntuple_diskusage(total, total - free, free)
        reserve = max(GIB, min(total // 20, 4 * GIB))
        return usage, reserve, max(0, free - reserve)
    return space


@pytest.fixture
def explicit(tmp_path, monkeypatch):
    directory = tmp_path / 'scratch'
    directory.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', str(directory))
    return str(directory)


@pytest.mark.parametrize('free, status', [(50 * GIB, 'ok'), (10 * GIB, 'advisory'),
                                          (5 * GIB, 'critical')])
def test_assessment_classes(explicit, monkeypatch, free, status):
    monkeypatch.setattr(frame_storage, '_space', _fake_space({explicit: free}))
    # 4 GiB reserve: available is 46, 6 and 1 GiB.
    result = frame_storage.assess_scratch(2 * GIB, 4 * GIB, 3 * GIB)
    assert result['status'] == status
    assert result['directory'] == explicit and result['explicit'] is True
    assert result['required_bytes'] == 7 * GIB
    assert result['available_bytes'] == free - 4 * GIB
    assert result['reserve_bytes'] == 4 * GIB


def test_threshold_is_the_cache_admission_rule(explicit, monkeypatch):
    # Exactly enough for expected + optional after the reserve is not advisory.
    monkeypatch.setattr(frame_storage, '_space', _fake_space({explicit: 11 * GIB}))
    assert frame_storage.assess_scratch(GIB, 4 * GIB, 3 * GIB)['status'] == 'ok'
    monkeypatch.setattr(frame_storage, '_space', _fake_space({explicit: 11 * GIB - 1}))
    assert frame_storage.assess_scratch(GIB, 4 * GIB, 3 * GIB)['status'] == 'advisory'


def test_unqueryable_volume_is_unknown_not_a_prompt(explicit, monkeypatch):
    monkeypatch.setattr(frame_storage, '_space', _fake_space({explicit: OSError('gone')}))
    result = frame_storage.assess_scratch(GIB, GIB)
    assert result['status'] == 'unknown' and 'gone' in result['reason']


def test_without_explicit_directory_select_scratch_candidates_are_used(tmp_path, monkeypatch):
    small, large = tmp_path / 'small', tmp_path / 'large'
    small.mkdir(), large.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(frame_storage, '_scratch_candidates',
                        lambda explicit, paths: iter([str(small), str(large)]))
    monkeypatch.setattr(frame_storage, '_space', _fake_space(
        {str(small): 6 * GIB, str(large): 60 * GIB}))
    result = frame_storage.assess_scratch(GIB, 2 * GIB, GIB, placement_goal=3 * GIB)
    assert result['directory'] == str(large) and result['explicit'] is False
    assert result['status'] == 'ok'


def test_automatic_assessment_skips_unwritable_candidate(tmp_path, monkeypatch):
    unwritable, fallback = tmp_path / 'unwritable', tmp_path / 'fallback'
    unwritable.mkdir(), fallback.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(frame_storage, '_scratch_candidates',
                        lambda explicit, paths: iter([str(unwritable), str(fallback)]))
    monkeypatch.setattr(frame_storage, '_space', _fake_space(
        {str(unwritable): 60 * GIB, str(fallback): 5 * GIB}))
    real_mkdtemp = frame_storage.tempfile.mkdtemp

    def mkdtemp(*args, **kwargs):
        if kwargs.get('dir') == str(unwritable):
            raise PermissionError('unwritable')
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(frame_storage.tempfile, 'mkdtemp', mkdtemp)
    assessment = frame_storage.assess_scratch(0, 3 * GIB, GIB, placement_goal=GIB)
    assert assessment['directory'] == str(fallback)
    assert assessment['status'] == 'advisory'
    with frame_storage.scratch_directory():
        selected = frame_storage.select_scratch(0, preferred_size=GIB)
        assert os.path.dirname(selected) == str(fallback)


def test_automatic_assessment_searches_all_candidates_for_placement_goal(tmp_path, monkeypatch):
    small, large = tmp_path / 'small', tmp_path / 'large'
    small.mkdir(), large.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(frame_storage, '_scratch_candidates',
                        lambda explicit, paths: iter([str(small), str(large)]))
    monkeypatch.setattr(frame_storage, '_space', _fake_space(
        {str(small): 5 * GIB, str(large): 60 * GIB}))
    assessment = frame_storage.assess_scratch(0, 3 * GIB, GIB, placement_goal=2 * GIB)
    assert assessment['directory'] == str(large)
    assert assessment['status'] == 'ok'
    with frame_storage.scratch_directory():
        selected = frame_storage.select_scratch(0, preferred_size=2 * GIB)
        assert os.path.dirname(selected) == str(large)


def test_automatic_assessment_ignores_later_full_volume(tmp_path, monkeypatch):
    roomy, full = tmp_path / 'roomy', tmp_path / 'full'
    roomy.mkdir(), full.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(frame_storage, '_scratch_candidates',
                        lambda explicit, paths: iter([str(roomy), str(full)]))
    monkeypatch.setattr(frame_storage, '_space', _fake_space(
        {str(roomy): 60 * GIB, str(full): 4 * GIB}))
    assessment = frame_storage.assess_scratch(GIB, 3 * GIB, GIB, placement_goal=4 * GIB)
    assert assessment['status'] == 'ok'
    assert assessment['directory'] == str(roomy)
    with frame_storage.scratch_directory():
        selected = frame_storage.select_scratch(GIB, preferred_size=4 * GIB)
        assert os.path.dirname(selected) == str(roomy)


@pytest.mark.parametrize('same_volume', [True, False])
def test_automatic_low_space_warns_without_claiming_directory(tmp_path, monkeypatch, same_volume):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir(), second.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(frame_storage, '_scratch_candidates',
                        lambda explicit, paths: iter([str(first), str(second)]))
    monkeypatch.setattr(frame_storage, '_space', _fake_space(
        {str(first): 5 * GIB, str(second): 5 * GIB}))
    monkeypatch.setattr(frame_storage, '_scratch_volume',
                        lambda path: 1 if same_volume or path == str(first) else 2)
    assessment = frame_storage.assess_scratch(0, 2 * GIB, GIB)
    assert assessment['status'] == 'advisory'
    assert assessment['directory'] is None
    assert assessment['volume_count'] == (1 if same_volume else 2)
    assert assessment['available_bytes'] == GIB


def test_automatic_assessment_warns_about_volume_selected_at_runtime(tmp_path, monkeypatch):
    small, large = tmp_path / 'small', tmp_path / 'large'
    small.mkdir(), large.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(frame_storage, '_scratch_candidates',
                        lambda explicit, paths: iter([str(small), str(large)]))
    monkeypatch.setattr(frame_storage, '_space', _fake_space(
        {str(small): 5 * GIB, str(large): 60 * GIB}))
    assessment = frame_storage.assess_scratch(0, 3 * GIB, GIB, placement_goal=GIB)
    assert assessment['status'] == 'advisory'
    assert assessment['directory'] == str(small)
    with frame_storage.scratch_directory() as scratch:
        selected = frame_storage.select_scratch(0, preferred_size=GIB)
        assert os.path.dirname(selected) == str(small)


def test_untiled_dct_output_estimate_never_causes_critical(tmp_path, monkeypatch):
    from core.storage_preflight import render_storage_estimate

    frames = [np.zeros((40, 50, 3), np.uint8) for _ in range(2)]
    estimate = render_storage_estimate(frames, algorithm='dct', alignment='none')
    assert estimate[0] == 0
    monkeypatch.setattr(frame_storage, '_space', _fake_space({str(tmp_path): 4 * GIB}))
    assert frame_storage.assess_scratch(*estimate, directory=str(tmp_path))['status'] == 'advisory'


def test_directory_validation(tmp_path):
    with pytest.raises(OSError):
        frame_storage.validate_storage_directory(tmp_path / 'missing')
    assert frame_storage.validate_storage_directory(tmp_path) == str(tmp_path)
    if os.name != 'nt' and os.geteuid() != 0:
        locked = tmp_path / 'locked'
        locked.mkdir()
        locked.chmod(0o500)
        try:
            with pytest.raises(OSError):
                frame_storage.validate_storage_directory(locked)
        finally:
            locked.chmod(0o700)


def test_runtime_admission_still_applies_after_a_preflight(explicit, monkeypatch):
    monkeypatch.setattr(frame_storage, '_space', _fake_space({explicit: 50 * GIB}))
    assert frame_storage.assess_scratch(GIB, GIB)['status'] == 'ok'
    monkeypatch.setattr(frame_storage, '_space', _fake_space({explicit: 4 * GIB}))  # space vanished
    with pytest.raises(frame_storage.ScratchSpaceError) as raised:
        frame_storage.store_frame(np.zeros((64, 64), np.uint8), prefer_ram=False)
    assert explicit in str(raised.value)


def test_estimate_follows_the_pipeline_materialization_rules(tmp_path, monkeypatch):
    from dataclasses import replace
    from core import storage_preflight
    from core.algorithms import get_fusion
    from core.source_frames import scan_source
    from core.storage_preflight import render_storage_estimate
    from utils.image_utils import imwrite_auto
    arrays = [np.zeros((40, 50, 3), np.uint8) for _ in range(3)]
    # Streaming sources do not require a tiled output buffer in every backend.
    minimum, expected, optional, _ = render_storage_estimate(
        arrays, algorithm='stackmffv3', alignment='ecc')
    assert minimum == 0 and expected > minimum
    assert optional == 0          # in-memory arrays are never cached
    # Aligned sources may already be reusable without materialization.
    descriptor = replace(get_fusion('dct'), streaming_sources=False)
    with monkeypatch.context() as patch:
        patch.setattr(storage_preflight, 'get_fusion', lambda algorithm: descriptor)
        minimum, *_ = render_storage_estimate(arrays, algorithm='dct', alignment='ecc')
        # ... and then stores one cleaned copy of every frame.
        cleaned, *_ = render_storage_estimate(arrays, algorithm='dct', alignment='ecc',
                                              cleanup=True)
    assert minimum == 0
    assert cleaned == minimum
    # A streaming render keeps sensor cleanup lazy: it adds no mandatory disk.
    lazy, *_ = render_storage_estimate(arrays, algorithm='dct', alignment='none', cleanup=True)
    assert lazy == 0
    paths = []
    for index, image in enumerate(arrays):
        path = tmp_path / f'f{index}.png'
        assert imwrite_auto(str(path), image)
        paths.append(str(path))
    sources = [scan_source(path, header_only=True)[0] for path in paths]
    _, _, optional, source_paths = render_storage_estimate(
        sources, algorithm='stackmffv3', alignment='none')
    assert optional >= sum(frame.nbytes for frame in sources)
    assert sorted(source_paths) == sorted(paths)


# ---- GUI and control channel -------------------------------------------------

@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    from main import OpenFocus
    from utils import recovery, settings_store
    # Individual asynchronous tests opt in explicitly, independent of platform.
    monkeypatch.setattr(frame_storage, 'automatic_preflight_may_block', lambda: False)
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    settings.clear()
    settings.setValue('ui/first_run_done', '1')
    settings.setValue('updates/last_check_epoch', 9999999999)
    settings.setValue('rendering/ai_cpu_warning_dismissed', True)
    monkeypatch.setattr(settings_store, '_settings_singleton', settings)
    monkeypatch.setattr(recovery, '_base_dir', lambda: str(tmp_path / 'recovery'))
    monkeypatch.setenv('OPENFOCUS_CONTROL_FILE', str(tmp_path / 'control.json'))
    instance = OpenFocus()
    instance.experimental_features = True
    instance.use_gpu = False
    instance.restore_last_stack = False
    instance.set_alignment_method('none')
    instance.chk_quick_preview.setChecked(False)
    monkeypatch.setattr(instance, 'current_fusion_method', lambda: 'dct')
    instance.raw_images = [np.zeros((8, 8, 3), np.uint8)] * 2
    instance.image_filenames = ['a.tif', 'b.tif']
    instance.excluded_frames = set()
    yield instance
    instance.raw_images, instance.image_filenames = [], []
    instance._force_quit = True
    instance.close()
    instance.deleteLater()
    qapp.processEvents()


class _Signal:
    def connect(self, *args):
        pass


class FakeWorker:
    """Records how the render manager would start the isolated render."""

    created = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.storage_directory = None
        self.finished_signal = self.error_signal = _Signal()
        self.progress_signal = self.stage_signal = _Signal()
        FakeWorker.created.append(self)

    def start(self):
        pass

    def isRunning(self):
        return False


@pytest.fixture
def fake_worker(monkeypatch):
    from controllers import render_manager
    FakeWorker.created = []
    monkeypatch.setattr(render_manager, 'RenderProcessWorker', FakeWorker)
    return FakeWorker


def _assessment(status, directory='/scratch'):
    return {'status': status, 'directory': directory, 'explicit': True, 'free_bytes': 9 * GIB,
            'reserve_bytes': 4 * GIB, 'available_bytes': 5 * GIB, 'minimum_bytes': GIB,
            'expected_bytes': 2 * GIB, 'optional_bytes': 4 * GIB, 'required_bytes': 6 * GIB}


def _assess_as(monkeypatch, statuses):
    """Make assess_scratch return the given statuses in order (per directory)."""
    calls = []

    def assess(*args, directory=None):
        calls.append(directory)
        status = statuses[directory] if isinstance(statuses, dict) else statuses
        return _assessment(status, directory or '/scratch')
    monkeypatch.setattr(frame_storage, 'assess_scratch', assess)
    return calls


def test_small_render_never_prompts(window, fake_worker, monkeypatch):
    manager = window.render_manager
    monkeypatch.setattr(manager, '_confirm_storage', lambda *a: pytest.fail('prompted'))
    assert manager.start_render() == 'started'
    assert fake_worker.created[0].storage_directory is None


@pytest.mark.parametrize('language', ['en', 'zh', 'ja', 'es', 'de', 'fr', 'ru', 'ko', 'pt', 'it', 'nl', 'pl'])
def test_neutral_storage_warning_displays_localized_upper_bound(window, monkeypatch, language):
    from locales import trans

    monkeypatch.setattr(trans, 'current_lang', language)
    assessment = _assessment('advisory', None)
    message = window.render_manager._storage_message(assessment)
    bound = trans.t('storage_available_upper_bound').format(
        size=frame_storage.readable_bytes(assessment['available_bytes']))
    assert bound in message
    assert frame_storage.readable_bytes(assessment['required_bytes']) in message
    assert 'None' not in message


def test_gui_defers_slow_automatic_discovery(window, fake_worker, monkeypatch, qapp):
    import threading
    from PyQt6.QtCore import QTimer
    from PyQt6.QtTest import QTest

    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(frame_storage, 'automatic_preflight_may_block', lambda: True)
    monkeypatch.setattr(frame_storage, '_storage_directory', None)

    def slow_assess(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return _assessment('ok')

    monkeypatch.setattr(frame_storage, 'assess_scratch', slow_assess)
    events = []
    QTimer.singleShot(0, lambda: events.append('responsive'))
    manager = window.render_manager
    try:
        assert manager.start_render() == 'preflight'
        qapp.processEvents()
        assert events == ['responsive'] and not fake_worker.created
        release.set()
        for _ in range(100):
            if fake_worker.created:
                break
            QTest.qWait(10)
        assert entered.is_set() and len(fake_worker.created) == 1
        assert manager.last_storage_assessment['status'] == 'ok'
    finally:
        release.set()


@pytest.mark.parametrize('change', ['source', 'source_replaced', 'settings', 'cancel'])
def test_delayed_storage_preflight_rejects_stale_or_cancelled_request(
        window, fake_worker, monkeypatch, qapp, change):
    from controllers import render_manager

    probes = []
    monkeypatch.setattr(frame_storage, 'automatic_preflight_may_block', lambda: True)
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(render_manager._StorageProbe, 'start', lambda probe, estimate: probes.append(probe))
    manager = window.render_manager
    assert manager.start_render() == 'preflight'
    if change == 'source':
        window.source_revision = getattr(window, 'source_revision', 0) + 1
    elif change == 'source_replaced':
        window.raw_images = list(window.raw_images)
    elif change == 'settings':
        window.thread_count += 1
    else:
        assert manager.start_render() == 'cancelled'
    probes[0].completed.emit(_assessment('ok'))
    qapp.processEvents()
    assert not fake_worker.created
    assert manager._pending_storage is None
    assert manager.last_state == ('cancelled' if change == 'cancel' else 'discarded')


def test_cpu_confirmation_is_not_repeated_after_storage_preflight(window, fake_worker, monkeypatch, qapp):
    from controllers import render_manager

    probes, cpu_prompts, storage_prompts = [], [], []
    monkeypatch.setattr(frame_storage, 'automatic_preflight_may_block', lambda: True)
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(render_manager._StorageProbe, 'start', lambda probe, estimate: probes.append(probe))
    manager = window.render_manager
    monkeypatch.setattr(manager, '_ai_confirmation_needed', lambda method: True)
    monkeypatch.setattr(manager, '_confirm_cpu_ai_render', lambda method: cpu_prompts.append(method) or True)
    monkeypatch.setattr(manager, '_confirm_storage',
                        lambda assessment, frames: storage_prompts.append(assessment) or None)
    assert manager.start_render() == 'preflight'
    assert len(cpu_prompts) == 1 and not storage_prompts
    probes[0].completed.emit(_assessment('advisory'))
    qapp.processEvents()
    assert len(cpu_prompts) == 1 and len(storage_prompts) == 1
    assert not fake_worker.created


@pytest.mark.parametrize('outcome', ['advisory', 'cancel', 'stale'])
def test_pending_storage_preserves_previous_result(window, fake_worker, monkeypatch, qapp, outcome):
    from controllers import render_manager
    from core.control.server import _render_state

    probes = []
    monkeypatch.setattr(frame_storage, 'automatic_preflight_may_block', lambda: True)
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(render_manager._StorageProbe, 'start', lambda probe, estimate: probes.append(probe))
    manager = window.render_manager
    previous = {'output': 'earlier result'}
    manager.last_result = previous
    manager.last_state, manager.last_error = 'error', 'earlier render failed'
    assert manager.start_render(interactive=False) == 'preflight'
    assert _render_state(window)['result'] == previous
    if outcome == 'cancel':
        assert manager.cancel_storage_preflight()
    elif outcome == 'stale':
        window.source_revision = getattr(window, 'source_revision', 0) + 1
    probes[0].completed.emit(_assessment('advisory'))
    qapp.processEvents()
    assert manager.last_result is previous
    assert _render_state(window)['result'] == previous
    state = _render_state(window)
    if outcome == 'advisory':
        assert state['state'] == 'error' and state['error'] == 'earlier render failed'
    else:
        assert state['state'] == ('cancelled' if outcome == 'cancel' else 'discarded')
        assert state['error'] is None
    assert not fake_worker.created


@pytest.mark.parametrize('storage_status,expected', [
    ('advisory', 'confirm_storage'), ('critical', 'insufficient_storage'),
])
def test_rejected_storage_preflight_preserves_previous_control_error(
        window, fake_worker, monkeypatch, tmp_path, qapp, storage_status, expected):
    from controllers import render_manager
    from core.control.server import ControlServer

    probes = []
    monkeypatch.setattr(frame_storage, 'automatic_preflight_may_block', lambda: True)
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(render_manager._StorageProbe, 'start', lambda probe, estimate: probes.append(probe))
    manager = window.render_manager
    manager.last_state, manager.last_error = 'error', 'earlier render failed'
    server = ControlServer(window, socket_name=f'OpenFocus-previous-error-{os.getpid()}-{id(tmp_path)}')
    try:
        assert manager.start_render(interactive=False) == 'preflight'
        outcomes = []
        manager.set_storage_completion(outcomes.append)
        probes[0].completed.emit(_assessment(storage_status))
        qapp.processEvents()
        assert outcomes == [expected]
        state = server._cmd_render_state({})
        assert state['state'] == 'error' and state['error'] == 'earlier render failed'
        assert not state['busy'] and not fake_worker.created
    finally:
        manager.cancel_storage_preflight()
        server.close()


def test_gui_still_assesses_explicit_directory_on_slow_platform(window, monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(frame_storage, 'sys', SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    calls = _assess_as(monkeypatch, 'critical')
    assessment = window.render_manager._storage_preflight(None, window.raw_images)
    assert assessment['status'] == 'critical'
    assert calls == [None]


@pytest.mark.parametrize('setup', ['preview', 'roi', 'compare'])
def test_drafts_roi_and_compare_are_not_assessed(window, monkeypatch, setup):
    manager = window.render_manager
    _assess_as(monkeypatch, 'critical')
    if setup == 'preview':
        window.chk_quick_preview.setChecked(True)
    elif setup == 'roi':
        window.roi_mode_active = True
    assert manager._storage_preflight('dct' if setup == 'compare' else None,
                                      window.raw_images) is None


def test_advisory_prompts_once_and_continue_starts_the_render(window, fake_worker, monkeypatch):
    manager = window.render_manager
    _assess_as(monkeypatch, 'advisory')
    prompts = []
    monkeypatch.setattr(manager, '_confirm_storage',
                        lambda assessment, frames: prompts.append(assessment)
                        or {'directory': None, 'keep': False})
    assert manager.start_render() == 'started'
    assert len(prompts) == 1 and prompts[0]['status'] == 'advisory'
    assert fake_worker.created[0].storage_directory is None


def test_cancel_in_the_prompt_starts_nothing_and_unlocks(window, fake_worker, monkeypatch):
    manager = window.render_manager
    _assess_as(monkeypatch, 'critical')
    monkeypatch.setattr(manager, '_confirm_storage', lambda *a: None)
    assert manager.start_render() == 'cancelled'
    assert not fake_worker.created
    assert window.btn_render.isEnabled() and window.combo_fusion.isEnabled()


@pytest.mark.parametrize('keep', [False, True])
def test_chosen_directory_binds_the_next_render_and_persists_only_on_yes(
        window, fake_worker, monkeypatch, tmp_path, keep):
    from utils.settings_store import get_settings
    manager = window.render_manager
    chosen = str(tmp_path / 'roomy')
    os.mkdir(chosen)
    _assess_as(monkeypatch, 'advisory')
    monkeypatch.setattr(manager, '_confirm_storage', lambda *a: {'directory': chosen, 'keep': keep})
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    assert manager.start_render() == 'started'
    worker = fake_worker.created[0]
    if keep:
        assert worker.storage_directory is None          # the global setting carries it
        assert frame_storage._storage_directory == chosen
        assert window.storage_directory == chosen
        assert get_settings().value('rendering/storage_directory') == chosen
    else:
        assert worker.storage_directory == chosen        # this render only
        assert frame_storage._storage_directory is None
        assert not get_settings().value('rendering/storage_directory', '')


def _click_modals(qapp, script):
    """Click modal buttons by controlId, in order, as they appear."""
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QPushButton
    clicked = []
    timer = QTimer()

    def step():
        dialog = QApplication.activeModalWidget()
        if dialog is None or not script:
            return
        for button in dialog.findChildren(QPushButton):
            if button.property('controlId') == script[0]:
                clicked.append(script.pop(0))
                button.click()
                return
    timer.timeout.connect(step)
    timer.start(20)
    return timer, clicked


def test_dialog_validates_reassesses_and_asks_to_keep(window, monkeypatch, tmp_path, qapp):
    from utils.file_dialogs import FileDialog
    manager = window.render_manager
    full, roomy = str(tmp_path / 'full'), str(tmp_path / 'roomy')
    os.mkdir(full), os.mkdir(roomy)
    picks = [str(tmp_path / 'missing'), full, roomy]
    monkeypatch.setattr(FileDialog, 'getExistingDirectory', staticmethod(lambda *a: picks.pop(0)))
    warnings = []
    from controllers import render_manager
    monkeypatch.setattr(render_manager, 'show_warning_box', lambda *a: warnings.append(a))
    calls = _assess_as(monkeypatch, {full: 'advisory', roomy: 'ok'})
    timer, clicked = _click_modals(qapp, ['btn_storage_choose', 'btn_storage_choose',
                                          'btn_storage_choose', 'btn_storage_keep'])
    try:
        choice = manager._confirm_storage(_assessment('critical'), window.raw_images)
    finally:
        timer.stop()
    assert choice == {'directory': roomy, 'keep': True}
    assert len(warnings) == 1                     # the missing directory was rejected
    assert calls == [full, roomy]                 # each valid choice was re-assessed
    assert clicked[-1] == 'btn_storage_keep'


def test_critical_dialog_offers_no_continue(window, qapp):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QPushButton
    seen = []

    def inspect():
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            return
        buttons = {b.property('controlId'): b for b in dialog.findChildren(QPushButton)}
        seen.append(set(buttons) - {None})
        buttons['btn_storage_cancel'].click()
    timer = QTimer()
    timer.timeout.connect(inspect)
    timer.start(20)
    try:
        assert window.render_manager._confirm_storage(_assessment('critical'), []) is None
    finally:
        timer.stop()
    assert seen[0] == {'btn_storage_choose', 'btn_storage_cancel'}


@pytest.mark.parametrize('status,confirmed,expected', [
    ('advisory', False, 'confirm_required'),
    ('critical', True, 'insufficient_storage'),
    ('advisory', True, 'started'),
])
def test_control_render_reply_waits_for_storage_preflight(
        window, fake_worker, monkeypatch, tmp_path, qapp, status, confirmed, expected):
    from controllers import render_manager
    from core.control.protocol import encode
    from core.control.server import ControlServer, _DeferredResponse

    probes = []
    monkeypatch.setattr(frame_storage, 'automatic_preflight_may_block', lambda: True)
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(render_manager._StorageProbe, 'start', lambda probe, estimate: probes.append(probe))
    server = ControlServer(window, socket_name=f'OpenFocus-async-storage-{os.getpid()}-{id(tmp_path)}')
    try:
        response = server.handle_line(encode({'id': '1', 'cmd': 'render.start',
                                              'args': {'confirm': confirmed},
                                              'token': server.token}), token=server.token)
        assert isinstance(response, _DeferredResponse)
        delivered = []
        response.start(lambda success, value: delivered.append((success, value)))
        assert server._cmd_render_state({})['busy'] is True
        assert not fake_worker.created and not delivered
        probes[0].completed.emit(_assessment(status))
        qapp.processEvents()
        assert len(delivered) == 1
        success, value = delivered[0]
        if expected == 'started':
            assert success and value['started'] and value['storage']['status'] == status
            assert len(fake_worker.created) == 1
        else:
            assert not success and value.code == expected
            assert value.details['storage']['status'] == status
            assert not fake_worker.created
    finally:
        window.render_manager.cancel_storage_preflight()
        server.close()


@pytest.mark.parametrize('action', ['disconnect', 'cancel'])
def test_control_cancels_pending_storage_request(
        window, fake_worker, monkeypatch, tmp_path, qapp, action):
    from controllers import render_manager
    from core.control.protocol import encode
    from core.control.server import ControlServer

    probes = []
    monkeypatch.setattr(frame_storage, 'automatic_preflight_may_block', lambda: True)
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(render_manager._StorageProbe, 'start', lambda probe, estimate: probes.append(probe))
    server = ControlServer(window, socket_name=f'OpenFocus-cancel-storage-{os.getpid()}-{id(tmp_path)}')
    try:
        response = server.handle_line(encode({'id': '1', 'cmd': 'render.start',
                                              'args': {}, 'token': server.token}), token=server.token)
        delivered = []
        handle = response.start(lambda success, value: delivered.append((success, value)))
        if action == 'disconnect':
            handle.cancel()
        else:
            assert server._cmd_render_cancel({})['cancelling'] is True
        assert delivered and delivered[0][1].code == 'internal'
        probes[0].completed.emit(_assessment('ok'))
        qapp.processEvents()
        assert not fake_worker.created and window.render_manager._pending_storage is None
    finally:
        server.close()


def test_control_channel_statuses_and_storage_in_replies(window, fake_worker, monkeypatch, tmp_path):
    from core.control.protocol import encode
    from core.control.server import ControlServer
    server = ControlServer(window, socket_name=f'OpenFocus-storage-{os.getpid()}-{id(tmp_path)}')
    try:
        def call(cmd, **args):
            return server.handle_line(encode({'id': '1', 'cmd': cmd, 'args': args,
                                              'token': server.token}), token=server.token)
        _assess_as(monkeypatch, 'advisory')
        reply = call('render.start')
        assert reply['error']['code'] == 'confirm_required'
        assert reply['error']['details']['storage']['status'] == 'advisory'
        assert not fake_worker.created
        reply = call('render.start', confirm=True)
        assert reply['result']['started'] is True
        assert reply['result']['storage']['status'] == 'advisory'
        _assess_as(monkeypatch, 'critical')
        reply = call('render.start', confirm=True)
        assert reply['error']['code'] == 'insufficient_storage'
        assert reply['error']['details']['storage']['status'] == 'critical'
        assert len(fake_worker.created) == 1
        # An agent fixes it by choosing another directory.
        bad = call('option.set', name='storage_directory', value=str(tmp_path / 'missing'))
        assert bad['error']['code'] == 'invalid_argument'
        good = call('option.set', name='storage_directory', value=str(tmp_path))
        assert good['ok'] and frame_storage._storage_directory == str(tmp_path)
        assert call('app.state')['result']['options']['storage_directory'] == str(tmp_path)
    finally:
        server.close()
        frame_storage.configure_storage_directory(None)


# ---- the chosen directory reaches the spawned render child -------------------

def _report_directory_child(connection, cancel_event, request, budget, directory, language):
    connection.send(('error', f'DIRECTORY={directory}', None))
    connection.close()


def test_one_render_directory_reaches_the_child(qapp, tmp_path, monkeypatch):
    from PyQt6.QtTest import QTest
    from core import render_process
    configured, chosen = tmp_path / 'configured', tmp_path / 'chosen'
    configured.mkdir(), chosen.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', str(configured))
    monkeypatch.setattr(render_process, '_render_child', _report_directory_child)
    from core.source_frames import scan_source
    from utils.image_utils import imwrite_auto
    sources = []
    for index in range(2):
        path = tmp_path / f'source-{index}.png'
        assert imwrite_auto(str(path), np.full((8, 8, 3), index * 50, np.uint8))
        sources.append(scan_source(str(path))[0])
    seen = []
    for override in (str(chosen), None):
        worker = render_process.RenderProcessWorker(
            sources, algorithm='dct', thread_count=1, use_gpu=False)
        worker.storage_directory = override
        errors = []
        worker.error_signal.connect(errors.append)
        worker.start()
        deadline = time.monotonic() + 30
        while worker.isRunning() and time.monotonic() < deadline:
            QTest.qWait(10)
        worker.wait(1000)
        QTest.qWait(50)            # deliver the queued error signal
        seen.append(errors)
    assert seen[0] == [f'DIRECTORY={chosen}']
    assert seen[1] == [f'DIRECTORY={configured}']     # one-render choice does not stick
    assert not list(chosen.glob('openfocus-frame-*'))
    assert not list(configured.glob('openfocus-frame-*'))
