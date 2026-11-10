"""Detailed (DEBUG) logging: runtime switch, what it records, child processes, CLI."""
import logging
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from conftest import make_two_focus_stack
from utils import logging_setup as logs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def log_folder(monkeypatch, tmp_path):
    logs.shutdown_logging()
    monkeypatch.setenv("OPENFOCUS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("OPENFOCUS_LOG_LEVEL", raising=False)
    logs.setup_logging()
    yield logs.logs_dir()
    logs.shutdown_logging()


def _stack(count=3, size=(150, 110)):
    first, second = make_two_focus_stack(size)
    frames = [first, second]
    while len(frames) < count:
        frames.append(((first.astype(np.uint16) + second) // 2).astype(np.uint8))
    return frames


def _tiled_fusion():
    from core.multi_focus_fusion import MultiFocusFusion
    fusion = MultiFocusFusion(algorithm='guided_filter', use_gpu=False, tile_enabled=True,
                              tile_block_size=64, tile_overlap=16, tile_threshold=32)
    return fusion.fuse(input_source=_stack(), img_resize=None, thread_count=2)


def test_runtime_level_survives_repeated_setup_and_returns_to_the_default(log_folder):
    logger = logs.get_logger(__name__)
    assert not logs.debug_enabled()
    assert logs.set_debug_logging(True) == logging.DEBUG
    assert logs.debug_enabled()
    logs.setup_logging()                     # every new window calls this again
    logger.debug("detail while on")
    logs.set_debug_logging(False)
    logger.debug("detail while off")
    text = logs.session_log_excerpt()
    assert "detail while on" in text and "detail while off" not in text
    # The log itself shows where its detail starts and stops.
    changes = [line for line in text.splitlines() if "event=log_level.changed" in line]
    assert ['"level": "DEBUG"' in line for line in changes] == [True, False]
    assert not logs.debug_enabled()


def test_switching_off_returns_to_the_environment_level(log_folder, monkeypatch):
    monkeypatch.setenv("OPENFOCUS_LOG_LEVEL", "WARNING")
    logs.setup_logging()
    logs.set_debug_logging(True)
    assert logs.debug_enabled()
    logs.set_debug_logging(False)
    logs.log_action("quiet_action")
    logs.get_logger(__name__).warning("still recorded")
    text = logs.session_log_excerpt()
    assert "event=quiet_action" not in text and "still recorded" in text


def test_a_change_between_quiet_levels_is_still_recorded(log_folder):
    logs.set_log_level("warning")
    logs.set_log_level("error")
    logs.set_log_level(None)
    changes = [line for line in logs.session_log_excerpt().splitlines()
               if "event=log_level.changed" in line]
    assert ['"level": "%s"' % name in line for line, name in
            zip(changes, ("WARNING", "ERROR", "INFO"))] == [True, True, True]


def test_moving_the_log_folder_keeps_the_chosen_detail(log_folder, tmp_path):
    logs.set_log_level("debug")
    logs.setup_logging(log_dir=tmp_path / "elsewhere")
    logs.get_logger(__name__).debug("detail after the move")
    assert "detail after the move" in logs.session_log_excerpt()


def test_debug_events_are_recorded_only_with_detail(log_folder):
    logs.log_debug("hidden_plan", workers=1)
    logs.set_debug_logging(True)
    logs.log_debug("visible_plan", workers=2, shape=(3, 4))
    text = logs.session_log_excerpt()
    assert "hidden_plan" not in text
    line = next(line for line in text.splitlines() if "event=visible_plan" in line)
    assert " DEBUG " in line and '"workers": 2' in line


def test_detail_explains_tiling_planning_and_scratch_without_changing_pixels(log_folder):
    reference = _tiled_fusion()
    quiet = logs.session_log_excerpt(1_000_000)
    logs.set_debug_logging(True)
    detailed = _tiled_fusion()
    text = logs.session_log_excerpt(1_000_000)
    np.testing.assert_array_equal(detailed, reference)
    for event in ("fusion_tiling", "fusion_tile", "execution_plan"):
        assert f"event={event} " not in quiet
        assert f"event={event} " in text
    tiling = next(line for line in text.splitlines() if "event=fusion_tiling" in line)
    assert '"block": 64' in tiling and '"overlap": 16' in tiling and '"tiles": 12' in tiling
    assert text.count("event=fusion_tile ") == 12
    # Pixels and paths stay out of these records.
    assert not any("ndarray" in line or "array(" in line for line in text.splitlines())


def test_advanced_settings_switch_applies_at_once_and_is_not_a_template_setting(
        qapp, log_folder):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QWidget
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from utils.project_file import TEMPLATE_SETTINGS
    from utils.settings_store import _PERSISTED_FIELDS

    window = QWidget()
    window.saved = []
    window.persist_settings = lambda: window.saved.append(window.debug_logging)
    window.update_manager = SimpleNamespace(auto_download=True)
    window.update_manager.set_auto_download = lambda value: setattr(
        window.update_manager, "auto_download", value)
    try:
        dialog = AdvancedSettingsDialog(window)
        assert dialog.chk_debug_logging.property("controlId") == "chk_debug_logging"
        assert dialog.btn_open_logs.property("controlId") == "btn_open_logs"
        assert not dialog.chk_debug_logging.isChecked()
        dialog.chk_debug_logging.setChecked(True)
        dialog.reject()                        # Cancel changes nothing
        assert not logs.debug_enabled()
        dialog.deleteLater()

        dialog = AdvancedSettingsDialog(window)
        dialog.chk_debug_logging.setChecked(True)
        dialog._apply()
        assert window.debug_logging is True and window.saved == [True]
        assert logs.debug_enabled()
        dialog.deleteLater()

        dialog = AdvancedSettingsDialog(window)
        assert dialog.chk_debug_logging.isChecked()
        dialog.chk_debug_logging.setChecked(False)
        dialog._apply()
        assert not logs.debug_enabled()
        dialog.deleteLater()
    finally:
        window.deleteLater()
        qapp.processEvents()
    assert "debug_logging" in _PERSISTED_FIELDS
    assert "debug_logging" not in TEMPLATE_SETTINGS


def _wait(worker, qapp):
    from PyQt6.QtTest import QTest
    deadline = time.monotonic() + 60
    while worker.isRunning() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert not worker.isRunning()
    worker.wait(1000)
    qapp.processEvents()


@pytest.mark.parametrize('detailed', [True, False])
def test_isolated_render_records_the_same_detail_as_its_window(
        qapp, log_folder, tmp_path, monkeypatch, detailed):
    from core import frame_storage
    from core.render_process import RenderProcessWorker
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', str(scratch))
    logs.set_debug_logging(detailed)
    worker = RenderProcessWorker(
        _stack(), algorithm='guided_filter', use_gpu=False, thread_count=1,
        exposure_match=False, tile_enabled=True, tile_block_size=64, tile_overlap=16,
        tile_threshold=32)
    results, errors = [], []
    worker.finished_signal.connect(lambda *value: results.append(value))
    worker.error_signal.connect(errors.append)
    worker.start()
    _wait(worker, qapp)
    assert not errors and results
    del results
    own = logs.session_log_path()
    child = [path for path in log_folder.glob('openfocus-*.log') if path != own]
    assert len(child) == 1
    text = child[0].read_text(encoding='utf-8')
    assert 'event=render.started' in text
    assert ('event=fusion_tiling ' in text) == detailed
    assert ('event=render_resources ' in text) == detailed


def _run_cli(*argv, env):
    return subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'main.py'), *argv],
                          capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300,
                          env={**os.environ, **env})


def test_cli_log_level_selects_detail_and_names_the_log_file(tmp_path):
    from utils.image_utils import imwrite_auto
    stack = tmp_path / 'stack'
    stack.mkdir()
    for index, image in enumerate(_stack()):
        assert imwrite_auto(str(stack / f'f{index}.png'), image)
    tiles = ('--method', 'guided_filter', '--threads', '1', '--tile-size', '64',
             '--tile-overlap', '16', '--tile-threshold', '32')
    env = {'OPENFOCUS_LOG_DIR': str(tmp_path / 'logs'), 'OPENFOCUS_LOG_LEVEL': ''}
    plain = _run_cli('--input', str(stack), '--output', str(tmp_path / 'plain.png'), *tiles,
                     env=env)
    assert plain.returncode == 0, plain.stderr
    assert 'Log file:' not in plain.stderr
    quiet = {path: path.read_text(encoding='utf-8')
             for path in (tmp_path / 'logs').glob('openfocus-*.log')}
    assert quiet and not any(' DEBUG ' in text for text in quiet.values())

    detailed = _run_cli('--input', str(stack), '--output', str(tmp_path / 'detailed.png'),
                        *tiles, '--log-level', 'debug', env=env)
    assert detailed.returncode == 0, detailed.stderr
    named = next(line for line in detailed.stderr.splitlines() if line.startswith('Log file: '))
    path = named.removeprefix('Log file: ').strip()
    assert path not in {str(known) for known in quiet}
    text = open(path, encoding='utf-8').read()
    assert 'event=fusion_tiling ' in text and 'event=render_resources ' in text
    import cv2
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / 'detailed.png')),
                                  cv2.imread(str(tmp_path / 'plain.png')))

    invalid = _run_cli('--input', str(stack), '--output', str(tmp_path / 'x.png'),
                       '--log-level', 'chatty', env=env)
    assert invalid.returncode == 2
    assert not (tmp_path / 'x.png').exists()
