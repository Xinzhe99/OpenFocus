"""Behavioral boundaries for settings drafts, flat fields, and feedback."""
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QDialog, QWidget


@pytest.fixture
def settings_window(qapp):
    window = QWidget()
    window.tile_enabled = True
    window.tile_block_size = 1024
    window.tile_overlap = 256
    window.tile_threshold = 2048
    window.thread_count = 4
    window.reg_downscale_width = 1024
    window.align_cache_enabled = True
    window.ai_batch_size = 2
    window._fusion_parameters_by_method = {"stackmffv3": {"mode": "soft"}}
    window.fusion_parameters_for = lambda method, depth=False: dict(
        window._fusion_parameters_by_method.get(method, {}))
    window.set_fusion_parameters = lambda method, values: window._fusion_parameters_by_method.__setitem__(
        method, dict(values))
    window.resource_preference = "balanced"
    window.restore_last_stack = False
    window.ask_load_scale = True
    window.saved = []
    window.persist_settings = lambda: window.saved.append(window.tile_block_size)
    window.update_manager = SimpleNamespace(auto_download=True)
    window.update_manager.set_auto_download = lambda value: setattr(
        window.update_manager, "auto_download", value)
    yield window
    window.deleteLater()
    qapp.processEvents()


def test_settings_overlap_is_visible_and_cancel_is_a_draft(settings_window):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    dialog = AdvancedSettingsDialog(settings_window)
    dialog.spin_block.setValue(256)
    assert dialog.spin_overlap.maximum() == 128
    assert dialog.spin_overlap.value() == 128
    dialog.chk_auto_update.setChecked(False)
    dialog.reject()
    assert settings_window.tile_block_size == 1024
    assert settings_window.tile_overlap == 256
    assert settings_window.update_manager.auto_download
    assert not settings_window.saved
    dialog.deleteLater()


def test_settings_submit_commits_visible_values_and_update_preference(settings_window):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    dialog = AdvancedSettingsDialog(settings_window)
    dialog.spin_block.setValue(256)
    dialog.chk_auto_update.setChecked(False)
    dialog._apply()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert settings_window.tile_block_size == 256
    assert settings_window.tile_overlap == 128
    assert settings_window.update_manager.auto_download is False
    assert settings_window.saved == [256]
    dialog.deleteLater()


def test_resource_mode_is_a_visible_draft(settings_window):
    """Resource mode commits like every other field and defaults to balanced."""
    from dialogs.advanced_settings import AdvancedSettingsDialog
    dialog = AdvancedSettingsDialog(settings_window)
    assert dialog.combo_resources.currentData() == "balanced"
    dialog.combo_resources.setCurrentIndex(
        dialog.combo_resources.findData("responsive"))
    dialog.reject()
    assert settings_window.resource_preference == "balanced"

    dialog = AdvancedSettingsDialog(settings_window)
    dialog.combo_resources.setCurrentIndex(
        dialog.combo_resources.findData("maximum"))
    dialog._apply()
    assert settings_window.resource_preference == "maximum"
    dialog.deleteLater()


def test_resource_mode_only_changes_concurrency(settings_window):
    """It must never reach the fusion parameters that shape the image."""
    from dialogs.advanced_settings import AdvancedSettingsDialog
    dialog = AdvancedSettingsDialog(settings_window)
    dialog.combo_resources.setCurrentIndex(
        dialog.combo_resources.findData("responsive"))
    dialog.spin_block.setValue(512)
    dialog._apply()
    assert settings_window.resource_preference == "responsive"
    assert settings_window.tile_block_size == 512
    assert settings_window.fusion_parameters_for("stackmffv3") == {"mode": "soft"}
    dialog.deleteLater()


def test_settings_failure_does_not_partially_commit(settings_window):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    dialog = AdvancedSettingsDialog(settings_window)
    dialog.spin_block.setValue(512)
    dialog.chk_auto_update.setChecked(False)

    def fail():
        raise OSError("disk full")

    settings_window.persist_settings = fail
    dialog._apply()
    assert settings_window.tile_block_size == 1024
    assert settings_window.update_manager.auto_download is True
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert "disk full" in dialog.status.text()
    dialog.deleteLater()


def test_settings_invalid_text_prevents_any_commit(settings_window):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    dialog = AdvancedSettingsDialog(settings_window)
    dialog.spin_threads.lineEdit().setText("-1")
    dialog.chk_auto_update.setChecked(False)
    dialog._apply()
    assert settings_window.thread_count == 4
    assert settings_window.update_manager.auto_download
    assert not settings_window.saved
    dialog.deleteLater()


def test_automatic_thread_limit_commits(settings_window):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    dialog = AdvancedSettingsDialog(settings_window)
    dialog.spin_threads.setValue(0)
    dialog._apply()
    assert settings_window.thread_count == 0
    assert settings_window.saved
    dialog.deleteLater()


@pytest.fixture
def dust_window(qapp):
    window = QWidget()
    window.raw_images = [np.full((8, 12, 3), 100, np.uint16),
                         np.full((8, 12, 3), 200, np.uint16)]
    window.base_images = list(window.raw_images)
    window.image_filenames = ["白场.tif", "sample.tif"]
    window.commits = []
    window.apply_flat_field_to_stack = lambda *a, **k: window.commits.append((a, k)) or True
    yield window
    window.deleteLater()
    qapp.processEvents()


def _publish_preview(dialog):
    from dialogs.dustmap import _FlatFieldWorker
    raw, base = dialog._sources()
    worker = _FlatFieldWorker("preview", dialog._revision, raw, base,
                             white=dialog._current_white(), target=0)
    worker.run()
    assert worker.result is not None
    dialog._worker = worker
    dialog._worker_finished()


def test_white_source_and_sigma_changes_invalidate_gain(dust_window):
    from dialogs.dustmap import DustMapDialog
    dialog = DustMapDialog(dust_window)
    dialog.rb_stack.setChecked(True)
    _publish_preview(dialog)
    assert dialog.btn_apply.isEnabled()
    dialog.combo_stack.setCurrentIndex(1)
    assert dialog._gain is None
    assert not dialog.btn_apply.isEnabled()
    _publish_preview(dialog)
    dialog.spin_sigma.setValue(7)
    assert dialog._gain is None
    assert not dialog.btn_apply.isEnabled()
    assert not dust_window.commits
    dialog.deleteLater()


def test_white_worker_rejects_base_domain_without_mutation(dust_window):
    from dialogs.dustmap import _FlatFieldWorker
    raw = dust_window.raw_images
    base = [np.full((16, 24, 3), 100, np.uint16)]
    worker = _FlatFieldWorker("apply", 1, raw, base,
                             gain=np.ones((8, 12, 3), np.float32))
    worker.run()
    assert worker.result is None
    assert worker.error
    assert all(np.all(frame == value) for frame, value in zip(raw, (100, 200)))
    assert np.all(base[0] == 100)
    worker.deleteLater()


def test_old_white_worker_result_cannot_enable_apply(dust_window):
    from dialogs.dustmap import DustMapDialog, _FlatFieldWorker
    dialog = DustMapDialog(dust_window)
    dialog.rb_stack.setChecked(True)
    raw, base = dialog._sources()
    worker = _FlatFieldWorker("preview", dialog._revision, raw, base, white=raw[0])
    worker.run()
    dialog.combo_stack.setCurrentIndex(1)
    dialog._worker = worker
    dialog._worker_finished()
    assert dialog._gain is None
    assert not dialog.btn_apply.isEnabled()
    dialog.deleteLater()


def test_failed_flatfield_commit_keeps_dialog_and_selected_frame(dust_window):
    from dialogs.dustmap import DustMapDialog, _FlatFieldWorker
    dialog = DustMapDialog(dust_window)
    dialog.rb_stack.setChecked(True)
    dialog.combo_stack.setCurrentIndex(1)
    dialog.chk_remove.setChecked(True)
    _publish_preview(dialog)
    raw, base = dialog._sources()
    worker = _FlatFieldWorker("apply", dialog._revision, raw, base, gain=dialog._gain)
    worker.remove_index = 1
    worker.run()
    calls = []
    dust_window.apply_flat_field_to_stack = lambda *a, **kw: calls.append(kw) or False
    dialog._worker = worker
    dialog._worker_finished()
    assert calls[0]["remove_index"] == 1
    assert len(calls[0]["prepared"][0]) == 2
    assert len(dust_window.raw_images) == 2
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.deleteLater()


def test_feedback_copy_matches_visible_log_and_export_failure_is_inline(qapp, monkeypatch, tmp_path):
    from dialogs import feedback
    monkeypatch.setattr(feedback.logging_setup, "session_log_excerpt", lambda *_a: "session only")
    monkeypatch.setattr(feedback.logging_setup, "logs_dir", lambda: tmp_path)
    dialog = feedback.FeedbackDialog(None)
    dialog._copy_log()
    assert qapp.clipboard().text() == dialog.log_view.toPlainText() == "session only"
    monkeypatch.setattr(feedback.QFileDialog, "getSaveFileName", lambda *a: (str(tmp_path / "report.zip"), ""))

    def fail(_dest):
        raise OSError("disk full")

    monkeypatch.setattr(feedback.logging_setup, "export_feedback_package", fail)
    dialog._export_package()
    assert "disk full" in dialog.log_hint.text()
    assert "report.zip" in dialog.log_hint.text()
    assert not dialog._export_path
    dialog.deleteLater()


def test_feedback_cancel_does_not_write(qapp, monkeypatch, tmp_path):
    from dialogs import feedback
    monkeypatch.setattr(feedback.logging_setup, "session_log_excerpt", lambda *_a: "log")
    monkeypatch.setattr(feedback.logging_setup, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(feedback.QFileDialog, "getSaveFileName", lambda *a: ("", ""))
    calls = []
    monkeypatch.setattr(feedback.logging_setup, "export_feedback_package", lambda *a: calls.append(a))
    dialog = feedback.FeedbackDialog(None)
    dialog._export_package()
    assert not calls
    dialog.deleteLater()


def test_feedback_failed_replace_preserves_previous_archive(monkeypatch, tmp_path):
    from utils import logging_setup
    monkeypatch.setattr(logging_setup, "session_log_excerpt", lambda *_a: "log")
    dest = tmp_path / "report.zip"
    dest.write_bytes(b"previous report")

    def fail(*args):
        raise OSError("locked destination")

    monkeypatch.setattr(logging_setup.os, "replace", fail)
    with pytest.raises(OSError, match="locked destination"):
        logging_setup.export_feedback_package(dest, include_all_logs=False)
    assert dest.read_bytes() == b"previous report"
    assert list(tmp_path.iterdir()) == [dest]


def test_environment_missing_torch_keeps_classic_methods_available(monkeypatch):
    import sys
    from dialogs.about import EnvironmentInfoDialog
    monkeypatch.setitem(sys.modules, "torch", None)
    text = EnvironmentInfoDialog.environment_report()
    assert "Guided Filter / GFG-FGF / OpenFocus A/B/C: CPU" in text
    assert "StackMFF-V3: CPU" not in text


def test_environment_probe_does_not_block_dialog_and_copy_uses_report(qapp, monkeypatch):
    from dialogs import about
    monkeypatch.setattr(about._EnvironmentProbe, "start", lambda self: None)
    dialog = about.EnvironmentInfoDialog()
    assert not dialog.copy_button.isEnabled()
    dialog._probe.text = "CPU methods available; optional AI missing"
    dialog._probe_finished()
    dialog._copy()
    assert qapp.clipboard().text() == dialog._probe.text
    dialog.deleteLater()


def test_welcome_theme_updates_existing_hint(qapp):
    from dialogs.welcome import WelcomeDialog
    from utils.design_tokens import palette
    window = SimpleNamespace(ui_theme="dark")
    window.apply_theme = lambda theme: setattr(window, "ui_theme", theme)
    dialog = WelcomeDialog(main_window=window)
    dialog._choose_theme("light")
    assert palette("light")["textMuted"] in dialog._demo_hint.styleSheet()
    dialog.deleteLater()
