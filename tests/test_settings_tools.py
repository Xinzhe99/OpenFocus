"""Behavioral boundaries for settings drafts, flat fields, and feedback."""
from types import SimpleNamespace
from pathlib import Path

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

def test_advanced_form_resists_centered_style_and_wraps_long_rows(qapp):
    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtWidgets import (
        QLabel, QLineEdit, QProxyStyle, QStyle, QStyleFactory, QVBoxLayout,
    )
    from dialogs.advanced_settings import _section

    class CenteredFormStyle(QProxyStyle):
        def styleHint(self, hint, option=None, widget=None, returnData=None):
            if hint == QStyle.StyleHint.SH_FormLayoutFormAlignment:
                return (Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop).value
            return super().styleHint(hint, option, widget, returnData)

    host = QWidget()
    style = CenteredFormStyle(QStyleFactory.create("Fusion"))
    style.setParent(host)
    host.setStyle(style)
    form = _section(QVBoxLayout(host), "Settings")
    label = QLabel("A long translated setting")
    label.setFixedWidth(180)
    field = QLineEdit()
    field.setFixedWidth(110)
    form.addRow(label, field)
    try:
        form.setGeometry(QRect(0, 0, 600, 100))
        assert label.geometry().left() < 20
        assert label.geometry().top() <= field.geometry().bottom()
        assert field.geometry().top() <= label.geometry().bottom()
        form.setGeometry(QRect(0, 0, 240, 100))
        assert field.geometry().top() > label.geometry().top()
    finally:
        host.deleteLater()


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


def test_storage_directory_browse_cancel_and_save(settings_window, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from core import frame_storage
    from dialogs import advanced_settings
    from utils import settings_store

    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    store = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, 'get_settings', lambda: store)
    monkeypatch.setattr(advanced_settings.FileDialog, 'getExistingDirectory',
                        lambda *args: str(tmp_path))
    settings_window.persist_settings = lambda: settings_store.save_window_settings(settings_window)

    dialog = advanced_settings.AdvancedSettingsDialog(settings_window)
    dialog._browse_storage_directory()
    dialog.reject()
    assert frame_storage.storage_directory() == frame_storage.tempfile.gettempdir()
    assert not store.contains('rendering/storage_directory')

    dialog = advanced_settings.AdvancedSettingsDialog(settings_window)
    dialog.storage_directory.setText(str(tmp_path))
    dialog._apply()
    assert dialog.result() == QDialog.DialogCode.Accepted
    image = frame_storage.create_frame((2, 2), np.uint8)
    assert Path(image.filename).parent == tmp_path
    restored = SimpleNamespace()
    settings_store.load_window_settings(restored)
    assert restored.storage_directory == str(tmp_path)
    del image

    dialog = advanced_settings.AdvancedSettingsDialog(settings_window)
    dialog.storage_directory.clear()
    dialog._apply()
    assert frame_storage.storage_directory() == frame_storage.tempfile.gettempdir()
    assert store.value('rendering/storage_directory') == ''


def test_invalid_storage_directory_keeps_previous_settings(settings_window, tmp_path, monkeypatch):
    from core import frame_storage
    from dialogs import advanced_settings

    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    previous = tmp_path / 'previous'
    previous.mkdir()
    frame_storage.configure_storage_directory(previous)
    settings_window.storage_directory = str(previous)
    dialog = advanced_settings.AdvancedSettingsDialog(settings_window)
    dialog.storage_directory.setText(str(tmp_path / 'missing'))
    dialog.spin_block.setValue(512)
    dialog._apply()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert frame_storage.storage_directory() == settings_window.storage_directory == str(previous)
    assert settings_window.tile_block_size == 1024
    assert dialog.status.text()
    assert not settings_window.saved


def test_storage_directory_rolls_back_when_save_fails(settings_window, tmp_path, monkeypatch):
    from core import frame_storage
    from dialogs import advanced_settings

    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    previous, requested = tmp_path / 'previous', tmp_path / 'requested'
    previous.mkdir()
    requested.mkdir()
    frame_storage.configure_storage_directory(previous)
    settings_window.storage_directory = str(previous)
    def fail():
        raise OSError('disk full')
    settings_window.persist_settings = fail
    dialog = advanced_settings.AdvancedSettingsDialog(settings_window)
    dialog.storage_directory.setText(str(requested))
    dialog._apply()
    assert frame_storage.storage_directory() == settings_window.storage_directory == str(previous)
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_saved_storage_directory_restores_or_reports_fallback(tmp_path, monkeypatch):
    import main
    from core import frame_storage

    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    window = SimpleNamespace(storage_directory=str(tmp_path))
    assert main._restore_storage_directory(window) is None
    assert frame_storage.storage_directory() == str(tmp_path)
    missing = str(tmp_path / 'missing')
    window.storage_directory = missing
    warning = main._restore_storage_directory(window)
    assert window.storage_directory == ''
    assert frame_storage.storage_directory() == frame_storage.tempfile.gettempdir()
    assert missing in warning


def test_settings_preserve_supported_tile_and_registration_ranges(settings_window):
    """Values the renderer accepts must not be clamped on the way through."""
    settings_window.tile_block_size = 8192
    settings_window.tile_overlap = 4096
    settings_window.tile_threshold = 60000
    settings_window.reg_downscale_width = 8192

    from dialogs.advanced_settings import AdvancedSettingsDialog
    dialog = AdvancedSettingsDialog(settings_window)
    assert dialog.spin_block.value() == 8192
    assert dialog.spin_overlap.value() == 4096
    assert dialog.spin_threshold.value() == 60000
    assert dialog.spin_reg_width.value() == 8192

    dialog._apply()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert settings_window.tile_block_size == 8192
    assert settings_window.tile_overlap == 4096
    assert settings_window.tile_threshold == 60000
    assert settings_window.reg_downscale_width == 8192
    dialog.deleteLater()


def test_tile_controls_and_diagnostics_are_drafts(settings_window):
    from dialogs.advanced_settings import AdvancedSettingsDialog

    dialog = AdvancedSettingsDialog(settings_window)
    dialog.chk_tile.setChecked(False)
    assert dialog.spin_block.isHidden()
    assert dialog.spin_overlap.isHidden()
    assert dialog.spin_threshold.isHidden()
    assert dialog.tile_note.isHidden()
    dialog.chk_tile.setChecked(True)
    assert not dialog.spin_block.isHidden()
    assert not dialog.tile_note.isHidden()
    dialog.chk_collect_alignment_report.setChecked(True)
    dialog.reject()
    assert not getattr(settings_window, 'collect_alignment_report', False)
    dialog.deleteLater()

    dialog = AdvancedSettingsDialog(settings_window)
    dialog.chk_collect_alignment_report.setChecked(True)
    dialog._apply()
    assert settings_window.collect_alignment_report is True
    assert settings_window.saved
    dialog.deleteLater()

def test_alignment_report_opens_from_settings_without_applying_draft(settings_window, qapp):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QPlainTextEdit
    from controllers.render_manager import RenderManager
    from dialogs.advanced_settings import AdvancedSettingsDialog

    settings_window.collect_alignment_report = False
    settings_window.render_manager = RenderManager(settings_window)
    settings_window.render_manager.alignment_diagnostics = (
        [{'frame_index': 0, 'method': 'ecc', 'status': 'aligned'}],
        ['source.tif'], [0])
    settings = AdvancedSettingsDialog(settings_window)
    settings.show()
    settings.chk_collect_alignment_report.setChecked(True)
    shown = []

    def close_report():
        report = QApplication.activeModalWidget()
        text = report.findChild(QPlainTextEdit) if report is not None else None
        shown.append((report.parentWidget() is settings if report else False,
                      text.toPlainText() if text else None))
        if report is not None:
            report.reject()

    try:
        QTimer.singleShot(0, close_report)
        settings.btn_alignment_report.click()
        assert shown and shown[0][0] and 'source.tif' in shown[0][1]
        assert settings.chk_collect_alignment_report.isChecked()
        assert settings_window.collect_alignment_report is False
        assert not settings_window.saved
    finally:
        settings.reject()
        settings.deleteLater()
        qapp.processEvents()


def test_disabled_tiling_ignores_invalid_draft_and_preserves_settings(settings_window):
    from dialogs.advanced_settings import AdvancedSettingsDialog

    dialog = AdvancedSettingsDialog(settings_window)
    dialog.spin_block.setValue(512)
    dialog.chk_tile.setChecked(False)
    dialog.spin_threshold.lineEdit().setText('')
    dialog._apply()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert settings_window.tile_enabled is False
    assert settings_window.tile_block_size == 1024
    assert settings_window.tile_overlap == 256
    assert settings_window.tile_threshold == 2048
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

def test_registration_failure_policy_is_shared_and_persisted(settings_window, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from core.algorithms import resolve_registration_parameters
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from utils import settings_store

    store = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, "get_settings", lambda: store)
    window = settings_window
    window._registration_parameters_by_method = {"openfocus": {"reference_frame": 4}}
    window.current_alignment_method = lambda: "openfocus"
    window.registration_parameters_for = lambda method: resolve_registration_parameters(
        method, window._registration_parameters_by_method.get(method))
    window.persist_settings = lambda: settings_store.save_parameter_memories(window)

    dialog = AdvancedSettingsDialog(window)
    dialog.combo_align_failure.setCurrentIndex(dialog.combo_align_failure.findData("fail"))
    dialog.reject()
    assert window.registration_parameters_for("openfocus")["failure_policy"] == "continue"

    dialog = AdvancedSettingsDialog(window)
    dialog.combo_align_failure.setCurrentIndex(dialog.combo_align_failure.findData("fail"))
    dialog._apply()
    restored = SimpleNamespace()
    settings_store.load_parameter_memories(restored)
    for method in ("homography", "ecc", "openfocus", "both"):
        parameters = resolve_registration_parameters(
            method, restored._registration_parameters_by_method.get(method))
        assert parameters["failure_policy"] == "fail"
    assert restored._registration_parameters_by_method["openfocus"]["reference_frame"] == 4


def test_advanced_settings_preserve_high_saved_thread_limit(settings_window):
    from dialogs.advanced_settings import AdvancedSettingsDialog

    settings_window.thread_count = 300
    dialog = AdvancedSettingsDialog(settings_window)
    dialog._apply()
    assert settings_window.thread_count == 300



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


def test_output_prefix_preview_validation_and_cancel(settings_window):
    from dialogs.advanced_settings import AdvancedSettingsDialog
    settings_window.output_name_prefix = "old"
    dialog = AdvancedSettingsDialog(settings_window)
    dialog.output_prefix.setText("specimen-12")
    assert dialog.output_prefix_preview.text().startswith("specimen-12_OpenFocus_")
    dialog.reject()
    assert settings_window.output_name_prefix == "old"
    assert not settings_window.saved

    dialog = AdvancedSettingsDialog(settings_window)
    dialog.output_prefix.setText("../source")
    dialog.spin_block.setValue(512)
    dialog._apply()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert settings_window.tile_block_size == 1024
    assert settings_window.output_name_prefix == "old"
    assert not settings_window.saved

    dialog.output_prefix.setText("標本_12")
    dialog._apply()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert settings_window.output_name_prefix == "標本_12"
    assert settings_window.saved == [512]
    dialog.deleteLater()


def test_output_prefix_default_and_safe_suggested_name(settings_window):
    from controllers.export_manager import ExportManager
    from utils.export_paths import valid_output_prefix
    manager = ExportManager(settings_window)
    manager._fusion_tag = lambda: "Fusion"
    manager._align_tag = lambda: "Align"
    original = manager.generate_default_filename()
    assert original.startswith("OpenFocus_")
    settings_window.output_name_prefix = "sample"
    name = manager.generate_default_filename()
    assert name.startswith("sample_OpenFocus_")
    assert manager.filename_from_label(name) == name
    settings_window.output_name_prefix = "../bad"
    assert manager.generate_default_filename().startswith("OpenFocus_")
    assert not valid_output_prefix("a" * 41)
    assert not valid_output_prefix("a/b")
    assert not valid_output_prefix("a b")


def test_output_prefix_is_translated_in_every_locale():
    from locales import trans
    for table in trans.translations.values():
        for key in ("output_prefix_section", "output_prefix_label",
                    "output_prefix_preview", "output_prefix_hint"):
            assert table.get(key)


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
    monkeypatch.setattr(feedback.FileDialog, "getSaveFileName", lambda *a: (str(tmp_path / "report.zip"), ""))

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
    monkeypatch.setattr(feedback.FileDialog, "getSaveFileName", lambda *a: ("", ""))
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


def test_environment_missing_runtime_keeps_classic_methods_available(monkeypatch):
    import sys
    from dialogs.about import EnvironmentInfoDialog
    from locales import trans

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    text = EnvironmentInfoDialog.environment_report()
    assert "Guided Filter / GFG-FGF / OpenFocus A/B/C: CPU" in text
    assert f"StackMFF V1/V2/V3: {trans.t('env_not_installed')}" in text
    assert trans.t('env_stackmff_unavailable') in text


@pytest.mark.parametrize("kind,device", [
    ("N/A", "CPU"), ("DIRECTML", "DirectML"), ("COREML", "CoreML"),
])
def test_environment_reports_neural_provider(monkeypatch, kind, device):
    import sys
    from dialogs.about import EnvironmentInfoDialog
    from locales import trans
    import core.multi_focus_fusion as mff

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(__version__="1.0"))
    monkeypatch.setattr(mff, "ensure_neural_device_kind", lambda: kind)
    text = EnvironmentInfoDialog.environment_report()
    assert f"StackMFF V1/V2/V3: {device}" in text
    if kind == "N/A":
        assert trans.t('env_no_gpu') in text
        assert trans.t('env_cpu_mode') in text
    else:
        assert trans.t('env_gpu_accel') in text
        assert trans.t('env_no_gpu') not in text


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


def test_environment_dialog_closes_while_probe_is_running(qapp, monkeypatch):
    import threading
    from dialogs import about

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow_report():
        entered.set()
        release.wait(5)
        finished.set()
        return "CPU"

    monkeypatch.setattr(about.EnvironmentInfoDialog, "environment_report", staticmethod(slow_report))
    dialog = about.EnvironmentInfoDialog()
    dialog.show()
    try:
        assert entered.wait(2)
        dialog.reject()
        assert not dialog.isVisible()
        assert not dialog._probe_timer.isActive()
        from PyQt6 import sip
        from PyQt6.QtCore import QCoreApplication, QEvent
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(dialog)
    finally:
        release.set()
        assert finished.wait(5)
        from PyQt6.QtTest import QTest
        QTest.qWait(50)


def test_welcome_theme_updates_existing_hint(qapp):
    from dialogs.welcome import WelcomeDialog
    from utils.design_tokens import palette
    window = SimpleNamespace(ui_theme="dark")
    window.apply_theme = lambda theme: setattr(window, "ui_theme", theme)
    dialog = WelcomeDialog(main_window=window)
    dialog._choose_theme("light")
    assert palette("light")["textMuted"] in dialog._demo_hint.styleSheet()
    dialog.deleteLater()


@pytest.mark.parametrize("initial", [True, False])
@pytest.mark.parametrize("dismiss", ["reject", "close", "escape"])
def test_usage_settings_are_drafts_until_ok(settings_window, tmp_path, initial, dismiss):
    from PyQt6.QtCore import QSettings, Qt
    from PyQt6.QtTest import QTest
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from utils.usage_analytics import UsageAnalytics, ENABLED_KEY, ID_KEY

    store = QSettings(str(tmp_path / 'usage.ini'), QSettings.Format.IniFormat)
    if not initial:
        store.setValue(ENABLED_KEY, False)
    calls = []
    reporter = UsageAnalytics(settings=store, sender=lambda payload: calls.append(payload) or True)
    settings_window.usage_analytics = reporter
    dialog = AdvancedSettingsDialog(settings_window)
    try:
        assert dialog.chk_usage.isChecked() is initial
        dialog.chk_usage.setChecked(not initial)
        if dismiss == 'escape':
            dialog.show()
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
        else:
            getattr(dialog, dismiss)()
        assert reporter.enabled() is initial
        assert not store.contains(ID_KEY)
        assert not calls
    finally:
        reporter.shutdown()
        dialog.deleteLater()


@pytest.mark.parametrize("initial", [True, False])
def test_usage_ok_persists_without_consent_popup(settings_window, tmp_path, monkeypatch, initial):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QMessageBox
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from utils.usage_analytics import UsageAnalytics, ENABLED_KEY, ID_KEY

    store = QSettings(str(tmp_path / 'usage.ini'), QSettings.Format.IniFormat)
    store.setValue(ENABLED_KEY, initial)
    reporter = UsageAnalytics(settings=store, sender=lambda payload: True)
    settings_window.usage_analytics = reporter
    def unexpected_popup(*args):
        pytest.fail('Advanced Settings must not request additional usage consent')
    monkeypatch.setattr(QMessageBox, 'exec', unexpected_popup)
    dialog = AdvancedSettingsDialog(settings_window)
    try:
        dialog.chk_usage.setChecked(not initial)
        dialog._apply()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert reporter.enabled() is (not initial)
        reopened = QSettings(str(tmp_path / 'usage.ini'), QSettings.Format.IniFormat)
        assert reopened.value(ENABLED_KEY, type=bool) is (not initial)
        assert reopened.contains(ID_KEY) is (not initial)
        next_dialog = AdvancedSettingsDialog(settings_window)
        assert next_dialog.chk_usage.isChecked() is (not initial)
        next_dialog.reject()
        next_dialog.deleteLater()
    finally:
        reporter.shutdown()
        dialog.deleteLater()


def test_failed_settings_apply_does_not_enable_reporting(settings_window, tmp_path):
    from PyQt6.QtCore import QSettings
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from utils.usage_analytics import UsageAnalytics, ENABLED_KEY, ID_KEY

    store = QSettings(str(tmp_path / 'usage.ini'), QSettings.Format.IniFormat)
    store.setValue(ENABLED_KEY, False)
    calls = []
    reporter = UsageAnalytics(settings=store, sender=lambda payload: calls.append(payload) or True)
    settings_window.usage_analytics = reporter
    def fail_save():
        raise OSError('settings unavailable')
    settings_window.persist_settings = fail_save
    dialog = AdvancedSettingsDialog(settings_window)
    try:
        dialog.chk_usage.setChecked(True)
        dialog._apply()
        assert dialog.result() != QDialog.DialogCode.Accepted
        assert not reporter.enabled()
        assert not store.contains(ID_KEY) and not calls
    finally:
        reporter.shutdown()
        dialog.deleteLater()


def test_image_save_dialogs_start_where_the_last_image_was_saved(tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    import utils.settings_store as store
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(store, "get_settings", lambda: settings)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))   # Windows home is USERPROFILE
    assert store.image_save_dir() == str(home)
    saved = tmp_path / "exports"
    saved.mkdir()
    store.remember_image_save(str(saved / "stack.tif"))
    assert store.image_save_dir() == str(saved)
    assert store.image_save_default("next.png") == str(saved / "next.png")
    assert store.image_save_default("/elsewhere/next.png") == str(saved / "next.png")
    assert store.get_last_dialog_dir() == str(saved)
    saved.rmdir()                                       # a vanished folder is not offered
    assert store.image_save_dir() == str(home)
