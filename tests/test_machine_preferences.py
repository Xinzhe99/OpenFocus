"""Opening a project or recovery snapshot keeps this machine's thread count.

thread_count describes this computer (Advanced Settings). A project or crash
recovery snapshot saved elsewhere must not replace it, and must not write the
foreign value back to the saved settings. Applying a template stays an explicit
choice (test_project_file.test_template_restores_registration_defaults_and_automatic_threads).
"""
import pytest


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    from main import OpenFocus
    from utils import recovery, settings_store
    settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    settings.clear()
    settings.setValue('ui/first_run_done', '1')
    settings.setValue('updates/last_check_epoch', 9999999999)
    settings.setValue('rendering/thread_count', 0)
    monkeypatch.setattr(settings_store, '_settings_singleton', settings)
    monkeypatch.setattr(recovery, '_base_dir', lambda: str(tmp_path / 'recovery'))
    monkeypatch.setenv('OPENFOCUS_CONTROL_FILE', str(tmp_path / 'control.json'))
    instance = OpenFocus()
    instance.restore_last_stack = False
    yield instance, settings
    instance._force_quit = True
    instance.close()
    instance.deleteLater()
    qapp.processEvents()


def test_project_or_recovery_snapshot_keeps_the_machine_thread_count(window):
    from utils.project_file import _apply_after_load, _collect_state
    instance, settings = window
    assert instance.thread_count == 0
    # The same path runs for project open, crash recovery and the
    # after-update restore: apply_project -> _apply_after_load.
    state = _collect_state(instance)
    state['settings']['thread_count'] = 2
    _apply_after_load(instance, state)
    assert instance.thread_count == 0
    assert int(settings.value('rendering/thread_count')) == 0
