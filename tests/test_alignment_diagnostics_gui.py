from types import SimpleNamespace

from controllers.render_manager import RenderManager
from locales import trans


def test_alignment_report_retained_after_failure_and_cleared_on_cancel(monkeypatch):
    window = SimpleNamespace(statusBar=lambda: SimpleNamespace(showMessage=lambda *args: None))
    manager = RenderManager(window)
    shown = []
    monkeypatch.setattr('controllers.render_manager.show_message_box',
                        lambda window, title, message, details, icon: shown.append(details))
    manager._restore_ui_controls = lambda: None
    manager._dispose_worker = lambda: None
    manager._hide_progress_bar = lambda: None
    manager._render_context = {'filenames': ['bad.tif'], 'included': [4]}
    manager.worker = SimpleNamespace(alignment_report=[
        {'frame_index': 0, 'method': 'ecc', 'status': 'failed', 'reason': 'ECC failed to converge'}])
    manager.on_render_error('alignment failed')
    assert manager.alignment_diagnostics[0][0]['status'] == 'failed'
    assert manager.alignment_diagnostics[1:] == (['bad.tif'], [4])
    assert len(shown) == 1 and 'bad.tif' in shown[0] and '5' in shown[0]
    assert trans.t('alignment_reason_ecc_failed') in shown[0]
    assert 'bad.tif' not in manager.last_error
    manager.on_render_error('CANCELLED: render cancelled')
    assert not manager.alignment_diagnostics[0]

