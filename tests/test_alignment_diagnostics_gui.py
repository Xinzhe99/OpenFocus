from types import SimpleNamespace

from controllers.render_manager import RenderManager


def test_alignment_report_retained_after_failure_and_cleared_on_cancel():
    window = SimpleNamespace(statusBar=lambda: SimpleNamespace(showMessage=lambda *args: None))
    manager = RenderManager(window)
    manager._render_interactive = False
    manager._restore_ui_controls = lambda: None
    manager._dispose_worker = lambda: None
    manager._hide_progress_bar = lambda: None
    manager._render_context = {'filenames': ['bad.tif'], 'included': [4]}
    manager.worker = SimpleNamespace(alignment_report=[
        {'frame_index': 0, 'method': 'ecc', 'status': 'failed'}])
    manager.on_render_error('alignment failed')
    assert manager.alignment_diagnostics[0][0]['status'] == 'failed'
    assert manager.alignment_diagnostics[1:] == (['bad.tif'], [4])
    manager.on_render_error('CANCELLED: render cancelled')
    assert not manager.alignment_diagnostics[0]

