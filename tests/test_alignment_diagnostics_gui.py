from types import SimpleNamespace

from controllers.render_manager import RenderManager
from locales import trans


def test_alignment_report_maps_selected_frames_and_stages():
    manager = RenderManager(None)
    manager.alignment_diagnostics = (
        [
            {'frame_index': 0, 'method': 'homography', 'status': 'reference'},
            {'frame_index': 1, 'method': 'homography', 'status': 'fallback'},
            {'frame_index': 1, 'method': 'ecc', 'status': 'aligned'},
        ],
        ['first.tif', 'third.tif'], [0, 2],
    )
    old_language = trans.current_lang
    try:
        trans.current_lang = 'en'
        lines = manager.alignment_report_text().splitlines()
        assert lines == [
            'Frame 1 — first.tif: homography — Reference',
            'Frame 3 — third.tif: homography — Fallback',
            'Frame 3 — third.tif: ecc — Aligned',
        ]
        manager.alignment_diagnostics = ([], ['first.tif'], [0])
        assert manager.alignment_report_text() == trans.t('alignment_report_unavailable')
    finally:
        trans.current_lang = old_language


def test_alignment_report_dialog_displays_outcomes(qapp, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QPlainTextEdit, QWidget

    window = QWidget()
    manager = RenderManager(window)
    manager.alignment_diagnostics = (
        [{'frame_index': 0, 'method': 'ecc', 'status': 'fallback'}],
        ['frame.tif'], [2],
    )
    shown = []
    monkeypatch.setattr(QDialog, 'exec', lambda dialog: shown.append(
        dialog.findChild(QPlainTextEdit).toPlainText()))
    manager.show_alignment_report()
    assert len(shown) == 1
    assert 'frame.tif' in shown[0] and 'ecc' in shown[0] and '3' in shown[0]


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


def test_alignment_report_translations_exist_for_every_language():
    keys = ('alignment_report_collect', 'alignment_report_collect_hint',
            'alignment_report_title', 'alignment_report_unavailable',
            'alignment_report_frame', 'alignment_report_reference',
            'alignment_report_aligned', 'alignment_report_fallback',
            'alignment_report_failed', 'alignment_report_cached',
            'alignment_report_available')
    for code in trans.PACK_ORDER:
        pack = trans.translations[code]
        assert all(key in pack and pack[key] for key in keys), code
        assert pack['alignment_report_frame'].format(
            number=2, name='image.tif', method='ecc', status='ok')
