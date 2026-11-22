import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('PyQt6')


def _publish_preview(shared):
    from core.render_process import _LivePreviewMailbox

    mailbox = _LivePreviewMailbox(shared)
    rgb = np.full((2, 3, 3), 31, np.uint8)
    mailbox.publish(rgb, 1, 4)
    rgb[:] = 63
    mailbox.publish(rgb, 2, 4)
    rgb[:] = 255


def test_preview_mailbox_shares_only_latest_snapshot_with_spawned_child():
    from core.render_process import _LivePreviewMailbox

    context = multiprocessing.get_context('spawn')
    mailbox = _LivePreviewMailbox.create(context)
    assert mailbox.take() is None
    child = context.Process(target=_publish_preview, args=(mailbox.shared,))
    child.start()
    try:
        child.join(15)
        assert child.exitcode == 0
        assert mailbox.take() == (bytes([63]) * 18, 3, 2, 2, 4)
        assert mailbox.take() is None
    finally:
        if child.is_alive():
            child.kill()
            child.join()
        child.close()


def test_preview_mailbox_drops_contention_and_oversized_pixels():
    from core.render_process import _LivePreviewMailbox

    mailbox = _LivePreviewMailbox.create(multiprocessing.get_context('spawn'))
    frame = np.full((2, 3, 3), 31, np.uint8)
    mailbox.publish(frame, 1, 4)
    with mailbox.shared[2]:
        mailbox.publish(frame, 2, 4)
        assert mailbox.take() is None
    mailbox.publish(np.zeros((1, 513, 3), np.uint8), 3, 4)
    assert mailbox.take() == (frame.tobytes(), 3, 2, 1, 4)


@pytest.fixture
def live_view(qapp, monkeypatch):
    from controllers import render_manager
    from tests.test_gui_units import _render_window, _patched_render_manager
    from ui.image_panels import create_result_panel
    from utils import recovery

    frames = [np.full((20, 30, 3), color, np.uint8) for color in (5, 15)]
    previous = np.full_like(frames[0], 63)
    window = _render_window(frames)
    window.set_fusion_method('openfocus_a')
    window.fusion_result = previous
    window.fusion_results = [previous]
    window.output_list.addItem('Previous result')
    window.output_list.setCurrentRow(0)
    window.export_manager = SimpleNamespace(generate_default_filename=lambda **kwargs: 'Render')
    panel = create_result_panel()
    window.live_preview_page = panel.live_preview_page
    window.view_stack = panel.view_stack
    manager, _ = _patched_render_manager(monkeypatch, window)

    def publish_result(label, **kwargs):
        window.fusion_results.insert(0, window.fusion_result)
        window.output_list.insertItem(0, label)

    monkeypatch.setattr(window.output_manager, 'update_output_list_for_fusion', publish_result)
    monkeypatch.setattr(render_manager.QApplication, 'alert', lambda *_: None)
    monkeypatch.setattr(recovery, 'write_snapshot', lambda *_: None)
    yield manager, panel, previous
    manager.dismiss_live_preview()
    panel.widget.deleteLater()


@pytest.mark.parametrize('outcome', ['finish', 'stale', 'cancel', 'error'])
def test_live_canvas_keeps_previews_out_of_output_history(live_view, outcome):
    manager, panel, previous = live_view
    window = manager.window
    page, stack = panel.live_preview_page, panel.view_stack
    normal_page = stack.currentWidget()
    assert manager.start_render() == 'started'
    worker = manager.worker
    assert worker.kwargs['live_preview']
    rgb = bytes([135, 50, 38]) * 6
    frames = iter([None, (rgb, 3, 2, 1, 2), (rgb, 3, 2, 2, 2)])
    worker.take_live_preview = lambda: next(frames, None)
    manager._poll_live_preview()
    assert stack.currentWidget() is normal_page
    manager._poll_live_preview()
    assert stack.currentWidget() is page
    assert page._image.pixelColor(0, 0).getRgb() == (135, 50, 38, 255)
    assert window.fusion_result is previous
    assert window.output_list.count() == 1
    assert window.fusion_results[0] is previous
    assert '1/2' in page.status.text()
    manager._poll_live_preview()
    assert '2/2' in page.status.text()

    final = np.full_like(window.raw_images[0], 222)
    if outcome == 'stale':
        window.source_revision = 1
        manager._poll_live_preview()
        assert stack.currentWidget() is normal_page
    elif outcome == 'cancel':
        assert manager.start_render() == 'busy'
        assert stack.currentWidget() is normal_page
    if outcome == 'error':
        manager.on_render_error('injected render failure')
    else:
        manager.on_render_finished(window.raw_images, final, False, 0, 1, 'CPU')

    assert stack.currentWidget() is normal_page
    assert not page.status.text()
    assert page._image.isNull()
    assert not manager._live_preview_timer.isActive()
    if outcome == 'finish':
        assert manager.last_state == 'done'
        assert window.fusion_result is window.output_manager.displayed_image is final
        assert window.fusion_results[0] is final
        assert window.fusion_results[1] is previous
        assert window.output_list.count() == 2
    else:
        assert manager.last_state == {'stale': 'discarded', 'cancel': 'cancelled', 'error': 'error'}[outcome]
        assert window.fusion_result is previous
        assert window.fusion_results[0] is previous
        assert window.output_list.count() == 1


@pytest.mark.parametrize('mode', ['quick', 'other_method', 'wipe', 'history', 'control'])
def test_preview_is_requested_only_for_ordinary_desktop_fusion(live_view, mode):
    manager, _, _ = live_view
    window = manager.window
    if mode == 'quick':
        window.chk_quick_preview.setChecked(True)
    elif mode == 'other_method':
        window.set_fusion_method('guided_filter')
    elif mode == 'wipe':
        window.wipe_active = True
    elif mode == 'history':
        window.output_list.addItem('Older result')
        window.output_list.setCurrentRow(1)

    assert manager.start_render(interactive=mode != 'control') == 'started'
    assert not manager.worker.kwargs['live_preview']
    assert manager._live_preview_timer is None


def test_switching_to_wipe_dismisses_live_preview(live_view):
    from main import OpenFocus

    manager, panel, _ = live_view
    window = manager.window
    window.set_fusion_method('openfocus_b')
    window.render_manager = manager
    window.wipe_bar = panel.wipe_bar
    window._setup_wipe_sliders = lambda: None
    window._update_wipe_images = lambda: None

    assert manager.start_render() == 'started'
    assert manager.worker.kwargs['live_preview']
    manager.worker.take_live_preview = lambda: (bytes(18), 3, 2, 1, 2)
    manager._poll_live_preview()
    assert panel.view_stack.currentWidget() is panel.live_preview_page

    OpenFocus.set_wipe_mode(window, True)
    assert panel.view_stack.currentIndex() == 1
    assert panel.live_preview_page._image.isNull()
    manager._poll_live_preview()
    assert panel.view_stack.currentIndex() == 1
