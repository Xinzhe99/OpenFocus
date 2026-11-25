"""Regression tests for client maintainability refactors and lifecycle invariants."""
from __future__ import annotations

import os
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QMessageBox, QSlider, QWidget

from controllers.export_manager import ExportManager, _window_unavailable
from controllers.output_manager import OutputManager, OutputRecord
from controllers.source_manager import SourceManager
from utils.export_paths import resolve_output_subfolder
from utils.frame_pixels import (
    block_signals,
    is_qobject_alive,
    safe_disconnect,
    window_unavailable,
)


@pytest.fixture(autouse=True)
def _application(qapp):
    return qapp


# ---------------------------------------------------------------------------
# 1. Shared Qt lifecycle helpers
# ---------------------------------------------------------------------------

class _DummyEmitter(QObject):
    ping = pyqtSignal()


def test_qt_lifecycle_helpers_cover_qobjects_and_test_doubles(qapp):
    assert not is_qobject_alive(None)
    assert is_qobject_alive(SimpleNamespace())

    widget = QWidget()
    assert is_qobject_alive(widget)
    assert not window_unavailable(widget)

    widget._shutting_down = True
    assert window_unavailable(widget)
    assert _window_unavailable(widget)

    sip.delete(widget)
    assert not is_qobject_alive(widget)
    assert window_unavailable(widget)

    plain = SimpleNamespace(_shutting_down=False)
    assert not window_unavailable(plain)
    plain._shutting_down = True
    assert window_unavailable(plain)
    assert window_unavailable(None)


def test_safe_disconnect_and_block_signals(qapp):
    emitter = _DummyEmitter()
    seen = []
    slot = lambda: seen.append(1)
    emitter.ping.connect(slot)
    emitter.ping.emit()
    assert seen == [1]

    safe_disconnect(emitter.ping, slot)
    safe_disconnect(emitter.ping, slot)  # second disconnect is a safe no-op
    safe_disconnect(None, slot)
    emitter.ping.emit()
    assert seen == [1]

    slider = QSlider()
    emitted = []
    slider.valueChanged.connect(emitted.append)
    with pytest.raises(RuntimeError):
        with block_signals(slider):
            slider.setValue(5)
            raise RuntimeError("boom")
    # Signal blocking must be restored even when the body raises.
    assert not slider.signalsBlocked()
    assert emitted == []
    slider.setValue(6)
    assert emitted == [6]

    # Plain Python test double also works without requiring a QObject.
    calls = []

    class FakeWidget:
        def blockSignals(self, flag):
            calls.append(flag)
            return False

    with block_signals(FakeWidget()):
        pass
    assert calls == [True, False]


# ---------------------------------------------------------------------------
# 2. Batch subfolder resolution & BatchWorker / BatchProcessingDialog helpers
# ---------------------------------------------------------------------------

def test_resolve_output_subfolder_enforces_containment(tmp_path):
    source = tmp_path / "stack_a"
    source.mkdir()

    resolved = resolve_output_subfolder(str(source), "OpenFocus_Output")
    assert resolved == os.path.realpath(str(source / "OpenFocus_Output"))

    for bad in ("", "   ", "../escape", os.path.abspath(str(tmp_path / "other"))):
        with pytest.raises(ValueError):
            resolve_output_subfolder(str(source), bad)

    if os.name == "nt":
        other_drive = 'Y:' if source.drive.upper() == 'Z:' else 'Z:'
        with pytest.raises(ValueError):
            resolve_output_subfolder(str(source), other_drive + 'elsewhere')


def test_batch_worker_shared_reference_and_tile_helpers(tmp_path):
    from core.workers import BatchWorker

    source = tmp_path / "stack"
    source.mkdir()
    mask = source / "defect.png"
    flat = source / "flat.png"
    frame = source / "frame_01.png"
    for p in (mask, flat, frame):
        p.write_bytes(b"")

    worker = BatchWorker(
        [str(source)],
        "subfolder",
        "nested_out",
        {
            "defect_mask_path": str(mask),
            "flat_field_path": str(flat),
            "runtime_settings": {"tile_block_size": 512, "tile_enabled": True},
        },
        tile_enabled=True,
        tile_block_size=256,
        tile_overlap=32,
        tile_threshold=1024,
    )
    assert worker._is_correction_reference(str(mask))
    assert worker._is_correction_reference(str(flat))
    assert not worker._is_correction_reference(str(frame))

    of_tiles = worker._resolve_tile_kwargs("openfocus_a")
    assert of_tiles["tile_enabled"] is False
    assert of_tiles["tile_block_size"] == 512

    gff_tiles = worker._resolve_tile_kwargs("gff")
    assert gff_tiles["tile_enabled"] is True
    assert gff_tiles["tile_block_size"] == 512

    out_dir = worker._get_output_path_for_single_folder(str(source))
    assert out_dir == os.path.realpath(str(source / "nested_out"))
    assert os.path.isdir(out_dir)

    worker.output_path = "../outside"
    with pytest.raises(ValueError, match="inside the source folder"):
        worker._get_output_path_for_single_folder(str(source))


def test_batch_dialog_done_cancels_imports_and_stops_sensor_inspection(qapp):
    from dialogs.batch import BatchProcessingDialog

    dialog = BatchProcessingDialog()
    cancelled = []
    stopped = []

    class FakeWorker:
        def cancel(self):
            cancelled.append(True)

    dialog._import_worker = FakeWorker()
    dialog._import_busy = True
    dialog.sensor_cleanup.stop_inspection = lambda: stopped.append(True)

    dialog.done(0)
    assert cancelled == [True]
    assert stopped == [True]
    assert dialog._import_worker is None
    assert not dialog._import_busy
    dialog.deleteLater()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# 3. SourceManager lifecycle, timers & sequence helpers
# ---------------------------------------------------------------------------

def test_source_manager_review_retirement_and_timer_cleanup(qapp, monkeypatch):
    class FakeReviewWorker:
        def __init__(self, frames):
            self.frames = tuple(frames)
            self._cancelled = False
            self.error = None
            self.results = []
            self.deleted = False

        def cancel(self):
            self._cancelled = True

        def deleteLater(self):
            self.deleted = True

    class FakeWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.raw_images = [np.zeros((8, 8, 3), dtype=np.uint8)]
            self.base_images = list(self.raw_images)
            self.image_filenames = ["a.png"]
            self.image_source_paths = ["/tmp/a.png"]
            self.image_source_exifs = [b"exif"]
            self.image_icc_profiles = [b"icc"]
            self.excluded_frames = set()
            self.source_transforms = []
            self.unrecoverable_source_changes = []
            self.file_list = QListWidget(self)
            self.source_images_label = SimpleNamespace(setText=lambda *_: None, setToolTip=lambda *_: None)
            self.label_manager = SimpleNamespace(reset_labels=lambda: None)
            self.transform_manager = SimpleNamespace(
                invalidate_processing_results=lambda **_: None,
                reload_image_stack=lambda **_: True,
            )
            self.status_messages = []

        def _set_review_status(self, text):
            self.status_messages.append(text)

        def _show_status_message(self, text, ms):
            self.status_messages.append(text)

    win = FakeWindow()
    mgr = SourceManager(win)
    assert isinstance(mgr._retiring_review_workers, set)

    # Retiring a worker before _clear_review has ever been called must not raise.
    stray = FakeReviewWorker(win.raw_images)
    mgr._finish_stack_review(stray)
    assert stray.deleted

    # Active review worker + timers are cancelled & stopped on _clear_review().
    active = FakeReviewWorker(win.raw_images)
    mgr._review_worker = active
    mgr._review_status_key = "stack_review_cancelled"
    mgr._review_notice_timer.start(5000)
    mgr._render_prewarm_stack = (1, id(win.raw_images), 0)
    mgr._render_prewarm_timer.start(5000)

    mgr._clear_review()
    assert active._cancelled
    assert mgr._review_worker is None
    assert active in mgr._retiring_review_workers
    assert not mgr._review_notice_timer.isActive()

    # Late finish of retired worker must not overwrite status key.
    mgr._finish_stack_review(active)
    assert active.deleted
    assert active not in mgr._retiring_review_workers
    assert mgr._review_status_key is None

    # Shutdown guard prevents _finish_stack_review and _expire_review_notice from touching widgets.
    win._shutting_down = True
    late = FakeReviewWorker(win.raw_images)
    mgr._review_worker = late
    mgr._review_status_key = "stack_review_cancelled"
    mgr._finish_stack_review(late)
    assert late.deleted
    assert mgr._review_worker is None
    assert mgr._review_status_key is None

    mgr._review_status_key = "stack_review_cancelled"
    mgr._expire_review_notice()
    assert mgr._review_status_key == "stack_review_cancelled"

    # clear_image_stack stops pending prewarm timer.
    win._shutting_down = False
    monkeypatch.setattr("utils.recovery.write_snapshot", lambda _w: None)
    mgr.clear_image_stack()
    assert not mgr._render_prewarm_timer.isActive()
    assert mgr._render_prewarm_stack is None

    # _pop_frame_at keeps all 6 source sequences aligned.
    win.image_filenames = ["a.png", "b.png"]
    win.raw_images = [1, 2]
    win.base_images = [10, 20]
    win.image_source_paths = ["/a", "/b"]
    win.image_source_exifs = [b"1", b"2"]
    win.image_icc_profiles = [b"p1", b"p2"]
    mgr._pop_frame_at(0)
    assert win.image_filenames == ["b.png"]
    assert win.raw_images == [2]
    assert win.base_images == [20]
    assert win.image_source_paths == ["/b"]
    assert win.image_source_exifs == [b"2"]
    assert win.image_icc_profiles == [b"p2"]

    win.deleteLater()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# 4. OutputManager context menu ownership, atomic publish & multi-delete wipe sync
# ---------------------------------------------------------------------------

def test_output_record_snapshot_and_context_menu_ownership(qapp):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    record = OutputRecord.create(img, "Fusion A")
    record.scale_bar_config = {"px_um": 1.0, "nested": {"x": 1}}
    record.watermark_config = {"text": "OF", "nested": {"y": 2}}

    snap = record.snapshot_for_export()
    assert snap is not record
    snap.scale_bar_config["nested"]["x"] = 99
    snap.watermark_config["nested"]["y"] = 88
    assert record.scale_bar_config["nested"]["x"] == 1
    assert record.watermark_config["nested"]["y"] == 2

    window = QWidget()
    window.output_list = QListWidget(window)
    window.fusion_results = [img]
    item = QListWidgetItem("Fusion A")
    window.output_list.addItem(item)
    mgr = OutputManager(window)

    before_actions = set(window.findChildren(QAction))
    menu = mgr._build_output_context_menu(item)
    # Actions must belong to the menu/submenu, NOT accumulate directly on window.
    direct_window_actions = [a for a in window.children() if isinstance(a, QAction)]
    assert direct_window_actions == []
    assert set(window.findChildren(QAction)) != before_actions

    menu.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    assert set(window.findChildren(QAction)) == before_actions

    window.deleteLater()
    qapp.processEvents()


def test_output_manager_atomic_publish_rollback_and_multidelete_wipe_sync(qapp, monkeypatch):
    img_a = np.zeros((12, 12, 3), dtype=np.uint8)
    img_b = np.full((12, 12, 3), 255, dtype=np.uint8)

    class FakeView:
        def __init__(self):
            self._base_pixmap = "initial_pixmap"
            self.text = ""
            self.cleared = False

        def set_display_pixmap(self, pm):
            self._base_pixmap = pm

        def clear(self):
            self.cleared = True

        def setText(self, text):
            self.text = text

    window = QWidget()
    window.output_list = QListWidget(window)
    window.output_label = SimpleNamespace(setText=lambda *_: None)
    window.lbl_result_img = FakeView()
    window.result_control_bar = SimpleNamespace(setVisible=lambda *_: None)
    window.fusion_results = []
    window.fusion_result = None
    window.wipe_active = True
    wipe_refreshes = []
    window.refresh_wipe_controls = lambda: wipe_refreshes.append(window.fusion_result)

    mgr = OutputManager(window)
    rec_a = OutputRecord.create(img_a, "First")
    mgr._insert_output(img_a, "First", rec_a)
    window.fusion_result = img_a
    assert window.output_list.count() == 1

    # Simulated display failure during publish_compare_output rolls back cleanly.
    monkeypatch.setattr(mgr, "display_specific_fusion_result", lambda _img: False)
    rec_b = OutputRecord.create(img_b, "Compare", kind="compare")
    with pytest.raises(RuntimeError):
        mgr.publish_compare_output(rec_b)

    assert window.output_list.count() == 1
    assert window.fusion_result is img_a
    assert window.lbl_result_img._base_pixmap == "initial_pixmap"

    # Multi-delete refreshes wipe controls just like single-item delete.
    monkeypatch.setattr(QMessageBox, "exec", lambda _self: QMessageBox.StandardButton.Yes)
    window.output_list.item(0).setSelected(True)
    mgr.delete_selected_output_images()
    assert window.output_list.count() == 0
    assert window.fusion_result is None
    assert wipe_refreshes == [None]

    window.deleteLater()
    qapp.processEvents()


def test_output_manager_validate_export_path_preserves_one_arg_contract():
    calls = []
    window = SimpleNamespace(
        export_manager=SimpleNamespace(_safe_export_path=lambda path: calls.append((path,)) or path)
    )
    mgr = OutputManager(window)
    rec_no_sources = OutputRecord.create(None, "A")
    mgr._validate_export_path("/tmp/out.png", rec_no_sources)
    assert calls == [("/tmp/out.png",)]

    calls2 = []
    window.export_manager = SimpleNamespace(
        _safe_export_path=lambda path, extra=(): calls2.append((path, tuple(extra))) or path
    )
    rec_with_sources = OutputRecord.create(None, "B")
    rec_with_sources.source_paths = ("/tmp/src.png",)
    mgr._validate_export_path("/tmp/out.png", rec_with_sources)
    assert calls2 == [("/tmp/out.png", ("/tmp/src.png",))]


# ---------------------------------------------------------------------------
# 6. core/render_process shared child & parent exchange helpers
# ---------------------------------------------------------------------------

def test_render_process_shared_helpers_preserve_stack_and_poll_idle(monkeypatch):
    from core import render_process
    from utils import diagnostics, logging_setup

    monkeypatch.setattr(logging_setup, 'setup_logging', lambda: None)
    monkeypatch.setattr(diagnostics, 'install_exception_hooks', lambda: None)
    monkeypatch.setattr(render_process.trans, 'current_lang', 'en')

    class FakeConn:
        def __init__(self, messages=()):
            self.sent = []
            self._incoming = list(messages)
            self.closed = False

        def send(self, msg):
            self.sent.append(msg)

        def poll(self, _timeout=0):
            return bool(self._incoming)

        def recv(self):
            return self._incoming.pop(0)

        def close(self):
            self.closed = True

    # _render_child still appears at stack[0] on setup error.
    conn = FakeConn()
    render_process._render_child(conn, Event(), ValueError("bad request"), None, None, "en")
    assert conn.closed
    assert conn.sent and conn.sent[-1][0] == "error"
    details = conn.sent[-1][4]
    assert details["error"]["stack"][0]["function"] == "_render_child"

    # _poll_child_exchange stops cleanly on ('idle',) when stop_on_idle=True.
    worker = render_process.RenderProcessWorker(
        [np.zeros((8, 8, 3), dtype=np.uint8)], algorithm=None, need_fusion=False
    )
    exchange = render_process._RenderExchange(worker)

    class AliveProc:
        def is_alive(self):
            return True

    parent = FakeConn([("stage", "fuse"), ("idle",)])
    idle_confirmed = worker._poll_child_exchange(
        AliveProc(), parent, Event(), exchange, stop_on_idle=True
    )
    assert idle_confirmed is True
    assert exchange.last_stage == "fuse"


# ---------------------------------------------------------------------------
# 7. CompareDialog & RetouchDialog lifecycle guards
# ---------------------------------------------------------------------------

def test_compare_dialog_cancels_display_workers_and_guards_finalize(qapp):
    from dialogs.compare import CompareDialog

    cancelled = []

    class FakeDisplayWorker:
        def cancel(self):
            cancelled.append(True)

    dummy = SimpleNamespace(
        _display_workers={"openfocus_a": FakeDisplayWorker(), "gff": FakeDisplayWorker()},
        _results={"openfocus_a": {"key": 1}},
        _full_results={"openfocus_a": {"key": 1}},
        _errors={"openfocus_a": (1, "err")},
        _stale_methods=set(),
    )
    dummy._cancel_display_workers = lambda methods=None: CompareDialog._cancel_display_workers(dummy, methods)

    CompareDialog._clear_compare_results(dummy)
    assert len(cancelled) == 2
    assert dummy._results == {}
    assert dummy._full_results == {}
    assert dummy._errors == {}
    assert dummy._stale_methods == {"openfocus_a"}

    # _source_preview_finished ignores None sender safely.
    no_sender = SimpleNamespace(sender=lambda: None)
    CompareDialog._source_preview_finished(no_sender)


def test_retouch_dialog_reuses_notice_timer_and_guards_post_close_callbacks(qapp):
    from core.render_context import build_render_context
    from dialogs.retouch import RetouchDialog

    frame = np.full((32, 32, 3), 120, dtype=np.uint8)
    result = frame.copy()
    ctx = build_render_context(
        [frame, frame.copy()],
        result,
        filenames=["f1.png", "f2.png"],
        algorithm="openfocus_a",
        alignment_options=(),
        aligned=False,
    )
    record = OutputRecord.create(result, "Test Output", context=ctx)

    dialog = RetouchDialog(
        None,
        record=record,
        apply_callback=lambda _rec, _img: (True, ""),
    )
    try:
        timer_before = dialog._notice_timer
        timers_before = len(dialog.findChildren(QTimer))

        dialog._set_notice("First notice")
        dialog._set_notice("Second notice")
        dialog._set_notice("Third notice")

        # Notice timer must be reused, never allocating new QTimer children.
        assert dialog._notice_timer is timer_before
        assert len(dialog.findChildren(QTimer)) == timers_before
        assert dialog._notice_timer.isActive()

        # Modal-guard helper reflects active blockers cleanly.
        assert not dialog._action_blocked()
        dialog.preview_mode = "before"
        assert dialog._action_blocked()
        assert dialog._action_blocked(allow_stroke=True)
        dialog.preview_mode = None
        assert not dialog._action_blocked()

        # Shutdown stops _notice_timer and post-close callbacks safely no-op even after doc=None.
        dialog._shutdown_sharpness()
        assert not dialog._notice_timer.isActive()
        dialog.doc = None
        dialog._clear_notice()
        dialog._fit()
        dialog.canvas_resized()
        dialog._elide_title()
        dialog._update_status()
    finally:
        dialog._allow_close = True
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
