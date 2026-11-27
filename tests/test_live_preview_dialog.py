"""Comprehensive UI & integration tests for dialogs.live_preview.LivePreviewDialog."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow

from locales import trans


def _make_synthetic_stack(n: int = 6, h: int = 120, w: int = 160) -> list[np.ndarray]:
    """Create a vivid multi-focus synthetic BGR stack with distinct sharp bands."""
    frames: list[np.ndarray] = []
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.zeros((h, w, 3), dtype=np.uint8)
    # Rich color gradient + fine checkerboard texture across bands.
    base[..., 0] = np.clip((xx * 255) // max(1, w - 1), 0, 255).astype(np.uint8)
    base[..., 1] = np.clip((yy * 255) // max(1, h - 1), 0, 255).astype(np.uint8)
    base[..., 2] = 180
    checker = (((xx // 4) + (yy // 4)) % 2 == 0).astype(np.uint8) * 60
    sharp = np.clip(base.astype(np.int16) + checker[..., None] - 30, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 4.0)

    band_w = max(1, w // n)
    for i in range(n):
        frame = blurred.copy()
        x0 = i * band_w
        x1 = w if i == n - 1 else (i + 1) * band_w
        frame[:, x0:x1] = sharp[:, x0:x1]
        frames.append(frame)
    return frames


class _MockLiveSession:
    """Fast deterministic session double for UI/worker/coalescing tests."""

    def __init__(self, frames: list[np.ndarray], *, render_delay_s: float = 0.0):
        self.frames = [f.copy() for f in frames]
        self.unaligned_frames = [np.roll(f, 4, axis=1) for f in frames]
        self.render_delay_s = float(render_delay_s)
        self.prepared = False
        self.closed = False
        self.render_calls: list[dict] = []
        self.source_requests: list[tuple[int, bool]] = []
        self._lock = threading.Lock()

    @property
    def proxy_size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return (w, h)

    def prepare(self, progress_callback=None, should_cancel=None):
        n = len(self.frames)
        for i in range(n):
            if should_cancel and should_cancel():
                return
            if progress_callback:
                progress_callback(i + 1, n, "prepare")
        self.prepared = True

    def get_source_frame(self, index: int, *, aligned: bool = True) -> np.ndarray:
        with self._lock:
            self.source_requests.append((int(index), bool(aligned)))
        pool = self.frames if aligned else self.unaligned_frames
        return pool[max(0, min(int(index), len(pool) - 1))]

    def render(self, request=None, *, method=None, radius=8, smoothing=4.0, feather=3,
               substack_range=None, should_cancel=None):
        if request is not None:
            method = getattr(request, "method", method)
            radius = getattr(request, "radius", radius)
            smoothing = getattr(request, "smoothing", smoothing)
            feather = getattr(request, "feather", feather)
            substack_range = getattr(request, "substack_range", substack_range)
        if self.render_delay_s > 0:
            time.sleep(self.render_delay_s)
        with self._lock:
            self.render_calls.append({
                "method": method,
                "radius": int(radius),
                "smoothing": float(smoothing),
                "feather": int(feather),
                "substack_range": substack_range,
            })
        n = len(self.frames)
        start, end = substack_range if substack_range is not None else (0, n - 1)
        out = self.frames[min(start, n - 1)].copy()
        # Encode radius/method slightly into corner pixel for verification.
        out[0, 0, 0] = int(radius) % 256
        out[0, 0, 1] = int(round(smoothing)) % 256
        out[0, 0, 2] = int(feather) % 256
        hist = np.zeros(n, dtype=np.float32)
        active = max(1, end - start + 1)
        for idx in range(start, end + 1):
            hist[idx] = 1.0 / active
        return SimpleNamespace(
            image=out,
            depth_histogram=hist,
            elapsed_ms=12.5,
        )

    def close(self) -> None:
        self.closed = True


def _wait_until(qapp, predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return bool(predicate())


def _make_mock_host(frames: list[np.ndarray], *, with_enqueue: bool = False):
    applied_methods: list[str] = []
    applied_params: list[tuple[str, dict]] = []
    render_started: list[bool] = []
    enqueued_jobs: list[dict] = []
    selection_changed: list[bool] = []
    undo_pushed: list[bool] = []

    host = SimpleNamespace(
        raw_images=list(frames),
        image_filenames=[f"img_{i:02d}.png" for i in range(len(frames))],
        excluded_frames=set(),
        ui_theme="light",
        source_revision=1,
        push_undo=lambda: undo_pushed.append(True),
        set_fusion_method=lambda m: applied_methods.append(m),
        set_fusion_parameters=lambda m, p: applied_params.append((m, dict(p))),
        _show_status_message=lambda *_a, **_k: None,
        source_manager=SimpleNamespace(
            processing_selection_changed=lambda: selection_changed.append(True),
            decorate_source_items=lambda: None,
        ),
        render_manager=SimpleNamespace(
            start_render=lambda: render_started.append(True),
        ),
        _applied_methods=applied_methods,
        _applied_params=applied_params,
        _render_started=render_started,
        _enqueued_jobs=enqueued_jobs,
        _selection_changed=selection_changed,
        _undo_pushed=undo_pushed,
    )
    if with_enqueue:
        # Match OpenFocus.enqueue_render_job's exact keyword-only signature.
        def _enqueue_render_job(
            *,
            method=None,
            align_method=None,
            method_params=None,
            align_params=None,
            frame_indices=None,
            options=None,
            label=None,
            interactive=True,
        ):
            enqueued_jobs.append({
                "method": method,
                "align_method": align_method,
                "method_params": dict(method_params or {}),
                "align_params": align_params,
                "frame_indices": list(frame_indices) if frame_indices is not None else None,
                "options": options,
                "label": label,
                "interactive": interactive,
            })

        host.enqueue_render_job = _enqueue_render_job
    return host


def test_right_panel_has_live_preview_button_next_to_render(qapp):
    from ui.right_panel import create_right_panel

    parts = create_right_panel()
    try:
        assert parts.btn_live_preview is not None
        assert parts.btn_live_preview.property("controlId") == "btn_live_preview"
        assert not parts.btn_live_preview.isEnabled()
        assert parts.btn_live_preview.full_text() == trans.t("btn_live_preview")
        # Both Render and Live Preview share the same parent layout row.
        assert parts.btn_live_preview.parentWidget() is parts.btn_render.parentWidget()
    finally:
        parts.widget.deleteLater()
        qapp.processEvents()


def test_live_preview_i18n_complete_across_all_12_languages():
    from locales.live_preview_i18n import LIVE_PREVIEW_TRANSLATIONS

    assert set(trans.PACK_ORDER) == set(LIVE_PREVIEW_TRANSLATIONS)
    en_keys = set(LIVE_PREVIEW_TRANSLATIONS["en"])
    assert "btn_live_preview" in en_keys
    assert "action_live_preview" in en_keys
    assert "live_preview_section_histogram" in en_keys
    assert "live_preview_substack_range" in en_keys
    assert "live_preview_show_aligned" in en_keys
    for code in trans.PACK_ORDER:
        table = trans.translations[code]
        missing = en_keys - set(table)
        assert not missing, f"{code} missing keys: {sorted(missing)}"


def test_dialog_async_prepare_render_and_histogram(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(6)
    session = _MockLiveSession(frames)
    dlg = LivePreviewDialog(
        None,
        images=frames,
        initial_method="openfocus_b",
        session_factory=lambda _sz: session,
        theme="light",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None and not dlg.loading_overlay.isVisible())
        assert session.prepared
        assert dlg.canvas.has_image()
        assert "OpenFocus - B" in dlg.lbl_perf_badge.text()
        assert "160×120" in dlg.lbl_perf_badge.text()
        assert dlg._last_histogram is not None
        assert len(dlg._last_histogram) == 6
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_method_switching_hides_inapplicable_controls_and_preserves_values(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(5)
    session = _MockLiveSession(frames)
    dlg = LivePreviewDialog(
        None,
        images=frames,
        initial_method="openfocus_b",
        session_factory=lambda _sz: session,
        theme="light",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None)

        # OpenFocus B: Radius (1..50), Smoothing (0..34), Feather (0..10) all visible.
        assert dlg.row_radius.isVisible()
        assert dlg.row_smoothing.isVisible()
        assert dlg.row_feather.isVisible()
        assert dlg.row_smoothing.slider.maximum() == 34

        dlg.row_radius.slider.setValue(15)
        dlg.row_smoothing.slider.setValue(11)
        dlg.row_feather.slider.setValue(6)

        # Switch to OpenFocus A: Radius visible, Smoothing & Feather hidden.
        dlg.method_radios["openfocus_a"].click()
        qapp.processEvents()
        assert dlg.current_method() == "openfocus_a"
        assert dlg.row_radius.isVisible()
        assert not dlg.row_smoothing.isVisible()
        assert not dlg.row_feather.isVisible()

        # Switch to OpenFocus C: Radius & Feather hidden, Smoothing visible (0..14).
        dlg.method_radios["openfocus_c"].click()
        qapp.processEvents()
        assert dlg.current_method() == "openfocus_c"
        assert not dlg.row_radius.isVisible()
        assert dlg.row_smoothing.isVisible()
        assert not dlg.row_feather.isVisible()
        assert dlg.row_smoothing.slider.maximum() == 14

        # Switch back to OpenFocus B: tuned values restored!
        dlg.method_radios["openfocus_b"].click()
        qapp.processEvents()
        assert dlg.row_radius.value() == 15
        assert dlg.row_smoothing.value() == 11
        assert dlg.row_feather.value() == 6
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_latest_wins_slider_coalescing(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(5)
    session = _MockLiveSession(frames, render_delay_s=0.03)
    dlg = LivePreviewDialog(
        None,
        images=frames,
        initial_method="openfocus_b",
        session_factory=lambda _sz: session,
        theme="dark",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None)
        initial_calls = len(session.render_calls)

        # Rapidly burst 25 slider updates.
        for radius in range(10, 35):
            dlg.row_radius.slider.setValue(radius)
        qapp.processEvents()

        # Wait for the final request (radius=34) to land.
        assert _wait_until(
            qapp,
            lambda: session.render_calls and session.render_calls[-1]["radius"] == 34,
            timeout_s=5.0,
        )
        burst_calls = len(session.render_calls) - initial_calls
        assert burst_calls < 15, f"Expected latest-wins coalescing, got {burst_calls} renders for 25 drags"
        assert session.render_calls[-1]["radius"] == 34
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_stack_browser_scrub_show_aligned_and_space_peek(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(6)
    session = _MockLiveSession(frames)
    dlg = LivePreviewDialog(
        None,
        images=frames,
        initial_method="openfocus_b",
        session_factory=lambda _sz: session,
        theme="light",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None)

        # Scrub stack slider -> switches to Source Slice view instantaneously.
        dlg.stack_slider.setValue(3)
        qapp.processEvents()
        assert dlg._view_mode == "source"
        assert not dlg.btn_view_fused.isChecked()
        assert "4 / 6" in dlg.lbl_frame_counter.text()
        assert session.source_requests[-1] == (3, True)

        # Toggle Show Aligned off -> requests unaligned frame immediately.
        dlg.chk_show_aligned.setChecked(False)
        qapp.processEvents()
        assert session.source_requests[-1] == (3, False)

        # 1-click switch back to Live Fused Preview.
        dlg.btn_view_fused.click()
        qapp.processEvents()
        assert dlg._view_mode == "fused"
        assert dlg.btn_view_fused.isChecked()
        assert dlg.canvas._overlay_badge == ""

        # Hold Space to peek source slice, release Space to restore fused preview.
        press = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(dlg, press)
        qapp.processEvents()
        assert dlg._peeking_source is True
        assert dlg.canvas._overlay_badge != ""

        release = QKeyEvent(QKeyEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(dlg, release)
        qapp.processEvents()
        assert dlg._peeking_source is False
        assert dlg._view_mode == "fused"
        assert dlg.canvas._overlay_badge == ""
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_substack_range_slider_histogram_sync_and_reset(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(8)
    session = _MockLiveSession(frames)
    dlg = LivePreviewDialog(
        None,
        images=frames,
        initial_method="openfocus_b",
        session_factory=lambda _sz: session,
        theme="light",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None)
        assert not dlg.btn_reset_substack.isEnabled()
        assert dlg.current_substack_range() == (0, 7)

        # Trim substack via dual-handle slider.
        dlg.substack_slider.set_range(2, 5, emit=True)
        assert _wait_until(
            qapp,
            lambda: session.render_calls and session.render_calls[-1]["substack_range"] == (2, 5),
        )
        assert dlg.current_substack_range() == (2, 5)
        assert dlg.btn_reset_substack.isEnabled()
        assert "3" in dlg.lbl_substack_range.text() and "6" in dlg.lbl_substack_range.text()
        assert dlg.histogram_widget._start_idx == 2
        assert dlg.histogram_widget._end_idx == 5

        # Enforces min span of 2 frames (inclusive diff >= 1).
        dlg.substack_slider.set_range(4, 4, emit=True)
        assert dlg.substack_slider.end_value() - dlg.substack_slider.start_value() >= 1

        # Reset restores full stack range.
        dlg.btn_reset_substack.click()
        assert _wait_until(
            qapp,
            lambda: session.render_calls and session.render_calls[-1]["substack_range"] == (0, 7),
        )
        assert dlg.current_substack_range() == (0, 7)
        assert not dlg.btn_reset_substack.isEnabled()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_theme_switching_light_dark_gray_renders_cleanly(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(6)
    session = _MockLiveSession(frames)
    dlg = LivePreviewDialog(
        None,
        images=frames,
        initial_method="openfocus_b",
        session_factory=lambda _sz: session,
        theme="light",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None)
        light_img = dlg.grab().toImage()

        dlg.apply_theme("dark")
        qapp.processEvents()
        dark_img = dlg.grab().toImage()

        dlg.apply_theme("gray")
        qapp.processEvents()
        gray_img = dlg.grab().toImage()

        assert light_img.pixel(10, 10) != dark_img.pixel(10, 10)
        assert dark_img.pixel(10, 10) != gray_img.pixel(10, 10)
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_apply_parameters_and_substack_sync_to_main_window(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(6)
    host = _make_mock_host(frames, with_enqueue=False)
    session = _MockLiveSession(frames)
    dlg = LivePreviewDialog(
        host,
        images=frames,
        initial_method="openfocus_b",
        session_factory=lambda _sz: session,
        theme="light",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None)
        assert not dlg.btn_enqueue.isVisible()

        dlg.row_radius.slider.setValue(14)
        dlg.row_smoothing.slider.setValue(9)
        dlg.row_feather.slider.setValue(5)
        dlg.substack_slider.set_range(1, 4, emit=True)
        qapp.processEvents()

        dlg.btn_apply.click()
        qapp.processEvents()

        assert host._applied_methods[-1] == "openfocus_b"
        m, params = host._applied_params[-1]
        assert m == "openfocus_b"
        assert params["radius"] == 14
        assert params["smoothing"] == 9.0
        assert params["feather"] == 5
        assert host.excluded_frames == {0, 5}
        assert host._selection_changed == [True]
        assert host._undo_pushed == [True]
        # Apply Parameters keeps dialog open.
        assert dlg.isVisible()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_apply_and_render_closes_and_releases_memory(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(5)
    host = _make_mock_host(frames, with_enqueue=False)
    session = _MockLiveSession(frames)
    dlg = LivePreviewDialog(
        host,
        images=frames,
        initial_method="openfocus_c",
        session_factory=lambda _sz: session,
        theme="light",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None)
        dlg.row_smoothing.slider.setValue(7)
        qapp.processEvents()

        dlg.btn_apply_render.click()
        qapp.processEvents()

        assert host._applied_methods[-1] == "openfocus_c"
        assert host._applied_params[-1][1]["smoothing"] == 7.0
        assert host._render_started == [True]
        assert session.closed is True
        assert not dlg._worker.isRunning()
        assert not dlg.isVisible()
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_enqueue_hook_visible_when_host_supports_queue(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(5)
    host = _make_mock_host(frames, with_enqueue=True)
    session = _MockLiveSession(frames)
    emitted: list[dict] = []
    dlg = LivePreviewDialog(
        host,
        images=frames,
        initial_method="openfocus_b",
        session_factory=lambda _sz: session,
        theme="light",
    )
    dlg.enqueue_requested.connect(emitted.append)
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None)
        assert dlg.btn_enqueue.isVisible()

        dlg.substack_slider.set_range(1, 3, emit=True)
        dlg.btn_enqueue.click()
        qapp.processEvents()

        assert len(emitted) == 1
        assert emitted[0]["method"] == "openfocus_b"
        assert emitted[0]["substack_range"] == (1, 3)
        assert len(host._enqueued_jobs) == 1
        assert host._enqueued_jobs[0]["method"] == "openfocus_b"
        assert host._enqueued_jobs[0]["method_params"] == dlg.current_parameters()
        assert host._enqueued_jobs[0]["frame_indices"] == [1, 2, 3]
        # Enqueue keeps preview dialog open for queueing more variations!
        assert dlg.isVisible()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_close_button_releases_session_memory_immediately(qapp):
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(4)
    session = _MockLiveSession(frames)
    dlg = LivePreviewDialog(
        None,
        images=frames,
        initial_method="openfocus_a",
        session_factory=lambda _sz: session,
        theme="light",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None)
        dlg.btn_close.click()
        qapp.processEvents()
        assert session.closed is True
        assert not dlg._worker.isRunning()
    finally:
        dlg.deleteLater()
        qapp.processEvents()


def test_real_engine_end_to_end_abc_and_substack(qapp):
    """End-to-end integration test against core.live_preview.LivePreviewSession."""
    live_preview_mod = pytest.importorskip("core.live_preview")
    assert hasattr(live_preview_mod, "LivePreviewSession")
    from dialogs.live_preview import LivePreviewDialog

    frames = _make_synthetic_stack(6, h=140, w=200)
    dlg = LivePreviewDialog(
        None,
        images=frames,
        initial_method="openfocus_b",
        alignment_method="none",
        theme="light",
    )
    try:
        dlg.show()
        assert _wait_until(qapp, lambda: dlg._fused_pixmap is not None and not dlg.loading_overlay.isVisible(), timeout_s=15.0)
        assert dlg._last_histogram is not None
        assert len(dlg._last_histogram) == 6
        assert float(np.sum(dlg._last_histogram)) > 0.0

        # Switch to OpenFocus A and C with real engine.
        for mid in ("openfocus_a", "openfocus_c", "openfocus_b"):
            seq_before = dlg._request_seq
            dlg.method_radios[mid].click()
            assert _wait_until(
                qapp,
                lambda sb=seq_before, m=mid: dlg._request_seq > sb and _method_tag(m) in dlg.lbl_perf_badge.text(),
                timeout_s=10.0,
            )

        # Trim substack and verify histogram zeroes outer slices.
        dlg.substack_slider.set_range(1, 4, emit=True)
        assert _wait_until(
            qapp,
            lambda: dlg._last_histogram is not None
            and float(dlg._last_histogram[0]) == 0.0
            and float(dlg._last_histogram[5]) == 0.0
            and float(np.sum(dlg._last_histogram[1:5])) > 0.0,
            timeout_s=10.0,
        )
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def _method_tag(method_id: str) -> str:
    return {
        "openfocus_a": "OpenFocus - A",
        "openfocus_b": "OpenFocus - B",
        "openfocus_c": "OpenFocus - C",
    }[method_id]
