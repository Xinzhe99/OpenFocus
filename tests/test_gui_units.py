"""Offscreen GUI tests: settings persistence round-trip and the wipe widget."""
import os
import shutil
import tempfile

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")
QSettings = pytest.importorskip("PyQt6.QtCore").QSettings

from PyQt6.QtWidgets import QApplication  # noqa: E402

TEMP_SETTINGS = os.path.join(tempfile.gettempdir(), "openfocus_pytest_qsettings")


@pytest.fixture(scope="session")
def qapp():
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, TEMP_SETTINGS)
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def clean_settings():
    shutil.rmtree(TEMP_SETTINGS, ignore_errors=True)
    os_makedirs(TEMP_SETTINGS)
    yield
    shutil.rmtree(TEMP_SETTINGS, ignore_errors=True)


def os_makedirs(path):
    import os
    os.makedirs(path, exist_ok=True)


def test_settings_round_trip(qapp, clean_settings):
    from utils.settings_store import load_window_settings, save_window_settings, add_recent_file

    class FakeWindow:
        thread_count = 4
        tile_enabled = True
        tile_block_size = 1024
        tile_overlap = 256
        tile_threshold = 2048
        reg_downscale_width = 1024
        stackmffv4_batch_size = 2
        use_gpu = True
        recent_files = []

    w = FakeWindow()
    load_window_settings(w)  # defaults on empty store
    assert w.thread_count == 4 and w.use_gpu is True

    w.thread_count = 11
    w.use_gpu = False
    w.tile_block_size = 640
    demo = r"C:\stacks\demo"
    add_recent_file(w, demo)
    add_recent_file(w, demo)  # dedupe
    save_window_settings(w)

    w2 = FakeWindow()
    load_window_settings(w2)
    assert w2.thread_count == 11
    assert w2.use_gpu is False
    assert w2.tile_block_size == 640
    # add_recent_file stores the abspath (form differs across platforms)
    assert w2.recent_files == [os.path.abspath(demo)]


def test_recent_files_cap(qapp, clean_settings):
    from utils.settings_store import load_window_settings, add_recent_file, MAX_RECENT_FILES

    class FakeWindow:
        recent_files = []

    w = FakeWindow()
    load_window_settings(w)
    for i in range(MAX_RECENT_FILES + 3):
        add_recent_file(w, rf"C:\stacks\s{i}")
    assert len(w.recent_files) == MAX_RECENT_FILES


def test_wipe_widget_paint_and_interactions(qapp):
    from PyQt6.QtGui import QColor, QPixmap
    from widgets.wipe_compare import WipeCompareWidget

    widget = WipeCompareWidget()
    widget.resize(200, 100)

    left = QPixmap(120, 60)
    left.fill(QColor(200, 30, 30))
    right = QPixmap(120, 60)
    right.fill(QColor(30, 30, 200))
    widget.set_images(left, right, "A", "B")
    assert widget.has_images()

    widget.grab()  # must not raise

    # Divider drag: press on the divider (x=100 of a 200px widget), drag far
    # right; the ratio must clamp to 0.98
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QEvent
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(100, 50),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    widget.mousePressEvent(press)
    move = QMouseEvent(QEvent.Type.MouseMove, QPointF(500, 50),
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)
    widget.mouseMoveEvent(move)
    widget.mouseReleaseEvent(press)
    assert widget._divider_ratio == pytest.approx(0.98)

    widget.reset_view()
    assert widget._divider_ratio == 0.5 and widget._zoom == 1.0
    widget.clear()
    assert not widget.has_images()


def test_every_language_defines_every_key(qapp):
    """Non-English packs must cover every English key, not fall back silently.

    The self-update strings added in v1.22-v1.24 shipped English-only, so
    zh/ja/es users saw raw buttons in the update dialog.
    """
    from locales import trans

    english = set(trans.translations['en'])
    assert english, "English translation table is empty"

    for lang, table in trans.translations.items():
        if lang == 'en':
            continue
        missing = english - set(table)
        assert not missing, f"{lang} is missing {len(missing)} keys: {sorted(missing)}"


def test_translated_keys_used_in_code_exist(qapp):
    """A typo'd key renders as its own name in the UI; catch it in CI."""
    import os
    import re

    from locales import trans

    pattern = re.compile(r"""trans\.t\(\s*['"]([A-Za-z0-9_]+)['"]""")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    english = set(trans.translations['en'])
    used = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", "dist", "build", ".git", ".pytest_cache")]
        for name in filenames:
            if name.endswith(".py"):
                with open(os.path.join(dirpath, name), encoding="utf-8", errors="ignore") as fh:
                    used.update(pattern.findall(fh.read()))

    unknown = sorted(k for k in used if k not in english)
    assert not unknown, f"trans.t() keys with no English entry: {unknown}"


# ---------------------------------------------------------------------------
# Controller regressions (release audit)
#
# ui.styles is imported first on purpose: dialogs <-> ui is a cycle that only
# resolves when the ui package is loaded before the dialogs package.
# ---------------------------------------------------------------------------
import ui.styles  # noqa: E402  (import order, see above)

from PyQt6.QtCore import QRectF  # noqa: E402
from PyQt6.QtGui import QAction  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QCheckBox, QListWidget, QPushButton, QRadioButton, QSlider, QWidget,
)

from locales import trans  # noqa: E402


class _RecordingBox:
    """Stands in for QMessageBox: records instead of blocking on exec()."""

    def __init__(self):
        self.warnings = []

    def warning(self, parent, title, text, *args, **kwargs):
        self.warnings.append((title, text))
        return 0


class _StubStatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, text, timeout=0):
        self.messages.append(str(text))

    def clearMessage(self):
        self.messages.append("")


class _StubOutputManager:
    def show_fusion_result(self):
        pass

    def update_output_list_for_fusion(self):
        pass

    def update_output_count(self):
        pass


class _FakeTransformManager:
    """Only what the source controller calls."""

    def __init__(self):
        self.invalidated = []
        self.reloaded = []

    def invalidate_processing_results(self, clear_output_view=False, preserve_outputs=False):
        self.invalidated.append((clear_output_view, preserve_outputs))

    def reload_image_stack(self, initial_index=0):
        self.reloaded.append(initial_index)


class _StubLabelManager:
    def __init__(self):
        self.resets = 0

    def reset_labels(self):
        self.resets += 1


def _source_window(filenames):
    class FakeWindow:
        pass

    w = FakeWindow()
    w.file_list = QListWidget()
    for name in filenames:
        w.file_list.addItem(name)
    w.image_filenames = list(filenames)
    w.raw_images = []
    w.base_images = []
    w.image_source_paths = []
    w.current_display_index = 0
    w.current_scale_factor = 1.0
    w.current_folder_path = ""
    w.transform_manager = _FakeTransformManager()
    w.label_manager = _StubLabelManager()
    return w


def _render_window(frames):
    class FakeWindow:
        pass

    w = FakeWindow()
    w.raw_images = list(frames)
    w.base_images = list(frames)
    w.image_filenames = [f"f{i}.png" for i in range(len(frames))]
    w.image_source_paths = []
    w.file_list = QListWidget()
    for name in w.image_filenames:
        w.file_list.addItem(name)
    w.aligned_images = []
    w.is_images_aligned = False
    w.last_alignment_options = None
    w.registration_results = []
    w.fusion_result = None
    w.current_result_index = -1
    w.roi_aligned_images = []
    w.roi_mode_active = False
    w.align_cache_enabled = True
    w.current_folder_path = os.path.join(tempfile.gettempdir(), "of-render-stack")
    w.current_scale_factor = 1.0
    w.reg_downscale_width = 1024
    w.thread_count = 2
    w.use_gpu = False
    w.tile_enabled = False
    w.wipe_active = False
    w.btn_render = QPushButton("Render")
    w.btn_reset = QPushButton("Reset")
    w.rb_a = QRadioButton()
    w.rb_a.setChecked(True)
    for attr in ("rb_b", "rb_c", "rb_gfg", "rb_d"):
        setattr(w, attr, QRadioButton())
    w.cb_align_homography = QCheckBox()
    w.cb_align_ecc = QCheckBox()
    w.chk_quick_preview = QCheckBox()
    w.slider_smooth = QSlider()
    w.slider_smooth.setValue(31)
    w.result_slider = QSlider()
    w.result_control_bar = QWidget()
    w.add_label_action = QAction()
    w.output_list = QListWidget()
    w.lbl_result_img = _StubResultView()
    w.output_manager = _StubOutputManager()
    w.transform_manager = _FakeTransformManager()
    w.label_manager = _StubLabelManager()
    w.status = _StubStatusBar()
    w.views_shown = []
    w.statusBar = lambda: w.status
    w.update_result_view = lambda index: w.views_shown.append(index)
    return w


def _fake_worker_class():
    class FakeRenderWorker:
        """Stands in for RenderWorker: records the constructor, no thread."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.is_cancelled = False
            self._running = True
            self.started = 0
            self.finished_signal = _StubSignal()
            self.error_signal = _StubSignal()
            self.progress_signal = _StubSignal()

        def isRunning(self):
            return self._running

        def start(self):
            self.started += 1

    return FakeRenderWorker


class _StubSignal:
    def __init__(self):
        self.slot = None

    def connect(self, slot):
        self.slot = slot


def _patched_render_manager(monkeypatch, window, ai_available=False):
    """RenderManager with a scripted worker and no dialogs: returns (manager,
    the recorded save_aligned calls, the recorded worker stack)."""
    from controllers import render_manager as render_module
    from utils import align_cache

    monkeypatch.setattr(render_module, "RenderWorker", _fake_worker_class())
    monkeypatch.setattr(render_module, "is_stackmffv4_available", lambda: ai_available)
    for name in ("show_custom_message_box", "show_message_box", "show_warning_box"):
        monkeypatch.setattr(render_module, name, lambda *a, **k: None)

    saved = []
    monkeypatch.setattr(
        align_cache, "save_aligned",
        lambda folder, names, images, options, width, scale, shape: saved.append(
            {"filenames": list(names), "images": images, "options": options, "shape": shape}))

    return render_module.RenderManager(window), saved


def _finish_alignment(manager, window, stack):
    """Registration-only result: same count, different (aligned) frames."""
    processed = [frame + 7 for frame in stack]
    manager.on_render_finished(processed, None, True, 1.0, 0.0, "CPU")
    return processed


def test_deleting_a_middle_frame_keeps_names_aligned_with_images(qapp):
    """_build_load_options aliased the name list into the frame list and reset
    scale_factor to 1.0, so one delete dropped two frames and desynced names
    from images."""
    from controllers.source_manager import SourceManager

    folder = os.path.join(tempfile.gettempdir(), "of-stack")
    names = ["f0.png", "f1.png", "f2.png", "f3.png"]
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(4)]

    w = _source_window(names)
    sm = SourceManager(w)
    sm._load_source = (folder, None, None)
    sm._apply_load_options(sm._build_load_options(frames, names, 0.5))

    assert w.current_scale_factor == 0.5
    assert w.raw_images is not w.base_images
    assert [id(x) for x in w.raw_images] == [id(x) for x in w.base_images]
    assert w.image_source_paths == [os.path.join(folder, n) for n in names]

    sm.delete_source_image(w.file_list.item(1))  # delete a middle frame

    assert w.image_filenames == ["f0.png", "f2.png", "f3.png"]
    assert [int(img[0, 0, 0]) for img in w.raw_images] == [0, 2, 3]
    assert [int(img[0, 0, 0]) for img in w.base_images] == [0, 2, 3]
    assert w.image_source_paths == [os.path.join(folder, n)
                                    for n in ("f0.png", "f2.png", "f3.png")]
    assert len(w.image_filenames) == len(w.raw_images) == len(w.base_images)


def test_quick_preview_render_does_not_publish_its_draft(qapp, monkeypatch):
    frames = [np.zeros((60, 1400, 3), dtype=np.uint8) + i for i in range(3)]
    w = _render_window(frames)
    rm, saved = _patched_render_manager(monkeypatch, w)
    w.chk_quick_preview.setChecked(True)
    rm.start_render()

    handed = rm.worker.args[0]
    assert handed is not w.raw_images
    assert max(handed[0].shape[:2]) <= 1200
    assert rm._render_context["source"] is handed
    assert rm._preview_mode is True

    _finish_alignment(rm, w, handed)

    assert w.aligned_images == []
    assert w.is_images_aligned is False
    assert saved == []
    assert rm._preview_mode is False


def test_alignment_published_when_the_stack_is_untouched(qapp, monkeypatch):
    """The same render on the real frames must publish and cache, keyed on the
    source frames it was handed (not on the returned ones)."""
    frames = [np.zeros((30, 40, 3), dtype=np.uint8) + i for i in range(3)]
    w = _render_window(frames)
    rm, saved = _patched_render_manager(monkeypatch, w)
    rm.start_render()

    handed = rm.worker.args[0]
    assert handed is w.raw_images

    processed = _finish_alignment(rm, w, handed)

    assert w.aligned_images == processed
    assert w.is_images_aligned is True
    assert w.last_alignment_options == (False, False)
    assert len(saved) == 1
    assert saved[0]["shape"] == (30, 40)
    assert saved[0]["filenames"] == w.image_filenames


def test_alignment_dropped_when_a_frame_is_deleted_mid_render(qapp, monkeypatch):
    frames = [np.zeros((30, 40, 3), dtype=np.uint8) + i for i in range(3)]
    w = _render_window(frames)
    rm, saved = _patched_render_manager(monkeypatch, w)
    rm.start_render()
    handed = rm.worker.args[0]

    from controllers.source_manager import SourceManager
    SourceManager(w).delete_source_image(w.file_list.item(1))
    assert len(w.raw_images) == 2

    _finish_alignment(rm, w, handed)  # the worker still returns 3 frames

    assert w.aligned_images == []
    assert w.is_images_aligned is False
    assert saved == []


def test_render_started_while_a_render_runs_cancels_it(qapp, monkeypatch):
    frames = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(2)]
    w = _render_window(frames)
    rm, _saved = _patched_render_manager(monkeypatch, w)
    rm.start_render()
    first = rm.worker

    rm.start_render()  # a second click while the first is running cancels it

    assert first.is_cancelled is True
    assert rm.worker is first          # and starts no second thread
    assert w.btn_render.text() == trans.t("btn_cancel_render")


def test_cancelled_render_clears_preview_mode(qapp, monkeypatch):
    frames = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(2)]
    w = _render_window(frames)
    rm, _saved = _patched_render_manager(monkeypatch, w)
    w.chk_quick_preview.setChecked(True)
    rm.start_render()

    rm.on_render_error("CANCELLED: user requested")

    assert rm._preview_mode is False
    assert rm._render_context is None
    assert rm.worker is None
    assert w.btn_render.isEnabled()
    assert w.status.messages[-1] == trans.t("msg_render_cancelled")


def test_compare_all_second_click_cancels_the_run(qapp, monkeypatch):
    """The compare button stayed enabled while btn_render (the only cancel
    affordance) was disabled, so a compare-all could not be stopped."""
    frames = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(3)]
    w = _render_window(frames)
    rm, _saved = _patched_render_manager(monkeypatch, w)

    rm.start_compare_all()
    assert rm._compare_mode is True
    assert len(rm._compare_queue) == 3
    assert rm._compare_current == "guided_filter"
    assert not w.btn_render.isEnabled()
    assert not w.cb_align_homography.isEnabled()

    running = rm.worker
    rm.start_compare_all()  # second click cancels

    assert rm._compare_mode is False
    assert rm._compare_queue == []
    assert rm._compare_current is None
    assert running.is_cancelled is True
    assert w.status.messages[-1] == trans.t("msg_render_cancelled")


def test_compare_all_cancel_between_renders_unlocks_the_ui(qapp, monkeypatch):
    frames = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(3)]
    w = _render_window(frames)
    rm, _saved = _patched_render_manager(monkeypatch, w)
    rm.start_compare_all()
    rm.worker._running = False  # between two queued renders nothing runs

    rm.start_compare_all()

    assert rm._compare_queue == [] and rm._compare_mode is False
    assert w.btn_render.isEnabled()
    assert w.cb_align_homography.isEnabled() and w.slider_smooth.isEnabled()


def test_roi_reset_clears_rect_button_and_state(qapp):
    """The half-reset dropped roi_aligned_images but kept the rectangle, so the
    next render cropped the new stack with the previous stack's ROI."""
    from controllers.transform_manager import TransformManager

    class FakeWindow:
        pass

    w = FakeWindow()
    w.roi_aligned_images = [np.zeros((8, 8, 3), dtype=np.uint8)]
    w.roi_aligned_raw_count = 3
    w.roi_mode_active = True
    w.lbl_result_img = _StubResultView()
    w.lbl_result_img.roi_mode = True
    w.lbl_result_img.set_roi_rect(QRectF(1, 1, 4, 4))
    w.btn_preview_roi = QPushButton("ROI")
    w.btn_preview_roi.setCheckable(True)
    w.btn_preview_roi.setChecked(True)
    toggled = []
    w.btn_preview_roi.toggled.connect(toggled.append)

    tm = TransformManager(w)
    tm.reset_roi_selection()

    assert w.roi_aligned_images == []
    assert w.roi_aligned_raw_count == 0
    assert w.roi_mode_active is False
    assert w.lbl_result_img.roi_mode is False
    assert w.lbl_result_img.get_roi_rect() is None
    assert w.btn_preview_roi.isChecked() is False
    assert toggled == []  # signals blocked: no re-entry into the ROI handler

    tm.reset_roi_selection()  # idempotent: called on every stack mutation
    assert w.roi_aligned_images == [] and toggled == []


class _StubResultView:
    """What the ROI code of TransformManager / RenderManager expects."""

    def __init__(self):
        self._roi_rect = None
        self.roi_mode = False
        self.pixmaps = []
        self.cleared = 0

    def get_roi_rect(self):
        return self._roi_rect

    def set_roi_rect(self, rect):
        self._roi_rect = rect

    def set_display_pixmap(self, pixmap):
        self.pixmaps.append(pixmap)

    def clear(self):
        self.cleared += 1
        self.pixmaps = []


def test_sharpness_curve_follows_the_applied_theme(qapp, monkeypatch):
    """The dark theme is a QSS skin: palette-based detection reported light and
    drew light grey text on a dark background."""
    from ui import styles
    from widgets.sharpness_curve import SharpnessCurveWidget

    curve = SharpnessCurveWidget()
    curve.resize(200, 40)
    # The trap this test exists for: under the dark QSS the palette still says light
    assert curve.palette().color(curve.backgroundRole()).lightness() > 128

    monkeypatch.setattr(styles, "CURRENT_THEME", "dark", raising=False)
    dark = curve._theme_colors()
    monkeypatch.setattr(styles, "CURRENT_THEME", "light", raising=False)
    light = curve._theme_colors()

    assert light[2] != dark[2]
    assert dark[2].lightness() > light[2].lightness()  # text brighter on dark
    assert dark[0].lightness() > light[0].lightness()  # curve stays visible
    assert dark[3].lightness() > light[3].lightness()  # out-of-focus alert
    # the axis keeps the opposite sense: a faint grey rule on white, a mid
    # grey one on the dark background
    assert light[1].lightness() > dark[1].lightness()

    for theme in ("dark", "light"):
        monkeypatch.setattr(styles, "CURRENT_THEME", theme, raising=False)
        curve.set_data([1.0, 0.9, 0.2, 0.8])
        curve.grab()  # must paint in both themes


def test_batch_dialog_checks_the_output_target_before_starting(qapp, monkeypatch, tmp_path):
    """An unwritable target used to fail inside the worker, after the dialog
    had already closed."""
    import PyQt6.QtWidgets as qt_widgets
    from dialogs.batch import BatchProcessingDialog

    box = _RecordingBox()
    monkeypatch.setattr(qt_widgets, "QMessageBox", box)

    stack = tmp_path / "stack"
    stack.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    import cv2
    for i in range(3):
        assert cv2.imwrite(str(src / f"f{i}.png"), np.zeros((20, 20, 3), np.uint8))

    dlg = BatchProcessingDialog()
    dlg.rb_multiple_folders.setChecked(True)
    dlg.folder_paths = [str(src)]
    dlg.rb_subfolder.setChecked(True)
    dlg.subfolder_name.setText("")

    dlg.start_batch_processing()
    assert box.warnings, "a blank subfolder name must be rejected"
    assert dlg.result() != 1

    dlg.subfolder_name.setText("OpenFocus_Output")
    dlg.folder_paths = [str(tmp_path / "deleted")]
    dlg.start_batch_processing()
    assert len(box.warnings) == 2, "a missing source folder must be rejected"

    dlg.folder_paths = [str(src)]
    dlg.rb_custom_folder.setChecked(True)
    dlg.custom_folder_path.setText(str(tmp_path / "nowhere" / "out"))
    dlg.start_batch_processing()
    assert len(box.warnings) == 3, "a custom path under a missing parent must be rejected"

    dlg.custom_folder_path.setText(str(tmp_path / "out"))
    dlg.start_batch_processing()
    assert len(box.warnings) == 3, "a writable custom target must start"
    assert dlg.result() == 1


class _StubResizeDialog:
    """Stands in for DownsampleDialog and records what it was offered."""

    def __init__(self, chosen):
        self.chosen = chosen
        self.offered = []

    def __call__(self, parent=None, initial_scale=1.0, max_scale=1.0):
        self.offered.append((initial_scale, max_scale))
        dialog = self

        def exec():
            return 1

        dialog.exec = exec
        dialog.get_scale_factor = lambda: self.chosen
        return dialog


def _resize_window(base_shape, base_scale, current_scale):
    class FakeWindow:
        pass

    w = FakeWindow()
    w.base_images = [np.zeros(base_shape, dtype=np.uint8) for _ in range(3)]
    for i, img in enumerate(w.base_images):
        img[:, :] = i
    w.raw_images = [cv2.resize(img, (base_shape[1] // 2, base_shape[0] // 2),
                               interpolation=cv2.INTER_AREA) for img in w.base_images]
    w.base_scale_factor = base_scale
    w.current_scale_factor = current_scale
    w.image_filenames = ["f0.png", "f1.png", "f2.png"]
    w.reload_calls = 0
    from controllers.transform_manager import TransformManager
    w.transform_manager = TransformManager(w)
    w.transform_manager._invalidate_processing_results = lambda **kw: None
    w.transform_manager.reload_image_stack = lambda *a, **kw: setattr(
        w, "reload_calls", w.reload_calls + 1)
    return w


def test_resizing_starts_from_the_loaded_base_and_cannot_exceed_it(qapp, monkeypatch):
    """The old code sized and resized from the *current* frames, so a stack
    loaded at 50% and reduced to 25% ended up at 12.5% of the original, and
    asking for 100% silently kept the 50% decode while reporting 100%."""
    from controllers import transform_manager as tm

    # originals were 400x200; base_images hold the 50% decode (200x100)
    w = _resize_window((100, 200, 3), 0.5, 0.5)
    stub = _StubResizeDialog(0.25)
    monkeypatch.setattr(tm, "DownsampleDialog", stub)
    monkeypatch.setattr(tm, "show_success_box", lambda *a, **kw: None)

    w.transform_manager.resize_all_images()

    assert stub.offered == [(0.5, 0.5)], "dialog must offer the base scale as its ceiling"
    assert [img.shape[:2] for img in w.raw_images] == [(50, 100)] * 3
    assert w.current_scale_factor == 0.25
    assert w.reload_calls == 1

    # asking for the ceiling must hand back the base frames, not an upscale
    w2 = _resize_window((100, 200, 3), 0.5, 0.25)
    stub2 = _StubResizeDialog(0.5)
    monkeypatch.setattr(tm, "DownsampleDialog", stub2)
    w2.transform_manager.resize_all_images()
    assert [img.shape[:2] for img in w2.raw_images] == [(100, 200)] * 3
    assert [int(img[0, 0, 0]) for img in w2.raw_images] == [0, 1, 2]


def test_downsample_dialog_respects_max_scale(qapp):
    from dialogs.settings import DownsampleDialog

    dlg = DownsampleDialog(initial_scale=0.25, max_scale=0.5)
    assert dlg.slider.maximum() == 50
    assert dlg.spinbox.maximum() == 50
    assert dlg.get_scale_factor() == 0.25

    over = DownsampleDialog(initial_scale=0.9, max_scale=0.4)
    assert over.get_scale_factor() == 0.4
