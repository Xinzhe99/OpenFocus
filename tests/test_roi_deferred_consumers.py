"""Real ROI widgets consume bounded background pixels from aligned descriptors."""
import time
from threading import get_ident

import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QLabel, QWidget

from core import source_frames, workers
from utils.linear_dng import LinearDNGMetadata, develop_linear
from utils.image_utils import to_display_uint8


def _until(qapp, predicate):
    deadline = time.monotonic() + 15
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(.005)
    assert predicate()


class _LinearPixels(np.ndarray):
    pass


def _aligned(qapp, dtype, linear):
    metadata = (LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1.0, 1024 / 1667), 'Canon', 'Canon EOS R6') if linear else None)
    originals, sources, threads = [], [], []
    for index in range(2):
        values = [7000 + index * 2000, 11000 + index * 2000, 14000 + index * 2000] if dtype == np.uint16 else [40 + index * 20, 70 + index * 20, 100 + index * 20]
        image = np.full((256, 320, 3), values, dtype)
        if linear:
            image = image.view(_LinearPixels)
            image.linear_metadata = metadata
        originals.append(image)
        def pixels(image):
            threads.append(get_ident())
            return np.array(image, copy=True)
        sources.append(source_frames.process_frame(image, pixels))
    worker = workers.ROIAlignmentWorker(sources, thread_count=1)
    results, errors = [], []
    worker.finished_signal.connect(lambda frames, elapsed: results.append(frames))
    worker.error_signal.connect(errors.append)
    worker.start()
    try:
        _until(qapp, lambda: not worker.isRunning())
        qapp.processEvents()
        assert not errors and len(results) == 1
        assert all(isinstance(frame, source_frames.ProcessedFrame) for frame in results[0])
        assert all(frame.source is source for frame, source in zip(results[0], sources))
        return results[0], originals, metadata, threads
    finally:
        if worker.isRunning():
            worker.cancel()
        assert worker.wait(15000)


@pytest.mark.parametrize('method', ['homography', 'ecc', 'both', 'openfocus'])
def test_roi_worker_matches_selected_registration(qapp, method):
    import cv2
    from core.registration import ImageRegistration

    rng = np.random.default_rng(25)
    first = rng.integers(20, 230, (160, 192, 3), dtype=np.uint8)
    second = cv2.warpAffine(first, np.float32([[1, 0, 2.5], [0, 1, -1.5]]),
                            (192, 160), borderMode=cv2.BORDER_REPLICATE)
    frames = [first, second]
    parameters = {'failure_policy': 'continue'}
    direct = ImageRegistration(method=method, parameters=parameters,
                               downscale_width=128, brightness_reference='last',
                               defer_warps=True).process(frames, thread_count=1)
    worker = workers.ROIAlignmentWorker(frames, reg_downscale_width=128,
                                        thread_count=1, alignment=method,
                                        registration_parameters=parameters,
                                        exposure_match=True)
    results, errors = [], []
    worker.finished_signal.connect(lambda images, elapsed: results.append(images))
    worker.error_signal.connect(errors.append)
    worker.start()
    try:
        _until(qapp, lambda: not worker.isRunning())
        qapp.processEvents()
        assert not errors and len(results) == 1
        for expected, actual in zip(direct, results[0]):
            np.testing.assert_allclose(source_frames.resolve_frame(actual),
                                       source_frames.resolve_frame(expected),
                                       rtol=0, atol=1)
    finally:
        if worker.isRunning():
            worker.cancel()
        assert worker.wait(15000)


def _expected_color(image, metadata):
    pixels = develop_linear(image, metadata) if metadata is not None else image
    return tuple(int(value) for value in to_display_uint8(pixels)[0, 0, ::-1])


@pytest.mark.parametrize('dtype,linear', [(np.uint8, False), (np.uint16, False), (np.uint16, True)])
def test_roi_options_construct_and_select_deferred_frames_without_gui_resolution(qapp, dtype, linear):
    from dialogs.roi import ROIRenderOptionsDialog
    gui_thread = get_ident()
    frames, originals, metadata, threads = _aligned(qapp, dtype, linear)
    parent = QWidget()
    parent.linear_dng_metadata = metadata
    before = len(threads)
    dialog = ROIRenderOptionsDialog(2, parent, frames=frames, roi_size=(80, 70))
    assert len(threads) == before
    assert dialog.frame_container.isHidden()
    try:
        dialog.rb_paste.setChecked(True)
        _until(qapp, lambda: dialog.lbl_preview.pixmap() is not None and not dialog.lbl_preview.pixmap().isNull())
        for index in (0, 1):
            dialog.combo_frames.setCurrentIndex(index)
            _until(qapp, lambda: not dialog._preview_workers)
            pixmap = dialog.lbl_preview.pixmap()
            assert pixmap.width() <= 240 and pixmap.height() <= 120
            color = pixmap.toImage().pixelColor(0, 0)
            assert (color.red(), color.green(), color.blue()) == _expected_color(originals[index], metadata)
            assert str(frames[index].shape[1]) in dialog.lbl_output.text()
            assert str(frames[index].shape[0]) in dialog.lbl_output.text()
        assert threads and all(thread != gui_thread for thread in threads)
        assert frames[0].dtype == dtype
    finally:
        dialog.reject()
        _until(qapp, lambda: not dialog._preview_workers)
        dialog.deleteLater()
        parent.deleteLater()


@pytest.mark.parametrize('dtype,linear', [(np.uint8, False), (np.uint16, False), (np.uint16, True)])
def test_main_roi_display_preserves_aligned_coordinates_and_background_development(qapp, monkeypatch, dtype, linear):
    import main
    frames, originals, metadata, threads = _aligned(qapp, dtype, linear)
    window = _roi_window(frames, metadata)
    errors = []
    monkeypatch.setattr(main, 'show_error_box', lambda *args: errors.append(args))
    gui_thread = get_ident()
    try:
        for index in (0, 1):
            window._display_roi_aligned_image(index)
            _until(qapp, lambda: window.lbl_result_img._base_pixmap is not None)
            pixmap = window.lbl_result_img._base_pixmap
            assert pixmap.width() <= 768 and pixmap.height() <= 768
            size = window.lbl_result_img._source_size
            assert (size.width(), size.height()) == (frames[index].shape[1], frames[index].shape[0])
            color = pixmap.toImage().pixelColor(0, 0)
            assert (color.red(), color.green(), color.blue()) == _expected_color(originals[index], metadata)
            if index == 0:
                window.lbl_result_img.set_roi_rect(QRectF(25, 20, 80, 70))
            else:
                assert window.lbl_result_img.get_roi_rect() == QRectF(25, 20, 80, 70)
        assert not errors
        assert threads and all(thread != gui_thread for thread in threads)
    finally:
        _close_roi_window(qapp, window)


def test_roi_preview_cancel_discards_native_completion_and_releases_source(qapp):
    from threading import Event
    entered, release = Event(), Event()
    pixels = np.full((256, 320, 3), 7000, np.uint16)
    def process(image):
        entered.set()
        assert release.wait(10)
        return image.copy()
    frame = source_frames.process_frame(pixels, process)
    request = ('preview', 0, (0, 0, 320, 256), (238, 118))
    worker = workers.ROIFramePreviewWorker(frame, ('roi', 1), request)
    results, errors = [], []
    worker.ready.connect(lambda *args: results.append(args))
    worker.failed.connect(lambda *args: errors.append(args))
    worker.start()
    try:
        _until(qapp, entered.is_set)
        worker.request_cancel()
        release.set()
        _until(qapp, lambda: not worker.isRunning())
        qapp.processEvents()
        assert not results and not errors
        assert worker.frame is None
    finally:
        worker.request_cancel()
        release.set()
        assert worker.wait(15000)


def _held_previews(count=40):
    from threading import Event
    entered, release = Event(), Event()
    resolved = []
    base = np.zeros((256, 320, 3), np.uint8)
    frames = []
    for index in range(count):
        def process(image, index=index):
            resolved.append(index)
            if index == 0:
                entered.set()
                assert release.wait(15)
            return np.full_like(image, index * 3 + 5)
        frames.append(source_frames.process_frame(base, process))
    return frames, entered, release, resolved


def _roi_window(frames, metadata=None):
    import main
    from widgets.magnifier_label import MagnifierLabel

    class ROIWindow(QWidget):
        _display_roi_aligned_image = main.OpenFocus._display_roi_aligned_image
        _request_roi_region = main.OpenFocus._request_roi_region
        _release_roi_regions = main.OpenFocus._release_roi_regions

        def __init__(self):
            super().__init__()
            self.roi_aligned_images = frames
            self.roi_mode_active = True
            self.linear_dng_metadata = metadata
            self.source_revision = 7
            self._roi_preview_workers = {}
            self._roi_display_key = None
            self._roi_display_serial = 0
            self._roi_regions_connected = False
            self._shutting_down = False
            self.lbl_result_img = MagnifierLabel(parent=self)
            self.lbl_result_info = QLabel(self)
            self.result_control_bar = QWidget(self)

    return ROIWindow()


def _close_roi_window(qapp, window, release=None):
    window._shutting_down = True
    if window._roi_display_key is not None:
        window._release_roi_regions(window._roi_display_key)
    for worker in tuple(window._roi_preview_workers.values()):
        worker.next_request = None
        worker.request_cancel()
    if release is not None:
        release.set()
    _until(qapp, lambda: not window._roi_preview_workers)
    window.deleteLater()


@pytest.mark.parametrize('close', [False, True])
def test_roi_dialog_rapid_selection_retires_before_starting_latest(qapp, monkeypatch, close):
    from dialogs.roi import ROIRenderOptionsDialog
    frames, entered, release, resolved = _held_previews()
    created = []
    original = workers.ROIFramePreviewWorker
    class ObservedWorker(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)
    monkeypatch.setattr(workers, 'ROIFramePreviewWorker', ObservedWorker)
    dialog = ROIRenderOptionsDialog(len(frames), frames=frames)
    try:
        dialog.rb_paste.setChecked(True)
        _until(qapp, entered.is_set)
        for index in range(1, len(frames)):
            dialog.combo_frames.setCurrentIndex(index)
            qapp.processEvents()
            assert len(dialog._preview_workers) == 1
        assert len(created) == 1 and resolved == [0]
        if close:
            dialog.reject()
        release.set()
        _until(qapp, lambda: not dialog._preview_workers)
        if close:
            assert len(created) == 1 and resolved == [0]
        else:
            assert len(created) == 2 and resolved == [0, len(frames) - 1]
            color = dialog.lbl_preview.pixmap().toImage().pixelColor(0, 0)
            assert color.red() == (len(frames) - 1) * 3 + 5
    finally:
        dialog.reject()
        release.set()
        _until(qapp, lambda: not dialog._preview_workers)
        dialog.deleteLater()


@pytest.mark.parametrize('close', [False, True])
def test_main_roi_rapid_selection_coalesces_across_frame_generations(qapp, monkeypatch, close):
    frames, entered, release, resolved = _held_previews()
    window = _roi_window(frames)
    created = []
    original = workers.ROIFramePreviewWorker
    class ObservedWorker(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)
    monkeypatch.setattr(workers, 'ROIFramePreviewWorker', ObservedWorker)
    try:
        window._display_roi_aligned_image(0)
        _until(qapp, entered.is_set)
        for index in range(1, len(frames)):
            window._display_roi_aligned_image(index)
            qapp.processEvents()
            assert len(window._roi_preview_workers) == 1
        assert len(created) == 1 and resolved == [0]
        if close:
            window._shutting_down = True
            window._release_roi_regions(window._roi_display_key)
        release.set()
        _until(qapp, lambda: not window._roi_preview_workers)
        if close:
            assert len(created) == 1 and resolved == [0]
            assert window.lbl_result_img._base_pixmap is None
        else:
            assert len(created) == 2 and resolved == [0, len(frames) - 1]
            color = window.lbl_result_img._base_pixmap.toImage().pixelColor(0, 0)
            assert color.red() == (len(frames) - 1) * 3 + 5
    finally:
        _close_roi_window(qapp, window, release)


@pytest.mark.parametrize('key_name', ['Key_Delete', 'Key_Backspace'])
@pytest.mark.parametrize('redraw', [False, True])
def test_roi_preview_delivery_preserves_live_delete_or_redraw(qapp, key_name, redraw):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    frames, entered, release, _resolved = _held_previews(2)
    window = _roi_window(frames)
    label = window.lbl_result_img
    label.roi_mode = True
    original_rect = QRectF(25, 20, 80, 70)
    replacement_rect = QRectF(55, 40, 90, 60)
    label.set_roi_rect(original_rect)
    try:
        window._display_roi_aligned_image(0)
        _until(qapp, entered.is_set)
        label.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, getattr(Qt.Key, key_name),
                                      Qt.KeyboardModifier.NoModifier))
        assert label.get_roi_rect() is None
        if redraw:
            label.set_roi_rect(replacement_rect)
        release.set()
        _until(qapp, lambda: label._base_pixmap is not None)
        assert label.get_roi_rect() == (replacement_rect if redraw else None)
    finally:
        _close_roi_window(qapp, window, release)


def test_main_roi_region_worker_bound_survives_cross_frame_requests(qapp):
    frames, entered, release, resolved = _held_previews(8)
    window = _roi_window(frames)
    try:
        window._display_roi_aligned_image(0)
        _until(qapp, entered.is_set)
        for index in range(len(frames)):
            if index:
                window._display_roi_aligned_image(index)
            for kind in ('view', 'loupe'):
                window._request_roi_region(window._roi_display_key,
                    (kind, index, (0, 0, 100, 80), (100, 80)))
            qapp.processEvents()
            assert len(window._roi_preview_workers) <= 3
        assert resolved == [0]
        release.set()
        _until(qapp, lambda: not window._roi_preview_workers)
        assert all(index in (0, len(frames) - 1) for index in resolved)
        color = window.lbl_result_img._base_pixmap.toImage().pixelColor(0, 0)
        assert color.red() == (len(frames) - 1) * 3 + 5
    finally:
        _close_roi_window(qapp, window, release)
