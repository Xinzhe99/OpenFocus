"""Display-path invariants: conversions, resize repaint and the wipe cache.

These guard the paths a user feels as "the app is laggy". Each case pins
behaviour that a performance change could quietly break — not the timings
themselves (those live in ``tools/ui_perf_probe.py``).

No ``OpenFocus`` window is constructed here: creating and destroying the
main window without a real event loop aborts the pytest process (see the
note in ``tests/test_window_geometry.py``).
"""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QImage  # noqa: E402

from utils.image_utils import cv2_to_pixmap, to_display_uint8  # noqa: E402


def _rgb_array(pixmap):
    """Pixels of a pixmap as an (H, W, 3) RGB array.

    RGBA8888 is used deliberately: Format_RGB32 is 0xffRRGGBB, which on a
    little-endian host lays out as BGRA in memory and would make every
    assertion silently compare BGR. The array is copied out — QImage owns
    the buffer, so a view would dangle and numpy would read freed memory.
    """
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    return np.frombuffer(ptr, np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4)[:, :image.width(), :3].copy()


class TestBitDepthToPixmap:
    def test_bgr_channel_order_survives_the_conversion(self, qapp):
        """QImage.Format_BGR888 must not swap red and blue.

        The fast path feeds OpenCV's buffer straight to QImage; an RGB888
        mistake would silently turn every image's channels inside out.
        """
        bgr = np.zeros((4, 5, 3), np.uint8)
        bgr[..., 0] = 200   # blue
        bgr[..., 2] = 30    # red
        got = _rgb_array(cv2_to_pixmap(bgr))
        assert got.shape[:2] == (4, 5)
        assert tuple(got[0, 0]) == (30, 0, 200)

    @pytest.mark.parametrize(
        "value,expected", [(0, 0), (257, 1), (32768, 128), (65535, 255)])
    def test_uint16_is_scaled_not_truncated(self, qapp, value, expected):
        pixmap = cv2_to_pixmap(np.full((2, 3, 3), value, np.uint16))
        assert tuple(_rgb_array(pixmap)[0, 0]) == (expected,) * 3

    def test_grayscale_stays_grayscale(self, qapp):
        gray = np.full((3, 4), 120, np.uint8)
        got = _rgb_array(cv2_to_pixmap(gray))
        assert got.shape[:2] == (3, 4)
        assert tuple(got[0, 0]) == (120, 120, 120)

    def test_uint8_conversion_does_not_copy(self):
        """Display conversion must be free for the common 8-bit case.

        Every repaint goes through to_display_uint8; copying a 24 MP frame
        here cost 13 ms per repaint on its own.
        """
        img = np.zeros((8, 8, 3), np.uint8)
        assert to_display_uint8(img) is img

    def test_uint16_keeps_the_array_shape(self):
        """convertScaleAbs returns (N,1) for 1-D input; the shape must hold."""
        flat = np.array([0, 257, 65535], np.uint16)
        assert to_display_uint8(flat).shape == flat.shape


class TestMagnifierRepaintsAfterResize:
    def test_composite_matches_the_label_and_is_not_blank(self, qapp):
        """A resize must repaint from the pixmap already held.

        The label rebuilt its composite from the source on every resize
        (a full-image smooth scale); the replacement draws only the visible
        region, so this pins size and content rather than the method.
        """
        from widgets.magnifier_label import MagnifierLabel

        label = MagnifierLabel()
        label.resize(600, 400)
        label.show()
        pixmap = cv2_to_pixmap(np.random.RandomState(0).randint(
            0, 255, (300, 400, 3), dtype=np.uint8))
        label.set_display_pixmap(pixmap)
        qapp.processEvents()
        try:
            for size in ((480, 320), (640, 460), (600, 400)):
                label.resize(*size)
                qapp.processEvents()
                composite = label.pixmap()
                assert composite is not None and not composite.isNull()
                assert (composite.width(), composite.height()) == size
                assert _rgb_array(composite).std() > 1.0
        finally:
            label.close()

    def test_zoom_factor_survives_a_resize(self, qapp):
        from widgets.magnifier_label import MagnifierLabel

        label = MagnifierLabel()
        label.resize(500, 350)
        label.show()
        label.set_display_pixmap(cv2_to_pixmap(
            np.full((200, 300, 3), 60, np.uint8)))
        qapp.processEvents()
        try:
            label._zoom_factor = 2.5
            label._cached_scaled_pixmap = None
            label._composite_key = None
            label._update_scaled_pixmap()
            label.resize(560, 390)
            qapp.processEvents()
            assert label._zoom_factor == 2.5
        finally:
            label.close()


class TestWipePixmapCache:
    """The A/B wipe caches one pixmap per side across slider steps."""

    def _window(self):
        from types import SimpleNamespace
        from controllers.label_manager import LabelManager
        from utils.label_model import Label
        from main import OpenFocus

        window = SimpleNamespace(
            fusion_results=[],
            output_manager=SimpleNamespace(record_for_image=lambda _image: None),
            _bgr_to_pixmap=cv2_to_pixmap)
        window.label_manager = LabelManager(window)
        window.caption = Label.from_dict({
            "target": "registered", "format": "A", "enabled": False,
            "x_location": 5, "y_location": 20, "font_size": 12})
        window.label_manager.labels.add(window.caption)
        window._wipe_pixmap = lambda kind, index, image: \
            OpenFocus._wipe_pixmap(window, kind, index, image)
        return window

    def test_cached_pixmap_tracks_the_frame(self):
        window = self._window()
        image = np.full((40, 60, 3), 80, np.uint8)
        first = window._wipe_pixmap("right", 0, image)
        assert np.all(_rgb_array(first) == 80)

        other = np.full((40, 60, 3), 200, np.uint8)
        assert tuple(_rgb_array(window._wipe_pixmap("right", 0, other))[0, 0]) \
            == (200, 200, 200)

    def test_label_toggle_invalidates_the_cache(self):
        """A cached unlabeled pixmap must not outlive the label toggle."""
        window = self._window()
        image = np.full((40, 60, 3), 80, np.uint8)
        unlabeled = _rgb_array(window._wipe_pixmap("right", 0, image)).copy()

        window.caption.enabled = True
        labeled = _rgb_array(window._wipe_pixmap("right", 0, image))
        assert not np.array_equal(unlabeled, labeled), (
            "labels were enabled but the cached pixmap was reused")

        window.caption.enabled = False
        restored = _rgb_array(window._wipe_pixmap("right", 0, image))
        assert np.array_equal(unlabeled, restored)


    def test_left_side_never_gets_registered_labels(self):
        window = self._window()
        image = np.full((40, 60, 3), 80, np.uint8)
        window.caption.enabled = True
        plain = _rgb_array(window._wipe_pixmap("left", 0, image))
        assert tuple(plain[0, 0]) == (80, 80, 80)


