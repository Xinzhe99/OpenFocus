"""Scale-bar tests: metadata detection, nice-length rounding, drawing."""
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
PIL = pytest.importorskip("PIL")


def _make_tiff(path, desc=None, xres=None, resunit=None, arr=None):
    from PIL import Image
    arr = arr if arr is not None else np.zeros((200, 300, 3), np.uint8)
    tiffinfo = {}
    if desc is not None:
        tiffinfo[270] = desc.encode() if isinstance(desc, str) else desc
    if xres is not None:
        tiffinfo[282] = xres
    if resunit is not None:
        tiffinfo[296] = resunit
    Image.fromarray(arr).save(path, format="TIFF", **({"tiffinfo": tiffinfo} if tiffinfo else {}))


class TestDetection:
    @pytest.mark.parametrize("unit", [None, "", "pixel", "unknown"])
    def test_imagej_requires_explicit_physical_unit(self, unit):
        from utils.scalebar import _calibration_from_tags
        desc = "ImageJ=1.11a\n" + ("unit=" + unit if unit is not None else "")
        assert _calibration_from_tags(desc, 2000, 1) is None
        assert _calibration_from_tags(desc, 2000, 3) == 5

    @pytest.mark.parametrize("unit,expected", [("micron", .0005), ("µm", .0005), ("nm", None)])
    def test_imagej_explicit_unit(self, unit, expected):
        from utils.scalebar import _calibration_from_tags
        assert _calibration_from_tags("ImageJ=1.11a\nunit=" + unit, 2000, 1) == expected

    @pytest.mark.parametrize("suffix", [".tif", ".dng"])
    def test_unscaled_imagej_file_is_uncalibrated(self, tmp_path, suffix):
        import tifffile
        from utils.scalebar import detect_px_size_um
        path = tmp_path / ("uncalibrated" + suffix)
        tifffile.imwrite(path, np.zeros((16, 16), np.uint16), imagej=True)
        assert detect_px_size_um(str(path)) is None


    def test_imagej_micron(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "ij.tif")
        _make_tiff(p, desc="ImageJ=1.53c\nimages=3\nunit=micron\n",
                   xres=(2000, 1), resunit=1)
        assert detect_px_size_um(p) == pytest.approx(0.0005)

    def test_ome_tiff(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "ome.tif")
        desc = ('<?xml version="1.0"?><OME><Image><Pixels '
                'PhysicalSizeX="0.325" PhysicalSizeXUnit="µm"/></Pixels>'
                '</Image></OME>')
        _make_tiff(p, desc=desc)
        assert detect_px_size_um(p) == pytest.approx(0.325)

    def test_centimeter_unit(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "cm.tif")
        _make_tiff(p, xres=(2000, 1), resunit=3)
        assert detect_px_size_um(p) == pytest.approx(5.0)

    def test_uncalibrated(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "plain.tif")
        _make_tiff(p)
        assert detect_px_size_um(p) is None

    def test_non_tiff(self, tmp_path):
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "x.png")
        cv2.imwrite(p, np.zeros((10, 10, 3), np.uint8))
        assert detect_px_size_um(p) is None

    def test_linear_dng_centimeter_resolution(self, tmp_path):
        """PIL rejects LinearRaw DNG payloads; tags must still be readable."""
        import tifffile
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "calibrated.dng")
        tifffile.imwrite(p, np.zeros((16, 16, 3), np.uint16), photometric=34892,
                         planarconfig="contig", resolution=(20000, 20000),
                         resolutionunit=3)
        assert detect_px_size_um(p) == pytest.approx(0.5)

    def test_dng_imagej_description(self, tmp_path):
        import tifffile
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "imagej.dng")
        tifffile.imwrite(p, np.zeros((16, 16, 3), np.uint16), photometric=34892,
                         planarconfig="contig", resolution=(2, 2),
                         resolutionunit=2,
                         description="ImageJ=1.53c\nunit=micron\n")
        assert detect_px_size_um(p) == pytest.approx(0.5)

    def test_dng_print_dpi_is_not_calibration(self, tmp_path):
        import tifffile
        from utils.scalebar import detect_px_size_um
        p = str(tmp_path / "camera.dng")
        tifffile.imwrite(p, np.zeros((16, 16, 3), np.uint16),
                         resolution=(300, 300), resolutionunit=2)
        assert detect_px_size_um(p) is None


class TestNiceLength:
    def test_rounding(self):
        from utils.scalebar import nice_bar_um
        assert nice_bar_um(12) == 10
        assert nice_bar_um(37) == 50
        assert nice_bar_um(8) == 10
        assert nice_bar_um(120) == 100
        assert nice_bar_um(0.0) == 1.0

    def test_format(self):
        from utils.scalebar import format_length
        assert format_length(50) == "50 µm"
        assert format_length(1500) == "1.5 mm"
        assert format_length(0.4) == "400 nm"


class TestDrawing:
    def test_bar_pixels_and_label(self):
        from utils.scalebar import draw_scale_bar
        img = np.zeros((1000, 1000, 3), np.uint8)  # black → white bar
        out = draw_scale_bar(img, 0.1, "bottom-right", "white")
        assert out is not img
        # bar_um: nice(1000*0.1*0.12=12) = 10 → bar_px = 100
        corner = out[-80:-20, -130:-20]  # bottom-right region
        assert corner.max() > 200  # white bar/text present

    def test_passthrough_without_calibration(self):
        from utils.scalebar import draw_scale_bar, burn
        img = np.zeros((50, 50, 3), np.uint8)
        assert draw_scale_bar(img, 0) is img
        assert burn(img, None) is img
        assert burn(img, {"px_um": 0, "position": "x", "color": "y"}) is img

    def test_all_positions_and_colors(self):
        from utils.scalebar import draw_scale_bar
        img = np.full((800, 800, 3), 255, np.uint8)  # white → auto black
        for pos in ("bottom-right", "bottom-left", "top-right", "top-left"):
            for col in ("auto", "white", "black"):
                out = draw_scale_bar(img, 0.5, pos, col)
                assert out.shape == img.shape
                # something drawn in the matching quadrant
                q = {
                    "bottom-right": out[500:, 500:],
                    "bottom-left": out[500:, :300],
                    "top-right": out[:300, 500:],
                    "top-left": out[:300, :300],
                }[pos]
                assert q.min() < 128  # dark pixels on white canvas

    def test_label_contains_length(self):
        # font path: Windows has Arial, so µm renders; fallback 'um' also fine
        from utils.scalebar import draw_scale_bar, format_length, nice_bar_um
        img = np.zeros((1000, 1000, 3), np.uint8)
        out = draw_scale_bar(img, 0.1, "bottom-right", "white")
        assert out[-100:, -300:].max() > 200


@pytest.mark.parametrize("shape", [(16, 16), (8, 200), (5, 520), (220, 20)])
def test_tiny_or_extreme_frames_do_not_crash(shape):
    """The tight-frame fallback pushed bar_y1 past bar_y2, so PIL rejected
    the rectangle and every small export failed with ValueError."""
    from utils.scalebar import burn

    img = np.full(shape + (3,), 128, np.uint8)
    out = burn(img, {"px_um": 0.5, "position": "bottom-right", "color": "auto"})
    assert out.shape == img.shape and out.dtype == np.uint8


def test_drawn_length_matches_the_printed_length():
    """A clamped bar must not keep the unclamped physical label.

    The old code forced at least 8 px and printed the length computed before
    clamping, so a 20x20 frame at 1 µm/px showed a "2 µm" bar that was 8 px
    long — a 4x scale error in the exported figure.
    """
    from utils.scalebar import draw_scale_bar_ex

    img = np.full((200, 400, 3), 90, np.uint8)
    _out, info = draw_scale_bar_ex(img, 1.0)
    assert info["too_small"] is False
    assert info["bar_px"] == int(round(info["bar_um"] / 1.0))

    tiny = np.full((20, 20, 3), 90, np.uint8)
    out, info = draw_scale_bar_ex(tiny, 1.0)
    assert info["too_small"] is True
    assert np.array_equal(out, tiny), "a bar that cannot state its scale was drawn"

@pytest.mark.parametrize("position", ["top-left", "top-right", "bottom-left", "bottom-right", "custom"])
def test_scaled_preview_retains_native_bar_length(position):
    from utils.scalebar import draw_scale_bar_ex
    native = np.full((800, 1200, 3), 90, np.uint8)
    preview = np.full((400, 600, 3), 90, np.uint8)
    cfg = dict(px_um=.65, position=position, color="white", custom_x=.4, custom_y=.7)
    _, native_info = draw_scale_bar_ex(native, **cfg)
    scaled, info = draw_scale_bar_ex(preview, **cfg, canvas_size=(1200, 800),
                                     render_scale=.5, region_mean=lambda _: 90)
    assert info["bar_um"] == native_info["bar_um"]
    assert info["bar_px"] == native_info["bar_px"]
    assert np.any(scaled != preview)



def test_sixteen_bit_frame_keeps_style_arguments():
    """The 16-bit branch used to recurse with position/colour only."""
    from utils.scalebar import draw_scale_bar_ex

    img = np.full((240, 400, 3), 40000, np.uint16)
    out, info = draw_scale_bar_ex(
        img, 1.0, "custom", "white", font_size=18, text_align="center",
        custom_x=0.2, custom_y=0.7)
    assert out.dtype == np.uint16
    assert info["too_small"] is False
    # The custom position must have been honoured: the lower-left band is
    # touched while the default bottom-right corner is not.
    assert (out[150:210, 40:200] != img[150:210, 40:200]).any()
    assert np.array_equal(out[-40:, -120:], img[-40:, -120:])

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("position", ["top-left", "bottom-right", "custom"])
def test_grayscale_bar_preserves_layout_and_pixels(dtype, position):
    from utils.scalebar import burn, draw_scale_bar, draw_scale_bar_ex
    image = np.tile(np.linspace(0, np.iinfo(dtype).max, 400, dtype=dtype), (240, 1))
    original = image.copy()
    kwargs = dict(position=position, color="white", custom_x=.2, custom_y=.7)
    drawn = draw_scale_bar(image, .5, **kwargs)
    extended, info = draw_scale_bar_ex(image, .5, **kwargs)
    burned = burn(image, dict(px_um=.5, **kwargs))
    assert drawn.shape == image.shape and drawn.dtype == dtype
    assert not info["too_small"] and np.any(drawn != image)
    np.testing.assert_array_equal(drawn, extended)
    np.testing.assert_array_equal(drawn, burned)
    np.testing.assert_array_equal(image, original)
    color = np.repeat(image[:, :, None], 3, axis=2)
    np.testing.assert_array_equal(drawn, draw_scale_bar(color, .5, **kwargs)[:, :, 0])


def test_existing_lazy_preview_never_resolves_full_source(qapp, monkeypatch):
    from core.source_frames import SourceFrame
    from dialogs.scalebar import ScaleBarDialog
    def no_decode(*args, **kwargs):
        raise AssertionError("full source decoded during fit preview")
    monkeypatch.setattr(SourceFrame, "full_resolution", no_decode)
    preview = np.full((120, 200), 100, np.uint8)
    frame = SourceFrame("unused.tif", 0, (24000, 40000), np.dtype("uint8"), preview, ())
    host = _scale_dialog_host(frame)
    dialog = ScaleBarDialog(host)
    _preview_scale(qapp, dialog)
    assert dialog._ok_button.isEnabled()
    assert max(dialog._preview_cache[1].shape[:2]) <= 1200
    dialog.size_spin.setValue(16)
    _preview_scale(qapp, dialog)
    _close_scale(qapp, dialog)
    host.deleteLater()

def test_processed_preview_worker_retained_until_cancel_finishes(qapp):
    import threading
    from core.source_frames import ProcessedFrame
    from dialogs.scalebar import ScaleBarDialog
    started, release = threading.Event(), threading.Event()
    threads = []
    image = np.zeros((240, 400), np.uint16)
    def process(pixels):
        threads.append(threading.get_ident())
        started.set()
        assert release.wait(10)
        return pixels.copy()
    frame = ProcessedFrame(image, process, image.shape, image.dtype)
    host = _scale_dialog_host(frame)
    dialog = ScaleBarDialog(host)
    try:
        _wait_scale(qapp, started.is_set)
        assert threads[0] != threading.get_ident()
        dialog.reject()
        assert dialog._pixel_worker is not None
        assert not dialog._ok_button.isEnabled()
    finally:
        release.set()
        _wait_scale(qapp, lambda: dialog._pixel_worker is None and dialog._region_worker is None)
        host.deleteLater()

def test_baked_output_is_read_only(qapp):
    from types import SimpleNamespace
    from dialogs.scalebar import ScaleBarDialog
    from utils.image_utils import pixmap_to_cv2
    image = np.full((240, 400, 3), 90, np.uint8)
    cfg = dict(px_um=1.0, mode="manual", manual=1.0, position="bottom-right",
               color="white", font_family="", font_size=0.0, font_bold=False,
               text_align="auto", custom_x=.95, custom_y=.95)
    record = SimpleNamespace(kind="fusion", label="baked", image=image, output_id="B",
                             context=None, scale_bar_config=cfg.copy(),
                             scale_bar_initialized=True, annotations_baked=True)
    host = _scale_dialog_host(image)
    host.scale_bar_target = "output"
    host.output_manager = SimpleNamespace(record_for_row=lambda _: record,
                                          record_for_item=lambda _: record)
    host.output_list = SimpleNamespace(count=lambda: 1, currentItem=lambda: object())
    dialog = ScaleBarDialog(host)
    _preview_scale(qapp, dialog)
    assert not dialog._ok_button.isEnabled()
    np.testing.assert_array_equal(pixmap_to_cv2(dialog.preview_lbl._base_pixmap), image)
    dialog._save_and_accept()
    assert record.scale_bar_config == cfg
    _close_scale(qapp, dialog)
    host.deleteLater()




def test_manual_calibration_is_converted_to_working_pixels():
    """0.5 µm/px loaded at 50% is 1 µm per working pixel."""
    from types import SimpleNamespace
    from utils.scalebar import effective_px_um

    assert effective_px_um(SimpleNamespace(
        scale_um_per_px_manual=0.5, current_scale_factor=0.5,
        source_px_um=None)) == pytest.approx(1.0)
    # Metadata is already expressed for the loaded pixels.
    assert effective_px_um(SimpleNamespace(
        scale_um_per_px_manual=0.0, current_scale_factor=0.5,
        source_px_um=0.8)) == pytest.approx(0.8)

def test_state_preserves_uninitialized_output():
    from types import SimpleNamespace
    from utils.scalebar import capture_state, restore_state
    record = SimpleNamespace(output_id="one", scale_bar_config=None,
                             scale_bar_initialized=False)
    host = SimpleNamespace(output_manager=SimpleNamespace(record_for_row=lambda _: record),
                           output_list=SimpleNamespace(count=lambda: 1))
    saved = capture_state(host)
    record.scale_bar_config = {"px_um": 5}
    record.scale_bar_initialized = True
    restore_state(host, saved)
    assert record.scale_bar_config is None
    assert record.scale_bar_initialized is False



def _scale_dialog_host(image):
    from PyQt6.QtWidgets import QWidget
    host = QWidget()
    host.raw_images = [] if image is None else [image]
    host.fusion_results = []
    host.scale_bar_target = "source"
    host.scale_um_per_px_manual = 0.5
    host.current_scale_factor = 1.0
    host.source_px_um = None
    return host

def test_output_draft_keeps_stack_calibration(qapp):
    from types import SimpleNamespace
    from dialogs.scalebar import ScaleBarDialog
    from utils.scalebar import effective_px_um
    image = np.zeros((240, 400, 3), np.uint8)
    cfg = dict(px_um=2.0, mode="manual", manual=2.0, position="bottom-right",
               color="white", font_family="", font_size=0.0, font_bold=False,
               text_align="auto", custom_x=.95, custom_y=.95)
    record = SimpleNamespace(kind="fusion", label="B", image=image, output_id="B",
                             context=SimpleNamespace(source_px_um=None, source_scale_factor=1.0),
                             scale_bar_config=cfg.copy(), scale_bar_initialized=True,
                             annotations_baked=False)
    host = _scale_dialog_host(image)
    host.current_scale_factor = .5
    host.scale_bar_target = "output"
    host.output_manager = SimpleNamespace(record_for_row=lambda _: record,
                                          record_for_item=lambda _: record)
    host.output_list = SimpleNamespace(count=lambda: 1, currentItem=lambda: object())
    dialog = ScaleBarDialog(host)
    _preview_scale(qapp, dialog)
    dialog.px_spin.setValue(3.0)
    dialog.target_combo.setCurrentIndex(dialog.target_combo.findData("source"))
    assert dialog.manual_value() == .5
    assert dialog.effective_px_um() == 1.0
    dialog.target_combo.setCurrentIndex(dialog.target_combo.findData("output"))
    assert dialog.manual_value() == 3.0
    _preview_scale(qapp, dialog)
    dialog._save_and_accept()
    _wait_scale(qapp, lambda: dialog._pixel_worker is None and dialog._region_worker is None)
    assert effective_px_um(host) == 1.0
    assert record.scale_bar_config["px_um"] == 3.0
    assert record.scale_bar_initialized
    host.deleteLater()

@pytest.mark.parametrize("failure", ["prepare", "persist", "undo"])
def test_scale_commit_failure_restores_state_and_history(qapp, monkeypatch, failure):
    from dialogs.scalebar import ScaleBarDialog
    from utils import recovery, scalebar
    image = np.zeros((240, 400, 3), np.uint8)
    host = _scale_dialog_host(image)
    host._undo_stack, host._redo_stack, host._project_dirty = ["old"], ["redo"], False
    snapshots = []
    monkeypatch.setattr(recovery, "write_snapshot", lambda _: snapshots.append(True))
    attempts = []
    def prepare(previous):
        assert scalebar.capture_state(host) == previous
        if failure == "prepare":
            raise OSError("prepare failed")
        return previous
    def persist():
        attempts.append(True)
        if failure == "persist" and len(attempts) == 1:
            raise OSError("settings failed")
    def undo(_previous, *, prepared):
        assert prepared == _previous
        host._project_dirty = True
        host._undo_stack.append("partial")
        host._redo_stack.clear()
        if failure == "undo":
            raise OSError("undo failed")
    host.persist_settings, host.push_scalebar_undo = persist, undo
    host.prepare_scalebar_undo = prepare
    previous = scalebar.capture_state(host)
    dialog = ScaleBarDialog(host)
    _preview_scale(qapp, dialog)
    dialog.px_spin.setValue(2.0)
    dialog._save_and_accept()
    assert scalebar.capture_state(host) == previous
    assert host._undo_stack == ["old"] and host._redo_stack == ["redo"]
    assert host._project_dirty is False and snapshots == []
    assert dialog._error.text()
    _close_scale(qapp, dialog)
    host.deleteLater()

def test_baked_transition_rejects_pending_output_commit(qapp):
    from types import SimpleNamespace
    from dialogs.scalebar import ScaleBarDialog
    image = np.zeros((240, 400, 3), np.uint8)
    cfg = dict(px_um=1.0, mode="manual", manual=1.0, position="bottom-right",
               color="white", font_family="", font_size=0.0, font_bold=False,
               text_align="auto", custom_x=.95, custom_y=.95)
    record = SimpleNamespace(kind="fusion", label="result", image=image, output_id="B",
                             context=None, scale_bar_config=cfg.copy(),
                             scale_bar_initialized=True, annotations_baked=False)
    host = _scale_dialog_host(image)
    host.scale_bar_target = "output"
    host.output_manager = SimpleNamespace(record_for_row=lambda _: record,
                                          record_for_item=lambda _: record)
    host.output_list = SimpleNamespace(count=lambda: 1, currentItem=lambda: object())
    dialog = ScaleBarDialog(host)
    _preview_scale(qapp, dialog)
    dialog.px_spin.setValue(2.0)
    record.annotations_baked = True
    dialog._save_and_accept()
    assert record.scale_bar_config == cfg
    assert host.scale_um_per_px_manual == .5
    assert dialog._error.text()
    _close_scale(qapp, dialog)
    host.deleteLater()




def _wait_scale(qapp, predicate):
    from PyQt6.QtTest import QTest
    for _ in range(2000):
        qapp.processEvents()
        if predicate():
            return
        QTest.qWait(5)
    pytest.fail("scale bar worker did not finish")


def _close_scale(qapp, dialog):
    dialog.reject()
    _wait_scale(qapp, lambda: dialog._pixel_worker is None and dialog._region_worker is None)


def _preview_scale(qapp, dialog):
    _wait_scale(qapp, lambda: dialog._preview_cache is not None
                and dialog._pixel_worker is None and not dialog._preview_timer.isActive())


@pytest.mark.parametrize("source_px_um, expected_mode", [
    (None, "manual"),
    (0.8, "metadata"),
])
def test_scale_dialog_defaults_to_available_calibration(qapp, source_px_um, expected_mode):
    from dialogs.scalebar import ScaleBarDialog

    host = _scale_dialog_host(np.zeros((240, 400, 3), np.uint8))
    host.scale_um_per_px_manual = 0.0
    host.source_px_um = source_px_um
    dialog = ScaleBarDialog(host)
    assert dialog.calibration_mode() == expected_mode
    assert dialog.effective_px_um() == source_px_um
    assert dialog.px_spin.isEnabled() == (expected_mode == "manual")
    if source_px_um is None:
        assert not dialog._ok_button.isEnabled()
    _close_scale(qapp, dialog)
    host.deleteLater()


def test_scale_dialog_uses_selected_output_metadata_for_default(qapp):
    from types import SimpleNamespace
    from dialogs.scalebar import ScaleBarDialog

    image = np.zeros((240, 400, 3), np.uint8)
    record = SimpleNamespace(kind="fusion", label="result", image=image,
                             output_id="result", context=None, scale_bar_config=None)
    host = _scale_dialog_host(image)
    host.scale_bar_target = "output"
    host.source_px_um = 0.8
    host.scale_um_per_px_manual = 0.0
    host.output_manager = SimpleNamespace(
        record_for_row=lambda row: record, record_for_item=lambda item: record)
    host.output_list = SimpleNamespace(count=lambda: 1, currentItem=lambda: "item")
    dialog = ScaleBarDialog(host)
    assert dialog.calibration_mode() == "manual"
    assert dialog.effective_px_um() is None
    assert not dialog._ok_button.isEnabled()
    _close_scale(qapp, dialog)
    host.deleteLater()


def test_preview_uses_native_geometry_and_uint16_display(qapp):
    from dialogs.scalebar import ScaleBarDialog
    from utils.image_utils import pixmap_to_cv2, to_display_uint8
    from utils.scalebar import draw_scale_bar_ex

    image = np.full((797, 1200, 3), 20000, np.uint16)
    host = _scale_dialog_host(image)
    dialog = ScaleBarDialog(host)
    dialog.resize(960, 640)
    dialog.show()
    _preview_scale(qapp, dialog)
    native, _info = draw_scale_bar_ex(image, **dialog._draw_config())
    assert np.array_equal(pixmap_to_cv2(dialog.preview_lbl._base_pixmap),
                          to_display_uint8(native))
    scale, x, y = dialog.preview_lbl._get_layout_params()
    assert x >= -1 and y >= -1
    assert x + image.shape[1] * scale <= dialog.preview_lbl.width() + 1
    assert y + image.shape[0] * scale <= dialog.preview_lbl.height() + 1
    _close_scale(qapp, dialog)
    host.deleteLater()


def test_preview_develops_native_raw_source_before_drawing(qapp):
    from dialogs.scalebar import ScaleBarDialog
    from utils.linear_dng import LinearDNGMetadata, develop_linear
    from utils.image_utils import pixmap_to_cv2, to_display_uint8
    from utils.scalebar import draw_scale_bar_ex

    metadata = LinearDNGMetadata(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (0.5, 1.0, 0.7), "Maker", "Camera")
    image = np.full((797, 1200, 3), 7000, np.uint16)
    host = _scale_dialog_host(image)
    host.linear_dng_metadata = metadata
    dialog = ScaleBarDialog(host)
    _preview_scale(qapp, dialog)
    expected, _info = draw_scale_bar_ex(develop_linear(image, metadata),
                                        **dialog._draw_config())
    np.testing.assert_array_equal(
        pixmap_to_cv2(dialog.preview_lbl._base_pixmap),
        to_display_uint8(expected))
    _close_scale(qapp, dialog)
    host.deleteLater()


def test_output_preview_uses_record_metadata_not_current_stack(qapp):
    from types import SimpleNamespace
    from dialogs.scalebar import ScaleBarDialog
    from utils.linear_dng import LinearDNGMetadata, develop_linear
    from utils.image_utils import pixmap_to_cv2, to_display_uint8
    from utils.scalebar import draw_scale_bar_ex

    def metadata(model, neutral):
        return LinearDNGMetadata(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            neutral, "Maker", model)

    record_meta = metadata("Historical", (0.5, 1.0, 0.7))
    stack_meta = metadata("Current", (0.8, 1.0, 0.6))
    image = np.full((240, 400, 3), 7000, np.uint16)
    record = SimpleNamespace(kind="fusion", label="historical", image=image,
                             output_id="o1", context=None,
                             scale_bar_config=None,
                             linear_dng_metadata=record_meta,
                             annotations_baked=False)
    host = _scale_dialog_host(image)
    host.scale_bar_target = "output"
    host.linear_dng_metadata = stack_meta
    host.output_manager = SimpleNamespace(
        record_for_row=lambda _row: record,
        record_for_item=lambda _item: record)
    host.output_list = SimpleNamespace(count=lambda: 1,
                                       currentItem=lambda: "item")
    dialog = ScaleBarDialog(host)
    _preview_scale(qapp, dialog)
    expected, _info = draw_scale_bar_ex(develop_linear(image, record_meta),
                                        **dialog._draw_config())
    np.testing.assert_array_equal(
        pixmap_to_cv2(dialog.preview_lbl._base_pixmap),
        to_display_uint8(expected))

    # A historical record without its own metadata must not inherit the
    # currently loaded stack's calibration.
    record.linear_dng_metadata = None
    assert dialog._preview_metadata() is None
    _close_scale(qapp, dialog)
    host.deleteLater()


@pytest.mark.parametrize("image", [None, np.zeros((8, 8, 3), np.uint8)])
def test_invalid_scale_target_cannot_commit(qapp, image):
    from dialogs.scalebar import ScaleBarDialog
    host = _scale_dialog_host(image)
    dialog = ScaleBarDialog(host)
    assert not dialog._ok_button.isEnabled()
    dialog._save_and_accept()
    assert not getattr(host, "scale_bar_enabled", False)
    assert host.scale_um_per_px_manual == 0.5
    _close_scale(qapp, dialog)
    host.deleteLater()


def test_scale_draft_cancel_leaves_calibration_and_position(qapp):
    from dialogs.scalebar import ScaleBarDialog
    host = _scale_dialog_host(np.zeros((240, 400, 3), np.uint8))
    host.scale_bar_custom_x = 0.0
    dialog = ScaleBarDialog(host)
    dialog.px_spin.setValue(2.0)
    dialog.custom_x.setValue(100)
    _close_scale(qapp, dialog)
    assert host.scale_um_per_px_manual == 0.5
    assert host.scale_bar_custom_x == 0.0
    host.deleteLater()


def test_historical_output_keeps_its_calibration():
    from types import SimpleNamespace
    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from utils.scalebar import export_cfg

    image = np.zeros((240, 400, 3), np.uint8)
    context = build_render_context([image], image, pixel_size_um=0.65)
    record = OutputRecord.create(image, "A", context=context)
    host = SimpleNamespace(scale_bar_enabled=True, scale_bar_target="all",
                           source_px_um=3.0, scale_um_per_px_manual=0.0)
    assert export_cfg(host, record=record)["px_um"] == 0.65
    record.scale_bar_config = export_cfg(host, record=record)
    record.scale_bar_initialized = True
    host.source_px_um = 8.0
    host.scale_um_per_px_manual = 11.0
    assert export_cfg(host, record=record)["px_um"] == 0.65


def test_export_calibration_uses_record_context_and_never_guesses():
    from types import SimpleNamespace
    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from utils.scalebar import calibration_px_um

    image = np.zeros((240, 400, 3), np.uint8)
    context = build_render_context([image], image, pixel_size_um=0.65)
    record = OutputRecord.create(image, "A", context=context)
    legacy = OutputRecord.create(image, "legacy")
    host = SimpleNamespace(source_px_um=3.0, scale_um_per_px_manual=0.0,
                           current_scale_factor=1.0)
    assert calibration_px_um(host, record) == 0.65
    assert calibration_px_um(host, legacy) is None  # no provenance, no guess
    assert calibration_px_um(host) == 3.0


def test_unavailable_registered_target_is_disabled_and_not_preselected(qapp):
    from dialogs.scalebar import ScaleBarDialog

    host = _scale_dialog_host(np.zeros((240, 400, 3), np.uint8))
    host.registration_results = []
    host.scale_bar_target = "registered"
    dialog = ScaleBarDialog(host)
    _preview_scale(qapp, dialog)
    index = dialog.target_combo.findData("registered")
    assert not dialog.target_combo.model().item(index).isEnabled()
    assert dialog.target_combo.currentData() == "source"
    assert dialog._ok_button.isEnabled()
    _close_scale(qapp, dialog)
    assert host.scale_bar_target == "registered"
    host.deleteLater()


def test_empty_output_selector_is_hidden(qapp):
    from dialogs.scalebar import ScaleBarDialog

    host = _scale_dialog_host(np.zeros((240, 400, 3), np.uint8))
    host.scale_bar_target = "all"
    dialog = ScaleBarDialog(host)
    output_index = dialog.target_combo.findData("output")
    assert not dialog.target_combo.model().item(output_index).isEnabled()
    assert dialog.output_combo.count() == 0
    assert not dialog._target_form.isRowVisible(dialog.output_combo)
    _close_scale(qapp, dialog)
    host.deleteLater()


def test_restored_record_keeps_a_tiny_stored_calibration(qapp):
    from types import SimpleNamespace
    from dialogs.scalebar import ScaleBarDialog

    image = np.zeros((240, 400, 3), np.uint8)
    record = SimpleNamespace(
        kind="fusion", label="result", image=image, output_id="o1",
        context=None,
        scale_bar_config={
            "mode": "manual", "manual": 0.0008, "px_um": 0.0008,
            "position": "bottom-right", "color": "auto",
            "font_family": "", "font_size": 0.0, "font_bold": False,
            "text_align": "auto", "custom_x": 0.95, "custom_y": 0.95,
        })
    host = _scale_dialog_host(image)
    host.scale_bar_target = "all"
    # Two decimals are enough for the host value; the record's 0.0008 must
    # not be rounded away while restoring it.
    host.scale_um_per_px_manual = 5.0
    host.output_manager = SimpleNamespace(
        record_for_row=lambda row: record,
        record_for_item=lambda item: record)
    host.output_list = SimpleNamespace(count=lambda: 1,
                                       currentItem=lambda: "item")
    dialog = ScaleBarDialog(host)
    assert dialog.mode_combo.currentData() == "manual"
    assert dialog.px_spin.decimals() >= 4
    assert dialog.manual_value() == pytest.approx(0.0008)
    _close_scale(qapp, dialog)
    host.deleteLater()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_calibration_never_draws_a_physical_claim(value):
    from utils.scalebar import draw_scale_bar
    image = np.zeros((100, 200, 3), np.uint8)
    assert draw_scale_bar(image, value) is image
