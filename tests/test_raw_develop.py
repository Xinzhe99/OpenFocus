"""Tests for OpenFocus RAW Develop Parameters, External DNG Fallback, and Dialog UI."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import raw_develop as rd
from core.raw_develop import (
    DEMOSAIC_ALGORITHMS,
    HIGHLIGHT_MODES,
    OUTPUT_COLOR_SPACES,
    WB_MODES,
    RawDevelopSettings,
    build_dng_converter_command,
    build_linear_rawpy_postprocess_kwargs,
    build_rawpy_postprocess_kwargs,
    compute_custom_wb_multipliers,
    convert_raw_via_external_dng,
    detect_external_dng_converter,
    load_raw_develop_settings,
    open_raw_with_dng_fallback,
    render_raw_develop_preview,
    reset_raw_develop_settings,
    save_raw_develop_settings,
    validate_dng_converter_path,
)

SAMPLE_RAF_DIR = Path(r"D:\OpenFocus_20261109\OpenFocus测试素材\20261104-悦悦GAKU-RAF")


def _first_sample_raf() -> Path | None:
    if not SAMPLE_RAF_DIR.is_dir():
        return None
    for entry in sorted(SAMPLE_RAF_DIR.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".raf":
            return entry
    return None


# ---------------------------------------------------------------------------
# 1. Model defaults, normalization, and QSettings persistence
# ---------------------------------------------------------------------------


def test_raw_develop_settings_defaults_and_normalization():
    defaults = RawDevelopSettings()
    assert defaults.wb_mode == "camera"
    assert defaults.wb_temp_shift == 0.0
    assert defaults.wb_tint_shift == 0.0
    assert defaults.highlight_mode == "clip"
    assert defaults.demosaic == "ahd"
    assert defaults.exposure_ev == 0.0
    assert defaults.auto_bright is False
    assert defaults.output_color == "srgb"
    assert defaults.external_dng_converter_path == ""
    assert defaults.is_default()

    dirty = RawDevelopSettings(
        wb_mode="UNKNOWN",
        wb_temp_shift=9.5,
        wb_tint_shift=-4.2,
        highlight_mode="invalid",
        demosaic="bogus",
        exposure_ev=5.0,
        auto_bright="true",  # type: ignore[arg-type]
        output_color="cmyk",
        external_dng_converter_path="  /opt/dng  ",
    ).normalized()
    assert dirty.wb_mode == "camera"
    assert dirty.wb_temp_shift == 1.0
    assert dirty.wb_tint_shift == -1.0
    assert dirty.highlight_mode == "clip"
    assert dirty.demosaic == "ahd"
    assert dirty.exposure_ev == 2.0
    assert dirty.auto_bright is True
    assert dirty.output_color == "srgb"
    assert dirty.external_dng_converter_path == "/opt/dng"
    assert not dirty.is_default()

    reloaded = RawDevelopSettings.from_dict(dirty.to_dict())
    assert reloaded == dirty
    assert RawDevelopSettings.from_dict(None) == RawDevelopSettings()


def test_raw_develop_qsettings_roundtrip_and_reset(tmp_path):
    QtCore = pytest.importorskip("PyQt6.QtCore")
    store = QtCore.QSettings(str(tmp_path / "raw_develop.ini"), QtCore.QSettings.Format.IniFormat)

    assert load_raw_develop_settings(store) == RawDevelopSettings()

    custom = RawDevelopSettings(
        wb_mode="custom",
        wb_temp_shift=0.35,
        wb_tint_shift=-0.2,
        highlight_mode="rebuild",
        demosaic="dcb",
        exposure_ev=0.75,
        auto_bright=True,
        output_color="prophoto",
        external_dng_converter_path=str(tmp_path / "dng.exe"),
    )
    saved = save_raw_develop_settings(custom, store)
    assert saved == custom.normalized()

    reopened = QtCore.QSettings(str(tmp_path / "raw_develop.ini"), QtCore.QSettings.Format.IniFormat)
    loaded = load_raw_develop_settings(reopened)
    assert loaded == custom.normalized()

    reset = reset_raw_develop_settings(reopened)
    assert reset == RawDevelopSettings()
    assert load_raw_develop_settings(reopened) == RawDevelopSettings()


# ---------------------------------------------------------------------------
# 2. build_rawpy_postprocess_kwargs across every mode & graceful fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wb_mode", WB_MODES)
def test_build_postprocess_kwargs_wb_modes(wb_mode):
    cfg = RawDevelopSettings(wb_mode=wb_mode, wb_temp_shift=0.5, wb_tint_shift=-0.25)
    kwargs = build_rawpy_postprocess_kwargs(cfg, bps=16, half_size=False)
    assert kwargs["output_bps"] == 16
    assert kwargs["no_auto_bright"] is True
    if wb_mode == "camera":
        assert kwargs["use_camera_wb"] is True
        assert kwargs["use_auto_wb"] is False
        assert "user_wb" not in kwargs
    elif wb_mode == "auto":
        assert kwargs["use_camera_wb"] is False
        assert kwargs["use_auto_wb"] is True
        assert "user_wb" not in kwargs
    elif wb_mode == "daylight":
        assert kwargs["use_camera_wb"] is False
        assert kwargs["use_auto_wb"] is False
        assert "user_wb" not in kwargs
    elif wb_mode == "custom":
        assert kwargs["use_camera_wb"] is False
        assert kwargs["use_auto_wb"] is False
        r, g, b, g2 = kwargs["user_wb"]
        assert r > 1.0 and b < 1.0
        assert g > 1.0 and g == g2


def test_custom_wb_multipliers_respect_camera_wb_and_direction():
    warm = compute_custom_wb_multipliers(0.6, 0.0, camera_wb=[2.0, 1.0, 1.5, 1.0])
    cool = compute_custom_wb_multipliers(-0.6, 0.0, camera_wb=[2.0, 1.0, 1.5, 1.0])
    assert warm[0] > cool[0]
    assert warm[2] < cool[2]

    magenta = compute_custom_wb_multipliers(0.0, 0.5)
    green = compute_custom_wb_multipliers(0.0, -0.5)
    assert magenta[1] < 1.0 < green[1]


@pytest.mark.parametrize(
    "highlight_mode,expected_code",
    [("clip", 0), ("unclip", 1), ("blend", 2), ("rebuild", 5)],
)
def test_build_postprocess_kwargs_highlight_modes(highlight_mode, expected_code):
    rawpy = pytest.importorskip("rawpy")
    cfg = RawDevelopSettings(highlight_mode=highlight_mode)
    kwargs = build_rawpy_postprocess_kwargs(cfg, bps=16)
    val = kwargs["highlight_mode"]
    assert int(getattr(val, "value", val)) == expected_code

    # Without HighlightMode enum on a minimal mock rawpy, falls back to int code.
    kwargs_minimal = build_rawpy_postprocess_kwargs(cfg, rawpy_module=SimpleNamespace())
    assert kwargs_minimal["highlight_mode"] == expected_code


@pytest.mark.parametrize("demosaic", DEMOSAIC_ALGORITHMS)
def test_build_postprocess_kwargs_demosaic_algorithms(demosaic):
    rawpy = pytest.importorskip("rawpy")
    cfg = RawDevelopSettings(demosaic=demosaic)
    kwargs = build_rawpy_postprocess_kwargs(cfg, bps=16)
    assert "demosaic_algorithm" in kwargs
    assert isinstance(kwargs["demosaic_algorithm"], rawpy.DemosaicAlgorithm)


def test_build_postprocess_kwargs_demosaic_graceful_fallback():
    class _StubMember:
        def __init__(self, name, supported):
            self.name = name
            self.isSupported = supported

    class _PartialDemosaic:
        AHD = _StubMember("AHD", True)
        DCB = _StubMember("DCB", False)  # simulates LibRaw build without DCB

    fake_rawpy = SimpleNamespace(DemosaicAlgorithm=_PartialDemosaic)
    kwargs = build_rawpy_postprocess_kwargs(
        RawDevelopSettings(demosaic="dcb"), rawpy_module=fake_rawpy
    )
    assert kwargs["demosaic_algorithm"] is _PartialDemosaic.AHD

    kwargs_missing = build_rawpy_postprocess_kwargs(
        RawDevelopSettings(demosaic="ppg"), rawpy_module=fake_rawpy
    )
    assert kwargs_missing["demosaic_algorithm"] is _PartialDemosaic.AHD


@pytest.mark.parametrize("output_color", OUTPUT_COLOR_SPACES)
def test_build_postprocess_kwargs_output_color_and_exposure(output_color):
    rawpy = pytest.importorskip("rawpy")
    cfg = RawDevelopSettings(
        output_color=output_color,
        exposure_ev=1.0,
        auto_bright=True,
    )
    kwargs = build_rawpy_postprocess_kwargs(cfg, bps=8, half_size=True)
    assert kwargs["output_bps"] == 8
    assert kwargs["half_size"] is True
    assert kwargs["no_auto_bright"] is False
    assert kwargs["exp_shift"] == pytest.approx(2.0, rel=1e-5)
    assert isinstance(kwargs["output_color"], rawpy.ColorSpace)

    default_kwargs = build_rawpy_postprocess_kwargs(None)
    assert default_kwargs["use_camera_wb"] is True
    assert default_kwargs["no_auto_bright"] is True
    assert default_kwargs["output_bps"] == 16
    assert default_kwargs["gamma"] == (2.4, 12.92)
    assert "exp_shift" not in default_kwargs


def test_build_linear_postprocess_kwargs():
    rawpy = pytest.importorskip("rawpy")
    default_lin = build_linear_rawpy_postprocess_kwargs(None, half_size=True)
    assert default_lin["output_color"] == rawpy.ColorSpace.raw
    assert default_lin["user_wb"] == [1, 1, 1, 1]
    assert default_lin["no_auto_scale"] is True
    assert default_lin["gamma"] == (1, 1)
    assert default_lin["half_size"] is True
    assert "demosaic_algorithm" not in default_lin

    custom_lin = build_linear_rawpy_postprocess_kwargs(
        RawDevelopSettings(demosaic="dcb", highlight_mode="blend"), half_size=False
    )
    assert custom_lin["demosaic_algorithm"] == rawpy.DemosaicAlgorithm.DCB
    assert custom_lin["highlight_mode"] == rawpy.HighlightMode.Blend


# ---------------------------------------------------------------------------
# 3. External DNG Converter validation, CLI invocation & fallback (mocked)
# ---------------------------------------------------------------------------


def test_validate_and_build_dng_converter_command(tmp_path):
    assert validate_dng_converter_path("") == (False, "empty")
    assert validate_dng_converter_path(str(tmp_path / "missing.exe")) == (False, "not_found")
    assert validate_dng_converter_path(str(tmp_path)) == (False, "not_file")

    exe = tmp_path / ("Adobe DNG Converter.exe" if os.name == "nt" else "dng_converter")
    exe.write_bytes(b"stub")
    if os.name != "nt":
        os.chmod(exe, 0o755)
    ok, resolved = validate_dng_converter_path(str(exe))
    assert ok and resolved == str(exe)

    cmd = build_dng_converter_command(str(exe), str(tmp_path / "shot.cr3"), str(tmp_path), "shot.dng")
    assert cmd == [str(exe), "-u", "-p0", "-d", str(tmp_path), "-o", "shot.dng", str(tmp_path / "shot.cr3")]


def test_convert_raw_via_external_dng_and_fallback(tmp_path, monkeypatch):
    exe = tmp_path / ("dng.exe" if os.name == "nt" else "dng")
    exe.write_bytes(b"stub")
    if os.name != "nt":
        os.chmod(exe, 0o755)
    source = tmp_path / "brand_new_camera.cr3"
    source.write_bytes(b"raw-bytes")

    invoked_cmds = []

    def fake_run(cmd, **kwargs):
        assert kwargs.get("shell") is False
        invoked_cmds.append(cmd)
        out_dir = cmd[cmd.index("-d") + 1]
        out_name = cmd[cmd.index("-o") + 1]
        (Path(out_dir) / out_name).write_bytes(b"converted-dng")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rd.subprocess, "run", fake_run)
    converted = convert_raw_via_external_dng(str(source), str(exe), output_dir=str(tmp_path / "out"))
    assert Path(converted).is_file()
    assert len(invoked_cmds) == 1

    # Test open_raw_with_dng_fallback: native fails, fallback succeeds, temp DNG cleaned up.
    opened_files = []

    class _StubRawPy:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def open_file(self, path):
            opened_files.append(path)
            if path == str(source):
                raise RuntimeError("Unsupported camera model")

    fake_rawpy = SimpleNamespace(RawPy=_StubRawPy)
    cfg = RawDevelopSettings(external_dng_converter_path=str(exe))
    with open_raw_with_dng_fallback(str(source), cfg, rawpy_module=fake_rawpy) as (_raw, opened):
        assert opened != str(source)
        assert opened.endswith(".dng")
        assert Path(opened).is_file()
        temp_dng = Path(opened)
    assert not temp_dng.exists()
    assert opened_files[0] == str(source)


def test_detect_external_dng_converter_uses_candidates(tmp_path, monkeypatch):
    exe = tmp_path / ("Adobe DNG Converter.exe" if os.name == "nt" else "dng_converter")
    exe.write_bytes(b"stub")
    if os.name != "nt":
        os.chmod(exe, 0o755)
    monkeypatch.setattr(rd, "_candidate_dng_converter_paths", lambda: [str(tmp_path / "nope"), str(exe)])
    assert detect_external_dng_converter() == os.path.abspath(str(exe))


# ---------------------------------------------------------------------------
# 4. Integration with core/image_loader.py and utils/linear_dng.py
# ---------------------------------------------------------------------------


def test_read_raw_wires_explicit_develop_settings(tmp_path, monkeypatch):
    import core.image_loader as il

    captured = {}

    class _FakeRaw:
        sizes = SimpleNamespace(width=32, height=24, raw_width=32, raw_height=24, pixel_aspect=1.0, flip=0)
        raw_type = "flat"
        raw_pattern = np.zeros((2, 2), dtype=np.uint8)
        num_colors = 3
        camera_whitebalance = [2.0, 1.0, 1.5, 1.0]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def open_file(self, _path):
            pass

        def unpack(self):
            pass

        def postprocess(self, **kwargs):
            captured.update(kwargs)
            return np.full((24, 32, 3), 1000, dtype=np.uint16)

    monkeypatch.setattr(il, "RAWPY_AVAILABLE", True)
    monkeypatch.setattr(
        il,
        "rawpy",
        SimpleNamespace(RawPy=_FakeRaw, RawType=SimpleNamespace(Flat="flat", Stack="stack")),
        raising=False,
    )
    raw_file = tmp_path / "sample.raf"
    raw_file.write_bytes(b"0" * 64)

    cfg = RawDevelopSettings(
        wb_mode="custom",
        wb_temp_shift=0.4,
        highlight_mode="blend",
        exposure_ev=0.5,
        auto_bright=True,
    )
    bgr = il.decode_image(str(raw_file), raw_develop_settings=cfg)
    assert bgr is not None and bgr.shape == (24, 32, 3) and bgr.dtype == np.uint16
    assert captured["use_camera_wb"] is False
    assert captured["no_auto_bright"] is False
    assert captured["highlight_mode"] == 2
    assert captured["exp_shift"] == pytest.approx(2.0 ** 0.5, rel=1e-5)
    assert "user_wb" in captured


def test_develop_linear_backward_compatible_and_exposure_aware():
    from utils.linear_dng import LinearDNGMetadata, develop_linear

    meta = LinearDNGMetadata(
        color_matrix=((0.8, -0.1, -0.1), (-0.4, 1.2, 0.2), (-0.1, 0.2, 0.6)),
        as_shot_neutral=(0.6, 1.0, 0.7),
    )
    linear = np.full((16, 16, 3), 10000, dtype=np.uint16)
    baseline = develop_linear(linear, meta)
    same = develop_linear(linear, meta, develop_settings=RawDevelopSettings())
    np.testing.assert_array_equal(baseline, same)

    brighter = develop_linear(linear, meta, develop_settings=RawDevelopSettings(exposure_ev=1.0))
    assert float(brighter.mean()) > float(baseline.mean())


# ---------------------------------------------------------------------------
# 5. Real RAW half-size preview decode (skipped cleanly in CI if absent)
# ---------------------------------------------------------------------------


def test_real_raf_half_size_preview_decode():
    sample = _first_sample_raf()
    if sample is None:
        pytest.skip(f"Local RAW sample folder not available: {SAMPLE_RAF_DIR}")
    pytest.importorskip("rawpy")

    preview_base = render_raw_develop_preview(str(sample), RawDevelopSettings(), max_side=480)
    assert preview_base.dtype == np.uint8
    assert preview_base.ndim == 3 and preview_base.shape[2] == 3
    assert max(preview_base.shape[:2]) <= 480

    preview_dark = render_raw_develop_preview(
        str(sample), RawDevelopSettings(exposure_ev=-1.0), max_side=360
    )
    preview_bright = render_raw_develop_preview(
        str(sample), RawDevelopSettings(exposure_ev=1.0), max_side=360
    )
    assert float(preview_bright.mean()) > float(preview_dark.mean()) + 15.0

    preview_warm = render_raw_develop_preview(
        str(sample), RawDevelopSettings(wb_mode="custom", wb_temp_shift=0.6), max_side=360
    )
    preview_cool = render_raw_develop_preview(
        str(sample), RawDevelopSettings(wb_mode="custom", wb_temp_shift=-0.6), max_side=360
    )
    # BGR order: index 2 is Red, index 0 is Blue.
    warm_ratio = float(preview_warm[:, :, 2].mean()) / max(1.0, float(preview_warm[:, :, 0].mean()))
    cool_ratio = float(preview_cool[:, :, 2].mean()) / max(1.0, float(preview_cool[:, :, 0].mean()))
    assert warm_ratio > cool_ratio * 1.3


# ---------------------------------------------------------------------------
# 6. RawDevelopDialog UI, Light/Dark/Gray themes, and Menu integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme", ["light", "dark", "gray"])
def test_raw_develop_dialog_themes_controls_and_apply(qapp, tmp_path, monkeypatch, theme):
    QtCore = pytest.importorskip("PyQt6.QtCore")
    from dialogs.raw_develop import RawDevelopDialog

    fake_raw = tmp_path / "preview.raf"
    fake_raw.write_bytes(b"stub")
    stub_bgr = np.full((180, 240, 3), (90, 140, 210), dtype=np.uint8)
    monkeypatch.setattr("dialogs.raw_develop.render_raw_develop_preview", lambda *_a, **_k: stub_bgr)

    store = QtCore.QSettings(str(tmp_path / "dialog.ini"), QtCore.QSettings.Format.IniFormat)
    dialog = RawDevelopDialog(
        sample_paths=[str(fake_raw)],
        qsettings=store,
        theme=theme,
    )
    try:
        dialog.show()
        # Wait briefly for async preview worker to emit previewReady.
        for _ in range(40):
            qapp.processEvents()
            if dialog._current_pixmap is not None and not dialog._current_pixmap.isNull():
                break
            QtCore.QThread.msleep(15)
        assert dialog._current_pixmap is not None and not dialog._current_pixmap.isNull()
        assert dialog._theme == theme

        # WB temp/tint sliders disabled unless custom WB mode is selected.
        assert not dialog.slider_wb_temp.isEnabled()
        custom_idx = dialog.combo_wb_mode.findData("custom")
        dialog.combo_wb_mode.setCurrentIndex(custom_idx)
        assert dialog.slider_wb_temp.isEnabled()
        assert dialog.slider_wb_tint.isEnabled()

        dialog.slider_wb_temp.setValue(45)
        dialog.slider_exposure.setValue(-50)
        dialog.combo_highlight.setCurrentIndex(dialog.combo_highlight.findData("blend"))
        dialog.combo_demosaic.setCurrentIndex(dialog.combo_demosaic.findData("dcb"))
        dialog.combo_output_color.setCurrentIndex(dialog.combo_output_color.findData("adobe"))
        dialog.chk_auto_bright.setChecked(True)

        # Reset restores defaults.
        dialog.btn_reset.click()
        assert dialog.current_settings() == RawDevelopSettings()
        assert not dialog.slider_wb_temp.isEnabled()

        # Re-apply a valid setting and persist.
        dialog.combo_highlight.setCurrentIndex(dialog.combo_highlight.findData("rebuild"))
        dialog.slider_exposure.setValue(75)
        dialog.btn_apply.click()
        assert dialog.applied_settings is not None
        assert dialog.applied_settings.highlight_mode == "rebuild"
        assert dialog.applied_settings.exposure_ev == pytest.approx(0.75)
        assert load_raw_develop_settings(store).highlight_mode == "rebuild"
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_raw_develop_menu_action_and_translations(qapp):
    from locales import trans
    from tests.test_menu_translations import _menu_window
    from ui.menus import setup_menus

    window = _menu_window(qapp)
    called = []
    window.show_raw_develop_dialog = lambda: called.append(True)
    setup_menus(window)
    try:
        action = window.ui_objs["action_raw_develop"]
        assert action in window.ui_objs["menu_settings"].actions()
        assert action.property("trans_key") == "raw_develop_title"
        action.trigger()
        assert called == [True]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()

    for code in trans.PACK_ORDER:
        table = trans.translations[code]
        assert table.get("raw_develop_title"), f"{code}: missing raw_develop_title"
        assert table.get("raw_develop_group_dng"), f"{code}: missing raw_develop_group_dng"
