"""Transparent PNG CLI processing and explicit rejection of unsupported data."""
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from core.image_loader import ImageStackLoader


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def _stack(folder, dtype, hidden):
    folder.mkdir()
    maximum = np.iinfo(dtype).max
    yy, xx = np.indices((64, 80))
    pattern = (((xx * 11 + yy * 17) % 256) * (maximum // 255)).astype(dtype)
    for index, values in enumerate((pattern, cv2.GaussianBlur(pattern, (7, 7), 0))):
        image = np.full((80, 104, 4), hidden, dtype=dtype)
        image[:, :, 3] = 0
        image[8:72, 12:92, :3] = values[:, :, None]
        image[8:72, 12:92, 3] = maximum // 2
        assert cv2.imwrite(str(folder / f"{index}.png"), image)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_cli_transparent_png_excludes_hidden_pixels_and_preserves_alpha(tmp_path, dtype):
    first, second = tmp_path / "first", tmp_path / "second"
    _stack(first, dtype, 0)
    _stack(second, dtype, np.iinfo(dtype).max)
    results = []
    for index, folder in enumerate((first, second)):
        out = tmp_path / f"out{index}.png"
        result = _run("--input", folder, "--output", out, "--method", "guided_filter",
                      "--cpu", "--threads", "1")
        assert result.returncode == 0, result.stderr
        results.append(cv2.imread(str(out), cv2.IMREAD_UNCHANGED))
    actual, other = results
    assert actual.dtype == dtype and actual.shape == (80, 104, 4)
    expected_alpha = np.zeros((80, 104), dtype)
    expected_alpha[8:72, 12:92] = np.iinfo(dtype).max // 2
    np.testing.assert_array_equal(actual[:, :, 3], expected_alpha)
    np.testing.assert_array_equal(actual, other)
    assert np.any(actual[8:72, 12:92, :3])
    assert not np.any(actual[expected_alpha == 0, :3])
    cropped = tmp_path / "cropped"
    cropped.mkdir()
    for index in range(2):
        source = cv2.imread(str(first / f"{index}.png"), cv2.IMREAD_UNCHANGED)
        assert cv2.imwrite(str(cropped / f"{index}.png"), source[8:72, 12:92, :3])
    opaque_out = tmp_path / "opaque-result.png"
    result = _run("--input", cropped, "--output", opaque_out, "--method", "guided_filter",
                  "--cpu", "--threads", "1")
    assert result.returncode == 0, result.stderr
    expected = cv2.imread(str(opaque_out), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual[8:72, 12:92, :3], expected)


def test_cli_batch_preserves_transparent_png(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _stack(first, np.uint8, 0)
    _stack(second, np.uint8, 255)
    output = tmp_path / "results"
    result = _run("--input", first, second, "--output-dir", output,
                  "--method", "guided_filter", "--cpu", "--threads", "1")
    assert result.returncode == 0, result.stderr
    a = cv2.imread(str(output / "first.png"), cv2.IMREAD_UNCHANGED)
    b = cv2.imread(str(output / "second.png"), cv2.IMREAD_UNCHANGED)
    assert a is not None and b is not None
    assert a.shape == (80, 104, 4)
    np.testing.assert_array_equal(a, b)


def test_desktop_rejects_entire_stack_with_transparent_frame(tmp_path):
    folder = tmp_path / "frames"
    _stack(folder, np.uint8, 99)
    assert cv2.imwrite(str(folder / "opaque.png"), np.zeros((80, 104, 3), np.uint8))
    ok, message, frames, _ = ImageStackLoader().load_from_folder(str(folder))
    assert not ok and not frames
    assert "transparent" in message.lower() or "PNG" in message


@pytest.mark.parametrize("option,value", [("--align", "ecc"), ("--depth-map", "depth.png")])
def test_cli_rejects_incompatible_processing(tmp_path, option, value):
    folder = tmp_path / "frames"
    _stack(folder, np.uint8, 1)
    out = tmp_path / "out.png"
    result = _run("--input", folder, "--output", out, "--method", "guided_filter", option, value)
    assert result.returncode != 0
    assert "transparent" in result.stderr.lower()
    assert not out.exists()


def test_cli_rejects_different_masks_and_non_png_output(tmp_path):
    folder = tmp_path / "frames"
    _stack(folder, np.uint8, 1)
    path = folder / "1.png"
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    frame[8, 12, 3] = 0
    assert cv2.imwrite(str(path), frame)
    out = tmp_path / "out.png"
    result = _run("--input", folder, "--output", out, "--method", "guided_filter")
    assert result.returncode != 0 and not out.exists()
    assert "alpha" in result.stderr.lower()
    _stack(tmp_path / "good", np.uint8, 1)
    out = tmp_path / "out.jpg"
    result = _run("--input", tmp_path / "good", "--output", out, "--method", "guided_filter")
    assert result.returncode != 0 and not out.exists()
    assert "png" in result.stderr.lower()


def test_transparent_tiff_is_rejected(tmp_path):
    path = tmp_path / "source.tif"
    pixels = np.zeros((16, 16, 4), np.uint8)
    pixels[4:12, 4:12, :] = 255
    assert cv2.imwrite(str(path), pixels)
    ok, message, frames, _ = ImageStackLoader().load_from_filepaths([str(path)])
    assert not ok and not frames
    assert "PNG" in message


def test_transparent_later_tiff_page_rejects_entire_container(tmp_path):
    path = tmp_path / "pages.tif"
    opaque = np.full((16, 16, 4), 255, np.uint8)
    transparent = opaque.copy()
    transparent[:4, :, 3] = 0
    assert cv2.imwritemulti(str(path), [opaque, transparent])
    ok, message, frames, _ = ImageStackLoader().load_from_filepaths([str(path)])
    assert not ok and not frames
    assert "PNG" in message


def test_hidden_pixels_cannot_enter_tiled_fusion(tmp_path):
    from types import SimpleNamespace
    from core.cli import _transparent_region
    from core.multi_focus_fusion import MultiFocusFusion

    outputs = []
    for index, hidden in enumerate((0, 255)):
        folder = tmp_path / str(index)
        _stack(folder, np.uint8, hidden)
        frames = [cv2.imread(str(folder / f"{i}.png"), cv2.IMREAD_UNCHANGED)
                  for i in range(2)]
        crops, region = _transparent_region(frames, SimpleNamespace(align="none"))
        assert region is not None
        fusion = MultiFocusFusion(algorithm="guided_filter", tile_enabled=True,
                                  tile_threshold=16, tile_block_size=32,
                                  tile_overlap=8)
        outputs.append(fusion.fuse(crops, thread_count=1))
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_registration_rejects_transparent_arrays():
    from core.registration import ImageRegistration
    frame = np.zeros((16, 16, 4), np.uint8)
    frame[4:12, 4:12, 3] = 255
    with pytest.raises(ValueError, match="transparent"):
        ImageRegistration(method="ecc").process([frame, frame])
