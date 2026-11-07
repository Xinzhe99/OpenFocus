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


def test_desktop_loader_preserves_transparent_frames(tmp_path):
    folder = tmp_path / "frames"
    _stack(folder, np.uint8, 99)
    ok, message, frames, _ = ImageStackLoader().load_from_folder(str(folder))
    assert ok, message
    assert len(frames) == 2 and all(frame.shape == (80, 104, 4) for frame in frames)
    assert np.all(frames[0][8:72, 12:92, 3] == 127)
    assert not np.any(frames[0][:8, :, 3])


@pytest.mark.parametrize("option,value", [("--align", "ecc"), ("--depth-map", "depth.png")])
def test_cli_rejects_incompatible_processing(tmp_path, option, value):
    folder = tmp_path / "frames"
    _stack(folder, np.uint8, 1)
    out = tmp_path / "out.png"
    method = "openfocus_b" if option == "--depth-map" else "guided_filter"
    result = _run("--input", folder, "--output", out, "--method", method, option, value)
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


def test_later_tiff_alpha_failure_preserves_existing_workspace(tmp_path):
    import tifffile
    previous = np.full((8, 9, 3), 73, np.uint16)
    loader = ImageStackLoader()
    loader.images, loader.image_paths = [previous], ['previous.tif']
    loader.source_exif = b'previous metadata'
    path = tmp_path / 'alpha-pages.tif'
    opaque = np.full((900, 1100, 4), 65535, np.uint16)
    transparent = opaque.copy()
    transparent[601, 643, 3] = 0  # A single pixel a reduced preview can miss.
    with tifffile.TiffWriter(path) as writer:
        writer.write(opaque, photometric='rgb', extrasamples='unassalpha')
        writer.write(transparent, photometric='rgb', extrasamples='unassalpha')
    ok, message, frames, _ = loader.load_from_filepaths([str(path)])
    assert not ok and not frames and 'PNG' in message
    assert loader.images == [previous] and loader.images[0] is previous
    assert loader.image_paths == ['previous.tif']
    assert loader.source_exif == b'previous metadata' and path.is_file()


def test_fully_opaque_rgba_tiff_still_loads_original_color(tmp_path):
    import tifffile
    from core.source_frames import resolve_frame
    path = tmp_path / 'opaque.tif'
    rgba = np.full((32, 48, 4), 65535, np.uint16)
    rgba[:, :, :3] = (1000, 20000, 60000)
    tifffile.imwrite(path, rgba, photometric='rgb', extrasamples='unassalpha')
    ok, message, frames, _ = ImageStackLoader().load_from_filepaths([str(path)])
    assert ok, message
    assert frames[0].shape == (32, 48, 3)
    np.testing.assert_array_equal(resolve_frame(frames[0]), rgba[:, :, 2::-1])


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


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_desktop_render_and_folder_batch_preserve_alpha(tmp_path, qapp, dtype):
    from core.workers import RenderWorker, BatchWorker

    folder = tmp_path / 'frames'
    _stack(folder, dtype, np.iinfo(dtype).max)
    ok, message, frames, _ = ImageStackLoader().load_from_folder(str(folder))
    assert ok, message
    worker = RenderWorker(frames, algorithm='guided_filter', alignment='none',
                          fusion_parameters={'kernel_size': 9}, use_gpu=False,
                          tile_enabled=False, thread_count=1, exposure_match=False)
    observed = {}
    worker.finished_signal.connect(lambda *args: observed.update(result=args[1]))
    worker.error_signal.connect(lambda message: observed.update(error=message))
    worker.run()
    assert 'error' not in observed, observed.get('error')
    fused = observed['result']
    assert fused.shape == frames[0].shape and fused.dtype == dtype
    np.testing.assert_array_equal(fused[:, :, 3], frames[0][:, :, 3])
    assert not np.any(fused[fused[:, :, 3] == 0, :3])

    output = tmp_path / 'out'
    output.mkdir()
    batch = BatchWorker([str(folder)], 'custom', str(output), {
        'reg_methods': [], 'fusion_method': 'guided_filter',
        'fusion_params': {'kernel_size': 9}, 'format': 'png'},
        thread_count=1, use_gpu=False, exposure_match=False)
    batch.process_single_folder(str(folder))
    saved = cv2.imread(str(output / 'frames.png'), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(saved, fused)


@pytest.mark.parametrize('options', [dict(alignment='ecc'), dict(preview=True),
                                     dict(roi_rect=(0, 0, 8, 8))])
def test_desktop_render_rejects_unsupported_modes(tmp_path, qapp, options):
    from core.workers import RenderWorker
    folder = tmp_path / 'frames'
    _stack(folder, np.uint8, 0)
    ok, _, frames, _ = ImageStackLoader().load_from_folder(str(folder))
    assert ok
    worker = RenderWorker(frames, algorithm='guided_filter', use_gpu=False, **options)
    errors, results = [], []
    worker.error_signal.connect(errors.append)
    worker.finished_signal.connect(lambda *args: results.append(args))
    worker.run()
    assert errors and not results


def test_desktop_rejects_distinct_masks_in_render_and_batch(tmp_path, qapp):
    from core.workers import RenderWorker, BatchWorker
    folder = tmp_path / 'frames'
    _stack(folder, np.uint8, 0)
    path = folder / '1.png'
    changed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    changed[8, 12, 3] = 0
    assert cv2.imwrite(str(path), changed)
    ok, _, frames, _ = ImageStackLoader().load_from_folder(str(folder))
    assert ok
    worker = RenderWorker(frames, algorithm='guided_filter', alignment='none',
                          use_gpu=False, thread_count=1)
    errors, results = [], []
    worker.error_signal.connect(errors.append)
    worker.finished_signal.connect(lambda *args: results.append(args))
    worker.run()
    assert errors and not results
    output = tmp_path / 'out'
    output.mkdir()
    batch = BatchWorker([str(folder)], 'custom', str(output), {
        'reg_methods': [], 'fusion_method': 'guided_filter', 'format': 'png'},
        thread_count=1, use_gpu=False)
    with pytest.raises(ValueError, match='alpha'):
        batch.process_single_folder(str(folder))
    assert not (output / 'frames.png').exists()


def test_desktop_batch_rejects_non_png_before_writing(tmp_path, qapp):
    from core.workers import BatchWorker
    folder = tmp_path / 'frames'
    _stack(folder, np.uint8, 0)
    output = tmp_path / 'out'
    output.mkdir()
    batch = BatchWorker([str(folder)], 'custom', str(output), {
        'reg_methods': [], 'fusion_method': 'guided_filter', 'format': 'jpg'},
        thread_count=1, use_gpu=False)
    with pytest.raises(ValueError):
        batch.process_single_folder(str(folder))
    assert not (output / 'frames.jpg').exists()


def test_display_composites_alpha_without_using_hidden_color():
    from utils.image_utils import composite_transparent_for_display
    first = np.zeros((24, 24, 4), np.uint8)
    second = first.copy()
    second[:, :, :3] = 255
    np.testing.assert_array_equal(composite_transparent_for_display(first),
                                  composite_transparent_for_display(second))
    displayed = composite_transparent_for_display(first)
    assert displayed[0, 0, 0] != displayed[0, 12, 0]


def test_transparent_export_rejects_other_formats(tmp_path):
    from utils.image_utils import imwrite_auto
    image = np.zeros((8, 8, 4), np.uint8)
    image[2:6, 2:6, 3] = 127
    assert not imwrite_auto(str(tmp_path / 'lost.jpg'), image)
    assert not imwrite_auto(str(tmp_path / 'lost.tif'), image)
    assert imwrite_auto(str(tmp_path / 'kept.png'), image)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / 'kept.png'), cv2.IMREAD_UNCHANGED), image)


@pytest.mark.parametrize('extension,dtype,channels', [
    ('.png', np.uint8, 1), ('.png', np.uint16, 1),
    ('.png', np.uint8, 3), ('.png', np.uint16, 3),
    ('.png', np.uint8, 4), ('.png', np.uint16, 4),
    ('.jpg', np.uint8, 3), ('.tif', np.uint16, 3)])
def test_opaque_decode_matches_original_color_read(tmp_path, extension, dtype, channels):
    from core.image_loader import decode_image
    rng = np.random.default_rng(42)
    maximum = np.iinfo(dtype).max
    shape = (20, 24) if channels == 1 else (20, 24, channels)
    pixels = rng.integers(0, maximum + 1, size=shape, dtype=dtype)
    if channels == 4:
        pixels[:, :, 3] = maximum
    path = tmp_path / ('source' + extension)
    assert cv2.imwrite(str(path), pixels)
    expected = cv2.imdecode(np.fromfile(str(path), np.uint8),
                            cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    np.testing.assert_array_equal(decode_image(str(path)), expected)
