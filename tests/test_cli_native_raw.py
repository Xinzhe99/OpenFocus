"""Small end-to-end camera-linear CLI stacks."""
from dataclasses import replace

import numpy as np
import pytest
import tifffile

pytest.importorskip("rawpy")
pytest.importorskip("PyQt6.QtCore")

from core.cli import run_cli
from utils.linear_dng import LinearDNGMetadata, develop_linear, write_linear_dng


def _write_tiff_named_dng(path, pixels, metadata):
    temporary = path.with_suffix(".dng")
    write_linear_dng(temporary, pixels, metadata)
    temporary.replace(path)


@pytest.fixture
def native_stack(tmp_path):
    folder = tmp_path / "相机 stack"
    folder.mkdir()
    metadata = LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1.0, 1024 / 1667), "Example", "Camera")
    y, x = np.indices((128, 128))
    pixels = np.stack((2000 + x * 40, 5000 + y * 40,
                       10000 + (x + y) * 30), axis=2).astype(np.uint16)
    paths = [folder / "一.dng", folder / "二.tiff"]
    write_linear_dng(paths[0], pixels, metadata)
    _write_tiff_named_dng(paths[1], pixels, metadata)
    return folder, paths, pixels, metadata


def _args(paths, output):
    return ["--native-raw", "--input", *(str(p) for p in paths),
            "--output", str(output), "--method", "guided_filter", "--cpu",
            "--threads", "1", "--no-exposure-match"]


def test_native_cli_dng_and_developed_tiff(native_stack, tmp_path):
    _, paths, _, metadata = native_stack
    dng = tmp_path / "融合.dng"
    tiff = tmp_path / "显影.tiff"
    assert run_cli(_args(paths, dng)) == 0
    assert run_cli(_args(paths, tiff)) == 0
    with tifffile.TiffFile(dng) as source:
        page = source.pages[0]
        assert page.photometric.value == 34892
        assert page.bitspersample == 16
        assert page.tags[271].value == metadata.camera_make
        assert page.tags[272].value == metadata.camera_model
        linear = page.asarray()[:, :, ::-1]
    from utils.image_utils import srgb_icc_profile

    with tifffile.TiffFile(tiff) as result:
        assert result.pages[0].tags[34675].value == srgb_icc_profile()
        developed = result.pages[0].asarray()[:, :, ::-1]
    assert developed.dtype == np.uint16
    np.testing.assert_allclose(developed, develop_linear(linear, metadata), atol=4)


@pytest.mark.parametrize("change", [
    {"camera_model": "Other"},
    {"as_shot_neutral": (0.7, 1.0, 0.8)},
])
def test_native_cli_rejects_incompatible_camera_and_preserves_sources(native_stack, tmp_path, capsys, change):
    from locales import trans

    _, paths, pixels, metadata = native_stack
    _write_tiff_named_dng(paths[1], pixels, replace(metadata, **change))
    originals = [path.read_bytes() for path in paths]
    output = tmp_path / "result.dng"
    assert run_cli(_args(paths, output)) != 0
    assert not output.exists()
    assert [path.read_bytes() for path in paths] == originals
    assert trans.t("linear_raw_mismatch") in capsys.readouterr().err
    assert run_cli(_args(paths, paths[0])) != 0
    assert paths[0].read_bytes() == originals[0]


def test_native_cli_rejects_ordinary_tiff_and_rgb_dng_export(native_stack, tmp_path):
    _, paths, pixels, _ = native_stack
    ordinary = tmp_path / "普通.tiff"
    tifffile.imwrite(ordinary, pixels, photometric="rgb")
    output = tmp_path / "result.dng"
    assert run_cli(_args([paths[0], ordinary], output)) != 0
    assert run_cli(["--input", str(paths[0]), str(ordinary), "--output", str(output)]) != 0
    assert not output.exists()


def test_native_batch_produces_dng_and_tiff(native_stack, tmp_path):
    folder, _, _, _ = native_stack
    output = tmp_path / "results"
    common = ["--native-raw", "--input", str(folder), "--output-dir", str(output),
              "--method", "guided_filter", "--cpu", "--threads", "1",
              "--no-exposure-match"]
    assert run_cli(common) == 0
    assert (output / f"{folder.name}.dng").exists()
    assert run_cli(common + ["--native-batch-format", "tiff"]) == 0
    assert tifffile.imread(output / f"{folder.name}.tiff").dtype == np.uint16


def test_native_cli_rejects_unsafe_options_and_summary_overwrite(native_stack, tmp_path):
    _, paths, _, _ = native_stack
    original = paths[0].read_bytes()
    output = tmp_path / "result.dng"
    assert run_cli(_args(paths, output) + ["--group-size", "2"]) != 0
    assert run_cli(_args(paths, output) + ["--scale-bar"]) != 0
    assert run_cli(_args(paths, output) + ["--json", str(paths[0])]) != 0
    assert paths[0].read_bytes() == original
    assert not output.exists()


def test_native_cli_warns_when_complex_profile_is_developed(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    from core.cli import _save_native
    from utils import linear_dng

    metadata = SimpleNamespace(profile_tags=((50936, "profile"),))
    pixels = np.full((8, 8, 3), 10000, dtype=np.uint16)
    monkeypatch.setattr(linear_dng, "develop_linear", lambda image, profile: image)
    _save_native(str(tmp_path / "developed.tiff"), pixels, metadata)
    assert "color is approximate" in capsys.readouterr().err


@pytest.mark.parametrize('alignment', ['none', 'ecc'])
@pytest.mark.parametrize('tiled', [False, True])
def test_native_cli_materializes_before_registration_and_normalizes_at_pixel_boundary(
        native_stack, tmp_path, monkeypatch, alignment, tiled):
    from core import cli, frame_storage, multi_focus_fusion, registration
    from core.source_frames import SourceFrame
    from utils import image_utils

    _, paths, original, metadata = native_stack
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    options = ['--align', alignment, '--tile-size', '32', '--tile-overlap', '4',
               '--tile-threshold', '32', '--tile-enabled' if tiled else '--no-tile-enabled']
    args = cli.build_parser().parse_args(_args(paths, tmp_path / 'fused.dng') + options)
    frames, loaded_metadata = cli._load_native_stack([str(path) for path in paths], args)
    assert all(isinstance(frame, SourceFrame) for frame in frames)
    assert loaded_metadata.camera_model == metadata.camera_model

    registered = []
    def process(self, images, **kwargs):
        assert all(isinstance(image, np.memmap) and image.dtype == np.uint16 for image in images)
        for image in images:
            np.testing.assert_array_equal(image, original)
        registered.extend(images)
        return images
    monkeypatch.setattr(registration.ImageRegistration, 'process', process)

    normalized_shapes = []
    normalize = image_utils.normalize_fuse_input
    def normalize_one(images):
        assert len(images) == 1
        normalized_shapes.append(images[0].shape)
        if tiled:
            assert max(images[0].shape[:2]) <= 32
        else:
            assert isinstance(images[0], np.memmap)
        return normalize(images)
    monkeypatch.setattr(image_utils, 'normalize_fuse_input', normalize_one)
    monkeypatch.setattr(multi_focus_fusion, 'normalize_fuse_input', normalize_one)

    fused = cli._fuse_stack_admitted(frames, args)
    assert bool(registered) == (alignment != 'none')
    assert normalized_shapes
    assert fused.dtype == np.uint16 and fused.shape == original.shape
    np.testing.assert_allclose(fused, original, rtol=0, atol=4)
    assert all(isinstance(image, np.memmap) for image in args._depth_frames)
    assert all(image.shape == original.shape for image in args._depth_frames)

