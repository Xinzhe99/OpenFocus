"""Color profiles belong to the pixels that produced each exported result."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

from core.image_loader import ImageStackLoader, common_icc_profile
from utils.image_utils import imwrite_auto


@pytest.fixture
def rgb_profile():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@pytest.mark.parametrize("extension", [".png", ".tif"])
@pytest.mark.parametrize("channels", [3, 4])
def test_16bit_export_keeps_pixels_exif_and_icc(tmp_path, rgb_profile, extension, channels):
    image = np.random.default_rng(3).integers(0, 65536, (17, 23, channels), dtype=np.uint16)
    if channels == 4 and extension == ".tif":
        image[:, :, 3] = np.iinfo(np.uint16).max
    exif = Image.Exif()
    exif[271] = "Camera"
    path = tmp_path / f"结果{extension}"
    assert imwrite_auto(str(path), image, exif=exif.tobytes(), icc_profile=rgb_profile)
    back = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(back, image)
    with Image.open(path) as saved:
        assert saved.info.get("icc_profile") == rgb_profile
        assert saved.getexif()[271] == "Camera"


@pytest.mark.parametrize("extension", [".png", ".tif", ".jpg", ".webp"])
def test_8bit_formats_keep_the_profile(tmp_path, rgb_profile, extension):
    image = np.full((16, 18, 3), (20, 90, 160), dtype=np.uint8)
    path = tmp_path / f"out{extension}"
    assert imwrite_auto(str(path), image, exif=b"", icc_profile=rgb_profile)
    with Image.open(path) as saved:
        assert saved.info.get("icc_profile") == rgb_profile
    if extension in (".png", ".tif", ".webp"):
        back = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(back, image)


def test_loader_does_not_attribute_mixed_or_previous_profiles(tmp_path, rgb_profile):
    tagged = tmp_path / "tagged"
    mixed = tmp_path / "mixed"
    tagged.mkdir()
    mixed.mkdir()
    for folder, profiles in [(tagged, [rgb_profile, rgb_profile]), (mixed, [rgb_profile, None])]:
        for index, profile in enumerate(profiles):
            kwargs = {"icc_profile": profile} if profile else {}
            Image.fromarray(np.full((12, 15, 3), 50 + index * 50, np.uint8)).save(
                folder / f"{index}.png", **kwargs)
    loader = ImageStackLoader()
    assert loader.load_from_folder(str(tagged), thread_count=1)[0]
    assert loader.source_icc_profile == rgb_profile
    assert loader.load_from_folder(str(mixed), thread_count=1)[0]
    assert loader.source_icc_profile == b""
    assert common_icc_profile([str(mixed / "0.png"), str(mixed / "1.png")]) == b""


def test_historical_output_uses_its_own_profile(tmp_path, rgb_profile):
    pytest.importorskip("PyQt6")
    from controllers.output_manager import OutputManager, OutputRecord
    from core.render_context import build_render_context

    image = np.full((12, 15, 3), 120, np.uint8)
    context = build_render_context([image], image, icc_profile=rgb_profile)
    record = OutputRecord.create(image, "old", context=context)
    window = SimpleNamespace(scale_bar_enabled=False,
        image_loader=SimpleNamespace(source_icc_profile=b"different"),
        export_manager=SimpleNamespace(_safe_export_path=lambda path: path))
    labels = SimpleNamespace(prepare_bgr_image=lambda _target, pixels, _index, **_kw: pixels)
    path = tmp_path / "old.png"
    OutputManager(window)._write_record(record, str(path), labels)
    with Image.open(path) as saved:
        assert saved.info.get("icc_profile") == rgb_profile
