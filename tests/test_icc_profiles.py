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


def test_invalid_rgb_profile_is_not_propagated(tmp_path):
    invalid = bytearray(128)
    invalid[:4] = (128).to_bytes(4, "big")
    invalid[16:20] = b"RGB "
    invalid[36:40] = b"acsp"
    path = tmp_path / "invalid.png"
    Image.new("RGB", (12, 12)).save(path, icc_profile=bytes(invalid))
    assert common_icc_profile([str(path)]) == b""


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


@pytest.mark.parametrize("second", ["same", "different", "missing"])
def test_multipage_tiff_requires_matching_profile_on_every_page(tmp_path, rgb_profile, second):
    from PIL import TiffImagePlugin

    other = bytearray(rgb_profile)
    other[-1] ^= 1
    path = tmp_path / "pages.tif"
    with TiffImagePlugin.AppendingTiffWriter(str(path), True) as writer:
        for profile in (rgb_profile, {"same": rgb_profile, "different": bytes(other), "missing": b""}[second]):
            tags = TiffImagePlugin.ImageFileDirectory_v2()
            if profile:
                tags[34675] = profile
            Image.new("RGB", (12, 12)).save(writer, format="TIFF", tiffinfo=tags)
            writer.newFrame()
    assert common_icc_profile([str(path)]) == (rgb_profile if second == "same" else b"")


def test_video_clears_previous_profile_and_metadata(tmp_path, rgb_profile, monkeypatch):
    path = tmp_path / "clip.mp4"
    path.touch()
    frame = np.zeros((8, 8, 3), np.uint8)

    class Capture:
        def __init__(self, _path):
            self.frames = iter([frame])
        def isOpened(self):
            return True
        def get(self, _prop):
            return 1
        def read(self):
            image = next(self.frames, None)
            return image is not None, image
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    loader = ImageStackLoader()
    loader.source_icc_profile = rgb_profile
    loader.source_icc_profiles = {"old": rgb_profile}
    loader.source_exif = b"old"
    assert loader.load_from_video(str(path))[0]
    assert loader.source_icc_profile == b""
    assert loader.source_icc_profiles == {}
    assert loader.source_exif == b""


def test_append_profiles_follow_frames_and_exclusions(qapp, tmp_path, rgb_profile):
    from controllers.source_manager import SourceManager
    from core.image_loader import stack_icc_profile
    from tests.test_gui_units import _source_window

    other = bytearray(rgb_profile)
    other[-1] ^= 1
    window = _source_window([])
    window.raw_images = []
    window.image_filenames = []
    window.image_source_paths = []
    window.image_icc_profiles = []
    window.image_loader = ImageStackLoader()
    manager = SourceManager(window)
    for index, profile in enumerate((rgb_profile, rgb_profile, bytes(other), None)):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (12, 12)).save(path, **({"icc_profile": profile} if profile else {}))
        ok, _, images, names = window.image_loader.load_from_filepaths([str(path)], thread_count=1)
        assert ok
        manager._load_source = (None, None, [str(path)])
        manager._apply_load_options(manager._build_load_options(images, names, 1.0), append=index > 0)
        assert stack_icc_profile(window) == (rgb_profile if index < 2 else b"")
    assert stack_icc_profile(window, [0, 1]) == rgb_profile
    assert stack_icc_profile(window, [0, 2]) == b""
    assert stack_icc_profile(window, [0, 3]) == b""


def test_project_frame_order_uses_cached_source_profiles(qapp, tmp_path, rgb_profile):
    from controllers.source_manager import SourceManager
    from core.image_loader import stack_icc_profile
    from tests.test_gui_units import _source_window
    from utils.project_file import _prepare_project_load

    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in paths:
        Image.new("RGB", (12, 12)).save(path, icc_profile=rgb_profile)
    window = _source_window([])
    window.raw_images = []
    window.image_filenames = []
    window.image_loader = ImageStackLoader()
    ok, _, images, names = window.image_loader.load_from_filepaths([str(p) for p in paths], thread_count=1)
    assert ok
    manager = SourceManager(window)
    manager._load_source = (None, None, [str(p) for p in paths])
    options = manager._build_load_options(images, names, 1.0)
    state = {"sources": [str(p) for p in paths], "frame_order": [
        {"source": 1, "name": names[1]}, {"source": 0, "name": names[0]}]}
    options = _prepare_project_load(window, state, options)
    manager._apply_load_options(options)
    assert window.image_source_paths == [str(paths[1]), str(paths[0])]
    assert stack_icc_profile(window) == rgb_profile
