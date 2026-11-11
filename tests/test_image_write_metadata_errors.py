"""JPEG metadata limits surface a specific error instead of returning False."""
import numpy as np
import pytest


class _Meta:
    def __init__(self, size):
        self.size = size

    def tobytes(self):
        return b"E" * self.size


def test_jpeg_exif_too_large_raises_specific_error(tmp_path, monkeypatch):
    import utils.image_utils as iu
    monkeypatch.setattr(iu, "export_exif", lambda _data: _Meta(70000))
    with pytest.raises(iu.ImageWriteError) as caught:
        iu.imwrite_auto(str(tmp_path / "large.jpg"),
                        np.zeros((8, 8, 3), np.uint8), exif=b"source")
    assert caught.value.code == "exif_too_large"
    assert "EXIF" in str(caught.value)
    assert not (tmp_path / "large.jpg").exists()


def test_jpeg_icc_too_large_raises_specific_error(tmp_path, monkeypatch):
    import utils.image_utils as iu
    monkeypatch.setattr(iu, "export_exif", lambda _data: _Meta(4))
    with pytest.raises(iu.ImageWriteError) as caught:
        iu.imwrite_auto(str(tmp_path / "large-icc.jpg"),
                        np.zeros((8, 8, 3), np.uint8), exif=b"source",
                        icc_profile=b"I" * (255 * 65519 + 1))
    assert caught.value.code == "icc_too_large"
    assert not (tmp_path / "large-icc.jpg").exists()


def test_jpeg_normal_metadata_still_writes(tmp_path, monkeypatch):
    import utils.image_utils as iu
    monkeypatch.setattr(iu, "export_exif", lambda _data: _Meta(4))
    path = tmp_path / "ok.jpg"
    assert iu.imwrite_auto(str(path), np.zeros((8, 8, 3), np.uint8), exif=b"source")
    assert path.is_file()


def test_real_large_exif_jpeg_is_a_clear_error(tmp_path):
    from PIL import Image
    import utils.image_utils as iu
    exif = Image.Exif()
    exif[270] = "D" * 70000  # ImageDescription
    path = tmp_path / "real-large.jpg"
    with pytest.raises(iu.ImageWriteError) as caught:
        iu.imwrite_auto(str(path), np.zeros((8, 8, 3), np.uint8),
                        exif=exif.tobytes())
    assert caught.value.code == "exif_too_large"
    assert "EXIF" in str(caught.value)
    assert not path.exists()


def test_cancelled_jpeg_write_propagates_and_leaves_no_file(tmp_path):
    from core.registration import RegistrationCancelled
    import utils.image_utils as iu
    path = tmp_path / "cancelled.jpg"
    with pytest.raises(RegistrationCancelled):
        iu.imwrite_auto(str(path), np.zeros((8, 8, 3), np.uint8),
                        exif=b"", should_cancel=lambda: True)
    assert not path.exists()


def test_invalid_exif_snapshot_is_dropped_not_retained(monkeypatch):
    import core.image_loader as loader
    monkeypatch.setattr(loader, "_read_exif_bytes", lambda _path: b"not-an-exif-block")
    assert loader._read_safe_exif("/invalid.jpg") == b""


def test_large_parseable_exif_snapshot_fails_explicitly(monkeypatch):
    from PIL import Image
    import core.image_loader as loader
    exif = Image.Exif()
    # A real TIFF-structured EXIF block larger than the 16 MiB snapshot
    # budget; it is parseable and sanitizing keeps the long description.
    exif[270] = "D" * (16 * 1024 * 1024 + 16)
    raw = exif.tobytes()
    assert len(raw) > 16 * 1024 * 1024
    monkeypatch.setattr(loader, "_read_exif_bytes", lambda _path: raw)
    with pytest.raises(loader.SourceExifTooLarge) as caught:
        loader._read_safe_exif("/huge.jpg")
    exc = caught.value
    assert exc.path == "/huge.jpg"
    assert exc.limit == 16 * 1024 * 1024
    assert exc.size > exc.limit
    assert "/huge.jpg" in str(exc)
    assert "export without metadata" not in str(exc)


def _write_oversized_exif(tmp_path, fmt, ext):
    from PIL import Image
    path = tmp_path / ("huge-exif" + ext)
    exif = Image.Exif()
    exif[270] = "D" * (16 * 1024 * 1024 + 16)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(
        path, format=fmt, exif=exif.tobytes())
    return path


@pytest.mark.parametrize(("fmt", "ext"), [("PNG", ".png"), ("TIFF", ".tif")])
def test_real_oversized_exif_loader_raises_english(tmp_path, fmt, ext):
    from core.image_loader import ImageStackLoader
    from utils.image_utils import SourceExifTooLarge

    path = _write_oversized_exif(tmp_path, fmt, ext)
    with pytest.raises(SourceExifTooLarge) as caught:
        ImageStackLoader().load_from_filepaths([str(path)])
    exc = caught.value
    assert exc.filename == str(path)
    assert exc.size > exc.limit
    assert "export without metadata" not in str(exc)


@pytest.mark.parametrize(("fmt", "ext"), [("PNG", ".png"), ("TIFF", ".tif")])
def test_real_oversized_exif_worker_reports_localized_path(qapp, tmp_path, fmt, ext):
    from core.image_loader import ImageStackLoader
    from core.workers import StackLoadWorker
    from locales import trans

    path = _write_oversized_exif(tmp_path, fmt, ext)
    previous = trans.current_lang
    trans.set_language("zh")
    try:
        expected = trans.t('source_exif_snapshot_too_large').format(path=str(path))
        worker = StackLoadWorker(ImageStackLoader(), filepaths=[str(path)])
        messages = []
        worker.finished_load.connect(
            lambda ok, message, images, names: messages.append((ok, message, images, names)))
        worker.run()
    finally:
        trans.set_language(previous)

    assert messages == [(False, expected, [], [])]
