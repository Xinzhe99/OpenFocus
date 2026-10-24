"""Disk-level guarantees for image exports and calibrated depth maps."""
import json

import cv2
import numpy as np
import pytest

from utils import image_utils


@pytest.mark.parametrize("writer", ["single", "stack", "physical"])
def test_failed_commit_keeps_the_existing_file(tmp_path, monkeypatch, writer):
    destination = tmp_path / "原始.tif"
    previous = b"original file must survive"
    destination.write_bytes(previous)

    def refuse_replace(*_args):
        raise PermissionError("destination locked")

    monkeypatch.setattr(image_utils.os, "replace", refuse_replace)
    image = np.arange(120, dtype=np.uint16).reshape(10, 12)
    if writer == "single":
        success = image_utils.imwrite_auto(str(destination), image, exif=b"")
    elif writer == "stack":
        success = image_utils.imwrite_multi_tiff(str(destination), [image, image])
    else:
        success = image_utils.imwrite_float_tiff(
            str(destination), image.astype(np.float32), {"unit": "um"})
    assert not success
    assert destination.read_bytes() == previous
    assert set(tmp_path.iterdir()) == {destination}


def test_unicode_sixteen_bit_export_preserves_pixels(tmp_path):
    destination = tmp_path / "标注.png"
    image = np.arange(720, dtype=np.uint16).reshape(12, 20, 3) * 83
    assert image_utils.imwrite_auto(str(destination), image, exif=b"")
    loaded = cv2.imdecode(np.fromfile(destination, np.uint8), cv2.IMREAD_UNCHANGED)
    assert loaded.dtype == np.uint16
    assert np.array_equal(loaded, image)


def test_physical_tiff_preserves_units_negative_depth_and_nodata(tmp_path):
    from PIL import Image

    destination = tmp_path / "物理深度.tif"
    values = np.array([[-2.75, np.nan], [0.0, 1.25]], dtype=np.float32)
    metadata = {"unit": "µm", "z_start": -2.75, "capture_indices": [0, 2, 5]}
    assert image_utils.imwrite_float_tiff(str(destination), values, metadata)
    with Image.open(destination) as loaded:
        actual = np.asarray(loaded)
        assert actual.dtype == np.float32
        assert np.array_equal(actual, values, equal_nan=True)
        assert json.loads(loaded.tag_v2[270]) == metadata
