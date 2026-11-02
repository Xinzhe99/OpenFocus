"""Focused contracts for opt-in sensor-coordinate cleanup."""
import cv2
import numpy as np
import pytest
from PIL import Image

from utils.sensor_cleanup import (
    SensorCleanupCancelled,
    cleanup_sensor_image,
    load_sensor_defect_mask,
    validate_sensor_source_orientation,
)


def _save_mask(tmp_path, mask):
    path = tmp_path / "mask.png"
    assert cv2.imwrite(str(path), mask)
    return path


def test_disabled_cleanup_leaves_input_untouched():
    image = np.arange(48, dtype=np.uint16).reshape(4, 4, 3)
    assert cleanup_sensor_image(image) is image


def test_mask_repairs_from_original_unmasked_neighbors_and_preserves_alpha(tmp_path):
    image = np.full((10, 10, 4), (1001, 2002, 3003, 4004), dtype=np.uint16)
    image[4, 4, :3] = 65535
    image[4, 5, :3] = 0
    image[4, 4, 3] = 777
    original = image.copy()
    mask = np.zeros((10, 10), np.uint8)
    mask[4, 4:6] = 255
    result = cleanup_sensor_image(image, _save_mask(tmp_path, mask))
    np.testing.assert_array_equal(result[4, 4, :3], [1001, 2002, 3003])
    np.testing.assert_array_equal(result[4, 5, :3], [1001, 2002, 3003])
    np.testing.assert_array_equal(result[..., 3], image[..., 3])
    np.testing.assert_array_equal(image, original)
    assert result.dtype == np.uint16


def test_mask_is_loaded_once_and_reports_extent_and_coverage(tmp_path):
    mask = np.zeros((10, 20), np.uint16)
    mask[3, 4] = 1
    path = _save_mask(tmp_path, mask)
    loaded = load_sensor_defect_mask(path)
    assert (loaded.width, loaded.height, loaded.coverage) == (20, 10, 0.005)
    path.unlink()
    image = np.full((10, 20), 31, np.uint8)
    image[3, 4] = 250
    assert cleanup_sensor_image(image, loaded)[3, 4] == 31
    with pytest.raises(ValueError, match="dimensions"):
        cleanup_sensor_image(image[:9], loaded)


@pytest.mark.parametrize("mask", [
    np.full((10, 10), 127, np.uint8),
    np.full((10, 10, 3), 255, np.uint8),
    np.pad(np.full((2, 6), 255, np.uint8), ((0, 8), (0, 4))),
])
def test_reject_invalid_mask(tmp_path, mask):
    with pytest.raises(ValueError):
        load_sensor_defect_mask(_save_mask(tmp_path, mask))


def test_reject_mixed_nonzero_mask_values(tmp_path):
    mask = np.zeros((10, 10), np.uint8)
    mask[0, :2] = [1, 255]
    with pytest.raises(ValueError, match="one nonzero"):
        load_sensor_defect_mask(_save_mask(tmp_path, mask))


def test_reject_defect_without_clean_neighbors(tmp_path):
    mask = np.zeros((20, 20), np.uint8)
    mask[4:7, 4:7] = 255
    with pytest.raises(ValueError, match="no unmasked neighbor"):
        cleanup_sensor_image(np.zeros((20, 20), np.uint8), _save_mask(tmp_path, mask))


@pytest.mark.parametrize("dtype,base,noise", [(np.uint8, 90, 3), (np.uint16, 30001, 700)])
def test_denoise_is_mild_preserves_edges_precision_and_alpha(dtype, base, noise):
    image = np.full((20, 20, 4), base, dtype)
    image[:, 10:, :3] = base * 2
    image[5, 5, :3] = base + noise
    image[..., 3] = np.arange(20, dtype=dtype)
    result = cleanup_sensor_image(image, sensor_denoise=True)
    assert base < result[5, 5, 0] < base + noise
    assert result[8, 9, 0] == base
    assert result[8, 10, 0] == base * 2
    np.testing.assert_array_equal(result[..., 3], image[..., 3])
    assert result.dtype == dtype
    if dtype == np.uint16:
        assert result[5, 5, 0] % 257 != 0


def test_cancel_before_and_during_processing(tmp_path):
    image = np.zeros((600, 600), np.uint8)
    with pytest.raises(SensorCleanupCancelled):
        cleanup_sensor_image(image, sensor_denoise=True, cancel_callback=lambda: True)
    calls = iter([False, False, False, True])
    with pytest.raises(SensorCleanupCancelled):
        cleanup_sensor_image(image, sensor_denoise=True, cancel_callback=lambda: next(calls))
    with pytest.raises(SensorCleanupCancelled):
        load_sensor_defect_mask(_save_mask(tmp_path, image), lambda: True)


def test_mask_memory_rejected_before_encoded_read_and_decode(tmp_path, monkeypatch):
    from core import resource_policy
    from utils import sensor_cleanup

    mask = np.zeros((300, 400), np.uint16)
    path = _save_mask(tmp_path, mask)
    requests = []

    def reject(amount):
        requests.append(amount)
        raise resource_policy.ResourceLimitError("full")

    monkeypatch.setattr(resource_policy, "require_memory", reject)
    monkeypatch.setattr(sensor_cleanup.np, "fromfile", lambda *a, **k: pytest.fail("read before admission"))
    monkeypatch.setattr(sensor_cleanup.cv2, "imdecode", lambda *a, **k: pytest.fail("decode before admission"))
    with pytest.raises(resource_policy.ResourceLimitError):
        load_sensor_defect_mask(path)
    assert requests == [path.stat().st_size + mask.size * 16]


def test_cleanup_admits_copy_and_denoise_buffers_before_copy(monkeypatch):
    from core import resource_policy

    image = np.zeros((300, 400, 4), np.uint16)
    requests = []

    def reject(amount):
        requests.append(amount)
        raise resource_policy.ResourceLimitError("full")

    monkeypatch.setattr(resource_policy, "require_memory", reject)
    with pytest.raises(resource_policy.ResourceLimitError):
        cleanup_sensor_image(image, sensor_denoise=True)
    assert requests[0] >= 2 * image.nbytes + 258 * 258 * 32


def test_stack_admission_accounts_for_all_copies_and_mask(tmp_path, monkeypatch):
    from core import resource_policy
    from utils.sensor_cleanup import require_sensor_cleanup_stack

    images = [np.zeros((300, 400, 3), np.uint8) for _ in range(3)]
    path = _save_mask(tmp_path, np.zeros((300, 400), np.uint8))
    requests = []
    monkeypatch.setattr(resource_policy, "require_memory", requests.append)
    require_sensor_cleanup_stack(images, path, True)
    assert requests[0] >= (sum(image.nbytes for image in images) + images[0].nbytes
                           + 258 * 258 * 32 + path.stat().st_size + 300 * 400 * 16)


@pytest.mark.parametrize('orientation', [None, 1])
def test_sensor_orientation_accepts_neutral_jpeg(tmp_path, orientation):
    path = tmp_path / 'neutral.jpg'
    image = Image.fromarray(np.zeros((16, 16, 3), np.uint8))
    exif = image.getexif()
    if orientation is not None:
        exif[274] = orientation
    image.save(path, exif=exif)
    validate_sensor_source_orientation([path])


def test_sensor_orientation_rejects_rotated_square_jpeg(tmp_path):
    path = tmp_path / 'rotated.jpg'
    image = Image.fromarray(np.zeros((16, 16, 3), np.uint8))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    with pytest.raises(ValueError, match='orientation.*6'):
        validate_sensor_source_orientation([path])


def test_sensor_orientation_checks_each_tiff_page(tmp_path):
    import struct

    # Two uncompressed 1x1 TIFF pages, with a rotation on the second page.
    def ifd(offset, next_offset, orientation):
        entries = [(256, 4, 1, 1), (257, 4, 1, 1), (258, 3, 1, 8),
                   (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, offset),
                   (274, 3, 1, orientation), (278, 4, 1, 1), (279, 4, 1, 1)]
        return (struct.pack('<H', len(entries))
                + b''.join(struct.pack('<HHII', *entry) for entry in entries)
                + struct.pack('<I', next_offset))

    ifd_size = 2 + 9 * 12 + 4
    second = 8 + ifd_size
    pixels = second + ifd_size
    path = tmp_path / 'pages.tif'
    path.write_bytes(b'II' + struct.pack('<HI', 42, 8)
                     + ifd(pixels, second, 1) + ifd(pixels + 1, 0, 6) + b'\0\0')
    with pytest.raises(ValueError, match='page 2'):
        validate_sensor_source_orientation([path])


def test_sensor_orientation_rejects_raw_unknown_and_cancellation(tmp_path):
    raw = tmp_path / 'source.dng'
    raw.write_bytes(b'raw')
    unknown = tmp_path / 'broken.jpg'
    unknown.write_bytes(b'not an image')
    with pytest.raises(ValueError, match='RAW'):
        validate_sensor_source_orientation([raw])
    with pytest.raises(ValueError, match='cannot be inspected'):
        validate_sensor_source_orientation([unknown])
    with pytest.raises(SensorCleanupCancelled):
        validate_sensor_source_orientation([unknown], lambda: True)
