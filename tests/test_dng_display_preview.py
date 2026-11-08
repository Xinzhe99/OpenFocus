"""Real LibRaw-readable sensor files and selected display recovery boundaries."""
from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rawpy
import tifffile
from PIL import Image

from core import source_frames as sources
from utils import linear_dng
from tests.test_cfa_dng import MATRIX, Sensor, export_sensor


@pytest.fixture(params=['cfa', 'linear'])
def no_preview_dng(request, tmp_path, monkeypatch):
    if request.param == 'cfa':
        return export_sensor(tmp_path, monkeypatch, Sensor())
    path = tmp_path / '无预览.dng'
    pixels = np.empty((128, 160, 3), np.uint16)
    pixels[:] = (7000, 11000, 14000)
    pixels[20:90, 40:120] = (14000, 5000, 9000)
    metadata = linear_dng.LinearDNGMetadata(MATRIX, (.5, 1., 2 / 3), 'Nikon', 'Nikon Synthetic')
    linear_dng.write_linear_dng(path, pixels, metadata)
    return path


@pytest.mark.parametrize('linear', [False, True])
def test_existing_dng_recovers_real_content_and_develops_only_once(
        no_preview_dng, monkeypatch, linear):
    frame = sources.scan_source(no_preview_dng, linear=linear)[0]
    assert not frame.preview_is_available
    before = no_preview_dng.read_bytes()
    cache_bytes = sources.SOURCE_CACHE.bytes
    developed = []
    original = linear_dng.develop_linear

    def develop(image, metadata):
        developed.append(image.shape)
        return original(image, metadata)

    monkeypatch.setattr(linear_dng, 'develop_linear', develop)
    actual = sources.ensure_display_preview(frame)
    assert frame.preview_is_available and frame.preview_is_display
    assert max(actual.shape[:2]) <= sources.PREVIEW_EDGE
    assert actual.dtype == np.uint16 and np.mean(actual) > 500
    assert np.ptp(actual) > 1000
    np.testing.assert_array_equal(sources.display_image(frame), actual)
    np.testing.assert_array_equal(sources.ensure_display_preview(frame), actual)
    assert len(developed) == int(linear)
    assert sources.SOURCE_CACHE.bytes == cache_bytes
    assert no_preview_dng.read_bytes() == before


@pytest.mark.parametrize('flip', [0, 6])
def test_conversion_embeds_bounded_display_ifd_without_changing_sensor_or_profile(
        tmp_path, monkeypatch, flip):
    sensor = Sensor(flip=flip)
    baseline = export_sensor(tmp_path, monkeypatch, sensor)
    with tifffile.TiffFile(baseline) as tif:
        profile = {code: tif.pages[0].tags[code].value for code in
                   (271, 272, 274, 33421, 33422, 50706, 50707, 50708, 50713,
                    50714, 50717, 50719, 50720, 50721, 50728, 50778, 50829)}
    rgb = np.empty((1000, 1600, 3), np.uint8)
    rgb[:] = (220, 70, 20)
    rgb[:500, :800] = (10, 150, 240)
    encoded = BytesIO()
    Image.fromarray(rgb).save(encoded, format='JPEG')
    sensor.extract_thumb = lambda: SimpleNamespace(format=rawpy.ThumbFormat.JPEG,
                                                   data=encoded.getvalue())
    output = export_sensor(tmp_path, monkeypatch, sensor)
    with tifffile.TiffFile(output) as tif:
        root = tif.pages[0]
        assert len(tif.pages) == 1 and len(root.pages) == 1
        display = root.pages[0]
        assert display.photometric == 2 and display.is_reduced
        assert display.compression == 1 and display.dtype == np.uint8
        assert max(display.shape[:2]) <= sources.PREVIEW_EDGE
        assert display.tags[50970].value == 2
        assert display.tags[274].value == root.tags[274].value
        assert max(display.shape[:2]) <= max(sensor.sizes.width, sensor.sizes.height)
        assert all(root.tags[code].value == value for code, value in profile.items())
        np.testing.assert_array_equal(root.asarray(), sensor.raw_image_visible)
        assert 50721 not in display.tags and 50728 not in display.tags
    with open(output, 'rb') as stream, rawpy.imread(stream) as raw:
        np.testing.assert_array_equal(raw.raw_image, sensor.raw_image_visible)
        assert raw.postprocess(output_bps=16).shape == (128, 128, 3)
    frame = sources.scan_source(output, linear=True)[0]
    assert frame.preview_is_available and frame.preview_is_display
    # The unrotated IFD is displayed using the unchanged sensor orientation.
    with Image.open(BytesIO(encoded.getvalue())) as image:
        expected = sources._pil_preview(image)
        expected = sources._small(expected, edge=128)
    if flip == 6:
        expected = cv2.rotate(expected, cv2.ROTATE_90_CLOCKWISE)
    np.testing.assert_array_equal(sources.display_image(frame), expected)
    copied = tmp_path / 'verbatim.dng'
    linear_dng.convert_raw_to_dng(copied, output)
    assert copied.read_bytes() == output.read_bytes()


def test_linear_conversion_display_has_oriented_pixels_and_exact_profile(tmp_path, monkeypatch):
    source = tmp_path / 'camera.raw'
    source.write_bytes(b'non-CFA container')
    output = tmp_path / 'linear-preview.dng'
    sensor = Sensor(flip=6)
    sensor.raw_type = rawpy.RawType.Stack
    sensor.extract_thumb = lambda: SimpleNamespace(
        format=rawpy.ThumbFormat.BITMAP,
        data=np.full((40, 80, 3), (230, 80, 30), np.uint8))
    pixels = np.full((128, 160, 3), (7000, 11000, 14000), np.uint16)
    metadata = linear_dng.LinearDNGMetadata(MATRIX, (.5, 1., 2 / 3), 'Nikon', 'Nikon Synthetic')
    with monkeypatch.context() as context:
        context.setattr(rawpy, 'RawPy', lambda: sensor)
        context.setattr(linear_dng, 'load_linear_raw', lambda path: (pixels, metadata))
        linear_dng.convert_raw_to_dng(output, source)
    with tifffile.TiffFile(output) as tif:
        np.testing.assert_array_equal(tif.pages[0].asarray(), pixels[:, :, ::-1])
        assert tif.pages[0].pages[0].shape == (80, 40, 3)
    assert linear_dng._linear_dng_calibration(output) == metadata
    with open(output, 'rb') as stream, rawpy.imread(stream) as raw:
        assert raw.postprocess(output_bps=16).shape == pixels.shape
    frame = sources.scan_source(output, linear=True)[0]
    assert frame.preview_is_available and frame.preview_is_display
    np.testing.assert_array_equal(sources.display_image(frame)[0, 0], [30, 80, 230])


@pytest.mark.parametrize('boundary', ['cancel', 'changed', 'failed', 'calibration'])
def test_selected_decode_failure_never_publishes_pixels(
        no_preview_dng, monkeypatch, boundary):
    from core import image_loader
    linear = boundary == 'calibration'
    frame = sources.scan_source(no_preview_dng, linear=linear)[0]
    old_preview = frame.preview
    old_cache = sources.SOURCE_CACHE.bytes
    cancelled = False
    original = image_loader.decode_image

    def decode(*args, **kwargs):
        nonlocal cancelled
        result = original(*args, **kwargs)
        if boundary == 'cancel':
            cancelled = True
        elif boundary == 'changed':
            with no_preview_dng.open('ab') as stream:
                stream.write(b'changed')
        elif boundary == 'failed':
            return None
        return result

    if linear:
        original_linear = linear_dng.load_linear_raw

        def load(*args, **kwargs):
            image, metadata = original_linear(*args, **kwargs)
            return image, replace(metadata, as_shot_neutral=(1., 1., 1.))

        monkeypatch.setattr(linear_dng, 'load_linear_raw', load)
    else:
        monkeypatch.setattr(image_loader, 'decode_image', decode)
    error = {'cancel': InterruptedError, 'changed': OSError,
             'failed': ValueError, 'calibration': ValueError}[boundary]
    with pytest.raises(error):
        sources.ensure_display_preview(frame, should_cancel=lambda: cancelled)
    assert frame.preview is old_preview
    assert not frame.preview_is_available
    assert sources.SOURCE_CACHE.bytes == old_cache


def test_selected_memory_denial_precedes_sensor_unpack(no_preview_dng, monkeypatch):
    from core import image_loader, resource_policy
    frame = sources.scan_source(no_preview_dng)[0]
    attempts = []

    def denied(size, *args, **kwargs):
        attempts.append(size)
        raise resource_policy.ResourceLimitError('denied sensor allocation')

    monkeypatch.setattr(resource_policy, 'require_memory', denied)
    monkeypatch.setattr(image_loader, 'decode_image',
                        lambda *a, **k: pytest.fail('sensor unpacked without admission'))
    with pytest.raises(resource_policy.ResourceLimitError, match='denied'):
        sources.ensure_display_preview(frame)
    assert len(attempts) == 1 and attempts[0] >= 128 * 128 * 64
    assert not frame.preview_is_available


@pytest.mark.parametrize('failure', ['missing', 'malformed', 'budget'])
def test_unusable_embedded_preview_keeps_lossless_sensor_export(tmp_path, monkeypatch, failure):
    sensor = Sensor()
    if failure == 'malformed':
        sensor.extract_thumb = lambda: SimpleNamespace(format=rawpy.ThumbFormat.JPEG, data=b'bad jpeg')
    elif failure == 'budget':
        def rejected():
            raise MemoryError('bounded decoder denied')
        sensor.extract_thumb = rejected
    output = export_sensor(tmp_path, monkeypatch, sensor)
    with open(output, 'rb') as stream, rawpy.imread(stream) as raw:
        np.testing.assert_array_equal(raw.raw_image, sensor.raw_image_visible)
    with tifffile.TiffFile(output) as tif:
        assert not tif.pages[0].subifds


def test_conversion_cancellation_after_preview_does_not_write_output(tmp_path, monkeypatch):
    sensor = Sensor()
    cancelled = False

    def thumb():
        nonlocal cancelled
        cancelled = True
        return SimpleNamespace(format=rawpy.ThumbFormat.BITMAP,
                               data=np.full((40, 60, 3), 90, np.uint8))

    sensor.extract_thumb = thumb
    source, output = tmp_path / 'source.nef', tmp_path / 'cancelled.dng'
    source.write_bytes(b'keep source')
    monkeypatch.setattr(rawpy, 'RawPy', lambda: sensor)
    with pytest.raises(InterruptedError):
        linear_dng.convert_raw_to_dng(output, source, should_cancel=lambda: cancelled)
    assert not output.exists() and source.read_bytes() == b'keep source'


def test_bulk_previews_never_recover_sensors_but_selected_worker_does(
        no_preview_dng, monkeypatch, qapp):
    from PyQt6.QtCore import QThread
    from PyQt6.QtTest import QSignalSpy
    from core import image_loader, source_previews
    frames = [sources.scan_source(no_preview_dng, header_only=True)[0] for _ in range(3)]
    original = image_loader.decode_image
    decoded = []

    def decode(*args, **kwargs):
        decoded.append(QThread.currentThread() is qapp.thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_image', decode)
    bulk = source_previews.SourcePreviewWorker(frames, 'bulk')
    bulk.start()
    assert bulk.wait(10000)
    assert decoded == []
    assert all(not frame.preview_is_available for frame in frames)
    selected = source_previews.SourcePreviewWorker(frames, 'selected', [1], selected_display=True)
    ready, failed = QSignalSpy(selected.ready), QSignalSpy(selected.failed)
    selected.start()
    assert selected.wait(10000)
    qapp.processEvents()
    assert not failed and len(ready) == 1
    assert decoded == [False]
    assert [frame.preview_is_available for frame in frames] == [False, True, False]


@pytest.mark.parametrize('linear', [False, True])
def test_initial_stack_worker_recovers_only_first_source(no_preview_dng, monkeypatch, qapp, linear):
    from PyQt6.QtCore import QThread
    from PyQt6.QtTest import QSignalSpy
    from core import image_loader
    from core.workers import StackLoadWorker
    second = no_preview_dng.with_name('second.dng')
    second.write_bytes(no_preview_dng.read_bytes())
    decoded = []
    owner = linear_dng if linear else image_loader
    name = 'load_linear_raw' if linear else 'decode_image'
    original = getattr(owner, name)

    def decode(path, *args, **kwargs):
        decoded.append((str(path), QThread.currentThread() is qapp.thread()))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(owner, name, decode)
    worker = StackLoadWorker(image_loader.ImageStackLoader(),
                             filepaths=[str(no_preview_dng), str(second)], linear_raw=linear)
    result = QSignalSpy(worker.finished_load)
    worker.start()
    assert worker.wait(10000)
    qapp.processEvents()
    assert len(result) == 1 and result[0][0], result[0][1] if result else 'No load result'
    assert decoded == [(str(no_preview_dng), False)]
    first, other = result[0][2]
    assert first.preview_is_available and first.preview_is_display
    assert other.preview is None


def test_selected_cancel_discards_native_result_and_releases_admission(
        no_preview_dng, monkeypatch, qapp):
    from threading import Event
    from PyQt6.QtTest import QSignalSpy
    from core import image_loader, source_previews
    frame = sources.scan_source(no_preview_dng)[0]
    original = image_loader.decode_image
    entered, release = Event(), Event()
    old_preview = frame.preview
    cache_bytes = sources.SOURCE_CACHE.bytes

    def decode(*args, **kwargs):
        image = original(*args, **kwargs)
        entered.set()
        assert release.wait(5), 'Native decode was not released'
        return image

    monkeypatch.setattr(image_loader, 'decode_image', decode)
    old = source_previews.SourcePreviewWorker([frame], 'cancelled', [0], selected_display=True)
    ready, failed = QSignalSpy(old.ready), QSignalSpy(old.failed)
    old.start()
    try:
        assert entered.wait(5)
        old.request_cancel()
    finally:
        release.set()
        assert old.wait(10000)
    qapp.processEvents()
    assert not ready and not failed
    assert frame.preview is old_preview and not frame.preview_is_available
    assert sources.SOURCE_CACHE.bytes == cache_bytes
    # A new selected request must acquire both preview and heavy-job leases.
    current = source_previews.SourcePreviewWorker([frame], 'replacement', [0], selected_display=True)
    current_ready = QSignalSpy(current.ready)
    current.start()
    assert current.wait(10000)
    qapp.processEvents()
    assert len(current_ready) == 1 and frame.preview_is_available


def test_conversion_rejects_source_replaced_during_preview(tmp_path, monkeypatch):
    source, output = tmp_path / 'source.nef', tmp_path / 'replaced.dng'
    source.write_bytes(b'initial camera container')
    sensor = Sensor()

    def thumb():
        source.write_bytes(b'a different camera container')
        return SimpleNamespace(format=rawpy.ThumbFormat.BITMAP,
                               data=np.full((40, 60, 3), 90, np.uint8))

    sensor.extract_thumb = thumb
    monkeypatch.setattr(rawpy, 'RawPy', lambda: sensor)
    with pytest.raises(OSError, match='Source changed'):
        linear_dng.convert_raw_to_dng(output, source)
    assert not output.exists()


def test_recovered_display_applies_source_transforms_with_bounded_pixels(no_preview_dng):
    from core.image_loader import decode_image
    expected = decode_image(str(no_preview_dng))
    frame = sources.scan_source(no_preview_dng)[0]
    frame = sources.transform_frame(frame, 'rotate', code=cv2.ROTATE_90_CLOCKWISE)
    frame = sources.transform_frame(frame, 'resize', width=1200, height=600)
    expected = cv2.rotate(expected, cv2.ROTATE_90_CLOCKWISE)
    expected = cv2.resize(expected, (768, 384), interpolation=cv2.INTER_AREA)
    actual = sources.ensure_display_preview(frame)
    assert frame.shape == (600, 1200, 3) and frame.preview_is_display
    assert frame.preview_is_available and actual.shape == (384, 768, 3)
    np.testing.assert_array_equal(actual, expected)
