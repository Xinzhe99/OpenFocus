"""Original pixels are an explicit operation, independent of workspace previews."""

import cv2
import numpy as np
import pytest
from PIL import Image

from core import image_loader
from core import source_frames as sources


def _jpeg(tmp_path, name='原始图.jpg', shape=(900, 1200)):
    path = tmp_path / name
    y, x = np.indices(shape)
    pixels = np.stack((x % 251, y % 239, (x + y) % 241), axis=-1).astype(np.uint8)
    Image.fromarray(pixels).save(path, quality=95)
    return str(path)


def test_open_is_preview_only_with_original_metadata(tmp_path, monkeypatch):
    path = _jpeg(tmp_path)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('workspace open invoked original decoder')
    monkeypatch.setattr(image_loader, 'decode_image', forbidden)
    ok, message, frames, names = image_loader.ImageStackLoader().load_from_filepaths([path])
    assert ok, message
    assert names == ['原始图.jpg']
    assert frames[0].shape == (900, 1200, 3)
    preview = sources.display_image(frames[0])
    assert max(preview.shape[:2]) <= sources.PREVIEW_EDGE
    assert preview.shape[:2] != frames[0].shape[:2]
    assert sources.SOURCE_CACHE.bytes == 0 or sources.SOURCE_CACHE.bytes <= sources.SOURCE_CACHE.max_bytes


def test_full_pixels_use_original_and_cache_evicts_by_bytes(tmp_path, monkeypatch):
    paths = [_jpeg(tmp_path, f'{index}.jpg', (64, 80)) for index in range(3)]
    cache = sources.FrameCache(64 * 80 * 3)
    monkeypatch.setattr(sources, 'SOURCE_CACHE', cache)
    calls = []
    original = image_loader.decode_image
    def decode(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)
    monkeypatch.setattr(image_loader, 'decode_image', decode)
    frames = [sources.scan_source(path)[0] for path in paths]
    first = sources.resolve_frame(frames[0])
    assert sources.resolve_frame(frames[0]) is first
    sources.resolve_frame(frames[1])
    assert cache.bytes <= cache.max_bytes
    sources.resolve_frame(frames[0])
    assert len(calls) == 3
    assert first.shape == (64, 80, 3)
    np.testing.assert_array_equal(first, original(paths[0]))


@pytest.mark.parametrize('cached_frames', [1, 2])
def test_cache_releases_old_pixels_before_next_decode(tmp_path, monkeypatch, cached_frames):
    import weakref
    from utils.image_utils import imwrite_auto

    pixels = [np.full((32, 40, 3), value, np.uint16) for value in (257, 19001, 65001)]
    paths = [str(tmp_path / f'cache-{index}.png') for index in range(len(pixels))]
    for path, expected in zip(paths, pixels):
        assert imwrite_auto(path, expected)
    frames = [sources.scan_source(path, header_only=True)[0] for path in paths]
    cache = sources.FrameCache(cached_frames * pixels[0].nbytes)
    monkeypatch.setattr(sources, 'SOURCE_CACHE', cache)
    original = image_loader.decode_image
    decoded = []

    def decode(path, **kwargs):
        live_bytes = sum(image.nbytes for ref in decoded if (image := ref()) is not None)
        assert live_bytes + pixels[0].nbytes <= cache.max_bytes
        image = original(path, **kwargs)
        decoded.append(weakref.ref(image))
        return image

    monkeypatch.setattr(image_loader, 'decode_image', decode)
    for frame, expected in zip(frames, pixels):
        actual = sources.resolve_frame(frame)
        np.testing.assert_array_equal(actual, expected)
        del actual
    np.testing.assert_array_equal(sources.resolve_frame(frames[0]), pixels[0])


def test_oversized_frame_is_not_retained(tmp_path, monkeypatch):
    cache = sources.FrameCache(1)
    monkeypatch.setattr(sources, 'SOURCE_CACHE', cache)
    frame = sources.scan_source(_jpeg(tmp_path, shape=(30, 40)))[0]
    assert sources.resolve_frame(frame).shape == frame.shape
    assert cache.bytes == 0


def test_transforms_remain_lazy_then_materialize_exact_pixels(tmp_path, monkeypatch):
    path = _jpeg(tmp_path)
    frame = sources.scan_source(path)[0]
    original_decoder = image_loader.decode_image
    def forbidden(*_args, **_kwargs):
        raise AssertionError('transform decoded original pixels')
    monkeypatch.setattr(image_loader, 'decode_image', forbidden)
    rotated = sources.transform_frame(frame, 'rotate', code=cv2.ROTATE_90_CLOCKWISE)
    resized = sources.transform_frame(rotated, 'resize', width=450, height=600)
    assert resized.shape == (600, 450, 3)
    assert max(sources.display_image(resized).shape[:2]) <= sources.PREVIEW_EDGE
    monkeypatch.setattr(image_loader, 'decode_image', original_decoder)
    materialized = sources.materialize_frames([resized])
    expected = cv2.resize(cv2.rotate(original_decoder(path), cv2.ROTATE_90_CLOCKWISE),
                          (450, 600), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(materialized[0], expected)


def test_tiff_pages_keep_original_depth_and_page_identity(tmp_path, monkeypatch):
    path = str(tmp_path / '多页.tif')
    original = [np.full((48, 64, 3), value, np.uint16) for value in (1024, 50000)]
    import tifffile
    # OpenCV filename-based writers are not Unicode-safe on Windows; create
    # the actual requested container through Python's Unicode filesystem API.
    with tifffile.TiffWriter(path) as writer:
        for page in original:
            writer.write(page[:, :, ::-1], photometric='rgb')
    assert (tmp_path / '多页.tif').is_file()
    frames = sources.scan_source(path)
    assert [frame.page for frame in frames] == [0, 1]
    assert all(frame.dtype == np.uint16 for frame in frames)
    for frame, expected in zip(frames, original):
        np.testing.assert_array_equal(sources.resolve_frame(frame), expected)
    assert (tmp_path / '多页.tif').is_file()


def test_png_header_retains_rgb16_samples(tmp_path):
    path = str(tmp_path / '16位.png')
    original = np.full((48, 64, 3), (1000, 30000, 60000), np.uint16)
    ok, data = cv2.imencode('.png', original)
    assert ok
    data.tofile(path)
    frame = sources.scan_source(path)[0]
    assert frame.dtype == np.uint16
    np.testing.assert_array_equal(sources.resolve_frame(frame), original)


def test_changed_source_never_returns_cached_stale_pixels(tmp_path, monkeypatch):
    path = _jpeg(tmp_path, shape=(30, 40))
    frame = sources.scan_source(path)[0]
    sources.resolve_frame(frame)
    monkeypatch.setattr(sources, '_fingerprint', lambda _path: (0, 0))
    with pytest.raises(OSError, match='Source changed'):
        sources.resolve_frame(frame)


def test_pathless_result_arrays_remain_supported():
    image = np.full((24, 32, 3), 60000, np.uint16)
    assert sources.display_image(image) is image
    assert sources.resolve_frame(image) is image
    materialized = sources.materialize_frames([image])
    np.testing.assert_array_equal(materialized[0], image)


def test_materialize_parallel_window_preserves_order_and_progress(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from core import frame_storage
    from core import resource_policy

    paths = [_jpeg(tmp_path, f'parallel-{i}.jpg', (8, 10)) for i in range(4)]
    frames = [sources.scan_source(path, header_only=True)[0] for path in paths]
    monkeypatch.setattr(resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=2, max_in_flight=2))
    second_started = threading.Event()
    lock = threading.Lock()
    active = peak = 0

    def decode(frame):
        nonlocal active, peak
        index = paths.index(frame.path)
        assert frame_storage._scratch.get() is not None
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if index == 0:
                assert second_started.wait(5)
            elif index == 1:
                second_started.set()
            return np.full(frame.shape, index, frame.dtype)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(sources, 'resolve_frame', decode)
    progress = []
    result = sources.materialize_frames(frames, progress_callback=lambda done, total:
                                        progress.append((done, total)))
    assert peak == 2
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    for index, image in enumerate(result):
        np.testing.assert_array_equal(
            image, np.full(frames[index].shape, index, frames[index].dtype))


def test_materialize_waits_for_running_decode_after_failure(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from core import resource_policy

    paths = [_jpeg(tmp_path, f'failure-{i}.jpg', (8, 10)) for i in range(2)]
    frames = [sources.scan_source(path, header_only=True)[0] for path in paths]
    monkeypatch.setattr(resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=2, max_in_flight=2))
    started = threading.Event()
    completed = threading.Event()
    release = threading.Event()

    def decode(frame):
        if frame is frames[0]:
            started.set()
            assert release.wait(5)
            completed.set()
            return np.zeros(frame.shape, frame.dtype)
        assert started.wait(5)
        threading.Timer(0.1, release.set).start()
        raise ValueError('decode failure')

    monkeypatch.setattr(sources, 'resolve_frame', decode)
    with pytest.raises(ValueError, match='decode failure'):
        sources.materialize_frames(frames)
    assert completed.is_set()


def test_materialize_cancellation_waits_for_running_decode(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from core import resource_policy
    from core.registration import RegistrationCancelled

    paths = [_jpeg(tmp_path, f'cancel-{i}.jpg', (8, 10)) for i in range(2)]
    frames = [sources.scan_source(path, header_only=True)[0] for path in paths]
    monkeypatch.setattr(resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=2, max_in_flight=2))
    cancelled = threading.Event()
    released = threading.Event()
    completed = threading.Event()
    started = threading.Barrier(2)

    def decode(frame):
        started.wait(timeout=5)
        if frame is frames[0]:
            cancelled.set()
            threading.Timer(0.1, released.set).start()
        assert released.wait(5)
        if frame is frames[1]:
            completed.set()
        return np.zeros(frame.shape, frame.dtype)

    monkeypatch.setattr(sources, 'resolve_frame', decode)
    progress = []
    with pytest.raises(RegistrationCancelled):
        sources.materialize_frames(frames, cancelled.is_set,
                                   progress_callback=lambda *args: progress.append(args))
    assert completed.is_set()
    assert not progress


def test_manifest_opens_before_remaining_previews(tmp_path, monkeypatch):
    paths = [_jpeg(tmp_path, f'{index}.jpg', (90, 120)) for index in range(4)]
    original = Image.Image.thumbnail
    previews = []
    def thumbnail(image, *args, **kwargs):
        previews.append(image.size)
        return original(image, *args, **kwargs)
    monkeypatch.setattr(Image.Image, 'thumbnail', thumbnail)
    ok, message, frames, _ = image_loader.ImageStackLoader().load_from_filepaths(paths)
    assert ok, message
    assert len(previews) == 1
    assert sources.preview_ready(frames[0])
    assert all(not sources.preview_ready(frame) for frame in frames[1:])


def test_preview_pixels_do_not_pin_all_frames_in_ram(tmp_path, monkeypatch):
    cache = sources.FrameCache(90 * 120 * 3)
    monkeypatch.setattr(sources, 'PREVIEW_CACHE', cache)
    frames = [sources.scan_source(_jpeg(tmp_path, f'{index}.jpg', (90, 120)))[0]
              for index in range(5)]
    assert all(isinstance(frame.preview, sources.PreviewPixels) for frame in frames)
    for frame in frames:
        preview = sources.display_image(frame)
        assert preview.shape == (90, 120, 3)
        assert cache.bytes <= cache.max_bytes
    # Disk-owned previews remain available after eviction, without decoding originals.
    assert sources.display_image(frames[0]).shape == (90, 120, 3)
    assert cache.bytes <= cache.max_bytes


def test_raw_manifest_does_not_unpack_sensor_or_postprocess(tmp_path, monkeypatch):
    from types import SimpleNamespace
    rawpy = pytest.importorskip('rawpy')
    path = tmp_path / '原生.raf'
    path.write_bytes(b'header fixture')
    opened = []
    class HeaderRaw:
        sizes = SimpleNamespace(width=80, height=60, pixel_aspect=1.0, flip=6)
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def open_file(self, filename): opened.append(filename)
        def unpack(self): raise AssertionError('manifest unpacked RAW sensor')
        def postprocess(self, **_kwargs): raise AssertionError('manifest demosaiced RAW')
        def extract_thumb(self): raise AssertionError('manifest decoded thumbnail before publication')
    monkeypatch.setattr(rawpy, 'RawPy', HeaderRaw)
    monkeypatch.setattr(image_loader, 'is_raw_source', lambda _path: True)
    frame = sources.scan_source(str(path), header_only=True)[0]
    assert opened == [str(path)]
    assert frame.shape == (80, 60, 3)
    assert frame.orientation == 6 and frame.dtype == np.uint16
    assert not sources.preview_ready(frame)


def test_large_raw_missing_preview_never_unpacks_sensor(monkeypatch):
    from types import SimpleNamespace
    from core import batch_converter
    monkeypatch.setattr(batch_converter, '_embedded_raw_jpegs', lambda _path, **_kwargs: iter(()))
    class LargeRaw:
        sizes = SimpleNamespace(width=8000, height=6000)
        def extract_thumb(self): raise AssertionError('large native thumbnail decode')
        def unpack(self): raise AssertionError('large sensor unpack')
        def postprocess(self, **_kwargs): raise AssertionError('large RAW demosaic')
    preview = sources._raw_preview(LargeRaw(), 1, 'missing-preview.raf')
    assert max(preview.shape[:2]) <= sources.PREVIEW_EDGE
    assert preview.dtype == np.uint8


def test_large_nondraft_preview_never_calls_pillow_decoder():
    class HeaderImage:
        mode = 'RGB'
        size = (20000, 12000)
        width, height = size
        def getbands(self): return ('R', 'G', 'B')
        def draft(self, *_args): return None
        def thumbnail(self, *_args): raise AssertionError('oversized PNG/BMP decode')
        def convert(self, *_args): raise AssertionError('oversized PNG/BMP pixel conversion')
    preview = sources._pil_preview(HeaderImage(), itemsize=2)
    assert max(preview.shape[:2]) <= sources.PREVIEW_EDGE
    assert preview.dtype == np.uint8


def test_large_single_strip_tiff_preview_refuses_before_decode():
    from types import SimpleNamespace
    def forbidden(**_kwargs):
        raise AssertionError('oversized TIFF strip decoded')
    page = SimpleNamespace(imagelength=12000, imagewidth=20000, samplesperpixel=3,
                           parent=SimpleNamespace(filehandle=SimpleNamespace(path='header-only.tif')),
                           photometric=2,
                           planarconfig=1, dtype=np.dtype(np.uint16), is_memmappable=False,
                           is_tiled=False, rowsperstrip=12000, databytecounts=(1000,),
                           segments=forbidden)
    preview = sources._tiff_preview(page)
    assert max(preview.shape[:2]) <= sources.PREVIEW_EDGE


def test_mappable_tiff_preview_samples_without_segment_decoder():
    from types import SimpleNamespace
    sample = np.array([[[1000, 2000, 3000]]], np.uint16)
    # Header/mock only: broadcast_to does not allocate this virtual sensor grid.
    mapped = np.broadcast_to(sample, (4000, 5000, 3))
    def asarray(*, out):
        assert out == 'memmap'
        return mapped
    def forbidden(**_kwargs): raise AssertionError('mapped TIFF invoked strip decoder')
    page = SimpleNamespace(imagelength=4000, imagewidth=5000, samplesperpixel=3,
                           parent=SimpleNamespace(filehandle=SimpleNamespace(path='mapped-header.tif')),
                           planarconfig=1, dtype=np.dtype(np.uint16), is_memmappable=True,
                           asarray=asarray, segments=forbidden, photometric=2)
    preview = sources._tiff_preview(page)
    assert max(preview.shape[:2]) <= sources.PREVIEW_EDGE
    assert preview.dtype == np.uint16
    np.testing.assert_array_equal(preview[0, 0], [3000, 2000, 1000])


def test_progressive_jpeg_admits_original_coefficient_scratch():
    class ProgressiveHeader:
        mode = 'RGB'
        size = (6000, 4000)
        width, height = size
        info = {'progressive': True}
        def getbands(self): return ('R', 'G', 'B')
        def draft(self, _mode, target):
            self.width, self.height = target
        def thumbnail(self, *_args): raise AssertionError('progressive coefficient allocation')
        def convert(self, *_args): raise AssertionError('progressive pixels decoded')
    preview = sources._pil_preview(ProgressiveHeader())
    assert isinstance(preview, sources._UnavailablePreview)
    assert max(preview.shape[:2]) <= sources.PREVIEW_EDGE


def test_small_lzw_preview_uses_bundled_opencv_codec(tmp_path, monkeypatch):
    import tifffile
    path = str(tmp_path / '压缩预览.tif')
    expected = np.full((24, 32, 3), (1000, 22000, 57000), np.uint16)
    ok, data = cv2.imencode('.tiff', expected, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
    assert ok
    data.tofile(path)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('LZW preview required imagecodecs')
    monkeypatch.setattr(tifffile.TiffPage, 'segments', forbidden)
    frame = sources.scan_source(path)[0]
    assert frame.preview_is_available
    np.testing.assert_array_equal(sources.display_image(frame), expected)


def test_oversized_lzw_preview_never_calls_full_decoder(monkeypatch):
    from types import SimpleNamespace
    def forbidden(*_args, **_kwargs): raise AssertionError('oversized LZW decode')
    monkeypatch.setattr(cv2, 'imdecodemulti', forbidden)
    page = SimpleNamespace(imagelength=12000, imagewidth=20000, samplesperpixel=3,
                           parent=SimpleNamespace(filehandle=SimpleNamespace(path='lzw-header.tif')),
                           photometric=2,
                           planarconfig=1, dtype=np.dtype(np.uint16), compression=5, tags={})
    preview = sources._tiff_preview(page)
    assert isinstance(preview, sources._UnavailablePreview)


@pytest.mark.parametrize('orientation,code', [(3, cv2.ROTATE_180),
                                            (6, cv2.ROTATE_90_CLOCKWISE),
                                            (8, cv2.ROTATE_90_COUNTERCLOCKWISE)])
def test_tiff_orientation_is_applied_once_for_previews_and_originals(tmp_path, orientation, code):
    import tifffile
    path = str(tmp_path / f'orientation-{orientation}.tif')
    rgb = np.arange(12 * 20 * 3, dtype=np.uint16).reshape(12, 20, 3)
    with tifffile.TiffWriter(path) as writer:
        for offset in (0, 1000):
            writer.write(rgb + offset, photometric='rgb',
                         extratags=[(274, 'H', 1, orientation, False)])
    frames = sources.scan_source(path)
    for frame, offset in zip(frames, (0, 1000)):
        expected = cv2.rotate(np.ascontiguousarray((rgb + offset)[:, :, ::-1]), code)
        assert frame.shape == expected.shape
        np.testing.assert_array_equal(sources.display_image(frame), expected)
        np.testing.assert_array_equal(sources.resolve_frame(frame), expected)


def _rgb16_png_with_exif(path, placement):
    from utils.image_utils import _png_insert_exif
    y, x = np.indices((12, 20), dtype=np.uint16)
    pixels = np.stack((x * 1800 + y * 101 + 123,
                       y * 2200 + x * 61 + 456,
                       (x + y) * 900 + 789), axis=-1).astype(np.uint16)
    ok, encoded = cv2.imencode('.png', pixels)
    assert ok
    png = encoded.tobytes()
    if placement is not None:
        exif = Image.Exif()
        exif[274], exif[271], exif[272] = 6, 'Header-only camera', 'RGB16 PNG'
        tiff_exif = exif.tobytes()[6:]  # PNG eXIf contains TIFF bytes, not Exif\0\0.
        prefixed = _png_insert_exif(png, tiff_exif)
        if placement == 'before':
            png = prefixed
        else:
            assert placement == 'after' and png[-8:-4] == b'IEND'
            chunk = prefixed[33:33 + 12 + len(tiff_exif)]
            png = png[:-12] + chunk + png[-12:]
    path.write_bytes(png)
    return pixels


@pytest.mark.parametrize('placement', [None, 'before', 'after'])
def test_rgb16_png_metadata_never_loads_pixels_and_originals_remain_exact(tmp_path, monkeypatch, placement):
    from PIL import PngImagePlugin
    from utils.image_utils import read_image_exif
    path = tmp_path / '带方向的16位.png'
    original = _rgb16_png_with_exif(path, placement)
    attempts = []
    def forbidden(image, *_args, **_kwargs):
        attempts.append(image.filename)
        raise AssertionError('PNG metadata loaded image pixels')
    with monkeypatch.context() as guard:
        guard.setattr(PngImagePlugin.PngImageFile, 'load', forbidden)
        with Image.open(path) as image:
            position = image.fp.tell()
            exif = read_image_exif(image)
            assert image.fp.tell() == position
            assert exif.get(274, 1) == (1 if placement is None else 6)
        frame = sources.scan_source(path, header_only=True)[0]
        metadata = image_loader._read_exif_bytes(str(path))
        assert attempts == []
        assert frame.dtype == np.uint16 and frame.preview is None
        assert frame.orientation == (1 if placement is None else 6)
        assert frame.shape == ((12, 20, 3) if placement is None else (20, 12, 3))
        if placement is None:
            assert metadata == b''
        else:
            exported = Image.Exif()
            exported.load(metadata)
            assert exported[271] == 'Header-only camera' and exported[272] == 'RGB16 PNG'
    expected = original if placement is None else cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
    decoded = sources.resolve_frame(frame)
    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, expected)


@pytest.mark.parametrize('placement', [None, 'before', 'after'])
def test_png_stack_open_only_loads_the_first_preview_pixels(tmp_path, monkeypatch, placement):
    from pathlib import Path
    from PIL import PngImagePlugin
    paths = [tmp_path / f'帧-{index}.png' for index in range(3)]
    for path in paths:
        _rgb16_png_with_exif(path, placement)
    load = PngImagePlugin.PngImageFile.load
    decoded_paths = []
    def preview_only(image, *args, **kwargs):
        path = Path(image.filename)
        decoded_paths.append(path)
        if path != paths[0]:
            raise AssertionError('metadata decoded a non-selected PNG frame')
        return load(image, *args, **kwargs)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, 'load', preview_only)
    ok, message, frames, _ = image_loader.ImageStackLoader().load_from_filepaths(
        [str(path) for path in paths])
    assert ok, message
    assert len(frames) == 3 and set(decoded_paths) == {paths[0]}
    assert sources.preview_ready(frames[0])
    assert all(not sources.preview_ready(frame) for frame in frames[1:])
    assert all(frame.dtype == np.uint16 for frame in frames)
    assert all(frame.shape == ((12, 20, 3) if placement is None else (20, 12, 3)) for frame in frames)


def test_raw_source_dimensions_are_header_only_and_unicode_safe(tmp_path, monkeypatch):
    from types import SimpleNamespace
    rawpy = pytest.importorskip('rawpy')
    path = tmp_path / '原生尺寸.raf'
    path.write_bytes(b'RAW header fixture')
    class HeaderRaw:
        sizes = SimpleNamespace(width=80, height=60)
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def open_file(self, filename):
            assert isinstance(filename, str) and filename == str(path)
        def open_buffer(self, *_args): raise AssertionError('RAW metadata read the complete encoded file')
        def unpack(self): raise AssertionError('RAW dimensions unpacked sensor pixels')
        def postprocess(self, **_kwargs): raise AssertionError('RAW dimensions demosaiced sensor pixels')
    def forbidden(*_args, **_kwargs):
        raise AssertionError('RAW dimensions used the unpacking convenience decoder')
    monkeypatch.setattr(rawpy, 'RawPy', HeaderRaw)
    monkeypatch.setattr(rawpy, 'imread', forbidden)
    monkeypatch.setattr(image_loader, 'RAWPY_AVAILABLE', True)
    assert image_loader.source_dimensions(path) == (80, 60)


def _dng_preview_fixture(tmp_path, monkeypatch, *, bitmap=True, orientation=1,
                         compression=None, previews=None):
    """A real TIFF/DNG preview IFD beside a header-only 48MP CFA sensor."""
    from types import SimpleNamespace
    import tifffile
    rawpy = pytest.importorskip('rawpy')
    path = tmp_path / '大传感器预览.dng'
    rgb = np.arange(24 * 32 * 3, dtype=np.uint16).reshape(24, 32, 3) * 23
    previews = [rgb] if previews is None else previews
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((8, 8), np.uint16), photometric=32803, subifds=len(previews),
                     extratags=[(50706, 'B', 4, (1, 4, 0, 0), False)])
        for preview in previews:
            writer.write(preview if bitmap else np.zeros(preview.shape[:2], np.uint16),
                         photometric='rgb' if bitmap else 32803, subfiletype=1,
                         compression=compression, rowsperstrip=4,
                         extratags=[(274, 'H', 1, orientation, False)])
    # Inflate only the sensor header. Preview data stay real and small; attempting
    # to decode the CFA strip would fail rather than allocate a giant fixture.
    with tifffile.TiffFile(path, mode='r+b') as tif:
        tif.pages[0].tags['ImageWidth'].overwrite(8000)
        tif.pages[0].tags['ImageLength'].overwrite(6000)
    class HeaderRaw:
        sizes = SimpleNamespace(width=8000, height=6000, pixel_aspect=1.0, flip=6)
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def open_file(self, filename): assert filename == str(path)
        def unpack(self): raise AssertionError('preview unpacked sensor')
        def postprocess(self, **_kwargs): raise AssertionError('preview demosaiced sensor')
        def extract_thumb(self): raise AssertionError('preview allocated unadmitted LibRaw thumbnail')
    monkeypatch.setattr(rawpy, 'RawPy', HeaderRaw)
    return path, rgb


def test_raw_bitmap_preview_uses_smaller_admitted_ifd(tmp_path, monkeypatch):
    from core import resource_policy
    large = np.full((512, 768, 3), (1000, 2000, 3000), np.uint16)
    small = np.arange(120 * 160 * 3, dtype=np.uint16).reshape(120, 160, 3)
    path, _ = _dng_preview_fixture(tmp_path, monkeypatch, previews=[large, small])
    def admit(required, **_kwargs):
        if required > 1024 * 1024:
            raise resource_policy.ResourceLimitError('preview admission')
        return 1024 * 1024
    monkeypatch.setattr(resource_policy, 'require_memory', admit)
    frame = sources.scan_source(path)[0]
    assert frame.preview_is_available and frame.shape == (8000, 6000, 3)
    expected = cv2.rotate(np.ascontiguousarray(small[:, :, ::-1]), cv2.ROTATE_90_CLOCKWISE)
    np.testing.assert_array_equal(sources.display_image(frame), expected)


@pytest.mark.parametrize('orientation', [1, 3, 6, 8])
def test_large_dng_bitmap_subifd_is_real_oriented_display_only(tmp_path, monkeypatch, orientation):
    path, rgb = _dng_preview_fixture(tmp_path, monkeypatch, orientation=orientation)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('preview invoked full-resolution decoder')
    monkeypatch.setattr(image_loader, 'decode_image', forbidden)
    frame = sources.scan_source(path)[0]
    assert frame.shape == (8000, 6000, 3) and frame.dtype == np.uint16
    assert frame.preview_is_available and frame.preview_is_display
    assert frame.fingerprint == sources._fingerprint(path)
    expected = cv2.rotate(np.ascontiguousarray(rgb[:, :, ::-1]), cv2.ROTATE_90_CLOCKWISE)
    np.testing.assert_array_equal(sources.display_image(frame), expected)
    from core import resource_policy
    with monkeypatch.context() as guard:
        guard.setattr(resource_policy, 'require_memory', lambda *_args, **_kwargs: None)
        with pytest.raises(AssertionError, match='full-resolution'):
            sources.resolve_frame(frame)


def test_dng_cfa_pyramid_never_masquerades_as_display_pixels(tmp_path, monkeypatch):
    path, _ = _dng_preview_fixture(tmp_path, monkeypatch, bitmap=False)
    frame = sources.scan_source(path)[0]
    assert not frame.preview_is_available
    assert max(sources.display_image(frame).shape[:2]) <= sources.PREVIEW_EDGE


def test_dng_preview_cancellation_does_not_publish_partial_pixels(tmp_path, monkeypatch):
    import tifffile
    path, _ = _dng_preview_fixture(tmp_path, monkeypatch, compression='deflate')
    frame = sources.scan_source(path, header_only=True)[0]
    cancelled = False
    segments = tifffile.TiffPage.segments
    def cancel_between_segments(page, **kwargs):
        nonlocal cancelled
        for segment in segments(page, **kwargs):
            cancelled = True
            yield segment
    monkeypatch.setattr(tifffile.TiffPage, 'segments', cancel_between_segments)
    with pytest.raises(InterruptedError, match='cancelled'):
        sources.ensure_preview(frame, should_cancel=lambda: cancelled)
    assert frame.preview is None


def test_preview_rejects_source_replaced_between_identity_check_and_scan(tmp_path, monkeypatch):
    from dataclasses import replace
    path = _jpeg(tmp_path, shape=(30, 40))
    frame = sources.scan_source(path, header_only=True)[0]
    replacement = replace(sources.scan_source(path)[0], fingerprint=(1, 2))
    monkeypatch.setattr(sources, 'scan_source', lambda *_args, **_kwargs: [replacement])
    with pytest.raises(OSError, match='Source changed'):
        sources.ensure_preview(frame)
    assert frame.preview is None


def test_large_raf_embedded_jpeg_is_admitted_without_libraw_thumbnail(tmp_path):
    from io import BytesIO
    from types import SimpleNamespace
    import struct
    pytest.importorskip('rawpy')
    encoded = BytesIO()
    Image.new('RGB', (96, 64), (20, 80, 160)).save(encoded, format='JPEG')
    payload = encoded.getvalue()
    header = bytearray(92)
    header[:16] = b'FUJIFILMCCD-RAW '
    struct.pack_into('>II', header, 84, len(header), len(payload))
    path = tmp_path / '大传感器.raf'
    path.write_bytes(header + payload)
    class LargeRaw:
        sizes = SimpleNamespace(width=10000, height=10000)
        def extract_thumb(self): raise AssertionError('unadmitted LibRaw thumbnail')
    preview = sources._raw_preview(LargeRaw(), 6, path)
    with Image.open(BytesIO(payload)) as image:
        expected = cv2.rotate(np.asarray(image)[:, :, ::-1].copy(), cv2.ROTATE_90_CLOCKWISE)
    np.testing.assert_array_equal(preview, expected)


@pytest.mark.parametrize('rows_per_strip', [1, 24])
def test_large_raw_lzw_thumbnail_uses_only_its_encoded_ranges(tmp_path, monkeypatch, rows_per_strip):
    import tifffile
    path = tmp_path / '大容器小缩略图.tif'
    expected = np.arange(24 * 32 * 3, dtype=np.uint16).reshape(24, 32, 3) * 19
    ok, encoded = cv2.imencode('.tiff', expected, [cv2.IMWRITE_TIFF_COMPRESSION, 5,
                                               cv2.IMWRITE_TIFF_ROWSPERSTRIP, rows_per_strip])
    assert ok
    path.write_bytes(encoded.tobytes())
    # A full-container admission would reject this file even though the real
    # compressed preview is tiny. Truncation extends without a giant allocation.
    with path.open('r+b') as stream:
        stream.truncate(sources.PREVIEW_DECODE_BUDGET)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('LZW thumbnail required optional tifffile codec')
    monkeypatch.setattr(tifffile.TiffPage, 'segments', forbidden)
    with tifffile.TiffFile(path) as tif:
        actual = sources._raw_lzw_preview(tif.pages[0])
    np.testing.assert_array_equal(actual, expected)


def test_camera_linear_processing_never_uses_embedded_display_samples(tmp_path, monkeypatch):
    from dataclasses import replace
    path, _ = _dng_preview_fixture(tmp_path, monkeypatch)
    display = sources.scan_source(path)[0]
    linear = replace(display, linear_metadata=object())
    calibrated = np.full((24, 32, 3), (123, 456, 789), np.uint16)
    calls = []
    def resolve(frame):
        calls.append(frame)
        return calibrated
    monkeypatch.setattr(sources, 'resolve_frame', resolve)
    np.testing.assert_array_equal(sources.processing_preview(linear), calibrated)
    assert calls == [linear]
    assert not np.array_equal(sources.display_image(linear), calibrated)


@pytest.mark.parametrize('offset,length', [(92, 33 * 1024 * 1024), (2**31, 100), (92, 0)])
def test_raf_preview_rejects_unbounded_or_invalid_encoded_range(tmp_path, monkeypatch, offset, length):
    import struct
    from core import batch_converter
    header = bytearray(92)
    header[:16] = b'FUJIFILMCCD-RAW '
    struct.pack_into('>II', header, 84, offset, length)
    path = tmp_path / 'invalid.raf'
    path.write_bytes(header)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('invalid JPEG payload was admitted')
    monkeypatch.setattr(batch_converter.resource_policy, 'require_memory', forbidden)
    assert list(batch_converter._embedded_raw_jpegs(path)) == []


def _preview_jpeg(color=(41, 91, 161)):
    from io import BytesIO
    with BytesIO() as output:
        Image.new('RGB', (8, 6), color).save(output, format='JPEG')
        return output.getvalue()


def _interchange_tiff(tmp_path, payloads, name='interchange.tif'):
    """TIFF IFD whose 513/514 tags point at independently encoded JPEGs."""
    import tifffile
    path = tmp_path / name
    count = len(payloads)
    with tifffile.TiffWriter(path) as writer:
        writer.write(np.zeros((2, 2, 3), np.uint8), photometric='rgb',
                     extratags=[(513, 'I', count, (0,) * count, False),
                                (514, 'I', count, (0,) * count, False)])
    ranges = []
    with path.open('ab') as stream:
        for payload in payloads:
            ranges.append((stream.tell(), len(payload)))
            stream.write(payload)
    return path, ranges


def _set_interchange_tags(path, offsets, lengths):
    import tifffile
    with tifffile.TiffFile(path, mode='r+b') as tif:
        page = tif.pages[0]
        page.tags['JPEGInterchangeFormat'].overwrite(tuple(offsets))
        page.tags['JPEGInterchangeFormatLength'].overwrite(tuple(lengths))


@pytest.mark.parametrize('value,expected', [
    (163096, (163096,)),
    ((163096,), (163096,)),
    ((163096, 595456), (163096, 595456)),
    ([163096], (163096,)),
])
def test_tiff_offset_tags_accept_scalars_and_vectors(value, expected):
    from types import SimpleNamespace
    from core import batch_converter
    assert batch_converter._tiff_tag_integers(SimpleNamespace(value=value)) == expected


@pytest.mark.parametrize('value', [
    None, '', '163096', [], (), (163096, 'length'), (163096, 595456.5), (-1,),
])
def test_tiff_offset_tags_reject_malformed_values(value):
    from types import SimpleNamespace
    from core import batch_converter
    assert batch_converter._tiff_tag_integers(SimpleNamespace(value=value)) is None


def test_embedded_raw_jpegs_read_singleton_tuple_interchange_tags(tmp_path):
    import tifffile
    from core import batch_converter
    payload = _preview_jpeg()
    path, ranges = _interchange_tiff(tmp_path, [payload])
    _set_interchange_tags(path, (ranges[0][0],), (ranges[0][1],))
    with tifffile.TiffFile(path) as tif:
        # The real NEFs report these single-value tags as one-element tuples.
        assert tif.pages[0].tags['JPEGInterchangeFormat'].value == (ranges[0][0],)
    assert list(batch_converter._embedded_raw_jpegs(path)) == [payload]


def test_raw_preview_decodes_singleton_tuple_interchange_tag(tmp_path):
    from io import BytesIO
    from types import SimpleNamespace
    pytest.importorskip('rawpy')
    payload = _preview_jpeg((10, 80, 140))
    path, ranges = _interchange_tiff(tmp_path, [payload])
    _set_interchange_tags(path, (ranges[0][0],), (ranges[0][1],))
    class HeaderRaw:
        sizes = SimpleNamespace(width=8000, height=6000)
        def extract_thumb(self):
            raise AssertionError('tuple preview fell back to LibRaw')
    preview = sources._raw_preview(HeaderRaw(), 1, path)
    with Image.open(BytesIO(payload)) as image:
        expected = np.asarray(image)[:, :, ::-1]
    np.testing.assert_array_equal(preview, expected)


def test_embedded_raw_jpegs_accept_paired_multiple_interchange_values(tmp_path):
    from core import batch_converter
    payloads = [_preview_jpeg((220, 30, 40)), _preview_jpeg((20, 180, 60))]
    path, ranges = _interchange_tiff(tmp_path, payloads)
    _set_interchange_tags(path, [item[0] for item in ranges], [item[1] for item in ranges])
    assert list(batch_converter._embedded_raw_jpegs(path)) == payloads


@pytest.mark.parametrize('offset_count,length_count', [(2, 1), (1, 2)])
def test_embedded_raw_jpegs_reject_unequal_interchange_vector_lengths(
        tmp_path, offset_count, length_count):
    from core import batch_converter
    payloads = [_preview_jpeg((220, 30, 40)), _preview_jpeg((20, 180, 60))]
    path, ranges = _interchange_tiff(tmp_path, payloads)
    _set_interchange_tags(path,
                          [item[0] for item in ranges[:offset_count]],
                          [item[1] for item in ranges[:length_count]])
    # Neither a truncated prefix nor one broadcast length may be admitted.
    assert list(batch_converter._embedded_raw_jpegs(path)) == []


def test_embedded_raw_jpegs_ignore_malformed_interchange_ranges(tmp_path):
    from core import batch_converter
    payload = _preview_jpeg()
    path, ranges = _interchange_tiff(tmp_path, [payload])
    offset, length = ranges[0]
    file_size = path.stat().st_size
    for bad_offset, bad_length in ((0, 8), (offset, 0), (file_size + 1024, length),
                                   (offset, 33 * 1024 * 1024)):
        _set_interchange_tags(path, (bad_offset,), (bad_length,))
        assert list(batch_converter._embedded_raw_jpegs(path)) == []


def test_embedded_raw_jpegs_cancel_between_paired_values(tmp_path):
    from core import batch_converter
    payloads = [_preview_jpeg((220, 30, 40)), _preview_jpeg((20, 180, 60))]
    path, ranges = _interchange_tiff(tmp_path, payloads)
    _set_interchange_tags(path, [item[0] for item in ranges], [item[1] for item in ranges])
    cancelled = False
    iterator = batch_converter._embedded_raw_jpegs(path, should_cancel=lambda: cancelled)
    assert next(iterator) == payloads[0]
    cancelled = True
    with pytest.raises(InterruptedError, match='cancelled'):
        next(iterator)


def test_embedded_raw_jpegs_cancel_before_reading_any_range(tmp_path):
    from core import batch_converter
    payload = _preview_jpeg()
    path, _ranges = _interchange_tiff(tmp_path, [payload])
    with pytest.raises(InterruptedError, match='cancelled'):
        next(batch_converter._embedded_raw_jpegs(path, should_cancel=lambda: True))


def _multiple_raw_jpegs(tmp_path, monkeypatch):
    """Real progressive and baseline JPEG strips in a RAW TIFF container."""
    from io import BytesIO
    from types import SimpleNamespace
    import tifffile
    rawpy = pytest.importorskip('rawpy')
    path = tmp_path / '多预览.dng'
    specs = [((1024, 768), True), ((160, 120), False)]
    payloads = []
    for size, progressive in specs:
        with BytesIO() as output:
            Image.new('RGB', size, (11, 41, 211)).save(output, format='JPEG', progressive=progressive)
            payloads.append(output.getvalue())
    # Start with ordinary tiny IFDs, then point their strips at the independently
    # encoded JPEGs. This creates real YCbCr JPEG pages without imagecodecs.
    with tifffile.TiffWriter(path) as writer:
        for _ in specs:
            writer.write(np.zeros((1, 1, 3), np.uint8), photometric='rgb')
    ranges = []
    with path.open('ab') as stream:
        for payload in payloads:
            ranges.append((stream.tell(), len(payload)))
            stream.write(payload)
    with tifffile.TiffFile(path, mode='r+b') as tif:
        for page, (size, _), (offset, length) in zip(tif.pages, specs, ranges):
            for name, value in (('ImageWidth', size[0]), ('ImageLength', size[1]),
                                ('Compression', 7), ('PhotometricInterpretation', 6),
                                ('StripOffsets', offset), ('StripByteCounts', length),
                                ('RowsPerStrip', size[1])):
                page.tags[name].overwrite(value)
    class HeaderRaw:
        sizes = SimpleNamespace(width=8000, height=6000, pixel_aspect=1.0, flip=6)
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def open_file(self, filename): assert filename == str(path)
        def unpack(self): raise AssertionError('JPEG preview unpacked sensor')
        def postprocess(self, **_kwargs): raise AssertionError('JPEG preview demosaiced sensor')
        def extract_thumb(self): raise AssertionError('JPEG preview used unadmitted LibRaw extraction')
    monkeypatch.setattr(rawpy, 'RawPy', HeaderRaw)
    # Exercise the same coefficient-admission boundary with a smaller real
    # progressive image, avoiding a 144MB encoder allocation in the regression.
    monkeypatch.setattr(sources, 'PREVIEW_DECODE_BUDGET', 2 * 1024 * 1024)
    return path, payloads


@pytest.mark.parametrize('consumer', ['source', 'converter'])
def test_raw_consumers_try_smaller_jpeg_after_progressive_admission_refusal(tmp_path, monkeypatch, consumer):
    from io import BytesIO
    from PIL import JpegImagePlugin
    from core import batch_converter
    path, payloads = _multiple_raw_jpegs(tmp_path, monkeypatch)
    with Image.open(BytesIO(payloads[1])) as image:
        expected = image.transpose(Image.Transpose.ROTATE_270)
        if consumer == 'converter':
            expected.thumbnail((96, 72), Image.Resampling.LANCZOS)
        expected = np.asarray(expected).copy()
    loads = []
    original_load = JpegImagePlugin.JpegImageFile.load
    def admitted_only(image, *args, **kwargs):
        assert image.size != (1024, 768), 'rejected progressive JPEG decoded pixels'
        loads.append(image.size)
        return original_load(image, *args, **kwargs)
    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, 'load', admitted_only)
    if consumer == 'source':
        frame = sources.scan_source(path)[0]
        assert frame.preview_is_available and frame.shape == (8000, 6000, 3)
        assert frame.fingerprint == sources._fingerprint(path)
        actual = sources.display_image(frame)[:, :, ::-1]
    else:
        actual = batch_converter.thumbnail_rgb(str(path))
    np.testing.assert_array_equal(actual, expected)
    assert (160, 120) in loads


def test_raw_jpeg_fallback_cancels_before_reading_next_preview(tmp_path, monkeypatch):
    path, _ = _multiple_raw_jpegs(tmp_path, monkeypatch)
    frame = sources.scan_source(path, header_only=True)[0]
    cancelled = False
    original = sources._pil_preview
    attempts = []
    def cancel_after_admission(image, **kwargs):
        nonlocal cancelled
        attempts.append(image.size)
        result = original(image, **kwargs)
        cancelled = True
        return result
    monkeypatch.setattr(sources, '_pil_preview', cancel_after_admission)
    with pytest.raises(InterruptedError, match='cancelled'):
        sources.ensure_preview(frame, should_cancel=lambda: cancelled)
    assert frame.preview is None and attempts == [(1024, 768)]


@pytest.mark.parametrize('replace_at', ['before_decode', 'after_decode'])
def test_original_rejects_source_replaced_during_decode(tmp_path, monkeypatch, replace_at):
    import os
    path = tmp_path / '替换中的原图.png'
    Image.new('RGB', (40, 30), (20, 40, 60)).save(path)
    frame = sources.scan_source(path, header_only=True)[0]
    cache = sources.FrameCache()
    monkeypatch.setattr(sources, 'SOURCE_CACHE', cache)
    decode = image_loader.decode_image

    def replace_source():
        Image.new('RGB', (40, 30), (160, 180, 200)).save(path)
        os.utime(path, ns=(frame.fingerprint[1] + 1_000_000_000,
                           frame.fingerprint[1] + 1_000_000_000))

    def decode_during_replacement(filename, **kwargs):
        if replace_at == 'before_decode':
            replace_source()
        pixels = decode(filename, **kwargs)
        if replace_at == 'after_decode':
            replace_source()
        return pixels

    monkeypatch.setattr(image_loader, 'decode_image', decode_during_replacement)
    with pytest.raises(OSError, match='Source changed'):
        sources.resolve_frame(frame)
    assert cache.bytes == 0
    monkeypatch.setattr(image_loader, 'decode_image', decode)
    reopened = sources.scan_source(path, header_only=True)[0]
    np.testing.assert_array_equal(sources.resolve_frame(reopened),
                                  np.full((30, 40, 3), (200, 180, 160), np.uint8))


@pytest.mark.parametrize('cancel_at', ['decode', 'store'])
def test_materialization_discards_last_frame_after_cancellation(tmp_path, monkeypatch, cancel_at):
    import gc
    from core import frame_storage
    from core.registration import RegistrationCancelled
    path = _jpeg(tmp_path, shape=(30, 40))
    frame = sources.scan_source(path, header_only=True)[0]
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    monkeypatch.setattr(sources, 'SOURCE_CACHE', sources.FrameCache())
    cancelled = False
    decode, store = image_loader.decode_image, sources.store_frame

    def decode_and_cancel(filename, **kwargs):
        nonlocal cancelled
        pixels = decode(filename, **kwargs)
        cancelled = cancel_at == 'decode'
        return pixels

    def store_and_cancel(image, **kwargs):
        nonlocal cancelled
        mapped = store(image, **kwargs)
        cancelled = True
        return mapped

    monkeypatch.setattr(image_loader, 'decode_image', decode_and_cancel)
    monkeypatch.setattr(sources, 'store_frame', store_and_cancel)
    with pytest.raises(RegistrationCancelled):
        sources.materialize_frames([frame], should_cancel=lambda: cancelled)
    gc.collect()
    assert not list(tmp_path.glob('openfocus-frame-*.bin'))
    monkeypatch.setattr(sources, 'store_frame', store)
    mapped = sources.materialize_frames([frame])[0]
    np.testing.assert_array_equal(mapped, decode(path))


def test_materialization_rejects_known_stack_before_decode(monkeypatch, tmp_path):
    from core import frame_storage
    paths = [_jpeg(tmp_path, f'early-{i}.jpg', (30, 40)) for i in range(2)]
    frames = [sources.scan_source(path, header_only=True)[0] for path in paths]
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    monkeypatch.setattr(frame_storage, '_space', lambda path: (None, 0, frames[0].nbytes))
    def forbidden(*args, **kwargs):
        raise AssertionError('decoded before stack admission')
    monkeypatch.setattr(sources.SourceFrame, 'full_resolution', forbidden)
    with pytest.raises(frame_storage.ScratchSpaceError) as error:
        sources.materialize_frames(frames, staging_bytes=10)
    assert error.value.required == 2 * frames[0].nbytes + 10
    assert not list(tmp_path.glob('openfocus-frame-*.bin'))


def test_existing_maps_are_not_charged_as_new_allocations(monkeypatch, tmp_path):
    from core import frame_storage
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    existing = frame_storage.store_frame(np.arange(16, dtype=np.uint8), prefer_ram=False)
    monkeypatch.setattr(frame_storage, '_space', lambda path: (None, 0, 20))
    result = sources.materialize_frames([existing, np.zeros(10, np.uint8)], staging_bytes=10)
    assert result[0] is existing
    np.testing.assert_array_equal(result[1], np.zeros(10, np.uint8))


def test_generator_is_not_consumed_before_processing(monkeypatch, tmp_path):
    from core import frame_storage
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    first = np.arange(10, dtype=np.uint8)
    stored = []
    original = sources.store_frame
    def record(image, **kwargs):
        stored.append(True)
        return original(image, **kwargs)
    def frames():
        yield first
        assert stored, 'generator was exhausted before the first frame was stored'
        yield first
    monkeypatch.setattr(sources, 'store_frame', record)
    mapped = sources.materialize_frames(frames())
    assert len(mapped) == 2
    np.testing.assert_array_equal(mapped[1], first)


def test_scratch_headers_skip_unreadable_and_propagate_cancel(monkeypatch, tmp_path):
    from core.registration import RegistrationCancelled
    path = _jpeg(tmp_path, shape=(20, 30))
    headers = sources.scratch_source_headers([path, str(tmp_path / 'missing.jpg')])
    assert len(headers) == 1 and headers[0].shape == (20, 30, 3)
    monkeypatch.setattr(sources, 'scan_source', lambda *args, **kwargs: (_ for _ in ()).throw(InterruptedError()))
    with pytest.raises(RegistrationCancelled):
        sources.scratch_source_headers([path])
