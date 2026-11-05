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
    assert isinstance(materialized[0], np.memmap)
    expected = cv2.resize(cv2.rotate(original_decoder(path), cv2.ROTATE_90_CLOCKWISE),
                          (450, 600), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(materialized[0], expected)


def test_tiff_pages_keep_original_depth_and_page_identity(tmp_path, monkeypatch):
    path = str(tmp_path / '多页.tif')
    original = [np.full((48, 64, 3), value, np.uint16) for value in (1024, 50000)]
    assert cv2.imwritemulti(path, original)
    frames = sources.scan_source(path)
    assert [frame.page for frame in frames] == [0, 1]
    assert all(frame.dtype == np.uint16 for frame in frames)
    for frame, expected in zip(frames, original):
        np.testing.assert_array_equal(sources.resolve_frame(frame), expected)


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
    assert isinstance(materialized[0], np.memmap)
    np.testing.assert_array_equal(materialized[0], image)


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
    monkeypatch.setattr(batch_converter, '_embedded_raw_jpeg', lambda _path: None, raising=False)
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
                           planarconfig=1, dtype=np.dtype(np.uint16), is_memmappable=True,
                           asarray=asarray, segments=forbidden, photometric=2)
    preview = sources._tiff_preview(page)
    assert max(preview.shape[:2]) <= sources.PREVIEW_EDGE
    assert preview.dtype == np.uint16
    np.testing.assert_array_equal(preview[0, 0], [3000, 2000, 1000])
