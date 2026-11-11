"""Layered output retains processed pixels, geometry, masks and composite."""
import os
import cv2

import numpy as np
import pytest
from PIL import Image

from core.render_context import RenderContext, SourceFrame
from utils import layered_tiff


def _context(dtype):
    base = np.empty((18, 20, 3), dtype)
    base[:] = (21, 47, 101) if dtype == np.uint8 else (2100, 4700, 10100)
    source = np.empty((10, 11, 3), dtype)
    source[:] = (13, 67, 149) if dtype == np.uint8 else (1300, 6700, 14900)
    source[0, :] = 0
    frame = SourceFrame('first', 'échantillon.png', source, 3, 4, True)
    second = SourceFrame('second', 'second.png', base.copy())
    context = RenderContext('ctx', 'test', (), 20, 18, base.dtype, (frame, second))
    return base, context


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_layers_preserve_composite_sources_and_mask(tmp_path, monkeypatch, dtype):
    from psdtags import PsdChannelId, PsdKey, TiffImageSourceData
    import tifffile

    monkeypatch.setattr(layered_tiff, 'ram_budget', lambda snapshot: 512 * 1024 * 1024)
    monkeypatch.setattr(layered_tiff, 'probe_resources', lambda: type('Snapshot', (), {'ram_known': True})())
    image, context = _context(dtype)
    path = tmp_path / '层 照片.tif'
    layered_tiff.export_layered_tiff(path, image, context)
    # The TIFF composite is read separately from the layer-tag parser.
    with tifffile.TiffFile(path) as tif:
        composite = tif.asarray()
        assert composite.dtype == dtype
        np.testing.assert_array_equal(composite, image[..., ::-1])
    if dtype == np.uint8:
        with Image.open(path) as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), image[..., ::-1])
    metadata = TiffImageSourceData.fromtiff(path)
    assert metadata.layers.key == (PsdKey.LAYER if dtype == np.uint8 else PsdKey.LAYER_16)
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(decoded, image)
    assert len(metadata.layers.layers) == 3
    result, frame, second = metadata.layers.layers
    assert not int(result.flags) & 2 and int(frame.flags) & 2
    assert tuple(frame.rectangle) == (4, 3, 14, 14)
    assert any(getattr(info, 'value', '') == 'échantillon.png' for info in frame.info)
    assert int(second.flags) & 2
    assert any(getattr(info, 'value', '') == 'second.png' for info in second.info)
    channel = {c.channelid: c.data for c in frame.channels}
    np.testing.assert_array_equal(channel[PsdChannelId.CHANNEL0], context.frames[0].image[..., 2])
    np.testing.assert_array_equal(channel[PsdChannelId.CHANNEL2], context.frames[0].image[..., 0])
    alpha = channel[PsdChannelId.TRANSPARENCY_MASK]
    assert not alpha[0].any() and np.all(alpha[1:] == np.iinfo(dtype).max)


def test_preflight_rejects_invalid_context_and_never_overwrites_source(tmp_path, monkeypatch):
    monkeypatch.setattr(layered_tiff, 'ram_budget', lambda snapshot: 512 * 1024 * 1024)
    monkeypatch.setattr(layered_tiff, 'probe_resources', lambda: type('Snapshot', (), {'ram_known': True})())
    image, context = _context(np.uint8)
    source_path = tmp_path / 'source.tif'
    source_path.write_bytes(b'original')
    context.frames[0].name = str(source_path)
    with pytest.raises(ValueError, match='source image'):
        layered_tiff.export_layered_tiff(source_path, image, context)
    assert source_path.read_bytes() == b'original'
    context.frames[0].offset_x = 100
    with pytest.raises(ValueError, match='geometry'):
        layered_tiff.export_layered_tiff(tmp_path / 'out.tif', image, context)
    context.frames[0].offset_x = 3
    context.frames[0].image = context.frames[0].image.astype(np.uint16)
    with pytest.raises(ValueError, match='bit depth'):
        layered_tiff.export_layered_tiff(tmp_path / 'out.tif', image, context)
    assert not (tmp_path / 'out.tif').exists()


def test_admission_and_cancellation_keep_existing_file(tmp_path, monkeypatch):
    image, context = _context(np.uint16)
    path = tmp_path / 'existing.tif'
    path.write_bytes(b'previous')
    monkeypatch.setattr(layered_tiff, 'probe_resources', lambda: type('Snapshot', (), {'ram_known': True})())
    monkeypatch.setattr(layered_tiff, 'ram_budget', lambda snapshot: 1)
    with pytest.raises(ValueError, match='memory budget'):
        layered_tiff.export_layered_tiff(path, image, context)
    monkeypatch.setattr(layered_tiff, 'probe_resources', lambda: type('Snapshot', (), {'ram_known': False})())
    with pytest.raises(ValueError, match='memory budget'):
        layered_tiff.export_layered_tiff(path, image, context)
    assert path.read_bytes() == b'previous'
    monkeypatch.setattr(layered_tiff, 'probe_resources', lambda: type('Snapshot', (), {'ram_known': True})())
    assert path.read_bytes() == b'previous'
    monkeypatch.setattr(layered_tiff, 'ram_budget', lambda snapshot: 512 * 1024 * 1024)
    cancelled = [False]
    def interrupt_after_sync(fd):
        cancelled[0] = True
    monkeypatch.setattr(os, 'fsync', interrupt_after_sync)
    with pytest.raises(ValueError, match='cancelled'):
        layered_tiff.export_layered_tiff(path, image, context, cancelled=lambda: cancelled[0])
    assert path.read_bytes() == b'previous'
    assert list(tmp_path.iterdir()) == [path]


def test_unsupported_layer_names_and_missing_sources_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(layered_tiff, 'ram_budget', lambda snapshot: 512 * 1024 * 1024)
    monkeypatch.setattr(layered_tiff, 'probe_resources', lambda: type('Snapshot', (), {'ram_known': True})())
    image, context = _context(np.uint8)
    with pytest.raises(ValueError, match='no retained'):
        layered_tiff.export_layered_tiff(tmp_path / 'out.tif', image, None)
    context.frames[0].name = '\U0001f600.png'
    with pytest.raises(ValueError, match='name'):
        layered_tiff.export_layered_tiff(tmp_path / 'out.tif', image, context)


def test_export_preserves_rgb_profile_and_rejects_excess_frames(tmp_path, monkeypatch):
    from PIL import ImageCms
    import tifffile

    monkeypatch.setattr(layered_tiff, 'ram_budget', lambda snapshot: 512 * 1024 * 1024)
    monkeypatch.setattr(layered_tiff, 'probe_resources', lambda: type('Snapshot', (), {'ram_known': True})())
    image, context = _context(np.uint8)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    path = tmp_path / 'profile.tif'
    layered_tiff.export_layered_tiff(path, image, context, icc_profile=profile)
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].tags[34675].value == profile
    context.frames = context.frames * 129
    with pytest.raises(ValueError, match='Too many'):
        layered_tiff.export_layered_tiff(tmp_path / 'too-many.tif', image, context)


@pytest.mark.parametrize('kind', ['path', 'string', 'hardlink'])
def test_historical_source_snapshot_is_protected_before_preflight(tmp_path, kind):
    from dataclasses import replace
    image, context = _context(np.uint8)
    source = tmp_path / 'source.tif'
    source.write_bytes(b'original')
    target = source
    if kind == 'hardlink':
        target = tmp_path / 'alias.tif'
        try:
            target.hardlink_to(source)
        except OSError:
            pytest.skip('hard links unavailable')
    sources = str(source) if kind == 'string' else source
    context = replace(context, source_paths=sources)
    with pytest.raises(ValueError, match='source image'):
        layered_tiff.export_layered_tiff(target, image, context)
    assert source.read_bytes() == b'original'


def test_legacy_context_without_source_paths_protects_absolute_frame(tmp_path):
    from types import SimpleNamespace
    source = tmp_path / 'old.tif'
    source.write_bytes(b'original')
    context = SimpleNamespace(frames=(SimpleNamespace(name=str(source)),))
    with pytest.raises(ValueError, match='source image'):
        layered_tiff.export_layered_tiff(source, None, context)
    assert source.read_bytes() == b'original'


def test_legacy_context_without_frames_reports_missing_sources(tmp_path):
    from types import SimpleNamespace
    image, _ = _context(np.uint8)
    with pytest.raises(ValueError, match='no retained'):
        layered_tiff.export_layered_tiff(tmp_path / 'out.tif', image, SimpleNamespace())
