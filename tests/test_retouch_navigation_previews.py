"""Captured source pixels for Retouch navigation, in each frame's local coordinates."""
from dataclasses import replace

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from core import source_frames as sources
from core.render_context import build_render_context
from utils import retouch


def _forbid_decode(*args, **kwargs):
    pytest.fail('Retouch preparation requested original pixels')


def test_exif_oriented_jpeg_keeps_upright_colors(tmp_path, monkeypatch):
    path = tmp_path / 'portrait.jpg'
    rgb = np.zeros((48, 80, 3), np.uint8)
    rgb[:24, :40] = (230, 30, 20)
    rgb[24:, 40:] = (10, 160, 210)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(rgb).save(path, quality=96, exif=exif)
    with Image.open(path) as image:
        expected = np.asarray(ImageOps.exif_transpose(image))[:, :, ::-1]
    source, = sources.scan_source(path)
    context = build_render_context([source], expected)
    monkeypatch.setattr(sources.SourceFrame, 'full_resolution', _forbid_decode)
    monkeypatch.setattr(sources, 'scan_source', _forbid_decode)
    doc = retouch.prepare_retouch_document(expected, context)
    np.testing.assert_array_equal(doc.source_navigation_previews[0], expected)
    assert doc.source_navigation_previews[0].dtype == np.uint8


def test_linear_capture_is_developed_once_after_exposure(tmp_path, monkeypatch):
    from utils.image_utils import to_display_uint8
    from utils.linear_dng import LinearDNGMetadata, develop_linear

    metadata = LinearDNGMetadata(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), (0.8, 1., 0.7))
    pixels = np.full((32, 48, 3), (2300, 6100, 9000), np.uint16)
    frame = sources.SourceFrame(str(tmp_path / 'linear.dng'), 0, pixels.shape, pixels.dtype,
                                pixels, (0, 0), linear_metadata=metadata)
    matched = sources.exposure_frames(sources.normalize_frames([frame]), [2.])
    context = build_render_context(matched, pixels, linear_dng_metadata=metadata)
    monkeypatch.setattr(sources.SourceFrame, 'full_resolution', _forbid_decode)
    monkeypatch.setattr(sources.ProcessedFrame, 'full_resolution', _forbid_decode)
    doc = retouch.prepare_retouch_document(pixels, context)
    expected = to_display_uint8(develop_linear(pixels * 2, metadata))
    np.testing.assert_array_equal(doc.source_navigation_previews[0], expected)
    np.testing.assert_array_equal(doc.source_previews[0], expected)
    assert doc.source_navigation_previews[0].dtype == np.uint8


def test_ecc_and_measured_exposure_capture_matches_render_roi(monkeypatch):
    from core.registration import ImageRegistration

    rng = np.random.default_rng(405)
    scene = cv2.GaussianBlur(rng.integers(15, 220, (144, 192, 3), dtype=np.uint8), (5, 5), 1)
    moved = cv2.warpAffine(scene, np.float32([[1, 0, 3], [0, 1, -2]]), (192, 144),
                           borderMode=cv2.BORDER_REFLECT) // 2
    engine = ImageRegistration('ecc', downscale_width=192, defer_warps=True)
    aligned = engine.process([scene, moved], thread_count=1)
    matched = sources.match_exposure_frames(aligned, reference_index=0)
    native = [sources.resolve_frame(frame) for frame in matched]
    assert float(native[1].mean()) > 1.8 * float(moved.mean())
    roi = (19, 23, 80, 75)
    result = np.zeros_like(native[0])
    context = build_render_context(matched, result, roi_rect=roi, roi_mode='paste', aligned=True)
    monkeypatch.setattr(sources.ProcessedFrame, 'full_resolution', _forbid_decode)
    doc = retouch.prepare_retouch_document(result, context)
    for index, image in enumerate(native):
        assert (doc.frames[index].offset_x, doc.frames[index].offset_y) == roi[:2]
        np.testing.assert_array_equal(doc.source_navigation_previews[index], image[23:98, 19:99])
        assert doc.source_navigation_previews[index].dtype == np.uint8


def test_scaled_capture_samples_fractional_roi_and_preserves_display_domain(tmp_path, monkeypatch):
    yy, xx = np.indices((20, 40))
    pixels = np.stack([xx * 5, yy * 10, 200 - xx * 5], axis=2).astype(np.uint8)
    frame = sources.SourceFrame(str(tmp_path / 'capture.dng'), 0, (200, 400, 3), np.dtype('uint16'),
                                pixels, (0, 0), preview_is_display=True)
    monkeypatch.setattr(sources.SourceFrame, 'full_resolution', _forbid_decode)
    result = np.zeros((40, 80, 3), np.uint16)
    context = build_render_context([frame], result, roi_rect=(15, 35, 80, 40), roi_mode='crop')
    doc = retouch.prepare_retouch_document(result, context)
    preview, = doc.source_navigation_previews
    assert preview.shape == (4, 8, 3)
    np.testing.assert_allclose(preview[0, 0], (7.5, 35, 192.5), atol=0.5)
    np.testing.assert_allclose(preview[-1, -1], (42.5, 65, 157.5), atol=0.5)
    assert preview.dtype == np.uint8


def test_unknown_processing_missing_pixels_and_linear_display_exposure_are_unavailable(tmp_path, monkeypatch):
    pixels = np.full((80, 100, 3), (10, 30, 60), np.uint8)
    frame = sources.SourceFrame(str(tmp_path / 'capture.dng'), 0, pixels.shape, np.dtype('uint16'),
                                pixels, (0, 0), linear_metadata=object(), preview_is_display=True)
    frames = [sources.process_frame(frame, _forbid_decode), replace(frame, preview=None),
              replace(frame, preview_is_available=False), sources.exposure_frames([frame], [2.0])[0]]
    result = np.zeros(frame.shape, np.uint16)
    context = build_render_context(frames, result)
    monkeypatch.setattr(sources, 'scan_source', _forbid_decode)
    monkeypatch.setattr(sources.SourceFrame, 'full_resolution', _forbid_decode)
    monkeypatch.setattr(sources.ProcessedFrame, 'full_resolution', _forbid_decode)
    doc = retouch.prepare_retouch_document(result, context)
    assert doc.source_navigation_previews == [None] * 4
    assert [icon is not None for icon in doc.source_previews] == [True, False, True, True]
    assert all(icon.dtype == np.uint8 for icon in doc.source_previews if icon is not None)


def test_navigation_stack_adapts_to_actual_pixel_bytes_and_preserves_strip(monkeypatch):
    # Mixed native depths exercise the byte cap independently of frame count.
    images = [np.full((180, 180, 3), i, dtype) for i, dtype in
              enumerate((np.uint8, np.uint16, np.float32) * 3)]
    context = build_render_context(images, images[0])
    budget = 370000
    monkeypatch.setattr(retouch, '_NAVIGATION_PREVIEW_BYTES', budget)
    doc = retouch.prepare_retouch_document(images[0], context)
    captures = doc.source_navigation_previews
    assert len({image.shape for image in captures}) == 1
    assert 64 < captures[0].shape[0] < 180
    assert sum(image.nbytes for image in captures) <= budget
    assert all(image.dtype == np.uint8 for image in captures)
    assert all(image.shape == (64, 64, 3) for image in doc.source_previews)
    monkeypatch.setattr(retouch, '_NAVIGATION_PREVIEW_BYTES', 0)
    doc = retouch.prepare_retouch_document(images[0], context)
    assert doc.source_navigation_previews == [None] * len(images)
    assert all(image.shape == (64, 64, 3) for image in doc.source_previews)
