"""Bounded, validated alignment pixels survive isolated render handoffs."""
import gc
from pathlib import Path

import numpy as np
import pytest

from core import frame_storage as storage, frame_resolution_cache as cache
from core import source_frames as sources
from core.registration import RegistrationCancelled


def _identity(image):
    return image


def _cached_only_child(*args):
    from core import image_loader, render_process
    import cv2
    def forbidden(*args, **kwargs):
        raise AssertionError('Retained render decoded or warped a source')
    image_loader.decode_image = forbidden
    cv2.warpPerspective = forbidden
    render_process._render_child(*args)


def _retained(tmp_path, monkeypatch):
    from utils.image_utils import imwrite_auto
    monkeypatch.setattr(storage, '_storage_directory', str(tmp_path))
    pixels = np.arange(12 * 16 * 3, dtype=np.uint16).reshape(12, 16, 3)
    path = tmp_path / 'source.tif'
    assert imwrite_auto(str(path), pixels)
    source = sources.scan_source(str(path), header_only=True)[0]
    aligned = sources.crop_frame(source, 0, 0, 16, 12)
    with cache.render_scope([source]):
        sources.resolve_frame(aligned)
        result = cache.retain_aligned_frames([aligned])[0]
    assert isinstance(result, sources.RetainedFrame)
    return result, pixels, path


def test_retained_frame_checks_source_and_cancellation_without_decode(tmp_path, monkeypatch):
    from core import image_loader
    retained, expected, path = _retained(tmp_path, monkeypatch)
    monkeypatch.setattr(image_loader, 'decode_image', lambda *a, **k: pytest.fail('decoded'))
    actual = sources.resolve_frame(retained)
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    assert sources.frame_resolution_bytes(retained) == 0
    assert sources.source_cache_growth_bytes([retained]) == 0
    with pytest.raises(RegistrationCancelled):
        sources.resolve_frame(retained, should_cancel=lambda: True)
    path.write_bytes(b'changed')
    with pytest.raises(OSError, match='Source changed'):
        sources.resolve_frame(retained)


def test_retention_is_optional_bounded_and_does_not_fill_misses(tmp_path, monkeypatch):
    retained, _, _ = _retained(tmp_path, monkeypatch)
    aligned = retained.source
    monkeypatch.setattr(sources, 'resolve_frame', lambda *a, **k: pytest.fail('resolved'))
    with cache.render_scope([aligned]) as scope:
        assert cache.retain_aligned_frames([aligned]) == [aligned]
        monkeypatch.setattr(cache, 'RETAINED_FRAME_LIMIT_BYTES', 1)
        monkeypatch.setattr(scope, 'get', lambda frame: pytest.fail('over-budget lookup'))
        assert cache.retain_aligned_frames([aligned]) == [aligned]
    assert cache.retain_aligned_frames([aligned]) == [aligned]


def test_source_change_after_fusion_forfeits_retention_without_failing(tmp_path, monkeypatch):
    from utils.image_utils import imwrite_auto
    monkeypatch.setattr(storage, '_storage_directory', str(tmp_path))
    path = tmp_path / 'source.tif'
    assert imwrite_auto(str(path), np.zeros((12, 16, 3), np.uint16))
    source = sources.scan_source(str(path), header_only=True)[0]
    aligned = sources.crop_frame(source, 0, 0, 16, 12)
    with cache.render_scope([source]):
        sources.resolve_frame(aligned)
        path.write_bytes(b'changed after fusion')
        assert cache.retain_aligned_frames([aligned]) == [aligned]


def test_unplanned_scope_is_not_planned_for_retention(tmp_path, monkeypatch):
    retained, _, _ = _retained(tmp_path, monkeypatch)
    aligned = retained.source
    with cache.render_scope([aligned]) as scope:
        monkeypatch.setattr(scope, '_plan', lambda: pytest.fail('planned'))
        assert cache.retain_aligned_frames([aligned]) == [aligned]


def test_repeated_access_snapshot_keeps_retained_frames(tmp_path, monkeypatch):
    retained, expected, _ = _retained(tmp_path, monkeypatch)
    monkeypatch.setattr(sources, 'materialize_frames',
                        lambda *a, **k: pytest.fail('copied retained pixels'))
    frames = [retained]
    assert sources.materialize_for_repeated_access(frames, required=True) is frames
    np.testing.assert_array_equal(sources.resolve_frame(frames[0]), expected)


def test_retained_mapping_transfer_preserves_provenance_and_releases(tmp_path, monkeypatch):
    from core.frame_transport import FrameEncoder, FrameDecoder
    retained, expected, _ = _retained(tmp_path, monkeypatch)
    encoder = FrameEncoder()
    payload = encoder.encode(retained)
    decoded = FrameDecoder().decode(payload)
    assert isinstance(decoded, sources.RetainedFrame)
    np.testing.assert_array_equal(sources.resolve_frame(decoded), expected)
    assert decoded.source.source.path == retained.source.source.path
    assert decoded.linear_metadata == retained.linear_metadata
    path = Path(storage.describe_frame(decoded.pixels)['path'])
    del retained, encoder, decoded
    sources.SOURCE_CACHE.clear()
    gc.collect()
    assert not path.exists()


def test_changed_processing_does_not_reuse_retained_result(tmp_path, monkeypatch):
    retained, expected, _ = _retained(tmp_path, monkeypatch)
    changed = sources.process_frame(retained, lambda pixels: pixels + np.uint16(5))
    np.testing.assert_array_equal(sources.resolve_frame(changed), expected + 5)


def test_retained_preview_stays_geometric_without_reading_full_pixels(tmp_path, monkeypatch):
    from core.aligned_previews import can_preview, registered_preview
    retained, _, _ = _retained(tmp_path, monkeypatch)
    expected = registered_preview(retained.source)
    monkeypatch.setattr(sources.RetainedFrame, 'full_resolution',
                        lambda *a, **k: pytest.fail('read full retained image'))
    assert can_preview(retained)
    actual = registered_preview(retained)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


@pytest.mark.parametrize('crop', [False, True])
def test_history_releases_retained_files_but_preserves_pixels(tmp_path, monkeypatch, crop):
    from core.render_context import build_render_context
    retained, expected, _ = _retained(tmp_path, monkeypatch)
    path = Path(storage.describe_frame(retained.pixels)['path'])
    frame = sources.crop_frame(retained, 1, 1, 8, 6) if crop else retained
    expected = expected[1:7, 1:9] if crop else expected
    context = build_render_context([frame], expected)
    assert context is not None
    del retained, frame
    sources.SOURCE_CACHE.clear()
    gc.collect()
    assert not path.exists()
    np.testing.assert_array_equal(sources.resolve_frame(context.frames[0].image), expected)


def test_spawned_repeated_render_uses_retained_pixels(qapp, tmp_path, monkeypatch):
    from tests.test_render_process import _sources, _start, _wait
    from core import render_process
    from core.workers import RenderWorker
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(storage, '_storage_directory', str(scratch))
    frames = _sources(tmp_path)
    options = dict(algorithm='openfocus_b', alignment='openfocus', thread_count=1,
                   tile_enabled=False, use_gpu=False, sort_mode='off', exposure_match=False)
    first = render_process.RenderProcessWorker(frames, **options)
    results, errors = _start(first)
    _wait(first, qapp)
    assert not errors and len(results) == 1
    aligned = results[0][0]
    assert all(isinstance(f, sources.RetainedFrame) for f in aligned)
    paths = [Path(storage.describe_frame(f.pixels)['path']) for f in aligned]
    assert all(p.exists() for p in paths)
    # A different fusion method must run in a fresh child without decoding or warping.
    options['algorithm'] = 'openfocus_c'
    reference = RenderWorker(frames, **options)
    expected = []
    reference.finished_signal.connect(lambda *args: expected.append(args))
    reference._run_admitted()
    second = render_process.RenderProcessWorker(frames, aligned_images=aligned,
        is_images_aligned=True,
        last_alignment_options=(first.alignment, first.registration_parameters, first._reverse_registration),
        **options)
    monkeypatch.setattr(render_process, '_render_child', _cached_only_child)
    newer, errors = _start(second)
    _wait(second, qapp)
    assert not errors and len(newer) == 1
    assert second.child_pid != first.child_pid
    assert second.alignment_reused
    assert all(isinstance(f, sources.RetainedFrame) and
               not isinstance(f.source, sources.RetainedFrame) for f in newer[0][0])
    np.testing.assert_array_equal(newer[0][1], expected[0][1])
    # Qt owns signal closures until the completed workers are disposed.
    from PyQt6.QtCore import QCoreApplication, QEvent
    first.deleteLater()
    second.deleteLater()
    reference.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    results.clear()
    newer.clear()
    expected.clear()
    del first, second, aligned, results, newer, reference, expected
    sources.SOURCE_CACHE.clear()
    gc.collect()
    assert all(not p.exists() for p in paths)
