"""Retouch records reproduce pixels and reject changed inputs transactionally."""
import json

import numpy as np
import pytest

from core.render_context import build_render_context
from utils.retouch import RetouchDocument


def document(dtype):
    base = np.full((40, 48, 3), 100 if dtype == np.uint8 else 12000, dtype)
    frames = [np.full_like(base, 190 if dtype == np.uint8 else 50000),
              np.full_like(base, 40 if dtype == np.uint8 else 4000)]
    context = build_render_context(frames, base, filenames=['one', 'two'])
    doc = RetouchDocument(base, context)
    doc.initialize_checkpoint()
    return doc


def stroke(doc, tool, frame, points):
    doc.set_frame(frame)
    doc.set_tool(tool)
    doc.set_brush(9, 0.6, 0.7)
    doc.exposure = 0.5
    assert doc.begin_stroke(*points[0])
    for point in points[1:]:
        doc.stroke_to(*point)
    assert doc.end_stroke() == 'committed'


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_checkpoint_roundtrip_undo_redo_and_branch(tmp_path, dtype):
    original = document(dtype)
    stroke(original, 'source', 1, [(10, 10), (20, 15), (22, 18)])
    stroke(original, 'exposure', 0, [(15, 10), (16, 13)])
    stroke(original, 'restore', 0, [(20, 15), (22, 18)])
    assert original.adjust_tones(0.3, 0.2) == 'committed'
    final = original.working.copy()
    assert original.undo()
    current = original.working.copy()
    path = tmp_path / 'history.ofretouch'
    original.save_checkpoint(path)
    restored = document(dtype)
    restored.load_checkpoint(path)
    np.testing.assert_array_equal(restored.working, current)
    assert restored.redo()
    np.testing.assert_array_equal(restored.working, final)
    assert original.undo() and restored.undo() and restored.undo()
    np.testing.assert_array_equal(restored.working, original.working)
    stroke(restored, 'source', 0, [(5, 5), (8, 9)])
    assert not restored.can_redo
    branch = tmp_path / 'branch.ofretouch'
    restored.save_checkpoint(branch)
    again = document(dtype)
    again.load_checkpoint(branch)
    np.testing.assert_array_equal(again.working, restored.working)
    assert not again.can_redo
    original_again = document(dtype)
    original_again.load_checkpoint(path)
    np.testing.assert_array_equal(original_again.working, current)


def test_changed_source_and_cancel_never_modify_current_document(tmp_path):
    source = document(np.uint16)
    stroke(source, 'source', 0, [(10, 10), (20, 20)])
    path = tmp_path / 'history.ofretouch'
    source.save_checkpoint(path)
    target = document(np.uint16)
    stroke(target, 'source', 1, [(5, 5), (6, 6)])
    before = target.working.copy()
    target.frames[0].image[2, 2] += 1
    with pytest.raises(ValueError):
        target.load_checkpoint(path)
    np.testing.assert_array_equal(target.working, before)
    target.frames[0].image[2, 2] -= 1
    with pytest.raises(ValueError):
        target.load_checkpoint(path, cancelled=lambda: True)
    np.testing.assert_array_equal(target.working, before)
    assert target.undo()


def test_geometry_and_malformed_operation_are_rejected_without_edits(tmp_path):
    source = document(np.uint8)
    stroke(source, 'source', 0, [(10, 10), (20, 20)])
    path = tmp_path / 'history.ofretouch'
    source.save_checkpoint(path)
    target = document(np.uint8)
    target.frames[0].offset_x = 1
    with pytest.raises(ValueError):
        target.load_checkpoint(path)
    np.testing.assert_array_equal(target.working, target.base)
    target.frames[0].offset_x = 0
    record = json.loads(path.read_text(encoding='utf-8'))
    record['operations'][0]['points'][0] = ['bad', 0]
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError):
        target.load_checkpoint(path)
    np.testing.assert_array_equal(target.working, target.base)


def test_aligned_mask_offset_restore_all_and_evicted_history(tmp_path):
    from core.render_context import RenderContext, SourceFrame

    base = np.full((32, 32, 3), 90, np.uint8)
    image = np.full((20, 20, 3), 200, np.uint8)
    image[0, :] = 0

    def make():
        frame = SourceFrame('frame', 'source', image.copy(), 5, 6, True)
        context = RenderContext('id', 'test', (), 32, 32, base.dtype, (frame,))
        doc = RetouchDocument(base, context, max_strokes=2)
        doc.initialize_checkpoint()
        return doc

    doc = make()
    stroke(doc, 'source', 0, [(10, 10), (15, 15)])
    stroke(doc, 'exposure', 0, [(12, 12), (14, 14)])
    assert doc.restore_all()
    assert doc.undo()
    before = doc.working.copy()
    path = tmp_path / 'aligned.ofretouch'
    doc.save_checkpoint(path)
    loaded = make()
    loaded.load_checkpoint(path)
    np.testing.assert_array_equal(loaded.working, before)
    assert loaded.redo()
    np.testing.assert_array_equal(loaded.working, base)
    assert loaded.undo() and loaded.undo() and not loaded.undo()
    mismatch = make()
    mismatch.frames[0].aligned = False
    with pytest.raises(ValueError):
        mismatch.load_checkpoint(path)
    np.testing.assert_array_equal(mismatch.working, base)


@pytest.mark.parametrize('change_source', [False, True])
def test_edit_during_initial_scan_is_verified_before_save(monkeypatch, tmp_path, change_source):
    import threading
    import utils.retouch_checkpoint as checkpoint

    base = np.full((24, 24, 3), 80, np.uint8)
    frame = np.full_like(base, 180)
    doc = RetouchDocument(base, build_render_context([frame], base))
    entered, release = threading.Event(), threading.Event()
    original = checkpoint.provenance

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint, 'provenance', delayed)
    errors = []
    def initialize():
        try:
            doc.initialize_checkpoint()
        except ValueError as exc:
            errors.append(exc)
    worker = threading.Thread(target=initialize)
    worker.start()
    assert entered.wait(5)
    stroke(doc, 'source', 0, [(8, 8), (10, 10)])
    if change_source:
        frame[:] = 220
    release.set()
    worker.join(5)
    assert not worker.is_alive() and not errors and doc.checkpoint_ready
    assert doc.undo() and doc.redo()
    path = tmp_path / 'history.ofretouch'
    if change_source:
        with pytest.raises(ValueError, match='does not match'):
            doc.save_checkpoint(path)
        assert not path.exists()
    else:
        doc.save_checkpoint(path)
        restored = RetouchDocument(base, build_render_context([frame], base))
        restored.initialize_checkpoint()
        restored.load_checkpoint(path)
        np.testing.assert_array_equal(restored.working, doc.working)


def test_cancel_save_after_fsync_preserves_existing_checkpoint(tmp_path, monkeypatch):
    import utils.retouch_checkpoint as checkpoint

    doc = document(np.uint8)
    stroke(doc, 'source', 0, [(8, 8), (10, 10)])
    path = tmp_path / 'existing.ofretouch'
    path.write_bytes(b'original')
    cancelled = [False]
    def cancel_after_sync(fd):
        cancelled[0] = True
    monkeypatch.setattr(checkpoint.os, 'fsync', cancel_after_sync)
    with pytest.raises(ValueError, match='cancelled'):
        doc.save_checkpoint(path, lambda: cancelled[0])
    assert path.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [path]


def test_record_limit_reports_failure_without_changing_result():
    doc = document(np.uint8)
    stroke(doc, 'source', 0, [(8, 8), (10, 10)])
    before = doc.working.copy()
    doc._point_count = 250000
    assert not doc.begin_stroke(12, 12)
    assert doc.last_status == 'record_limit'
    np.testing.assert_array_equal(doc.working, before)
    assert doc.is_dirty and doc.undo()


def test_save_rejects_history_that_does_not_match_replay(tmp_path):
    doc = document(np.uint8)
    stroke(doc, 'source', 0, [(8, 8), (10, 10)])
    unchanged = doc.working.copy()
    tile = next(iter(doc.history._undo[-1].values()))
    tile[0, 0, 0] ^= 1
    path = tmp_path / 'existing.ofretouch'
    path.write_bytes(b'existing')
    with pytest.raises(ValueError, match='history does not match'):
        doc.save_checkpoint(path)
    assert path.read_bytes() == b'existing'
    np.testing.assert_array_equal(doc.working, unchanged)


def test_low_memory_preserves_checkpoint_and_current_edits(tmp_path, monkeypatch):
    from core import resource_policy

    doc = document(np.uint16)
    stroke(doc, 'source', 0, [(10, 10), (20, 20)])
    path = tmp_path / 'history.ofretouch'
    doc.save_checkpoint(path)
    saved = path.read_bytes()
    before = doc.working.copy()
    snapshot = resource_policy.ResourceSnapshot(
        total_ram=2 * 1024 ** 3, available_ram=1024 ** 3)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda: snapshot)

    with pytest.raises(resource_policy.ResourceLimitError):
        doc.save_checkpoint(path)
    assert path.read_bytes() == saved
    np.testing.assert_array_equal(doc.working, before)
    assert doc.can_undo

    with pytest.raises(resource_policy.ResourceLimitError):
        doc.load_checkpoint(path)
    np.testing.assert_array_equal(doc.working, before)
    assert doc.can_undo


def test_legacy_empty_stroke_retains_its_history_through_migration(tmp_path):
    from utils.retouch_checkpoint import provenance

    doc = document(np.uint16)
    operation = {'type': 'stroke', 'frame': 0, 'tool': 'restore',
                 'diameter': 9, 'hardness': 0.6, 'opacity': 0.7,
                 'exposure': 0.5, 'points': [[10, 10], [12, 12]]}
    record = {'version': 1, **provenance(doc), 'operations': [operation],
              'cursor': 1, 'undo_count': 1, 'redo_count': 0,
              'max_strokes': 100, 'max_bytes': doc.history.max_bytes}
    path = tmp_path / 'legacy-empty.ofretouch'
    path.write_text(json.dumps(record), encoding='utf-8')
    doc.load_checkpoint(path)
    np.testing.assert_array_equal(doc.working, doc.base)
    assert doc.can_undo and not doc.can_redo and not doc.is_dirty
    assert doc.undo()
    migrated = tmp_path / 'migrated-empty.ofretouch'
    doc.save_checkpoint(migrated)
    restored = document(np.uint16)
    restored.load_checkpoint(migrated)
    assert not restored.can_undo and restored.can_redo
    assert restored.redo() and restored.can_undo
    np.testing.assert_array_equal(restored.working, doc.base)


@pytest.mark.parametrize('dtype,maximum', [(np.uint8, 255), (np.uint16, 65535),
                                           (np.float32, 255)])
def test_tones_channels_ranges_and_legacy_math(dtype, maximum):
    from utils.retouch import adjust_tones

    image = np.array([[[.2, .4, .7, .123], [.8, .8, .8, .777]]]) * maximum
    image = image.astype(dtype)
    legacy = image.astype(np.float32)
    legacy[..., :3] *= 2 ** .3
    legacy[..., :3] -= maximum / 2
    legacy[..., :3] *= 2 ** -.2
    legacy[..., :3] += maximum / 2
    np.clip(legacy[..., :3], 0, maximum, out=legacy[..., :3])
    if np.issubdtype(dtype, np.integer):
        np.rint(legacy[..., :3], out=legacy[..., :3])
    np.testing.assert_array_equal(adjust_tones(image, .3, -.2), legacy.astype(dtype))
    result = adjust_tones(image, 0, 0, highlights=.6, shadows=-.4,
                          saturation=.7, temperature=.8)
    np.testing.assert_array_equal(result[..., 3], image[..., 3])
    assert result.dtype == dtype and np.all((result >= 0) & (result <= maximum))
    assert not np.array_equal(result[..., :3], image[..., :3])


def test_tones_luminance_masks_saturation_and_temperature():
    from utils.retouch import adjust_tones

    gray = np.array([[0, 32, 128, 224, 255]], np.uint8)
    high = adjust_tones(gray, 0, 0, highlights=1)
    shadow = adjust_tones(gray, 0, 0, shadows=1)
    assert high[0, 3] - gray[0, 3] > high[0, 1] - gray[0, 1]
    assert shadow[0, 1] - gray[0, 1] > shadow[0, 3] - gray[0, 3]
    assert high[0, 0] == shadow[0, 0] == 0
    assert high[0, 4] == shadow[0, 4] == 255
    np.testing.assert_array_equal(adjust_tones(gray, 0, 0, saturation=1, temperature=1), gray)
    color = np.array([[[40, 100, 160]]], np.uint8)
    desaturated = adjust_tones(color, 0, 0, saturation=-1)
    assert desaturated[0, 0, 0] == desaturated[0, 0, 1] == desaturated[0, 0, 2]
    warm = adjust_tones(color, 0, 0, temperature=1)
    cold = adjust_tones(color, 0, 0, temperature=-1)
    assert warm[0, 0, 0] < color[0, 0, 0] < cold[0, 0, 0]
    assert cold[0, 0, 2] < color[0, 0, 2] < warm[0, 0, 2]


@pytest.mark.parametrize('name', ['highlights', 'shadows', 'saturation', 'temperature'])
@pytest.mark.parametrize('value', [-1.01, 1.01, float('nan'), float('inf')])
def test_tones_reject_invalid_values(name, value):
    from utils.retouch import adjust_tones

    doc = document(np.uint8)
    with pytest.raises(ValueError):
        adjust_tones(doc.working, 0, 0, **{name: value})
    with pytest.raises(ValueError):
        doc.adjust_tones(0, 0, **{name: value})
    np.testing.assert_array_equal(doc.working, doc.base)
    assert not doc.can_undo


def test_extended_tones_undo_replay_and_cancel(tmp_path):
    base = np.full((260, 260, 3), (50, 110, 190), np.uint8)
    context = build_render_context([base.copy()], base, filenames=['one'])
    doc = RetouchDocument(base, context)
    doc.initialize_checkpoint()
    settings = dict(highlights=.7, shadows=-.5, saturation=.4, temperature=-.3)
    calls = iter([False, False, True])
    assert doc.adjust_tones(0, 0, lambda: next(calls), **settings) == 'cancelled'
    np.testing.assert_array_equal(doc.working, base)
    assert not doc.can_undo
    assert doc.adjust_tones(0, 0, **settings) == 'committed'
    edited = doc.working.copy()
    assert doc.undo() and not doc.can_undo
    np.testing.assert_array_equal(doc.working, base)
    path = tmp_path / 'tones.json'
    doc.save_checkpoint(path)
    restored = RetouchDocument(base, context)
    restored.initialize_checkpoint()
    restored.load_checkpoint(path)
    assert restored.redo()
    np.testing.assert_array_equal(restored.working, edited)


def test_legacy_tone_checkpoint_and_invalid_new_fields(tmp_path):
    from utils.retouch_checkpoint import provenance

    doc = document(np.uint8)
    operation = {'type': 'tones', 'exposure': .3, 'contrast': -.2}
    record = {'version': 1, **provenance(doc), 'operations': [operation],
              'cursor': 1, 'undo_count': 1, 'redo_count': 0,
              'max_strokes': 100, 'max_bytes': doc.history.max_bytes}
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    doc.load_checkpoint(path)
    expected = document(np.uint8)
    assert expected.adjust_tones(.3, -.2) == 'committed'
    np.testing.assert_array_equal(doc.working, expected.working)
    for name in ('highlights', 'shadows', 'saturation', 'temperature'):
        operation[name] = 1.01
        path.write_text(json.dumps(record), encoding='utf-8')
        with pytest.raises(ValueError):
            doc.load_checkpoint(path)
        np.testing.assert_array_equal(doc.working, expected.working)
        del operation[name]


def test_replay_borrows_owned_base_and_keeps_working_independent(monkeypatch):
    from core import frame_storage
    original = document(np.uint16)
    allocations = []
    store = frame_storage.store_frame

    def track(image, **kwargs):
        allocations.append(image)
        return store(image, **kwargs)

    monkeypatch.setattr(frame_storage, 'store_frame', track)
    replay = RetouchDocument._for_replay(original, max_strokes=10, max_bytes=100000)
    assert replay.base is original.base
    assert len(allocations) == 1 and allocations[0] is original.base
    replay.working[:] = 0
    assert np.all(original.base == 12000)
    assert np.all(original.working == 12000)


def test_snapshot_replay_allocates_working_without_copying_base(monkeypatch):
    from core import frame_storage
    original = document(np.uint8)
    def fail_copy(*args, **kwargs):
        raise AssertionError('snapshot replay copied its base')
    monkeypatch.setattr(frame_storage, 'store_frame', fail_copy)
    replay = RetouchDocument._for_replay(original, max_strokes=10,
                                          max_bytes=100000, snapshot=True)
    assert replay.base is original.base
    replay.working[:] = 7
    assert np.all(original.base == 100) and np.all(original.working == 100)


def test_document_snapshot_isolates_noncontiguous_input(monkeypatch):
    from core import frame_storage
    monkeypatch.setattr(frame_storage, '_try_ram_frame', lambda *args: None)
    pixels = np.full((40, 96, 3), 75, np.uint8)
    base = pixels[:, ::2]
    context = build_render_context([base.copy()], base, filenames=['one'])
    doc = RetouchDocument(base, context)
    assert isinstance(doc.base, np.memmap) and isinstance(doc.working, np.memmap)
    pixels[:] = 5
    assert np.all(doc.base == 75) and np.all(doc.working == 75)
    doc.working[:] = 10
    assert np.all(doc.base == 75)
