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
    return RetouchDocument(base, context)


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
        return RetouchDocument(base, context, max_strokes=2)

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
