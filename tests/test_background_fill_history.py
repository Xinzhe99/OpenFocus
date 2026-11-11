"""Background diagnostics and exact checkpoint replay through document consumers."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import cv2
import numpy as np
import pytest

from core.background_fill import BackgroundFillResult, MissingData, missing_data_overlay
from core.render_context import build_render_context
import utils.retouch as retouch


def _document(base=None, frames=None, **limits):
    if base is None:
        base = np.full((64, 64, 3), 80, np.uint8)
    if frames is None:
        frames = [np.full_like(base, 40), np.full_like(base, 160)]
    # A fresh session owns different arrays, so identity-based cache hits cannot
    # accidentally make a checkpoint test pass.
    frames = [frame.copy() for frame in frames]
    context = build_render_context(frames, base, filenames=['near', 'far'][:len(frames)])
    doc = retouch.RetouchDocument(base, context, **limits)
    doc.initialize_checkpoint()
    return doc


@pytest.fixture
def controlled_background(monkeypatch):
    """Exercise document state transitions with known changed/no-op results."""
    def replace(frames, fused, *, preset='auto', **kwargs):
        value = {'auto': 80, 'soft': 100, 'sharp': 140}[preset]
        current = kwargs.get('composite_onto')
        if current is None:
            current = fused
        image = current.copy() if preset == 'auto' else np.full_like(fused, value)
        layer = np.full(fused.shape[:2], value, np.uint8)
        return BackgroundFillResult(image, not np.array_equal(image, current), 0, 1,
                                    0.25, MissingData(*(layer.copy() for _ in range(4))))
    monkeypatch.setattr('core.background_fill.replace_background', replace)
    return replace


def _maps(doc):
    fill = doc.background_fill
    return None if fill is None else [layer.copy() for _, layer, _ in fill.missing.layers()]


def _assert_state(left, right):
    np.testing.assert_array_equal(left.working, right.working)
    assert left.is_dirty == right.is_dirty
    assert left.can_undo == right.can_undo
    assert left.can_redo == right.can_redo
    left_maps, right_maps = _maps(left), _maps(right)
    assert (left_maps is None) == (right_maps is None)
    if left_maps is not None:
        for expected, actual in zip(left_maps, right_maps):
            np.testing.assert_array_equal(expected, actual)


def test_noop_maps_export_and_checkpoint_without_dirty_or_undo(tmp_path, controlled_background):
    doc = _document()
    assert doc.auto_background() == 'empty'
    assert not doc.is_dirty and not doc.can_undo and not doc.can_redo
    np.testing.assert_array_equal(doc.working, doc.base)
    fill = doc.background_fill
    assert fill is not None
    overlay = missing_data_overlay(doc.working, fill.missing)
    assert overlay.shape == doc.working.shape and not np.array_equal(overlay, doc.working)
    path = tmp_path / 'analysis.ofretouch'
    doc.save_checkpoint(path)
    fresh = _document()
    fresh.load_checkpoint(path)
    _assert_state(doc, fresh)
    assert doc.adjust_tones(0.2, 0) == 'committed'
    assert doc.undo()
    assert fresh.adjust_tones(0.2, 0) == 'committed'
    assert fresh.undo()
    _assert_state(doc, fresh)


def test_noop_analysis_follows_undo_redo_restore_and_replaced_branch(tmp_path, controlled_background):
    doc = _document()
    assert doc.auto_background('soft') == 'committed'
    assert doc.auto_background() == 'empty'
    no_op_maps = _maps(doc)
    assert doc.auto_background('sharp') == 'committed'
    assert doc.undo()
    for expected, actual in zip(no_op_maps, _maps(doc)):
        np.testing.assert_array_equal(expected, actual)
    path = tmp_path / 'with-redo.ofretouch'
    doc.save_checkpoint(path)
    fresh = _document()
    fresh.load_checkpoint(path)
    _assert_state(doc, fresh)
    assert doc.redo() and fresh.redo()
    _assert_state(doc, fresh)
    assert fresh.restore_all() and fresh.background_fill is None
    assert fresh.undo() and fresh.background_fill is not None
    assert fresh.undo()
    assert fresh.adjust_tones(0.1, 0) == 'committed'
    assert not fresh.can_redo
    for expected, actual in zip(no_op_maps, _maps(fresh)):
        np.testing.assert_array_equal(expected, actual)
    branch = tmp_path / 'branch.ofretouch'
    fresh.save_checkpoint(branch)
    again = _document()
    again.load_checkpoint(branch)
    _assert_state(fresh, again)
    assert again.undo() and fresh.undo()
    _assert_state(fresh, again)


def test_tiny_history_releases_unreachable_maps_but_preserves_baseline(tmp_path, controlled_background):
    # One full tile plus two four-layer map sets fit; a second undo entry
    # and its diagnostic state do not. The count limit remains the default 100.
    doc = _document(max_bytes=64 * 64 * 11)
    references = []
    for index in range(8):
        assert doc.auto_background('soft' if index % 2 == 0 else 'sharp') == 'committed'
        references.append(weakref.ref(doc.background_fill.missing.background))
    gc.collect()
    assert all(reference() is None for reference in references[:-2])
    assert references[-2]() is not None and references[-1]() is not None
    assert doc.undo() and not doc.can_undo
    np.testing.assert_array_equal(doc.background_fill.missing.background, references[-2]())
    path = tmp_path / 'evicted.ofretouch'
    doc.save_checkpoint(path)
    fresh = _document()
    fresh.load_checkpoint(path)
    _assert_state(doc, fresh)
    assert doc.redo() and fresh.redo()
    _assert_state(doc, fresh)


def test_insufficient_diagnostic_budget_rejects_atomically(controlled_background):
    doc = _document(max_bytes=64 * 64 * 7)
    assert doc.auto_background('soft') == 'committed'
    pixels, fill = doc.working.copy(), doc.background_fill
    # The changed image's before/after maps alone cannot share this budget.
    assert doc.auto_background('sharp') == 'oversize'
    np.testing.assert_array_equal(doc.working, pixels)
    assert doc.background_fill is fill and doc.can_undo and not doc.can_redo
    assert doc.undo()
    np.testing.assert_array_equal(doc.working, doc.base)
    assert not doc.is_dirty and doc.background_fill is None
    small = _document(max_bytes=64 * 64 * 3)
    assert small.auto_background() == 'oversize'
    assert not small.is_dirty and not small.can_undo and small.background_fill is None
    high_depth = _document(np.full((64, 64, 3), 80, np.uint16),
                           max_bytes=64 * 64 * 4)
    assert high_depth.auto_background() == 'empty'
    assert high_depth.background_fill is not None
    assert not high_depth.is_dirty and not high_depth.can_undo


def test_cancelled_application_preserves_pixels_maps_and_redo(controlled_background):
    doc = _document()
    assert doc.auto_background('soft') == 'committed'
    assert doc.auto_background('sharp') == 'committed'
    assert doc.undo()
    pixels, fill = doc.working.copy(), doc.background_fill
    assert doc.auto_background('sharp', cancelled=lambda: not np.array_equal(
        doc.working, pixels)) == 'cancelled'
    np.testing.assert_array_equal(doc.working, pixels)
    assert doc.background_fill is fill and doc.can_redo
    assert doc.redo()


def test_mask_memory_rejection_precedes_materialization(monkeypatch):
    from core.resource_policy import ResourceLimitError
    doc = _document()
    for frame in doc.frames:
        frame.aligned = True
        def forbidden():
            pytest.fail('validity mask was allocated before resource admission')
        monkeypatch.setattr(frame, 'valid_mask', forbidden)
    def reject(required):
        raise ResourceLimitError('bounded test budget')
    monkeypatch.setattr('core.resource_policy.require_memory', reject)
    with pytest.raises(ResourceLimitError):
        doc.auto_background()
    assert not doc.is_dirty and not doc.can_undo and doc.background_fill is None


@pytest.fixture(scope='module')
def native_scene():
    rng = np.random.default_rng(719)
    height, width = 256, 320
    yy, xx = np.mgrid[:height, :width]
    disc = ((xx - width * 0.45) ** 2 + (yy - height * 0.5) ** 2 < 57 ** 2).astype(np.float32)
    subject = np.clip(150 + 60 * rng.standard_normal((height, width, 3)), 0, 255).astype(np.float32)
    subject[..., 0] *= 0.6
    background = cv2.GaussianBlur(rng.standard_normal((height, width, 3)).astype(np.float32), (0, 0), 1.2)
    background = np.clip(90 + 140 * background, 0, 255)
    frames, energies = [], []
    for index in range(5):
        subject_sigma = 0.3 + 1.4 * abs(index - 1)
        mask = cv2.GaussianBlur(disc, (0, 0), subject_sigma)[..., None]
        sharp = cv2.GaussianBlur(subject, (0, 0), subject_sigma)
        soft = cv2.GaussianBlur(background, (0, 0), 4 + 1.2 * (4 - index))
        frame = sharp * mask + soft * (1 - mask) + rng.normal(0, 2, subject.shape)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        gray = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        energies.append(cv2.GaussianBlur(lap * lap, (0, 0), 4))
    best = np.argmax(np.stack(energies), axis=0)
    fused = np.take_along_axis(np.stack(frames), best[None, ..., None], axis=0)[0]
    return frames, fused


def _in_qthread(call):
    from PyQt6.QtCore import QThread
    from utils.platform_utils import configure_native_worker_stack

    class Worker(QThread):
        def run(self):
            try:
                self.value = call()
            except BaseException as exc:
                self.error = exc
    worker = Worker()
    configure_native_worker_stack(worker)
    worker.start()
    assert worker.wait(120000), 'background operation did not finish'
    if hasattr(worker, 'error'):
        raise worker.error
    return worker.value


@pytest.mark.parametrize('kind', ['uint8', 'uint16', 'gray16'])
def test_five_native_background_edits_replay_cold_exactly(qapp, tmp_path, native_scene, kind):
    frames, base = native_scene
    if kind != 'uint8':
        frames = [frame.astype(np.uint16) * 257 for frame in frames]
        base = base.astype(np.uint16) * 257
    if kind == 'gray16':
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    doc = _document(base, frames)
    previous_threads = cv2.getNumThreads()
    try:
        for index in range(5):
            assert doc.adjust_tones(0.07, 0) == 'committed'
            cv2.setNumThreads(2 + index % 3)
            effective_threads = cv2.getNumThreads()
            call = lambda: doc.auto_background('soft' if index % 2 == 0 else 'sharp', True)
            assert (_in_qthread(call) if index % 2 == 0 else call()) == 'committed'
            assert cv2.getNumThreads() == effective_threads
        # Five background edits retain exact image and tile history across
        # worker/main-thread execution and a different native load context.
        assert doc.undo() and doc.undo()
        path = tmp_path / 'native.ofretouch'
        cv2.setNumThreads(1)
        doc.save_checkpoint(path)
        fresh = _document(base, frames)
        cv2.setNumThreads(4)
        _in_qthread(lambda: fresh.load_checkpoint(path))
        _assert_state(doc, fresh)
        while doc.can_undo:
            assert doc.undo() and fresh.undo()
            _assert_state(doc, fresh)
        while doc.can_redo:
            assert doc.redo() and fresh.redo()
            _assert_state(doc, fresh)
        if kind == 'uint8':
            inputs = tmp_path / 'inputs.npz'
            output = tmp_path / 'cold.npz'
            np.savez(inputs, base=base, frames=np.stack(frames))
            code = '''
import sys
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from core.render_context import build_render_context
from utils.retouch import RetouchDocument
source = np.load(sys.argv[1])
base, frames = source['base'], list(source['frames'])
context = build_render_context(frames, base, filenames=['near', 'far'])
doc = RetouchDocument(base, context)
doc.initialize_checkpoint()
cv2.setNumThreads(3)
with threadpool_limits(limits=2):
    doc.load_checkpoint(sys.argv[2])
    while doc.can_redo:
        doc.redo()
np.savez(sys.argv[3], image=doc.working,
         **{name: layer for name, layer, colour in doc.background_fill.missing.layers()})
'''
            subprocess.run([sys.executable, '-c', code, str(inputs), str(path), str(output)],
                           cwd=Path(__file__).resolve().parents[1], check=True,
                           capture_output=True, text=True, timeout=120,
                           env={**os.environ, 'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '3'})
            with np.load(output) as cold:
                np.testing.assert_array_equal(cold['image'], doc.working)
                for name, layer, _ in doc.background_fill.missing.layers():
                    np.testing.assert_array_equal(cold[name], layer)
    finally:
        cv2.setNumThreads(previous_threads)


def test_save_rejects_history_mismatch_even_when_final_pixels_match(
        tmp_path, monkeypatch, controlled_background):
    doc = _document()
    assert doc.auto_background('soft') == 'committed'
    assert doc.auto_background('sharp') == 'committed'
    pixels, fill = doc.working.copy(), doc.background_fill
    path = tmp_path / 'existing.ofretouch'
    path.write_bytes(b'existing checkpoint')
    import utils.retouch_checkpoint as checkpoints
    write_payload = checkpoints._write_payload
    def altered_history(archive, name, array, cancelled):
        if name.startswith('undo/1/'):
            array = array + 1
        return write_payload(archive, name, array, cancelled)
    monkeypatch.setattr(checkpoints, '_write_payload', altered_history)
    with pytest.raises(ValueError, match='history does not match'):
        doc.save_checkpoint(path)
    assert path.read_bytes() == b'existing checkpoint'
    np.testing.assert_array_equal(doc.working, pixels)
    assert doc.background_fill is fill and doc.can_undo


def test_background_checkpoint_rejects_changed_source_transactionally(
        tmp_path, controlled_background):
    doc = _document()
    assert doc.auto_background('soft') == 'committed'
    path = tmp_path / 'source-verified.ofretouch'
    doc.save_checkpoint(path)
    target = _document()
    assert target.auto_background('sharp') == 'committed'
    pixels, fill = target.working.copy(), target.background_fill
    target.frames[0].image[0, 0, 0] += 1
    with pytest.raises(ValueError, match='sources or source geometry'):
        target.load_checkpoint(path)
    np.testing.assert_array_equal(target.working, pixels)
    assert target.background_fill is fill and target.can_undo


def _legacy_record(doc, operations, cursor, undo_count, redo_count):
    from utils.retouch_checkpoint import provenance
    return {'version': 1, **provenance(doc), 'operations': operations,
            'cursor': cursor, 'undo_count': undo_count, 'redo_count': redo_count,
            'max_strokes': 100, 'max_bytes': 64 * 64 * 3}


@pytest.mark.parametrize('stacked', ['missing', 'wrong_shape'])
@pytest.mark.parametrize('analysis', [False, True])
def test_legacy_stacked_replay_rejects_missing_source_transactionally(
        tmp_path, controlled_background, stacked, analysis):
    target = _document()
    assert target.auto_background('soft') == 'committed'
    assert target.auto_background('sharp') == 'committed'
    assert target.undo()
    pixels = target.working.copy()
    history, fill = target.history, target.background_fill
    operations, cursor = list(target._operations), target._cursor
    analyses, fills = target._background_analyses, target._background_fills

    background = {'type': 'auto_background', 'preset': 'auto' if analysis else 'soft',
                  'clean': True, 'from_stacked': True}
    # Replay a prior operation before reaching the invalid background source.
    tones = {'type': 'tones', 'exposure': 0.2, 'contrast': 0}
    record = _legacy_record(target, [tones] if analysis else [tones, background],
                            1 if analysis else 2, 1 if analysis else 2, 0)
    if analysis:
        record['background_analyses'] = [{'cursor': 1, 'operation': background}]
    path = tmp_path / 'stacked-replay.ofretouch'
    path.write_text(json.dumps(record), encoding='utf-8')
    target.stacked_image = (None if stacked == 'missing'
                            else np.zeros((63, 64, 3), np.uint8))
    original_stacked = target.stacked_image

    with pytest.raises(ValueError, match='Stacked image required'):
        target.load_checkpoint(path)
    np.testing.assert_array_equal(target.working, pixels)
    assert target.history is history and target.background_fill is fill
    assert target._operations == operations and target._cursor == cursor
    assert target._background_analyses is analyses and target._background_fills is fills
    assert target.stacked_image is original_stacked
    assert target.can_undo and target.can_redo


@pytest.mark.parametrize('source', ['missing', 'wrong_shape'])
def test_interactive_background_keeps_base_fallback(monkeypatch, source):
    def replace(frames, fused, **kwargs):
        image = np.clip(fused.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        layer = np.zeros(fused.shape[:2], np.uint8)
        return BackgroundFillResult(image, True, 0, 1, 0.25,
                                    MissingData(*(layer.copy() for _ in range(4))))

    monkeypatch.setattr('core.background_fill.replace_background', replace)
    target = _document()
    target.stacked_image = None if source == 'missing' else np.zeros((1, 1, 3), np.uint8)
    assert target.auto_background('soft') == 'committed'
    np.testing.assert_array_equal(target.working, np.full_like(target.base, 100))


@pytest.mark.parametrize('cursor', [1, 2])
def test_legacy_tile_only_background_checkpoint_preserves_reachable_history(
        tmp_path, controlled_background, cursor):
    target = _document()
    # Original v1 allowed one RGB8 tile (12288 bytes), while each diagnostic
    # map set alone occupies 16384 bytes. The second edit evicted the first.
    record = _legacy_record(target, [
        {'type': 'auto_background', 'preset': 'soft', 'clean': True},
        {'type': 'auto_background', 'preset': 'sharp', 'clean': True},
    ], cursor, cursor - 1, 2 - cursor)
    path = tmp_path / 'legacy.ofretouch'
    path.write_text(json.dumps(record), encoding='utf-8')
    target.load_checkpoint(path)
    expected = 100 if cursor == 1 else 140
    np.testing.assert_array_equal(target.working, np.full_like(target.base, expected))
    np.testing.assert_array_equal(target.background_fill.missing.background,
                                  np.full(target.base.shape[:2], expected, np.uint8))
    assert target.can_undo == (cursor == 2)
    assert target.can_redo == (cursor == 1)
    if cursor == 2:
        assert target.undo()
    assert not target.can_undo and target.can_redo
    np.testing.assert_array_equal(target.working, np.full_like(target.base, 100))
    np.testing.assert_array_equal(target.background_fill.missing.background,
                                  np.full(target.base.shape[:2], 100, np.uint8))
    migrated = tmp_path / 'migrated.ofretouch'
    target.save_checkpoint(migrated)
    fresh = _document()
    fresh.load_checkpoint(migrated)
    _assert_state(target, fresh)
    assert target.redo() and fresh.redo()
    _assert_state(target, fresh)
    np.testing.assert_array_equal(fresh.working, np.full_like(fresh.base, 140))
    assert not fresh.can_redo


def test_legacy_nonbackground_checkpoint_keeps_original_budget_and_history(tmp_path):
    target = _document()
    record = _legacy_record(target, [{'type': 'tones', 'exposure': 0.2, 'contrast': 0}],
                            1, 1, 0)
    path = tmp_path / 'legacy-tones.ofretouch'
    path.write_text(json.dumps(record), encoding='utf-8')
    target.load_checkpoint(path)
    assert target.history.max_bytes == 64 * 64 * 3
    expected = _document()
    assert expected.adjust_tones(0.2, 0) == 'committed'
    _assert_state(expected, target)
    assert target.undo() and not target.can_undo
    np.testing.assert_array_equal(target.working, target.base)


def test_legacy_diagnostic_memory_rejection_is_transactional(
        tmp_path, monkeypatch, controlled_background):
    from core.resource_policy import ResourceLimitError
    target = _document()
    record = _legacy_record(target, [
        {'type': 'auto_background', 'preset': 'soft', 'clean': True},
        {'type': 'auto_background', 'preset': 'sharp', 'clean': True},
    ], 2, 1, 0)
    path = tmp_path / 'legacy-memory.ofretouch'
    path.write_text(json.dumps(record), encoding='utf-8')
    assert target.auto_background('sharp') == 'committed'
    pixels, fill = target.working.copy(), target.background_fill
    from utils.retouch_checkpoint import _aligned_mask_bytes
    pixels_count = target.height * target.width
    admission_limit = pixels_count * 4 + _aligned_mask_bytes(target)
    def admission(required, snapshot=None):
        if required > admission_limit:
            raise ResourceLimitError('legacy diagnostic budget cannot be admitted')
    def forbidden(*args, **kwargs):
        pytest.fail('background replay started before migration memory admission')
    monkeypatch.setattr('core.resource_policy.require_memory', admission)
    monkeypatch.setattr('core.background_fill.replace_background', forbidden)
    with pytest.raises(ResourceLimitError):
        target.load_checkpoint(path)
    np.testing.assert_array_equal(target.working, pixels)
    assert target.background_fill is fill and target.can_undo


@pytest.mark.parametrize('damage', [
    'duplicate', 'unexpected', 'missing', 'checksum', 'object', 'shape',
    'header_length', 'tile_coordinate', 'budget',
])
def test_exact_archive_rejects_malformed_payloads_transactionally(
        tmp_path, monkeypatch, controlled_background, damage):
    import io
    import struct
    import warnings
    import zipfile
    source = _document()
    assert source.auto_background('soft') == 'committed'
    assert source.auto_background('sharp') == 'committed'
    assert source.undo()
    original = tmp_path / 'valid.ofretouch'
    source.save_checkpoint(original)
    with zipfile.ZipFile(original) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    record = json.loads(members['metadata.json'])
    if damage == 'unexpected':
        members['../outside.npy'] = b'not a checkpoint member'
    elif damage == 'missing':
        del members[next(name for name in members if name.startswith('undo/'))]
    elif damage == 'checksum':
        data = members['working.npy']
        members['working.npy'] = data[:-1] + bytes([data[-1] ^ 1])
    elif damage in ('object', 'shape'):
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            'descr': '|O' if damage == 'object' else '|u1',
            'fortran_order': False,
            'shape': source.base.shape if damage == 'object' else (2 ** 30, 2 ** 30, 3)})
        members['working.npy'] = header.getvalue()
    elif damage == 'header_length':
        members['working.npy'] = b'\x93NUMPY\x02\x00' + struct.pack('<I', 0xffffffff)
    elif damage == 'tile_coordinate':
        record['snapshot']['undo'][0][0] = [999, 999]
    elif damage == 'budget':
        record['max_bytes'] = 1
    members['metadata.json'] = json.dumps(record).encode('utf-8')
    path = tmp_path / 'damaged.ofretouch'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Duplicate name:')
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name, data in members.items():
                archive.writestr(name, data)
            if damage == 'duplicate':
                archive.writestr('working.npy', members['working.npy'])
    target = _document()
    assert target.auto_background('soft') == 'committed'
    pixels, fill = target.working.copy(), target.background_fill
    if damage != 'checksum':
        def forbidden(*args, **kwargs):
            pytest.fail('a malformed snapshot allocated a payload array')
        monkeypatch.setattr(np, 'empty', forbidden)
    with pytest.raises(ValueError):
        target.load_checkpoint(path)
    np.testing.assert_array_equal(target.working, pixels)
    assert target.background_fill is fill and target.can_undo and not target.can_redo
    assert not (tmp_path.parent / 'outside.npy').exists()


@pytest.mark.parametrize('operation', ['save', 'load'])
def test_exact_archive_cancellation_after_payload_io_is_transactional(
        tmp_path, monkeypatch, controlled_background, operation):
    import threading
    import utils.retouch_checkpoint as checkpoints
    source = _document()
    assert source.auto_background('soft') == 'committed'
    assert source.auto_background('sharp') == 'committed'
    path = tmp_path / 'cancel.ofretouch'
    source.save_checkpoint(path)
    original = path.read_bytes()
    cancelled = threading.Event()
    target = _document()
    assert target.auto_background('soft') == 'committed'
    pixels, fill = target.working.copy(), target.background_fill
    if operation == 'save':
        write = checkpoints._write_payload
        def interrupted(*args, **kwargs):
            digest = write(*args, **kwargs)
            cancelled.set()
            return digest
        monkeypatch.setattr(checkpoints, '_write_payload', interrupted)
        with pytest.raises(ValueError, match='cancelled'):
            target.save_checkpoint(path, cancelled.is_set)
    else:
        read = checkpoints._read_payload
        def interrupted(*args, **kwargs):
            read(*args, **kwargs)
            cancelled.set()
        monkeypatch.setattr(checkpoints, '_read_payload', interrupted)
        with pytest.raises(ValueError, match='cancelled'):
            target.load_checkpoint(path, cancelled.is_set)
    assert path.read_bytes() == original
    np.testing.assert_array_equal(target.working, pixels)
    assert target.background_fill is fill and target.can_undo and not target.can_redo


def test_exact_archive_load_and_save_never_recompute_background(
        tmp_path, monkeypatch, controlled_background):
    doc = _document()
    assert doc.auto_background('soft') == 'committed'
    assert doc.auto_background('sharp') == 'committed'
    assert doc.undo()
    def forbidden(*args, **kwargs):
        pytest.fail('an exact checkpoint recomputed background pixels')
    monkeypatch.setattr('core.background_fill.replace_background', forbidden)
    path = tmp_path / 'no-recomputation.ofretouch'
    doc.save_checkpoint(path)
    fresh = _document()
    fresh.load_checkpoint(path)
    _assert_state(doc, fresh)
    while doc.can_redo:
        assert doc.redo() and fresh.redo()
        _assert_state(doc, fresh)


def test_archive_rejects_dishonest_directory_count_before_eager_parsing(
        tmp_path, monkeypatch, controlled_background):
    import struct
    import zipfile
    source = _document()
    assert source.auto_background('soft') == 'committed'
    path = tmp_path / 'directory-count.ofretouch'
    source.save_checkpoint(path)
    data = bytearray(path.read_bytes())
    end = data.rfind(b'PK\x05\x06')
    assert end >= 0
    struct.pack_into('<HH', data, end + 8, 1, 1)
    path.write_bytes(data)
    target = _document()
    def forbidden(*args, **kwargs):
        pytest.fail('ZipFile eagerly parsed an unvalidated member count')
    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    with pytest.raises(ValueError, match='member count'):
        target.load_checkpoint(path)
    assert not target.is_dirty and not target.can_undo and target.background_fill is None


def test_archive_admits_zip64_without_classic_sentinels(
        tmp_path, controlled_background):
    import struct
    source = _document()
    assert source.auto_background('soft') == 'committed'
    path = tmp_path / 'zip64-without-sentinels.ofretouch'
    source.save_checkpoint(path)
    data = path.read_bytes()
    end = data.rfind(b'PK\x05\x06')
    fields = struct.unpack('<4s4H2LH', data[end:end + 22])
    count, size, offset = fields[4:7]
    record = struct.pack('<4sQ2H2L4Q', b'PK\x06\x06', 44, 45, 45,
                         0, 0, count, count, size, offset)
    locator = struct.pack('<4sLQL', b'PK\x06\x07', 0, end, 1)
    path.write_bytes(data[:end] + record + locator + data[end:])
    target = _document()
    target.load_checkpoint(path)
    _assert_state(source, target)


def test_archive_rejects_hidden_zip64_directory_before_eager_parsing(
        tmp_path, monkeypatch, controlled_background):
    import struct
    import zipfile
    import utils.retouch_checkpoint as checkpoints
    source = _document()
    assert source.auto_background('soft') == 'committed'
    path = tmp_path / 'hidden-zip64.ofretouch'
    source.save_checkpoint(path)
    data = path.read_bytes()
    end = data.rfind(b'PK\x05\x06')
    fields = struct.unpack('<4s4H2LH', data[end:end + 22])
    count, size, offset = fields[4:7]
    # The classic directory is one entry whose comment hides a ZIP64 record.
    # ZipFile follows that record even though no classic field is a sentinel.
    classic_entry = bytearray(46)
    classic_entry[:4] = b'PK\x01\x02'
    struct.pack_into('<H', classic_entry, 32, 76)
    record = struct.pack('<4sQ2H2L4Q', b'PK\x06\x06', 44, 45, 45,
                         0, 0, count + 1, count + 1, size + 46, offset)
    locator = struct.pack('<4sLQL', b'PK\x06\x07', 0, end + 46, 1)
    classic_end = struct.pack('<4s4H2LH', b'PK\x05\x06', 0, 0,
                              1, 1, 122, end, 0)
    path.write_bytes(data[:end] + classic_entry + record + locator + classic_end)
    monkeypatch.setattr(checkpoints, 'MAX_ARCHIVE_MEMBERS', count)
    target = _document()
    def forbidden(*args, **kwargs):
        pytest.fail('ZipFile parsed a hidden ZIP64 directory before admission')
    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    with pytest.raises(ValueError, match='directory exceeds its limits'):
        target.load_checkpoint(path)
    assert not target.is_dirty and not target.can_undo and target.background_fill is None
