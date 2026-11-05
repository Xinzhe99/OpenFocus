"""Versioned, source-verified retouch operation records."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile

import numpy as np

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_POINTS = 250000
MAX_OPERATIONS = 100000


def _aligned_mask_bytes(doc):
    # Connected components, zero masks, labels and up to two cached masks.
    pixels = max((frame.image.shape[0] * frame.image.shape[1]
                  for frame in doc.frames
                  if frame.image is not None and frame.aligned), default=0)
    return pixels * 14


def _require_replay_memory(doc, operations, max_bytes):
    from core.resource_policy import require_memory

    frame_bytes = doc.base.nbytes
    count = len(operations)
    pixels = doc.height * doc.width
    history = min(max_bytes, count * frame_bytes)
    stroke_diameters = [op.get('diameter', 0) for op in operations
                        if isinstance(op, dict) and op.get('type') == 'stroke']
    diameter = max((value for value in stroke_diameters
                    if type(value) is int and 1 <= value <= 2048), default=0)
    dab_pixels = min(doc.width, diameter) * min(doc.height, diameter)
    # One replay document holds two image copies. A pending stroke can hold
    # another full image of history plus coverage and float blending buffers.
    scratch = (frame_bytes + pixels * 4 + dab_pixels * 48) if count else 0
    require_memory(2 * frame_bytes + history + scratch + _aligned_mask_bytes(doc))


def _fingerprint(array, cancelled=None):
    if array is None:
        return None
    digest = hashlib.sha256()
    # Hash in row order without copying an entire non-contiguous ROI.
    for row in array:
        if cancelled is not None and cancelled():
            raise ValueError('Retouch replay cancelled')
        digest.update(memoryview(np.ascontiguousarray(row)).cast('B'))
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": digest.hexdigest()}


def provenance(doc, cancelled=None):
    mask_bytes = _aligned_mask_bytes(doc)
    if mask_bytes:
        from core.resource_policy import require_memory
        require_memory(mask_bytes)
    frames = []
    for frame in doc.frames:
        if cancelled is not None and cancelled():
            raise ValueError('Retouch replay cancelled')
        image = frame.image
        mask = frame.valid_mask() if image is not None and frame.aligned else None
        frames.append({"id": str(frame.frame_id), "image": _fingerprint(image, cancelled),
                       "offset_x": int(frame.offset_x), "offset_y": int(frame.offset_y),
                       "aligned": bool(frame.aligned), "mask": _fingerprint(mask, cancelled)})
    return {"base": _fingerprint(doc.base, cancelled), "frames": frames}


def _check_provenance(doc, expected, cancelled=None):
    actual = provenance(doc, cancelled)
    if any(frame['image'] is None for frame in actual['frames']):
        raise ValueError('A processed source frame is missing')
    for frame in actual['frames']:
        shape = frame['image']['shape']
        x, y = frame['offset_x'], frame['offset_y']
        if len(shape) < 2 or x < 0 or y < 0 or x + shape[1] > doc.width or y + shape[0] > doc.height:
            raise ValueError('Invalid processed source geometry')
    if actual != expected:
        raise ValueError('Base image, processed sources or source geometry do not match the checkpoint')


def save_checkpoint(doc, path, cancelled=None):
    if doc.stroke_active:
        raise ValueError('Finish the current stroke before saving')
    if not doc.checkpoint_ready:
        raise ValueError('Checkpoint initialization is unavailable; start a new session')
    _check_provenance(doc, doc._provenance, cancelled)
    destination = os.path.abspath(os.fspath(path))
    for frame in doc.frames:
        name = frame.name
        if name and os.path.isfile(name) and (os.path.normcase(os.path.abspath(name)) == os.path.normcase(destination)
                or (os.path.exists(destination) and os.path.samefile(name, destination))):
            raise ValueError('Cannot overwrite a source image')
    # Reserve for JSON text and its UTF-8 encoding before materializing both.
    from core.resource_policy import require_memory
    record_bytes = (4096 + 96 * doc._point_count + 256 * len(doc._operations)
                    + sum(512 + 6 * len(str(frame.frame_id)) for frame in doc.frames))
    require_memory(record_bytes * 2)
    record = {"version": 1, **doc._provenance, "operations": doc._operations,
              "cursor": doc._cursor, "undo_count": len(doc.history._undo),
              "redo_count": len(doc.history._redo), "max_strokes": doc.history.max_strokes,
              "max_bytes": doc.history.max_bytes}
    data = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('Retouch history is too large to save')
    if cancelled is not None and cancelled():
        raise ValueError('Retouch checkpoint saving cancelled')
    directory = os.path.dirname(destination)
    fd, temp = tempfile.mkstemp(prefix='.retouch-', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if cancelled is not None and cancelled():
            raise ValueError('Retouch checkpoint saving cancelled')
        verified = _replay_checkpoint(doc, temp, cancelled)
        if not np.array_equal(verified.working, doc.working):
            raise ValueError('Retouch replay does not match the current image; start a new session')
        for expected, actual in ((doc.history._undo, verified.history._undo),
                                 (doc.history._redo, verified.history._redo)):
            if len(expected) != len(actual):
                raise ValueError('Retouch replay history does not match; start a new session')
            for left, right in zip(expected, actual):
                if cancelled is not None and cancelled():
                    raise ValueError('Retouch checkpoint saving cancelled')
                if left.keys() != right.keys() or any(
                        not np.array_equal(left[key], right[key]) for key in left):
                    raise ValueError('Retouch replay history does not match; start a new session')
        _check_provenance(doc, doc._provenance, cancelled)
        if cancelled is not None and cancelled():
            raise ValueError('Retouch checkpoint saving cancelled')
        os.replace(temp, destination)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _number(value, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError('Invalid retouch operation parameter')
    return value


def _integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('Invalid retouch history index')
    return value


def read_checkpoint(path):
    """Read checkpoint metadata without exceeding the file or memory budget."""
    with open(path, 'rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > MAX_FILE_BYTES:
            raise ValueError('Retouch history is too large')
        from core.resource_policy import require_memory
        # JSON objects and point lists occupy more memory than their encoding.
        require_memory(size * 12)
        data = stream.read(size + 1)
    if len(data) > size:
        raise ValueError('Retouch history changed while reading')
    try:
        return json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid retouch checkpoint') from exc


def _replay_checkpoint(doc, path, cancelled=None):
    if doc.stroke_active:
        raise ValueError('Finish the current stroke before loading')
    if not doc.checkpoint_ready:
        raise ValueError('Checkpoint initialization is unavailable; start a new session')
    record = read_checkpoint(path)
    try:
        if not isinstance(record, dict) or type(record.get('version')) is not int or record['version'] != 1:
            raise ValueError('Unsupported retouch history version')
        expected = {'base': record['base'], 'frames': record['frames']}
        if expected != doc._provenance:
            raise ValueError('Base image, processed sources or source geometry do not match the checkpoint')
        _check_provenance(doc, expected, cancelled)
        operations = record['operations']
        if not isinstance(operations, list) or len(operations) > MAX_OPERATIONS:
            raise ValueError('Invalid retouch operation list')
        cursor = _integer(record['cursor'], 0, len(operations))
        undo_count = _integer(record['undo_count'], 0, cursor)
        redo_count = _integer(record['redo_count'], 0, len(operations) - cursor)
        if redo_count != len(operations) - cursor:
            raise ValueError('Invalid redo history')
        max_strokes = _integer(record['max_strokes'], 1, 100000)
        max_bytes = _integer(record['max_bytes'], 1, 4 * 1024 * 1024 * 1024)
        _require_replay_memory(doc, operations, max_bytes)
        from utils.retouch import RetouchDocument
        replay = RetouchDocument(doc.base, doc.context, max_strokes=max_strokes, max_bytes=max_bytes)
        replay._provenance = doc._provenance
        points_count = 0
        for op in operations:
            if cancelled is not None and cancelled():
                raise ValueError('Retouch replay cancelled')
            if not isinstance(op, dict):
                raise ValueError('Invalid retouch operation')
            kind = op['type']
            if kind == 'stroke':
                frame = _integer(op['frame'], 0, len(replay.frames) - 1)
                tool = op['tool']
                if tool not in replay.TOOLS:
                    raise ValueError('Invalid retouch tool')
                diameter = _integer(op['diameter'], 1, 2048)
                hardness = _number(op['hardness'], 0, 1)
                opacity = _number(op['opacity'], 0.01, 1)
                exposure = _number(op['exposure'], -2, 2)
                points = op['points']
                if not isinstance(points, list) or not points:
                    raise ValueError('Invalid stroke points')
                points_count += len(points)
                if points_count > MAX_POINTS:
                    raise ValueError('Too many stroke points')
                limit = max(doc.width, doc.height) * 8 + 2048
                for point in points:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        raise ValueError('Invalid stroke point')
                    _number(point[0], -limit, limit)
                    _number(point[1], -limit, limit)
                replay.set_frame(frame)
                replay.set_tool(tool)
                replay.set_brush(diameter, hardness, opacity)
                replay.exposure = exposure
                if not replay.begin_stroke(*points[0]):
                    raise ValueError('Stroke starts outside its source')
                for point in points[1:]:
                    if cancelled is not None and cancelled():
                        raise ValueError('Retouch replay cancelled')
                    replay.stroke_to(*point)
                if replay.end_stroke() != 'committed':
                    raise ValueError('Stroke could not be replayed')
            elif kind == 'tones':
                exposure = _number(op['exposure'], -2, 2)
                contrast = _number(op['contrast'], -1, 1)
                if replay.adjust_tones(exposure, contrast, cancelled) != 'committed':
                    raise ValueError('Tone adjustment could not be replayed')
            elif kind == 'auto_background':
                preset = op['preset']
                clean = op['clean']
                soft_parts = op.get('keep_soft_parts', False)
                if (preset not in ('auto', 'soft', 'sharp') or type(clean) is not bool
                        or type(soft_parts) is not bool):
                    raise ValueError('Invalid auto background operation')
                status = replay.auto_background(preset, clean, cancelled,
                                                keep_soft_parts=soft_parts)
                if status == 'cancelled':
                    raise ValueError('Retouch replay cancelled')
                if status != 'committed':
                    raise ValueError('Auto background could not be replayed')
            elif kind == 'restore_all':
                if not replay.restore_all():
                    raise ValueError('Restore could not be replayed')
            else:
                raise ValueError('Unknown retouch operation')
        for _ in range(len(operations) - cursor):
            if cancelled is not None and cancelled():
                raise ValueError('Retouch replay cancelled')
            if not replay.undo():
                raise ValueError('Redo branch exceeds the history budget')
        if len(replay.history._undo) < undo_count or len(replay.history._redo) != redo_count:
            raise ValueError('Checkpoint history exceeds the history budget')
        if undo_count == 0:
            replay.history._undo.clear()
        else:
            replay.history._undo[:] = replay.history._undo[-undo_count:]
        if cancelled is not None and cancelled():
            raise ValueError('Retouch replay cancelled')
    except (KeyError, TypeError, OverflowError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid retouch checkpoint') from exc
    return replay


def load_checkpoint(doc, path, cancelled=None):
    replay = _replay_checkpoint(doc, path, cancelled)
    doc.working = replay.working
    doc.history = replay.history
    doc._operations = replay._operations
    doc._cursor = replay._cursor
    doc._point_count = replay._point_count
