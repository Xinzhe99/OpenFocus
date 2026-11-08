"""Versioned, source-verified retouch operation records."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
import struct
import tempfile
import zipfile

import numpy as np

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_POINTS = 250000
MAX_OPERATIONS = 100000
MAX_ARCHIVE_MEMBERS = 500000
PAYLOAD_CHUNK_BYTES = 1024 * 1024


def _aligned_mask_bytes(doc):
    # Connected components, zero masks, labels and up to two cached masks.
    pixels = max((frame.image.shape[0] * frame.image.shape[1]
                  for frame in doc.frames
                  if frame.image is not None and frame.aligned), default=0)
    return pixels * 14


def _require_replay_memory(doc, operations, max_bytes, diagnostic_bytes=0):
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
    require_memory(2 * frame_bytes + history + scratch + _aligned_mask_bytes(doc)
                   + diagnostic_bytes)


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
    from utils.path_identity import same_path
    sources = [frame.name for frame in doc.frames]
    sources.extend(doc.context.source_paths)
    if any(same_path(destination, source) for source in sources):
        raise ValueError('Cannot overwrite a source image')
    # Reserve for JSON text and its UTF-8 encoding before materializing both.
    from core.resource_policy import require_memory
    record_bytes = (4096 + 96 * doc._point_count + 256 * len(doc._operations)
                    + 384 * len(doc._background_analyses)
                    + sum(512 + 6 * len(str(frame.frame_id)) for frame in doc.frames))
    snapshot = (bool(doc._background_fills) or bool(doc._background_analyses)
                or any(operation.get('type') == 'auto_background' for operation in doc._operations))
    if snapshot:
        payload_count = (1 + sum(len(entry) for entry in doc.history._undo + doc.history._redo)
                         + 4 * len(doc._background_fills))
        if payload_count + 1 > MAX_ARCHIVE_MEMBERS:
            raise ValueError('Retouch history has too many payloads')
        record_bytes += payload_count * 512
    require_memory(record_bytes * 2)
    record = {"version": 1, **doc._provenance, "operations": doc._operations,
              "cursor": doc._cursor, "undo_count": len(doc.history._undo),
              "redo_count": len(doc.history._redo), "max_strokes": doc.history.max_strokes,
              "max_bytes": doc.history.max_bytes,
              "budget_scope": "history_and_diagnostics",
              "background_analyses": [{"cursor": state, "operation": operation}
                                      for state, operation in sorted(doc._background_analyses.items())]}
    data = None
    if not snapshot:
        data = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        if len(data) > MAX_FILE_BYTES:
            raise ValueError('Retouch history is too large to save')
    if cancelled is not None and cancelled():
        raise ValueError('Retouch checkpoint saving cancelled')
    directory = os.path.dirname(destination)
    fd, temp = tempfile.mkstemp(prefix='.retouch-', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            if snapshot:
                _write_snapshot(doc, stream, record, cancelled)
            else:
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
        if doc._background_fills.keys() != verified._background_fills.keys():
            raise ValueError('Retouch replay diagnostics do not match; start a new session')
        for state, expected_fill in doc._background_fills.items():
            actual_fill = verified._background_fills[state]
            if _fill_metadata(expected_fill) != _fill_metadata(actual_fill) or any(
                    not np.array_equal(left[1], right[1]) for left, right in zip(
                        expected_fill.missing.layers(), actual_fill.missing.layers())):
                raise ValueError('Retouch replay diagnostics do not match; start a new session')
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
        archive = stream.read(4) == b'PK\x03\x04'
    if archive:
        with _open_archive(path) as archive:
            return _archive_metadata(archive)
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
    with open(path, 'rb') as stream:
        archive = stream.read(4) == b'PK\x03\x04'
    if archive:
        return _load_snapshot(doc, path, cancelled)
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
        analyses = record.get('background_analyses', [])
        if not isinstance(analyses, list) or len(analyses) > max_strokes + 2:
            raise ValueError('Invalid background analyses')
        pending_analyses = {}
        for analysis in analyses:
            if not isinstance(analysis, dict):
                raise ValueError('Invalid background analysis')
            state = _integer(analysis['cursor'], 0, len(operations))
            operation = analysis['operation']
            if state in pending_analyses or not isinstance(operation, dict):
                raise ValueError('Invalid background analysis')
            if operation.get('type') != 'auto_background':
                raise ValueError('Invalid background analysis')
            pending_analyses[state] = operation
        scope = record.get('budget_scope')
        if scope not in (None, 'history_and_diagnostics'):
            raise ValueError('Unsupported retouch history budget scope')
        legacy_states = (_legacy_background_states(operations, pending_analyses,
                                                   cursor - undo_count)
                         if scope is None else None)
        diagnostic_bytes = (len(legacy_states) * doc.height * doc.width * 4
                            if legacy_states is not None else 0)
        if diagnostic_bytes > 4 * 1024 * 1024 * 1024:
            raise ValueError('Checkpoint diagnostics exceed the history budget')
        _require_replay_memory(doc, operations, max_bytes, diagnostic_bytes)
        from utils.retouch import RetouchDocument
        replay = RetouchDocument(doc.base, doc.context, max_strokes=max_strokes, max_bytes=max_bytes)
        replay._provenance = doc._provenance
        replay.stacked_image = doc.stacked_image
        replay._checkpoint_background_states = legacy_states
        points_count = 0
        for state, op in enumerate(operations):
            if state in pending_analyses:
                _replay_background(replay, pending_analyses[state], cancelled, 'empty')
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
                if replay.end_stroke(_preserve_empty=True) != 'committed':
                    raise ValueError('Stroke could not be replayed')
            elif kind == 'tones':
                exposure = _number(op['exposure'], -2, 2)
                contrast = _number(op['contrast'], -1, 1)
                adjustments = {name: _number(op.get(name, 0), -1, 1)
                               for name in ('highlights', 'shadows', 'saturation', 'temperature')}
                if replay.adjust_tones(exposure, contrast, cancelled, **adjustments) != 'committed':
                    raise ValueError('Tone adjustment could not be replayed')
            elif kind == 'auto_background':
                _replay_background(replay, op, cancelled, 'committed')
            elif kind == 'restore_all':
                if not replay.restore_all():
                    raise ValueError('Restore could not be replayed')
            else:
                raise ValueError('Unknown retouch operation')
        if len(operations) in pending_analyses:
            _replay_background(replay, pending_analyses[len(operations)], cancelled, 'empty')
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
        if legacy_states is not None:
            # Old v1 records charged tiles alone. Keep their exact retained
            # history, then give the combined budget explicit diagnostic
            # headroom, within the same hard limit accepted for new records.
            map_bytes = sum(replay._background_map_bytes(fill)
                            for fill in replay._background_fills.values())
            replay.history.max_bytes = min(4 * 1024 * 1024 * 1024, max_bytes + map_bytes)
            if replay.history._total_bytes() + map_bytes > replay.history.max_bytes:
                raise ValueError('Checkpoint diagnostics exceed the history budget')
            replay._checkpoint_background_states = None
        replay._enforce_background_budget()
        if cancelled is not None and cancelled():
            raise ValueError('Retouch replay cancelled')
    except (KeyError, TypeError, OverflowError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid retouch checkpoint') from exc
    return replay


def _legacy_background_states(operations, analyses, first):
    """Only diagnostics reachable in the legacy record need migration storage."""
    states = {index + 1 for index, operation in enumerate(operations)
              if isinstance(operation, dict) and operation.get('type') == 'auto_background'}
    states.update(analyses)
    if not states:
        return None
    earlier = [state for state in states if state <= first]
    baseline = max(earlier) if earlier else None
    if baseline is not None and any(isinstance(operation, dict)
                                    and operation.get('type') == 'restore_all'
                                    for operation in operations[baseline:first]):
        baseline = None
    return {state for state in states if state > first or state == baseline}


def _replay_background(replay, operation, cancelled, expected_status):
    _validate_background_operation(operation)
    stacked = replay.stacked_image
    # Older v1 records analysed the then-current pixels. New operations name
    # their immutable analysis source explicitly; snapshots never recompute it.
    if not operation.get('from_stacked'):
        replay.stacked_image = replay.base if operation.get('from_base') else replay.working
    try:
        status = replay.auto_background(
            operation['preset'], operation['clean'], cancelled,
            keep_soft_parts=operation.get('keep_soft_parts', False))
    finally:
        replay.stacked_image = stacked
    if status == 'cancelled':
        raise ValueError('Retouch replay cancelled')
    if status != expected_status:
        raise ValueError('Auto background could not be replayed')
    if status == 'committed':
        replay._operations[-1] = dict(operation)
    else:
        replay._background_analyses[replay._cursor] = dict(operation)


def load_checkpoint(doc, path, cancelled=None):
    replay = _replay_checkpoint(doc, path, cancelled)
    doc.working = replay.working
    doc.history = replay.history
    doc.history.base = doc.base
    doc._operations = replay._operations
    doc._cursor = replay._cursor
    # The replayed Auto background steps carry their missing-data maps; maps
    # from this session's earlier runs no longer describe the loaded image.
    doc._background_fills = replay._background_fills
    doc._background_analyses = replay._background_analyses
    doc._point_count = replay._point_count


def _cancel_checkpoint(cancelled, saving=False):
    if cancelled is not None and cancelled():
        raise ValueError('Retouch checkpoint saving cancelled' if saving
                         else 'Retouch replay cancelled')


def _fill_metadata(fill):
    return {name: getattr(fill, name) for name in (
        'changed', 'background_frame', 'far_end', 'subject_fraction',
        'preset_used', 'recorded_share', 'stats')}


class _DigestStream:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        self.digest.update(data)
        return self.stream.write(data)

    def read(self, count):
        data = self.stream.read(count)
        self.digest.update(data)
        return data


def _write_payload(archive, name, array, cancelled):
    with archive.open(name, 'w', force_zip64=True) as stream:
        output = _DigestStream(stream)
        np.lib.format.write_array_header_1_0(output, {
            'descr': np.lib.format.dtype_to_descr(array.dtype),
            'fortran_order': False, 'shape': array.shape})
        # Buffered iteration also bounds copies of non-contiguous map views.
        iterator = np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                             op_flags=['readonly'], order='C',
                             buffersize=max(1, PAYLOAD_CHUNK_BYTES // array.dtype.itemsize))
        for block in iterator:
            _cancel_checkpoint(cancelled, saving=True)
            output.write(memoryview(np.ascontiguousarray(block)).cast('B'))
        return output.digest.hexdigest()


def _write_snapshot(doc, stream, record, cancelled):
    record['version'] = 2
    snapshot = {'undo': [], 'redo': [], 'background_fills': []}
    record['snapshot'] = snapshot
    payloads = {}
    record['payloads'] = payloads
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        payloads['working.npy'] = _write_payload(archive, 'working.npy', doc.working, cancelled)
        for direction, entries in (('undo', doc.history._undo), ('redo', doc.history._redo)):
            for index, entry in enumerate(entries):
                coordinates = []
                snapshot[direction].append(coordinates)
                for (tx, ty), tile in sorted(entry.items()):
                    coordinates.append([tx, ty])
                    name = f'{direction}/{index}/{tx}_{ty}.npy'
                    payloads[name] = _write_payload(archive, name, tile, cancelled)
        for state, fill in sorted(doc._background_fills.items()):
            snapshot['background_fills'].append({'cursor': state, 'result': _fill_metadata(fill)})
            for layer_name, layer, _ in fill.missing.layers():
                name = f'maps/{state}/{layer_name}.npy'
                payloads[name] = _write_payload(archive, name, layer, cancelled)
        _cancel_checkpoint(cancelled, saving=True)
        data = json.dumps(record, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':')).encode('utf-8')
        if len(data) > MAX_FILE_BYTES:
            raise ValueError('Retouch history is too large to save')
        archive.writestr('metadata.json', data)


def _admit_archive_directory(stream, cancelled=None):
    """Bound ZIP's eager central-directory allocation before constructing it."""
    from core.resource_policy import require_memory
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    length = min(size, 65557)
    stream.seek(size - length)
    end = stream.read(length)
    position = end.rfind(b'PK\x05\x06')
    if position < 0 or len(end) - position < 22:
        raise ValueError('Invalid retouch checkpoint archive')
    fields = struct.unpack('<4s4H2LH', end[position:position + 22])
    _, disk, directory_disk, disk_count, count, directory_size, offset, comment = fields
    if disk or directory_disk or disk_count != count or position + 22 + comment != len(end):
        raise ValueError('Unsupported retouch checkpoint archive directory')
    directory_end = size - length + position
    locator = b''
    # ZipFile checks the locator even when classic fields fit without sentinels.
    if directory_end >= 20:
        stream.seek(directory_end - 20)
        locator = stream.read(20)
    zip64 = locator.startswith(b'PK\x06\x07')
    if (directory_size == 0xffffffff or offset == 0xffffffff) and not zip64:
        raise ValueError('Missing ZIP64 checkpoint directory')
    if zip64:
        if len(locator) != 20:
            raise ValueError('Invalid ZIP64 checkpoint directory')
        signature, disk, zip64_offset, disks = struct.unpack('<4sLQL', locator)
        if signature != b'PK\x06\x07' or disk or disks != 1:
            raise ValueError('Unsupported ZIP64 checkpoint directory')
        stream.seek(zip64_offset)
        header = stream.read(56)
        if len(header) != 56:
            raise ValueError('Invalid ZIP64 checkpoint directory')
        fields = struct.unpack('<4sQ2H2L4Q', header)
        if (fields[0] != b'PK\x06\x06' or fields[1] != 44 or fields[4] or fields[5]
                or fields[6] != fields[7] or zip64_offset + 56 != directory_end - 20):
            raise ValueError('Unsupported ZIP64 checkpoint directory')
        count, directory_size, offset = fields[7], fields[8], fields[9]
        directory_end = zip64_offset
    if (not 1 <= count <= MAX_ARCHIVE_MEMBERS or directory_size < count * 46
            or offset < 0 or offset + directory_size != directory_end):
        raise ValueError('Retouch checkpoint archive directory exceeds its limits')
    require_memory(directory_size * 4 + count * 1024 + 131072)
    # ZipFile walks directory bytes rather than trusting the EOCD count.
    # Confirm that count with fixed-size reads before its eager allocation.
    position = offset
    for _ in range(count):
        _cancel_checkpoint(cancelled)
        stream.seek(position)
        header = stream.read(46)
        if len(header) != 46 or header[:4] != b'PK\x01\x02':
            raise ValueError('Invalid checkpoint member directory')
        name_size, extra_size, comment_size = struct.unpack_from('<3H', header, 28)
        position += 46 + name_size + extra_size + comment_size
        if position > directory_end:
            raise ValueError('Checkpoint member directory exceeds its bounds')
    if position != directory_end:
        raise ValueError('Checkpoint archive member count does not match')
    stream.seek(0)


@contextmanager
def _open_archive(path, cancelled=None):
    try:
        with open(path, 'rb') as stream:
            _admit_archive_directory(stream, cancelled)
            with zipfile.ZipFile(stream, 'r') as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(set(names)) != len(names) or any(
                        info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1
                        or info.file_size != info.compress_size for info in infos):
                    raise ValueError('Duplicate or unsupported checkpoint archive members')
                yield archive
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError,
            struct.error, EOFError, OverflowError) as exc:
        raise ValueError('Invalid retouch checkpoint archive') from exc


def _archive_metadata(archive):
    from core.resource_policy import require_memory
    try:
        info = archive.getinfo('metadata.json')
    except KeyError as exc:
        raise ValueError('Retouch checkpoint metadata is missing') from exc
    if not 1 <= info.file_size <= MAX_FILE_BYTES:
        raise ValueError('Retouch checkpoint metadata exceeds its limit')
    require_memory(info.file_size * 12)
    try:
        with archive.open(info) as stream:
            data = stream.read(info.file_size + 1)
        if len(data) != info.file_size:
            raise ValueError('Retouch checkpoint metadata changed while reading')
        record = json.loads(data)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Invalid retouch checkpoint metadata') from exc
    if not isinstance(record, dict) or type(record.get('version')) is not int or record['version'] != 2:
        raise ValueError('Unsupported retouch history version')
    return record


def _snapshot_operations(doc, operations):
    if not isinstance(operations, list) or len(operations) > MAX_OPERATIONS:
        raise ValueError('Invalid retouch operation list')
    points_count = 0
    limit = max(doc.width, doc.height) * 8 + 2048
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError('Invalid retouch operation')
        kind = operation['type']
        if kind == 'stroke':
            _integer(operation['frame'], 0, len(doc.frames) - 1)
            if operation['tool'] not in doc.TOOLS:
                raise ValueError('Invalid retouch tool')
            _integer(operation['diameter'], 1, 2048)
            _number(operation['hardness'], 0, 1)
            _number(operation['opacity'], 0.01, 1)
            _number(operation['exposure'], -2, 2)
            points = operation['points']
            if not isinstance(points, list) or not points:
                raise ValueError('Invalid stroke points')
            points_count += len(points)
            if points_count > MAX_POINTS:
                raise ValueError('Too many stroke points')
            for point in points:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError('Invalid stroke point')
                _number(point[0], -limit, limit)
                _number(point[1], -limit, limit)
        elif kind == 'tones':
            _number(operation['exposure'], -2, 2)
            _number(operation['contrast'], -1, 1)
            for name in ('highlights', 'shadows', 'saturation', 'temperature'):
                _number(operation.get(name, 0), -1, 1)
        elif kind == 'auto_background':
            _validate_background_operation(operation)
        elif kind != 'restore_all':
            raise ValueError('Unknown retouch operation')
    return points_count


def _validate_background_operation(operation):
    if (not isinstance(operation, dict) or operation.get('type') != 'auto_background'
            or operation.get('preset') not in ('auto', 'soft', 'sharp')
            or type(operation.get('clean')) is not bool
            or type(operation.get('keep_soft_parts', False)) is not bool
            or type(operation.get('from_base', False)) is not bool
            or type(operation.get('from_stacked', False)) is not bool
            or (operation.get('from_base') and operation.get('from_stacked'))):
        raise ValueError('Invalid auto background operation')


def _snapshot_fill_metadata(doc, metadata):
    if not isinstance(metadata, dict) or type(metadata.get('changed')) is not bool:
        raise ValueError('Invalid background diagnostic metadata')
    _integer(metadata['background_frame'], 0, len(doc.frames) - 1)
    _integer(metadata['far_end'], 0, len(doc.frames) - 1)
    _number(metadata['subject_fraction'], 0, 1)
    _number(metadata['recorded_share'], 0, 1)
    if metadata['preset_used'] not in ('soft', 'sharp', 'forced') or not isinstance(metadata['stats'], dict):
        raise ValueError('Invalid background diagnostic metadata')
    pending = [metadata['stats']]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Invalid background diagnostic statistics')
    return {key: metadata[key] for key in ('changed', 'background_frame', 'far_end',
            'subject_fraction', 'preset_used', 'recorded_share', 'stats')}


def _payload_header(stream, expected_shape, expected_dtype):
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError('Unsupported checkpoint NumPy payload version')
    length_size = 2 if version == (1, 0) else 4
    encoded_length = stream.read(length_size)
    if len(encoded_length) != length_size:
        raise ValueError('Truncated checkpoint NumPy header')
    header_size = int.from_bytes(encoded_length, 'little')
    if not 1 <= header_size <= 4096:
        raise ValueError('Checkpoint NumPy header exceeds its limit')
    header = stream.read(header_size)
    if len(header) != header_size:
        raise ValueError('Truncated checkpoint NumPy header')
    # NumPy checks max_header_size after reading its declared header. Give
    # its parser only this already bounded header, never the payload stream.
    bounded = io.BytesIO(encoded_length + header)
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, fortran, dtype = reader(bounded, max_header_size=4096)
    if (shape != expected_shape or any(type(value) is not int for value in shape)
            or fortran or dtype.hasobject or dtype != expected_dtype):
        raise ValueError('Checkpoint payload shape or dtype does not match')


def _read_payload(archive, name, target, digest, cancelled):
    with archive.open(name) as stream:
        source = _DigestStream(stream)
        _payload_header(source, target.shape, target.dtype)
        view = memoryview(target).cast('B')
        position = 0
        while position < len(view):
            _cancel_checkpoint(cancelled)
            block = source.read(min(PAYLOAD_CHUNK_BYTES, len(view) - position))
            if not block:
                raise ValueError('Truncated checkpoint payload')
            view[position:position + len(block)] = block
            position += len(block)
        if source.read(1) or source.digest.hexdigest() != digest:
            raise ValueError('Checkpoint payload checksum does not match')


def _load_snapshot(doc, path, cancelled):
    from core.background_fill import BackgroundFillResult, MissingData
    from core.resource_policy import require_memory
    from utils.retouch import RetouchDocument, TILE
    try:
        with _open_archive(path, cancelled) as archive:
            record = _archive_metadata(archive)
            if record.get('budget_scope') != 'history_and_diagnostics':
                raise ValueError('Unsupported retouch history budget scope')
            expected = {'base': record['base'], 'frames': record['frames']}
            if expected != doc._provenance:
                raise ValueError('Base image, processed sources or source geometry do not match the checkpoint')
            _check_provenance(doc, expected, cancelled)
            operations = record['operations']
            points_count = _snapshot_operations(doc, operations)
            cursor = _integer(record['cursor'], 0, len(operations))
            undo_count = _integer(record['undo_count'], 0, cursor)
            redo_count = _integer(record['redo_count'], 0, len(operations) - cursor)
            if redo_count != len(operations) - cursor:
                raise ValueError('Invalid redo history')
            max_strokes = _integer(record['max_strokes'], 1, MAX_OPERATIONS)
            max_bytes = _integer(record['max_bytes'], 1, 4 * 1024 * 1024 * 1024)
            if undo_count + redo_count > max_strokes:
                raise ValueError('Checkpoint history exceeds its stroke limit')
            analyses = record.get('background_analyses', [])
            if not isinstance(analyses, list) or len(analyses) > max_strokes + 2:
                raise ValueError('Invalid background analyses')
            background_analyses = {}
            for analysis in analyses:
                if not isinstance(analysis, dict):
                    raise ValueError('Invalid background analysis')
                state = _integer(analysis['cursor'], 0, len(operations))
                if state in background_analyses:
                    raise ValueError('Duplicate background analysis')
                _validate_background_operation(analysis['operation'])
                background_analyses[state] = analysis['operation']
            snapshot, payloads = record['snapshot'], record['payloads']
            if not isinstance(snapshot, dict) or not isinstance(payloads, dict):
                raise ValueError('Invalid retouch snapshot manifest')
            # These plans contain only geometry and references to small JSON
            # records. No NumPy payload array exists before all headers pass.
            plan = [('working.npy', doc.base.shape, doc.base.dtype)]
            history_plan = {}
            retained_bytes = 0
            for direction, count in (('undo', undo_count), ('redo', redo_count)):
                entries = snapshot[direction]
                if not isinstance(entries, list) or len(entries) != count:
                    raise ValueError('Invalid snapshot history count')
                history_plan[direction] = []
                for index, coordinates in enumerate(entries):
                    if not isinstance(coordinates, list) or not coordinates:
                        raise ValueError('Invalid snapshot history tiles')
                    keys = set()
                    entry_plan = []
                    history_plan[direction].append(entry_plan)
                    for coordinate in coordinates:
                        if not isinstance(coordinate, list) or len(coordinate) != 2:
                            raise ValueError('Invalid snapshot tile coordinate')
                        tx = _integer(coordinate[0], 0, (doc.width - 1) // TILE)
                        ty = _integer(coordinate[1], 0, (doc.height - 1) // TILE)
                        if (tx, ty) in keys:
                            raise ValueError('Duplicate snapshot history tile')
                        keys.add((tx, ty))
                        shape = (min(TILE, doc.height - ty * TILE),
                                 min(TILE, doc.width - tx * TILE), *doc.base.shape[2:])
                        name = f'{direction}/{index}/{tx}_{ty}.npy'
                        entry_plan.append(((tx, ty), name, shape))
                        plan.append((name, shape, doc.base.dtype))
                        retained_bytes += math.prod(shape) * doc.base.dtype.itemsize
            fills = snapshot['background_fills']
            if not isinstance(fills, list) or len(fills) > max_strokes + 2:
                raise ValueError('Invalid snapshot diagnostic states')
            fill_plan = {}
            layer_names = ('background', 'hidden', 'out_of_range', 'featureless')
            for fill in fills:
                if not isinstance(fill, dict):
                    raise ValueError('Invalid snapshot diagnostics')
                state = _integer(fill['cursor'], 0, len(operations))
                if state in fill_plan or (state not in background_analyses and (
                        state == 0 or operations[state - 1].get('type') != 'auto_background')):
                    raise ValueError('Invalid snapshot diagnostic state')
                fill_plan[state] = _snapshot_fill_metadata(doc, fill['result'])
                for layer in layer_names:
                    plan.append((f'maps/{state}/{layer}.npy', (doc.height, doc.width), np.dtype('uint8')))
                    retained_bytes += doc.height * doc.width
            if not background_analyses.keys() <= fill_plan.keys():
                raise ValueError('Snapshot background analysis has no diagnostics')
            if retained_bytes > max_bytes:
                raise ValueError('Checkpoint history and diagnostics exceed the byte budget')
            names = {name for name, _shape, _dtype in plan}
            if set(payloads) != names or set(archive.namelist()) != names | {'metadata.json'}:
                raise ValueError('Unexpected or missing checkpoint archive payload')
            require_memory(2 * doc.base.nbytes + retained_bytes + PAYLOAD_CHUNK_BYTES
                           + len(plan) * 256)
            for name, shape, dtype in plan:
                _cancel_checkpoint(cancelled)
                digest = payloads[name]
                if not isinstance(digest, str) or len(digest) != 64 or any(
                        character not in '0123456789abcdef' for character in digest):
                    raise ValueError('Invalid checkpoint payload checksum')
                info = archive.getinfo(name)
                with archive.open(info) as stream:
                    _payload_header(stream, shape, dtype)
                    if info.file_size != stream.tell() + math.prod(shape) * dtype.itemsize:
                        raise ValueError('Checkpoint payload size does not match')
            replay = RetouchDocument(doc.base, doc.context, max_strokes=max_strokes, max_bytes=max_bytes)
            replay._provenance = doc._provenance
            replay._operations, replay._cursor, replay._point_count = operations, cursor, points_count
            replay._background_analyses = background_analyses
            _read_payload(archive, 'working.npy', replay.working, payloads['working.npy'], cancelled)
            for direction, entries in history_plan.items():
                destination = replay.history._undo if direction == 'undo' else replay.history._redo
                for entry_plan in entries:
                    entry = {}
                    destination.append(entry)
                    for key, name, shape in entry_plan:
                        tile = np.empty(shape, doc.base.dtype)
                        _read_payload(archive, name, tile, payloads[name], cancelled)
                        entry[key] = tile
            for state, metadata in fill_plan.items():
                layers = []
                for layer in layer_names:
                    name = f'maps/{state}/{layer}.npy'
                    data = np.empty((doc.height, doc.width), np.uint8)
                    _read_payload(archive, name, data, payloads[name], cancelled)
                    layers.append(data)
                replay._background_fills[state] = BackgroundFillResult(
                    image=None, missing=MissingData(*layers), **metadata)
            for key in replay.history._tile_keys(0, 0, doc.width, doc.height):
                _cancel_checkpoint(cancelled)
                replay.history._refresh_dirty(key)
            replay._prune_background_fills()
            if replay._background_fills.keys() != fill_plan.keys():
                raise ValueError('Snapshot contains unreachable diagnostics')
            replay.history._reserved_bytes = replay._background_map_bytes(replay.background_fill)
            replay.history.evicted = cursor > undo_count
            _check_provenance(doc, expected, cancelled)
            _cancel_checkpoint(cancelled)
            return replay
    except (KeyError, TypeError, OverflowError, UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid retouch snapshot checkpoint') from exc
