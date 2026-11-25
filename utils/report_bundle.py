"""Privacy-preserving diagnostics for optional error reports.

Only explicitly permitted facts and numeric event fields enter the archive. Raw
logs and exception messages are never included, even after pattern redaction.
"""
from dataclasses import dataclass
import builtins
import datetime
import hashlib
import io
import json
import math
import os
import platform as system_platform
import re
import traceback

import zipfile
from utils import diagnostics, logging_setup
from utils.account_client import safe_sign_in_evidence
from utils.output_sharpen import MAX_SHARPEN_STRENGTH

MAX_BUNDLE_BYTES = 2 * 1024 * 1024
_MAX_LOG_BYTES = 2 * 1024 * 1024
_LIVE_LOG_BYTES = 256 * 1024
_MAX_EVENTS = 2000
_MAX_FRAMES = 12
_EVENTS = frozenset('''session.started startup shutdown.started shutdown.completed ui.action ui.gesture
batch_finished render_start render_finished render.fusion_region load_stack export_image export_image_write
converter.page converter.item converter.batch converter.output converter.resources
converter.admission_failed converter.admitted cli cli.fusion cli.exit render.performance
comparison_alignment comparison_preview comparison_method scratch_selected
linear_source_manifest source_manifest fusion fusion.effective fusion_result fusion_tiling fusion_tile
registration_warp registration_plan registration_stage render_resources execution_plan
neural_tile_plan heavy_job.waited source_decode source_previews stack_load stack_load.decoded roi_alignment
render batch_import batch_image_write batch batch_folder batch_stack gif_save
settings.applied depth_map neural_provider_probe neural_provider_selected neural_session
ort_operator_trace render_child_error resource_admission diagnostic_failure
render.state render_dispatch render_dispatch.worker_started render_dispatch.admitted
render_dispatch.ipc_ready render_dispatch.request_encoded render_dispatch.spawn_started
render_dispatch.spawn_completed'''.split())
_NUMBERS = frozenset('''elapsed_ms seconds frame_count frames loaded_frames file_count
folder_count count workers preview_threads thread_count native_threads page pages files
success total index column stack_index scale scale_factor reg_downscale_width duration_sec
block_requested block_after_image_clamp block_after_memory_plan block block_size
overlap_requested overlap tiles required_bytes goal_bytes available_bytes budget
need_ram free_ram need_disk free_disk in_flight ram_budget vram_budget rss_bytes
peak_rss_bytes available_ram_bytes total_ram_bytes fixed_bytes per_worker_bytes
inference_bytes ram_ceiling waited_ms exit_code total_bytes radius sigma threshold levels
iterations kernel_size strength quality compression downscale width height smoothing feather N
max_shift_ratio rotation magnification reference_frame exposure_reference_index
tile_block_size tile_overlap tile_threshold ai_batch_size registration_downscale_width
group_number group_size group_overlap load_scale_factor roi_base_index button
angle_dx angle_dy pixel_dx pixel_dy'''.split())
_SHAPES = frozenset(('shape', 'first_shape'))
_DTYPES = frozenset(('uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'float16', 'float32', 'float64', 'bool'))
_NESTED = frozenset(('settings', 'parameters', 'fusion_parameters', 'registration_parameters'))
_METHODS = frozenset('''stackmffv3 stackmffv2 stackmff openfocus_a openfocus_b
openfocus_c guided_filter gfgfgf dct dtcwt none homography ecc openfocus both'''.split())
_METHOD_FIELDS = frozenset(('method', 'algorithm', 'registration', 'align', 'alignment'))
_BOOL_FIELDS = frozenset(('preview', 'use_gpu', 'gpu_requested', 'cancelled', 'tile_enabled',
    'roi_only', 'reverse_order', 'fallback', 'fallback_cpu', 'exposure_match',
    'sharpen_output', 'reverse_registration', 'source_order_reversed', 'sensor_denoise',
    'defect_mask', 'flat_field', 'grouped', 'need_fusion', 'full_frame_roi',
    'alignment_reused', 'alignment_computed'))
_CONTROLS = frozenset('''fusion_method alignment_method render compare_all quick_preview
exposure_match base_correct reverse_order sharpen_output wipe source_frame result_frame roi'''.split())
_CONTROL_ID = re.compile(r'^[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*){0,20}$')
_ENUMS = {
    'origin': frozenset(('render', 'batch', 'batch_group', 'cli', 'comparison')),
    'base_frame': frozenset(('none', 'first', 'last')),
    'sort_mode': frozenset(('asc', 'desc', 'off')),
    'registration_brightness_reference': frozenset(('first', 'last')),
    'resource_preference': frozenset(('balanced', 'responsive', 'maximum')),
    'mode': frozenset(('soft', 'hard')),
    'inference_scale': frozenset(('single', 'dual')),
    'failure_policy': frozenset(('continue', 'fail')),
    'roi_mode': frozenset(('crop', 'paste')),
    'action': frozenset(('press', 'move', 'release', 'scroll')),
    'error_category': frozenset(('resource_limit', 'memory', 'os', 'value', 'runtime', 'other')),
    'error_origin': frozenset(('child_worker', 'child_setup_or_transfer', 'child_exit')),
    'render_stage': frozenset(('prepare', 'align', 'fuse', 'finish', 'finalize', 'publish', 'unknown')),
    'resource_status': frozenset(('unknown', 'insufficient')),
}
_ADMISSION_BYTES = ('required_bytes', 'available_ram_bytes', 'total_ram_bytes', 'ram_ceiling',
                    'available_commit_bytes', 'process_commit_headroom_bytes')
_GPU_MODEL = re.compile(r'^(?:NVIDIA (?:GeForce (?:RTX|GTX) [1-9][0-9]{2,3}(?: Ti| SUPER| Laptop GPU)?|RTX [1-9][0-9]{3}(?: Ada Generation)?)|AMD Radeon (?:RX [1-9][0-9]{3}(?: XT| XTX)?|Graphics)|Intel (?:Arc A[0-9]{3}M?|Iris Xe Graphics|UHD Graphics [0-9]{3,4}))$')
_STACK_FUNCTIONS = {
    'workers.py': frozenset(('run', '_run_admitted')),
    'registration.py': frozenset(('_align_homography_impl', 'align', 'register')),
    'multi_focus_fusion.py': frozenset(('fuse', '_fuse_tiled_ai_batched', '_fuse_stackmffv3', '_fuse_openfocus_a', '_fuse_openfocus_b', '_fuse_openfocus_c')),
    'image_loader.py': frozenset(('load_from_filepaths', 'load_from_folder', 'load_from_video', 'load_linear_raw_stack')),
    'render_process.py': frozenset(('run', '_run_process', '_render_child', '_run_decoded_render_job')),
    'resource_policy.py': frozenset(('require_memory', 'plan_execution')),
    'source_frames.py': frozenset(('resolve', 'decode', 'full_resolution', '_decode', 'resolve_frame')),
    'batch_converter.py': frozenset(('convert_one', 'convert_batch', '_decode', '_decode_tiff')),
    'stackmffv3.py': frozenset(('fuse', '_blend_tile')),
    'depth_map.py': frozenset(('compute_focus_index',)),
    'cli.py': frozenset(('_run_with_logging', 'main')),
}
_EVENT_LINE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\s+\w+\s+.*?\bevent=([^\s]+)\s+(\{.*\})\s*$')
_ERROR_HEADER = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\s+(?:ERROR|CRITICAL)\s+pid=\d+\s+thread=\S+\s+session=\S+\s+operation=\S+\s+\S+.*$')
_OPERATION_ID = re.compile(r'^[0-9a-f]{12}$')
_EVENT_OPERATION = re.compile(r'\bsession=\S+ operation=([0-9a-f]{12}) ')
_FRAME = re.compile(r'^\s*File "([^"\n]+)", line (\d+), in ([^\s]+)\s*$')
_NATIVE_FRAME = re.compile(r'^\s*File "([^"\n]+)", line (\d+) in ([^\s]+)\s*$')


def _native_error(text):
    """Extract only a recognized native header and the current thread's safe frames."""
    kind = None
    current = False
    frames = []
    for line in text.splitlines():
        if line.startswith(('Fatal Python error:', 'Windows fatal exception:')):
            title = line.partition(':')[2].strip().lower()
            kind = {'segmentation fault': 'segmentation_fault', 'aborted': 'aborted',
                    'bus error': 'bus_error', 'access violation': 'access_violation'}.get(title, 'unknown')
            current = False
            frames = []
            continue
        if kind is None:
            continue
        if line.startswith('Current thread '):
            current = True
            continue
        if line.startswith('Thread ') or line.startswith('Extension modules:'):
            current = False
            continue
        if current and len(frames) < _MAX_FRAMES:
            match = _NATIVE_FRAME.match(line)
            if match:
                filename, number, function = match.groups()
                basename = re.split(r'[/\\]', filename)[-1]
                if len(number) <= 9 and 0 < int(number) < 10**9 and function in _STACK_FUNCTIONS.get(basename, ()):
                    frames.append((basename, int(number), function))
    return kind, list(reversed(frames))
_EXCEPTION = re.compile(r'^((?:core\.resource_policy\.)?[A-Za-z_][A-Za-z_0-9]*)(?::.*)?$')
_QUALIFIED_EXCEPTION = re.compile(r'^(?:[A-Za-z_][\w]*\.)+([A-Za-z_][A-Za-z_0-9]*):.*$', re.ASCII)
_VERSION = re.compile(r'^\d{1,6}(?:\.\d{1,6}){0,3}(?:(?:a|b|rc|\.post|\.dev)\d{1,6})?$')


@dataclass(frozen=True)
class ReportBundle:
    data: bytes
    fingerprint: str
    has_error: bool
    subject: str
    summary: str
    log_excerpt: str
    platform: str
    occurred_at: int
    app_version: str | None = None
    build_channel: str | None = None


def _platform():
    value = system_platform.system().lower()
    return {'windows': 'windows', 'darwin': 'macos', 'linux': 'linux'}.get(value, 'unknown')


def _exception_name(name):
    if name in ('ResourceLimitError', 'core.resource_policy.ResourceLimitError'):
        return 'ResourceLimitError'
    value = getattr(builtins, name, None)
    return name if isinstance(value, type) and issubclass(value, BaseException) else None


def _numeric_fields(fields, depth=0):
    result = {}
    if not isinstance(fields, dict):
        return result
    for key, value in fields.items():
        if key in _NUMBERS and type(value) in (int, float) and abs(value) <= 10**15 and math.isfinite(value):
            result[key] = value
        elif key == 'sharpen_strength' and type(value) is int and 0 <= value <= MAX_SHARPEN_STRENGTH:
            result[key] = value
        elif key in _SHAPES and isinstance(value, (list, tuple)) and 1 <= len(value) <= 4 and all(type(x) is int and 0 <= x <= 10**9 for x in value):
            result[key] = list(value)
        elif key in ('roi_rect', 'requested_region', 'input_region', 'output_region') and isinstance(value, (list, tuple)) and len(value) == 4 and all(type(x) is int and 0 <= x <= 10**9 for x in value):
            result[key] = list(value)
        elif key == 'fusion_order' and isinstance(value, list) and len(value) <= 4096 and all(type(x) is int and 0 <= x <= 10**9 for x in value):
            result[key] = list(value)
        elif key == 'points' and isinstance(value, list) and len(value) <= 128 and all(
                isinstance(point, list) and len(point) == 2 and all(type(x) is int and abs(x) <= 10**7 for x in point)
                for point in value):
            result[key] = value
        elif key == 'sources' and isinstance(value, list) and len(value) <= 4096:
            result[key] = [_numeric_fields(item, depth + 1) for item in value if isinstance(item, dict)]
        elif key == 'dtype' and isinstance(value, str) and value in _DTYPES:
            result[key] = value
        elif key == 'frame_shapes' and isinstance(value, list):
            shapes = []
            if len(value) > 32:
                result['frame_shapes_overflow'] = len(value) - 32
            for item in value[:32]:
                if not isinstance(item, dict):
                    continue
                shape, count, dtype = item.get('shape'), item.get('count'), item.get('dtype')
                if (isinstance(shape, (list, tuple)) and 1 <= len(shape) <= 4
                        and all(type(x) is int and 0 <= x <= 10**9 for x in shape)
                        and type(count) is int and 0 <= count <= 10**9
                        and isinstance(dtype, str) and dtype in _DTYPES):
                    shapes.append({'shape': list(shape), 'count': count, 'dtype': dtype})
            if shapes:
                result[key] = shapes
        elif key in _BOOL_FIELDS and type(value) is bool:
            result[key] = value
        elif key in _METHOD_FIELDS and isinstance(value, str) and value in _METHODS:
            result[key] = value
        elif key in ('registration', 'align', 'alignment') and isinstance(value, list) and len(value) <= 8 and all(isinstance(item, str) and item in _METHODS for item in value):
            result[key] = value
        elif key == 'adapter' and isinstance(value, str) and _GPU_MODEL.fullmatch(value):
            result['gpu_model'] = value
        elif key == 'control' and isinstance(value, str) and len(value) <= 200 and (value in _CONTROLS or _CONTROL_ID.fullmatch(value)):
            result[key] = value
        elif key in _ENUMS and isinstance(value, str) and value in _ENUMS[key]:
            result[key] = value
        elif key == 'value' and type(value) is bool:
            result[key] = value
        elif key == 'value' and type(value) in (int, float) and abs(value) <= 10**15 and math.isfinite(value):
            result[key] = value
        elif key in _NESTED and depth < 2:
            nested = _numeric_fields(value, depth + 1)
            if nested:
                result[key] = nested
    return result


def safe_exception(kind, frames):
    """Keep bounded line numbers and allowlisted application locations."""
    frames = list(frames)[-_MAX_FRAMES:]
    error = {'kind': _exception_name(kind) or 'Error',
             'stack_lines': [line for _, line, _ in frames if type(line) is int and 0 < line < 10**9]}
    stack = []
    for filename, line, function in frames:
        if not isinstance(filename, str) or not isinstance(function, str) or type(line) is not int:
            continue
        basename = re.split(r'[/\\]', filename)[-1]
        if 0 < line < 10**9 and function in _STACK_FUNCTIONS.get(basename, ()):
            stack.append({'file': basename, 'line': line, 'function': function})
    if stack:
        error['stack'] = stack
    return error


def _safe_admission(fields):
    if not isinstance(fields, dict):
        return {}
    values = {key: fields[key] for key in _ADMISSION_BYTES
              if type(fields.get(key)) is int and 0 <= fields[key] <= 10**15}
    if fields.get('resource_status') in ('unknown', 'insufficient'):
        values['resource_status'] = fields['resource_status']
    return values


def _safe_child_failure(fields):
    if not isinstance(fields, dict):
        return {}
    evidence = {key: value for key in ('error_category', 'error_origin', 'render_stage')
                if isinstance(value := fields.get(key), str) and value in _ENUMS[key]}
    if type(code := fields.get('exit_code')) is int and -(2**31) <= code < 2**32:
        evidence['exit_code'] = code
    raw_error = fields.get('error')
    if isinstance(raw_error, dict) and isinstance(raw_error.get('kind'), str):
        stack = raw_error.get('stack')
        frames = ((frame.get('file'), frame.get('line'), frame.get('function'))
                  for frame in stack[-_MAX_FRAMES:] if isinstance(frame, dict)) if isinstance(stack, list) else ()
        error = safe_exception(raw_error['kind'], frames)
        if isinstance(lines := raw_error.get('stack_lines'), list):
            error['stack_lines'] = [x for x in lines[-_MAX_FRAMES:] if type(x) is int and 0 < x < 10**9]
        evidence['error'] = error
    if admission := _safe_admission(fields.get('admission')):
        evidence['admission'] = admission
    return evidence


def _known_event(name):
    base, separator, outcome = name.rpartition('.')
    return name in _EVENTS or (separator and base in _EVENTS and outcome in ('started', 'completed', 'cancelled', 'failed'))


def _timestamp(stamp):
    return datetime.datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S,%f').isoformat(timespec='milliseconds')


def _events_and_error(text):
    events = []
    omitted = 0
    has_error = False
    frames = []
    error_name = None
    error_message = ''
    in_traceback = False
    error_timestamp = None
    traceback_timestamp = None
    previous_line = ''
    for line in text.splitlines():
        preceding_line, previous_line = previous_line, line
        match = _EVENT_LINE.match(line)
        if match:
            timestamp, name, payload = match.groups()
            # Sign-in evidence is handled only by its dedicated anonymous queue.
            if name == 'account_sign_in.failed':
                continue
            if name.endswith('.failed'):
                has_error = True
            if not _known_event(name):
                continue
            try:
                parsed = json.loads(payload)
                stamp = _timestamp(timestamp)
            except (ValueError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue
            if name == 'render_child_error':
                parameters = _safe_child_failure(parsed)
            elif name == 'resource_admission':
                parameters = _safe_admission(parsed)
            else:
                parameters = _numeric_fields(parsed)
            event = {'timestamp': stamp, 'event': name, 'parameters': parameters}
            if operation := _EVENT_OPERATION.search(line, 0, match.start(2)):
                event['operation_id'] = operation.group(1)
            if isinstance(parent := parsed.get('parent_operation_id'), str) and _OPERATION_ID.fullmatch(parent):
                event['parent_operation_id'] = parent
            events.append(event)
            if len(events) > _MAX_EVENTS:
                del events[0]
                omitted += 1
            continue
        if line.strip() == 'Traceback (most recent call last):':
            in_traceback = True
            frames = []
            traceback_timestamp = None
            header = _ERROR_HEADER.match(preceding_line)
            if header:
                try:
                    traceback_timestamp = _timestamp(header.group(1))
                except ValueError:
                    pass
            continue
        if in_traceback:
            frame = _FRAME.match(line)
            if frame:
                frames.append((frame.group(1), int(frame.group(2)), frame.group(3)))
                frames = frames[-_MAX_FRAMES:]
                continue
            match = _EXCEPTION.fullmatch(line) or _QUALIFIED_EXCEPTION.fullmatch(line)
            if match:
                error_name = match.group(1)
                error_message = line.partition(':')[2].strip()
                has_error = True
                error_timestamp = traceback_timestamp
                in_traceback = False
    return events, has_error, error_name, error_message, frames, omitted, error_timestamp


def _normalize_message(message):
    # Used only as fingerprint input, never emitted as report text.
    message = message.lower()
    message = re.sub(r'\b[0-9a-f]{8}-[0-9a-f-]{27,}\b', ' ', message)
    message = re.sub(r'(?:[a-z]:)?[\\/][^\s\'\"]+', ' ', message)
    message = re.sub(r'\b[0-9a-f]{8,}\b', ' ', message)
    message = re.sub(r'\b(?:0x)?[0-9a-f]*\d[0-9a-f]*\b', ' ', message)
    message = re.sub(r'\d+', ' ', message)
    return ' '.join(message.split())[:200]


def _fingerprint(kind, platform, frames, message):
    # Both slash conventions may occur regardless of the reporting host.
    top = frames[-3:][::-1]
    frame_text = '|'.join('{}:{}:{}'.format(re.split(r'[/\\]', filename)[-1], line, func) for filename, line, func in top)
    return hashlib.sha256(f'{kind}|{platform}|{frame_text}|{_normalize_message(message)}'.encode('utf-8', 'replace')).hexdigest()


def _safe_facts(facts, platform):
    result = {'platform': platform}
    for key in ('OpenFocus', 'python', 'numpy', 'opencv-python', 'opencv-python-headless', 'scipy', 'torch', 'PyQt6', 'dtcwt', 'Pillow'):
        value = facts.get(key)
        if isinstance(value, str) and len(value) <= 32 and _VERSION.fullmatch(value):
            result[key] = value
    for key in ('cpu_count',):
        value = facts.get(key)
        if type(value) is int and 0 <= value <= 65536:
            result[key] = value
    if type(facts.get('frozen')) is bool:
        result['frozen'] = facts['frozen']
    machine = facts.get('machine')
    if machine in ('AMD64', 'x86_64', 'arm64', 'aarch64', 'x86', 'i386', 'i686'):
        result['machine'] = machine
    ram = facts.get('ram')
    if isinstance(ram, str):
        match = re.fullmatch(r'(\d{1,6}\.\d) GB total, (\d{1,6}\.\d) GB free', ram)
        if match:
            result['ram_total_gib'] = float(match.group(1))
            result['ram_free_gib'] = float(match.group(2))
    gpu = facts.get('gpu (neural acceleration)')
    if type(gpu) is bool:
        result['gpu_accelerated'] = gpu
    for key in ('gpu', 'gpu_model', 'adapter'):
        gpu = facts.get(key)
        if isinstance(gpu, str) and _GPU_MODEL.fullmatch(gpu):
            result['gpu_model'] = gpu
            break
    return result


def _read_crash():
    path = diagnostics.crash_log_path()
    if path is None:
        return ''
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - _MAX_LOG_BYTES))
            return stream.read(_MAX_LOG_BYTES).decode('utf-8', 'replace')
    except OSError:
        return ''


def session_has_error(log_text=None, crash_text=None):
    live_error = log_text is None and logging_setup.session_had_error()
    if log_text is None:
        log_text = logging_setup.session_log_excerpt(_MAX_LOG_BYTES)
    if crash_text is None:
        crash_text = _read_crash()
    return bool(live_error or crash_text.strip() or _events_and_error(log_text)[1])


def _json_text(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def _excerpt(events):
    entries = []
    remaining = 7998
    for event in reversed(events[-25:]):
        encoded = _json_text(event)
        cost = len(encoded) + bool(entries)
        if cost <= remaining:
            entries.append(encoded)
            remaining -= cost
    return '[' + ','.join(reversed(entries)) + ']'


def _pack_report(events, error, capture, facts, platform, signature, has_error, occurred_at,
                 app_version=None, build_channel=None):
    kind = error['kind']
    display_kind = f"{kind} ({error['native_kind']})" if 'native_kind' in error else kind
    summary = f"{display_kind}; {len(error['stack_lines'])} stack frames"
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in (('system.json', facts), ('events.json', events),
                            ('error.json', error), ('capture.json', capture)):
            archive.writestr(name, _json_text(value))
    data = output.getvalue()
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError('report bundle exceeds 2 MiB')
    return ReportBundle(data, signature, has_error, f'{kind}: diagnostic report', summary,
                        _excerpt(events), platform, occurred_at, app_version, build_channel)


def snapshot_sign_in_failure(record):
    """Capture only sign-in evidence, including before the reporting choice is set."""
    evidence = safe_sign_in_evidence(record.report_sign_in)
    if not all(key in evidence for key in ('method', 'stage', 'category')):
        return None
    timestamp = datetime.datetime.fromtimestamp(record.created).isoformat(timespec='milliseconds')
    event = {'timestamp': timestamp, 'event': 'account_sign_in.failed', 'parameters': evidence}
    error = {'kind': 'AccountError', 'stack_lines': [], 'timestamp': timestamp}
    platform = _platform()
    signature = _fingerprint('AccountError', platform, (), _json_text({
        key: evidence[key] for key in ('method', 'stage', 'category', 'status') if key in evidence
    }))
    return _pack_report([event], error, {}, {'platform': platform}, platform,
                        signature, True, int(record.created))


def snapshot_failure(record, facts):
    """Capture this error with its bounded lead-up, while it is still in the log."""
    stamp = datetime.datetime.fromtimestamp(record.created).isoformat(timespec='milliseconds')
    frames = []
    kind, message = 'Error', ''
    if record.exc_info and record.exc_info[0] is not None:
        exc_type, exc, tb = record.exc_info
        kind, message = exc_type.__name__, str(exc) if exc is not None else ''
        frames = [(f.f_code.co_filename, line, f.f_code.co_name) for f, line in traceback.walk_tb(tb)]
    if not frames:
        frames = [(record.pathname, record.lineno, record.funcName)]
    platform = _platform()
    child = _safe_child_failure(getattr(record, 'report_child_evidence', None))
    error = child.get('error', safe_exception(kind, frames))
    parameters = dict(child)
    if admission := _safe_admission(getattr(record.exc_info[1], 'admission', None) if record.exc_info else None):
        parameters.setdefault('admission', admission)
    if child:
        child_frames = [(f['file'], f['line'], f['function']) for f in error.get('stack', ())]
        # The child signature uses its own code, including the exit code when no stack survived.
        category = ':'.join(str(child[k]) for k in ('error_origin', 'error_category', 'exit_code') if k in child)
        fingerprint_kind = f"{error['kind']}:{category}"
        signature = _fingerprint(fingerprint_kind, platform, child_frames or frames, '' if child_frames else message)
    else:
        signature = _fingerprint(kind, platform, frames, message)
    operation = getattr(record, 'event', None)
    if isinstance(operation, str) and operation.endswith('.failed') and _known_event(operation):
        parameters['failed_event'] = operation
    error['timestamp'] = stamp
    workload = _numeric_fields(getattr(record, 'report_workload', None))
    parameters.update(workload)
    parameters['error'] = error
    log_text = logging_setup.session_log_excerpt(_LIVE_LOG_BYTES)
    events, _, _, _, _, omitted, _ = _events_and_error(log_text)
    events = [event for event in events if event['timestamp'] <= stamp]
    failure = {'timestamp': stamp, 'event': 'diagnostic_failure', 'parameters': parameters}
    operation_id = getattr(record, 'operation_id', None)
    if isinstance(operation_id, str) and _OPERATION_ID.fullmatch(operation_id):
        failure['operation_id'] = operation_id
    if len(events) >= _MAX_EVENTS:
        del events[0]
        omitted += 1
    events.append(failure)
    capture = {'events_omitted': omitted, 'log_tail_trimmed': log_text.startswith('...(older lines trimmed;')}
    return _pack_report(events, error, capture, _safe_facts(facts, platform), platform,
                        signature, True, int(record.created))


def build_report_bundle(*, log_text=None, crash_text=None, facts=None, exc=None,
                        app_version=None, build_channel=None):
    """Build a bounded zip and safe metadata; raises ValueError if too large."""
    live_error = log_text is None and logging_setup.session_had_error()
    if log_text is None:
        log_text = logging_setup.session_log_excerpt(_MAX_LOG_BYTES)
    if crash_text is None:
        crash_text = _read_crash()
    if app_version is not None:
        # Only the version and channel in metadata describe the previous run.
        facts = {}
        if build_channel in ('release', 'development'):
            facts['frozen'] = build_channel == 'release'
    elif facts is None:
        facts = diagnostics.system_facts()
    platform = _platform()
    events, event_error, kind, message, frames, omitted, error_timestamp = _events_and_error(log_text)
    _, crash_error, crash_kind, crash_message, crash_frames, _, crash_timestamp = _events_and_error(crash_text)
    native_kind, native_frames = _native_error(crash_text)
    if exc is not None:
        kind = type(exc).__name__
        message = str(exc)
        frames = [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)]
        native_kind = None
        error_timestamp = None
    elif native_kind is not None:
        kind, message, frames = None, '', native_frames
        error_timestamp = None
    elif crash_kind:
        kind, message, frames = crash_kind, crash_message, crash_frames
        error_timestamp = crash_timestamp
    fingerprint_kind = f'native:{native_kind}' if native_kind is not None else kind or 'crash'
    fingerprint = _fingerprint(fingerprint_kind, platform, frames, message)
    error = safe_exception(kind or '', frames)
    if native_kind is not None:
        error.update(kind='Crash', native_kind=native_kind)
    elif kind is None and not (live_error and not crash_text.strip()):
        error['kind'] = 'Crash'
    if error_timestamp is not None:
        error['timestamp'] = error_timestamp
    capture = {'events_omitted': omitted,
               'log_tail_trimmed': log_text.startswith('...(older lines trimmed;')}
    occurred_at = (datetime.datetime.fromisoformat(error_timestamp).timestamp()
                   if error_timestamp else datetime.datetime.now().timestamp())
    return _pack_report(events, error, capture, _safe_facts(facts, platform), platform, fingerprint,
                        bool(live_error or event_error or crash_error or crash_text.strip() or exc is not None),
                        int(occurred_at), app_version, build_channel)
