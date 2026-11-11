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

MAX_BUNDLE_BYTES = 2 * 1024 * 1024
_MAX_LOG_BYTES = 2 * 1024 * 1024
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
ort_operator_trace'''.split())
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
    'failure_policy': frozenset(('continue', 'fail')),
    'roi_mode': frozenset(('crop', 'paste')),
    'action': frozenset(('press', 'move', 'release', 'scroll')),
}
_GPU_MODEL = re.compile(r'^(?:NVIDIA (?:GeForce (?:RTX|GTX) [1-9][0-9]{2,3}(?: Ti| SUPER| Laptop GPU)?|RTX [1-9][0-9]{3}(?: Ada Generation)?)|AMD Radeon (?:RX [1-9][0-9]{3}(?: XT| XTX)?|Graphics)|Intel (?:Arc A[0-9]{3}M?|Iris Xe Graphics|UHD Graphics [0-9]{3,4}))$')
_STACK_FUNCTIONS = {
    'workers.py': frozenset(('run', '_run_admitted')),
    'registration.py': frozenset(('_align_homography_impl', 'align', 'register')),
    'multi_focus_fusion.py': frozenset(('fuse', '_fuse_tiled_ai_batched', '_fuse_stackmffv3', '_fuse_openfocus_a', '_fuse_openfocus_b', '_fuse_openfocus_c')),
    'image_loader.py': frozenset(('load_from_filepaths', 'load_from_folder', 'load_from_video')),
    'source_frames.py': frozenset(('resolve', 'decode')),
    'depth_map.py': frozenset(('compute_focus_index',)),
    'cli.py': frozenset(('_run_with_logging', 'main')),
}
_EVENT_LINE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\s+\w+\s+.*?\bevent=([^\s]+)\s+(\{.*\})\s*$')
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
_EXCEPTION = re.compile(r'^([A-Za-z_][A-Za-z_0-9]*)(?::.*)?$')
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


def _platform():
    value = system_platform.system().lower()
    return {'windows': 'windows', 'darwin': 'macos', 'linux': 'linux'}.get(value, 'unknown')


def _exception_name(name):
    value = getattr(builtins, name, None)
    return name if isinstance(value, type) and issubclass(value, BaseException) else None


def _numeric_fields(fields, depth=0):
    result = {}
    if not isinstance(fields, dict):
        return result
    for key, value in fields.items():
        if key in _NUMBERS and type(value) in (int, float) and abs(value) <= 10**15 and math.isfinite(value):
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


def _events_and_error(text):
    events = []
    omitted = 0
    has_error = False
    frames = []
    error_name = None
    error_message = ''
    in_traceback = False
    for line in text.splitlines():
        match = _EVENT_LINE.match(line)
        if match:
            timestamp, name, payload = match.groups()
            if name.endswith('.failed'):
                has_error = True
            base, separator, outcome = name.rpartition('.')
            known = name in _EVENTS or (separator and base in _EVENTS and outcome in ('started', 'completed', 'cancelled', 'failed'))
            if not known:
                continue
            try:
                parsed = json.loads(payload)
                stamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f')
            except (ValueError, TypeError):
                continue
            if outcome == 'failed':
                has_error = True
            events.append({'timestamp': stamp.isoformat(timespec='milliseconds'), 'event': name, 'parameters': _numeric_fields(parsed)})
            if len(events) > _MAX_EVENTS:
                del events[0]
                omitted += 1
            continue
        if line.strip() == 'Traceback (most recent call last):':
            in_traceback = True
            frames = []
            continue
        if in_traceback:
            frame = _FRAME.match(line)
            if frame:
                frames.append((frame.group(1), int(frame.group(2)), frame.group(3)))
                frames = frames[-_MAX_FRAMES:]
                continue
            match = _EXCEPTION.match(line)
            if match:
                error_name = match.group(1)
                error_message = line.partition(':')[2].strip()
                has_error = True
                in_traceback = False
    return events, has_error, error_name, error_message, frames, omitted


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


def build_report_bundle(*, log_text=None, crash_text=None, facts=None, exc=None):
    """Build a bounded zip and safe metadata; raises ValueError if too large."""
    live_error = log_text is None and logging_setup.session_had_error()
    if log_text is None:
        log_text = logging_setup.session_log_excerpt(_MAX_LOG_BYTES)
    if crash_text is None:
        crash_text = _read_crash()
    if facts is None:
        facts = diagnostics.system_facts()
    platform = _platform()
    events, event_error, kind, message, frames, omitted = _events_and_error(log_text)
    _, crash_error, crash_kind, crash_message, crash_frames, _ = _events_and_error(crash_text)
    native_kind, native_frames = _native_error(crash_text)
    if exc is not None:
        kind = type(exc).__name__
        message = str(exc)
        frames = [(f.filename, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)]
        native_kind = None
    elif native_kind is not None:
        kind, message, frames = None, '', native_frames
    elif crash_kind:
        kind, message, frames = crash_kind, crash_message, crash_frames
    fingerprint_kind = f'native:{native_kind}' if native_kind is not None else kind or 'crash'
    fingerprint = _fingerprint(fingerprint_kind, platform, frames, message)
    safe_kind = ('Crash' if native_kind is not None else
                 _exception_name(kind or '') or ('Error' if kind or (live_error and not crash_text.strip()) else 'Crash'))
    safe_frames = [line for _, line, _ in frames[-_MAX_FRAMES:] if 0 < line < 10**9]
    subject = f'{safe_kind}: diagnostic report'
    summary = f'{safe_kind}; {len(safe_frames)} stack frames'
    if native_kind is not None:
        summary = f'{safe_kind} ({native_kind}); {len(safe_frames)} stack frames'
    error = {'kind': safe_kind, 'stack_lines': safe_frames}
    if native_kind is not None:
        error['native_kind'] = native_kind
    stack = []
    for filename, line, func in frames[-_MAX_FRAMES:]:
        basename = re.split(r'[/\\]', filename)[-1]
        if 0 < line < 10**9 and func in _STACK_FUNCTIONS.get(basename, ()):
            stack.append({'file': basename, 'line': line, 'function': func})
    if stack:
        error['stack'] = stack
    capture = {'events_omitted': omitted,
               'log_tail_trimmed': log_text.startswith('...(older lines trimmed;')}
    excerpt = json.dumps(events[-25:], ensure_ascii=True, separators=(',', ':'))[-8000:]
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in (('system.json', _safe_facts(facts, platform)), ('events.json', events),
                            ('error.json', error), ('capture.json', capture)):
            archive.writestr(name, json.dumps(value, ensure_ascii=True, separators=(',', ':'), allow_nan=False))
    data = output.getvalue()
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError('report bundle exceeds 2 MiB')
    return ReportBundle(data, fingerprint, bool(live_error or event_error or crash_error or crash_text.strip() or exc is not None), subject, summary, excerpt, platform, int(datetime.datetime.now().timestamp()))
