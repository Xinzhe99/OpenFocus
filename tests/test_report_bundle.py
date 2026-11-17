import io
import json
import zipfile

import pytest

from utils import report_bundle


def _event(name, fields):
    return ('2026-11-10 10:11:12,123 INFO pid=1 thread=Main session=x '
            'operation=x openfocus: event=' + name + ' ' + json.dumps(fields) + '\n')


def _contents(bundle):
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        return {name: json.loads(archive.read(name)) for name in archive.namelist()}


def test_bundle_allowlists_events_facts_and_all_metadata():
    secret = 'privateperson@example.com token-xyz /home/private/photo.png'
    fields = {'path': secret, 'sha256': secret, 'parameters': {'radius': 4, 'path': secret},
              'shape': [100, 200, 3], 'frame_count': 3, 'total_bytes': 60000,
              'frame_shapes': [{'shape': [100, 200, 3], 'count': 3, 'dtype': 'uint8'},
                               {'shape': [4, 5], 'count': 1, 'dtype': secret}],
              'first_dtype': secret, 'message': secret}
    text = _event('render.started', fields) + _event(secret, fields)
    text += ('Traceback (most recent call last):\n'
             '  File "/home/private/secret_photo.py", line 31, in secret_function\n'
             '    secret_source_code()\nValueError: ' + secret + '\n')
    facts = {'OpenFocus': '1.77.0', 'python': '3.12.1', 'cpu_count': 8,
             'machine': 'x86_64', 'platform': secret, 'log': secret, 'numpy': '1.2' + secret}
    bundle = report_bundle.build_report_bundle(log_text=text, crash_text='', facts=facts)
    contents = _contents(bundle)
    public = repr(contents) + bundle.subject + bundle.summary + bundle.log_excerpt
    for private in (secret, 'privateperson', 'token-xyz', 'photo.png',
                    'secret_photo.py', 'secret_function', 'secret_source_code', 'sha256'):
        assert private not in public
    assert contents['system.json']['OpenFocus'] == '1.77.0'
    assert 'numpy' not in contents['system.json']
    assert contents['events.json'] == [{'timestamp': '2026-11-10T10:11:12.123',
        'event': 'render.started', 'parameters': {'parameters': {'radius': 4},
        'shape': [100, 200, 3], 'frame_count': 3, 'total_bytes': 60000,
        'frame_shapes': [{'shape': [100, 200, 3], 'count': 3, 'dtype': 'uint8'}]}}]
    assert contents['error.json'] == {'kind': 'ValueError', 'stack_lines': [31]}
    assert bundle.has_error


def test_error_detection_for_failed_unhandled_and_crash():
    assert not report_bundle.session_has_error(_event('render.completed', {}), '')
    assert report_bundle.session_has_error(_event('new_stage.failed', {}), '')
    assert report_bundle.session_has_error('Traceback (most recent call last):\nValueError: secret\n', '')
    assert report_bundle.session_has_error('', 'native crash text')


def test_fingerprint_normalizes_and_excludes_version():
    def trace(path, message, kind='ValueError'):
        return ('Traceback (most recent call last):\n'
                f'  File "{path}/engine.py", line 12, in render\n'
                f'{kind}: {message}\n')
    first = report_bundle.build_report_bundle(log_text=trace('/home/alice', 'bad 123 /tmp/a'), crash_text='', facts={'OpenFocus': '1.0'})
    second = report_bundle.build_report_bundle(log_text=trace('/home/bob', 'BAD 456 /tmp/b'), crash_text='', facts={'OpenFocus': '9.0'})
    other = report_bundle.build_report_bundle(log_text=trace('/home/bob', 'BAD 456 /tmp/b', 'TypeError'), crash_text='', facts={})
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != other.fingerprint
    assert len(first.fingerprint) == 64


def test_numeric_values_shapes_and_version_fields_are_bounded():
    fields = {'shape': [1, 2, 3, 4, 5], 'first_shape': [1, 'secret'],
              'frame_count': float('nan'), 'frames': True, 'seconds': 10**30,
              'frame_shapes': [{'shape': [1, 2], 'count': True, 'dtype': 'uint8'}]}
    bundle = report_bundle.build_report_bundle(log_text=_event('render.started', fields), crash_text='',
                                               facts={'python': '3.12.1secret', 'cpu_count': True})
    contents = _contents(bundle)
    assert contents['events.json'][0]['parameters'] == {}
    assert 'python' not in contents['system.json']
    assert 'cpu_count' not in contents['system.json']


def test_rejects_oversized_archive(monkeypatch):
    monkeypatch.setattr(report_bundle, 'MAX_BUNDLE_BYTES', 1)
    with pytest.raises(ValueError, match='2 MiB'):
        report_bundle.build_report_bundle(log_text='', crash_text='', facts={})


def test_safe_processing_events_and_shape_overflow():
    shapes = [{'shape': [i + 1, 20, 3], 'count': 1, 'dtype': 'uint8'} for i in range(35)]
    text = _event('stack_load.decoded', {'frame_count': 35, 'frame_shapes': shapes})
    text += _event('render.started', {'method': 'stackmffv3', 'registration': ['homography', 'ecc'],
                                      'preview': True, 'use_gpu': False, 'cancelled': True,
                                      'parameters': {'smoothing': 4.0, 'N': 3}})
    text += _event('ui.action', {'control': 'quick_preview', 'value': True})
    text += _event('ui.action', {'control': 'private@example.com', 'value': 'private/path'})
    events = _contents(report_bundle.build_report_bundle(log_text=text, crash_text='', facts={}))['events.json']
    assert events[0]['parameters']['frame_count'] == 35
    assert len(events[0]['parameters']['frame_shapes']) == 32
    assert events[0]['parameters']['frame_shapes_overflow'] == 3
    assert events[1]['parameters'] == {'method': 'stackmffv3', 'registration': ['homography', 'ecc'],
                                        'preview': True, 'use_gpu': False, 'cancelled': True,
                                        'parameters': {'smoothing': 4.0, 'N': 3}}
    assert events[2]['parameters'] == {'control': 'quick_preview', 'value': True}
    assert events[3]['parameters'] == {}


def test_fusion_report_keeps_reproduction_settings_without_private_text():
    fields = {
        'origin': 'render', 'thread_count': 4, 'exposure_reference_index': 1,
        'fusion_order': [0, 1], 'registration': ['ecc'],
        'settings': {'algorithm': 'stackmffv3', 'exposure_match': True,
                     'sort_mode': 'asc', 'base_frame': 'none', 'tile_enabled': True,
                     'tile_block_size': 1024, 'fusion_parameters': {'mode': 'soft', 'path': '/private/photo.jpg'}},
        'sources': [{'index': 0, 'shape': [10, 20, 3], 'dtype': 'uint8', 'path': '/private/photo.jpg'},
                    {'index': 1, 'shape': [10, 20, 3], 'dtype': 'uint8'}],
        'path': '/private/photo.jpg',
    }
    text = _event('fusion.effective', fields)
    text += _event('ui.action', {'control': 'OpenFocus.ui_objs.action_save', 'value': True})
    text += _event('ui.gesture', {'control': 'OpenFocus.preview', 'action': 'release',
                                  'points': [[1, 2], [3, 4]], 'path': '/private/photo.jpg'})
    events = _contents(report_bundle.build_report_bundle(log_text=text, crash_text='', facts={}))['events.json']
    assert events[0]['parameters']['settings']['exposure_match'] is True
    assert events[0]['parameters']['settings']['fusion_parameters'] == {'mode': 'soft'}
    assert events[0]['parameters']['fusion_order'] == [0, 1]
    assert events[0]['parameters']['sources'][1]['shape'] == [10, 20, 3]
    assert events[1]['parameters']['control'] == 'OpenFocus.ui_objs.action_save'
    assert events[2]['parameters']['points'] == [[1, 2], [3, 4]]
    assert '/private/photo.jpg' not in repr(events)


def test_report_discloses_omitted_history(monkeypatch):
    monkeypatch.setattr(report_bundle, '_MAX_EVENTS', 2)
    text = '...(older lines trimmed; export recent log tails for more context)...\n'
    text += ''.join(_event('ui.action', {'control': 'render', 'index': i}) for i in range(3))
    contents = _contents(report_bundle.build_report_bundle(log_text=text, crash_text='', facts={}))
    assert [event['parameters']['index'] for event in contents['events.json']] == [1, 2]
    assert contents['capture.json'] == {'events_omitted': 1, 'log_tail_trimmed': True}


def test_safe_hardware_and_stack_identifiers():
    trace = ('Traceback (most recent call last):\n'
             '  File "/private/workers.py", line 42, in run\n'
             '  File "/private/workers.py", line 43, in secret_function\n'
             '  File "/private/photo.png", line 44, in run\n'
             'ValueError: private@example.com\n')
    facts = {'ram': '16.0 GB total, 8.5 GB free', 'gpu': 'NVIDIA GeForce RTX 4090',
             'gpu (neural acceleration)': True}
    contents = _contents(report_bundle.build_report_bundle(log_text=trace, crash_text='', facts=facts))
    assert contents['system.json']['ram_total_gib'] == 16.0
    assert contents['system.json']['ram_free_gib'] == 8.5
    assert contents['system.json']['gpu_model'] == 'NVIDIA GeForce RTX 4090'
    assert contents['system.json']['gpu_accelerated'] is True
    assert contents['error.json']['stack'] == [{'file': 'workers.py', 'line': 42, 'function': 'run'}]
    assert 'private' not in repr(contents)
    invalid = {'ram': '16.0 GB total, private free', 'gpu': 'NVIDIA GeForce RTX 4090 private'}
    safe = _contents(report_bundle.build_report_bundle(log_text='', crash_text='', facts=invalid))['system.json']
    assert 'gpu_model' not in safe and 'ram_total_gib' not in safe


def test_live_session_error_survives_missing_log_and_isolated_inputs(monkeypatch):
    monkeypatch.setattr(report_bundle.logging_setup, 'session_had_error', lambda: True)
    monkeypatch.setattr(report_bundle.logging_setup, 'session_log_excerpt', lambda max_bytes: '')
    assert report_bundle.session_has_error(crash_text='')
    bundle = report_bundle.build_report_bundle(crash_text='', facts={})
    assert bundle.has_error
    assert bundle.subject == 'Error: diagnostic report'
    assert _contents(bundle)['error.json']['kind'] == 'Error'
    assert not report_bundle.session_has_error(log_text='', crash_text='')
    assert not report_bundle.build_report_bundle(log_text='', crash_text='', facts={}).has_error
    assert report_bundle.session_has_error(log_text='', crash_text='native crash')


def _native_report(title, frames, directory='/home/alice', address='0x123'):
    crash = (title + '\nThread 0x999:\n  File "/secret/registration.py", line 99 in align\n'
             + f'Current thread {address} (most recent call first):\n'
             + ''.join(f'  File "{directory}/{name}", line {line} in {function}\n'
                       for name, line, function in frames)
             + '  File "/private/token-xyz.py", line 2 in privateperson@example.com\n'
             + 'Extension modules: token-xyz\n')
    old = 'Traceback (most recent call last):\n  File "x", line 2, in x\nValueError: old secret\n'
    return report_bundle.build_report_bundle(log_text=old, crash_text=crash, facts={})


def test_native_crash_safe_frames_and_fingerprint():
    frames = [('workers.py', 42, 'run'), ('registration.py', 99, 'align')]
    first = _native_report('Fatal Python error: Segmentation fault', frames)
    same = _native_report('Fatal Python error: Segmentation fault', frames, 'C:\\Users\\bob', '0x777')
    other = _native_report('Windows fatal exception: access violation', frames)
    assert first.fingerprint == same.fingerprint != other.fingerprint
    error = _contents(first)['error.json']
    assert error == {'kind': 'Crash', 'native_kind': 'segmentation_fault',
                     'stack_lines': [99, 42],
                     'stack': [{'file': 'registration.py', 'line': 99, 'function': 'align'},
                               {'file': 'workers.py', 'line': 42, 'function': 'run'}]}
    assert 'segmentation_fault' in first.summary
    public = repr(_contents(first)) + first.subject + first.summary + first.log_excerpt
    for secret in ('alice', 'privateperson', 'token-xyz', '0x123', '/secret', 'old secret'):
        assert secret not in public


def test_native_top_frames_and_selection_precedence():
    frames = [('workers.py', n, 'run') for n in range(1, 16)]
    first = _native_report('Fatal Python error: Aborted', frames)
    changed_top = _native_report('Fatal Python error: Aborted', [('workers.py', 44, 'run')] + frames[1:])
    changed_tail = _native_report('Fatal Python error: Aborted', frames[:12] + [('workers.py', 80, 'run')])
    assert first.fingerprint != changed_top.fingerprint
    assert first.fingerprint == changed_tail.fingerprint
    assert _contents(first)['error.json']['stack_lines'] == list(range(12, 0, -1))
    crash = 'Fatal Python error: Bus error\nThread 0x1:\n  File "workers.py", line 1 in run\n'
    without_current = report_bundle.build_report_bundle(log_text='', crash_text=crash, facts={})
    assert _contents(without_current)['error.json'] == {'kind': 'Crash', 'native_kind': 'bus_error', 'stack_lines': []}
    unknown = report_bundle.build_report_bundle(log_text='', crash_text='Fatal Python error: secret\n', facts={})
    assert _contents(unknown)['error.json']['native_kind'] == 'unknown'
    explicit = report_bundle.build_report_bundle(log_text='', crash_text=crash, facts={}, exc=ValueError('private'))
    assert _contents(explicit)['error.json']['kind'] == 'ValueError'
    assert 'native_kind' not in _contents(explicit)['error.json']
    truncated = report_bundle.build_report_bundle(log_text='', crash_text='Fatal Python error: Aborted\nCurrent thread 0x', facts={})
    assert _contents(truncated)['error.json']['stack_lines'] == []


def test_child_and_admission_diagnostics_are_strictly_sanitized():
    secret = 'privateperson@example.com /home/private/photo.png token-xyz'
    child = {'error_category': 'resource_limit', 'error_origin': 'child_worker',
             'render_stage': 'prepare', 'message': secret, 'control': 'secret_identifier',
             'required_bytes': 42}
    admission = {'resource_status': 'insufficient', 'required_bytes': 3000,
                 'available_ram_bytes': 2000, 'total_ram_bytes': 4000, 'ram_ceiling': 500,
                 'error_category': 'memory', 'message': secret}
    text = _event('render_child_error', child) + _event('resource_admission', admission)
    text += ('Traceback (most recent call last):\n'
             '  File "/home/private/render_process.py", line 10, in run\n'
             '  File "/home/private/render_process.py", line 11, in _run_process\n'
             '  File "/home/private/render_process.py", line 12, in _render_child\n'
             '  File "/home/private/image_loader.py", line 13, in load_linear_raw_stack\n'
             '  File "/home/private/resource_policy.py", line 14, in plan_execution\n'
             '  File "/home/private/resource_policy.py", line 15, in require_memory\n'
             'core.resource_policy.ResourceLimitError: ' + secret + '\n')
    bundle = report_bundle.build_report_bundle(log_text=text, crash_text='', facts={})
    contents = _contents(bundle)
    assert contents['events.json'][0]['parameters'] == {k: child[k] for k in
        ('error_category', 'error_origin', 'render_stage')}
    assert contents['events.json'][1]['parameters'] == {k: admission[k] for k in
        ('resource_status', 'required_bytes', 'available_ram_bytes', 'total_ram_bytes', 'ram_ceiling')}
    assert contents['error.json']['kind'] == 'ResourceLimitError'
    assert [item['function'] for item in contents['error.json']['stack']] == [
        'run', '_run_process', '_render_child', 'load_linear_raw_stack', 'plan_execution', 'require_memory']
    assert all(value not in repr(contents) + bundle.subject + bundle.summary + bundle.log_excerpt
               for value in ('privateperson', '/home/private', 'photo.png', 'token-xyz', 'secret_identifier'))


def test_new_events_reject_untrusted_values_and_do_not_mark_session_failed():
    text = _event('render_child_error', {'error_category': 'private', 'error_origin': 'private',
                  'render_stage': 'private'})
    text += _event('resource_admission', {'resource_status': 'unknown', 'required_bytes': -1,
                   'available_ram_bytes': 10**16, 'total_ram_bytes': True, 'ram_ceiling': 1.5})
    bundle = report_bundle.build_report_bundle(log_text=text, crash_text='', facts={})
    assert [e['parameters'] for e in _contents(bundle)['events.json']] == [{}, {'resource_status': 'unknown'}]
    assert not bundle.has_error


@pytest.mark.parametrize('category', ['resource_limit', 'memory', 'os', 'value', 'runtime', 'other'])
@pytest.mark.parametrize('origin', ['child_worker', 'child_setup_or_transfer'])
@pytest.mark.parametrize('stage', ['prepare', 'align', 'fuse', 'finish', 'unknown'])
def test_child_error_taxonomy_survives_bundle(category, origin, stage):
    fields = {'error_category': category, 'error_origin': origin, 'render_stage': stage}
    bundle = report_bundle.build_report_bundle(log_text=_event('render_child_error', fields),
                                                crash_text='', facts={})
    assert _contents(bundle)['events.json'][0]['parameters'] == fields


def test_traceback_timestamp_requires_adjacent_error_header():
    trace = 'Traceback (most recent call last):\nValueError: private\n'
    header = '2026-11-10 10:11:12,123 ERROR pid=1 thread=Main session=x operation=x openfocus: private\n'
    def error(text):
        return _contents(report_bundle.build_report_bundle(log_text=text, crash_text='', facts={}))['error.json']
    assert error(header + trace)['timestamp'] == '2026-11-10T10:11:12.123'
    assert 'timestamp' not in error(header + 'intervening line\n' + trace)
    assert 'timestamp' not in error(header.replace('ERROR', 'INFO') + trace)
    assert 'timestamp' not in error(header.replace('2026-11-10', '2026-99-99') + trace)
    assert 'timestamp' not in error(header + trace + trace)
