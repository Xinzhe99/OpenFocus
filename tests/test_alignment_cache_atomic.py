"""Cache identities and atomic generations survive cancellation and I/O errors."""
import json
import os
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from core.registration import RegistrationCancelled
from utils import align_cache


@pytest.fixture
def stack(tmp_path):
    folder = tmp_path / 'stack'
    folder.mkdir()
    rng = np.random.default_rng(1208)
    images = [rng.integers(0, 65535, (24, 32, 3), dtype=np.uint16)
              for _ in range(3)]
    names = [f'frame{index}.png' for index in range(len(images))]
    for name, image in zip(names, images):
        assert cv2.imwrite(str(folder / name), image)
    return str(folder), names, images


def _save(stack, **kwargs):
    folder, names, images = stack
    align_cache.save_aligned(folder, names, images, ('openfocus',),
                             1024, **kwargs)


def _load(stack, **kwargs):
    folder, names, _ = stack
    return align_cache.load_aligned(folder, names, ('openfocus',),
                                    1024, **kwargs)


@pytest.mark.parametrize('change', [
    {'brightness_reference': 'first'},
    {'equalize_brightness': False},
    {'backend': 'cupy-linear'},
    {'effective_params': {'max_shift_ratio': 0.25}},
    {'effective_params': {'max_shift_ratio': 0.08, 'failure_policy': 'fail', 'reference_frame': 1}},
    {'effective_params': {'max_shift_ratio': 0.08, 'failure_policy': 'continue', 'reference_frame': 2}},
    {'source_revision': 2},
    {'frame_ids': [2, 1, 0]},
    {'dtype': '<f4'},
])
def test_request_identity_prevents_cross_request_pixel_reuse(stack, change):
    identity = {'brightness_reference': 'last', 'equalize_brightness': True,
                'backend': 'opencv-lanczos4',
                'effective_params': {'max_shift_ratio': 0.08, 'failure_policy': 'continue', 'reference_frame': 1},
                'source_revision': 1, 'frame_ids': [0, 1, 2], 'dtype': '<u2'}
    _save(stack, identity=identity)
    loaded = _load(stack, identity=dict(reversed(list(identity.items()))))
    assert loaded is not None
    for expected, actual in zip(stack[2], loaded):
        np.testing.assert_array_equal(actual, expected)
    assert _load(stack, identity={**identity, **change}) is None


def test_cancelled_encoding_preserves_previous_complete_generation(stack, monkeypatch):
    _save(stack)
    previous = [image.copy() for image in stack[2]]
    cancelled = Event()
    original = align_cache.imwrite_auto

    def encode_and_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled.set()
        return result

    monkeypatch.setattr(align_cache, 'imwrite_auto', encode_and_cancel)
    replacement = (stack[0], stack[1], [np.zeros_like(image) for image in stack[2]])
    with pytest.raises(RegistrationCancelled):
        _save(replacement, should_cancel=cancelled.is_set)
    loaded = _load(stack)
    assert loaded is not None
    for expected, actual in zip(previous, loaded):
        np.testing.assert_array_equal(actual, expected)
    assert not list(Path(stack[0]).rglob('.pending-*'))


def test_failed_manifest_replace_preserves_previous_generation(stack, monkeypatch):
    _save(stack)
    original = align_cache.os.replace

    def fail_manifest(source, destination):
        if os.path.basename(destination) == 'meta.json':
            raise PermissionError('manifest is locked')
        return original(source, destination)

    monkeypatch.setattr(align_cache.os, 'replace', fail_manifest)
    _save((stack[0], stack[1], [np.zeros_like(image) for image in stack[2]]))
    loaded = _load(stack)
    assert loaded is not None
    for expected, actual in zip(stack[2], loaded):
        np.testing.assert_array_equal(actual, expected)


def test_cancelled_decode_never_returns_a_partial_stack(stack, monkeypatch):
    _save(stack)
    cancelled = Event()
    original = align_cache._read_frame
    reads = []

    def decode_and_cancel(path):
        image = original(path)
        reads.append(path)
        cancelled.set()
        return image

    monkeypatch.setattr(align_cache, '_read_frame', decode_and_cancel)
    with pytest.raises(RegistrationCancelled):
        _load(stack, should_cancel=cancelled.is_set)
    assert len(reads) == 1


def test_byte_limit_evicts_oldest_cache_and_preserves_unrelated_files(stack):
    _save(stack, identity={'revision': 1})
    root = Path(align_cache.cache_dir(stack[0]))
    unrelated = root / 'user-notes'
    unrelated.mkdir()
    (unrelated / 'keep.txt').write_text('unrelated', encoding='utf-8')
    manifest = next(root.glob('*/meta.json'))
    encoded_bytes = json.loads(manifest.read_text(encoding='utf-8'))['encoded_bytes']
    os.utime(manifest, (1, 1))
    _save(stack, identity={'revision': 2}, max_cache_bytes=encoded_bytes + 1)
    assert _load(stack, identity={'revision': 1}) is None
    assert _load(stack, identity={'revision': 2}) is not None
    assert (unrelated / 'keep.txt').read_text(encoding='utf-8') == 'unrelated'


def test_previous_cache_version_is_evicted_after_alignment_change(stack, monkeypatch):
    current_version = align_cache.CACHE_FORMAT_VERSION
    monkeypatch.setattr(align_cache, 'CACHE_FORMAT_VERSION', 3)
    _save(stack, identity={'revision': 1})
    root = Path(align_cache.cache_dir(stack[0]))
    old_manifest = next(root.glob('*/meta.json'))
    encoded_bytes = json.loads(old_manifest.read_text(encoding='utf-8'))['encoded_bytes']
    os.utime(old_manifest, (1, 1))

    monkeypatch.setattr(align_cache, 'CACHE_FORMAT_VERSION', current_version)
    assert _load(stack, identity={'revision': 1}) is None
    _save(stack, identity={'revision': 2}, max_cache_bytes=encoded_bytes + 1)
    assert not old_manifest.exists()
    assert _load(stack, identity={'revision': 2}) is not None


def test_over_budget_new_entry_does_not_remove_previous_cache(stack):
    _save(stack, identity={'revision': 1})
    _save(stack, identity={'revision': 2}, max_cache_bytes=1)
    assert _load(stack, identity={'revision': 1}) is not None
    assert _load(stack, identity={'revision': 2}) is None


def test_source_mutation_during_encoding_does_not_commit_stale_pixels(stack, monkeypatch):
    _save(stack)
    source = Path(stack[0]) / stack[1][0]
    contents, stat = source.read_bytes(), source.stat()
    original = align_cache.imwrite_auto

    def encode_and_mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_bytes(contents + b'changed during alignment cache encoding')
        return result

    monkeypatch.setattr(align_cache, 'imwrite_auto', encode_and_mutate)
    _save((stack[0], stack[1], [np.zeros_like(image) for image in stack[2]]))
    assert _load(stack) is None
    source.write_bytes(contents)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    loaded = _load(stack)
    assert loaded is not None
    for expected, actual in zip(stack[2], loaded):
        np.testing.assert_array_equal(actual, expected)


@pytest.fixture
def plan_stack(stack):
    from core.openfocus_alignment import estimate_alignment_plan
    folder, names, images = stack
    plan = estimate_alignment_plan(images, max_shift_ratio=0)
    return folder, names, images, plan


def _save_plan(stack, revision, **kwargs):
    folder, names, images, plan = stack
    align_cache.save_alignment_plan(folder, names, plan, images,
                                    identity={'revision': revision}, **kwargs)


def _load_plan(stack, revision):
    folder, names, images, _ = stack
    return align_cache.load_alignment_plan(folder, names, images,
                                           identity={'revision': revision})


def _plan_path(stack, revision):
    folder, names, images, _ = stack
    signature = align_cache._plan_signature(folder, names, images, {'revision': revision})
    return Path(align_cache.cache_dir(folder)) / 'plans' / signature / 'meta.json'


def test_plan_lru_budget_and_reestimated_pixels(plan_stack):
    from core.openfocus_alignment import estimate_alignment_plan
    from core.registration import apply_alignment_plan
    _save_plan(plan_stack, 1)
    first = _plan_path(plan_stack, 1)
    size = first.stat().st_size
    _save_plan(plan_stack, 2)
    second = _plan_path(plan_stack, 2)
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    assert _load_plan(plan_stack, 1) is not None  # refresh LRU
    _save_plan(plan_stack, 3, max_cache_bytes=2 * size + 64)
    third = _plan_path(plan_stack, 3)
    assert first.exists() and not second.exists() and third.exists()
    assert sum(path.stat().st_size for path in (first, third)) <= 2 * size + 64
    assert _load_plan(plan_stack, 2) is None
    images = plan_stack[2]
    rebuilt = estimate_alignment_plan(images, max_shift_ratio=0)
    for cached, fresh in zip(apply_alignment_plan(images, _load_plan(plan_stack, 1)),
                             apply_alignment_plan(images, rebuilt)):
        np.testing.assert_array_equal(cached, fresh)


@pytest.mark.parametrize('last_kind', ['plan', 'aligned'])
def test_shared_budget_evicts_other_kind_even_with_same_signature(plan_stack, monkeypatch, last_kind):
    stack = plan_stack[:3]
    folder, names, _ = stack
    signature = align_cache._signature(folder, names, ('openfocus',), 1024,
                                       identity={'revision': 1})
    monkeypatch.setattr(align_cache, '_plan_signature', lambda *args: signature)
    if last_kind == 'plan':
        _save(stack, identity={'revision': 1})
        old = Path(align_cache.cache_dir(folder)) / signature / 'meta.json'
        os.utime(old, (1, 1))
        _save_plan(plan_stack, 1)
        new = _plan_path(plan_stack, 1)
        budget = new.stat().st_size
        _save_plan(plan_stack, 1, max_cache_bytes=budget)
        assert not old.exists() and _load_plan(plan_stack, 1) is not None
    else:
        _save_plan(plan_stack, 1)
        old = _plan_path(plan_stack, 1)
        os.utime(old, (1, 1))
        _save(stack, identity={'revision': 1})
        new = Path(align_cache.cache_dir(folder)) / signature / 'meta.json'
        budget = json.loads(new.read_text(encoding='utf-8'))['encoded_bytes']
        _save(stack, identity={'revision': 1}, max_cache_bytes=budget)
        assert not old.exists() and _load(stack, identity={'revision': 1}) is not None


@pytest.mark.parametrize('failure', ['oversize', 'zero', 'cancel', 'json', 'replace'])
def test_failed_plan_publication_preserves_previous(plan_stack, monkeypatch, failure):
    _save_plan(plan_stack, 1)
    previous = _plan_path(plan_stack, 1).read_bytes()
    kwargs = {}
    if failure in ('oversize', 'zero'):
        kwargs['max_cache_bytes'] = 1 if failure == 'oversize' else 0
    elif failure == 'cancel':
        original = align_cache.json.dump
        cancelled = Event()
        def encode(*args, **kwargs):
            original(*args, **kwargs)
            cancelled.set()
        monkeypatch.setattr(align_cache.json, 'dump', encode)
        kwargs['should_cancel'] = cancelled.is_set
    elif failure == 'json':
        def fail(*args, **kwargs):
            raise OSError('write failed')
        monkeypatch.setattr(align_cache.json, 'dump', fail)
    else:
        def fail(*args, **kwargs):
            raise OSError('replace failed')
        monkeypatch.setattr(align_cache.os, 'replace', fail)
    if failure == 'cancel':
        with pytest.raises(RegistrationCancelled):
            _save_plan(plan_stack, 1, **kwargs)
    else:
        _save_plan(plan_stack, 1, **kwargs)
    assert _plan_path(plan_stack, 1).read_bytes() == previous
    assert _load_plan(plan_stack, 1) == plan_stack[3]
    assert not list(Path(plan_stack[0]).rglob('.plan-*'))


def test_prune_preserves_unowned_plan_entries_and_symlinks(plan_stack, tmp_path):
    _save_plan(plan_stack, 1)
    plans = _plan_path(plan_stack, 1).parent.parent
    unknown = plans / ('a' * 32)
    unknown.mkdir()
    (unknown / 'meta.json').write_text('{}', encoding='utf-8')
    (unknown / 'user.txt').write_text('keep', encoding='utf-8')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep.txt').write_text('keep', encoding='utf-8')
    link = plans / ('b' * 32)
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        link = None
    stray = plans / 'notes.txt'
    stray.write_text('keep', encoding='utf-8')
    os.utime(_plan_path(plan_stack, 1), (1, 1))
    _save_plan(plan_stack, 2, max_cache_bytes=_plan_path(plan_stack, 1).stat().st_size + 64)
    assert not _plan_path(plan_stack, 1).exists()
    assert (unknown / 'user.txt').read_text(encoding='utf-8') == 'keep'
    assert stray.read_text(encoding='utf-8') == 'keep'
    assert (outside / 'keep.txt').read_text(encoding='utf-8') == 'keep'
    if link is not None:
        assert link.is_symlink()


def test_failed_eviction_does_not_credit_unremoved_bytes(plan_stack, monkeypatch):
    _save_plan(plan_stack, 1)
    _save_plan(plan_stack, 2)
    first, second = (_plan_path(plan_stack, revision) for revision in (1, 2))
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))
    original = align_cache.shutil.rmtree
    def remove(path, *args, **kwargs):
        if Path(path) == first.parent:
            raise OSError('busy')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(align_cache.shutil, 'rmtree', remove)
    _save_plan(plan_stack, 3, max_cache_bytes=first.stat().st_size + 64)
    assert first.exists() and not second.exists() and _plan_path(plan_stack, 3).exists()


@pytest.mark.parametrize('invalid', ['version', 'signature', 'backend', 'count', 'extra'])
def test_prune_skips_unowned_plan_manifests(plan_stack, invalid):
    _save_plan(plan_stack, 1)
    path = _plan_path(plan_stack, 1)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if invalid == 'extra':
        (path.parent / 'user.txt').write_text('keep', encoding='utf-8')
    else:
        manifest[invalid] = {'version': -1, 'signature': 'wrong',
                             'backend': 'unknown', 'count': -1}[invalid]
        path.write_text(json.dumps(manifest), encoding='utf-8')
    os.utime(path, (1, 1))
    _save_plan(plan_stack, 2)
    budget = _plan_path(plan_stack, 2).stat().st_size
    _save_plan(plan_stack, 2, max_cache_bytes=budget)
    assert path.exists()


def test_prune_skips_plan_manifest_symlink(plan_stack, tmp_path):
    _save_plan(plan_stack, 1)
    plans = _plan_path(plan_stack, 1).parent.parent
    directory = plans / ('c' * 32)
    directory.mkdir()
    outside = tmp_path / 'outside.json'
    outside.write_text(json.dumps({'version': align_cache.PLAN_CACHE_VERSION,
        'signature': directory.name, 'backend': 'openfocus-wide4d-v3', 'count': 1}),
        encoding='utf-8')
    try:
        (directory / 'meta.json').symlink_to(outside)
    except OSError:
        pytest.skip('Symlink creation unavailable')
    os.utime(outside, (1, 1))
    _save_plan(plan_stack, 2, max_cache_bytes=_plan_path(plan_stack, 1).stat().st_size + 64)
    assert (directory / 'meta.json').is_symlink()
    assert outside.exists()


def test_prune_skips_symlinked_plans_root(stack, tmp_path):
    root = Path(align_cache.cache_dir(stack[0]))
    root.mkdir()
    outside = tmp_path / 'external-plans'
    directory = outside / ('d' * 32)
    directory.mkdir(parents=True)
    manifest = directory / 'meta.json'
    manifest.write_text(json.dumps({'version': align_cache.PLAN_CACHE_VERSION,
        'signature': directory.name, 'backend': 'openfocus-wide4d-v3', 'count': 1}),
        encoding='utf-8')
    try:
        (root / 'plans').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('Symlink creation unavailable')
    _save(stack, identity={'revision': 1})
    aligned = next(root.glob('*/meta.json'))
    budget = json.loads(aligned.read_text(encoding='utf-8'))['encoded_bytes']
    os.utime(manifest, (1, 1))
    _save(stack, identity={'revision': 1}, max_cache_bytes=budget)
    assert manifest.exists() and (root / 'plans').is_symlink()
