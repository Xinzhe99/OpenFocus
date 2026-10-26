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
    {'source_revision': 2},
    {'frame_ids': [2, 1, 0]},
    {'dtype': '<f4'},
])
def test_request_identity_prevents_cross_request_pixel_reuse(stack, change):
    identity = {'brightness_reference': 'last', 'equalize_brightness': True,
                'backend': 'opencv-lanczos4',
                'effective_params': {'max_shift_ratio': 0.08},
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
