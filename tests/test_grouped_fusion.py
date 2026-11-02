"""Focused coverage for the disk-backed grouped fusion engine."""

import tempfile
from collections import namedtuple

import cv2
import numpy as np
import pytest
from PIL import Image

from core import grouped_fusion as grouped
from core import resource_policy
from core.registration import RegistrationCancelled


def _files(tmp_path, values, dtype=np.uint8):
    paths = []
    for index, value in enumerate(values):
        path = tmp_path / f'{index}.png'
        image = np.full((8, 9, 3), value, dtype=dtype)
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


@pytest.fixture
def admitted(monkeypatch):
    monkeypatch.setattr(grouped.resource_policy, 'plan_execution', lambda *a, **k: None)


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    root = tmp_path / 'scratch'
    root.mkdir()
    original = tempfile.TemporaryDirectory
    monkeypatch.setattr(grouped.tempfile, 'TemporaryDirectory',
                        lambda **kw: original(dir=root, **kw))
    return root


def test_order_overlap_rounds_and_progress(tmp_path, admitted, scratch_root):
    paths = _files(tmp_path, range(7))
    groups, updates = [], []

    def fuse(images):
        groups.append([int(image[0, 0, 0]) for image in images])
        return np.maximum.reduce(images)

    result = grouped.grouped_fuse(paths, 3, 1, fuse,
                                  progress=lambda *args: updates.append(args))
    assert groups == [[0, 1, 2], [2, 3, 4], [4, 5, 6], [2, 4, 6]]
    assert updates == [(1, 1, 3), (1, 2, 3), (1, 3, 3), (2, 1, 1)]
    assert np.all(result == 6)
    assert not list(scratch_root.iterdir())


def test_two_frame_groups_carry_singletons_without_fusing(tmp_path, admitted):
    calls = []

    def fuse(images):
        calls.append([int(image[0, 0, 0]) for image in images])
        return np.maximum.reduce(images)

    result = grouped.grouped_fuse(_files(tmp_path, range(5)), 2, 0, fuse)
    assert calls == [[0, 1], [2, 3], [1, 3], [3, 4]]
    assert np.all(result == 4)


@pytest.mark.parametrize('dtype,values', [(np.uint8, [4, 100, 200, 252]),
                                          (np.uint16, [400, 10000, 40000, 64000])])
def test_quantization_and_approximation_quality(tmp_path, admitted, dtype, values):
    paths = _files(tmp_path, values, dtype)

    def mean(images):
        return np.rint(np.mean(np.stack(images).astype(np.float64), axis=0)).astype(dtype)

    result = grouped.grouped_fuse(paths, 2, 0, mean)
    assert result.dtype == dtype
    # Balanced averaging differs from whole-stack averaging by at most one
    # quantization level, including its intermediate TIFF round trip.
    assert abs(int(result[0, 0, 0]) - round(sum(values) / len(values))) <= 1


@pytest.mark.parametrize('dtype,scale,tolerance', [(np.uint8, 1, 2), (np.uint16, 257, 300)])
def test_real_fusion_of_identical_frames_matches_full_stack(tmp_path, dtype, scale, tolerance):
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import normalize_fuse_input, quantize_fuse_output

    y, x = np.indices((48, 48))
    pattern = (((x * 7 + y * 11) % 200 + 20) * scale).astype(dtype)
    image = np.repeat(pattern[..., None], 3, axis=2)
    paths = []
    for index in range(5):
        path = tmp_path / f'{index}.png'
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    fusion = MultiFocusFusion(algorithm='guided_filter', use_gpu=False, exposure_match=False)

    def fuse(images):
        values, is_16bit = normalize_fuse_input(images)
        return quantize_fuse_output(fusion.fuse(values, thread_count=1), is_16bit)

    reference = fuse([image] * len(paths))
    result = grouped.grouped_fuse(paths, 3, 1, fuse, thread_count=1)
    assert result.dtype == dtype
    assert np.mean(np.abs(result.astype(np.float64) - reference.astype(np.float64))) <= tolerance


@pytest.mark.parametrize('failure', ['cancel', 'callback'])
def test_scratch_cleanup_after_interruption(tmp_path, admitted, scratch_root, failure):
    paths = _files(tmp_path, range(7))
    cancelled = [False]

    def progress(stage, completed, total):
        if completed == 1:
            if failure == 'cancel':
                cancelled[0] = True
            else:
                raise RuntimeError('progress failed')

    with pytest.raises(RegistrationCancelled if failure == 'cancel' else RuntimeError):
        grouped.grouped_fuse(paths, 3, 1, np.maximum.reduce,
                             should_cancel=lambda: cancelled[0], progress=progress)
    assert not list(scratch_root.iterdir())


def test_cancellation_after_fusion_cleans_scratch(tmp_path, admitted, scratch_root):
    cancelled = [False]
    calls = [0]

    def fuse(images):
        calls[0] += 1
        if calls[0] == 2:
            cancelled[0] = True
        return np.maximum.reduce(images)

    with pytest.raises(RegistrationCancelled):
        grouped.grouped_fuse(_files(tmp_path, range(6)), 2, 0, fuse,
                             should_cancel=lambda: cancelled[0])
    assert not list(scratch_root.iterdir())


def test_resource_admission_precedes_decoding(tmp_path, monkeypatch, scratch_root):
    paths = _files(tmp_path, range(3))
    requests = []
    from core import multi_focus_fusion
    monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 12345)

    def reject(workload, **kwargs):
        requests.append((workload.fixed_bytes, kwargs))
        raise resource_policy.ResourceLimitError('memory ceiling')

    monkeypatch.setattr(grouped.resource_policy, 'plan_execution', reject)
    monkeypatch.setattr(grouped, 'decode_image', lambda path: pytest.fail('decoded before admission'))
    with pytest.raises(resource_policy.ResourceLimitError):
        grouped.grouped_fuse(paths, 2, 0, np.maximum.reduce,
                             thread_count=2, resource_preference='responsive')
    assert requests[0][0] >= 2 * 8 * 9 * 3
    assert requests[0][1] == {'preference': 'responsive', 'user_threads': 2,
                              'ram_budget_override': 12345}
    assert not list(scratch_root.iterdir())


def test_actual_memory_budget_rejects_group(tmp_path, monkeypatch):
    paths = _files(tmp_path, range(3))
    gib = 1024 ** 3
    snapshot = resource_policy.ResourceSnapshot(available_ram=gib + 1024,
                                                total_ram=2 * gib,
                                                physical_cores=2, logical_cpus=2)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args: snapshot)
    monkeypatch.setattr(grouped, 'decode_image', lambda path: pytest.fail('decoded over ceiling'))
    with pytest.raises(resource_policy.ResourceLimitError):
        grouped.grouped_fuse(paths, 2, 0, np.maximum.reduce)


def test_configured_memory_limit_rejects_group(tmp_path, monkeypatch):
    from core import multi_focus_fusion
    paths = _files(tmp_path, range(3))
    gib = 1024 ** 3
    snapshot = resource_policy.ResourceSnapshot(available_ram=8 * gib,
                                                total_ram=16 * gib,
                                                physical_cores=2, logical_cpus=2)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args: snapshot)
    monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 1000)
    monkeypatch.setattr(grouped, 'decode_image', lambda path: pytest.fail('decoded over ceiling'))
    with pytest.raises(resource_policy.ResourceLimitError):
        grouped.grouped_fuse(paths, 2, 0, np.maximum.reduce)


def test_finished_round_removes_only_consumed_scratch(tmp_path, admitted, scratch_root):
    observed = {}

    def progress(stage, completed, total):
        if stage in (3, 4) and completed == 1:
            directory = next(scratch_root.iterdir())
            observed[stage] = {path.name for path in directory.iterdir()}

    grouped.grouped_fuse(_files(tmp_path, range(10)), 2, 0,
                         np.maximum.reduce, progress=progress)
    assert observed[3] == {'1-5.tif', '2-1.tif', '2-2.tif', '3-1.tif'}
    assert observed[4] == {'1-5.tif', '3-1.tif'}
    assert not list(scratch_root.iterdir())


def test_scratch_disk_ceiling(tmp_path, admitted, monkeypatch, scratch_root):
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(grouped.shutil, 'disk_usage', lambda path: usage(1, 1, 0))
    with pytest.raises(resource_policy.ResourceLimitError, match='disk'):
        grouped.grouped_fuse(_files(tmp_path, range(4)), 2, 0, np.maximum.reduce)
    assert not list(scratch_root.iterdir())


def test_rejects_multipage_before_decode(tmp_path, admitted, monkeypatch):
    path = tmp_path / 'stack.tif'
    Image.new('RGB', (8, 8)).save(path, save_all=True,
                                   append_images=[Image.new('RGB', (8, 8))])
    monkeypatch.setattr(grouped, 'decode_image', lambda path: pytest.fail('decoded multipage'))
    with pytest.raises(ValueError, match='multipage'):
        grouped.grouped_fuse([path], 2, 0, np.maximum.reduce)


def test_scale_only_applied_to_originals(tmp_path, admitted):
    paths = _files(tmp_path, range(4))
    result = grouped.grouped_fuse(paths, 2, 0, np.maximum.reduce, scale_factor=0.5)
    assert result.shape == (4, 4, 3)


def test_preprocess_only_affects_original_frames(tmp_path, admitted):
    paths = _files(tmp_path, [10, 20, 30, 40, 50])
    result = grouped.grouped_fuse(
        paths, 2, 0, np.maximum.reduce,
        preprocess=lambda image: image + np.uint8(10))
    assert np.all(result == 60)


@pytest.mark.parametrize('group_size,overlap', [(1, 0), (2, 1), (3, 2), (3, -1)])
def test_invalid_group_geometry(tmp_path, group_size, overlap):
    with pytest.raises(ValueError):
        grouped.grouped_fuse(_files(tmp_path, [1]), group_size, overlap, np.maximum.reduce)


def test_rejects_invalid_fusion_result(tmp_path, admitted, scratch_root):
    with pytest.raises(ValueError, match='uint8 or uint16'):
        grouped.grouped_fuse(_files(tmp_path, range(4)), 2, 0,
                             lambda images: images[0].astype(np.float32))
    assert not list(scratch_root.iterdir())


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_different_crops_and_singleton_keep_source_coordinates(tmp_path, admitted, dtype):
    from core.registration import _record_grouped_crop
    y, x = np.indices((48, 52))
    image = np.stack([x, y, x + y], axis=-1).astype(dtype)
    paths = []
    for index in range(5):
        path = tmp_path / f'{index}.png'
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    calls = []

    def fuse(images):
        calls.append(len(images))
        h, w = images[0].shape[:2]
        left = 1 if len(calls) % 2 else 3
        top = 2 if len(calls) % 2 else 1
        _record_grouped_crop(top, h - 1, left, w - 2)
        return images[0][top:h - 1, left:w - 2].copy()

    result = grouped.grouped_fuse(paths, 2, 0, fuse)
    x0, y0 = map(int, result[0, 0, :2])
    assert result.dtype == dtype
    assert calls == [2, 2, 2, 2]
    np.testing.assert_array_equal(result, image[y0:y0 + result.shape[0], x0:x0 + result.shape[1]])


def test_real_registration_crops_across_rounds(tmp_path, monkeypatch):
    from core import registration as registration_module
    y, x = np.indices((64, 68))
    image = np.stack([x, y, x + y], axis=-1).astype(np.uint8)
    paths = []
    for index in range(5):
        path = tmp_path / f'{index}.png'
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    monkeypatch.setattr(registration_module, '_of_estimate_similarity',
                        lambda *args, **kwargs: (1.0, 0.0, 1.0, 0.0, 0.0))
    registration = registration_module.ImageRegistration(method='openfocus')

    def fuse(images):
        return registration.process(images, thread_count=1)[0].copy()

    result = grouped.grouped_fuse(paths, 2, 0, fuse)
    np.testing.assert_array_equal(result, image[12:52, 12:56])


def test_scaled_resource_admission_keeps_full_decoder_scratch(tmp_path, monkeypatch):
    paths = _files(tmp_path, [1, 2])
    requests = []
    def reject(workload, **kwargs):
        requests.append(workload.fixed_bytes)
        raise resource_policy.ResourceLimitError('stop before decode')
    monkeypatch.setattr(grouped.resource_policy, 'plan_execution', reject)
    with pytest.raises(resource_policy.ResourceLimitError):
        grouped.grouped_fuse(paths, 2, 0, np.maximum.reduce, scale_factor=0.5)
    encoded = max(path.stat().st_size for path in paths)
    assert requests == [2 * 4 * 4 * 6 + 8 * 9 * 24 + encoded]


def test_alpha_is_rejected_before_processing(tmp_path, monkeypatch):
    path = tmp_path / 'transparent.png'
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    image[2:14, 2:14, 3] = 255
    assert cv2.imwrite(str(path), image)
    monkeypatch.setattr(grouped, 'decode_image', lambda path: pytest.fail('decoded alpha input'))
    with pytest.raises(ValueError, match='alpha-channel'):
        grouped.grouped_fuse([path], 2, 0, np.maximum.reduce)


def test_projective_horizon_is_rejected_for_grouped_registration():
    from core.registration import grouped_crop_offsets, _compute_valid_region_from_transforms
    transform = np.eye(3)
    transform[2, 0] = -0.04
    token = grouped_crop_offsets.set([])
    try:
        with pytest.raises(ValueError, match='projective transform'):
            _compute_valid_region_from_transforms([transform], (64, 64))
    finally:
        grouped_crop_offsets.reset(token)
