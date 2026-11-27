"""Regression tests for 16-bit RAW memory admission, in-place decode, alignment, and OpenFocus A/B/C."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import rawpy

from core import image_loader, registration, resource_policy as policy, source_frames
from core.render_preflight import estimate_render_memory
from utils import linear_dng

GiB = 1024 ** 3
MiB = 1024 ** 2


def _laptop_16gb_snapshot():
    """Reproduce a 16 GiB RAM machine with ~8.4 GiB available (~6.05 GiB ceiling)."""
    return policy.ResourceSnapshot(
        available_ram=9_025_671_168,
        total_ram=16_844_877_824,
        physical_cores=8,
        logical_cpus=16,
        commit_headroom=42_031_034_368,
    )


def _raw_102mp_frames(tmp_path, count=29):
    encoded = 86_810_000
    return [
        source_frames.SourceFrame(
            str(tmp_path / f'_DSF{index:04d}.RAF'),
            0,
            (8746, 11662, 3),
            np.dtype(np.uint16),
            None,
            (encoded, 0),
            native_shape=(8746, 11662, 3),
            raw_sensor_shape=(8754, 11808),
        )
        for index in range(count)
    ]


@pytest.mark.parametrize('alignment', ['openfocus', 'ecc', 'homography'])
@pytest.mark.parametrize('algorithm', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
def test_102mp_raw_stack_admits_cleanly_on_16gb_machine(monkeypatch, tmp_path, alignment, algorithm):
    snapshot = _laptop_16gb_snapshot()
    monkeypatch.setattr(policy, 'probe_resources', lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(policy, 'probe_memory', lambda: snapshot)

    frames = _raw_102mp_frames(tmp_path, count=29)
    est = estimate_render_memory(frames, algorithm, {}, register=True, alignment=alignment)

    ceiling = policy.ram_ceiling(snapshot)
    assert ceiling == 9_025_671_168 - int(policy.RESERVE_RATIO * snapshot.total_ram)
    assert est.source_ram == policy.raw_decode_bytes(11808, 8754, frames[0].nbytes, 86_810_000)
    # One 102MP uint16 RAW decode must stay ~2.2 GiB (not the old inflated 6.89 GiB).
    assert 1.5 * GiB < est.source_ram < 2.5 * GiB
    assert est.registration_ram < 3.0 * GiB
    assert est.fusion_ram < 3.2 * GiB
    assert not est.ram_short


def test_raw_decode_admission_still_rejects_genuinely_insufficient_ram(tmp_path):
    frames = _raw_102mp_frames(tmp_path, count=1)
    tight = policy.ResourceSnapshot(
        available_ram=2 * GiB,
        total_ram=8 * GiB,
        physical_cores=4,
        logical_cpus=8,
    )
    required = source_frames.frame_resolution_bytes(frames[0], metadata_only=True)
    with pytest.raises(policy.ResourceLimitError):
        policy.require_memory(required, tight)


def test_read_raw_swaps_rgb_to_bgr_in_place_without_full_frame_copy(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    source_rgb = rng.integers(0, 65536, (48, 64, 3), dtype=np.uint16)
    expected_bgr = np.ascontiguousarray(source_rgb[:, :, ::-1])
    post_buffer = source_rgb.copy()
    post_ptr = post_buffer.ctypes.data

    class FakeRaw:
        sizes = SimpleNamespace(
            width=64, height=48, raw_width=64, raw_height=48, pixel_aspect=1.0, flip=0,
        )
        raw_type = 'flat'
        raw_pattern = np.zeros((2, 2), np.uint8)
        num_colors = 3

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def open_file(self, _path):
            pass

        def unpack(self):
            pass

        def postprocess(self, **_kwargs):
            return post_buffer

    path = tmp_path / 'sample.raf'
    path.write_bytes(b'raw')
    monkeypatch.setattr(image_loader, 'RAWPY_AVAILABLE', True)
    monkeypatch.setattr(
        image_loader, 'rawpy',
        SimpleNamespace(RawPy=FakeRaw, RawType=SimpleNamespace(Flat='flat', Stack='stack')),
        raising=False,
    )

    result = image_loader._read_raw(str(path), scale_factor=1.0)
    assert result is not None
    assert result.ctypes.data == post_ptr
    np.testing.assert_array_equal(result, expected_bgr)


def test_load_linear_raw_applies_lut_and_swap_in_place_with_bounded_scratch(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    height, width = 1024, 1024
    raw_rgb = rng.integers(0, 16384, (height, width, 3), dtype=np.uint16)
    post_buffer = raw_rgb.copy()
    post_ptr = post_buffer.ctypes.data
    expected_lut = np.minimum(
        np.rint(np.arange(65536, dtype=np.float64) * (65535.0 / 16383.0)), 65535,
    ).astype(np.uint16)
    expected_bgr = np.ascontiguousarray(expected_lut[raw_rgb][:, :, ::-1])

    class FakeRaw:
        sizes = SimpleNamespace(
            width=width, height=height, raw_width=width, raw_height=height,
            pixel_aspect=1.0, flip=0,
        )
        raw_type = rawpy.RawType.Flat
        raw_pattern = np.array([[0, 1], [3, 2]], np.uint8)
        num_colors = 3
        rgb_xyz_matrix = np.array([
            [0.8293, -0.1611, -0.1132],
            [-0.4759, 1.2711, 0.2275],
            [-0.1013, 0.2415, 0.5509],
        ])
        camera_whitebalance = [2.0, 1.0, 2.0, 1.0]
        black_level_per_channel = [0.0, 0.0, 0.0, 0.0]
        white_level = 16383

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def open_file(self, _path):
            pass

        def unpack(self):
            pass

        def postprocess(self, **_kwargs):
            return post_buffer

    path = tmp_path / 'linear.raf'
    path.write_bytes(b'raw')
    reservations = []
    monkeypatch.setattr(rawpy, 'RawPy', FakeRaw)
    monkeypatch.setattr(linear_dng, '_camera_identity', lambda *_args: ('Example', 'Camera'))
    monkeypatch.setattr(linear_dng, 'require_memory', reservations.append)

    image, metadata = linear_dng.load_linear_raw(path, scale_factor=1.0)
    assert metadata.camera_model == 'Camera'
    assert image.ctypes.data == post_ptr
    np.testing.assert_array_equal(image, expected_bgr)
    # Post-postprocess reservation must cover bounded 4 MiB band + LUT, not 2x full frame.
    assert len(reservations) == 3
    assert reservations[1] == policy.raw_decode_bytes(width, height, raw_rgb.nbytes, 3)
    assert reservations[2] <= 4 * MiB + 65536 * 10
    assert reservations[2] < raw_rgb.nbytes


def test_deferred_homography_and_ecc_admit_102mp_raw_on_16gb_machine(tmp_path, monkeypatch):
    snapshot = _laptop_16gb_snapshot()
    monkeypatch.setattr(policy, 'probe_resources', lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(policy, 'probe_memory', lambda: snapshot)

    frames = _raw_102mp_frames(tmp_path, count=29)
    for frame in frames:
        with open(frame.path, 'wb') as fh:
            fh.write(b'\x00')
    frames = [
        replace(frame, fingerprint=source_frames._fingerprint(frame.path))
        for frame in frames
    ]

    # Deferred ECC budget must not double-charge a packing copy or uint8 GPU stripe.
    ecc_budget = registration._plan_deferred_ecc(
        frames, 1024, True, policy.PREFERENCE_BALANCED, 0, None,
    )
    decode_bytes = source_frames.frame_resolution_bytes(frames[0])
    assert ecc_budget.prepare_bytes == decode_bytes
    assert ecc_budget.workers >= 1

    # Deferred Homography must plan from downscaled feature scratch + decode, not image.size * 16.
    plans = []
    real_plan = policy.plan_execution

    def capture_plan(workload, **kwargs):
        plan = real_plan(workload, **kwargs)
        plans.append((workload, plan))
        return plan

    monkeypatch.setattr(policy, 'plan_execution', capture_plan)
    monkeypatch.setattr(
        registration.ImageRegistration, '_process_homography',
        lambda self, imgs, **kwargs: imgs,
    )
    engine = registration.ImageRegistration('homography', defer_warps=True)
    result = engine.process(frames, thread_count=0, resource_preference='balanced')
    assert len(result) == 29
    assert len(plans) == 1
    workload, plan = plans[0]
    assert workload.fixed_bytes == 0
    assert workload.per_worker_bytes < 2.5 * GiB
    assert plan.workers >= 1


def test_perspective_scratch_and_gpu_staging_recognize_contiguous_raw_source_frames(tmp_path):
    from core.openfocus_alignment import SourceFingerprint
    from fusion_methods._openfocus_gpu_a import _source_staging_bytes

    frame = _raw_102mp_frames(tmp_path, count=1)[0]
    source_fp = SourceFingerprint(
        frame.shape, frame.dtype.str, '0' * 64, frame.path, 0, frame.fingerprint, '', 'pixel',
    )
    warp = registration.PerspectiveWarp(
        source_fp, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        '<f4', (11662, 8746), None, method='ecc',
    )
    scratch = registration._perspective_scratch_bytes(frame, warp, frame.shape)
    assert scratch is not None
    # Contiguous RAW SourceFrame needs ~1x output + small scratch, not the 2x fallback.
    assert frame.nbytes < scratch < frame.nbytes + 128 * MiB

    # GPU A staging should not charge an extra frame.nbytes packing copy for RAW SourceFrame.
    with open(frame.path, 'wb') as fh:
        fh.write(b'\x00')
    frame = replace(frame, fingerprint=source_frames._fingerprint(frame.path))
    staging = _source_staging_bytes(frame, frame.shape, None, native_upload=True)
    expected = source_frames.frame_resolution_bytes(frame) + frame.nbytes
    assert staging == expected
