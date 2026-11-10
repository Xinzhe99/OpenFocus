"""Bounded source preparation, ownership, and producer lifecycle contracts."""
import gc
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core import frame_storage, source_frames
from core.registration import RegistrationCancelled
from fusion_methods import openfocus_ab


def _source(tmp_path, name, value, dtype=np.uint16, alpha=False):
    shape = (12, 16, 4 if alpha else 3)
    image = np.full(shape, value, dtype)
    if alpha:
        image[:, :, 3] = 0
        image[2:10, 3:13, 3] = np.iinfo(dtype).max
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    path = tmp_path / name
    encoded.tofile(str(path))
    return source_frames.scan_source(str(path), header_only=True)[0], image


def test_collect_native_sources_keeps_metadata_and_repeatable_provenance(tmp_path, monkeypatch):
    from core import image_loader
    first, _ = _source(tmp_path, '原图8.png', 73, np.uint8)
    last, _ = _source(tmp_path, '原图16.png', 19001)
    calls = []
    decode = image_loader.decode_image

    def tracked(path, **kwargs):
        calls.append(path)
        return decode(path, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_image', tracked)
    frames = openfocus_ab._collect_images([first, last], normalize=False)
    assert frames == [first, last]
    assert calls == []
    matched = source_frames.match_exposure_frames(frames)
    assert calls == [last.path]
    assert [frame.dtype for frame in matched] == [np.uint8, np.uint16]
    assert matched[0].source is first and matched[1].source is last
    initial = source_frames.resolve_frame(matched[0]).copy()
    np.testing.assert_array_equal(source_frames.resolve_frame(matched[0]), initial)
    assert calls == [last.path, first.path]
    monkeypatch.setattr(source_frames, '_fingerprint', lambda path: (0, 0))
    with pytest.raises(OSError, match='Source changed'):
        source_frames.resolve_frame(matched[0])


def test_lazy_gain_does_not_modify_borrowed_input_and_retains_each_dtype():
    low = np.full((8, 9, 3), 64, np.uint8)
    high = np.full((8, 9, 3), 40000, np.uint16)
    low.flags.writeable = False
    high.flags.writeable = False
    frames = source_frames.match_exposure_frames([low, high])
    adjusted = source_frames.resolve_frame(frames[0])
    assert adjusted.dtype == np.uint8
    assert np.all(adjusted > low)
    assert np.all(low == 64) and np.all(high == 40000)
    assert not np.shares_memory(adjusted, low)
    assert source_frames.resolve_frame(frames[1]).dtype == np.uint16


def test_transparent_validation_keeps_lazy_crops_and_checks_every_alpha(tmp_path):
    first, pixels = _source(tmp_path, '透明1.png', 17000, alpha=True)
    last, _ = _source(tmp_path, '透明2.png', 21000, alpha=True)
    frames, region = source_frames.crop_transparent_frames([first, last])
    assert all(isinstance(frame, source_frames.ProcessedFrame) for frame in frames)
    assert frames[0].source is first and frames[1].source is last
    assert frames[0].dtype == np.uint16
    assert frames[0].shape == (8, 10, 3)
    np.testing.assert_array_equal(source_frames.resolve_frame(frames[0]), pixels[2:10, 3:13, :3])
    assert region[:2] == (pixels.shape, pixels.dtype)
    wrong = pixels.copy()
    wrong[2, 3, 3] = 0
    with pytest.raises(ValueError, match='alpha masks'):
        source_frames.crop_transparent_frames([first, wrong])


def test_ordered_producers_overlap_lazy_resolution_and_join_on_cancel():
    second_started = threading.Event()
    release_second = threading.Event()
    second_finished = threading.Event()
    cancelled = threading.Event()
    admitted = []
    frames = []

    def prepare(index):
        def process(image):
            admitted.append(index)
            if index == 0:
                assert second_started.wait(5)
            elif index == 1:
                second_started.set()
                try:
                    assert release_second.wait(5)
                finally:
                    second_finished.set()
            return image.copy()
        return process

    for index in range(5):
        frames.append(source_frames.process_frame(np.full((8, 9, 3), index, np.uint8), prepare(index)))

    def produce(index, should_cancel):
        openfocus_ab._check_cancel(should_cancel)
        image = source_frames.resolve_frame(frames[index])
        openfocus_ab._check_cancel(should_cancel)
        return image

    results = openfocus_ab._ordered_frame_results(len(frames), produce, 2, cancelled.is_set)
    index, image = next(results)
    assert index == 0 and np.all(image == 0)
    cancelled.set()
    exited = threading.Event()
    failures = []

    def cancel_consumer():
        try:
            next(results)
        except RegistrationCancelled:
            failures.append('cancelled')
        finally:
            exited.set()

    consumer = threading.Thread(target=cancel_consumer)
    consumer.start()
    try:
        assert not exited.wait(.05)
        assert not second_finished.is_set()
    finally:
        release_second.set()
        consumer.join(5)
    assert not consumer.is_alive() and failures == ['cancelled']
    assert second_finished.is_set()
    assert sorted(admitted) == [0, 1]


@pytest.mark.parametrize('ceiling', ['available_ram', 'shared_limit'])
def test_transient_ram_cap_counts_views_and_disk_fallback(tmp_path, monkeypatch, ceiling):
    from core import resource_policy
    gc.collect()
    baseline = frame_storage._ram_live_bytes
    monkeypatch.setattr(resource_policy, 'probe_memory',
                        lambda: SimpleNamespace(ram_known=True))
    monkeypatch.setattr(resource_policy, 'ram_budget', lambda snapshot: 2 * (baseline + 32))
    if ceiling == 'shared_limit':
        monkeypatch.setattr(resource_policy, 'ram_budget', lambda snapshot: 1024 ** 4)
        monkeypatch.setattr(resource_policy, 'FRAME_RAM_BUDGET_BYTES', baseline + 32)
    monkeypatch.setattr(resource_policy, 'require_memory', lambda *args, **kwargs: 1024)
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    first = frame_storage.create_transient_frame((16,), np.uint8)
    second = frame_storage.create_transient_frame((16,), np.uint8)
    view = first[2:]
    assert not isinstance(first, np.memmap) and not isinstance(second, np.memmap)
    overflow = frame_storage.create_transient_frame((1,), np.uint8)
    assert isinstance(overflow, np.memmap)
    del first
    gc.collect()
    assert frame_storage._ram_live_bytes == baseline + 32
    del view
    gc.collect()
    assert frame_storage._ram_live_bytes == baseline + 16
    third = frame_storage.create_transient_frame((16,), np.uint8)
    assert not isinstance(third, np.memmap)
    del second, third, overflow
    gc.collect()
    assert frame_storage._ram_live_bytes == baseline


def test_ram_snapshot_remains_independent_of_source_and_persistent_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    source = np.arange(36, dtype=np.uint16).reshape(6, 6)[:, ::-1]
    snapshot = frame_storage.store_frame(source, prefer_ram=True)
    expected = source.copy()
    source[:] = 0
    np.testing.assert_array_equal(snapshot, expected)
    assert not np.shares_memory(snapshot, source)
    persistent = frame_storage.store_frame(snapshot, prefer_ram=False)
    assert isinstance(persistent, np.memmap)
    snapshot[:] = 1
    np.testing.assert_array_equal(persistent, expected)


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
def test_native_fusion_rejects_pre_materialization_and_preserves_sources(tmp_path, monkeypatch, method):
    from core.multi_focus_fusion import MultiFocusFusion
    first, pixels = _source(tmp_path, f'{method}-8.png', 61, np.uint8)
    last, _ = _source(tmp_path, f'{method}-16.png', 16003)

    def forbidden(*args, **kwargs):
        raise AssertionError('OpenFocus path eagerly materialized the source stack')

    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    fusion = MultiFocusFusion(algorithm=method, tile_enabled=False, exposure_match=True)
    result = fusion.fuse([first, last], thread_count=1)
    assert result.shape == first.shape and result.dtype == np.float32
    np.testing.assert_array_equal(source_frames.resolve_frame(first), pixels)


def test_cli_lazy_loading_never_enters_full_decoder(tmp_path, monkeypatch):
    from core import cli, image_loader
    first, _ = _source(tmp_path, '命令行1.png', 33, np.uint8)
    last, _ = _source(tmp_path, '命令行2.png', 37001)

    def forbidden(*args, **kwargs):
        raise AssertionError('CLI eagerly decoded a header-backed source')

    monkeypatch.setattr(image_loader, 'iter_decoded_sources', forbidden)
    frames = cli._load_images([first.path, last.path], lazy=True)
    assert [frame.path for frame in frames] == [first.path, last.path]
    assert [frame.dtype for frame in frames] == [np.uint8, np.uint16]


def test_gui_render_publishes_lazy_cleaned_sources_for_retouch(qapp, tmp_path, monkeypatch):
    from core import workers
    first, pixels = _source(tmp_path, 'GUI原图1.png', 73, np.uint8)
    last, _ = _source(tmp_path, 'GUI原图2.png', 19001)

    def forbidden(*args, **kwargs):
        raise AssertionError('GUI eagerly materialized the source stack')

    monkeypatch.setattr(workers, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    worker = workers.RenderWorker([first, last], algorithm='openfocus_a',
        alignment='none', tile_enabled=False, thread_count=1, sensor_denoise=True)
    published = []
    worker.finished_signal.connect(lambda *args: published.append(args))
    worker._run_admitted()
    assert len(published) == 1
    sources, result = published[0][:2]
    assert all(isinstance(frame, source_frames.ProcessedFrame) for frame in sources)
    assert [frame.source for frame in sources] == [first, last]
    assert [frame.dtype for frame in sources] == [np.uint8, np.uint16]
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(source_frames.resolve_frame(first), pixels)


def test_comparison_without_alignment_keeps_original_descriptors(tmp_path, monkeypatch):
    from core import compare_worker
    first, _ = _source(tmp_path, '比较1.png', 31, np.uint8)
    last, _ = _source(tmp_path, '比较2.png', 43001)

    def forbidden(*args, **kwargs):
        raise AssertionError('Comparison eagerly materialized sources without alignment')

    monkeypatch.setattr(compare_worker, 'materialize_frames', forbidden)
    frames = compare_worker.align_frames([first, last], 'none', {}, lambda: False)
    assert frames == [first, last]


def test_roi_metadata_does_not_decode_source_pixels(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QRect
    from core import image_loader, workers
    first, pixels = _source(tmp_path, 'ROI原图.png', 7101)

    def forbidden(*args, **kwargs):
        raise AssertionError('ROI preparation decoded source pixels')

    decode = image_loader.decode_image
    monkeypatch.setattr(image_loader, 'decode_image', forbidden)
    worker = workers.RenderWorker([first], algorithm='openfocus_a', roi_rect=QRect(3, 2, 10, 8))
    cropped, base, bounds = worker._apply_roi_cropping([first])
    assert base is None and bounds == (3, 2, 10, 8)
    assert cropped[0].source is first and cropped[0].dtype == np.uint16
    monkeypatch.setattr(image_loader, 'decode_image', decode)
    np.testing.assert_array_equal(source_frames.resolve_frame(cropped[0]), pixels[2:10, 3:13])


def _original_source(frame):
    while isinstance(frame, source_frames.ProcessedFrame):
        frame = frame.source
    return frame


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
def test_gui_openfocus_plan_cache_cold_warm_keeps_originals(qapp, tmp_path, monkeypatch, method):
    from core import workers
    from utils import align_cache

    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    first, original = _source(tmp_path, f'{method}-first.png', 19001)
    last, _ = _source(tmp_path, f'{method}-last.png', 19001)
    frames = [first, last]
    request = dict(mode='load', save=True, folder=str(tmp_path),
                   filenames=[f'{method}-first.png', f'{method}-last.png'],
                   identity={'source_revision': 3})

    def forbidden(*args, **kwargs):
        raise AssertionError('Deferred alignment staged or cached a full aligned stack')

    monkeypatch.setattr(workers, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    monkeypatch.setattr(align_cache, 'load_aligned', forbidden)
    monkeypatch.setattr(align_cache, 'save_aligned', forbidden)

    def render():
        worker = workers.RenderWorker(frames, algorithm=method, alignment='openfocus',
            disk_cache=request, tile_enabled=False, thread_count=1)
        published = []
        worker.finished_signal.connect(lambda *args: published.append(args))
        worker._run_admitted()
        assert len(published) == 1
        aligned, result = published[0][:2]
        assert all(isinstance(frame, source_frames.ProcessedFrame) for frame in aligned)
        assert [_original_source(frame) for frame in aligned] == frames
        assert [frame.dtype for frame in aligned] == [np.dtype('uint16')] * 2
        assert result.dtype == np.uint16
        return worker, aligned, result

    cold, cold_frames, cold_result = render()
    assert cold.alignment_computed and not cold.alignment_reused
    assert cold._alignment_plan is not None
    assert not list((tmp_path / align_cache.CACHE_DIR_NAME).rglob('*.png'))
    monkeypatch.setattr(workers.ImageRegistration, 'process', forbidden)
    warm, warm_frames, warm_result = render()
    assert warm.alignment_reused and not warm.alignment_computed
    np.testing.assert_array_equal(warm_result, cold_result)
    np.testing.assert_array_equal(source_frames.resolve_frame(warm_frames[0]), original)
    np.testing.assert_array_equal(source_frames.resolve_frame(cold_frames[0]), original)
    np.testing.assert_array_equal(source_frames.resolve_frame(first), original)


def test_gui_plan_cache_invalidates_reference_and_changed_source(qapp, tmp_path):
    from core import workers

    first, _ = _source(tmp_path, 'cache-first.png', 17001)
    last, _ = _source(tmp_path, 'cache-last.png', 17001)
    request = dict(mode='load', save=True, folder=str(tmp_path),
                   filenames=['cache-first.png', 'cache-last.png'])
    cold = workers.RenderWorker([first, last], algorithm='openfocus_a', alignment='openfocus',
        disk_cache=request, tile_enabled=False, thread_count=1)
    cold._run_admitted()
    different_reference = workers.RenderWorker([first, last], algorithm='openfocus_a',
        alignment='openfocus', registration_parameters={'reference_frame': 2}, disk_cache=request)
    assert different_reference._load_cached_alignment() is None
    different_revision = dict(request, identity={'source_revision': 7})
    revised = workers.RenderWorker([first, last], algorithm='openfocus_a',
        alignment='openfocus', disk_cache=different_revision)
    assert revised._load_cached_alignment() is None
    with open(first.path, 'ab') as stream:
        stream.write(b'changed-source')
    stale = workers.RenderWorker([first, last], algorithm='openfocus_a',
        alignment='openfocus', disk_cache=request)
    assert stale._load_cached_alignment() is None
    with pytest.raises(OSError, match='Source changed'):
        source_frames.resolve_frame(first)


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_cli_openfocus_registration_keeps_descriptors_through_depth(
        tmp_path, monkeypatch, method, dtype):
    from core import cli
    first, original = _source(tmp_path, 'cli-aligned-first.png', 73, dtype)
    last, _ = _source(tmp_path, 'cli-aligned-last.png', 73, dtype)
    args = cli.build_parser().parse_args(['--input', first.path, last.path,
        '--output', str(tmp_path / 'fused.png'), '--method', method,
        '--align', 'openfocus', '--threads', '1', '--cpu', '--no-exposure-match',
        '--no-tile-enabled'])

    def forbidden(*args, **kwargs):
        raise AssertionError('CLI registration materialized all originals')

    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    result = cli._fuse_stack_admitted([first, last], args)
    assert result.dtype == dtype and result.shape == original.shape
    assert all(isinstance(frame, source_frames.ProcessedFrame) for frame in args._depth_frames)
    assert [_original_source(frame) for frame in args._depth_frames] == [first, last]
    np.testing.assert_array_equal(source_frames.resolve_frame(args._depth_frames[0]), original)


def test_comparison_openfocus_registration_reuses_plan_for_roi(tmp_path, monkeypatch):
    from core import compare_worker
    first, original = _source(tmp_path, 'compare-aligned-first.png', 18001)
    last, _ = _source(tmp_path, 'compare-aligned-last.png', 18001)

    def forbidden(*args, **kwargs):
        raise AssertionError('Comparison staged an aligned stack')

    monkeypatch.setattr(compare_worker, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    aligned = compare_worker.align_frames([first, last], 'openfocus',
                                         {'thread_count': 1}, lambda: False)
    assert [_original_source(frame) for frame in aligned] == [first, last]
    cropped = source_frames.crop_frame(aligned[0], 3, 2, 10, 8)
    assert _original_source(cropped) is first
    assert cropped.shape == (8, 10, 3) and cropped.dtype == np.uint16
    np.testing.assert_array_equal(source_frames.resolve_frame(cropped), original[2:10, 3:13])


def test_nested_resolution_charges_full_decode_and_warp_for_roi(tmp_path, monkeypatch):
    from core import resource_policy
    first, original = _source(tmp_path, 'charged.png', 18001)
    warp_bytes = first.nbytes + first.shape[1] * 64
    aligned = source_frames.process_frame(first, lambda pixels: pixels.copy(), scratch_bytes=warp_bytes)
    roi = source_frames.crop_frame(aligned, 3, 2, 10, 8)
    expected = source_frames.frame_resolution_bytes(first) + warp_bytes
    assert source_frames.frame_resolution_bytes(roi) == expected
    admitted = []
    monkeypatch.setattr(resource_policy, 'require_memory', lambda size, **kwargs: admitted.append(size))
    np.testing.assert_array_equal(source_frames.resolve_frame(roi), original[2:10, 3:13])
    assert expected in admitted
    assert expected > roi.nbytes


def test_processed_decode_cancellation_does_not_enter_warp(tmp_path, monkeypatch):
    first, original = _source(tmp_path, 'cancel-decode.png', 17001)
    cancelled = threading.Event()
    warped = []
    def decode(self, *, should_cancel=None):
        cancelled.set()
        return original
    monkeypatch.setattr(source_frames.SourceFrame, 'full_resolution', decode)
    aligned = source_frames.process_frame(first, lambda image: warped.append(image) or image,
                                          should_cancel=cancelled.is_set)
    with pytest.raises(RegistrationCancelled):
        source_frames.resolve_frame(aligned)
    assert warped == []


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
def test_batch_openfocus_registration_exports_without_staging(qapp, tmp_path, monkeypatch, method):
    from core import workers
    folder = tmp_path / 'batch-sources'
    folder.mkdir()
    first, original = _source(folder, 'first.png', 19001)
    last, _ = _source(folder, 'last.png', 19001)
    output = tmp_path / 'batch-output'
    output.mkdir()
    worker = workers.BatchWorker([], 'custom', str(output),
        {'format': 'png', 'fusion_method': method, 'alignment': 'openfocus',
         'save_aligned': True}, thread_count=1, tile_enabled=True, tile_threshold=8, use_gpu=False)

    def forbidden(*args, **kwargs):
        raise AssertionError('Batch staged all aligned frames before fusion')

    monkeypatch.setattr(workers, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    seen = []
    fuse = worker._fuse_one_stack
    def capture(fusion, algorithm, frames, params):
        assert not fusion.tile_enabled
        seen.extend(frames)
        return fuse(fusion, algorithm, frames, params)
    monkeypatch.setattr(worker, '_fuse_one_stack', capture)
    worker._process_single_stack([(first.path, first), (last.path, last)],
                                 str(output), 'aligned-batch', 0)
    assert [_original_source(frame) for frame in seen] == [first, last]
    assert worker._task_failures == [] and len(worker.written_paths) == 1
    assert len(worker.aligned_stacks[0]['files']) == 2
    for path in worker.aligned_stacks[0]['files']:
        decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded, original)


def test_gui_cleanup_openfocus_stays_lazy_and_skips_unserializable_cache(qapp, tmp_path, monkeypatch):
    from core import workers
    from utils import align_cache
    first, original = _source(tmp_path, 'cleanup-first.png', 19001)
    last, _ = _source(tmp_path, 'cleanup-last.png', 19001)

    def forbidden(*args, **kwargs):
        raise AssertionError('Cleanup alignment staged frames or persisted a callable')

    monkeypatch.setattr(workers, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    monkeypatch.setattr(align_cache, 'load_alignment_plan', forbidden)
    monkeypatch.setattr(align_cache, 'save_alignment_plan', forbidden)
    monkeypatch.setattr(align_cache, 'load_aligned', forbidden)
    monkeypatch.setattr(align_cache, 'save_aligned', forbidden)
    worker = workers.RenderWorker([first, last], algorithm='openfocus_a',
        alignment='openfocus', sensor_denoise=True, tile_enabled=False, thread_count=1,
        disk_cache=dict(mode='load', save=True, folder=str(tmp_path),
                        filenames=['cleanup-first.png', 'cleanup-last.png']))
    published = []
    worker.finished_signal.connect(lambda *args: published.append(args))
    worker._run_admitted()
    assert [_original_source(frame) for frame in published[0][0]] == [first, last]
    assert worker._alignment_plan is not None and worker.alignment_computed
    np.testing.assert_array_equal(source_frames.resolve_frame(first), original)


def test_deferred_nonidentity_plan_matches_eager_application_and_owns_sources(tmp_path):
    import weakref
    from core.registration import ImageRegistration, apply_alignment_plan
    y, x = np.indices((96, 128))
    texture = ((x * 1037 + y * 2291 + (x * y % 97) * 239) % 60000).astype(np.uint16)
    original = np.stack((texture, np.roll(texture, 3, axis=0), np.roll(texture, 5, axis=1)), axis=2)
    frames = []
    for index, image in enumerate((original, np.roll(original, 2, axis=1))):
        ok, encoded = cv2.imencode('.png', image)
        assert ok
        path = tmp_path / f'warped-{index}.png'
        encoded.tofile(str(path))
        frames.append(source_frames.scan_source(str(path), header_only=True)[0])
    retained = weakref.ref(frames[1])
    registration = ImageRegistration(method='openfocus', defer_warps=True)
    deferred = registration.process(frames, thread_count=1)
    eager = apply_alignment_plan(frames, registration.alignment_plan, defer_warps=False)
    expected = [np.array(image, copy=True) for image in eager]
    del eager, frames, registration
    gc.collect()
    assert retained() is not None
    for frame, pixels in zip(deferred, expected):
        assert isinstance(frame, source_frames.ProcessedFrame) and frame.dtype == np.uint16
        np.testing.assert_array_equal(source_frames.resolve_frame(frame), pixels)
        np.testing.assert_array_equal(source_frames.resolve_frame(frame), pixels)
    with open(retained().path, 'ab') as stream:
        stream.write(b'changed-after-render')
    with pytest.raises(OSError, match='Source changed'):
        source_frames.resolve_frame(deferred[1])


def test_gui_corrupt_plan_cache_falls_back_to_cold_estimation(qapp, tmp_path):
    from core import workers
    from utils import align_cache
    first, _ = _source(tmp_path, 'corrupt-first.png', 18001)
    last, _ = _source(tmp_path, 'corrupt-last.png', 18001)
    request = dict(mode='load', save=True, folder=str(tmp_path),
                   filenames=['corrupt-first.png', 'corrupt-last.png'])
    worker = workers.RenderWorker([first, last], algorithm='openfocus_a', alignment='openfocus',
        disk_cache=request, tile_enabled=False, thread_count=1)
    worker._run_admitted()
    cached_files = list((tmp_path / align_cache.CACHE_DIR_NAME).rglob('*.json'))
    assert cached_files
    for path in cached_files:
        path.write_text('{invalid-json', encoding='utf-8')
    retry = workers.RenderWorker([first, last], algorithm='openfocus_a', alignment='openfocus',
        disk_cache=request, tile_enabled=False, thread_count=1)
    retry._run_admitted()
    assert retry.alignment_computed and not retry.alignment_reused


def test_comparison_aligned_sources_outlive_deleted_qthread(qapp, tmp_path):
    from PyQt6 import sip
    from core import compare_worker
    first, original = _source(tmp_path, 'lifetime-first.png', 18001)
    last, _ = _source(tmp_path, 'lifetime-last.png', 18001)
    worker = compare_worker.AlignmentWorker([first, last], 'openfocus', {'thread_count': 1})
    published = []
    worker.ready.connect(published.append)
    worker.run()
    assert len(published) == 1
    aligned = published[0]
    sip.delete(worker)
    assert sip.isdeleted(worker)
    for frame in aligned:
        np.testing.assert_array_equal(source_frames.resolve_frame(frame), original)


def test_source_cache_charge_uses_full_decoded_source_under_roi(tmp_path):
    first, _ = _source(tmp_path, 'cache-charge.png', 19001)
    aligned = source_frames.process_frame(first, lambda pixels: pixels, scratch_bytes=64)
    roi = source_frames.crop_frame(aligned, 3, 2, 2, 2)
    assert source_frames.frame_source_cache_bytes(roi) == first.nbytes
    assert source_frames.frame_source_cache_bytes(roi) > roi.nbytes
    array = np.zeros((12, 16, 3), np.uint16)
    processed_array = source_frames.process_frame(array, lambda pixels: pixels.copy())
    assert source_frames.frame_source_cache_bytes(processed_array) == 0


def test_gui_alignment_only_publishes_repeatable_original_descriptors(qapp, tmp_path, monkeypatch):
    from core import workers
    first, original = _source(tmp_path, 'align-only-first.png', 19001)
    last, _ = _source(tmp_path, 'align-only-last.png', 19001)
    def forbidden(*args, **kwargs):
        raise AssertionError('Alignment-only publication staged the source stack')
    monkeypatch.setattr(workers, 'materialize_frames', forbidden)
    monkeypatch.setattr(source_frames, 'materialize_frames', forbidden)
    worker = workers.RenderWorker([first, last], algorithm=None, need_fusion=False,
                                  alignment='openfocus', thread_count=1)
    published = []
    worker.finished_signal.connect(lambda *args: published.append(args))
    worker._run_admitted()
    assert len(published) == 1 and published[0][1] is None
    assert [_original_source(frame) for frame in published[0][0]] == [first, last]
    for frame in published[0][0]:
        np.testing.assert_array_equal(source_frames.resolve_frame(frame), original)


def test_admission_charges_only_new_source_cache_growth_under_roi(tmp_path, monkeypatch):
    first, pixels = _source(tmp_path, 'resident-first.png', 18001)
    last, _ = _source(tmp_path, 'resident-last.png', 19001)
    frames = [source_frames.crop_frame(frame, 1, 1, 2, 2) for frame in (first, last)]
    cache = source_frames.FrameCache(first.nbytes * 2)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', cache)
    assert source_frames.source_cache_growth_bytes(frames) == first.nbytes * 2
    cache.put('first', pixels)
    assert source_frames.source_cache_growth_bytes(frames) == first.nbytes
    cache.put('last', pixels.copy())
    assert source_frames.source_cache_growth_bytes(frames) == 0
    assert source_frames.source_cache_growth_bytes([pixels]) == 0


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_native_resize_admission_charges_full_source_code_conversion(tmp_path, monkeypatch, dtype):
    from core import resource_policy
    from core import multi_focus_fusion
    first, _ = _source(tmp_path, 'resize-admission.png', 73, dtype)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 1024 ** 3)
    requests = []
    def admit(workload, **kwargs):
        requests.append(workload)
        return SimpleNamespace(workers=2)
    monkeypatch.setattr(resource_policy, 'plan_execution', admit)
    assert openfocus_ab._frame_pass_workers([first, first], (2, 2), 'openfocus_a', 2) == 2
    assert len(requests) == 1
    resize_workspace = (first.shape[0] * first.shape[1] + 2 * 2) * 3 * 8
    assert requests[0].per_worker_bytes >= resize_workspace + source_frames.frame_resolution_bytes(first)


def test_producer_admission_does_not_reserve_resident_source_cache_twice(tmp_path, monkeypatch):
    from core import resource_policy
    from core import multi_focus_fusion
    first, pixels = _source(tmp_path, 'producer-resident.png', 19001)
    cache = source_frames.FrameCache(first.nbytes)
    cache.put('already-resident', pixels)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', cache)
    monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 1024 ** 3)
    requests = []
    def admit(workload, **kwargs):
        requests.append(workload)
        return SimpleNamespace(workers=2)
    monkeypatch.setattr(resource_policy, 'plan_execution', admit)
    openfocus_ab._frame_pass_workers([first, first], None, 'openfocus_a', 2)
    assert requests[0].fixed_bytes == 4 * 1024 * 1024 + first.shape[0] * first.shape[1] * 16


def test_producer_window_stays_bounded_as_stack_length_grows(monkeypatch):
    from core import resource_policy
    from core import multi_focus_fusion

    monkeypatch.setattr(resource_policy, 'cpu_capacity', lambda *args: 8)
    monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 1024 ** 4)
    frame = SimpleNamespace(shape=(512, 512, 3), dtype=np.dtype(np.uint8))
    short = openfocus_ab._frame_pass_workers([frame] * 4, None, 'openfocus_a', 8)
    long = openfocus_ab._frame_pass_workers([frame] * 64, None, 'openfocus_a', 8)
    assert short == long == 2


@pytest.mark.parametrize('exposure', [False, True])
def test_large_tiff_b_snapshots_real_ecc_warps_before_whole_depth(tmp_path, monkeypatch, exposure):
    from core import multi_focus_fusion, registration

    rng = np.random.default_rng(301)
    descriptors, eager, deferred = [], [], []
    for index in range(3):
        image = rng.integers(0, 65536, (32, 40, 3), dtype=np.uint16)
        ok, encoded = cv2.imencode('.tif', image)
        assert ok
        path = tmp_path / f'ecc-{index}.tif'
        encoded.tofile(str(path))
        source = source_frames.scan_source(str(path), header_only=True)[0]
        pixels = source_frames.resolve_frame(source)
        identity = registration._perspective_source(source, pixels, None)
        matrix = np.float32([[1, 0, index * .5], [0, 1, -.25 * index], [0, 0, 1]])
        warp = registration.PerspectiveWarp(identity, tuple(matrix.ravel()), matrix.dtype.str,
                                             (40, 32))
        eager.append(warp(pixels))
        deferred.append(source_frames.process_frame(source, warp, scratch_bytes=source.nbytes * 2))
        descriptors.append(source)

    options = dict(algorithm='openfocus_b', tile_enabled=False, sort_mode='off',
                   exposure_match=exposure, use_gpu=False)
    expected = multi_focus_fusion.MultiFocusFusion(**options).fuse(eager, thread_count=1)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 512 * 1024 ** 2)
    original_cost = source_frames.frame_resolution_bytes
    def expensive(frame):
        if isinstance(frame, source_frames.ProcessedFrame) and not isinstance(frame.source, np.ndarray):
            return 129 * 1024 ** 2
        return original_cost(frame)
    monkeypatch.setattr(source_frames, 'frame_resolution_bytes', expensive)
    preparation = []
    def progress(done, total):
        pass
    progress.preparation = lambda done, total: preparation.append((done, total))
    actual = multi_focus_fusion.MultiFocusFusion(**options).fuse(
        deferred, thread_count=1, progress_callback=progress)
    np.testing.assert_array_equal(actual, expected)
    assert preparation == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_required_source_snapshots_fail_before_writing_when_scratch_is_full(tmp_path, monkeypatch):
    source, _ = _source(tmp_path, 'no-scratch.png', 17001)
    error = frame_storage.ScratchSpaceError(str(tmp_path), source.nbytes + 4096, 0)
    def no_space(size, *args, **kwargs):
        assert size == source.nbytes + 4096
        raise error
    monkeypatch.setattr(frame_storage, 'select_scratch', no_space)
    monkeypatch.setattr(source_frames, 'materialize_frames',
                        lambda *args, **kwargs: pytest.fail('snapshot writing started'))
    with pytest.raises(frame_storage.ScratchSpaceError) as caught:
        source_frames.materialize_for_repeated_access([source], required=True, staging_bytes=4096)
    assert caught.value is error


def test_required_source_snapshots_check_source_identity_after_staging(tmp_path):
    source, _ = _source(tmp_path, 'changed-after-snapshot.png', 17001)
    staged = source_frames.materialize_for_repeated_access([source], required=True)
    with open(source.path, 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(OSError, match='Source changed'):
        source_frames.resolve_frame(staged[0])
