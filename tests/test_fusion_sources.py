"""Real source-descriptor fusion preserves pixels without retaining decoded stacks."""
import os
import weakref

import cv2
import numpy as np
import pytest

from core import frame_storage, image_loader, source_frames
from core.multi_focus_fusion import MultiFocusFusion
from core.registration import RegistrationCancelled
from utils.image_utils import apply_exposure_gain, exposure_gains, normalize_fuse_input


CLASSICAL = ['guided_filter', 'gfgfgf', 'dct', 'dtcwt']


def _sources(tmp_path, count=3, mixed=True):
    rng = np.random.default_rng(951)
    arrays, descriptors = [], []
    for index in range(count):
        dtype = np.uint8 if mixed and index % 2 == 0 else np.uint16
        maximum = 256 if dtype == np.uint8 else 65536
        image = rng.integers(0, maximum, (32, 40, 3), dtype=dtype)
        path = tmp_path / f'frame{index:02d}.png'
        ok, encoded = cv2.imencode('.png', image)
        assert ok
        encoded.tofile(str(path))
        arrays.append(image)
        descriptors.extend(source_frames.scan_source(str(path), header_only=True))
    return arrays, descriptors


def _fusion(method, tiled=False, exposure=True):
    if method == 'dtcwt':
        pytest.importorskip('dtcwt')
    parameters = {'N': 1} if method == 'dtcwt' else {'kernel_size': 3}
    if method.startswith('stackmff'):
        pytest.importorskip('onnxruntime')
        parameters = {}
    return MultiFocusFusion(method, use_gpu=False, tile_enabled=tiled,
                            tile_threshold=24, tile_block_size=16, tile_overlap=8,
                            exposure_match=exposure, base_frame='none',
                            fusion_parameters=parameters)


@pytest.mark.parametrize('method', CLASSICAL)
@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.parametrize('exposure', [False, True])
def test_real_descriptor_fusion_matches_array_consumer(tmp_path, method, tiled, exposure):
    arrays, descriptors = _sources(tmp_path)
    expected = _fusion(method, tiled, exposure).fuse(arrays, thread_count=1)
    actual = _fusion(method, tiled, exposure).fuse(descriptors, thread_count=1)
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('method', CLASSICAL)
def test_source_normalization_exposure_and_resize_match_prepared_consumer(tmp_path, method):
    arrays, descriptors = _sources(tmp_path)
    normalized, _ = normalize_fuse_input(arrays)
    gains = exposure_gains(normalized)
    prepared = [apply_exposure_gain(frame, gain) for frame, gain in zip(normalized, gains)]
    resize = None if method == 'dct' else (33, 27)
    expected = _fusion(method, exposure=False).fuse(prepared, img_resize=resize, thread_count=1)
    actual = _fusion(method).fuse(descriptors, img_resize=resize, thread_count=1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape


@pytest.mark.parametrize('method', ['guided_filter', 'gfgfgf', 'dct'])
def test_cropped_descriptors_preserve_roi_coordinates_and_dtype(tmp_path, method):
    arrays, descriptors = _sources(tmp_path)
    descriptors = [source_frames.crop_frame(frame, 5, 3, 27, 25) for frame in descriptors]
    arrays = [frame[3:28, 5:32] for frame in arrays]
    actual = _fusion(method, tiled=True).fuse(descriptors, thread_count=1)
    expected = _fusion(method, tiled=True).fuse(arrays, thread_count=1)
    assert actual.shape == (25, 27, 3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('method', ['guided_filter', 'gfgfgf', 'dct'])
def test_prepared_sources_have_bounded_decoded_and_mapping_lifetimes(tmp_path, monkeypatch, method):
    _arrays, descriptors = _sources(tmp_path, count=8, mixed=False)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    decoded_refs, mapped_refs = [], []
    decode = image_loader.decode_image
    transient = frame_storage.create_transient_frame
    peaks = {'decoded': 0, 'mapped': 0}

    def tracked_decode(*args, **kwargs):
        image = decode(*args, **kwargs)
        decoded_refs.append(weakref.ref(image))
        peaks['decoded'] = max(peaks['decoded'], sum(ref() is not None for ref in decoded_refs))
        return image

    def tracked_mapping(*args, **kwargs):
        image = transient(*args, **kwargs)
        mapped_refs.append(weakref.ref(image))
        peaks['mapped'] = max(peaks['mapped'], sum(ref() is not None for ref in mapped_refs))
        return image

    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    monkeypatch.setattr(frame_storage, 'create_transient_frame', tracked_mapping)
    result = _fusion(method).fuse(descriptors, thread_count=1)
    assert result.shape == (32, 40, 3)
    assert peaks['decoded'] <= 2
    assert peaks['mapped'] <= 4
    assert not any(ref() is not None for ref in decoded_refs)
    assert not any(ref() is not None for ref in mapped_refs)


def test_tile_copies_release_uncacheable_full_sources(tmp_path, monkeypatch):
    arrays, descriptors = _sources(tmp_path, count=7)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    references = []
    decode = image_loader.decode_image

    def tracked_decode(*args, **kwargs):
        assert not any(ref() is not None for ref in references)
        image = decode(*args, **kwargs)
        references.append(weakref.ref(image))
        return image

    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    crops = [source_frames.resolve_region(frame, 7, 5, 13, 11) for frame in descriptors]
    for crop, array in zip(crops, arrays):
        assert crop.flags.owndata
        np.testing.assert_array_equal(crop, array[5:16, 7:20])
    assert not any(ref() is not None for ref in references)


@pytest.mark.parametrize('exposure', [False, True])
def test_fusion_cancel_after_first_decode_stops_before_next_source(tmp_path, monkeypatch, exposure):
    _arrays, descriptors = _sources(tmp_path, count=5)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    decoded = []
    decode = image_loader.decode_image

    def tracked_decode(path, **kwargs):
        image = decode(path, **kwargs)
        decoded.append(path)
        return image

    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    with pytest.raises(RegistrationCancelled):
        _fusion('guided_filter', exposure=exposure).fuse(
            descriptors, thread_count=1, should_cancel=lambda: bool(decoded))
    assert len(decoded) == 1


def test_changed_source_is_rejected_between_exposure_and_fusion(tmp_path, monkeypatch):
    _arrays, descriptors = _sources(tmp_path)
    changed = False

    def change_after_statistics():
        nonlocal changed
        # A change after a source has entered the cache must still be detected.
        if source_frames.SOURCE_CACHE.bytes and not changed:
            changed = True
            with open(descriptors[0].path, 'ab') as stream:
                stream.write(b'changed source identity')
        return False

    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(128 * 1024))
    with pytest.raises(OSError, match='Source changed'):
        _fusion('guided_filter').fuse(descriptors, thread_count=1,
                                     should_cancel=change_after_statistics)
    assert changed


@pytest.mark.parametrize('method', ['stackmff', 'stackmffv2', 'stackmffv3'])
@pytest.mark.parametrize('tiled', [False, True])
def test_real_neural_descriptor_fusion_keeps_all_focus_tile_semantics(tmp_path, method, tiled):
    arrays, descriptors = _sources(tmp_path)
    expected = _fusion(method, tiled).fuse(arrays, thread_count=1)
    actual = _fusion(method, tiled).fuse(descriptors, thread_count=1)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_neural_directory_staging_keeps_originals_unresolved(tmp_path, monkeypatch):
    from fusion_methods.neural_common import read_stack
    arrays, _descriptors = _sources(tmp_path)
    decode = image_loader.decode_image
    calls = []

    def tracked_decode(path, **kwargs):
        calls.append(path)
        return decode(path, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    frames = read_stack(str(tmp_path), (19, 17))
    assert not calls
    assert [frame.shape for frame in frames] == [(17, 19, 3)] * len(arrays)
    np.testing.assert_array_equal(source_frames.resolve_frame(frames[0]),
                                  cv2.resize(arrays[0], (19, 17)))
    assert len(calls) == 1


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_neural_directory_grayscale_preserves_native_precision(tmp_path, dtype):
    from fusion_methods.neural_common import read_stack
    from fusion_methods.stackmff_v2_tiled import _Source
    image = np.arange(19 * 23, dtype=dtype).reshape(19, 23)
    for index in range(2):
        ok, encoded = cv2.imencode('.png', image + index)
        assert ok
        encoded.tofile(str(tmp_path / f'{index}.png'))
    frames = read_stack(str(tmp_path), (17, 13))
    assert frames[0].shape == (13, 17)
    assert frames[0].dtype == dtype
    np.testing.assert_array_equal(source_frames.resolve_frame(frames[0]),
                                  cv2.resize(image, (17, 13)))
    # V2 historically decodes directory inputs as BGR, including spatial tiles.
    source = _Source(str(tmp_path), None, None, None)
    np.testing.assert_array_equal(source.get(0), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_neural_grayscale_tiff_pages_keep_orientation_depth_and_identity(tmp_path, dtype):
    import tifffile
    from fusion_methods.neural_common import read_stack
    pages = [np.arange(19 * 23, dtype=dtype).reshape(19, 23),
             np.full((19, 23), 173, dtype=dtype)]
    path = tmp_path / 'pages.tif'
    with tifffile.TiffWriter(str(path)) as writer:
        for page in pages:
            writer.write(page, photometric='minisblack',
                         extratags=[(274, 'H', 1, 6, False)])
    frames = read_stack(str(tmp_path), None)
    assert len(frames) == 2
    for frame, page in zip(frames, pages):
        assert frame.shape == (23, 19)
        assert frame.dtype == dtype
        np.testing.assert_array_equal(source_frames.resolve_frame(frame),
                                      cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE))
    with path.open('ab') as stream:
        stream.write(b'replaced container identity')
    with pytest.raises(OSError, match='Source changed'):
        source_frames.resolve_frame(frames[1])


def test_neural_lazy_directory_respects_explicit_decoder_budget(tmp_path):
    from core.resource_policy import ExecutionPlan, ResourceLimitError
    from fusion_methods.neural_common import read_stack
    _sources(tmp_path)
    with pytest.raises(ResourceLimitError):
        read_stack(str(tmp_path), None, execution_plan=ExecutionPlan(ram_budget=1))


def _write_sources(tmp_path, arrays):
    frames = []
    for index, image in enumerate(arrays):
        path = tmp_path / f'long{index}.png'
        ok, encoded = cv2.imencode('.png', image)
        assert ok
        encoded.tofile(str(path))
        frames.extend(source_frames.scan_source(str(path), header_only=True))
    return frames


def test_gfg_long_uncacheable_sources_resolve_once_per_source_pass(tmp_path, monkeypatch):
    from collections import Counter
    from fusion_methods.gfg_fgf import gfgfgf_impl
    image = np.random.default_rng(801).integers(0, 65536, (1024, 32, 3), dtype=np.uint16)
    arrays = [image, image.copy()]
    descriptors = _write_sources(tmp_path, arrays)
    expected = gfgfgf_impl(arrays, kernel_size=3, thread_count=1)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    counts = Counter()
    decode = image_loader.decode_image

    def tracked_decode(path, **kwargs):
        counts[path] += 1
        return decode(path, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    actual = gfgfgf_impl(descriptors, kernel_size=3, thread_count=1)
    np.testing.assert_array_equal(actual, expected)
    # Channel selection, decisions, weights and source reconstruction are fixed
    # passes; eight reconstruction strips must still read their source only once.
    assert counts == {descriptors[0].path: 4, descriptors[1].path: 1}


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_v3_long_uncacheable_color_sources_decode_once(tmp_path, monkeypatch, mode):
    from collections import Counter
    from fusion_methods.stackmffv3 import _blend_tile
    rng = np.random.default_rng(802)
    arrays = [rng.integers(0, 256, (1024, 17, 3), dtype=np.uint8),
              rng.integers(0, 65536, (1024, 17, 3), dtype=np.uint16)]
    descriptors = _write_sources(tmp_path, arrays)
    normalized, _ = normalize_fuse_input(arrays)
    stacked = np.stack(normalized)
    probabilities = rng.random((2, 1024, 17), dtype=np.float32)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    if mode == 'hard':
        selected = probabilities.argmax(axis=0)
        expected = stacked[selected, np.arange(1024)[:, None], np.arange(17)[None, :]]
    else:
        expected = (probabilities[:, :, :, None] * stacked).sum(axis=0)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    counts, references = Counter(), []
    decode = image_loader.decode_image

    def tracked_decode(path, **kwargs):
        assert not any(ref() is not None for ref in references)
        counts[path] += 1
        image = decode(path, **kwargs)
        references.append(weakref.ref(image))
        return image

    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    actual = _blend_tile(probabilities, descriptors, mode)
    np.testing.assert_array_equal(actual, expected)
    assert counts == {frame.path: 1 for frame in descriptors}
    assert not any(ref() is not None for ref in references)


@pytest.mark.parametrize('method', ['guided_filter', 'gfgfgf'])
def test_resized_descriptor_tiles_prepare_full_sources_once(tmp_path, monkeypatch, method):
    from collections import Counter
    rng = np.random.default_rng(803)
    arrays = [rng.integers(0, 65536, (192, 224, 3), dtype=np.uint16) for _ in range(2)]
    descriptors = _write_sources(tmp_path, arrays)
    normalized, _ = normalize_fuse_input(arrays)
    gains = exposure_gains(normalized)
    resized = [cv2.resize(apply_exposure_gain(image, gain), (288, 256))
               for image, gain in zip(normalized, gains)]

    def fusion(exposure):
        return MultiFocusFusion(method, use_gpu=False, tile_enabled=True,
                                tile_threshold=1, tile_block_size=64, tile_overlap=16,
                                exposure_match=exposure,
                                fusion_parameters={'kernel_size': 3})

    expected = fusion(False).fuse(resized, thread_count=1)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    counts, full_resizes, references = Counter(), [], []
    decode, resize = image_loader.decode_image, cv2.resize

    def tracked_decode(path, **kwargs):
        counts[path] += 1
        image = decode(path, **kwargs)
        references.append(weakref.ref(image))
        return image

    def tracked_resize(image, size, *args, **kwargs):
        if image.shape[:2] == (192, 224) and tuple(size) == (288, 256):
            full_resizes.append(image.shape)
        return resize(image, size, *args, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    monkeypatch.setattr(cv2, 'resize', tracked_resize)
    actual = fusion(True).fuse(descriptors, img_resize=(288, 256), thread_count=1)
    np.testing.assert_array_equal(actual, expected)
    assert len(full_resizes) == len(arrays)
    assert counts == {frame.path: 2 for frame in descriptors}
    assert not any(ref() is not None for ref in references)


def test_tiny_dct_mixed_size_sources_keep_per_frame_padding_geometry(tmp_path):
    from fusion_methods.dct import dct_focus_stack_fusion
    rng = np.random.default_rng(804)
    arrays = [rng.integers(0, 256, shape, dtype=np.uint8)
              for shape in ((3, 5, 3), (6, 7, 3))]
    descriptors = _write_sources(tmp_path, arrays)
    padded = [np.pad(image, ((0, 5), (0, 3), (0, 0)), mode='edge') for image in arrays]
    expected = dct_focus_stack_fusion(padded, kernel_size=3)[:3, :5]
    for inputs in (arrays, descriptors):
        fusion = MultiFocusFusion('dct', tile_enabled=False, exposure_match=False,
                                  fusion_parameters={'kernel_size': 3})
        np.testing.assert_array_equal(fusion.fuse(inputs, thread_count=1), expected)


def test_resized_source_change_at_last_tile_prevents_publication(tmp_path):
    _arrays, descriptors = _sources(tmp_path)
    changed = False

    def change_at_last_tile(done, total):
        nonlocal changed
        if done == total and not changed:
            changed = True
            with open(descriptors[0].path, 'ab') as stream:
                stream.write(b'changed after resize staging')

    fusion = MultiFocusFusion('guided_filter', tile_threshold=1, tile_block_size=16,
                              tile_overlap=4, exposure_match=False,
                              fusion_parameters={'kernel_size': 3})
    with pytest.raises(OSError, match='Source changed'):
        fusion.fuse(descriptors, img_resize=(56, 48), thread_count=1,
                    progress_callback=change_at_last_tile)
    assert changed


def test_direct_v2_long_resized_sources_stage_once_and_keep_identity(tmp_path, monkeypatch):
    from collections import Counter
    from fusion_methods.stackmff_v2_tiled import _Source
    rng = np.random.default_rng(805)
    arrays = [rng.integers(0, 256, (1000, 32, 3), dtype=np.uint8),
              rng.integers(0, 65536, (1000, 32, 3), dtype=np.uint16)]
    descriptors = _write_sources(tmp_path, arrays)
    gains = [.73, 1.14]
    normalized, _ = normalize_fuse_input(arrays)
    expected = [cv2.resize(apply_exposure_gain(image, gain), (40, 1024))
                for image, gain in zip(normalized, gains)]
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    decode, resize = image_loader.decode_image, cv2.resize
    counts, full_resizes = Counter(), []

    def tracked_decode(path, **kwargs):
        counts[path] += 1
        return decode(path, **kwargs)

    def tracked_resize(image, size, *args, **kwargs):
        if image.shape[:2] == (1000, 32) and tuple(size) == (40, 1024):
            full_resizes.append(image.shape)
        return resize(image, size, *args, **kwargs)

    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    monkeypatch.setattr(cv2, 'resize', tracked_resize)
    source = _Source(descriptors, (40, 1024), gains, None)
    for y in (0, 127, 255, 511, 767, 1013):
        for x in (0, 7, 13):
            for index, image in enumerate(expected):
                np.testing.assert_array_equal(source.get(index, (x, y, 13, 11)),
                                              image[y:y + 11, x:x + 13])
    assert counts == {frame.path: 1 for frame in descriptors}
    assert len(full_resizes) == len(arrays)
    with open(descriptors[0].path, 'ab') as stream:
        stream.write(b'changed after V2 staging')
    with pytest.raises(OSError, match='Source changed'):
        source.get(0, (0, 0, 13, 11))


@pytest.mark.parametrize('exposure', [False, True])
def test_tiled_v3_stages_deferred_frames_once_and_preserves_pixels(tmp_path, monkeypatch, exposure):
    from collections import Counter
    from fusion_methods import stackmffv3

    arrays, descriptors = _sources(tmp_path, count=3, mixed=False)
    transformed = [cv2.flip(image, 1) for image in arrays]
    processed = Counter()
    def flip(image, index):
        processed[index] += 1
        return cv2.flip(image, 1)
    deferred = [source_frames.process_frame(frame, lambda image, i=i: flip(image, i))
                for i, frame in enumerate(descriptors)]
    monkeypatch.setattr(stackmffv3, '_probabilities',
                        lambda batch, *args: np.full_like(batch, 1 / batch.shape[1]))
    expected = _fusion('stackmffv3', tiled=True, exposure=exposure).fuse(
        transformed, thread_count=1)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    decodes = Counter()
    decode = image_loader.decode_image
    def tracked_decode(path, **kwargs):
        decodes[path] += 1
        return decode(path, **kwargs)
    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    preparation, tiles = [], []
    def progress(done, total):
        tiles.append((done, total))
    progress.preparation = lambda done, total: preparation.append((done, total))

    actual = _fusion('stackmffv3', tiled=True, exposure=exposure).fuse(
        deferred, thread_count=1, progress_callback=progress)
    np.testing.assert_array_equal(actual, expected)
    passes = 2 if exposure else 1  # Exposure statistics precede the snapshot.
    assert decodes == {frame.path: passes for frame in descriptors}
    assert processed == {index: passes for index in range(len(descriptors))}
    assert preparation == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert tiles[0][0] == 0 and tiles[-1][0] == tiles[-1][1] and tiles[-1][1] > 1


def test_tiled_v3_rechecks_staged_source_after_last_tile(tmp_path, monkeypatch):
    from fusion_methods import stackmffv3

    _arrays, descriptors = _sources(tmp_path, count=2, mixed=False)
    monkeypatch.setattr(stackmffv3, '_probabilities',
                        lambda batch, *args: np.full_like(batch, 1 / batch.shape[1]))
    changed = []
    def progress(done, total):
        if done == total:
            with open(descriptors[0].path, 'ab') as stream:
                stream.write(b'changed')
            changed.append(True)
    with pytest.raises(OSError, match='Source changed'):
        _fusion('stackmffv3', tiled=True, exposure=False).fuse(
            descriptors, thread_count=1, progress_callback=progress)
    assert changed


def test_neural_source_staging_cancellation_stops_before_next_decode(tmp_path, monkeypatch):
    _arrays, descriptors = _sources(tmp_path, count=3, mixed=False)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    decode = image_loader.decode_image
    decoded = []
    def tracked_decode(path, **kwargs):
        decoded.append(path)
        return decode(path, **kwargs)
    monkeypatch.setattr(image_loader, 'decode_image', tracked_decode)
    cancelled = []
    def progress(done, total):
        if done == 1:
            cancelled.append(True)
    with pytest.raises(RegistrationCancelled):
        source_frames.materialize_for_repeated_access(
            descriptors, lambda: bool(cancelled), thread_count=1,
            progress_callback=progress)
    assert decoded == [descriptors[0].path]


def test_neural_source_staging_falls_back_only_before_writing(tmp_path, monkeypatch):
    _arrays, descriptors = _sources(tmp_path, count=2, mixed=False)
    error = frame_storage.ScratchSpaceError(str(tmp_path), 100, 0)
    def no_space(*args, **kwargs):
        raise error
    monkeypatch.setattr(frame_storage, 'select_scratch', no_space)
    monkeypatch.setattr(source_frames, 'materialize_frames',
                        lambda *args, **kwargs: pytest.fail('staging started'))
    assert source_frames.materialize_for_repeated_access(descriptors) is descriptors


def test_neural_source_staging_propagates_write_failure(tmp_path, monkeypatch):
    _arrays, descriptors = _sources(tmp_path, count=2, mixed=False)
    monkeypatch.setattr(frame_storage, 'select_scratch', lambda *args, **kwargs: str(tmp_path))
    def write_failure(*args, **kwargs):
        raise frame_storage.ScratchSpaceError(str(tmp_path), 100, 0)
    monkeypatch.setattr(source_frames, 'materialize_frames', write_failure)
    with pytest.raises(frame_storage.ScratchSpaceError):
        source_frames.materialize_for_repeated_access(descriptors)


def _classic_directory(tmp_path):
    import tifffile

    rng = np.random.default_rng(741)
    for name in ('img1.png', 'img10.PNG', 'cover.png', 'slice2_z1.png',
                 'slice10_z0.png'):
        image = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode('.png', image)
        assert ok
        encoded.tofile(str(tmp_path / name))
    with tifffile.TiffWriter(tmp_path / 'img2.tif') as tif:
        for _ in range(2):
            tif.write(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8),
                      photometric='rgb')
    (tmp_path / 'notes.txt').write_text('not an image')
    (tmp_path / 'sub.png').mkdir()


def _classic_direct(method, source, should_cancel=None):
    if method == 'dct':
        from fusion_methods.dct import dct_focus_stack_fusion
        return dct_focus_stack_fusion(source, kernel_size=3, thread_count=1,
                                      should_cancel=should_cancel)
    if method == 'dtcwt':
        from fusion_methods.dtcwt import _dtcwt_impl
        return _dtcwt_impl(source, None, 1, False, thread_count=1,
                           should_cancel=should_cancel)
    if method == 'guided_filter':
        from fusion_methods.gff import gff_impl
        return gff_impl(source, None, kernel_size=3, thread_count=1,
                        should_cancel=should_cancel)
    from fusion_methods.gfg_fgf import gfgfgf_impl
    return gfgfgf_impl(source, None, kernel_size=3, thread_count=1,
                       should_cancel=should_cancel)


@pytest.mark.parametrize('method', CLASSICAL)
def test_classic_directory_matches_shared_ordered_frames(tmp_path, monkeypatch, method):
    from fusion_methods.openfocus_ab import _collect_directory_sources, _collect_images
    from utils.sorting import natural_sort_key

    if method == 'dtcwt':
        pytest.importorskip('dtcwt')
    _classic_directory(tmp_path)
    frames, paths = _collect_directory_sources(str(tmp_path))
    expected_names = ['img1.png', 'img2.tif', 'img10.PNG', 'slice2_z1.png',
                      'slice10_z0.png', 'cover.png']
    assert [os.path.basename(path) for path in paths] == expected_names
    assert paths == sorted(paths, key=natural_sort_key)
    assert [(frame.path, frame.page) for frame in frames] == [
        (paths[0], 0), (paths[1], 0), (paths[1], 1), (paths[2], 0),
        (paths[3], 0), (paths[4], 0), (paths[5], 0)]
    assert [(f.path, f.page) for f in _collect_images(str(tmp_path), normalize=False)] == [
        (f.path, f.page) for f in frames]

    scanned = []
    scan = source_frames.scan_source
    def record(path, **kwargs):
        scanned.append(path)
        return scan(path, **kwargs)
    monkeypatch.setattr(source_frames, 'scan_source', record)
    actual = _classic_direct(method, str(tmp_path))
    assert scanned == paths
    expected = _classic_direct(method, frames)
    np.testing.assert_array_equal(actual, expected)
    if method == 'dct':
        from fusion_methods.dct import _collect_images_from_folder
        collected, collected_paths = _collect_images_from_folder(str(tmp_path))
        assert collected_paths == paths
        assert [(f.path, f.page) for f in collected] == [(f.path, f.page) for f in frames]


@pytest.mark.parametrize('method', CLASSICAL)
def test_classic_directory_corrupt_image_stops_scan(tmp_path, monkeypatch, method):
    if method == 'dtcwt':
        pytest.importorskip('dtcwt')
    _classic_directory(tmp_path)
    (tmp_path / 'img2.tif').write_bytes(b'broken image')
    scanned = []
    scan = source_frames.scan_source
    def record(path, **kwargs):
        scanned.append(path)
        return scan(path, **kwargs)
    monkeypatch.setattr(source_frames, 'scan_source', record)
    with pytest.raises(Exception):
        _classic_direct(method, str(tmp_path))
    assert [os.path.basename(path) for path in scanned] == ['img1.png', 'img2.tif']


@pytest.mark.parametrize('method', CLASSICAL)
def test_classic_directory_cancellation_stops_tiff_and_recovers(tmp_path, monkeypatch, method):
    import tifffile
    if method == 'dtcwt':
        pytest.importorskip('dtcwt')
    with tifffile.TiffWriter(tmp_path / 'img1.tif') as tif:
        for value in (30, 80):
            tif.write(np.full((32, 40, 3), value, np.uint8), photometric='rgb')
    ok, encoded = cv2.imencode('.png', np.full((32, 40, 3), 120, np.uint8))
    assert ok
    encoded.tofile(str(tmp_path / 'img2.png'))
    scanned = []
    scan = source_frames.scan_source
    def record(path, **kwargs):
        scanned.append(path)
        return scan(path, **kwargs)
    monkeypatch.setattr(source_frames, 'scan_source', record)
    with pytest.raises(RegistrationCancelled):
        _classic_direct(method, str(tmp_path), lambda: True)
    assert scanned == []

    original_frame = source_frames.SourceFrame
    created = []
    def create_frame(*args, **kwargs):
        frame = original_frame(*args, **kwargs)
        created.append(frame)
        return frame
    with monkeypatch.context() as context:
        context.setattr(source_frames, 'SourceFrame', create_frame)
        with pytest.raises(RegistrationCancelled):
            _classic_direct(method, str(tmp_path), lambda: bool(created))
    assert len(created) == 1
    assert scanned == [str(tmp_path / 'img1.tif')]
    from fusion_methods.openfocus_ab import _collect_directory_sources
    frames, _ = _collect_directory_sources(str(tmp_path))
    assert len(frames) == 3
    np.testing.assert_array_equal(_classic_direct(method, str(tmp_path)),
                                  _classic_direct(method, frames))


def test_classic_empty_and_missing_directory_policies(tmp_path):
    from fusion_methods.dct import _collect_images_from_folder
    from fusion_methods.dtcwt import _load_images
    from fusion_methods.gff import gff_impl
    from fusion_methods.gfg_fgf import gfgfgf_impl
    from fusion_methods.openfocus_ab import _collect_images

    directory = str(tmp_path)
    missing = str(tmp_path / 'missing')
    assert _collect_images_from_folder(directory) == ([], [])
    assert _collect_images_from_folder(missing) == ([], [])
    assert _load_images(directory, None) == []
    with pytest.raises(ValueError, match='Input path does not exist'):
        _load_images(missing, None)
    with pytest.raises(ValueError, match='at least 2'):
        _collect_images(directory, normalize=False)
    with pytest.raises(ValueError, match='Input folder is empty'):
        gff_impl(directory, None)
    with pytest.raises(ValueError, match='No images found'):
        gfgfgf_impl(directory)
    with pytest.raises(ValueError, match='Path not found'):
        gfgfgf_impl(missing)
