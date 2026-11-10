from types import SimpleNamespace

import numpy as np
import pytest

from core.image_loader import ImageStackLoader
from controllers.output_manager import OutputManager, OutputRecord
from utils.linear_dng import LinearDNGMetadata


def _metadata(model="Camera", neutral=(0.5, 1.0, 0.7)):
    return LinearDNGMetadata(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                              (0.0, 0.0, 1.0)), neutral, "Maker", model)


def test_linear_stack_load_is_atomic_and_checks_calibration(monkeypatch):
    from core import source_frames
    import utils.linear_dng as linear

    calls = []
    def inspect(path, scale, *, linear, header_only, should_cancel=None):
        assert linear and header_only
        assert callable(should_cancel) and not should_cancel()
        calls.append(path)
        return [SimpleNamespace(linear_metadata=_metadata(neutral=(0.5 + (len(calls) - 1) * .1, 1.0, 0.7)))]
    monkeypatch.setattr(source_frames, 'scan_source', inspect)
    monkeypatch.setattr(linear, 'load_linear_raw',
                        lambda *_args, **_kwargs: pytest.fail('manifest load decoded original RAW'))
    loader = ImageStackLoader()
    old = np.zeros((1, 1, 3), np.uint16)
    previous_profile = _metadata(model='Previous')
    loader.images = [old]
    loader.image_paths = ['previous.dng']
    loader.linear_dng_metadata = previous_profile
    ok, _, images, _ = loader.load_linear_raw_stack(['a.cr3', 'b.cr3'], .25, thread_count=1)
    assert not ok and images == []
    assert loader.images[0] is old
    assert loader.image_paths == ['previous.dng']
    assert loader.linear_dng_metadata is previous_profile
    assert calls == ['a.cr3', 'b.cr3']


def test_linear_stack_cancel_does_not_replace_previous_stack(monkeypatch):
    from core import source_frames
    loader = ImageStackLoader()
    old = np.zeros((1, 1, 3), np.uint16)
    loader.images = [old]
    cancelled, calls = [False], []
    loader.should_cancel = lambda: cancelled[0]
    def inspect(path, scale, *, linear, header_only, should_cancel=None):
        assert linear and header_only
        assert should_cancel is loader.should_cancel and not should_cancel()
        calls.append(path)
        cancelled[0] = True
        return [SimpleNamespace(linear_metadata=_metadata())]
    monkeypatch.setattr(source_frames, 'scan_source', inspect)
    ok, message, frames, _ = loader.load_linear_raw_stack(['a.cr3', 'b.cr3'], thread_count=1)
    assert not ok and message == 'CANCELLED' and frames == []
    assert loader.images[0] is old and calls == ['a.cr3']


@pytest.mark.parametrize('mismatch', [False, True])
def test_linear_manifest_preserves_input_order_when_scans_finish_out_of_order(monkeypatch, mismatch):
    import threading
    from core import resource_policy, source_frames
    from locales import trans
    second_finished = threading.Event()
    calls = []
    def plan(workload, *, preference, user_threads):
        assert user_threads == 2 and preference == 'responsive'
        assert workload.per_worker_bytes == 64 * 1024 * 1024
        return SimpleNamespace(workers=2)
    def inspect(path, scale, *, linear, header_only, should_cancel=None):
        if path == 'a.cr3':
            assert second_finished.wait(5)
            calls.append(path)
        else:
            calls.append(path)
            second_finished.set()
        model = 'Different' if mismatch and path == 'b.cr3' else 'Camera'
        return [SimpleNamespace(path=path, linear_metadata=_metadata(model=model))]
    monkeypatch.setattr(resource_policy, 'plan_execution', plan)
    monkeypatch.setattr(source_frames, 'scan_source', inspect)
    monkeypatch.setattr(source_frames, 'ensure_display_preview', lambda *_a, **_kw: None)
    loader = ImageStackLoader()
    old = object()
    loader.images = [old]
    ok, message, frames, names = loader.load_linear_raw_stack(
        ['a.cr3', 'b.cr3'], thread_count=2, resource_preference='responsive')
    assert calls == ['b.cr3', 'a.cr3']
    if mismatch:
        assert not ok and trans.t('linear_raw_mismatch') in message and 'b.cr3' in message
        assert not frames and loader.images == [old]
    else:
        assert ok and [frame.path for frame in frames] == ['a.cr3', 'b.cr3']
        assert names == ['a.cr3', 'b.cr3']


def test_linear_manifest_cancel_does_not_schedule_remaining_or_commit(monkeypatch):
    import threading
    from core import resource_policy, source_frames
    second_started = threading.Event()
    cancelled = threading.Event()
    calls = []
    def inspect(path, scale, *, linear, header_only, should_cancel=None):
        calls.append(path)
        if path == 'a.cr3':
            assert second_started.wait(5)
            cancelled.set()
        else:
            second_started.set()
        return [SimpleNamespace(linear_metadata=_metadata())]
    monkeypatch.setattr(resource_policy, 'plan_execution',
                        lambda *_a, **_kw: SimpleNamespace(workers=2))
    monkeypatch.setattr(source_frames, 'scan_source', inspect)
    monkeypatch.setattr(source_frames, 'ensure_display_preview',
                        lambda *_a, **_kw: pytest.fail('cancelled manifest prepared preview'))
    loader = ImageStackLoader()
    old = object()
    loader.images = [old]
    loader.should_cancel = cancelled.is_set
    ok, message, frames, _ = loader.load_linear_raw_stack(
        ['a.cr3', 'b.cr3', 'c.cr3'], thread_count=2)
    assert not ok and message == 'CANCELLED' and not frames
    assert set(calls) == {'a.cr3', 'b.cr3'} and loader.images == [old]



def test_unknown_camera_does_not_establish_stack_identity(monkeypatch):
    from core import source_frames
    generic = LinearDNGMetadata(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                                (0.5, 1., 0.7))
    monkeypatch.setattr(source_frames, 'scan_source', lambda *args, **kw:
                        [SimpleNamespace(linear_metadata=generic)])
    loader = ImageStackLoader()
    ok, _, _, _ = loader.load_linear_raw_stack(["one.cr3", "two.cr3"])
    assert not ok


def test_retouch_uses_linear_source_and_keeps_render_provenance():
    from core.render_context import build_render_context
    from utils.retouch import RetouchDocument
    frames = [np.full((16, 16, 3), value, np.uint16) for value in (9000, 31000)]
    fused = np.full((16, 16, 3), 2000, np.uint16)
    metadata = _metadata()
    context = build_render_context(frames, fused, filenames=["b.cr3", "d.cr3"],
                                   source_indices=[1, 3], linear_dng_metadata=metadata,
                                   source_paths=["b.cr3", "d.cr3"])
    doc = RetouchDocument(fused, context)
    doc.frame_index = 1
    doc.diameter = 6
    doc.hardness = 1.0
    assert doc.begin_stroke(8, 8)
    doc.end_stroke()
    assert np.all(doc.working[8, 8] == 31000)
    assert doc.working.dtype == np.uint16
    assert context.source_paths == ("b.cr3", "d.cr3")
    assert context.linear_dng_metadata == metadata
    assert doc.undo()
    np.testing.assert_array_equal(doc.working, fused)




def test_annotated_dng_develops_to_display_color(monkeypatch, tmp_path):
    import tifffile
    import rawpy
    from utils.linear_dng import develop_linear
    from controllers import output_manager as output

    image = np.full((128, 128, 3), 8000, np.uint16)
    image[100, 100] = (65000, 64000, 63000)
    metadata = _metadata(neutral=(0.5, 1.0, 0.7))
    record = OutputRecord.create(image, "result", context=SimpleNamespace(
        linear_dng_metadata=metadata, source_paths=()))
    window = SimpleNamespace(export_manager=SimpleNamespace(_safe_export_path=lambda *args: None))
    manager = OutputManager(window)
    monkeypatch.setattr(manager, "_annotation_state", lambda *args: {"labels": True, "scale": None})
    monkeypatch.setattr(output.scalebar, "burn", lambda img, cfg: img)
    def annotate(target, img, index, **kwargs):
        result = img.copy()
        result[32:64, 32:64] = (12000, 20000, 26000)
        return result
    labels = SimpleNamespace(prepare_bgr_image=annotate)
    path = tmp_path / "annotated.dng"
    manager._write_record(record, str(path), labels)
    raw = tifffile.imread(path)[:, :, ::-1].copy()
    rendered = develop_linear(raw, metadata)
    np.testing.assert_allclose(rendered[40, 40], (12000, 20000, 26000), atol=5)
    np.testing.assert_array_equal(raw[100, 100], image[100, 100])
    untouched = np.ones(image.shape[:2], dtype=bool)
    untouched[32:64, 32:64] = False
    np.testing.assert_array_equal(raw[untouched], image[untouched])
    with rawpy.imread(str(path)) as consumer:
        decoded = consumer.postprocess(use_camera_wb=True, no_auto_bright=True,
                                       output_bps=16, gamma=(2.4, 12.92))
    np.testing.assert_allclose(decoded[40, 40], (26000, 20000, 12000), atol=512)


def test_complex_dng_rejects_annotations_without_modifying_output(tmp_path, monkeypatch):
    from dataclasses import replace
    from locales import trans

    image = np.full((8, 8, 3), 5000, np.uint16)
    metadata = replace(_metadata(), profile_tags=((50936, 's', 7, 'profile'),))
    record = OutputRecord.create(image, 'result', context=SimpleNamespace(
        linear_dng_metadata=metadata, source_paths=()))
    manager = OutputManager(SimpleNamespace(export_manager=SimpleNamespace(
        _safe_export_path=lambda *args: None)))
    monkeypatch.setattr(manager, '_annotation_state',
                        lambda *args: {'labels': True, 'scale': None})
    target = tmp_path / 'existing.dng'
    target.write_bytes(b'previous')
    with pytest.raises(ValueError, match=trans.t('linear_dng_complex_annotations')):
        manager._write_record(record, str(target), SimpleNamespace())
    assert target.read_bytes() == b'previous'
    np.testing.assert_array_equal(record.image, image)


def test_annotated_dng_rejects_unrepresentable_color(monkeypatch, tmp_path):
    from controllers import output_manager as output
    image = np.full((4, 4, 3), 5000, np.uint16)
    metadata = LinearDNGMetadata(((1., -0.7, 0.), (0., 1., 0.), (0., 0., 1.)),
                                 (0.5, 1., 0.7), 'Maker', 'Camera')
    record = OutputRecord.create(image, 'result', context=SimpleNamespace(
        linear_dng_metadata=metadata, source_paths=()))
    manager = OutputManager(SimpleNamespace(export_manager=SimpleNamespace(_safe_export_path=lambda *args: None)))
    monkeypatch.setattr(manager, '_annotation_state', lambda *args: {'labels': True, 'scale': None})
    monkeypatch.setattr(output.scalebar, 'burn', lambda img, cfg: img)
    def annotate(target, img, index, **kwargs):
        result = img.copy()
        result[1, 1] = (0, 65535, 0)
        return result
    target = tmp_path / 'existing.dng'
    target.write_bytes(b'old')
    with pytest.raises(ValueError) as error:
        manager._write_record(record, str(target), SimpleNamespace(prepare_bgr_image=annotate))
    from locales import trans
    assert trans.t('linear_dng_gamut_error') in str(error.value)
    assert target.read_bytes() == b'old'
    np.testing.assert_array_equal(record.image, image)


def test_dng_export_keeps_the_records_result_calibration(tmp_path):
    from core.render_context import build_render_context
    from utils.scalebar import detect_px_size_um

    image = np.full((64, 64, 3), 7000, np.uint16)
    metadata = _metadata()
    context = build_render_context([image], image, pixel_size_um=0.65,
                                   linear_dng_metadata=metadata)
    record = OutputRecord.create(image, 'result', context=context)
    manager = OutputManager(SimpleNamespace(
        export_manager=SimpleNamespace(_safe_export_path=lambda *args: None)))
    labels = SimpleNamespace(
        prepare_bgr_image=lambda target, pixels, index, **kwargs: pixels)
    path = tmp_path / 'calibrated.dng'
    manager._write_record(record, str(path), labels)

    assert detect_px_size_um(str(path)) == pytest.approx(0.65)
    loader = ImageStackLoader()
    ok, message, _images, _names = loader.load_linear_raw_stack([str(path)])
    assert ok, message
    assert loader.px_size_um == pytest.approx(0.65)


def test_worker_resolves_scale_bar_against_developed_pixels(monkeypatch, tmp_path):
    import tifffile
    from controllers import output_manager as output
    from utils.linear_dng import develop_linear
    image = np.full((6, 6, 3), 12000, np.uint16)
    metadata = _metadata()
    record = OutputRecord.create(image, 'result', context=SimpleNamespace(
        linear_dng_metadata=metadata, source_paths=()))
    labels = SimpleNamespace(prepare_bgr_image=lambda target, img, index, **kw: img.copy())
    observed = []
    def burn(img, cfg):
        observed.append(img.copy())
        result = img.copy()
        result[1, 1] = (0, 0, 0) if img.mean() > 32767 else (65535, 65535, 65535)
        return result
    monkeypatch.setattr(output.scalebar, 'burn', burn)
    path = tmp_path / 'worker.dng'
    worker = output.LinearDngExportWorker(str(path), record, labels, {'labels': False, 'scale': {'color': 'auto'}})
    finished = []
    worker.completed.connect(lambda ok, message: finished.append((ok, message)))
    worker.run()
    assert finished and finished[0][0], finished
    expected = develop_linear(image, metadata)
    np.testing.assert_array_equal(observed[0], expected)
    raw = tifffile.imread(path)[:, :, ::-1].copy()
    expected_color = 0 if expected.mean() > 32767 else 65535
    np.testing.assert_allclose(develop_linear(raw, metadata)[1, 1], expected_color, atol=5)
    np.testing.assert_array_equal(raw[2:, 2:], image[2:, 2:])
