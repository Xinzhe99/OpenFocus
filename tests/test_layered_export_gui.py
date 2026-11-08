"""GUI integration for exporting the selected output as a layered TIFF."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QMainWindow

from controllers.export_manager import ExportManager, LayeredTiffWorker
from controllers.output_manager import OutputRecord
from core.render_context import RenderContext, SourceFrame


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _record(source):
    image = np.ones((3, 4, 3), np.uint8)
    context = RenderContext("id", "test", (), 4, 3, image.dtype,
                            (SourceFrame("1", str(source), image),),
                            icc_profile=b"profile")
    return OutputRecord.create(image, "selected", context=context)


def _window(record):
    window = QMainWindow()
    window.output_list = SimpleNamespace(currentItem=lambda: "selected")
    window.output_manager = SimpleNamespace(record_for_item=lambda item: record)
    window.image_source_paths = []
    window.image_loader = None
    return window



@pytest.mark.parametrize("protected", ["standalone", "frame"])
def test_input_paths_cannot_be_overwritten(qapp, monkeypatch, tmp_path, protected):
    source = tmp_path / "source.tif"
    source.write_bytes(b"original")
    record = _record(source if protected == "frame" else tmp_path / "other.tif")
    window = _window(record)
    if protected == "standalone":
        window._standalone_retouch_paths = {str(source)}
    manager = ExportManager(window)
    errors = []
    monkeypatch.setattr("controllers.export_manager.show_error_box",
                        lambda *args: errors.append(args))
    manager.export_layered_tiff(record, record, str(source))
    assert errors and manager.layered_worker is None
    assert source.read_bytes() == b"original"
    window.close()



def test_hardlinked_input_is_protected(qapp, monkeypatch, tmp_path):
    source = tmp_path / "source.tif"
    alias = tmp_path / "alias.tif"
    source.write_bytes(b"original")
    try:
        alias.hardlink_to(source)
    except OSError:
        pytest.skip("hard links unavailable")
    record = _record(source)
    window = _window(record)
    manager = ExportManager(window)
    errors = []
    monkeypatch.setattr("controllers.export_manager.show_error_box",
                        lambda *args: errors.append(args))
    manager.export_layered_tiff(record, record, str(alias))
    assert errors and manager.layered_worker is None
    assert source.read_bytes() == b"original"
    window.close()




def test_success_tracks_captured_output_and_pixels(qapp, monkeypatch, tmp_path):
    from tests.test_batch_export_contract import host

    window = host(qapp)
    record = _record(tmp_path / 'source.tif')
    record.kind = 'retouch'
    item = window.output_manager._insert_output(record.image, record.label, record)
    other = _record(tmp_path / 'other.tif')
    other.kind = 'retouch'
    other_item = window.output_manager._insert_output(other.image, other.label, other)
    manager = window.export_manager
    path = str(tmp_path / 'result.tif')
    monkeypatch.setattr(LayeredTiffWorker, 'start', lambda worker: None)
    manager.export_layered_tiff(record, record, path)
    captured = record.image
    window.output_list.setCurrentItem(other_item)
    record.image = record.image.copy()
    manager._layered_tiff_completed(True, path)
    assert record.exported and record.exported_image is captured
    assert record.export_path == path and record.changed_since_export and record.protected
    assert not other.exported and other.protected
    manager._layered_tiff_finished()
    window.close()


@pytest.mark.parametrize('annotation', ['none', 'label', 'scale', 'baked'])
def test_export_tracks_omitted_annotations(qapp, monkeypatch, tmp_path, annotation):
    from controllers.label_manager import LabelManager
    from tests.test_batch_export_contract import host
    from utils.label_model import Label

    window = host(qapp)
    window.label_manager = LabelManager(window)
    record = _record(tmp_path / 'source.tif')
    record.kind = 'retouch'
    item = window.output_manager._insert_output(record.image, record.label, record)
    if annotation in ('label', 'baked'):
        window.label_manager.labels.add(Label.from_dict({
            'target': 'output', 'output_id': record.output_id,
            'format': 'A', 'enabled': True}))
    if annotation == 'scale':
        record.scale_bar_config = {'px_um': .5, 'position': 'bottom-right'}
        record.scale_bar_initialized = True
    if annotation == 'baked':
        record.annotations_baked = True
    path = str(tmp_path / 'result.tif')
    monkeypatch.setattr(LayeredTiffWorker, 'start', lambda worker: None)
    manager = window.export_manager
    manager.export_layered_tiff(record, record, path)
    manager._layered_tiff_completed(True, path)
    assert record.exported and record.exported_image is record.image
    assert record.exported_annotations == {'labels': [], 'scale': None,
                                           'watermark': None}
    assert record.protected == (annotation in ('label', 'scale'))
    if annotation == 'scale':
        record.scale_bar_config = None
        window.output_manager.refresh_annotation_export_state()
        assert not record.protected
    manager._layered_tiff_finished()
    window.close()


@pytest.mark.parametrize('message', ['', 'write failed'])
def test_cancel_or_failure_does_not_mark_output_exported(qapp, monkeypatch, tmp_path, message):
    from tests.test_batch_export_contract import host

    window = host(qapp)
    record = _record(tmp_path / 'source.tif')
    record.kind = 'retouch'
    item = window.output_manager._insert_output(record.image, record.label, record)
    monkeypatch.setattr('controllers.export_manager.show_error_box', lambda *args: None)
    monkeypatch.setattr(LayeredTiffWorker, 'start', lambda worker: None)
    manager = window.export_manager
    manager.export_layered_tiff(record, record, str(tmp_path / 'result.tif'))
    manager._layered_tiff_completed(False, message)
    assert not record.exported and record.protected and not record.export_path
    manager._layered_tiff_finished()
    window.close()


def _wait_for_export(qapp, get_worker):
    import time
    from PyQt6.QtTest import QTest

    deadline = time.monotonic() + 20
    while get_worker() is not None and time.monotonic() < deadline:
        QTest.qWait(20)
    assert get_worker() is None, 'export did not finish'


@pytest.mark.parametrize('baked', [False, True])
def test_save_as_explains_unavailable_dng_without_overwriting(qapp, monkeypatch, tmp_path, baked):
    from locales import trans
    from tests.test_batch_export_contract import host
    from utils.linear_dng import LinearDNGMetadata

    window = host(qapp)
    image = np.full((16, 16, 3), 5000, np.uint16)
    record = OutputRecord.create(image, 'result')
    if baked:
        record.linear_dng_metadata = LinearDNGMetadata(
            ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1., 1., 1.))
        record.annotations_baked = True
    item = window.output_manager._insert_output(image, record.label, record)
    target = tmp_path / 'existing.dng'
    target.write_bytes(b'previous file')
    offered, warnings = [], []

    def select_dng(_parent, _title, _default, filters):
        offered.extend(filters.split(';;'))
        return str(target), trans.t('linear_dng_filter')

    monkeypatch.setattr('controllers.output_manager.FileDialog.getSaveFileName', select_dng)
    monkeypatch.setattr('controllers.output_manager.show_warning_box', lambda *args: warnings.append(args))
    window.output_manager.save_output_image_as(item)
    assert trans.t('linear_dng_filter') in offered
    assert warnings and target.read_bytes() == b'previous file'
    assert not record.exported and window.output_manager.linear_dng_worker is None
    window.close()


def test_save_as_linear_dng_writes_original_camera_pixels(qapp, monkeypatch, tmp_path):
    import tifffile
    from locales import trans
    from tests.test_batch_export_contract import host
    from utils.linear_dng import LinearDNGMetadata

    window = host(qapp)
    image = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3)
    record = OutputRecord.create(image, 'linear')
    record.linear_dng_metadata = LinearDNGMetadata(
        ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1., 1., 1.), 'Maker', 'Camera')
    item = window.output_manager._insert_output(image, record.label, record)
    target = tmp_path / 'linear.dng'
    errors = []
    monkeypatch.setattr('controllers.output_manager.FileDialog.getSaveFileName',
                        lambda *args: (str(target), trans.t('linear_dng_filter')))
    monkeypatch.setattr('controllers.output_manager.show_error_box', lambda *args: errors.append(args))
    window.output_manager.save_output_image_as(item)
    _wait_for_export(qapp, lambda: window.output_manager.linear_dng_worker)
    assert not errors and record.exported
    with tifffile.TiffFile(target) as tif:
        assert tif.pages[0].photometric.value == 34892
        np.testing.assert_array_equal(tif.asarray(), image[..., ::-1])
    window.close()


def test_save_as_layered_tiff_keeps_source_layers(qapp, monkeypatch, tmp_path):
    from locales import trans
    from psdtags import TiffImageSourceData
    from tests.test_batch_export_contract import host

    window = host(qapp)
    record = _record(tmp_path / 'source.tif')
    record.context.icc_profile = b''
    record.kind = 'retouch'
    item = window.output_manager._insert_output(record.image, record.label, record)
    target = tmp_path / 'layered.tif'
    errors = []
    monkeypatch.setattr('controllers.output_manager.FileDialog.getSaveFileName',
                        lambda *args: (str(target), trans.t('layered_tiff_filter')))
    monkeypatch.setattr('controllers.export_manager.show_error_box', lambda *args: errors.append(args))
    window.output_manager.save_output_image_as(item)
    _wait_for_export(qapp, lambda: window.export_manager.layered_worker)
    assert not errors and record.exported
    layers = TiffImageSourceData.fromtiff(target).layers.layers
    assert len(layers) == 2
    assert not int(layers[0].flags) & 2 and int(layers[1].flags) & 2
    window.close()


def test_save_as_layered_tiff_without_source_reports_error(qapp, monkeypatch, tmp_path):
    from locales import trans
    from tests.test_batch_export_contract import host

    window = host(qapp)
    image = np.full((16, 16, 3), 80, np.uint8)
    record = OutputRecord.create(image, 'result')
    item = window.output_manager._insert_output(image, record.label, record)
    target = tmp_path / 'layered.tif'
    warnings = []
    monkeypatch.setattr('controllers.output_manager.FileDialog.getSaveFileName',
                        lambda *args: (str(target), trans.t('layered_tiff_filter')))
    monkeypatch.setattr('controllers.export_manager.show_warning_box', lambda *args: warnings.append(args))
    window.output_manager.save_output_image_as(item)
    assert warnings and not target.exists() and not record.exported
    window.close()
