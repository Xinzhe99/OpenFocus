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
    monkeypatch.setattr("controllers.export_manager.QFileDialog.getSaveFileName",
                        lambda *args: (str(source), ""))
    errors = []
    monkeypatch.setattr("controllers.export_manager.show_error_box",
                        lambda *args: errors.append(args))
    manager.export_layered_tiff()
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
    window = _window(_record(source))
    manager = ExportManager(window)
    monkeypatch.setattr("controllers.export_manager.QFileDialog.getSaveFileName",
                        lambda *args: (str(alias), ""))
    errors = []
    monkeypatch.setattr("controllers.export_manager.show_error_box",
                        lambda *args: errors.append(args))
    manager.export_layered_tiff()
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
    monkeypatch.setattr('controllers.export_manager.QFileDialog.getSaveFileName',
                        lambda *args: (path, ''))
    monkeypatch.setattr(LayeredTiffWorker, 'start', lambda worker: None)
    manager.export_layered_tiff(item)
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
    monkeypatch.setattr('controllers.export_manager.QFileDialog.getSaveFileName',
                        lambda *args: (path, ''))
    monkeypatch.setattr(LayeredTiffWorker, 'start', lambda worker: None)
    manager = window.export_manager
    manager.export_layered_tiff(item)
    manager._layered_tiff_completed(True, path)
    assert record.exported and record.exported_image is record.image
    assert record.exported_annotations == {'labels': [], 'scale': None}
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
    monkeypatch.setattr('controllers.export_manager.QFileDialog.getSaveFileName',
                        lambda *args: (str(tmp_path / 'result.tif'), ''))
    monkeypatch.setattr('controllers.export_manager.show_error_box', lambda *args: None)
    monkeypatch.setattr(LayeredTiffWorker, 'start', lambda worker: None)
    manager = window.export_manager
    manager.export_layered_tiff(item)
    manager._layered_tiff_completed(False, message)
    assert not record.exported and record.protected and not record.export_path
    manager._layered_tiff_finished()
    window.close()
