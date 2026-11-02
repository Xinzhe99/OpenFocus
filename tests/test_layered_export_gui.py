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


def test_selected_record_is_passed_to_background_worker(qapp, monkeypatch, tmp_path):
    record = _record(tmp_path / "source.tif")
    window = _window(record)
    manager = ExportManager(window)
    monkeypatch.setattr("controllers.export_manager.QFileDialog.getSaveFileName",
                        lambda *args: (str(tmp_path / "result"), ""))
    started = []
    monkeypatch.setattr(LayeredTiffWorker, "start", lambda worker: started.append(worker))
    manager.export_layered_tiff()
    worker = started[0]
    assert worker.path == str(tmp_path / "result.tif")
    assert worker.image is record.image
    assert worker.context is record.context
    assert worker.icc_profile == b"profile"
    manager._cancel_layered_tiff()
    assert worker._cancelled.is_set()
    manager._layered_tiff_finished()
    window.close()


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


def test_worker_validation_error_is_visible(qapp, monkeypatch):
    window = _window(None)
    manager = ExportManager(window)
    errors = []
    monkeypatch.setattr("controllers.export_manager.show_error_box",
                        lambda *args: errors.append(args))
    manager._layered_tiff_completed(False, "unsafe source geometry")
    from locales import trans
    assert errors and errors[0][2] == (trans.t('msg_save_failed_text')
                                        + '\nunsafe source geometry')
    window.close()


def test_missing_provenance_shows_explanation(qapp, monkeypatch):
    from locales import trans
    window = _window(None)
    manager = ExportManager(window)
    warnings = []
    monkeypatch.setattr("controllers.export_manager.show_warning_box",
                        lambda *args: warnings.append(args))
    manager.export_layered_tiff()
    assert warnings and warnings[0][2] == trans.t('layered_tiff_no_source')
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


def test_worker_reports_validation_error_and_cancellation(qapp, monkeypatch, tmp_path):
    import utils.layered_tiff as layered_tiff

    record = _record(tmp_path / "source.tif")
    worker = LayeredTiffWorker(str(tmp_path / "result.tif"), record.image,
                               record.context, b"")
    messages = []
    worker.completed.connect(lambda success, text: messages.append((success, text)))

    def fail(*args, **kwargs):
        raise ValueError("unsafe source geometry")

    monkeypatch.setattr(layered_tiff, "export_layered_tiff", fail)
    worker.run()
    assert messages.pop() == (False, "unsafe source geometry")

    def cancelled(*args, **kwargs):
        assert kwargs["cancelled"]()
        raise ValueError("Layered export cancelled")

    monkeypatch.setattr(layered_tiff, "export_layered_tiff", cancelled)
    worker.cancel()
    worker.run()
    assert messages.pop() == (False, "")
