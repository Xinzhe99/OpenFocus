"""GUI integration for exporting the selected output as a layered TIFF."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QMainWindow

from controllers.export_manager import ExportManager
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


