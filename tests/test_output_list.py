"""Drag exports retain the color profile of each selected output."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageCms

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView

from controllers.output_manager import OutputRecord
from widgets.output_list import OutputListWidget


@pytest.mark.parametrize("extension", [".jpg", ".png"])
def test_drag_selected_outputs_use_their_own_profiles(qapp, tmp_path, monkeypatch, extension):
    from utils import scalebar
    import widgets.output_list as output_list

    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    other_profile = profile[:-1] + bytes([profile[-1] ^ 1])
    image = np.full((12, 15, 3), 120, np.uint8)
    baked_path = tmp_path / "baked.png"
    Image.fromarray(image).save(baked_path, icc_profile=other_profile)

    context_record = OutputRecord.create(image, "context", context=SimpleNamespace(icc_profile=profile))
    context_record.icc_profile = other_profile
    record_profile = OutputRecord.create(image, "record")
    record_profile.icc_profile = other_profile
    untagged = OutputRecord.create(image, "untagged", context=SimpleNamespace(icc_profile=b""))
    untagged.icc_profile = profile
    baked = OutputRecord.create(image, "baked", context=SimpleNamespace(icc_profile=profile))
    baked.annotations_baked = True
    baked.export_path = str(baked_path)
    records = [context_record, record_profile, untagged, baked, None]
    labels = SimpleNamespace(prepare_bgr_image=lambda _target, pixels, _index, **kwargs: pixels)
    widget = OutputListWidget()
    widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    for name in ("context", "record", "untagged", "baked", "unknown"):
        widget.addItem(name)
    manager = SimpleNamespace(
        record_for_item=lambda item: records[widget.row(item)],
        record_for_row=lambda row: records[row],
    )
    window = SimpleNamespace(
        output_manager=manager, fusion_results=[image] * len(records), fusion_result=image,
        image_loader=SimpleNamespace(source_icc_profile=other_profile),
        label_manager=SimpleNamespace(snapshot=lambda: labels), drag_export_format=extension,
    )
    widget.set_window(window)
    monkeypatch.setattr(scalebar, "export_cfg", lambda *args, **kwargs: None)
    monkeypatch.setattr(scalebar, "burn", lambda pixels, cfg: pixels)
    paths = []

    class Drag:
        def __init__(self, _widget):
            pass

        def setMimeData(self, mime):
            paths.extend(url.toLocalFile() for url in mime.urls())

        def exec(self, _actions):
            pass

    monkeypatch.setattr(output_list, "QDrag", Drag)
    for row in range(widget.count()):
        widget.item(row).setSelected(True)
    try:
        widget.startDrag(Qt.DropAction.CopyAction)
        assert len(paths) == len(records)
        for path, expected in zip(paths, [profile, other_profile, None, other_profile, None]):
            with Image.open(path) as saved:
                assert saved.info.get("icc_profile") == expected
    finally:
        widget.cleanup_temp_files()


def test_drag_native_raw_exports_the_developed_colors(qapp, tmp_path, monkeypatch):
    import cv2
    import rawpy
    from utils import scalebar
    from utils.linear_dng import LinearDNGMetadata, write_linear_dng

    image = np.full((128, 128, 3), [7000, 11000, 14000], np.uint16)
    metadata = LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1.0, 1024 / 1667), 'Canon', 'Canon EOS R6')
    source = tmp_path / 'reference.dng'
    write_linear_dng(source, image, metadata)
    with rawpy.imread(str(source)) as raw:
        expected = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)

    record = OutputRecord.create(image, 'linear')
    record.linear_dng_metadata = metadata
    labels = SimpleNamespace(prepare_bgr_image=lambda _target, pixels, _index, **kw: pixels)
    widget = OutputListWidget()
    widget.set_window(SimpleNamespace(
        fusion_results=[image], fusion_result=image,
        output_manager=SimpleNamespace(record_for_row=lambda _row: record),
        label_manager=SimpleNamespace(snapshot=lambda: labels), drag_export_format='.png'))
    monkeypatch.setattr(scalebar, 'export_cfg', lambda *args, **kwargs: None)
    path = widget._write_temp_jpg(widget._get_output_image_for_row(0), 'linear', record)
    try:
        saved = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert saved.dtype == np.uint16
        np.testing.assert_allclose(saved[:, :, ::-1], expected, atol=32)
    finally:
        import os
        os.remove(path)
