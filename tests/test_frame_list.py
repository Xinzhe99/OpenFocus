"""Regression test for the v1.37 hotfix: update_file_list crashed with
NameError: name 'Qt' is not defined as soon as a stack was loaded (the new
decorate_source_items uses Qt check states but the module never imported Qt).
Loading any folder — including the demo stack — hit it immediately."""
import os

import numpy as np
import pytest

QListWidgetItem = pytest.importorskip("PyQt6.QtWidgets").QListWidgetItem
Qt = pytest.importorskip("PyQt6.QtCore").Qt


def test_update_file_list_is_checkable_and_does_not_crash(qapp, tmp_path):
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QLabel
    from controllers.source_manager import SourceManager
    from tests.test_gui_units import _source_window

    names = ["a.png", "b.png", "c.png"]
    thumbs = []
    for _ in names:
        pm = QPixmap(8, 8)
        pm.fill()
        thumbs.append(pm)
    w = _source_window(names)
    w.source_images_label = QLabel()
    w.raw_images = [np.zeros((8, 8, 3), np.uint8) for _ in names]
    sm = SourceManager(w)

    # this is the call every load path ends with — it used to raise NameError
    sm.update_file_list(names, thumbs)

    assert w.file_list.count() == 3
    for row, item in enumerate([w.file_list.item(i) for i in range(3)]):
        assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert item.checkState() == Qt.CheckState.Checked
        assert item.data(Qt.ItemDataRole.UserRole) == row

    # unticking the middle frame (via the bound itemChanged handler) updates
    # the exclusion set
    w.file_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    sm.on_item_changed(w.file_list.item(1))
    assert w.excluded_frames == {1}
    # excluded frames stay excluded across a list rebuild (delete/reload)
    sm.update_file_list(names, thumbs)
    assert w.file_list.item(1).checkState() == Qt.CheckState.Unchecked
    # and a drag-drop order change maps back through the position tags:
    # simulate the widget-side drop by moving the last frame to the front
    moved = w.file_list.takeItem(2)
    w.file_list.insertItem(0, moved)  # rows now c,a,b -> UserRole order 2,0,1
    sm.apply_list_order()
    assert [os.path.basename(n) for n in w.image_filenames] == ["c.png", "a.png", "b.png"]
    assert w.excluded_frames == {2}  # old row 1 (b) moved to row 2 — still excluded
