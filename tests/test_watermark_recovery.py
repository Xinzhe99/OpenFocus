"""Watermark removal and undo/redo must update the crash-recovery snapshot.

The regression reads the real recovery project written by
``recovery.write_snapshot`` and replays its stored watermark settings through
the normal restore path, so it observes committed state instead of asserting
how many times a writer was called.
"""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
cv2 = pytest.importorskip("cv2")

from controllers.output_manager import OutputRecord  # noqa: E402
from utils import recovery  # noqa: E402
from utils import watermark as wm  # noqa: E402

SOURCE_TEXT = "© SOURCE"
REGISTERED_TEXT = "© REGISTERED"
OUTPUT_TEXT = "© OUTPUT"
RECORD_TEXT = "© RECORD"


class _Item:
    def __init__(self, record):
        self.record = record

    def data(self, role):
        return self.record


class _List:
    """List-widget subset: one row per record, first row selected."""

    def __init__(self, records):
        self.records = list(records)

    def count(self):
        return len(self.records)

    def currentItem(self):  # noqa: N802
        return _Item(self.records[0]) if self.records else None

    def item(self, row):
        return _Item(self.records[row]) if 0 <= row < len(self.records) else None


class _Manager:
    def __init__(self, records):
        self.records = list(records)

    def record_for_row(self, row):
        return self.records[row] if 0 <= row < len(self.records) else None

    def record_for_item(self, item):
        return item.record if item is not None else None


def _record(label, text):
    record = OutputRecord.create(None, label)
    wm.set_record_config(record, wm.WatermarkConfig(text=text))
    return record


def _window(tmp_path, records=()):
    from tests.test_project_file import _window as base_window
    from main import OpenFocus

    source = tmp_path / "frame.png"
    assert cv2.imwrite(str(source), np.full((8, 8, 3), 127, np.uint8))
    window = base_window([str(source)], str(tmp_path))
    for key, (attr, default) in wm.ATTRS.items():
        setattr(window, attr, default)
    for attr in wm.TARGET_CONFIG_ATTRS.values():
        setattr(window, attr, None)
    for enabled_attr, default in wm.SCOPE_ATTRS.values():
        setattr(window, enabled_attr, default)
    window.output_list = _List(records)
    window.output_manager = _Manager(records)
    window._undo_stack = []
    window._redo_stack = []
    window._undo_guard = False
    window.UNDO_LIMIT = OpenFocus.UNDO_LIMIT
    window.persisted = 0
    window.refreshed = 0
    window.dirty_marks = 0
    window.messages = []
    window._capture_watermark_state = lambda: wm.capture_state(window)
    window._capture_stack_state = lambda: {
        "watermark": wm.capture_state(window), "labels": None, "scalebar": None,
    }
    window.push_watermark_undo = MethodType(OpenFocus.push_watermark_undo, window)
    window._restore_stack_state = MethodType(OpenFocus._restore_stack_state, window)
    window._restore_watermark_state = MethodType(
        OpenFocus._restore_watermark_state, window)
    window.persist_settings = lambda: setattr(window, "persisted", window.persisted + 1)
    window.refresh_annotation_views = lambda: setattr(
        window, "refreshed", window.refreshed + 1)
    window._mark_project_dirty = lambda: setattr(
        window, "dirty_marks", window.dirty_marks + 1)
    window._refresh_undo_actions = lambda: None
    window._show_status_message = lambda text, timeout=0: window.messages.append(text)
    window.statusBar = lambda: SimpleNamespace(  # noqa: N802
        showMessage=lambda text, timeout=0: window.messages.append(text))
    return window


def _isolate_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path / "recovery"))


def _snapshot_watermark():
    state = recovery.load_snapshot_state()
    assert state is not None, "recovery snapshot is missing"
    return state["settings"]["watermark"]


def _seed(window):
    wm.set_target_config(window, "source", wm.WatermarkConfig(text=SOURCE_TEXT))
    wm.set_target_config(
        window, "registered", wm.WatermarkConfig(text=REGISTERED_TEXT))
    wm.set_target_config(window, "output", wm.WatermarkConfig(text=OUTPUT_TEXT))


def _replay(settings):
    """Apply recovered settings to a fresh host, like opening the session."""
    replayed = type("ReplayHost", (), {})()
    for key, (attr, default) in wm.ATTRS.items():
        setattr(replayed, attr, default)
    for attr in wm.TARGET_CONFIG_ATTRS.values():
        setattr(replayed, attr, None)
    for enabled_attr, default in wm.SCOPE_ATTRS.values():
        setattr(replayed, enabled_attr, default)
    wm.apply_settings(replayed, settings)
    return replayed


def test_removal_persists_and_keeps_other_targets(tmp_path, monkeypatch):
    _isolate_recovery(tmp_path, monkeypatch)
    record = _record("out", RECORD_TEXT)
    window = _window(tmp_path, [record])
    _seed(window)

    from main import OpenFocus
    OpenFocus.remove_watermark(window, "source")

    assert wm.target_config(window, "source") is None
    assert wm.target_config(window, "registered").text == REGISTERED_TEXT
    assert wm.target_config(window, "output").text == OUTPUT_TEXT
    assert wm.record_export_cfg(record)["text"] == RECORD_TEXT
    assert len(window._undo_stack) == 1

    settings = _snapshot_watermark()
    assert settings["source"] is None
    assert settings["registered"]["text"] == REGISTERED_TEXT
    assert settings["output"]["text"] == OUTPUT_TEXT
    assert settings["outputs"][record.output_id]["config"]["text"] == RECORD_TEXT

    replayed = _replay(settings)
    assert wm.target_config(replayed, "source") is None
    assert wm.target_config(replayed, "registered").text == REGISTERED_TEXT
    assert wm.target_config(replayed, "output").text == OUTPUT_TEXT


def test_undo_and_redo_refresh_recovery_for_record_binding(tmp_path, monkeypatch):
    _isolate_recovery(tmp_path, monkeypatch)
    record = _record("out", RECORD_TEXT)
    window = _window(tmp_path, [record])
    _seed(window)

    from main import OpenFocus
    OpenFocus.remove_watermark(window, "output")

    assert wm.record_export_cfg(record) is None
    removed = _snapshot_watermark()
    assert removed["outputs"][record.output_id]["config"] is None
    assert removed["source"]["text"] == SOURCE_TEXT
    assert removed["registered"]["text"] == REGISTERED_TEXT

    OpenFocus.undo(window)
    assert wm.record_export_cfg(record)["text"] == RECORD_TEXT
    restored = _snapshot_watermark()
    assert restored["outputs"][record.output_id]["config"]["text"] == RECORD_TEXT
    # Undo must not disturb the scopes the removal never touched.
    assert restored["source"]["text"] == SOURCE_TEXT
    assert restored["registered"]["text"] == REGISTERED_TEXT

    OpenFocus.redo(window)
    assert wm.record_export_cfg(record) is None
    replayed = _snapshot_watermark()
    assert replayed["outputs"][record.output_id]["config"] is None
    assert replayed["source"]["text"] == SOURCE_TEXT
    assert replayed["registered"]["text"] == REGISTERED_TEXT

    reapplied = _replay(replayed)
    assert wm.target_config(reapplied, "source").text == SOURCE_TEXT
    assert wm.target_config(reapplied, "registered").text == REGISTERED_TEXT
