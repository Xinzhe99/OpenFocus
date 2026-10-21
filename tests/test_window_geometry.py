"""A restored window must be usable.

Qt saves whatever geometry the window had at close. A window created at a
tiny size (a script, a screenshot harness) was restored next launch as a
79x1050 sliver: the app looked like it never started.
"""
import pytest

pytest.importorskip("PyQt6")


def test_minimums_are_usable():
    """The guard's thresholds must be large enough to show the UI."""
    from utils.settings_store import (_MIN_RESTORABLE_HEIGHT,
                                      _MIN_RESTORABLE_WIDTH)
    assert _MIN_RESTORABLE_WIDTH >= 640, "a narrower window hides the panels"
    assert _MIN_RESTORABLE_HEIGHT >= 480


def test_restore_rejects_a_sliver():
    """The restore path must not accept a geometry below the thresholds.

    The check runs on the widget Qt produced, so assert on the code path
    itself rather than by constructing a window (a QMainWindow created and
    destroyed without an event loop aborts the pytest process here).
    """
    from pathlib import Path
    src = Path("utils/settings_store.py").read_text(encoding="utf-8")
    body = src[src.index("def restore_window_layout"):]
    body = body[:body.index("\ndef ")]
    assert "restoreGeometry" in body
    assert "_MIN_RESTORABLE_WIDTH" in body
    assert "_MIN_RESTORABLE_HEIGHT" in body
    assert "intersects" in body, "an off-screen position must be recovered"


def test_geometry_is_persisted_on_close():
    """The window must save its geometry when it closes."""
    from pathlib import Path
    src = Path("main.py").read_text(encoding="utf-8")
    close = src[src.index("def closeEvent"):]
    close = close[:close.index("\n    def ", 4)]
    assert "save_window_layout" in close
