"""Project files must behave like any editor's document.

Regression: the window title never showed whether the current state was
saved, and closing (or opening another project) discarded the stack without
asking.
"""
import pytest

pytest.importorskip("PyQt6")


class _Host:
    """Minimal window carrying only what the title/dirty helpers touch."""

    def __init__(self, project_path=None):
        self.project_path = project_path
        self._project_dirty = False
        self.titles = []

    def setWindowTitle(self, text):  # noqa: N802 - Qt naming
        self.titles.append(text)


def _bind(host):
    import main
    for name in ("_update_project_title", "_mark_project_dirty",
                 "_clear_project_dirty"):
        setattr(host, name, main.OpenFocus.__dict__[name].__get__(host))
    return host


def test_title_marks_unsaved_changes():
    host = _bind(_Host())
    host._update_project_title()
    assert host.titles[-1] == "OpenFocus"

    host._mark_project_dirty()
    assert host.titles[-1] == "OpenFocus*"

    host.project_path = "C:/work/stack.ofproj"
    host._update_project_title()
    assert host.titles[-1] == "OpenFocus — stack.ofproj*"

    host._clear_project_dirty()
    assert host.titles[-1] == "OpenFocus — stack.ofproj"


def test_marking_twice_only_retitles_once():
    host = _bind(_Host())
    host._mark_project_dirty()
    count = len(host.titles)
    host._mark_project_dirty()
    assert len(host.titles) == count, "an already-dirty project must not retitle"


def test_stack_mutation_marks_the_project_dirty():
    """push_undo is the single funnel every stack mutation goes through."""
    from pathlib import Path
    src = Path("main.py").read_text(encoding="utf-8")
    block = src[src.index("def push_undo"):src.index("def _restore_stack_state")]
    assert "_mark_project_dirty()" in block, (
        "stack mutations must mark the project dirty, otherwise closing "
        "discards the work silently")


def _method_body(src, name):
    """Source of one method, up to the next method at the same indent."""
    start = src.index(f"    def {name}(")
    rest = src[start + 4:]
    nxt = rest.find("\n    def ")
    return src[start:] if nxt < 0 else src[start:start + 4 + nxt]


def test_close_and_open_ask_before_discarding():
    from pathlib import Path
    src = Path("main.py").read_text(encoding="utf-8")
    close = _method_body(src, "closeEvent")
    assert "_confirm_discard_project_changes" in close
    open_fn = src[src.index("def open_project_dialog"):
                   src.index("def save_project_dialog")]
    assert "_confirm_discard_project_changes" in open_fn
    assert "_clear_project_dirty()" in open_fn


def test_discard_prompt_defaults_to_cancel_and_offers_save():
    """The prompt used to default to discarding the unsaved project."""
    from pathlib import Path

    src = Path("main.py").read_text(encoding="utf-8")
    block = src[src.index("def _confirm_discard_project_changes"):
                src.index("def load_demo_stack")]
    assert "setDefaultButton(cancel)" in block, (
        "the safe choice must be the default")
    assert "save_project_dialog" in block, (
        "the user must be able to save before continuing")
    assert "btn_cancel" in block
