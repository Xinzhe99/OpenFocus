"""Render clicked while a stack is still loading starts when the load finishes."""


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, text, *_args):
        self.messages.append(text)

    def clearMessage(self):
        pass


class _Window:
    def __init__(self, frames=0):
        self._bar = _StatusBar()
        self.frames = frames

    def statusBar(self):
        return self._bar

    def effective_stack(self):
        return [object()] * self.frames, [], list(range(self.frames))


class _Worker:
    def __init__(self, cancelled=False):
        self.is_cancelled = cancelled

    def deleteLater(self):
        pass


def _setup(qapp, frames=0):
    from controllers.render_manager import RenderManager
    from controllers.source_manager import SourceManager
    window = _Window(frames)
    render = RenderManager(window)
    source = SourceManager.__new__(SourceManager)
    source.window = window
    source._load_worker = _Worker()
    window.source_manager = source
    window.render_manager = render
    return window, render, source


def test_render_during_load_is_queued_and_cancellable(qapp):
    window, render, _source = _setup(qapp)
    assert render.start_render() == "loading"
    assert render._render_after_load
    assert render.start_render() == "loading"          # second click cancels
    assert not render._render_after_load
    # The control channel never queues; it just reports the state.
    assert render.start_render(interactive=False) == "loading"
    assert not render._render_after_load


def test_queued_render_starts_when_frames_arrive(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    window, render, source = _setup(qapp, frames=3)
    render.start_render()
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    source._start_queued_render(source._load_worker)
    assert started == [render.start_render] and not render._render_after_load


def test_queued_render_is_dropped_when_the_load_fails(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    window, render, source = _setup(qapp, frames=0)
    render.start_render()
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    source._start_queued_render(_Worker(cancelled=True))
    assert not started and not render._render_after_load
