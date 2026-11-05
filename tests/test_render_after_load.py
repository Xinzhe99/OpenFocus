"""Render clicked while a stack is still loading starts when the load finishes."""


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, text, *_args):
        self.messages.append(text)

    def clearMessage(self):
        pass


class _Button:
    def setEnabled(self, _enabled):
        pass

    def setText(self, _text):
        pass


class _Window:
    def __init__(self, frames=0):
        self._bar = _StatusBar()
        self.frames = frames
        self.btn_render = _Button()

    def statusBar(self):
        return self._bar

    def effective_stack(self):
        return [object()] * self.frames, [], list(range(self.frames))


class _Worker:
    def __init__(self, cancelled=False, applied=False):
        self.is_cancelled = cancelled
        self._load_applied = applied

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
    assert render._render_after_load is None
    # The control channel never queues; it just reports the state.
    assert render.start_render(interactive=False) == "loading"
    assert render._render_after_load is None


def test_queued_render_starts_when_frames_arrive(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    window, render, source = _setup(qapp, frames=3)
    render.start_render()
    source._load_worker._load_applied = True
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    source._start_queued_render(source._load_worker)
    assert len(started) == 1 and render._render_after_load is None
    called = []
    render.start_render = lambda force_algorithm=None: called.append(force_algorithm)
    started[0]()
    assert called == [None]


def test_queued_render_is_dropped_when_the_load_fails(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    window, render, source = _setup(qapp, frames=0)
    render.start_render()
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    source._start_queued_render(_Worker(cancelled=True, applied=False))
    assert not started and render._render_after_load is None


def test_queued_render_is_dropped_when_failed_load_leaves_old_stack(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    _window, render, source = _setup(qapp, frames=3)
    render.start_render()
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    source._start_queued_render(_Worker(cancelled=False, applied=False))
    assert not started and render._render_after_load is None


def test_compare_request_keeps_method_and_is_cancelled_with_run(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    _window, render, source = _setup(qapp, frames=3)
    render._compare_mode = True
    render.start_render(force_algorithm="dct")
    assert render._render_after_load == {"force_algorithm": "dct", "compare": True}

    source._load_worker._load_applied = True
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    source._start_queued_render(source._load_worker)
    called = []
    render.start_render = lambda force_algorithm=None: called.append(force_algorithm)
    started[0]()
    assert called == ["dct"]

    render._render_after_load = {"force_algorithm": "openfocus_a", "compare": True}
    render._compare_mode = True
    render._compare_queue = ["dct"]
    render.start_compare_all()
    assert render._render_after_load is None


def test_render_click_cancelling_a_queued_compare_request_ends_the_run(qapp):
    _window, render, _source = _setup(qapp, frames=3)
    render._compare_mode = True
    render._compare_queue = ["openfocus_a"]
    render._compare_total = 2
    render.start_render(force_algorithm="dct")
    assert render._render_after_load == {"force_algorithm": "dct", "compare": True}
    render.start_render()                                   # the Render button cancels it
    assert render._render_after_load is None
    assert not render._compare_mode and render._compare_queue == []


def test_deferred_compare_render_does_not_run_after_the_compare_is_cancelled(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    _window, render, source = _setup(qapp, frames=3)
    render._compare_mode = True
    render.start_render(force_algorithm="dct")
    source._load_worker._load_applied = True
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    source._start_queued_render(source._load_worker)
    render._compare_mode = False                            # cancelled before the timer fired
    called = []
    render.start_render = lambda force_algorithm=None: called.append(force_algorithm)
    started[0]()
    assert called == []
