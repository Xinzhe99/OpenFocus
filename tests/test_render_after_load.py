"""Render clicked while a stack is still loading starts when the load finishes."""
import pytest


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

    def setToolTip(self, _text):
        pass


class _Window:
    def __init__(self, frames=0):
        self.experimental_features = True
        self._bar = _StatusBar()
        self.frames = frames
        self.raw_images = [object() for _ in range(frames)]
        self.source_revision = 0
        self.excluded_frames = set()
        self.btn_render = _Button()

    def statusBar(self):
        return self._bar

    def effective_stack(self):
        included = [index for index in range(len(self.raw_images))
                    if index not in self.excluded_frames]
        return [self.raw_images[index] for index in included], [], included


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
    source = SourceManager(window)
    source._load_worker = _Worker()
    window.source_manager = source
    window.render_manager = render
    return window, render, source


def _apply_and_finish_load(source):
    worker = source._load_worker
    worker._load_applied = True
    worker._load_stack = source.window.raw_images
    worker._load_revision = source.window.source_revision
    source._load_worker = None
    source._start_queued_render(worker)


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
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    _apply_and_finish_load(source)
    assert len(started) == 1
    called = []
    render.start_render = lambda force_algorithm=None: called.append(force_algorithm)
    started[0]()
    assert called == [None]
    assert render._render_after_load is None


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
    render._compare_current = "dct"
    render.start_render(force_algorithm="dct")
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    _apply_and_finish_load(source)
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
    render._compare_current = "dct"
    render._compare_total = 2
    render.start_render(force_algorithm="dct")
    render.start_render()                                   # the Render button cancels it
    assert render._render_after_load is None
    assert not render._compare_mode and render._compare_queue == []


def test_deferred_compare_render_does_not_run_after_the_compare_is_cancelled(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    _window, render, source = _setup(qapp, frames=3)
    render._compare_mode = True
    render._compare_current = "dct"
    render.start_render(force_algorithm="dct")
    started = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: started.append(fn))
    _apply_and_finish_load(source)
    render._compare_mode = False                            # cancelled before the timer fired
    called = []
    render.start_render = lambda force_algorithm=None: called.append(force_algorithm)
    started[0]()
    assert called == []


@pytest.mark.parametrize("change", ["cancel", "source", "revision", "selection", "load", "shutdown"])
def test_deferred_render_does_not_outlive_its_loaded_stack(qapp, monkeypatch, change):
    from PyQt6.QtCore import QTimer
    window, render, source = _setup(qapp, frames=3)
    render.start_render()
    deferred = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: deferred.append(fn))
    _apply_and_finish_load(source)
    assert len(deferred) == 1
    if change == "cancel":
        assert render.start_render() == "cancelled"
    elif change == "source":
        window.raw_images = list(window.raw_images)
    elif change == "revision":
        window.source_revision += 1
    elif change == "selection":
        window.excluded_frames.add(0)
    elif change == "load":
        source._load_worker = _Worker()
        source._load_generation = 1
    else:
        window._shutting_down = True
    called = []
    render.start_render = lambda **kwargs: called.append(kwargs)
    deferred[0]()
    assert called == []
    assert render._render_after_load is None


def test_control_poll_does_not_cancel_a_deferred_render(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    _window, render, source = _setup(qapp, frames=3)
    render.start_render()
    deferred = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: deferred.append(fn))
    _apply_and_finish_load(source)
    assert render.start_render(interactive=False) == "loading"
    called = []
    render.start_render = lambda **kwargs: called.append(kwargs)
    deferred[0]()
    assert called == [{"force_algorithm": None}]


def test_deferred_compare_cannot_join_a_restarted_comparison(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    _window, render, source = _setup(qapp, frames=3)
    render._compare_mode = True
    render._compare_current = "dct"
    render.start_render(force_algorithm="dct")
    deferred = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: deferred.append(fn))
    _apply_and_finish_load(source)
    render._compare_token = object()
    called = []
    render.start_render = lambda **kwargs: called.append(kwargs)
    deferred[0]()
    assert called == []
    assert render._compare_mode  # the new run owns this state


def test_replaced_load_cannot_satisfy_an_older_render_request(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    _window, render, source = _setup(qapp, frames=3)
    render.start_render()
    source._load_worker = _Worker()
    deferred = []
    monkeypatch.setattr(QTimer, "singleShot", lambda _ms, fn: deferred.append(fn))
    _apply_and_finish_load(source)
    assert deferred == []
    assert render._render_after_load is None


def test_waiting_caption_survives_refresh_and_cancel_stops_deferred_start(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QPushButton, QStatusBar
    from locales import trans
    window, render, source = _setup(qapp, frames=3)
    window._bar = QStatusBar()
    window.btn_render = QPushButton()
    deferred = []
    monkeypatch.setattr(QTimer, 'singleShot', lambda _ms, callback: deferred.append(callback))
    assert render.start_render() == 'loading'
    assert render.worker is None and render.last_state == 'queued'
    assert render._status_bar_label.text() == trans.t('render_stage_loading')
    assert not render._status_bar_label.isHidden()
    assert window.btn_render.text() == trans.t('btn_cancel_render')
    render.refresh_action_labels()
    assert window.btn_render.text() == trans.t('btn_cancel_render')
    _apply_and_finish_load(source)
    assert len(deferred) == 1
    assert render.start_render() == 'cancelled'
    assert render._status_bar_label.isHidden()
    assert window.btn_render.text() == trans.t('btn_render')
    called = []
    render.start_render = lambda **kwargs: called.append(kwargs)
    deferred[0]()
    assert called == [] and render.last_state == 'cancelled'
    window._bar.deleteLater()
    window.btn_render.deleteLater()


def test_resource_wait_preparation_fusion_and_late_cancel_stages_are_truthful(qapp):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QStatusBar
    from locales import trans
    window, render, _source = _setup(qapp)
    window._bar = QStatusBar()
    render.worker = SimpleNamespace(is_cancelled=False)
    for stage, key in [('waiting', 'render_stage_waiting'),
                       ('prepare', 'render_stage_prepare'),
                       ('align', 'render_stage_align'),
                       ('fuse', 'render_stage_fuse'),
                       ('finish', 'render_progress')]:
        render.on_render_stage(stage)
        assert render._status_bar_label.text() == trans.t(key)
        assert window.statusBar().currentMessage() == trans.t(key)
    render.on_render_stage('prepare')
    render.on_render_progress(1, 2)
    assert window.statusBar().currentMessage().startswith(trans.t('render_stage_prepare'))
    assert render._control_progress == (1, 2)
    render.on_render_stage('fuse')
    assert render._control_progress is None
    render.on_render_progress(1, 3)
    assert window.statusBar().currentMessage().startswith(trans.t('render_stage_fuse'))
    render.on_render_stage('finish')
    assert render._control_progress is None
    assert window.statusBar().currentMessage() == trans.t('render_progress')
    render.worker.is_cancelled = True
    render.on_render_stage('fuse')
    render.on_render_progress(1, 10)
    assert render._status_bar_label.text() == trans.t('depth_cancelling')
    assert window.statusBar().currentMessage() == trans.t('depth_cancelling')
    window._bar.deleteLater()


def test_render_progress_count_remains_visible_after_status_message_clears(qapp):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QStatusBar
    from locales import trans
    window, render, _source = _setup(qapp)
    window._bar = QStatusBar()
    render.worker = SimpleNamespace(is_cancelled=False)
    render.on_render_stage('fuse')
    render.on_render_progress(1, 228)
    assert render._status_bar_progress.maximum() == 228
    assert render._status_bar_progress.value() == 1
    assert render._status_bar_label.text().startswith(trans.t('render_stage_fuse'))
    assert '(1/228)' in render._status_bar_label.text()
    window.statusBar().clearMessage()
    assert '(1/228)' in render._status_bar_label.text()
    window._bar.deleteLater()
