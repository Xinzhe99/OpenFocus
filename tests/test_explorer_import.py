"""External import requests retain their selection and wait for active work."""
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication
from core.app import OpenFocusApplication, PlatformFileHandler


class Timer:
    def __init__(self):
        self.delays = []

    def start(self, delay):
        self.delays.append(delay)


def test_explorer_combines_folders_and_images_in_selection_order(tmp_path, monkeypatch):
    folder = tmp_path / "folder"
    folder.mkdir()
    first = folder / "frame2.png"
    second = folder / "frame10.png"
    first.touch()
    second.touch()
    image = tmp_path / "separate.png"
    image.touch()
    selected = []
    source = SimpleNamespace(
        _folder_import_paths=lambda path, *, strict: [str(second), str(first)],
        load_image_files_from_icon=lambda paths: selected.extend(paths),
    )
    app = SimpleNamespace(main_window=SimpleNamespace(source_manager=source))
    monkeypatch.setattr(PlatformFileHandler, "is_image", lambda path: path.endswith(".png"))

    OpenFocusApplication._process_explorer(app, [str(folder), str(image)])

    assert selected == [str(first), str(second), str(image)]


def test_external_import_waits_for_worker_and_preserves_request(tmp_path, monkeypatch):
    state = {"busy": True}
    source = SimpleNamespace(_reset_busy=lambda: state["busy"])
    received = []
    request = tmp_path / "openfocus-shell-request.json"
    request.write_text('["frame 1.png", "frame 2.png"]', encoding="utf-8")
    timer = Timer()
    app = SimpleNamespace(
        main_window=SimpleNamespace(source_manager=source),
        _import_queue=[(["frame 1.png", "frame 2.png"], True, str(request))],
        _import_processing=False,
        _import_timer=timer,
        _process_explorer=lambda paths: received.append(paths),
    )
    monkeypatch.setattr(QApplication, "activeModalWidget", lambda: None)

    OpenFocusApplication._drain_import_queue(app)
    assert not received
    assert len(app._import_queue) == 1
    assert timer.delays == [250]
    assert request.exists()

    state["busy"] = False
    OpenFocusApplication._drain_import_queue(app)
    assert received == [["frame 1.png", "frame 2.png"]]
    assert not app._import_queue

    assert not request.exists()


def test_external_import_waits_for_modal_dialog(monkeypatch):
    source = SimpleNamespace(_reset_busy=lambda: False)
    timer = Timer()
    app = SimpleNamespace(
        main_window=SimpleNamespace(source_manager=source),
        _import_queue=[(["frame.png"], True, None)],
        _import_processing=False,
        _import_timer=timer,
    )
    monkeypatch.setattr(QApplication, "activeModalWidget", lambda: object())

    OpenFocusApplication._drain_import_queue(app)
    assert len(app._import_queue) == 1
    assert timer.delays == [250]


def test_shell_request_file_must_be_private_temp_file(tmp_path, monkeypatch):
    from core import windows_shell

    monkeypatch.setattr(windows_shell.tempfile, "gettempdir", lambda: str(tmp_path))
    request = tmp_path / "openfocus-shell-栈.json"
    request.write_text('["栈/帧.png"]', encoding="utf-8")
    assert windows_shell.is_request_file(str(request))
    unrelated = tmp_path / "notes.json"
    unrelated.write_text("[]", encoding="utf-8")
    assert not windows_shell.is_request_file(str(unrelated))
    assert not windows_shell.is_request_file(str(tmp_path / "openfocus-shell-missing.json"))


@pytest.fixture
def external_import(tmp_path, monkeypatch):
    from core import app as app_module
    import utils

    folder = tmp_path / 'folder'
    folder.mkdir()
    folder2 = tmp_path / 'folder2'
    folder2.mkdir()
    paths = {name: tmp_path / name for name in
             ('frame2.png', 'frame10.png', 'separate.png', 'other.png',
              'clip.mp4', 'clip2.mp4', 'notes.txt')}
    for path in paths.values():
        path.touch()
    paths.update(folder=folder, folder2=folder2)
    calls = []
    warnings = []
    source = SimpleNamespace(
        _folder_import_paths=lambda path, *, strict: [str(paths['frame10.png']), str(paths['frame2.png'])],
        load_image_stack_from_icon=lambda value: calls.append(('folder', value)),
        load_multiple_folders_from_icon=lambda value: calls.append(('folders', value)),
        load_image_files_from_icon=lambda value: calls.append(('images', value)),
        load_video_stack_from_icon=lambda value: calls.append(('video', value)),
    )
    window = SimpleNamespace(source_manager=source)
    monkeypatch.setattr(PlatformFileHandler, 'is_image', lambda path: path.endswith('.png'))
    monkeypatch.setattr(PlatformFileHandler, 'is_video', lambda path: path.endswith('.mp4'))
    monkeypatch.setattr(utils, 'show_warning_box', lambda *args: warnings.append(args))
    monkeypatch.setattr(app_module.QApplication, 'instance', lambda: None)
    monkeypatch.setattr(app_module.QTimer, 'singleShot', lambda delay, callback: callback())

    def dispatch(entry, names):
        selected = [str(paths[name]) for name in names]
        if entry == 'application':
            app = SimpleNamespace(_ready=True, main_window=window)
            OpenFocusApplication._process(app, selected)
        else:
            app_module.process_command_line_args(window, selected)

    return paths, source, window, calls, warnings, dispatch


@pytest.mark.parametrize('entry', ['application', 'fallback'])
@pytest.mark.parametrize('names', [
    ['folder', 'separate.png'], ['separate.png', 'folder'],
])
def test_mixed_folder_images_preserve_selection_order(external_import, entry, names):
    paths, _, _, calls, warnings, dispatch = external_import
    dispatch(entry, names)
    group = [str(paths['frame2.png']), str(paths['frame10.png'])]
    separate = [str(paths['separate.png'])]
    expected = group + separate if names[0] == 'folder' else separate + group
    assert calls == [('images', expected)]
    assert not warnings


@pytest.mark.parametrize('entry', ['application', 'fallback'])
@pytest.mark.parametrize(('names', 'method'), [
    (['folder', 'folder2'], 'folders'),
    (['separate.png', 'other.png'], 'images'),
    (['clip.mp4'], 'video'),
])
def test_existing_external_import_dispatch(external_import, entry, names, method):
    paths, _, _, calls, warnings, dispatch = external_import
    dispatch(entry, names)
    selected = [str(paths[name]) for name in names]
    assert calls == [(method, selected[0] if method == 'video' else selected)]
    assert not warnings


@pytest.mark.parametrize('entry', ['application', 'fallback'])
@pytest.mark.parametrize('names', [
    ['clip.mp4', 'separate.png'], ['folder', 'clip.mp4'],
    ['clip.mp4', 'clip2.mp4'], ['clip.mp4', 'notes.txt'],
])
def test_incompatible_video_selection_is_rejected(external_import, entry, names):
    paths, _, _, calls, warnings, dispatch = external_import
    dispatch(entry, names)
    assert not calls
    assert len(warnings) == 1
    for name in names:
        assert str(paths[name]) in warnings[0][2]


@pytest.mark.parametrize('blocked_by', ['busy', 'modal'])
def test_mixed_queue_waits_and_imports_once(external_import, monkeypatch, blocked_by):
    paths, source, window, calls, warnings, _ = external_import
    state = {'blocked': True}
    source._reset_busy = lambda: state['blocked'] and blocked_by == 'busy'
    monkeypatch.setattr(QApplication, 'activeModalWidget',
                        lambda: object() if state['blocked'] and blocked_by == 'modal' else None)
    timer = Timer()
    request = paths['folder'].parent / 'openfocus-shell-queue.json'
    request.write_text('[]', encoding='utf-8')
    selected = [str(paths['folder']), str(paths['separate.png'])]
    app = SimpleNamespace(main_window=window, _ready=True,
                          _import_queue=[(selected, False, str(request))],
                          _import_processing=False, _import_timer=timer)
    app._process = lambda selection: OpenFocusApplication._process(app, selection)
    OpenFocusApplication._drain_import_queue(app)
    assert app._import_queue and request.exists() and not calls
    state['blocked'] = False
    OpenFocusApplication._drain_import_queue(app)
    assert calls == [('images', [str(paths['frame2.png']), str(paths['frame10.png']),
                                  str(paths['separate.png'])])]
    assert not app._import_queue and not request.exists() and not warnings


@pytest.mark.parametrize('outcome', ['failure', 'cancel'])
def test_mixed_queue_advances_after_failure_or_cancel(external_import, monkeypatch, outcome):
    paths, source, window, calls, warnings, _ = external_import
    source._reset_busy = lambda: False
    if outcome == 'failure':
        def fail(path, *, strict):
            raise OSError('folder unavailable')
        source._folder_import_paths = fail
    else:
        source.load_image_files_from_icon = lambda value: calls.append(('cancelled', value))
    monkeypatch.setattr(QApplication, 'activeModalWidget', lambda: None)
    request = paths['folder'].parent / 'openfocus-shell-finished.json'
    request.write_text('[]', encoding='utf-8')
    app = SimpleNamespace(main_window=window, _ready=True,
                          _import_queue=[([str(paths['folder']), str(paths['separate.png'])],
                                          False, str(request)),
                                         ([str(paths['clip.mp4'])], False, None)],
                          _import_processing=False, _import_timer=Timer())
    app._process = lambda selection: OpenFocusApplication._process(app, selection)
    OpenFocusApplication._drain_import_queue(app)
    assert not request.exists() and len(app._import_queue) == 1
    assert len(warnings) == (1 if outcome == 'failure' else 0)
    assert len(calls) == (0 if outcome == 'failure' else 1)
    OpenFocusApplication._drain_import_queue(app)
    assert calls[-1] == ('video', str(paths['clip.mp4']))
    assert not app._import_queue
