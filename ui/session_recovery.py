"""Startup decision for a crash snapshot whose sources may be slow to check."""
from queue import Empty, SimpleQueue
from threading import Event, Thread

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox, QProgressBar

from locales import trans
from utils import recovery
from utils.logging_setup import get_logger
from utils.ui_utils import show_warning_box, style_message_box


_log = get_logger(__name__)


def prompt_crash_recovery(parent):
    """Return validated state only when the user explicitly selects Restore."""
    path = recovery.crash_snapshot_path()
    if path is None:
        return None

    box = QMessageBox(parent)
    if hasattr(QMessageBox, 'Option'):
        box.setOption(QMessageBox.Option.DontUseNativeDialog)
    style_message_box(box)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(trans.t('recovery_title'))
    box.setText(trans.t('recovery_checking'))
    restore = box.addButton(trans.t('btn_restore'), QMessageBox.ButtonRole.AcceptRole)
    fresh = box.addButton(trans.t('btn_start_fresh'), QMessageBox.ButtonRole.RejectRole)
    restore.setEnabled(False)
    box.setDefaultButton(fresh)
    box.setEscapeButton(fresh)
    progress = QProgressBar(box)
    progress.setRange(0, 0)
    progress.setTextVisible(False)
    layout = box.layout()
    layout.addWidget(progress, layout.rowCount(), 0, 1, layout.columnCount())

    result = SimpleQueue()
    cancelled = Event()
    state = None
    timer = QTimer(box)

    def validate():
        # A stalled source can outlive this dialog or the whole application.
        # The worker reads only and never holds or calls a Qt widget.
        result.put(recovery.load_snapshot_state(path, should_cancel=cancelled.is_set))

    def check_result():
        nonlocal state
        if getattr(parent, '_shutting_down', False):
            box.reject()
            return
        try:
            state = result.get_nowait()
        except Empty:
            return
        timer.stop()
        if state is None:
            box.reject()
            return
        progress.hide()
        box.setText(trans.t('recovery_text').format(count=recovery.snapshot_frame_count(state)))
        restore.setEnabled(True)
        box.setDefaultButton(restore)

    timer.timeout.connect(check_result)
    timer.start(50)
    try:
        Thread(target=validate, name='openfocus-recovery-validation', daemon=True).start()
        box.exec()
        return state if box.clickedButton() is restore else None
    finally:
        cancelled.set()
        timer.stop()
        box.deleteLater()


def restore_crash_session(window):
    """Offer recovery and start replay with the main window's load controls."""
    source_manager = window.source_manager
    generation = getattr(source_manager, '_load_generation', 0)
    state = prompt_crash_recovery(window)
    # The control channel can change the stack during the modal prompt.
    if (state is None or window.raw_images or getattr(source_manager, '_load_worker', None) is not None
            or generation != getattr(source_manager, '_load_generation', 0)
            or getattr(window, '_shutting_down', False)):
        return False

    from utils.project_file import apply_project
    document = state.get('recovery_document')

    def restored():
        window.project_path = (document['project_path'] or None) if document else None
        window._project_dirty = document['dirty'] if document else True
        window._update_project_title()
        _log.info('Recovered the previous session')

    def failed(reason):
        if reason != 'CANCELLED' and not getattr(window, '_shutting_down', False):
            _log.warning('Crash recovery failed: %s', reason)
            show_warning_box(window, trans.t('recovery_title'), reason)

    try:
        apply_project(window, state, on_loaded=restored, on_error=failed, crash_restore=True)
    except Exception as exc:
        failed(str(exc))
        return False
    return True
