"""Startup decision for a crash snapshot whose sources may be slow to check."""
from queue import Empty, SimpleQueue
from threading import Thread

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox

from locales import trans
from utils import recovery
from utils.ui_utils import style_message_box


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

    result = SimpleQueue()
    state = None
    timer = QTimer(box)

    def validate():
        # A stalled source can outlive this dialog or the whole application.
        # The worker reads only and never holds or calls a Qt widget.
        result.put(recovery.load_snapshot_state(path))

    def check_result():
        nonlocal state
        try:
            state = result.get_nowait()
        except Empty:
            return
        timer.stop()
        if state is None:
            box.reject()
            return
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
        timer.stop()
        box.deleteLater()
