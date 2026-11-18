"""Dialog theme and empty-state behavior through real Qt widgets."""
import pytest

pytest.importorskip("PyQt6")


@pytest.fixture
def theme_environment(qapp):
    from PyQt6.QtGui import QPalette
    from ui import styles
    from locales import trans

    stylesheet = qapp.styleSheet()
    app_palette = QPalette(qapp.palette())
    theme = styles.CURRENT_THEME
    language = trans.current_lang
    try:
        yield
    finally:
        styles.set_current_theme(theme)
        qapp.setStyleSheet(stylesheet)
        qapp.setPalette(app_palette)
        if trans.current_lang != language:
            trans.set_language(language)
        qapp.processEvents()


@pytest.fixture
def invite_server(qapp, monkeypatch):
    from PyQt6.QtNetwork import QHostAddress, QTcpServer
    from dialogs.about import ContactInfoDialog

    server = QTcpServer()
    assert server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost))
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    monkeypatch.setattr(ContactInfoDialog, '_WECHAT_URL',
                        f'http://127.0.0.1:{server.serverPort()}/wechat-group.png')
    yield server
    server.close()
    server.deleteLater()


def test_contact_link_colour_follows_the_theme(qapp, theme_environment, invite_server):
    from PyQt6.QtGui import QPalette
    from PyQt6.QtWidgets import QWidget
    from dialogs.about import ContactInfoDialog
    from ui.styles import get_theme_style

    parent = QWidget()
    colours = []
    try:
        for theme in ("light", "dark"):
            qapp.setStyleSheet(get_theme_style(theme))
            parent.ui_theme = theme
            dialog = ContactInfoDialog(parent)
            try:
                dialog.show()
                qapp.processEvents()
                dialog.ensurePolished()
                browser = dialog.text_browser
                background = browser.palette().color(QPalette.ColorRole.Base)
                document = browser.document()
                block = document.begin()
                anchors = []
                while block.isValid():
                    iterator = block.begin()
                    while not iterator.atEnd():
                        fragment = iterator.fragment()
                        if fragment.isValid() and fragment.charFormat().isAnchor():
                            anchors.append(fragment.charFormat().foreground().color())
                        iterator += 1
                    block = block.next()
                assert anchors
                assert all(color.isValid() and color != background for color in anchors)
                colours.append(anchors)
            finally:
                dialog.close()
                dialog.deleteLater()
                qapp.processEvents()
        assert colours[0] != colours[1]
    finally:
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize('response', ['png', 'invalid', 'oversized', 'timeout',
                                      'redirect', 'cross_origin', 'no_bundle'])
def test_contact_wechat_invite_refresh(qapp, monkeypatch, invite_server, response):
    from pathlib import Path

    from PyQt6.QtNetwork import QHostAddress, QTcpServer

    from dialogs import about
    from dialogs.about import ContactInfoDialog
    from tests.test_control_channel import wait_until
    from locales import trans
    from utils import resource_path

    data = Path(resource_path('assets', 'readme', 'wechat-group.png')).read_bytes()
    if response == 'oversized':
        # Valid PNG followed by padding. Invalid image data alone would test decoding.
        data = data.ljust(ContactInfoDialog._WECHAT_MAX_BYTES + 1, b'\0')
    elif response in ('invalid', 'no_bundle'):
        data = b'invalid PNG'
    if response == 'no_bundle':
        monkeypatch.setattr(about, 'resource_path', lambda *parts: '')

    other_origin = QTcpServer()
    if response == 'cross_origin':
        assert other_origin.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost))

    def accept_connection():
        socket = invite_server.nextPendingConnection()
        if response == 'timeout':
            return

        def write_response():
            if not socket.canReadLine():
                return
            request = bytes(socket.readLine())
            socket.readAll()
            if response in ('redirect', 'cross_origin') and b' /wechat-group.png ' in request:
                location = ('/current.png' if response == 'redirect' else
                            f'http://127.0.0.1:{other_origin.serverPort()}/current.png')
                socket.write(f'HTTP/1.1 302 Found\r\nLocation: {location}\r\n\r\n'.encode('ascii'))
            else:
                headers = (f'HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n'
                           f'Content-Length: {len(data)}\r\nConnection: close\r\n\r\n')
                socket.write(headers.encode('ascii') + data)
            socket.disconnectFromHost()

        socket.readyRead.connect(write_response)
        if socket.bytesAvailable():
            write_response()

    invite_server.newConnection.connect(accept_connection)
    if response == 'timeout':
        monkeypatch.setattr(ContactInfoDialog, '_WECHAT_TIMEOUT_MS', 100)
    dialog = ContactInfoDialog()
    try:
        assert dialog._wechat_status.text() == trans.t('contact_wechat_loading')
        if response != 'no_bundle':
            bundled = dialog._wechat_qr.pixmap().cacheKey()
        assert wait_until(lambda: dialog._wechat_refresh is None)
        if response == 'no_bundle':
            assert dialog._wechat_qr.isHidden()
            assert dialog._wechat_status.text() == trans.t('contact_wechat_unavailable')
            return
        if response == 'cross_origin':
            assert not other_origin.hasPendingConnections()
        assert not dialog._wechat_qr.pixmap().isNull()
        if response in ('png', 'redirect'):
            assert dialog._wechat_status.isHidden()
        else:
            assert dialog._wechat_qr.pixmap().cacheKey() == bundled
            assert dialog._wechat_status.text() == trans.t('contact_wechat_bundled')
    finally:
        dialog.close()
        dialog.deleteLater()


def test_contact_menu_allows_main_window_input_while_refreshing(
        qapp, tmp_path, monkeypatch, theme_environment, invite_server):
    from PyQt6 import sip
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtNetwork import QAbstractSocket
    from PyQt6.QtTest import QTest
    from dialogs.about import ContactInfoDialog
    from tests.test_control_channel import _new_main_window, _destroy_main_window, wait_until

    monkeypatch.setattr(ContactInfoDialog, '_WECHAT_TIMEOUT_MS', 30_000)
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    contact_action = window.ui_objs['action_contact']

    def dismiss_modal_contact():
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, ContactInfoDialog):
            dialog.reject()

    def open_from_menu():
        # A regression to exec() must fail the assertion instead of hanging the test.
        QTimer.singleShot(0, dismiss_modal_contact)
        contact_action.trigger()
        dialogs = [d for d in window.findChildren(ContactInfoDialog) if d.isVisible()]
        assert len(dialogs) == 1 and qapp.activeModalWidget() is None
        return dialogs[0]

    try:
        for _ in range(2):
            dialog = open_from_menu()
            assert open_from_menu() is dialog
            assert wait_until(invite_server.hasPendingConnections)
            socket = invite_server.nextPendingConnection()
            assert wait_until(socket.canReadLine)

            combo = window.combo_align
            before = window.current_alignment_method()
            next_method = 'ecc' if before == 'none' else 'none'
            key = (Qt.Key.Key_Down if combo.findData(next_method) > combo.currentIndex()
                   else Qt.Key.Key_Up)
            QTest.keyClick(combo, key)
            assert window.current_alignment_method() == next_method
            assert dialog.isVisible()

            dialog.close()
            assert wait_until(lambda: sip.isdeleted(dialog))
            # Let the outstanding header read finish. The closed dialog's worker
            # must disconnect even though the server never sends the PNG body.
            socket.write(b'HTTP/1.1 200 OK\r\nContent-Length: 1024\r\n\r\n')
            assert wait_until(lambda: socket.state() == QAbstractSocket.SocketState.UnconnectedState)
    finally:
        _destroy_main_window(qapp, window)


def test_dustmap_preview_follows_the_theme(qapp, theme_environment):
    from PyQt6.QtGui import QColor, QPalette
    from PyQt6.QtWidgets import QWidget
    from dialogs.dustmap import DustMapDialog
    from ui.styles import get_theme_style, set_current_theme
    from utils.design_tokens import palette

    parent = QWidget()
    try:
        for theme in ("light", "dark"):
            set_current_theme(theme)
            qapp.setStyleSheet(get_theme_style(theme))
            dialog = DustMapDialog(parent)
            try:
                dialog.show()
                qapp.processEvents()
                dialog.ensurePolished()
                expected = QColor(palette(theme)["surfaceSunken"])
                for label in (dialog.lbl_gain, dialog.lbl_before, dialog.lbl_after):
                    assert label.text().strip()
                    image = label.grab().toImage()
                    actual = image.pixelColor(5, 5)
                    assert all(abs(a - b) <= 1 for a, b in zip(actual.getRgb()[:3], expected.getRgb()[:3]))
                    foreground = label.palette().color(QPalette.ColorRole.WindowText)
                    assert foreground != actual
            finally:
                dialog.close()
                dialog.deleteLater()
                qapp.processEvents()
    finally:
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


def test_batch_dialog_has_an_empty_state(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog
    from tests.test_batch_drag_drop import _make_stacks, _wait_import

    folders = _make_stacks(tmp_path, count=1, frames=1)
    dialog = BatchProcessingDialog(None)
    try:
        dialog.show()
        qapp.processEvents()
        assert dialog._folder_list_placeholder.isVisible()
        assert dialog._folder_list_placeholder.text().strip()
        assert not dialog.folder_list.isVisible()
        dialog.add_folders_to_list(folders)
        _wait_import(dialog, qapp)
        assert not dialog._import_busy
        assert dialog.folder_list.count() == 1
        assert dialog.folder_list.isVisible()
        assert not dialog._folder_list_placeholder.isVisible()
        dialog.folder_list.setCurrentRow(0)
        dialog.remove_selected_folders()
        qapp.processEvents()
        assert dialog.folder_list.count() == 0
        assert dialog._folder_list_placeholder.isVisible()
        assert not dialog.folder_list.isVisible()
    finally:
        worker = dialog._import_worker
        dialog.cancel_imports()
        if worker is not None:
            assert worker.wait(20000)
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
