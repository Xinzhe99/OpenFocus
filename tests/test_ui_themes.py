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


def test_contact_links_are_accessible_and_follow_the_theme(qapp, theme_environment, invite_server):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPalette, QTextDocument
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLabel, QScrollArea, QTextBrowser, QWidget
    from constants import DISCORD_INVITE_URL
    from dialogs.about import ContactInfoDialog
    from locales import trans
    from ui.styles import get_theme_style

    parent = QWidget()
    colours = []
    try:
        for theme in ('light', 'dark', 'gray'):
            qapp.setStyleSheet(get_theme_style(theme))
            parent.ui_theme = theme
            dialog = ContactInfoDialog(parent)
            try:
                dialog.show()
                dialog.activateWindow()
                qapp.processEvents()
                assert not dialog.findChildren(QTextBrowser)
                assert len(dialog.findChildren(QScrollArea)) == 1
                assert any(label.text() == trans.t('contact_zju')
                           for label in dialog.findChildren(QLabel))

                links = {
                    label.accessibleName(): label
                    for label in dialog.findChildren(QLabel, 'communityLink')
                }
                expected = {
                    'xiexinzhe@zju.edu.cn': 'mailto:xiexinzhe@zju.edu.cn',
                    'xiexinzheyyy@163.com': 'mailto:xiexinzheyyy@163.com',
                    'xiexinzhe123@gmail.com': 'mailto:xiexinzhe123@gmail.com',
                    trans.t('contact_github'): None,
                    'Discord': DISCORD_INVITE_URL,
                    trans.t('contact_wechat_group'): ContactInfoDialog._WECHAT_URL,
                }
                assert links.keys() == expected.keys()
                theme_colours = {}
                for name, url in expected.items():
                    link = links[name]
                    assert link.isVisible()
                    assert link.openExternalLinks()
                    assert link.focusPolicy() & Qt.FocusPolicy.TabFocus
                    assert link.textInteractionFlags() & Qt.TextInteractionFlag.LinksAccessibleByKeyboard

                    document = QTextDocument()
                    document.setHtml(link.text())
                    anchor = document.begin().begin().fragment().charFormat()
                    assert anchor.isAnchor()
                    href = anchor.anchorHref()
                    if url is None:
                        assert href.startswith('https://github.com/') and href.endswith('/OpenFocus')
                    else:
                        assert href == url
                    color = anchor.foreground().color()
                    background = link.parentWidget().palette().color(QPalette.ColorRole.Window)
                    assert color.isValid() and color != background
                    theme_colours[name] = color

                    # Observe the real keyboard activation without launching an external app.
                    link.setOpenExternalLinks(False)
                    activated = []
                    link.linkActivated.connect(activated.append)
                    link.setFocus(Qt.FocusReason.TabFocusReason)
                    QTest.keyClick(link, Qt.Key.Key_Tab)
                    QTest.keyClick(link, Qt.Key.Key_Return)
                    assert activated == [href]
                colours.append(theme_colours)

                contact_card = links['xiexinzhe@zju.edu.cn'].parentWidget()
                wechat_card = dialog._wechat_qr.parentWidget()
                assert wechat_card.x() > contact_card.x()
                dialog.resize(440, 480)
                qapp.processEvents()
                assert wechat_card.x() == contact_card.x()
                assert wechat_card.y() > contact_card.y()
            finally:
                dialog.close()
                dialog.deleteLater()
                qapp.processEvents()
        assert all(len({by_name[name].name() for by_name in colours}) == len(colours)
                   for name in colours[0])
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
            assert dialog._wechat_qr.width() == dialog._wechat_qr.height() == 192
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
    from utils.design_tokens import palette
    from PyQt6.QtWidgets import QLabel

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
        for theme in ('gray', 'light'):
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
            window.ui_objs[f'theme_{theme}'].trigger()
            assert all(palette(theme)['accentSoftText'] in link.text()
                       for link in dialog.findChildren(QLabel, 'communityLink'))

            dialog.close()
            assert wait_until(lambda: sip.isdeleted(dialog))
            # Let the outstanding header read finish. The closed dialog's worker
            # must disconnect even though the server never sends the PNG body.
            socket.write(b'HTTP/1.1 200 OK\r\nContent-Length: 1024\r\n\r\n')
            assert wait_until(lambda: socket.state() == QAbstractSocket.SocketState.UnconnectedState)
    finally:
        _destroy_main_window(qapp, window)


def test_status_sponsor_opens_published_link_without_displacing_messages(
        qapp, tmp_path, monkeypatch, theme_environment):
    import json
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QDesktopServices, QTextDocument
    from PyQt6.QtNetwork import QHostAddress, QTcpServer
    from PyQt6.QtTest import QTest
    from locales import trans
    from tests.test_control_channel import _new_main_window, _destroy_main_window, wait_until
    from ui.status_bar import SponsorLink

    server = QTcpServer()
    assert server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost))
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    monkeypatch.setattr(SponsorLink, '_endpoint',
                        f'http://127.0.0.1:{server.serverPort()}/v1/sponsor-placement')
    destination = 'https://example.org/sponsor'
    payload = json.dumps({'text': 'OpenFocus <Studio>', 'url': destination}).encode()
    response = (b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                + f'Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n'.encode()
                + payload)

    def accept():
        socket = server.nextPendingConnection()
        socket.disconnected.connect(socket.deleteLater)
        received = bytearray()

        def answer():
            received.extend(bytes(socket.readAll()))
            if b'\r\n\r\n' in received:
                socket.write(response)
                socket.disconnectFromHost()

        socket.readyRead.connect(answer)
        if socket.bytesAvailable():
            answer()

    server.newConnection.connect(accept)
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()))
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        window.resize(1240, window.height())
        label = window.lbl_status_sponsor
        trans.set_language('zh')
        assert wait_until(lambda: label.accessibleName() == 'OpenFocus <Studio>')
        bar = window.statusBar()
        bar.showMessage('Background notice', 5000)
        qapp.processEvents()
        doc = QTextDocument()
        doc.setHtml(label.text())
        assert doc.toPlainText().startswith(trans.t('status_credit'))
        assert doc.toPlainText().endswith('OpenFocus <Studio>')
        assert label.isVisible() and label.mapTo(bar, label.rect().topRight()).x() >= bar.width() - 15
        assert bar._message_label.text() == 'Background notice'
        assert bar._message_label.mapTo(bar, bar._message_label.rect().topRight()).x() < label.x()
        label.setFocus(Qt.FocusReason.TabFocusReason)
        QTest.keyClick(label, Qt.Key.Key_Return)
        pos = label.contentsRect().center()
        pos.setX(label.contentsRect().right() - 15)
        QTest.mouseMove(label, pos)
        QTest.mouseClick(label, Qt.MouseButton.LeftButton, pos=pos)
        assert opened == [destination, destination]
    finally:
        _destroy_main_window(qapp, window)
        server.close()
        server.deleteLater()


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


def test_depth_surface_uses_the_new_theme_without_moving_camera(qapp, theme_environment):
    import numpy as np
    from PyQt6.QtGui import QColor
    from ui.styles import get_theme_style, set_current_theme
    from widgets.depth_surface import DepthSurface

    surface = DepthSurface()
    surface.resize(390, 260)
    surface.set_surface(np.full((120, 160), 0.5, np.float32))
    try:
        for theme, surround in (("light", "#F9FAFB"), ("gray", "#606060"),
                                ("dark", "#12161B")):
            set_current_theme(theme)
            qapp.setStyleSheet(get_theme_style(theme))
            image = surface.grab().toImage()
            assert image.pixelColor(4, 4) == QColor(surround)
            assert image.pixelColor(image.rect().center()) != QColor(surround)
    finally:
        surface.deleteLater()
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
