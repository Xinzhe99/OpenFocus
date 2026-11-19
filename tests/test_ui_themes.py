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
    from dialogs import _community_download
    from dialogs.support import SupportUsDialog

    server = QTcpServer()
    assert server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost))
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    origin = f'http://127.0.0.1:{server.serverPort()}'
    monkeypatch.setattr(_community_download, 'QR_BASE_URL', f'{origin}/v1/qr')
    monkeypatch.setattr(SupportUsDialog, '_SPONSORS_URL', f'{origin}/v1/sponsors')
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
    from dialogs._community_download import QR_BASE_URL
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
                    (label.property('communityUrl') if label.property('communityUrl').startswith(QR_BASE_URL)
                     else label.accessibleName()): label
                    for label in dialog.findChildren(QLabel, 'communityLink')
                }
                expected = {
                    'xiexinzhe@zju.edu.cn': 'mailto:xiexinzhe@zju.edu.cn',
                    'xiexinzheyyy@163.com': 'mailto:xiexinzheyyy@163.com',
                    'xiexinzhe123@gmail.com': 'mailto:xiexinzhe123@gmail.com',
                    trans.t('contact_github'): None,
                    'Discord': DISCORD_INVITE_URL,
                    f'{QR_BASE_URL}/wechat-group': f'{QR_BASE_URL}/wechat-group',
                    f'{QR_BASE_URL}/wechat-official-account': f'{QR_BASE_URL}/wechat-official-account',
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
                wechat_card = dialog.findChild(QLabel, 'wechat-group').parentWidget()
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


@pytest.mark.parametrize('response', ['png', 'invalid', 'oversized', 'timeout', 'redirect',
                                      'unpublished', 'wrong_type', 'oversized_pixels'])
def test_contact_uses_only_current_published_qr(qapp, monkeypatch, invite_server, response):
    from io import BytesIO

    from PIL import Image
    from PyQt6.QtWidgets import QLabel

    from dialogs import _community_download
    from dialogs.about import ContactInfoDialog
    from locales import trans
    from tests.test_control_channel import wait_until

    def png(color):
        dimensions = (1025, 16) if response == 'oversized_pixels' else (16, 16)
        with BytesIO() as stream:
            Image.new('RGB', dimensions, color).save(stream, 'PNG')
            return stream.getvalue()

    data = png('red')
    if response == 'oversized':
        data = data.ljust(_community_download.QR_MAX_BYTES + 1, b'\0')
    elif response == 'invalid':
        data = b'invalid PNG'
    paths = []

    def accept_connection():
        socket = invite_server.nextPendingConnection()
        if response == 'timeout':
            return

        def write_response():
            if not socket.canReadLine():
                return
            request = bytes(socket.readLine()).split()
            paths.append(request[1])
            socket.readAll()
            if response == 'redirect' and request[1] != b'/redirected.png':
                socket.write(b'HTTP/1.1 302 Found\r\nLocation: /redirected.png\r\nContent-Length: 0\r\n\r\n')
            elif response == 'unpublished':
                socket.write(b'HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n')
            else:
                mime = 'text/html' if response == 'wrong_type' else 'image/png'
                headers = (f'HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\n'
                           f'Content-Length: {len(data)}\r\nConnection: close\r\n\r\n')
                socket.write(headers.encode('ascii') + data)
            socket.disconnectFromHost()

        socket.readyRead.connect(write_response)
        if socket.bytesAvailable():
            write_response()

    invite_server.newConnection.connect(accept_connection)
    if response == 'timeout':
        monkeypatch.setattr(_community_download.CommunityDownloads, 'timeout_seconds', 0.2)

    def check_dialog(color=None):
        dialog = ContactInfoDialog()
        dialog.show()
        try:
            qr = dialog.findChild(QLabel, 'wechat-group')
            status = qr.parentWidget().findChild(QLabel, 'communityMuted')
            assert qr.isHidden() and status.text() == trans.t('community_qr_loading')
            assert wait_until(lambda: status.isHidden() or status.text() != trans.t('community_qr_loading'))
            if color:
                assert qr.isVisible() and status.isHidden()
                assert qr.pixmap().toImage().pixelColor(0, 0).name() == color
            else:
                assert qr.isHidden()
                expected = 'community_qr_unpublished' if response == 'unpublished' else 'community_qr_unavailable'
                assert status.text() == trans.t(expected)
        finally:
            dialog.close()
            dialog.deleteLater()
            qapp.processEvents()

    if response == 'png':
        check_dialog('#ff0000')
        data = png('blue')
        check_dialog('#0000ff')
    else:
        check_dialog()
    assert b'/redirected.png' not in paths


@pytest.mark.parametrize('entry', ['action_contact', 'support_title'])
def test_community_dialogs_leave_main_window_responsive_during_pending_requests(
        qapp, tmp_path, monkeypatch, theme_environment, invite_server, entry):
    from PyQt6 import sip
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtNetwork import QAbstractSocket
    from PyQt6.QtTest import QTest
    from dialogs._community_download import CommunityDownloads
    from dialogs.about import ContactInfoDialog
    from dialogs.support import SupportUsDialog
    from tests.test_control_channel import _new_main_window, _destroy_main_window, wait_until
    from utils.design_tokens import palette
    from PyQt6.QtWidgets import QLabel

    monkeypatch.setattr(CommunityDownloads, 'timeout_seconds', 10)
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    action = window.ui_objs[entry]
    dialog_type = ContactInfoDialog if entry == 'action_contact' else SupportUsDialog

    def dismiss_modal():
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, dialog_type):
            dialog.reject()

    def open_from_menu():
        # A regression to exec() must fail the assertion instead of hanging the test.
        QTimer.singleShot(0, dismiss_modal)
        action.trigger()
        dialogs = [d for d in window.findChildren(dialog_type) if d.isVisible()]
        assert len(dialogs) == 1 and qapp.activeModalWidget() is None
        return dialogs[0]

    try:
        for theme in ('gray', 'light'):
            dialog = open_from_menu()
            assert open_from_menu() is dialog
            sockets = []
            for _ in range(2 if entry == 'action_contact' else 3):
                assert wait_until(invite_server.hasPendingConnections)
                socket = invite_server.nextPendingConnection()
                assert wait_until(socket.canReadLine)
                sockets.append(socket)

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
            for socket in sockets:
                # Completing the headers after close must neither read a body nor touch the UI.
                socket.write(b'HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 1024\r\n\r\n')
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
