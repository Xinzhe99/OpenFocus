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
    from dialogs.about import ContactInfoDialog
    from dialogs.support import SupportUsDialog

    server = QTcpServer()
    assert server.listen(QHostAddress(QHostAddress.SpecialAddress.LocalHost))
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    origin = f'http://127.0.0.1:{server.serverPort()}'
    monkeypatch.setattr(_community_download, 'QR_BASE_URL', f'{origin}/v1/qr')
    monkeypatch.setattr(ContactInfoDialog, '_CONTACT_INFO_URL', f'{origin}/v1/contact-info')
    monkeypatch.setattr(SupportUsDialog, '_SPONSORS_URL', f'{origin}/v1/sponsors')
    yield server
    server.close()
    server.deleteLater()


@pytest.fixture
def contact_replies(invite_server):
    reply = {'body': b'{}', 'pngs': {}}
    paths = []

    def accept_connection():
        socket = invite_server.nextPendingConnection()

        def write_response():
            if not socket.canReadLine():
                return
            path = bytes(socket.readLine()).split()[1]
            socket.readAll()
            if path.startswith(b'/v1/contact-info?'):
                paths.append(path)
                body = reply['body']
                headers = (f'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                           f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n')
                socket.write(headers.encode('ascii') + body)
            elif path in reply['pngs']:
                body = reply['pngs'][path]
                headers = (f'HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n'
                           f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n')
                socket.write(headers.encode('ascii') + body)
            else:
                socket.write(b'HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            socket.disconnectFromHost()

        socket.readyRead.connect(write_response)
        if socket.bytesAvailable():
            write_response()

    invite_server.newConnection.connect(accept_connection)
    yield reply, paths
    invite_server.newConnection.disconnect(accept_connection)


def test_contact_links_are_accessible_and_follow_the_theme(qapp, theme_environment, contact_replies):
    import json
    from PyQt6.QtCore import QPoint, QRect, Qt
    from PyQt6.QtGui import QPalette, QTextDocument
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLabel, QScrollArea, QTextBrowser, QWidget
    from dialogs.about import ContactInfoDialog
    from dialogs._community_download import QR_BASE_URL
    from locales import trans
    from tests.test_control_channel import wait_until
    from ui.styles import get_theme_style

    reply, _ = contact_replies
    trans.set_language('en')
    emails = ('first+reply@example.org', 'second@example.org', 'third@example.org')
    institution = 'Research <Lab> & Partners'
    github_url = 'https://github.com/openfocus-test/project?view=issues&state=open'
    discord_url = 'https://discord.gg/openfocus-test'
    reply['body'] = json.dumps(dict(emails=emails, institution=institution, github_url=github_url,
                                   discord_url=discord_url)).encode('utf-8')

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
                assert wait_until(lambda: any(label.isVisible()
                                              for label in dialog.findChildren(QLabel, 'contactInstitution')))
                institution_label = dialog.findChild(QLabel, 'contactInstitution')
                assert institution_label.textFormat() == Qt.TextFormat.PlainText
                assert institution_label.text() == institution

                links = {
                    (label.property('communityUrl') if label.property('communityUrl').startswith(QR_BASE_URL)
                     else label.accessibleName()): label
                    for label in dialog.findChildren(QLabel, 'communityLink')
                }
                expected = {
                    **{address: f'mailto:{address}' for address in emails},
                    trans.t('contact_github'): github_url,
                    'Discord': discord_url,
                    trans.t('contact_discord_join'): discord_url,
                    f'{QR_BASE_URL}/wechat-group': f'{QR_BASE_URL}/wechat-group',
                    f'{QR_BASE_URL}/wechat-official-account': f'{QR_BASE_URL}/wechat-official-account',
                }
                assert links.keys() == expected.keys()
                contact_card = links[emails[0]].parentWidget()
                discord_card = links[trans.t('contact_discord_join')].parentWidget()
                heading = next(label for label in contact_card.findChildren(QLabel, 'communitySection')
                               if label.text() == trans.t('contact_community_links'))
                assert heading.isVisible()
                assert heading.y() < links[trans.t('contact_github')].y() < links['Discord'].y()
                viewport = dialog.findChild(QScrollArea).viewport()
                for label in (dialog.findChild(QLabel, 'communityTitle'),
                              links[trans.t('contact_discord_join')]):
                    assert viewport.rect().contains(QRect(label.mapTo(viewport, QPoint()), label.size()))
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

                wechat_card = dialog.findChild(QLabel, 'wechat-group').parentWidget()
                assert contact_card.x() < discord_card.x() < wechat_card.x()
                dialog.resize(560, 480)
                qapp.processEvents()
                assert discord_card.x() > contact_card.x() and discord_card.y() == contact_card.y()
                assert wechat_card.x() == contact_card.x() and wechat_card.y() > contact_card.y()
                dialog.resize(440, 480)
                qapp.processEvents()
                assert wechat_card.x() == contact_card.x()
                assert discord_card.x() == contact_card.x()
                assert contact_card.y() < discord_card.y() < wechat_card.y()
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


def test_contact_refreshes_and_omits_missing_or_unsafe_details(qapp, theme_environment, contact_replies):
    import json
    from PyQt6.QtWidgets import QLabel
    from dialogs.about import ContactInfoDialog
    from locales import trans
    from tests.test_control_channel import wait_until

    reply, paths = contact_replies
    trans.set_language('en')
    first_address = 'first@example.org'
    reply['body'] = json.dumps(dict(emails=[first_address], institution='First institution')).encode('utf-8')
    dialog = ContactInfoDialog()

    def visible_labels():
        return [label for label in dialog.findChildren(QLabel) if label.isVisible()]

    def visible_text():
        return {label.text() for label in visible_labels()}

    def request_contact():
        dialog.show()
        assert trans.t('contact_details_loading') in visible_text()

    try:
        request_contact()
        assert wait_until(lambda: 'First institution' in visible_text())
        assert trans.t('contact_institution') in visible_text()
        assert trans.t('contact_email') in visible_text()
        dialog.close()

        trans.set_language('zh')
        long_address = ('a' * 64 + '@' + 'b' * 63 + '.' + 'c' * 63 + '.example.org')
        reply['body'] = json.dumps(dict(
            emails=[long_address, 'bad@example.org?subject=unwanted'],
            github_url='https://github.com@example.org/project',
            discord_url='https://discord.com/channels/not-an-invite',
        )).encode('utf-8')
        request_contact()
        assert wait_until(lambda: trans.t('contact_email') in visible_text())
        assert trans.t('contact_discord_unpublished') in visible_text()
        assert trans.t('contact_community_links') not in visible_text()
        assert trans.t('contact_institution') not in visible_text()
        assert 'First institution' not in visible_text()
        links = [label.property('communityUrl') for label in visible_labels()
                 if label.objectName() == 'communityLink']
        assert [url for url in links if '/v1/qr/' not in url] == [f'mailto:{long_address}']
        dialog.resize(440, 480)
        qapp.processEvents()
        assert dialog.width() == 440
        dialog.close()

        reply['body'] = b'{"emails": [], "institution": "", "github_url": "", "discord_url": ""}'
        request_contact()
        assert wait_until(lambda: trans.t('contact_details_empty') in visible_text())
        assert trans.t('contact_discord_unpublished') in visible_text()
        assert trans.t('contact_email') not in visible_text()
        assert trans.t('contact_institution') not in visible_text()
        dialog.close()

        reply['body'] = b'{"institution":"' + b'x' * (32 * 1024) + b'"}'
        request_contact()
        assert wait_until(lambda: trans.t('contact_details_unavailable') in visible_text())
        assert trans.t('contact_discord_unavailable') in visible_text()
        assert paths == [b'/v1/contact-info?lang=en'] + [b'/v1/contact-info?lang=zh'] * 3
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def _image_pixels(image):
    import numpy as np
    from PyQt6.QtGui import QImage

    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    bits = rgb.constBits()
    bits.setsize(rgb.sizeInBytes())
    rows = np.frombuffer(bits, dtype=np.uint8).reshape(rgb.height(), rgb.bytesPerLine())
    return rows[:, :rgb.width() * 3].reshape(rgb.height(), rgb.width(), 3).copy()


def test_discord_qr_decodes_the_current_published_invite(qapp, theme_environment, contact_replies):
    import json

    import cv2
    import numpy as np
    from PyQt6.QtWidgets import QLabel
    from dialogs.about import ContactInfoDialog
    from locales import trans
    from tests.test_control_channel import wait_until

    reply, paths = contact_replies
    trans.set_language('en')
    dialog = ContactInfoDialog()
    qr = dialog.findChild(QLabel, 'discord-invite')
    card = qr.parentWidget()
    status = card.findChild(QLabel, 'communityMuted')

    def open_with_invite(invite):
        reply['body'] = json.dumps({'discord_url': invite}).encode('utf-8')
        dialog.show()
        assert qr.isHidden() and qr.pixmap().isNull()
        assert status.isVisible() and status.text() == trans.t('contact_discord_loading')
        assert not any(link.isVisible() for link in card.findChildren(QLabel, 'communityLink'))

    try:
        for invite in ('https://discord.gg/OpenFocus1', 'https://discord.gg/Next?x=1&y=2'):
            open_with_invite(invite)
            assert wait_until(qr.isVisible)
            assert not status.isVisible()
            links = [link for link in card.findChildren(QLabel, 'communityLink') if link.isVisible()]
            assert len(links) == 1 and links[0].property('communityUrl') == invite

            # Decode Qt's painted widget at its actual device pixel ratio.
            image = qr.grab().toImage()
            pixels = _image_pixels(image)
            assert cv2.QRCodeDetector().detectAndDecode(pixels)[0] == invite
            badge_y, badge_x = np.nonzero(np.all(pixels == [88, 101, 242], axis=2))
            assert badge_x.size and badge_y.size
            assert abs(badge_x.min() + badge_x.max() + 1 - image.width()) <= 2
            assert abs(badge_y.min() + badge_y.max() + 1 - image.height()) <= 2
            badge = pixels[badge_y.min():badge_y.max() + 1, badge_x.min():badge_x.max() + 1]
            assert np.any(np.all(badge == 255, axis=2))
            dialog.close()

        open_with_invite('')
        assert wait_until(lambda: status.text() == trans.t('contact_discord_unpublished'))
        assert qr.isHidden() and qr.pixmap().isNull()
        assert not any(link.isVisible() for link in card.findChildren(QLabel, 'communityLink'))
        assert paths == [b'/v1/contact-info?lang=en'] * 3
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_community_qr_same_geometry_in_dialog_capture(qapp, theme_environment, contact_replies, tmp_path):
    import json
    from io import BytesIO

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from PyQt6.QtCore import QPoint, QRect
    from PyQt6.QtWidgets import QLabel

    from dialogs.about import ContactInfoDialog
    from dialogs.support import SupportUsDialog
    from locales import trans
    from tests.test_control_channel import wait_until

    reply, _ = contact_replies
    trans.set_language('en')
    invite = 'https://discord.gg/OpenFocus1'
    reply['body'] = json.dumps({'discord_url': invite}).encode('utf-8')
    mark = (40, 90, 180)
    codes = {
        'wechat-group': ('GROUP1', 1, (5, 79, 68, 4), True),
        'wechat-official-account': ('https://example.org/official', 6, (61, 3, 7, 38), False),
        'alipay': ('https://example.org/alipay', 5, (12, 45, 35, 11), False),
        'paypal': ('https://example.org/paypal', 6, (83, 38, 24, 12), True),
    }
    for slot, (payload, version, padding, transparent) in codes.items():
        params = cv2.QRCodeEncoder_Params()
        params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_H
        params.version = version
        modules = cv2.QRCodeEncoder_create(params).encode(payload)[2:-2, 2:-2]
        step = 6
        side = len(modules) * step
        artwork = Image.fromarray(modules).resize((side, side), Image.Resampling.NEAREST).convert('RGB')
        center = side // 2
        ImageDraw.Draw(artwork).rectangle((center - step, center - step, center + step, center + step), fill=mark)
        left, top, right, bottom = padding
        size = (left + side + right, top + side + bottom)
        # Transparent padding deliberately carries black RGB beneath its zero alpha.
        source = Image.new('RGBA', size, (0, 0, 0, 0)) if transparent else Image.new('RGB', size, 'white')
        source.paste(artwork, (left, top))
        stream = BytesIO()
        source.save(stream, 'PNG')
        reply['pngs'][f'/v1/qr/{slot}'.encode()] = stream.getvalue()

    for dialog_type, slots, box in (
        (ContactInfoDialog, ('discord-invite', 'wechat-group', 'wechat-official-account'), 152),
        (SupportUsDialog, ('alipay', 'paypal'), 228),
    ):
        dialog = dialog_type()
        try:
            dialog.show()
            if qapp.platformName() == 'windows':
                dialog.raise_()
            labels = [dialog.findChild(QLabel, slot) for slot in slots]
            assert wait_until(lambda: all(label.isVisible() for label in labels))
            assert wait_until(lambda: dialog.windowHandle().isExposed())
            qapp.processEvents()
            if qapp.platformName() == 'windows':
                qapp.sync()
                capture = dialog.screen().grabWindow(int(dialog.winId()))
            else:
                capture = dialog.grab()
            assert not capture.isNull()
            assert capture.save(str(tmp_path / f'{dialog_type.__name__}.png'))
            image = capture.toImage()
            ratio = dialog.devicePixelRatioF()
            side = round(box * ratio)
            for slot, label in zip(slots, labels):
                assert label.width() == label.height() == box
                assert abs(label.geometry().center().x() - label.parentWidget().rect().center().x()) <= 1
                origin = label.mapTo(dialog, QPoint())
                region = QRect(round(origin.x() * ratio), round(origin.y() * ratio), side, side)
                assert image.rect().contains(region)
                pixels = _image_pixels(image.copy(region))
                ys, xs = np.nonzero(np.any(pixels[1:-1, 1:-1] < 192, axis=2))
                width, height = np.ptp(xs) + 1, np.ptp(ys) + 1
                assert width == height == side * 7 // 10
                assert abs(xs.min() + xs.max() + 1 - (side - 2)) <= 2
                assert abs(ys.min() + ys.max() + 1 - (side - 2)) <= 2
                # Account for the one-pixel trim at the native capture boundary.
                version = 1 if slot == 'discord-invite' else codes[slot][1]
                modules = 17 + 4 * version
                quiet = min(xs.min(), ys.min(), side - 3 - xs.max(), side - 3 - ys.max())
                assert quiet + 1 >= 4 * width / modules
                payload = invite if slot == 'discord-invite' else codes[slot][0]
                assert cv2.QRCodeDetector().detectAndDecode(pixels[1:-1, 1:-1])[0] == payload
                if slot != 'discord-invite':
                    assert np.any(np.all(pixels == mark, axis=2))
        finally:
            dialog.close()
            dialog.deleteLater()
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
            socket.readAll()
            if request[1].startswith(b'/v1/contact-info?'):
                socket.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}')
                socket.disconnectFromHost()
                return
            paths.append(request[1])
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
                image = qr.pixmap().toImage()
                assert image.pixelColor(0, 0).name() == '#ffffff'
                assert image.pixelColor(image.width() // 2, image.height() // 2).name() == color
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
            for _ in range(3):
                assert wait_until(invite_server.hasPendingConnections)
                socket = invite_server.nextPendingConnection()
                assert wait_until(socket.canReadLine)
                path = bytes(socket.readLine()).split()[1]
                mime = b'image/png' if path.startswith(b'/v1/qr/') else b'application/json'
                sockets.append((socket, mime))

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
            for socket, mime in sockets:
                # Completing the headers after close must neither read a body nor touch the UI.
                socket.write(b'HTTP/1.1 200 OK\r\nContent-Type: ' + mime + b'\r\nContent-Length: 1024\r\n\r\n')
                assert wait_until(lambda: socket.state() == QAbstractSocket.SocketState.UnconnectedState)
    finally:
        _destroy_main_window(qapp, window)


def test_status_sponsor_opens_published_link_without_displacing_messages(
        qapp, tmp_path, monkeypatch, theme_environment):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QDesktopServices, QTextDocument
    from PyQt6.QtTest import QTest
    from locales import trans
    from tests.test_control_channel import _new_main_window, _destroy_main_window, wait_until
    from ui.status_bar import SponsorLink

    destination = 'https://example.org/sponsor'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            text = 'OpenFocus <Studio>' if self.path.endswith('?lang=zh') else 'OpenFocus Studio'
            body = json.dumps({'text': text, 'url': destination}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
    thread.start()
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    monkeypatch.setattr(SponsorLink, '_endpoint',
                        f'http://127.0.0.1:{server.server_port}/v1/sponsor-placement')
    trans.set_language('en')
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()))
    window = None
    try:
        window = _new_main_window(qapp, tmp_path, monkeypatch)
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
        if window is not None:
            _destroy_main_window(qapp, window)
        server.shutdown()
        thread.join()
        server.server_close()


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


@pytest.mark.parametrize('language', ['en', 'zh'])
def test_accelerator_download_priority_can_override_region(qapp, monkeypatch, language):
    from PyQt6.QtWidgets import QWidget
    from core.inference_pack import PackStatus
    from locales import trans
    from ui import accelerator_setup

    monkeypatch.setattr(trans, 'current_lang', language)
    window = QWidget()
    banner = accelerator_setup.AcceleratorBanner(window)
    banner.status = PackStatus(True, 'NVIDIA GPU', '', 100, False, ('TORCH_CUDA',), 'test', 200)
    installs = []
    monkeypatch.setattr(banner, '_start', lambda *args: installs.append(args))
    try:
        window.show()
        banner.open_settings()
        qapp.processEvents()
        dialog = banner.dialog
        assert dialog.install_button.isVisible()
        assert dialog.source.isVisible()
        assert dialog.source.currentIndex() == 0
        for index, expected in enumerate((None, True, False)):
            dialog.source.setCurrentIndex(index)
            banner.retranslate()
            dialog.install_button.click()
            assert installs[-1] == (True, None, expected)
    finally:
        if banner.dialog is not None:
            banner.dialog.close()
        window.close()
        window.deleteLater()
        qapp.processEvents()
