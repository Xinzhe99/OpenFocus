# dialogs/about.py
"""About and contact information dialogs."""

import json
import re
import threading
from html import escape
from unicodedata import category
from urllib.parse import urlencode, urlsplit

from PyQt6.QtCore import QEvent, QRect, QRectF, QTimer, Qt, QUrl
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QFontInfo, QImage
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QTextEdit,
)
from dialogs._community_download import CommunityDownloads
from locales import trans
from utils.account_client import BASE_URL


class _EnvironmentProbe(threading.Thread):
    def __init__(self):
        super().__init__(name="environment-probe", daemon=True)
        self.finished = threading.Event()

    def run(self):
        try:
            self.text = EnvironmentInfoDialog.environment_report()
        except Exception as exc:
            self.text = trans.t("msg_error") + "\n" + str(exc)
        finally:
            self.finished.set()


class EnvironmentInfoDialog(QDialog):
    """环境信息对话框（从 main.py 抽离）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('dialog_env_title'))
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(600, max(420, available.width() - 48)),
                        min(500, max(320, available.height() - 64)))
        else:
            self.resize(600, 500)

        # Styling comes from the generated application stylesheet.


        layout = QVBoxLayout(self)

        # 信息显示区
        self.text_edit = QTextEdit()
        self.text_edit.setToolTip(trans.t('env_subtitle'))
        self.text_edit.setReadOnly(True)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        family = fixed_font.family()
        if not QFontInfo(fixed_font).fixedPitch():
            family = next((name for name in QFontDatabase.families()
                           if QFontDatabase.isFixedPitch(name)), family)
        self.text_edit.setStyleSheet(f'font-family: "{family}";')
        layout.addWidget(self.text_edit)

        self._probe = _EnvironmentProbe()
        self.text_edit.setPlainText(trans.t("msg_gif_processing_title"))
        self._probe_timer = QTimer(self)
        self._probe_timer.setInterval(100)
        self._probe_timer.timeout.connect(self._poll_probe)
        self._probe.start()
        self._probe_timer.start()

    def _poll_probe(self):
        if self._probe.finished.is_set():
            self._probe_timer.stop()
            self._probe_finished()

    def _probe_finished(self):
        self.text_edit.setPlainText(self._probe.text)

    def done(self, result):
        self._probe_timer.stop()
        super().done(result)
        self.deleteLater()


    @staticmethod
    def environment_report():
        """Collect optional dependencies without touching GUI objects."""
        info_lines = []
        available = {}
        info_lines.append("=" * 60)
        from constants import app_display_version
        info_lines.append(f"OpenFocus v{app_display_version()}")
        info_lines.append("=" * 60)
        info_lines.append("")

        from utils.diagnostics import system_facts
        for key, value in system_facts().items():
            info_lines.append(f"{key}: {value}")
        info_lines.append("")

        # 检测 OpenCV
        info_lines.append("-" * 60)
        info_lines.append("OpenCV (cv2)")
        try:
            import cv2 as cv_check  # noqa: F401

            available["cv2"] = True
            info_lines.append(f"  ✓ {trans.t('env_installed').format(cv_check.__version__)}")
        except Exception as exc:
            info_lines.append(f"  {trans.t('env_not_installed')}: {exc}")
        info_lines.append("")

        # 检测 NumPy
        info_lines.append("-" * 60)
        info_lines.append("NumPy")
        try:
            import numpy as np_check  # noqa: F401

            available["numpy"] = True
            info_lines.append(f"  ✓ {trans.t('env_installed').format(np_check.__version__)}")
        except Exception as exc:
            info_lines.append(f"  {trans.t('env_not_installed')}: {exc}")
        info_lines.append("")

        # 检测 PyQt6
        info_lines.append("-" * 60)
        info_lines.append("PyQt6")
        try:
            from PyQt6.QtCore import PYQT_VERSION_STR  # noqa: F401

            info_lines.append(f"  ✓ {trans.t('env_installed').format(PYQT_VERSION_STR)}")
        except Exception as exc:
            info_lines.append(f"  {trans.t('env_not_installed')}: {exc}")
        info_lines.append("")

        # StackMFF runtime and the provider actually selected by its probe.
        info_lines.append("-" * 60)
        info_lines.append('StackMFF / AI inference')
        try:
            import onnxruntime
            from fusion_methods import neural_runtime
            from core.multi_focus_fusion import ensure_neural_device_kind
            available["onnxruntime"] = True
            kind = neural_runtime.admitted_kind(ensure_neural_device_kind())
            unadmitted = neural_runtime.unadmitted_accelerator()
            available["device"] = (neural_runtime.display_name(kind)
                                   if neural_runtime.is_accelerator(kind)
                                   else "CPU")
            info_lines.append(f"  ✓ {trans.t('env_installed').format(onnxruntime.__version__)}")
            diagnostics = neural_runtime.device_diagnostics()
            info_lines.append('  Registered runtime providers: ' + ', '.join(diagnostics['providers']))
            native = diagnostics.get('eligible_native', ())
            if native:
                info_lines.append('  Managed engines: ' + ', '.join(neural_runtime.display_name(k) for k in native))
            for index, name, _vendor, _memory in diagnostics['hardware']:
                info_lines.append(f'  Hardware DXGI {index}: {name}')
            if diagnostics['actual'] is not None:
                info_lines.append(f"  Last successful neural inference: {neural_runtime.display_name(diagnostics['actual'])} "
                                  f"({diagnostics['checkpoint']})")
            for reason in diagnostics.get('fallback_reasons', ()):
                info_lines.append('  Engine fallback: ' + reason)
            for error in diagnostics['probe_errors']:
                info_lines.append('  Provider probe: ' + error)
            if neural_runtime.is_accelerator(kind):
                if kind in native:
                    info_lines.append('  ' + trans.t('neural_provider_ready').format(available['device']))
                else:
                    info_lines.append(f"  ✓ {trans.t('env_gpu_accel')} ({available['device']})")
            else:
                if unadmitted:
                    # The provider was verified; say why it is not used
                    # instead of claiming no provider passed.
                    info_lines.append("  ⚠ " + trans.t('env_gpu_memory_unknown').format(
                        device=neural_runtime.display_name(unadmitted)))
                else:
                    info_lines.append(f"  ⚠ {trans.t('env_no_gpu')}")
                info_lines.append(f"  ✓ {trans.t('env_cpu_mode')}")
            info_lines.append('  ' + trans.t('msg_gpu_accel_hint'))
        except Exception as exc:
            available["onnxruntime"] = False
            info_lines.append(f"  {trans.t('env_not_installed')}: {exc}")
            info_lines.append(f"  ✗ {trans.t('env_stackmff_unavailable')}")
        info_lines.append("")
        from core.multi_focus_fusion import ensure_openfocus_gpu_available
        openfocus_gpu = ensure_openfocus_gpu_available()
        info_lines.append('-' * 60)
        info_lines.append('OpenFocus A/B/C / Guided Filter / GFG-FGF / DCT / DTCWT / OpenCL')
        info_lines.append('  OpenCL GPU available; CPU fallback' if openfocus_gpu
                          else '  CPU')
        info_lines.append('')


        # 检测 DTCWT
        info_lines.append("-" * 60)
        info_lines.append("DTCWT (Dual-Tree Complex Wavelet Transform)")
        try:
            import dtcwt  # noqa: F401

            available["dtcwt"] = True
            info_lines.append(f"  ✓ {trans.t('env_installed').format(dtcwt.__version__)}")
        except Exception as exc:
            info_lines.append(f"  {trans.t('env_dtcwt_unavailable')}: {exc}")
        info_lines.append("")

        # 检测 SciPy（DCT 的一致性中值滤波 + 深度图 DCT 度量所需）
        info_lines.append("-" * 60)
        info_lines.append("SciPy / DCT")
        try:
            import scipy  # noqa: F401
            available["scipy"] = True

            info_lines.append(f"  ✓ {trans.t('env_installed').format(scipy.__version__)}")
        except Exception as exc:
            info_lines.append(f"  {trans.t('env_not_installed')}: {exc}")
        info_lines.append("")

        # 总结
        info_lines.append("=" * 60)
        info_lines.append(trans.t('env_summary'))
        info_lines.append("=" * 60)
        info_lines.append(trans.t('env_core_dep'))
        info_lines.append(f"  {trans.t('env_core_desc')}")
        info_lines.append("")
        info_lines.append(trans.t('env_gpu_opt'))
        info_lines.append(f"  {trans.t('env_gpu_desc')}")
        info_lines.append("")
        info_lines.append(trans.t('env_fusion_alg'))
        info_lines.append(f"  {trans.t('env_fusion_desc')}")
        info_lines.append("")

        core = available.get("cv2", False) and available.get("numpy", False)
        methods = [
            ("OpenFocus A/B/C / Guided Filter / GFG-FGF", core,
             "OpenCL GPU available; CPU fallback" if openfocus_gpu else "CPU"),
            ("DCT", core and available.get("scipy", False),
             "OpenCL GPU available; CPU fallback" if openfocus_gpu else "CPU"),
            ("DTCWT", core and available.get("dtcwt", False),
             "OpenCL GPU available; CPU fallback" if openfocus_gpu else "CPU"),
            ("StackMFF V1/V2/V3", core and available.get("onnxruntime", False), available.get("device", "CPU")),
        ]
        summary = [trans.t("env_fusion_alg")]
        for name, ready, device in methods:
            summary.append(f"{name}: {device if ready else trans.t('env_not_installed')}")
        return "\n".join(summary + [""] + info_lines)


_EMAIL_LOCAL = re.compile(r'[A-Za-z0-9_+-]+(?:\.[A-Za-z0-9_+-]+)*')
_DOMAIN_LABEL = r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?'
_EMAIL_DOMAIN = re.compile(rf'{_DOMAIN_LABEL}(?:\.{_DOMAIN_LABEL})+')
_CONTROL_CATEGORIES = {'Cc', 'Cf', 'Cs'}


def _published_emails(value):
    if not isinstance(value, list) or len(value) > 8:
        return
    seen = set()
    for address in value:
        if not isinstance(address, str) or len(address) > 254:
            continue
        local, _, domain = address.partition('@')
        if len(local) > 64 or not _EMAIL_LOCAL.fullmatch(local) or not _EMAIL_DOMAIN.fullmatch(domain):
            continue
        key = address.lower()
        if key not in seen:
            seen.add(key)
            yield address


def _published_link(value, *hosts):
    if not isinstance(value, str) or not value:
        return ''
    if any(char.isspace() or char in '\\<>"`{}|^' or category(char) in _CONTROL_CATEGORIES
           for char in value):
        return ''
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ''
    authority = parsed.netloc.lower().removesuffix(':443')
    if parsed.scheme != 'https' or not parsed.netloc.isascii() or authority not in hosts:
        return ''
    url = QUrl(value, QUrl.ParsingMode.StrictMode)
    if not url.isValid():
        return ''
    normalized = url.adjusted(QUrl.UrlFormattingOption.NormalizePathSegments)
    path = normalized.path(QUrl.ComponentFormattingOption.FullyEncoded)
    if not path or path == '/':
        return ''
    if authority == 'discord.com' and not re.fullmatch(r'/invite/[^/]+', path):
        return ''
    return value


def _discord_qr(invite):
    import cv2
    from PyQt6.QtSvg import QSvgRenderer
    from utils import resource_path

    params = cv2.QRCodeEncoder_Params()
    params.correction_level = cv2.QRCodeEncoder_CORRECT_LEVEL_H
    try:
        encoded = cv2.QRCodeEncoder_create(params).encode(invite)
    except cv2.error:
        return None

    mark = QSvgRenderer(resource_path('assets', 'discord.svg'))
    if not mark.isValid():
        return None

    side = encoded.shape[0]
    image = QImage(encoded.data, side, side, encoded.strides[0],
                   QImage.Format.Format_Grayscale8).copy()
    # Skip OpenCV's two-module border. The shared renderer adds its quiet zone.
    modules = side - 4
    region = QRect(2, 2, modules, modules)

    def overlay(painter, target):
        # Cover only five center modules and keep the mark's 4:3 proportions.
        scale = target.width() / modules
        x = target.x() + target.width() / 2
        y = target.y() + target.height() / 2
        painter.fillRect(QRectF(x - 2.5 * scale, y - 2.5 * scale, 5 * scale, 5 * scale),
                         QColor('#5865F2'))
        mark.render(painter, QRectF(x - 2 * scale, y - 1.5 * scale, 4 * scale, 3 * scale))

    return image, region, overlay


class ContactInfoDialog(QDialog):
    """Show the current contact links and community codes from the account service."""

    _CONTACT_INFO_URL = f'{BASE_URL}/v1/contact-info'
    _MAX_CONTACT_INFO_BYTES = 32 * 1024
    _QR_SIZE = 152

    def __init__(self, parent=None):
        from ._community_layout import add_published_qr, community_body, community_card, community_label, community_qr

        super().__init__(parent)
        self._contact_cards = QGridLayout()
        self._contact_columns = 0
        title = trans.t('dialog_contact_title')
        self.setWindowTitle(title)
        self._contact_scroll, body = community_body(self, title, trans.t('contact_welcome'),
                                                    width=880, height=470)

        self._contact_cards.setSpacing(body.spacing())
        body.addLayout(self._contact_cards)
        body.addStretch()

        contact_card, contact = community_card()
        self._cards = [contact_card]
        contact.addWidget(community_label(trans.t('contact_info_title'), 'section'))
        self._contact_status = community_label(trans.t('contact_details_loading'), 'muted')
        contact.addWidget(self._contact_status)
        self._contact_details = QVBoxLayout()
        self._contact_details.setSpacing(contact.spacing())
        contact.addLayout(self._contact_details)
        contact.addStretch()

        discord_card, self._discord = community_card()
        self._cards.append(discord_card)
        self._discord_heading = community_label(trans.t('contact_discord_title'), 'section')
        self._discord_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._discord.addWidget(self._discord_heading)
        self._discord_qr = community_qr(self._discord, self._discord_heading.text(), 'discord-invite', self._QR_SIZE)
        self._discord_status = community_label('', 'muted')
        self._discord_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._discord.addWidget(self._discord_status)
        self._discord.addStretch()
        self._discord_link = None

        self._requests = CommunityDownloads(self)
        self._contact_request = 0
        for slot, label in (('wechat-group', 'contact_wechat_title'),
                            ('wechat-official-account', 'contact_wechat_official')):
            card, column = community_card()
            self._cards.append(card)
            add_published_qr(self, column, self._requests, trans.t(label), slot, self._QR_SIZE)
            column.addStretch()
        self._contact_reflow = QTimer(self, singleShot=True)
        self._contact_reflow.timeout.connect(self._arrange_contact_cards)
        self._contact_scroll.viewport().installEventFilter(self)
        self._contact_scroll.widget().installEventFilter(self)
        self._arrange_contact_cards()

        # 居中显示
        if parent:
            self.move(
                parent.x() + parent.width() // 2 - self.width() // 2,
                parent.y() + parent.height() // 2 - self.height() // 2,
            )

    def showEvent(self, event):
        super().showEvent(event)
        if event.spontaneous():
            return
        while self._contact_details.count():
            widget = self._contact_details.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()
        self._contact_status.setText(trans.t('contact_details_loading'))
        self._contact_status.show()
        self._discord_heading.setText(trans.t('contact_discord_title'))
        self._discord_qr.setAccessibleName(self._discord_heading.text())
        self._discord_qr.clear()
        self._discord_qr.hide()
        if self._discord_link is not None:
            self._discord.removeWidget(self._discord_link)
            self._discord_link.hide()
            self._discord_link.deleteLater()
            self._discord_link = None
        self._discord_status.setText(trans.t('contact_discord_loading'))
        self._discord_status.show()
        self._contact_request += 1
        request = self._contact_request
        query = urlencode({'lang': trans.current_lang})
        self._requests.get(f'{self._CONTACT_INFO_URL}?{query}', 'application/json',
                           self._MAX_CONTACT_INFO_BYTES,
                           lambda status, data: self._contact_finished(request, status, data))

    def _contact_finished(self, request, status, data):
        from ._community_layout import community_label, community_link

        if request != self._contact_request:
            return
        payload = None
        if status == 200 and data is not None:
            try:
                payload = json.loads(data.decode('utf-8'))
            except (ValueError, RecursionError):
                pass
        if not isinstance(payload, dict):
            self._contact_status.setText(trans.t('contact_details_unavailable'))
            self._discord_status.setText(trans.t('contact_discord_unavailable'))
            return

        emails = list(_published_emails(payload.get('emails')))
        if emails:
            self._contact_details.addWidget(community_label(trans.t('contact_email'), 'muted'))
            for address in emails:
                link = community_link(self, address, f'mailto:{address}')
                self._add_detail(link, address)

        institution = payload.get('institution')
        if (isinstance(institution, str) and 1 <= len(institution) <= 120 and institution.strip()
                and not any(category(char) in _CONTROL_CATEGORIES or char in '\u2028\u2029'
                            for char in institution)):
            self._contact_details.addWidget(community_label(trans.t('contact_institution'), 'muted'))
            label = community_label(institution)
            label.setObjectName('contactInstitution')
            self._add_detail(label, institution)

        github = _published_link(payload.get('github_url'), 'github.com')
        discord = _published_link(payload.get('discord_url'), 'discord.gg', 'discord.com')
        if github or discord:
            self._contact_details.addWidget(community_label(trans.t('contact_community_links'), 'section'))
        if github:
            self._contact_details.addWidget(community_link(self, trans.t('contact_github'), github))
        if discord:
            self._contact_details.addWidget(community_link(self, 'Discord', discord))

        self._show_discord_invite(discord)
        if self._contact_details.count():
            self._contact_status.hide()
        else:
            self._contact_status.setText(trans.t('contact_details_empty'))

    def _show_discord_invite(self, invite):
        from ._community_layout import community_link

        if not invite:
            self._discord_status.setText(trans.t('contact_discord_unpublished'))
            return
        self._discord_link = community_link(self, trans.t('contact_discord_join'), invite)
        self._discord_link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._discord.insertWidget(self._discord.count() - 1, self._discord_link)
        self._discord_link.show()
        code = _discord_qr(invite)
        if code is None:
            self._discord_status.setText(trans.t('contact_discord_qr_unavailable'))
            return
        image, region, overlay = code
        self._discord_qr.set_code(image, source=region, overlay=overlay)
        self._discord_status.hide()

    def _add_detail(self, label, value):
        # Long addresses and institution names must not set the cards' minimum width.
        policy = label.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        label.setSizePolicy(policy)
        label.setToolTip(f'<qt>{escape(value)}</qt>')
        self._contact_details.addWidget(label)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest):
            self._contact_reflow.start()
        return super().eventFilter(watched, event)

    def _arrange_contact_cards(self):
        # Reserve the scrollbar width even before it appears. Reflowing can
        # change the scroll range, but must not make the column count oscillate.
        scroll = self._contact_scroll
        available = scroll.maximumViewportSize().width() - scroll.verticalScrollBar().sizeHint().width()
        minimum_widths = []
        for index, card in enumerate(self._cards):
            minimum = card.minimumSizeHint().width()
            if index:
                # QR labels can be hidden while downloading. Reserve their width.
                margins = card.layout().contentsMargins()
                minimum = max(minimum, self._QR_SIZE + margins.left() + margins.right()
                              + 2 * card.frameWidth())
            minimum_widths.append(minimum)
        columns = 1
        for count in (4, 2):
            width = sum(max(minimum_widths[column::count]) for column in range(count))
            width += (count - 1) * self._contact_cards.horizontalSpacing()
            if width <= available:
                columns = count
                break
        if self._contact_columns == columns:
            return
        self._contact_columns = columns
        for card in self._cards:
            self._contact_cards.removeWidget(card)
        for index, card in enumerate(self._cards):
            row, column = divmod(index, columns)
            self._contact_cards.addWidget(card, row, column)
        for column in range(len(self._cards)):
            self._contact_cards.setColumnStretch(column, int(column < columns))
