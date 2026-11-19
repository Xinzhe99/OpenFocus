# dialogs/about.py
"""About and contact information dialogs."""

import json
import re
import threading
from html import escape
from unicodedata import category
from urllib.parse import urlencode, urlsplit

from PyQt6.QtCore import QTimer, Qt, QUrl
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QTextEdit,
)
from dialogs._community_download import CommunityDownloads
from locales import trans
from utils import get_monospace_font_family
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
        self.resize(600, 500)

        # Styling comes from the generated application stylesheet.


        layout = QVBoxLayout(self)

        # 信息显示区
        self.text_edit = QTextEdit()
        self.text_edit.setToolTip(trans.t('env_subtitle'))
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 10))
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
        info_lines.append('OpenFocus A/B/C / Guided Filter / GFG-FGF / DCT / OpenCL')
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
            ("DTCWT", core and available.get("dtcwt", False), "CPU"),
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


class ContactInfoDialog(QDialog):
    """Show contact links and current WeChat codes from the account service."""

    _CONTACT_INFO_URL = f'{BASE_URL}/v1/contact-info'
    _MAX_CONTACT_INFO_BYTES = 32 * 1024

    def __init__(self, parent=None):
        from ._community_layout import add_published_qr, community_body, community_card, community_label

        super().__init__(parent)
        self._contact_cards = QHBoxLayout()
        title = trans.t('dialog_contact_title')
        self.setWindowTitle(title)
        body = community_body(self, title, trans.t('contact_welcome'), width=820, height=410)

        self._contact_cards.setSpacing(body.spacing())
        body.addLayout(self._contact_cards)
        body.addStretch()

        contact_card, contact = community_card()
        self._contact_cards.addWidget(contact_card, 3)
        contact.addWidget(community_label(trans.t('contact_info_title'), 'section'))
        self._contact_status = community_label(trans.t('contact_details_loading'), 'muted')
        contact.addWidget(self._contact_status)
        self._contact_details = QVBoxLayout()
        self._contact_details.setSpacing(contact.spacing())
        contact.addLayout(self._contact_details)
        contact.addStretch()

        self._requests = CommunityDownloads(self)
        self._contact_request = 0
        for slot, label in (('wechat-group', 'contact_wechat_title'),
                            ('wechat-official-account', 'contact_wechat_official')):
            card, column = community_card()
            self._contact_cards.addWidget(card, 2)
            add_published_qr(self, column, self._requests, trans.t(label), slot, 192)
            column.addStretch()

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
        if github:
            self._contact_details.addWidget(community_link(self, trans.t('contact_github'), github))
        discord = _published_link(payload.get('discord_url'), 'discord.gg', 'discord.com')
        if discord:
            self._contact_details.addWidget(community_link(self, 'Discord', discord))

        if self._contact_details.count():
            self._contact_status.hide()
        else:
            self._contact_status.setText(trans.t('contact_details_empty'))

    def _add_detail(self, label, value):
        # Long addresses and institution names must not set the cards' minimum width.
        policy = label.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        label.setSizePolicy(policy)
        label.setToolTip(f'<qt>{escape(value)}</qt>')
        self._contact_details.addWidget(label)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        direction = (QHBoxLayout.Direction.LeftToRight if event.size().width() >= 760
                     else QHBoxLayout.Direction.TopToBottom)
        self._contact_cards.setDirection(direction)
