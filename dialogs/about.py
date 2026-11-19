# dialogs/about.py
"""About and contact information dialogs."""

from http.client import HTTPException
import threading
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPErrorProcessor, build_opener

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QWidget,
    QTextEdit,
    QPushButton,
)
from constants import DISCORD_INVITE_URL
from locales import trans
from utils import get_monospace_font_family, resource_path
from ui.styles import PRIMARY_BLUE


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
        info_lines.append('OpenFocus A/B/C / Guided Filter / GFG-FGF / OpenCL')
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
            ("OpenFocus A/B/C / Guided Filter / GFG-FGF", core, "OpenCL GPU available; CPU fallback" if openfocus_gpu else "CPU"),
            ("DCT", core and available.get("scipy", False), "CPU"),
            ("DTCWT", core and available.get("dtcwt", False), "CPU"),
            ("StackMFF V1/V2/V3", core and available.get("onnxruntime", False), available.get("device", "CPU")),
        ]
        summary = [trans.t("env_fusion_alg")]
        for name, ready, device in methods:
            summary.append(f"{name}: {device if ready else trans.t('env_not_installed')}")
        return "\n".join(summary + [""] + info_lines)


class _InviteResponses(HTTPErrorProcessor):
    # Inspect redirects before urllib follows them or drains their bodies.
    def http_response(self, request, response):
        return response

    https_response = http_response


def _origin(url):
    parts = urlsplit(url)
    port = parts.port if parts.port is not None else (443 if parts.scheme == 'https' else 80)
    return parts.scheme, parts.hostname, port


class _WechatInviteDownload(threading.Thread):
    """Own the network connection entirely off the GUI thread, including close."""

    def __init__(self, url, timeout_ms, max_bytes):
        super().__init__(name='wechat-invite-download', daemon=True)
        self.url = url
        self.deadline = time.monotonic() + timeout_ms / 1000
        self.max_bytes = max_bytes
        self.cancelled = threading.Event()
        self.png = None

    def run(self):
        try:
            opener = build_opener(_InviteResponses())
            url = self.url
            for _ in range(11):  # Initial request plus at most ten redirects.
                remaining = self.deadline - time.monotonic()
                if self.cancelled.is_set() or remaining <= 0:
                    return
                with opener.open(url, timeout=remaining) as reply:
                    if reply.status == 200:
                        self._read_png(reply)
                        return
                    location = reply.headers.get('Location')
                    if reply.status not in (301, 302, 303, 307, 308) or not location:
                        return
                    url = urljoin(url, location)
                    if _origin(url) != _origin(self.url):
                        return
        except (OSError, HTTPException, ValueError):
            pass  # The dialog already displays the offline invite.

    def _read_png(self, reply):
        data = bytearray()
        while not self.cancelled.is_set() and time.monotonic() < self.deadline:
            chunk = reply.read1(self.max_bytes + 1 - len(data))
            if not chunk:
                if not reply.length and time.monotonic() < self.deadline:
                    self.png = bytes(data)
                return
            data.extend(chunk)
            if len(data) > self.max_bytes:
                return


class ContactInfoDialog(QDialog):
    """Show contact links and refresh the WeChat invite from the website."""

    _WECHAT_URL = "https://openfocus.top/wechat-group.png"
    _WECHAT_TIMEOUT_MS = 4000
    _WECHAT_MAX_BYTES = 512 * 1024

    def __init__(self, parent=None):
        # Link colour must follow the theme: a hard-coded white link was
        # invisible on the white dialog (the GitHub URL simply vanished).
        link = "#0969da" if getattr(parent, "ui_theme", "dark") == "light" \
            else "#6cb6ff"
        contact_text = f"""
        <style>a {{ color: {link}; }}</style>
        <h3>{trans.t('contact_info_title')}</h3>
        
    <p>{trans.t('contact_email')}: xiexinzhe@zju.edu.cn</p>
    <p>{trans.t('contact_institution')}: {trans.t('contact_zju')}</p>
    <p>{trans.t('contact_github')}: <a href="https://github.com/Xinzhe99/OpenFocus">https://github.com/Xinzhe99/OpenFocus</a></p>
    <p>Discord: <a href="{DISCORD_INVITE_URL}">{DISCORD_INVITE_URL}</a></p>
    <p>{trans.t('contact_welcome')}</p>"""
        
        super().__init__(parent)
        self.setWindowTitle(trans.t('dialog_contact_title'))
        self.resize(500, 620)

        layout = QVBoxLayout(self)

        from PyQt6.QtWidgets import QTextBrowser
        from PyQt6.QtGui import QTextOption

        # 创建可滚动的文本浏览器
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(contact_text)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.addWidget(self.text_browser)

        self._wechat_qr = QLabel()
        self._wechat_qr.setAccessibleName(trans.t('contact_wechat_group'))
        self._wechat_qr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._wechat_qr.setFixedSize(320, 324)
        self._wechat_qr.setStyleSheet('background: white;')
        layout.addWidget(self._wechat_qr, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._wechat_status = QLabel(trans.t('contact_wechat_loading'))
        self._wechat_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._wechat_status.setWordWrap(True)
        layout.addWidget(self._wechat_status)

        group_link = QLabel(f'<a style="color: {link};" href="{self._WECHAT_URL}">'
                            f'{trans.t("contact_wechat_group")}</a>')
        group_link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group_link.setOpenExternalLinks(True)
        layout.addWidget(group_link)

        bundled = QPixmap(resource_path('assets', 'readme', 'wechat-group.png'))
        if bundled.isNull():
            self._wechat_qr.hide()
        else:
            self._show_wechat_qr(bundled)

        self._wechat_refresh = _WechatInviteDownload(
            self._WECHAT_URL, self._WECHAT_TIMEOUT_MS, self._WECHAT_MAX_BYTES)
        self.destroyed.connect(self._wechat_refresh.cancelled.set)
        self._wechat_timer = QTimer(self)
        self._wechat_timer.timeout.connect(self._poll_wechat_invite)
        self._wechat_refresh.start()
        self._wechat_timer.start(50)

        # 关闭按钮
        close_btn = QPushButton(trans.t('btn_close'))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        # 居中显示
        if parent:
            self.move(
                parent.x() + parent.width() // 2 - self.width() // 2,
                parent.y() + parent.height() // 2 - self.height() // 2,
            )

    def _show_wechat_qr(self, original):
        ratio = self.devicePixelRatioF()
        pixmap = original.scaled(
            round(self._wechat_qr.width() * ratio), round(self._wechat_qr.height() * ratio),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation,
        )
        pixmap.setDevicePixelRatio(ratio)
        self._wechat_qr.setPixmap(pixmap)
        self._wechat_qr.show()

    def _poll_wechat_invite(self):
        refresh = self._wechat_refresh
        pending = refresh.is_alive()
        if pending and time.monotonic() < refresh.deadline:
            return
        self._wechat_timer.stop()
        self._wechat_refresh = None
        image = QPixmap()
        if pending:
            refresh.cancelled.set()
        elif refresh.png:
            image.loadFromData(refresh.png, 'PNG')
        if image.isNull():
            has_fallback = not self._wechat_qr.isHidden()
            key = 'contact_wechat_bundled' if has_fallback else 'contact_wechat_unavailable'
            self._wechat_status.setText(trans.t(key))
        else:
            self._show_wechat_qr(image)
            self._wechat_status.hide()

    def done(self, result):
        self._wechat_timer.stop()
        if self._wechat_refresh is not None:
            self._wechat_refresh.cancelled.set()
            self._wechat_refresh = None
        super().done(result)


class _SupportCode(QLabel):
    """Scale the original payment image for the available logical size."""

    def __init__(self, filename):
        super().__init__()
        self._original = QPixmap(resource_path('assets', 'readme', filename))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(130, 160)
        self.setStyleSheet('background: white;')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        ratio = self.devicePixelRatioF()
        pixmap = self._original.scaled(
            round(self.width() * ratio), round(self.height() * ratio),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        pixmap.setDevicePixelRatio(ratio)
        self.setPixmap(pixmap)


class SupportUsDialog(QDialog):
    """Display the project's payment codes and acknowledgment information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('support_title'))
        self.resize(640, 600)
        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        body = QVBoxLayout(content)
        message = QLabel(trans.t('support_message'))
        message.setWordWrap(True)
        body.addWidget(message)
        codes = QHBoxLayout()
        for title, filename in (('Alipay', 'alipay-coffee.jpg'),
                                ('PayPal', 'paypal-coffee.png')):
            column = QVBoxLayout()
            label = QLabel(title)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(label)
            column.addWidget(_SupportCode(filename), 1)
            codes.addLayout(column, 1)
        body.addLayout(codes, 1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        close_btn = QPushButton(trans.t('btn_close'))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
