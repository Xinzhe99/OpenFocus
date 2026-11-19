# dialogs/about.py
"""About and contact information dialogs."""

import threading

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
)
from constants import DISCORD_INVITE_URL
from dialogs._community_download import CommunityDownloads
from locales import trans
from utils import get_monospace_font_family


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


class ContactInfoDialog(QDialog):
    """Show contact links and current WeChat codes from the account service."""

    def __init__(self, parent=None):
        from ._community_layout import (add_published_qr, community_body, community_card,
                                        community_label, community_link)

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
        contact.addWidget(community_label(trans.t('contact_email'), 'muted'))
        for address in ('xiexinzhe@zju.edu.cn', 'xiexinzheyyy@163.com', 'xiexinzhe123@gmail.com'):
            contact.addWidget(community_link(self, address, f'mailto:{address}'))
        contact.addSpacing(8)
        contact.addWidget(community_label(trans.t('contact_institution'), 'muted'))
        contact.addWidget(community_label(trans.t('contact_zju')))
        contact.addSpacing(8)
        contact.addWidget(community_link(self, trans.t('contact_github'), 'https://github.com/Xinzhe99/OpenFocus'))
        contact.addWidget(community_link(self, 'Discord', DISCORD_INVITE_URL))
        contact.addStretch()

        requests = CommunityDownloads(self)
        for slot, label in (('wechat-group', 'contact_wechat_title'),
                            ('wechat-official-account', 'contact_wechat_official')):
            card, column = community_card()
            self._contact_cards.addWidget(card, 2)
            add_published_qr(self, column, requests, trans.t(label), slot, 192)
            column.addStretch()

        # 居中显示
        if parent:
            self.move(
                parent.x() + parent.width() // 2 - self.width() // 2,
                parent.y() + parent.height() // 2 - self.height() // 2,
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        direction = (QHBoxLayout.Direction.LeftToRight if event.size().width() >= 760
                     else QHBoxLayout.Direction.TopToBottom)
        self._contact_cards.setDirection(direction)
