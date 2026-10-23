"""Visual + behavioural check for the update surface (05-updates).

No network, download or installer is exercised. It checks the restart
protection gate, the hidden-version rule, the newer-build rule and the banner
layout in light/dark and a narrow window.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path('F:/Working/OpenFocus')
WORK = Path(os.environ.get('OPENFOCUS_VISUAL_DIR')
            or (Path(os.environ.get('TEMP', '/tmp')) / 'openfocus_update_visual'))
WORK.mkdir(parents=True, exist_ok=True)

os.environ['APPDATA'] = str(WORK / 'profile')
os.environ['QT_QPA_PLATFORM'] = 'windows'
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QSettings, QTimer, QEventLoop
from PyQt6.QtWidgets import QApplication, QPushButton
import numpy as np

app = QApplication([])
from utils import settings_store
settings_store._settings_singleton = QSettings(str(WORK / 'audit.ini'),
                                               QSettings.Format.IniFormat)
settings_store._settings_singleton.setValue('ui/language', 'zh')
from locales import trans
trans.set_language('zh')
from main import OpenFocus
win = OpenFocus()
win.resize(1280, 800)
win.apply_theme('light')

out = WORK / 'evidence'
out.mkdir(exist_ok=True)
report = {'platform': 'Windows native Qt', 'screens': [], 'observations': []}


def settle(ms=200):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    app.processEvents()


def capture(widget, name):
    widget.show()
    settle()
    path = out / (name + '.png')
    if not widget.grab().save(str(path)):
        raise RuntimeError('capture failed: ' + name)
    report['screens'].append({
        'name': name, 'size': [widget.width(), widget.height()],
        'buttons': [b.text() for b in widget.findChildren(QPushButton)
                    if b.isVisible()],
    })
    print(name, widget.width(), widget.height(), flush=True)


win.show()
win.raw_images = [np.full((120, 160, 3), 90, np.uint8) for _ in range(2)]
win.base_images = list(win.raw_images)
win.image_filenames = ['u1.png', 'u2.png']
win.stack_images = [None] * 2
win.transform_manager.reload_image_stack(initial_index=0)

# Banner presentation (no network involved).
win.update_banner.show_for('v9.9.0', '1.67.0', ready=False)
settle()
capture(win, 'update_banner_zh')
report['observations'].append({
    'banner_tag': win.update_banner.tag,
    'dismiss_tooltip': win.update_banner.btn_dismiss.toolTip(),
})

# Hiding the version must survive the READY transition. The state callback is
# the real path: it used to call show_for directly and bypass the hidden tag.
win._on_banner_dismissed('v9.9.0')
win.update_banner.hide_banner()
win._on_update_state('ready', 'v9.9.0', 100)
settle()
report['observations'].append({
    'hidden_tag': win._banner_dismissed_tag(),
    'banner_visible_after_ready': win.update_banner.isVisible(),
})

# A manual check re-arms the banner for that version.
win._set_banner_dismissed_tag('')
win.update_banner.show_for('v9.9.0', '1.67.0', ready=True)
settle()
capture(win, 'update_banner_ready_zh')

# The protection gate refuses when no snapshot can be written.
from utils import recovery
original_write = recovery.write_snapshot
warnings = []
import main as main_module
original_warning = main_module.show_warning_box
main_module.show_warning_box = lambda *a, **k: warnings.append(a)
recovery.write_snapshot = lambda _w: False
try:
    result = win.prepare_for_update_restart()
finally:
    recovery.write_snapshot = original_write
    main_module.show_warning_box = original_warning
report['observations'].append({
    'restart_allowed_without_snapshot': bool(result),
    'warning_shown': bool(warnings),
})

# Version rule.
from constants import APP_VERSION
from controllers.update_manager import UpdateManager
report['observations'].append({
    'newer_ok': UpdateManager._is_newer_than_current('v999.0'),
    'same_version_rejected': not UpdateManager._is_newer_than_current(APP_VERSION),
    'older_rejected': not UpdateManager._is_newer_than_current('v0.0.1'),
})

# Dark + German + narrow.
win.apply_theme('dark')
trans.set_language('de')
win.update_ui_text()
win.update_banner.show_for('v9.9.0', '1.67.0', ready=True)
win.resize(960, 640)
settle(300)
capture(win, 'update_banner_dark_de_960')

(WORK / 'report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
win.hide()
win.status_timer.stop()
win.update_manager.shutdown()
print('REPORT', WORK / 'report.json', flush=True)
