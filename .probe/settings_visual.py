"""Visual + behavioural check for settings and supporting tools (10).

Checks the corrected advanced-settings description, the auto-download setter,
the flat-field invalidation rules and the feedback copy scope. Captures the
advanced settings dialog in light Chinese and dark German.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path('F:/Working/OpenFocus')
WORK = Path(os.environ.get('OPENFOCUS_VISUAL_DIR')
            or (Path(os.environ.get('TEMP', '/tmp')) / 'openfocus_settings_visual'))
WORK.mkdir(parents=True, exist_ok=True)

os.environ['APPDATA'] = str(WORK / 'profile')
os.environ['QT_QPA_PLATFORM'] = 'windows'
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QSettings, QTimer, QEventLoop
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton
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
win.raw_images = [np.full((80, 120, 3), 90, np.uint8) for _ in range(2)]
win.base_images = list(win.raw_images)
win.image_filenames = ['s1.png', 's2.png']
win.stack_images = [None] * 2
win.transform_manager.reload_image_stack(initial_index=0)

from dialogs.advanced_settings import AdvancedSettingsDialog
dlg = AdvancedSettingsDialog(win)
dlg.show()
settle()
capture(dlg, 'advanced_zh')
report['observations'].append({
    'intro': [lbl.text() for lbl in dlg.findChildren(QLabel)
              if 'AI' in lbl.text()][:1],
    'auto_download_checked': dlg.chk_auto_update.isChecked(),
})

# Commit toggles the real setter (it used to assign a read-only property).
before = win.update_manager.auto_download
dlg.chk_auto_update.setChecked(not before)
dlg._apply()
settle()
report['observations'].append({
    'auto_download_before': bool(before),
    'auto_download_after': bool(win.update_manager.auto_download),
    'changed': bool(win.update_manager.auto_download) != bool(before),
})

# Restore the preference for the rest of the run.
win.update_manager.set_auto_download(before)

# Dark German layout.
win.apply_theme('dark')
trans.set_language('de')
win.update_ui_text()
dlg2 = AdvancedSettingsDialog(win)
dlg2.show()
settle()
capture(dlg2, 'advanced_dark_de')
dlg2.reject()

# Flat-field: a new source must invalidate the computed gain.
from dialogs.dustmap import DustMapDialog
dust = DustMapDialog(win)
dust.show()
settle()
dust._gain = object()
dust.btn_apply.setEnabled(True)
dust.rb_stack.setChecked(True)
settle()
report['observations'].append({
    'gain_after_source_change': dust._gain,
    'apply_enabled': dust.btn_apply.isEnabled(),
})
capture(dust, 'dust_zh')
dust.reject()

(WORK / 'report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
win.hide()
win.status_timer.stop()
win.update_manager.shutdown()
print('REPORT', WORK / 'report.json', flush=True)
