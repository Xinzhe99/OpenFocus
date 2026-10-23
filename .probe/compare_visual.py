"""Visual + behavioural check for the comparison window (03-method-compare).

Checks that an excluded frame is not compared, that the view mode only changes
display, and that a full-frame candidate can be adopted as a new protected
output. Captures 1280x820, 960x640 and a dark German variant.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('F:/Working/OpenFocus')
WORK = Path(os.environ.get('OPENFOCUS_VISUAL_DIR')
            or (Path(os.environ.get('TEMP', '/tmp')) / 'openfocus_compare_visual'))
WORK.mkdir(parents=True, exist_ok=True)

os.environ['APPDATA'] = str(WORK / 'profile')
os.environ['QT_QPA_PLATFORM'] = 'windows'
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QSettings, QTimer, QEventLoop
from PyQt6.QtWidgets import QApplication, QPushButton
import cv2
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
win.use_gpu = False

out = WORK / 'evidence'
out.mkdir(exist_ok=True)
report = {'platform': 'Windows native Qt', 'screens': [], 'observations': []}


def settle(ms=200):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    app.processEvents()


def wait_until(predicate, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not predicate():
        settle(100)
    return predicate()


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
yy, xx = np.mgrid[:480, :720]
pattern = ((xx // 8 + yy // 8) % 2 * 150 + 50).astype(np.uint8)
base = np.dstack([pattern, ((xx / 720) * 230).astype(np.uint8),
                  ((yy / 480) * 230).astype(np.uint8)])
cv2.putText(base, 'COMPARE 01 02 03', (45, 245), cv2.FONT_HERSHEY_SIMPLEX,
            1.4, (245, 245, 245), 3)
blur = cv2.GaussianBlur(base, (0, 0), 5)
frames = []
for i in range(3):
    frame = blur.copy()
    frame[:, i * 240:(i + 1) * 240] = base[:, i * 240:(i + 1) * 240]
    frames.append(frame)
win.raw_images = frames
win.base_images = list(frames)
win.image_filenames = ['cmp_01.png', 'cmp_02.png', 'cmp_03.png']
win.stack_images = [None] * 3
win.current_scale_factor = 1.0
win.excluded_frames = {1}
win.transform_manager.reload_image_stack(initial_index=0)

from dialogs.compare import CompareDialog
dlg = CompareDialog(win)
dlg.show()
settle()
# The dialog auto-requests a preview on open; finish or cancel that first so
# the explicit run below is not mistaken for a cancel request.
dlg.align_combo.setCurrentIndex(dlg.align_combo.findData("none"))
dlg._preview_timer.stop()
if dlg._worker is not None:
    dlg._worker.cancel()
    wait_until(lambda: dlg._worker is None, 60)
dlg._results.clear()
dlg._errors.clear()
settle()
report['observations'].append({
    'frames_used': len(dlg._raw_frames),
    'total_frames': dlg._total_frames,
    'names': list(dlg._source_names),
    'summary': dlg.source_summary.text(),
})
assert len(dlg._raw_frames) == 2, "excluded frame must not be compared"

# Only DCT, full-frame computation so adoption is allowed.
for method, box in dlg._checks.items():
    box.setChecked(method == 'dct')
dlg.mode_combo.setCurrentIndex(dlg.mode_combo.findData(False))
dlg._preview_timer.stop()
dlg.run_comparison()
ok = wait_until(lambda: dlg._worker is None and dlg._results, 90)
report['observations'].append({
    'compare_done': bool(ok),
    'results': list(dlg._results),
    'full_results': list(dlg._full_results),
    'result_ok': {m: bool(r) for m, r in dlg._results.items()},
    'status': dlg.status_label.text(),
})
capture(dlg, 'compare_ready_zh')

# View mode must not recompute.
before_results = dict(dlg._results)
dlg.view_combo.setCurrentIndex(dlg.view_combo.findData('full'))
settle()
report['observations'].append({
    'view_mode': dlg._canvases['dct'].view_mode(),
    'can_show_full': dlg._canvases['dct'].can_show_full(),
    'no_recompute': list(before_results) == list(dlg._results),
})
capture(dlg, 'compare_full_view_zh')

# Selecting a candidate enables adoption.
dlg._select_method('dct')
settle()
report['observations'].append({
    'selected': dlg._selected_method,
    'adopt_params_enabled': dlg.btn_adopt_params.isEnabled(),
    'adopt_output_enabled': dlg.btn_adopt_output.isEnabled(),
})
outputs_before = win.output_list.count()
dlg._adopt_output()
settle()
report['observations'].append({
    'outputs_after': win.output_list.count(),
    'added': win.output_list.count() - outputs_before,
    'protected_after': len(win.output_manager.protected_records()),
})
capture(win, 'main_after_adopt_zh')

# Narrow + dark + German layout.
dlg2 = CompareDialog(win)
dlg2.resize(960, 640)
dlg2.show()
settle()
win.apply_theme('dark')
trans.set_language('de')
win.update_ui_text()
capture(dlg2, 'compare_small_dark_de')
dlg2.reject()

(WORK / 'report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
win.hide()
win.status_timer.stop()
win.update_manager.shutdown()
print('REPORT', WORK / 'report.json', flush=True)
