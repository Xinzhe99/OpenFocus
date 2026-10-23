"""Visual + behavioural check for the main workflow (08-main-workflow).

Checks participation counts, the ROI dialog naming/output summary, and the
renamed compare entry. Captures the loaded window, the ROI dialog and a
narrow window.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path('F:/Working/OpenFocus')
WORK = Path(os.environ.get('OPENFOCUS_VISUAL_DIR')
            or (Path(os.environ.get('TEMP', '/tmp')) / 'openfocus_workflow_visual'))
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


def settle(ms=220):
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
yy, xx = np.mgrid[:480, :720]
pattern = ((xx // 8 + yy // 8) % 2 * 150 + 50).astype(np.uint8)
base = np.dstack([pattern, ((xx / 720) * 230).astype(np.uint8),
                  ((yy / 480) * 230).astype(np.uint8)])
cv2.putText(base, 'WORKFLOW 01 02 03', (40, 245), cv2.FONT_HERSHEY_SIMPLEX,
            1.3, (245, 245, 245), 3)
blur = cv2.GaussianBlur(base, (0, 0), 5)
frames = []
for i in range(3):
    frame = blur.copy()
    frame[:, i * 240:(i + 1) * 240] = base[:, i * 240:(i + 1) * 240]
    frames.append(frame)
win.raw_images = frames
win.base_images = list(frames)
win.image_filenames = ['wf_01.png', 'wf_02.png', 'wf_03.png']
win.stack_images = [None] * 3
win.current_scale_factor = 1.0
win.transform_manager.reload_image_stack(initial_index=0)
settle()

before = win.source_images_label.text()
item = win.file_list.item(2)
from PyQt6.QtCore import Qt
item.setCheckState(Qt.CheckState.Unchecked)
settle()
after = win.source_images_label.text()
report['observations'].append({
    'count_before': before,
    'count_after': after,
    'included_indices': win.included_indices(),
    'effective_frames': len(win.effective_stack()[0]),
})
capture(win, 'main_participation_zh')

# ROI dialog: named frames and an output description.
from dialogs.roi import ROIRenderOptionsDialog
roi = ROIRenderOptionsDialog(
    3, win, frame_names=['wf_01.png', 'wf_02.png', 'wf_03.png'],
    roi_size=(320, 240), output_size=(720, 480))
roi.show()
settle()
roi.rb_paste.setChecked(True)
settle()
capture(roi, 'roi_named_zh')
report['observations'].append({
    'combo_labels': [roi.combo_frames.itemText(i)
                     for i in range(roi.combo_frames.count())],
    'output_text': roi.lbl_output.text(),
    'paste_index': roi.combo_frames.currentData(),
})
roi.reject()

# Renamed compare entry.
report['observations'].append({
    'compare_all_button': win.btn_compare_all.text(),
    'compare_all_hint': win.btn_compare_all.toolTip(),
})
capture(win, 'main_compare_entry_zh')

win.resize(960, 640)
settle(300)
capture(win, 'main_960_zh')

(WORK / 'report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
win.hide()
win.status_timer.stop()
win.update_manager.shutdown()
print('REPORT', WORK / 'report.json', flush=True)
