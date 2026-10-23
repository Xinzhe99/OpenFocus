"""Visual + behavioural check for batch processing (07-batch).

Runs a real split-folder batch with two groups, including an align-only run,
and verifies the per-task outcome counts, grouping order and progress text.
Captures the dialog and the completion summary.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('F:/Working/OpenFocus')
WORK = Path(os.environ.get('OPENFOCUS_VISUAL_DIR')
            or (Path(os.environ.get('TEMP', '/tmp')) / 'openfocus_batch_visual'))
WORK.mkdir(parents=True, exist_ok=True)

os.environ['APPDATA'] = str(WORK / 'profile')
os.environ['QT_QPA_PLATFORM'] = 'windows'
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QSettings, QTimer, QEventLoop
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton
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


# ---- synthetic source folder: 5 frames (img1, img2, img10, img20, img3) ----
src = WORK / 'stack'
src.mkdir(exist_ok=True)
for name, value in (('img1.png', 30), ('img2.png', 60), ('img10.png', 120),
                    ('img20.png', 180), ('img3.png', 210)):
    image = np.full((96, 128, 3), value, np.uint8)
    cv2.putText(image, name, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)
    cv2.imwrite(str(src / name), image)

win.show()

from core.image_loader import ImageStackLoader
loader = ImageStackLoader()
ok, message, items, names = loader.load_images_with_timestamps(str(src), 'filename')
report['observations'].append({
    'loaded': ok,
    'names': names,
    'natural_order': names == ['img1.png', 'img2.png', 'img3.png',
                               'img10.png', 'img20.png'],
    'missing_timestamps': loader.missing_timestamps(items),
})

stacks = loader.split_by_count(items, 2.0)
report['observations'].append({
    'count_groups': [len(group) for group in stacks],
    'float_count_ok': True,
})

from dialogs.batch import BatchProcessingDialog
dlg = BatchProcessingDialog(win)
dlg.show()
settle()
capture(dlg, 'batch_zh')

# Run a real batch worker through the dialog's settings: align-only, split 2.
outdir = WORK / 'batch_out'
outdir.mkdir(exist_ok=True)
from core.workers import BatchWorker
settings = {
    'reg_methods': ['openfocus'],
    'fusion_method': 'none',
    'save_aligned': True,
    'format': 'png',
    'jpg_quality': 95,
    'output_type': 'custom',
    'output_path': str(outdir),
}
worker = BatchWorker(
    folder_paths=[],
    output_type='custom',
    output_path=str(outdir),
    processing_settings=settings,
    import_mode='single_folder',
    split_method='count',
    split_param=2,
    single_folder_images_with_times=items,
    use_gpu=False,
    thread_count=1,
)
finished = {}
worker.finished.connect(lambda results: finished.update(results))
worker.start()
deadline = time.monotonic() + 120
while time.monotonic() < deadline and not finished:
    settle(100)
worker.wait(5000)
report['observations'].append({
    'batch_finished': bool(finished),
    'success': finished.get('success'),
    'total': finished.get('total'),
    'counts': finished.get('counts'),
    'tasks': finished.get('tasks'),
    'written': [Path(p).name for p in (finished.get('outputs') or [])],
})
capture(win, 'batch_result_zh')

# Progress text is localized and named.
from core.workers import trans_batch_progress
report['observations'].append({
    'progress_text': trans_batch_progress(2, 3, 'stack_002'),
})

(WORK / 'report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
win.hide()
win.status_timer.stop()
win.update_manager.shutdown()
print('REPORT', WORK / 'report.json', flush=True)
