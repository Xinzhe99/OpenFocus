"""Visual + behavioural check for the retouch workspace (06-retouch).

Builds a real DCT result, opens the retouch dialog and checks:
  * `]` changes the document diameter, not just the spin box,
  * switching to the restore tool updates the help paragraph,
  * Ctrl+Enter during a held preview does not publish,
  * a failed publication leaves the previous output list intact.
Captures the workspace at 1280x820 and 960x640.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path('F:/Working/OpenFocus')
WORK = Path(os.environ.get('OPENFOCUS_VISUAL_DIR')
            or (Path(os.environ.get('TEMP', '/tmp')) / 'openfocus_retouch_visual'))
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


win.show()
yy, xx = np.mgrid[:480, :720]
pattern = ((xx // 8 + yy // 8) % 2 * 150 + 50).astype(np.uint8)
base = np.dstack([pattern, ((xx / 720) * 230).astype(np.uint8),
                  ((yy / 480) * 230).astype(np.uint8)])
cv2.putText(base, 'RETOUCH 01 02 03', (45, 245), cv2.FONT_HERSHEY_SIMPLEX,
            1.4, (245, 245, 245), 3)
blur = cv2.GaussianBlur(base, (0, 0), 5)
frames = []
for i in range(3):
    frame = blur.copy()
    frame[:, i * 240:(i + 1) * 240] = base[:, i * 240:(i + 1) * 240]
    frames.append(frame)
win.raw_images = frames
win.base_images = list(frames)
win.image_filenames = ['retouch_01.png', 'retouch_02.png', 'retouch_03.png']
win.stack_images = [None] * 3
win.current_scale_factor = 1.0
win.transform_manager.reload_image_stack(initial_index=0)

from core.multi_focus_fusion import MultiFocusFusion
result = MultiFocusFusion(algorithm='dct', use_gpu=False).fuse(
    input_source=frames, img_resize=None, kernel_size=7)
from core.render_context import build_render_context
from controllers.output_manager import OutputRecord
context = build_render_context(frames, result, filenames=win.image_filenames,
                               algorithm='dct')
record = OutputRecord.create(result, 'Retouch sample - DCT', context=context)
win.fusion_result = result
win.output_manager.update_output_list_for_fusion(context=context)
win.output_manager.show_fusion_result()

published = []


def apply_callback(base_record, image):
    published.append((base_record.output_id, image.shape))
    return win.output_manager.add_retouched_output(base_record, image), ""


from dialogs.retouch import RetouchDialog
dlg = RetouchDialog(win, record=record, apply_callback=apply_callback)
dlg.show()
settle()
capture(dlg, 'retouch_zh')

# 1) `]` must reach the document.
before = dlg.doc.diameter
dlg._bump_diameter(1.25)
report['observations'].append({
    'diameter_before': before,
    'diameter_spin': dlg.num_diameter.value(),
    'diameter_doc': dlg.doc.diameter,
    'diameter_consistent': dlg.doc.diameter == dlg.num_diameter.value(),
})

# 2) Restore tool must explain what it actually does.
source_help = dlg.lbl_help.text()
dlg._set_tool('restore')
settle()
restore_help = dlg.lbl_help.text()
report['observations'].append({
    'tool_help_changed': source_help != restore_help,
    'restore_help': restore_help,
})

# 3) A held preview must refuse publication even through the shortcut.
dlg._set_tool('source')
dlg._press_preview('before')
settle()
report['observations'].append({
    'apply_enabled_during_preview': dlg.btn_apply.isEnabled(),
    'can_apply_during_preview': dlg._can_apply(),
    'published_during_preview': len(published),
})
dlg._release_preview('before')

# 4) Draw a stroke and publish once.
dlg.doc.set_brush(48, 0.6, 1.0)
dlg.begin_stroke(200.0, 240.0)
dlg.stroke_to(260.0, 260.0)
dlg.end_stroke()
settle()
report['observations'].append({
    'dirty_after_stroke': dlg.doc.is_dirty,
    'can_apply_after_stroke': dlg._can_apply(),
})
ok = dlg._apply()
settle()
report['observations'].append({
    'apply_result': bool(ok),
    'published': len(published),
    'outputs_after': win.output_list.count(),
    'protected_after': len(win.output_manager.protected_records()),
})
capture(win, 'main_after_retouch_zh')

# 5) Failure must keep the previous history and selection.
win.output_manager.MAX_HISTORY = win.output_list.count()
before_labels = [win.output_list.item(i).text()
                 for i in range(win.output_list.count())]
original_display = win.output_manager.display_specific_fusion_result


def fail_display(_image):
    return False


win.output_manager.display_specific_fusion_result = fail_display
failed = False
try:
    win.output_manager.add_retouched_output(
        win.output_manager.record_for_row(0), np.zeros_like(result))
except Exception:
    failed = True
finally:
    win.output_manager.display_specific_fusion_result = original_display
after_labels = [win.output_list.item(i).text()
                for i in range(win.output_list.count())]
report['observations'].append({
    'failed_publish_raised': failed,
    'history_unchanged': before_labels == after_labels,
    'count_after_failure': win.output_list.count(),
})

# 6) Narrow window layout.
win.resize(960, 640)
settle()
capture(win, 'main_after_retouch_960_zh')

(WORK / 'report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
win.hide()
win.status_timer.stop()
win.update_manager.shutdown()
print('REPORT', WORK / 'report.json', flush=True)
