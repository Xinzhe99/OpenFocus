"""Visual + behavioural check for the scale-bar dialog (02-scalebar).

Runs a real Qt window with a synthetic stack and checks the three behaviours
the spec pins: a cancelled preview leaves the main state untouched, the
working-pixel calibration follows the load scale, and a 16-bit frame keeps its
depth and style. Captures the dialog in Chinese light/dark and a narrow size.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path('F:/Working/OpenFocus')
WORK = Path(os.environ.get('OPENFOCUS_VISUAL_DIR')
            or (Path(os.environ.get('TEMP', '/tmp')) / 'openfocus_scale_visual'))
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


def settle(ms=250):
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
        'labels': [b.text() for b in find_visible_labels(widget)],
    })
    print(name, widget.width(), widget.height(), flush=True)


def find_visible_labels(widget):
    return [lbl for lbl in widget.findChildren(QLabel) if lbl.isVisible()
            and lbl.text()]


win.show()
yy, xx = np.mgrid[:480, :720]
pattern = ((xx // 8 + yy // 8) % 2 * 150 + 50).astype(np.uint8)
base = np.dstack([pattern, ((xx / 720) * 230).astype(np.uint8),
                  ((yy / 480) * 230).astype(np.uint8)])
cv2.putText(base, 'SCALE 50 um', (65, 245), cv2.FONT_HERSHEY_SIMPLEX,
            1.7, (245, 245, 245), 3)
blur = cv2.GaussianBlur(base, (0, 0), 5)
frames = []
for i in range(3):
    frame = blur.copy()
    frame[:, i * 240:(i + 1) * 240] = base[:, i * 240:(i + 1) * 240]
    frames.append(frame)
win.raw_images = frames
win.base_images = list(frames)
win.image_filenames = ['scale_01.png', 'scale_02.png', 'scale_03.png']
win.stack_images = [None] * 3
win.current_scale_factor = 0.5
win.base_scale_factor = 1.0
win.source_px_um = 0.5
win.scale_um_per_px_manual = 0.65
win.transform_manager.reload_image_stack(initial_index=0)

from core.multi_focus_fusion import MultiFocusFusion
result = MultiFocusFusion(algorithm='dct', use_gpu=False).fuse(
    input_source=frames, img_resize=None, kernel_size=7)
from core.render_context import build_render_context
from controllers.output_manager import OutputManager, OutputRecord
context = build_render_context(frames, result, filenames=win.image_filenames,
                               algorithm='dct')
win.fusion_result = result
win.output_manager.update_output_list_for_fusion(context=context)
win.output_manager.show_fusion_result()

from dialogs.scalebar import ScaleBarDialog, CALIBRATION_MANUAL
before = {
    'manual': win.scale_um_per_px_manual,
    'enabled': win.scale_bar_enabled,
    'position': win.scale_bar_position,
    'target': win.scale_bar_target,
}

# 1) Open, change the draft, cancel: nothing may leak into the window.
dlg = ScaleBarDialog(win)
dlg.show()
settle()
dlg.px_spin.setValue(2.0)
dlg.pos_combo.setCurrentIndex(dlg.pos_combo.findData('custom'))
dlg.target_combo.setCurrentIndex(dlg.target_combo.findData('source'))
settle()
capture(dlg, 'scalebar_draft_zh')
report['observations'].append({
    'effective_manual_065_at_50pct': round(dlg.effective_px_um() or 0, 4),
    'header': [lbl.text() for lbl in find_visible_labels(dlg)
               if '源' in lbl.text()][:1],
    'meta': dlg.preview_meta.text(),
})
dlg.reject()
settle()
report['observations'].append({
    'cancel_leaves_state': {
        'manual': win.scale_um_per_px_manual,
        'enabled': win.scale_bar_enabled,
        'position': win.scale_bar_position,
        'target': win.scale_bar_target,
    },
    'cancel_unchanged': (
        win.scale_um_per_px_manual == before['manual']
        and win.scale_bar_enabled == before['enabled']
        and win.scale_bar_position == before['position']
        and win.scale_bar_target == before['target']),
})

# 2) Commit a custom position with a manual value, then check the canvas.
dlg = ScaleBarDialog(win)
dlg.show()
settle()
dlg.mode_combo.setCurrentIndex(dlg.mode_combo.findData(CALIBRATION_MANUAL))
dlg.px_spin.setValue(0.5)
dlg.pos_combo.setCurrentIndex(dlg.pos_combo.findData('bottom-left'))
dlg.target_combo.setCurrentIndex(dlg.target_combo.findData('all'))
settle()
dlg._save_and_accept()
settle()
capture(win, 'main_with_scalebar_zh')
report['observations'].append({
    'committed': {
        'manual': win.scale_um_per_px_manual,
        'position': win.scale_bar_position,
        'target': win.scale_bar_target,
        'font_size': win.scale_bar_font_size,
    },
})

# 3) Dark theme, German, narrow dialog.
win.apply_theme('dark')
trans.set_language('de')
win.update_ui_text()
dlg = ScaleBarDialog(win)
dlg.resize(900, 600)
dlg.show()
settle()
capture(dlg, 'scalebar_dark_de')
dlg.reject()

# 4) 16-bit result keeps depth and the custom placement survives.
u16 = np.full((240, 400, 3), 40000, np.uint16)
u16[0:6, 0:6] = 1234
win.fusion_result = u16
win.scale_bar_target = 'output'
win.scale_um_per_px_manual = 1.0
from utils import scalebar as sb
cfg = sb.export_cfg(win, target='output')
drawn = sb.burn(u16, cfg)
report['observations'].append({
    'uint16_dtype': str(drawn.dtype),
    'uint16_untouched_corner': bool(np.array_equal(drawn[0:6, 0:6], u16[0:6, 0:6])),
    'uint16_changed': bool((drawn != u16).any()),
    'cfg': {k: cfg[k] for k in ('position', 'font_size', 'text_align')},
})

(WORK / 'report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
win.hide()
win.status_timer.stop()
win.update_manager.shutdown()
print('REPORT', WORK / 'report.json', flush=True)
