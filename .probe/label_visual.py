"""Visual check for the label manager (01-labels) on a real Qt window.

Builds the same isolated session as the audit script: temporary QSettings,
synthetic stack, a DCT result. Captures the dialog in Chinese/light,
Chinese/dark, and German/light at 960x640 so the layout requirements can be
inspected. Nothing is written outside the temp directory.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path('F:/Working/OpenFocus')
WORK = Path(os.environ.get('OPENFOCUS_VISUAL_DIR')
            or (Path(os.environ.get('TEMP', '/tmp')) / 'openfocus_labels_visual'))
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
win.setWindowTitle('OpenFocus label manager - isolated synthetic session')
win.resize(1280, 800)
win.apply_theme('light')
win.combo_fusion.setCurrentIndex(win.combo_fusion.findData('dct'))
win.combo_align.setCurrentIndex(win.combo_align.findData('none'))
win.use_gpu = False

out = WORK / 'evidence'
out.mkdir(exist_ok=True)
report = {'platform': 'Windows native Qt', 'language': 'zh',
          'theme': 'light', 'screens': [], 'observations': []}
keep = []


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
        'title': widget.windowTitle(),
        'buttons': [b.text() for b in widget.findChildren(QPushButton)
                    if b.isVisible()],
        'labels': [b.text() for b in widget.findChildren(QLabel)
                   if b.isVisible() and b.text()],
    })
    print(name, widget.width(), widget.height(), flush=True)


win.show()
yy, xx = np.mgrid[:480, :720]
pattern = ((xx // 8 + yy // 8) % 2 * 150 + 50).astype(np.uint8)
base = np.dstack([pattern, ((xx / 720) * 230).astype(np.uint8),
                  ((yy / 480) * 230).astype(np.uint8)])
cv2.putText(base, 'FOCUS 01 02 03', (65, 245), cv2.FONT_HERSHEY_SIMPLEX,
            1.7, (245, 245, 245), 3)
blur = cv2.GaussianBlur(base, (0, 0), 5)
frames = []
for i in range(3):
    frame = blur.copy()
    frame[:, i * 240:(i + 1) * 240] = base[:, i * 240:(i + 1) * 240]
    frames.append(frame)
win.raw_images = frames
win.base_images = list(frames)
win.image_filenames = ['focus_01.png', 'focus_02.png', 'focus_03.png']
win.stack_images = [None] * 3
win.current_scale_factor = 1.0
win.transform_manager.reload_image_stack(initial_index=0)
win.source_px_um = 0.5
from core.multi_focus_fusion import MultiFocusFusion
result = MultiFocusFusion(algorithm='dct', use_gpu=False).fuse(
    input_source=frames, img_resize=None, kernel_size=7)
from core.render_context import build_render_context
from controllers.output_manager import OutputRecord
context = build_render_context(frames, result, filenames=win.image_filenames,
                               algorithm='dct')
record = OutputRecord.create(result, 'Label sample - DCT', context=context)
win.fusion_result = result
win.output_manager.update_output_list_for_fusion(context=context)
win.output_manager.show_fusion_result()
capture(win, 'main_with_result_zh')


def label_capture(name, prepare=None):
    def _run():
        dlg = app.activeModalWidget()
        if dlg is None:
            raise RuntimeError('label modal not available')
        if prepare is not None:
            prepare(dlg)
            settle()
        capture(dlg, name)
        dlg.reject()
    QTimer.singleShot(300, _run)


def add_chinese_caption(dlg):
    """Type a Chinese caption into the draft so the font path is visible."""
    from PyQt6.QtWidgets import QLineEdit
    manager = win.label_manager
    manager.format_edit.setText('样品 {value}')
    manager.range_edit.setText('')
    manager._refresh_preview()
    # Read back what the dialog itself believes it will draw.
    report['observations'].append({
        'dialog_format': manager.format_edit.text(),
        'dialog_range': manager.range_edit.text(),
        'range_status': manager._range_status.text(),
        'preview_status': manager.preview_status.text(),
        'preview_total': manager._preview_total.text(),
        'line_edits': len([e for e in dlg.findChildren(QLineEdit)
                           if e.isVisible()]),
    })


label_capture('labels_zh', add_chinese_caption)
win.label_manager.show_add_label_dialog()

# Committed state must be visible on the canvases and in the export.
from utils.label_model import Label
from utils.validators import LabelConfig
committed = Label(LabelConfig())
committed.config.update_config({
    'target': 'output', 'format': '样品 {value}', 'range': 'All',
    'font_size': 40, 'x_location': 24, 'y_location': 90,
    'font_family': 'Microsoft YaHei', 'transparent_bg': True,
    'font_color': (255, 255, 255), 'bg_color': (0, 0, 0),
})
win.label_manager.labels.add(committed)
win.label_manager._bump_revision()
win.label_manager._sync_delete_actions()
win.label_manager._refresh_canvases()
capture(win, 'main_with_label_zh')

win.resize(960, 640)
capture(win, 'main_with_label_960_zh')

win.apply_theme('dark')
trans.set_language('de')
win.update_ui_text()
label_capture('labels_dark_de')
win.label_manager.show_add_label_dialog()

report['observations'].append({
    'committed_labels': len(win.label_manager.labels),
    'export_state_version': win.label_manager.export_state().get('version'),
    'render_changed_pixels': bool(
        (win.label_manager.prepare_bgr_image('output', result, 0) != result).any()),
})
(WORK / 'report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
win.hide()
win.status_timer.stop()
win.update_manager.shutdown()
print('REPORT', WORK / 'report.json', flush=True)
