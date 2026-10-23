"""Visual + behavioural check for the depth map and 3D view (04-depth-3d).

Checks request identity, stale-export blocking, 16-bit texture conversion and
the recorded source summary. Captures the 2D map, 3D tab and a narrow window.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('F:/Working/OpenFocus')
WORK = Path(os.environ.get('OPENFOCUS_VISUAL_DIR')
            or (Path(os.environ.get('TEMP', '/tmp')) / 'openfocus_depth_visual'))
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
cv2.putText(base, 'DEPTH 01 02 03', (60, 245), cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (245, 245, 245), 3)
blur = cv2.GaussianBlur(base, (0, 0), 6)
frames = []
for i in range(3):
    frame = blur.copy()
    frame[:, i * 240:(i + 1) * 240] = base[:, i * 240:(i + 1) * 240]
    frames.append(frame)
win.raw_images = frames
win.base_images = list(frames)
win.image_filenames = ['depth_01.png', 'depth_02.png', 'depth_03.png']
win.stack_images = [None] * 3
win.current_scale_factor = 1.0
win.excluded_frames = {2}
win.transform_manager.reload_image_stack(initial_index=0)

from core.multi_focus_fusion import MultiFocusFusion
result = MultiFocusFusion(algorithm='dct', use_gpu=False).fuse(
    input_source=frames[:2], img_resize=None, kernel_size=7)
win.fusion_result = result

from dialogs.depthmap import DepthMapDialog
dlg = DepthMapDialog(win)
dlg.show()
dlg.method_combo.setCurrentIndex(dlg.method_combo.findData('dct'))
settle()
ok = wait_until(lambda: dlg._index01 is not None, 90)
report['observations'].append({
    'computed': bool(ok),
    'source_summary': dlg.lbl_source.text(),
    'status': dlg.status_lbl.text(),
    'save_enabled': dlg.btn_save_color.isEnabled(),
    'texture_matches': dlg._matching_texture() is not None,
})
capture(dlg, 'depth_map_zh')

# Stale: switching to a SLOW method (no auto recompute) must block exports.
# A fast method deliberately recomputes immediately, which makes the map
# current again rather than stale.
dlg.method_combo.setCurrentIndex(dlg.method_combo.findData('dtcwt'))
settle()
report['observations'].append({
    'stale_status': dlg.status_lbl.text(),
    'save_enabled_after_stale': dlg.btn_save_color.isEnabled(),
    'surface_cleared': dlg.surface._height is None,
})
capture(dlg, 'depth_stale_zh')

# Recompute, then check the 3D tab and the 16-bit texture path.
dlg.method_combo.setCurrentIndex(dlg.method_combo.findData('dct'))
dlg._compute()
wait_until(lambda: dlg._index01 is not None and dlg._worker is None, 90)
settle()
dlg.tabs.setCurrentIndex(1)
settle(400)
report['observations'].append({
    'recomputed': dlg._index01 is not None,
    'surface_error': dlg._surface_error,
    'surface_state_text': dlg.lbl_surface_state.text(),
})
capture(dlg, 'depth_3d_zh')

# 16-bit texture: the surface must use the same display conversion.
u16 = (result.astype(np.uint16) * 257)
dlg.window_main.fusion_result = u16
dlg._map_texture = dlg._capture_texture_identity()
dlg._refresh_surface()
settle(300)
report['observations'].append({
    'uint16_texture_ok': dlg._surface_error == "",
    'uint16_surface_pixels': dlg.surface._height is not None,
})

dlg.resize(960, 640)
settle(300)
capture(dlg, 'depth_3d_small_zh')
dlg.reject()

(WORK / 'report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
win.hide()
win.status_timer.stop()
win.update_manager.shutdown()
print('REPORT', WORK / 'report.json', flush=True)
