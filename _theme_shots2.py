# -*- coding: utf-8 -*-
"""v1.20 screenshots: light main + welcome + theme buttons; dark regression."""
import sys, os, time, shutil
sys.path.insert(0, "F:/Working/OpenFocus")
os.chdir("F:/Working/OpenFocus")
from PyQt6.QtCore import QSettings
t = "F:/Working/OpenFocus/.temp_qsettings"
shutil.rmtree(t, ignore_errors=True)
os.makedirs(t, exist_ok=True)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, t)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap
app = QApplication(sys.argv)
import main as main_module
import numpy as np

w = main_module.OpenFocus()
w.show()
app.processEvents()

# light theme main window
w.apply_theme('light')
app.processEvents()
w.raw_images = [(np.random.rand(400, 600, 3) * 255).astype(np.uint8) for _ in range(3)]
w.stack_images = [None] * 3
w.stack_slider.setRange(0, 2)
w.current_display_index = 0
w.update_source_view(0)
app.processEvents()
w.grab().save("F:/Working/OpenFocus/.shot2_light_main.png")

# welcome dialog in light theme with theme buttons
w.first_run = True
from dialogs.welcome import WelcomeDialog
dlg = WelcomeDialog(w, main_window=w)
dlg.show()
app.processEvents()
dlg.grab().save("F:/Working/OpenFocus/.shot2_light_welcome.png")
dlg.close()

# dark regression
w.apply_theme('dark')
app.processEvents()
dlg = WelcomeDialog(w, main_window=w)
dlg.show()
app.processEvents()
dlg.grab().save("F:/Working/OpenFocus/.shot2_dark_welcome.png")
dlg.close()
w.grab().save("F:/Working/OpenFocus/.shot2_dark_main.png")
w.close()
app.processEvents()
shutil.rmtree(t, ignore_errors=True)
print('SCREENSHOTS SAVED')
