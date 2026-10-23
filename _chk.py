import sys
sys.path.insert(0,'.')
from PyQt6.QtWidgets import QApplication
app=QApplication([])
import ctypes
from ctypes import wintypes
u=ctypes.windll.user32
found=[]
u.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(
    lambda h,l: (found.append(h), True)[1] or True), 0)
n=0
for h in found:
    t=u.GetWindowTextLengthW(wintypes.HWND(h))
    if not t: continue
    b=ctypes.create_unicode_buffer(t+1); u.GetWindowTextW(wintypes.HWND(h),b,t+1)
    if 'OpenFocus' in b.value and u.IsWindowVisible(wintypes.HWND(h)):
        pm=app.primaryScreen().grabWindow(h)
        im=pm.toImage(); s=c2=0
        for y in range(0,im.height(),10):
            for x in range(0,im.width(),10):
                cc=im.pixelColor(x,y); s+=cc.red()+cc.green()+cc.blue(); c2+=1
        print('%s %dx%d lum=%.0f'%(b.value, im.width(), im.height(), s/max(1,c2)))
        n+=1
print('windows:', n)
