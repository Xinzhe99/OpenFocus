# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for OpenFocus with cross-platform drag-and-drop support.

Platform-specific configurations:
- macOS: Info.plist for dock drag-and-drop, app bundle structure
- Linux: Desktop file integration
- Windows: Command-line argument handling (built-in)
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Determine platform
import sys
IS_WIN = sys.platform.startswith('win')
IS_MAC = sys.platform == 'darwin'
IS_LINUX = sys.platform == 'linux'

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('weights', 'weights'),
        ('locales', 'locales'),
    ],
    hiddenimports=[
        'cv2',
        'numpy',
        'scipy',
        'dtcwt',
        'torch',
        'torchvision',
        'imageio',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe_kwargs = {
    'name': 'OpenFocus',
    'debug': False,
    'bootloader_ignore_signals': False,
    'strip': False,
    'upx': True,
    'console': False,
    'icon': 'assets/OpenFocus.ico',
}

if IS_WIN:
    exe_kwargs['uac_admin'] = False

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    **exe_kwargs
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OpenFocus',
)

# macOS: Create app bundle with Info.plist
if IS_MAC:
    app_bundle = BUNDLE(
        coll,
        name='OpenFocus.app',
        icon='assets/OpenFocus.icns',
        bundle_identifier='com.openfocus.app',
        info_plist='assets/Info.plist',
    )
