"""Collect the lazily loaded TIFF LZW codec and its distribution license."""
from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata


hiddenimports = ['imagecodecs._imcd', 'imagecodecs._shared', 'imagecodecs._shared_cython']
binaries = collect_dynamic_libs('imagecodecs')
datas = copy_metadata('imagecodecs')
