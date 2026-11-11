"""Collect the lazily loaded TIFF LZW codec and its distribution license."""
from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata


hiddenimports = ['imagecodecs._lzw', 'imagecodecs._shared']
binaries = collect_dynamic_libs('imagecodecs')
datas = copy_metadata('imagecodecs')
