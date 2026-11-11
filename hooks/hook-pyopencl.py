"""Collect PyOpenCL resources, native loader libraries and license metadata."""
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata


hiddenimports = ['pyopencl._cl']
datas = collect_data_files('pyopencl') + copy_metadata('pyopencl')
binaries = collect_dynamic_libs(
    'pyopencl', search_patterns=['*.dll', '*.dylib', 'lib*.so*']
)
