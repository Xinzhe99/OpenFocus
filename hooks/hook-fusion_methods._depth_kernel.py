"""Retain the CPU compiler and LLVM redistribution notices in frozen builds."""
from PyInstaller.utils.hooks import copy_metadata


datas = copy_metadata('numba') + copy_metadata('llvmlite')
