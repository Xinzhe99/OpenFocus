"""Collect lazy utility exports from their single runtime declaration."""
from utils import _EXPORTS

hiddenimports = sorted(set(_EXPORTS.values()) | {'utils.layered_tiff'})
