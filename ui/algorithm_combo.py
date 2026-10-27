"""Category-grouped algorithm selectors shared by every dropdown.

Fusion methods declare ``category`` in the registry; this module turns that
into non-selectable section headers, so the main window, batch dialog and
depth dialog all present the same order and grouping.
"""
from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtGui import QBrush, QFont, QPalette
from PyQt6.QtWidgets import QComboBox

from core.algorithms import fusion_algorithms
from locales import trans

CATEGORY_ORDER = ('ai', 'spatial', 'transform')
CATEGORY_KEYS = {
    'ai': 'group_ai_methods',
    'spatial': 'group_spatial_methods',
    'transform': 'group_transform_methods',
}


def _grouped(specs):
    """Registry order inside each family; families in display order."""
    return sorted(specs, key=lambda spec: CATEGORY_ORDER.index(spec.category))


def _add_header(combo, category):
    combo.addItem(trans.t(CATEGORY_KEYS[category]), None)
    item = combo.model().item(combo.count() - 1)
    item.setEnabled(False)
    # Section labels sit a step below the method text so the list stays
    # readable instead of shouting each header at full size.
    font = QFont(combo.font())
    if font.pointSizeF() > 0:
        font.setPointSizeF(max(6.0, font.pointSizeF() - 1.0))
    elif font.pixelSize() > 0:
        font.setPixelSize(max(6, font.pixelSize() - 1))
    font.setBold(False)
    item.setFont(font)
    palette = combo.palette()
    item.setForeground(QBrush(palette.color(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)))


def fill_fusion_combo(combo, specs=None, *, none_label=None, none_hint=None):
    """Rebuild a method combo with family headers; preserves the selection."""
    specs = list(fusion_algorithms() if specs is None else specs)
    current = combo.currentData()
    with QSignalBlocker(combo):
        combo.clear()
        if none_label:
            combo.addItem(trans.t(none_label), 'none')
            if none_hint:
                combo.setItemData(combo.count() - 1, trans.t(none_hint),
                                  Qt.ItemDataRole.ToolTipRole)
        last = None
        for spec in _grouped(specs):
            if spec.category != last:
                _add_header(combo, spec.category)
                last = spec.category
            combo.addItem(trans.t(spec.label_key), spec.id)
            if spec.hint_key:
                combo.setItemData(combo.count() - 1, trans.t(spec.hint_key),
                                  Qt.ItemDataRole.ToolTipRole)
        index = combo.findData(current)
        if index >= 0:
            combo.setCurrentIndex(index)


def retranslate_fusion_combo(combo, specs=None, *, none_label=None,
                             none_hint=None):
    """Same list in the active language, keeping the selected method."""
    fill_fusion_combo(combo, specs, none_label=none_label, none_hint=none_hint)
