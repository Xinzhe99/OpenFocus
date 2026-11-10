"""Category-grouped algorithm selectors shared by every dropdown.

Fusion methods declare ``category`` in the registry; this module turns that
into non-selectable section headers, so the main window, batch dialog and
depth dialog all present the same order and grouping.
"""
import importlib.util

from PyQt6.QtCore import QEvent, QObject, QSignalBlocker, Qt
from PyQt6.QtGui import QBrush, QFont, QPalette
from PyQt6.QtWidgets import QComboBox, QToolTip

from core.algorithms import fusion_algorithms
from core.experimental import (alignment_is_experimental, experimental_enabled,
                               fusion_is_experimental)
from locales import trans

CATEGORY_ORDER = ('ai', 'spatial', 'transform')
CATEGORY_KEYS = {
    'ai': 'group_ai_methods',
    'spatial': 'group_spatial_methods',
    'transform': 'group_transform_methods',
}
_LOCKED_HINT_ROLE = Qt.ItemDataRole.UserRole + 101


class _LockedItemHint(QObject):
    def __init__(self, combo):
        super().__init__(combo)
        self.combo = combo

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.ToolTip):
            position = event.position().toPoint() if hasattr(event, 'position') else event.pos()
            index = self.combo.view().indexAt(position)
            hint = index.data(_LOCKED_HINT_ROLE) if index.isValid() else None
            if hint:
                global_pos = event.globalPosition().toPoint() if hasattr(event, 'globalPosition') else event.globalPos()
                QToolTip.showText(global_pos, hint, watched)
                return True
        return super().eventFilter(watched, event)


def _install_locked_item_hint(combo):
    if not hasattr(combo, '_experimental_hint_filter'):
        combo._experimental_hint_filter = _LockedItemHint(combo)
        combo.view().viewport().installEventFilter(combo._experimental_hint_filter)



def _package_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def fusion_unavailable_reason(spec, ai_available):
    """Return a localized explanation for a missing optional backend."""
    if spec.neural and ai_available is False:
        return trans.t("msg_stackmff_unavailable_text")
    if spec.id == "dtcwt" and not _package_available("dtcwt"):
        return trans.t("env_dtcwt_unavailable")
    return ""

def experimental_method_label(spec, *, alignment=False):
    label = trans.t(spec.label_key)
    is_experimental = (alignment_is_experimental(spec.id) if alignment
                       else fusion_is_experimental(spec.id))
    if is_experimental:
        label = f"{label} ({trans.t('experimental_label')})"
    return label


def apply_method_availability(combo, specs, window, *, alignment=False,
                              ai_available=None, recommended=None):
    """Update labels and disabled explanations without changing the selection."""
    _install_locked_item_hint(combo)
    enabled = experimental_enabled(window)
    for spec in specs:
        index = combo.findData(spec.id)
        if index < 0:
            continue
        experimental = (alignment_is_experimental(spec.id) if alignment
                        else fusion_is_experimental(spec.id))
        label = experimental_method_label(spec, alignment=alignment)
        if spec.id == recommended:
            label += f" ({trans.t('recommended')})"
        combo.setItemText(index, label)
        dependency = '' if alignment else fusion_unavailable_reason(spec, ai_available)
        locked = trans.t('experimental_locked_hint') if experimental and not enabled else ''
        combo.setItemData(index, locked or None, _LOCKED_HINT_ROLE)
        reason = '\n\n'.join(text for text in (locked, dependency) if text)
        item = combo.model().item(index)
        if item is not None:
            item.setEnabled(not reason)
        hint = trans.t(spec.hint_key) if spec.hint_key else ''
        combo.setItemData(index, '\n\n'.join(text for text in (hint, reason) if text),
                          Qt.ItemDataRole.ToolTipRole)


def selected_item_enabled(combo):
    index = combo.currentIndex()
    item = combo.model().item(index) if index >= 0 else None
    return item is not None and item.isEnabled()



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
            combo.addItem(experimental_method_label(spec), spec.id)
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
