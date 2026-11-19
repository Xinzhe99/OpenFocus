"""Category-grouped algorithm selectors shared by every dropdown.

Fusion methods declare ``category`` in the registry; this module turns that
into non-selectable section headers, so the main window, batch dialog and
depth dialog all present the same order and grouping.
"""
import importlib.util

from PyQt6.QtCore import QEvent, QObject, QSignalBlocker, Qt
from PyQt6.QtGui import QBrush, QFont, QPalette
from PyQt6.QtWidgets import QComboBox, QToolTip

from core.algorithms import FUSION_DEFAULT, fusion_algorithms, get_fusion
from core.experimental import alignment_is_experimental, alignment_is_locked, experimental_enabled
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
    if spec.id == 'dtcwt' and not all(map(_package_available, ('dtcwt', 'scipy', 'six'))):
        return trans.t("env_dtcwt_unavailable")
    return ""


def fusion_lock_reason(spec, window, ai_available=None):
    """Explain every condition preventing a desktop user from running this method."""
    dependency = fusion_unavailable_reason(spec, ai_available)
    opt_in = (trans.t('experimental_locked_hint')
              if spec.experimental and not experimental_enabled(window) else '')
    return '\n\n'.join(filter(None, (opt_in, dependency)))


def fusion_label(spec):
    label = trans.t(spec.label_key)
    return f"{label} ({trans.t('experimental_label')})" if spec.experimental else label


def desktop_fusion_lock_reason(window, method):
    if method is None or method == 'none':
        return ''
    return fusion_lock_reason(get_fusion(method), window,
                              getattr(window, '_stackmff_available', None))

def apply_method_availability(combo, specs, window, *, alignment=False,
                              ai_available=None, recommended=None):
    """Update labels and disabled explanations without changing the selection."""
    _install_locked_item_hint(combo)
    for spec in specs:
        index = combo.findData(spec.id)
        if index < 0:
            continue
        if alignment:
            label = trans.t(spec.label_key)
            if alignment_is_experimental(spec.id):
                label += f" ({trans.t('experimental_label')})"
        else:
            label = fusion_label(spec)
        if spec.id == recommended:
            label += f" ({trans.t('recommended')})"
        combo.setItemText(index, label)
        if alignment:
            reason = trans.t('experimental_locked_hint') if alignment_is_locked(window, spec.id) else ''
        else:
            reason = fusion_lock_reason(spec, window, ai_available)
        combo.setItemData(index, reason or None, _LOCKED_HINT_ROLE)
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
            combo.addItem(fusion_label(spec), spec.id)
            if spec.hint_key:
                combo.setItemData(combo.count() - 1, trans.t(spec.hint_key),
                                  Qt.ItemDataRole.ToolTipRole)
        default = 'none' if none_label else FUSION_DEFAULT
        index = combo.findData(current if current is not None else default)
        if index >= 0:
            combo.setCurrentIndex(index)


def retranslate_fusion_combo(combo, specs=None, *, none_label=None,
                             none_hint=None):
    """Same list in the active language, keeping the selected method."""
    fill_fusion_combo(combo, specs, none_label=none_label, none_hint=none_hint)
