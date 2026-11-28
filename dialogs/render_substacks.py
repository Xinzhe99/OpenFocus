"""Lightweight dialog to configure single-stack substack rendering."""
from __future__ import annotations

from typing import Sequence

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from core.algorithms import get_fusion, get_registration
from core.substack_utils import (
    MIN_SUBSTACK_SIZE,
    SubstackRange,
    default_chunk_size,
    default_substack_count,
    max_overlap_for_chunk,
    max_overlap_for_count,
    max_substack_count,
    partition_substacks,
)
from locales import trans
from ui import styles as ui_styles
from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette


class RenderSubstacksDialog(QDialog):
    """Configure how an active source stack is split into overlapping substacks."""

    def __init__(
        self,
        parent=None,
        *,
        frame_count: int = 0,
        included_indices: Sequence[int] | None = None,
        fusion_method: str | None = None,
        alignment_method: str | None = None,
        theme: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("renderSubstacksDialog")
        self.setProperty("controlId", "render_substacks_dialog")

        if frame_count <= 0 and parent is not None and hasattr(parent, "effective_stack"):
            frames, _names, inc = parent.effective_stack()
            frame_count = len(frames)
            if included_indices is None:
                included_indices = inc

        self._frame_count = max(MIN_SUBSTACK_SIZE, int(frame_count))
        self._included_indices = (
            tuple(int(i) for i in included_indices)
            if included_indices is not None and len(included_indices) == self._frame_count
            else tuple(range(self._frame_count))
        )
        self._fusion_method = fusion_method or (
            parent.current_fusion_method()
            if parent is not None and callable(getattr(parent, "current_fusion_method", None))
            else "openfocus_b"
        )
        self._alignment_method = alignment_method or (
            parent.current_alignment_method()
            if parent is not None and callable(getattr(parent, "current_alignment_method", None))
            else "ecc"
        )
        self._theme = theme or getattr(parent, "ui_theme", None) or ui_styles.CURRENT_THEME
        self._ranges: list[SubstackRange] = []

        self._build_ui()
        self.retranslate()
        self.apply_theme(self._theme)
        self._update_preview()
        self.setMinimumWidth(460)
        self.resize(500, 490)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"])
        root.setSpacing(SPACE["md"])

        # Header
        self._title_label = QLabel()
        self._title_label.setObjectName("substackDialogTitle")
        root.addWidget(self._title_label)

        # Split strategy group
        self._split_group = QGroupBox()
        split_grid = QGridLayout(self._split_group)
        split_grid.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        split_grid.setHorizontalSpacing(SPACE["md"])
        split_grid.setVerticalSpacing(SPACE["md"])

        self._mode_group = QButtonGroup(self)
        self.radio_chunk_size = QRadioButton()
        self.radio_chunk_size.setProperty("controlId", "substack_radio_chunk_size")
        self.radio_chunk_size.setChecked(True)
        self.radio_count = QRadioButton()
        self.radio_count.setProperty("controlId", "substack_radio_count")
        self._mode_group.addButton(self.radio_chunk_size)
        self._mode_group.addButton(self.radio_count)

        self.spin_chunk_size = QSpinBox()
        self.spin_chunk_size.setProperty("controlId", "substack_spin_chunk_size")
        self.spin_chunk_size.setRange(MIN_SUBSTACK_SIZE, self._frame_count)
        self.spin_chunk_size.setValue(default_chunk_size(self._frame_count))
        self.spin_chunk_size.setMinimumWidth(96)

        max_count = max(2, max_substack_count(self._frame_count))
        self.spin_count = QSpinBox()
        self.spin_count.setProperty("controlId", "substack_spin_count")
        self.spin_count.setRange(2, max_count)
        self.spin_count.setValue(max(2, default_substack_count(self._frame_count)))
        self.spin_count.setMinimumWidth(96)

        self._overlap_label = QLabel()
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setProperty("controlId", "substack_spin_overlap")
        self.spin_overlap.setRange(0, max_overlap_for_chunk(self.spin_chunk_size.value()))
        self.spin_overlap.setValue(min(1, self.spin_overlap.maximum()))
        self.spin_overlap.setMinimumWidth(96)
        self._overlap_label.setBuddy(self.spin_overlap)

        split_grid.addWidget(self.radio_chunk_size, 0, 0)
        split_grid.addWidget(self.spin_chunk_size, 0, 1, Qt.AlignmentFlag.AlignRight)
        split_grid.addWidget(self.radio_count, 1, 0)
        split_grid.addWidget(self.spin_count, 1, 1, Qt.AlignmentFlag.AlignRight)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("substackDivider")
        split_grid.addWidget(sep, 2, 0, 1, 2)

        split_grid.addWidget(self._overlap_label, 3, 0)
        split_grid.addWidget(self.spin_overlap, 3, 1, Qt.AlignmentFlag.AlignRight)
        split_grid.setColumnStretch(0, 1)
        root.addWidget(self._split_group)

        # Live range preview group
        self._preview_group = QGroupBox()
        preview_layout = QVBoxLayout(self._preview_group)
        preview_layout.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        preview_layout.setSpacing(SPACE["sm"])

        self.preview_summary_label = QLabel()
        self.preview_summary_label.setObjectName("substackSummaryPill")
        self.preview_summary_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_summary_label)

        self.preview_list = QListWidget()
        self.preview_list.setObjectName("substackPreviewList")
        self.preview_list.setProperty("controlId", "substack_preview_list")
        self.preview_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.preview_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preview_list.setMinimumHeight(120)
        self.preview_list.setMaximumHeight(190)
        preview_layout.addWidget(self.preview_list, 1)
        root.addWidget(self._preview_group, 1)

        # Optional nested auto-restack checkbox
        self.chk_auto_restack = QCheckBox()
        self.chk_auto_restack.setProperty("controlId", "substack_chk_auto_restack")
        self.chk_auto_restack.setChecked(False)
        root.addWidget(self.chk_auto_restack)

        # Footer action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACE["sm"])
        btn_row.addStretch(1)
        self.btn_cancel = QPushButton()
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_render = QPushButton()
        self.btn_render.setObjectName("primaryButton")
        self.btn_render.setProperty("controlId", "substack_btn_render")
        self.btn_render.setDefault(True)
        self.btn_render.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_render)
        root.addLayout(btn_row)

        # Wire live updates
        self.radio_chunk_size.toggled.connect(self._on_mode_changed)
        self.radio_count.toggled.connect(self._on_mode_changed)
        self.spin_chunk_size.valueChanged.connect(self._on_split_changed)
        self.spin_count.valueChanged.connect(self._on_split_changed)
        self.spin_overlap.valueChanged.connect(self._update_preview)
        self._sync_spin_enabled()

    def _method_display(self) -> tuple[str, str]:
        try:
            fusion_name = (
                trans.t("radio_no_fusion")
                if self._fusion_method == "none"
                else trans.t(get_fusion(self._fusion_method).label_key)
            )
        except ValueError:
            fusion_name = str(self._fusion_method)
        try:
            align_name = trans.t(get_registration(self._alignment_method).label_key)
        except ValueError:
            align_name = str(self._alignment_method)
        return fusion_name, align_name

    def _current_max_overlap(self) -> int:
        if self.selected_mode() == "chunk_size":
            return max_overlap_for_chunk(self.spin_chunk_size.value())
        return max_overlap_for_count(self._frame_count, self.spin_count.value())

    def _sync_spin_enabled(self) -> None:
        by_chunk = self.radio_chunk_size.isChecked()
        self.spin_chunk_size.setEnabled(by_chunk)
        self.spin_count.setEnabled(not by_chunk)

    def _on_mode_changed(self) -> None:
        self._sync_spin_enabled()
        self._on_split_changed()

    def _on_split_changed(self) -> None:
        max_ov = self._current_max_overlap()
        with QSignalBlocker(self.spin_overlap):
            self.spin_overlap.setMaximum(max_ov)
            if self.spin_overlap.value() > max_ov:
                self.spin_overlap.setValue(max_ov)
        self._update_preview()

    def _update_preview(self) -> None:
        max_ov = self._current_max_overlap()
        overlap_tip = trans.t("substack_overlap_hint").format(max_overlap=max_ov)
        self._overlap_label.setToolTip(overlap_tip)
        self.spin_overlap.setToolTip(overlap_tip)
        mode = self.selected_mode()
        self._ranges = partition_substacks(
            self._frame_count,
            mode=mode,
            chunk_size=self.spin_chunk_size.value(),
            substack_count=self.spin_count.value(),
            overlap=self.spin_overlap.value(),
            included_indices=self._included_indices,
        )
        self.preview_summary_label.setText(
            trans.t("substack_preview_summary").format(
                substacks=len(self._ranges),
                frames=self._frame_count,
                overlap=self.spin_overlap.value(),
            )
        )
        self.preview_list.clear()
        for sr in self._ranges:
            text = trans.t("substack_preview_item").format(
                index=sr.index,
                start=sr.display_start,
                end=sr.display_end,
                count=sr.count,
            )
            self.preview_list.addItem(QListWidgetItem(text))
        self.btn_render.setEnabled(bool(self._ranges))

    def retranslate(self) -> None:
        fusion_name, align_name = self._method_display()
        self.setWindowTitle(trans.t("substack_dialog_title"))
        self._title_label.setText(trans.t("substack_dialog_title"))
        self._split_group.setToolTip(
            trans.t("substack_dialog_subtitle").format(
                frames=self._frame_count,
                method=fusion_name,
                alignment=align_name,
            )
        )
        self._split_group.setTitle(trans.t("substack_group_split"))
        self.radio_chunk_size.setText(trans.t("substack_mode_chunk_size"))
        self.radio_count.setText(trans.t("substack_mode_count"))
        self._overlap_label.setText(trans.t("substack_overlap_label"))
        self._preview_group.setTitle(trans.t("substack_group_preview"))
        self.chk_auto_restack.setText(trans.t("substack_auto_restack"))
        self.chk_auto_restack.setToolTip(trans.t("substack_auto_restack_hint"))
        self.btn_cancel.setText(trans.t("btn_cancel_generic"))
        self.btn_render.setText(trans.t("substack_btn_render"))
        self._update_preview()

    def apply_theme(self, theme: str = "light") -> None:
        """Apply Light / Dark / Gray tokens from ``utils.design_tokens``."""
        self._theme = theme if theme in ("light", "dark", "gray") else "light"
        t = palette(self._theme)
        self.setStyleSheet(
            f"""
            QDialog#renderSubstacksDialog {{
                background-color: {t["bg"]};
                color: {t["text"]};
            }}
            QLabel#substackDialogTitle {{
                color: {t["text"]};
                font-size: {FONT["title"]}pt;
                font-weight: {WEIGHT["semibold"]};
            }}
            QLabel#mutedLabel {{
                color: {t["textMuted"]};
            }}
            QFrame#substackDivider {{
                color: {t["border"]};
                background-color: {t["border"]};
                max-height: 1px;
                margin-top: {SPACE["xs"]}px;
                margin-bottom: {SPACE["xs"]}px;
            }}
            QLabel#substackSummaryPill {{
                background-color: {t["accentSoft"]};
                color: {t["accentSoftText"]};
                border: 1px solid {t["border"]};
                border-radius: {RADIUS["sm"]}px;
                padding: {SPACE["xs"]}px {SPACE["sm"]}px;
                font-weight: {WEIGHT["medium"]};
            }}
            QListWidget#substackPreviewList {{
                background-color: {t["surfaceSunken"]};
                color: {t["text"]};
                border: 1px solid {t["border"]};
                border-radius: {RADIUS["sm"]}px;
                padding: {SPACE["xs"]}px;
            }}
            QListWidget#substackPreviewList::item {{
                padding: {SPACE["xs"]}px {SPACE["sm"]}px;
                border-bottom: 1px solid {t["border"]};
            }}
            """
        )

    # ------------------------------------------------------------------
    # Public configuration accessors
    # ------------------------------------------------------------------
    def selected_mode(self) -> str:
        return "chunk_size" if self.radio_chunk_size.isChecked() else "count"

    def chunk_size(self) -> int:
        return int(self.spin_chunk_size.value())

    def substack_count(self) -> int:
        return int(self.spin_count.value())

    def overlap(self) -> int:
        return int(self.spin_overlap.value())

    def auto_restack(self) -> bool:
        return bool(self.chk_auto_restack.isChecked())

    def substack_ranges(self) -> list[SubstackRange]:
        return list(self._ranges)

    def plan(self) -> dict:
        return {
            "mode": self.selected_mode(),
            "chunk_size": self.chunk_size(),
            "substack_count": self.substack_count(),
            "overlap": self.overlap(),
            "auto_restack": self.auto_restack(),
            "ranges": self.substack_ranges(),
        }
