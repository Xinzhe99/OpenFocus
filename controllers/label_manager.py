"""Label management: a list of captions, each with its own scope and style.

Previously this controller held exactly two configurations (source stack and
"registered" stack) and adding a label overwrote the one for the chosen
target. It also could not edit or delete a single caption, and the two target
names hid the fact that fused and retouched outputs share the "registered"
slot.

The model now lives in utils.label_model (LabelSet). This module owns the
dialog, the commit transaction and the render/export entry points every other
module calls. Target names are explicit: input / registered / output.
"""
from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFontComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.styles import get_add_label_dialog_style
from utils import (
    LabelAdder,
    build_label_text,
    cv2_to_pixmap,
    pixmap_to_cv2,
    show_success_box,
    show_warning_box,
)
from utils.label_model import Label, LabelSet, normalize_target
from utils.validators import LabelConfig, parse_label_range
from locales import trans
from widgets.stepper import StepperSpinBox


class LabelManager:
    """Owns the label set, its dialog and every render/export call site."""

    def __init__(self, window: Any) -> None:
        self.window = window
        self._labels = LabelSet()
        # Bumped on every committed change. Caches (A/B pixmaps) key on it so
        # editing a caption cannot keep serving an old drawing.
        self._revision = 0
        self._draw_errors = {}
        self._clear_dialog_state()

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------
    @property
    def labels(self) -> LabelSet:
        return self._labels

    @property
    def revision(self) -> int:
        return self._revision

    def _bump_revision(self) -> None:
        self._revision += 1
        self._draw_errors.clear()
        window = self.window
        if hasattr(window, "invalidate_annotation_caches"):
            window.invalidate_annotation_caches()
        elif hasattr(window, "_wipe_pix_cache"):
            window._wipe_pix_cache.clear()


    # ------------------------------------------------------------------
    # Removal / reset
    # ------------------------------------------------------------------
    def delete_registered_labels(self) -> None:
        if not any(label.target == "registered" for label in self._labels):
            return
        draft = self._labels.copy()
        draft.remove_target("registered")
        self._replace_labels(draft)

    def delete_output_labels(self) -> None:
        if not any(label.target == "output" for label in self._labels):
            return
        draft = self._labels.copy()
        draft.remove_target("output")
        self._replace_labels(draft)

    def delete_input_labels(self) -> None:
        window = self.window
        if not any(label.target == "input" for label in self._labels):
            return
        draft = self._labels.copy()
        draft.remove_target("input")
        self._replace_labels(draft)

    def delete_all_labels(self) -> None:
        window = self.window
        if len(self._labels):
            self._replace_labels(LabelSet())

    def reset_labels(self) -> None:
        """Drop every caption (used when a different stack is loaded)."""
        if not len(self._labels):
            self._sync_delete_actions()
            return
        self._labels.clear()
        self._after_model_change(refresh_source=True, refresh_output=True)

    def _after_model_change(self, refresh_source: bool = False,
                            refresh_output: bool = False) -> None:
        self._bump_revision()
        self._sync_delete_actions()
        window = self.window
        if hasattr(window, "_mark_project_dirty"):
            window._mark_project_dirty()
        try:
            from utils import recovery
            recovery.write_snapshot(window)
        except Exception:
            pass
        if refresh_source and getattr(window, "stack_images", None) \
                and getattr(window, "current_display_index", -1) >= 0:
            try:
                window.update_source_view(window.current_display_index)
            except Exception:
                pass
        if refresh_output:
            try:
                if getattr(window, "fusion_result", None) is not None:
                    window.display_fusion_result()
                elif getattr(window, "registration_results", None) \
                        and getattr(window, "current_result_index", -1) >= 0:
                    window.update_result_view(window.current_result_index)
            except Exception:
                pass

    def _sync_delete_actions(self) -> None:
        """Mirror stored labels, including disabled ones, into target actions."""
        window = self.window
        if hasattr(window, "del_input_label_action"):
            window.del_input_label_action.setEnabled(
                any(label.target == "input" for label in self._labels))
        if hasattr(window, "del_reg_label_action"):
            window.del_reg_label_action.setEnabled(
                any(label.target == "registered" for label in self._labels))
        if hasattr(window, "del_output_label_action"):
            window.del_output_label_action.setEnabled(
                any(label.target == "output" for label in self._labels))
        if hasattr(window, "del_all_labels_action"):
            window.del_all_labels_action.setEnabled(len(self._labels) > 0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def export_state(self) -> dict:
        """Serialize the canonical model; legacy input is migrated on read."""
        return {"version": 2, "labels_v2": self._labels.to_dict()}

    def import_state(self, state: dict) -> None:
        """Restore captions, migrating the legacy two-slot format."""
        try:
            if not state:
                self._labels = LabelSet()
            elif isinstance(state.get("labels_v2"), list):
                self._labels = LabelSet.from_dict(state["labels_v2"])
            else:
                self._labels = self._migrate_legacy(state)
            self._bump_revision()
            self._sync_delete_actions()
            self._refresh_canvases()
        except Exception as exc:
            print(f"label import failed: {exc}")

    @staticmethod
    def _migrate_legacy(state: dict) -> LabelSet:
        """Old projects: input slot -> source, registered -> registered+output.

        A legacy "registered" caption was drawn on both aligned frames and
        fused results, so it must keep that reach; splitting it into two
        copies preserves the previous visible behaviour.
        """
        labels = LabelSet()
        for key, target in (("input", "input"), ("registered", "registered")):
            block = state.get(key) or {}
            config = dict(block.get("config") or {})
            if not config:
                continue
            config.pop("target_stack", None)
            config["target"] = target
            config["enabled"] = bool(block.get("enabled", True))
            label = Label.from_dict(config)
            labels.add(label)
            if target == "registered":
                twin = Label.from_dict({**config, "target": "output"})
                labels.add(twin)
        return labels

    def scale_stack_labels(self, x_ratio: float, y_ratio: float) -> None:
        """Resize annotations inside the caller's existing stack undo step."""
        changed = False
        for label in self._labels:
            if label.target == "output":
                continue
            config = label.config
            config.x_location = int(round(config.x_location * x_ratio))
            config.y_location = int(round(config.y_location * y_ratio))
            config.font_size = max(1, int(round(config.font_size * min(x_ratio, y_ratio))))
            changed = True
        if changed:
            self._bump_revision()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _selected_output(self):
        manager = getattr(self.window, "output_manager", None)
        widget = getattr(self.window, "output_list", None)
        if manager is not None and widget is not None:
            return manager.record_for_item(widget.currentItem())
        return None

    def frame_ids(self, target: str) -> list[str]:
        """Stable file/page identities; ordering follows the target stack."""
        if hasattr(self, "_snapshot_frame_ids"):
            return self._snapshot_frame_ids.get(target, [])
        window = self.window
        if target == "registered":
            identities = getattr(window, "registration_frame_ids", None)
            if identities is not None:
                return list(identities)
        import os
        names = list(getattr(window, "image_filenames", None) or [])
        paths = list(getattr(window, "image_source_paths", None) or [])
        folder = getattr(window, "current_folder_path", "") or ""
        return [os.path.normcase(os.path.abspath(
            paths[i] if i < len(paths) and paths[i] else os.path.join(folder, name)))
            + "::" + name for i, name in enumerate(names)]

    def snapshot(self):
        frozen = LabelManager(self.window)
        frozen._labels = self._labels.copy()
        frozen._snapshot_frame_ids = {
            target: self.frame_ids(target) for target in ("input", "registered")}
        return frozen

    def _draw_for(self, image, target: str, index: int, output_id=None):
        """Resolve identity and numbering once for every display/export path."""
        wanted = normalize_target(target)
        index = 0 if wanted == "output" else index
        if wanted == "output" and output_id is None:
            record = self._selected_output()
            output_id = getattr(record, "output_id", None)
        identities = self.frame_ids(wanted) if wanted != "output" else []
        drawn, error = image, ""
        for label in self._labels.for_target(wanted):
            config = label.config
            if wanted == "output" and config.output_id and config.output_id != output_id:
                continue
            adder = LabelAdder()
            if config.frame_ids is not None and wanted != "output":
                if not 0 <= index < len(identities) or identities[index] not in config.frame_ids:
                    continue
                import copy
                adder.config = copy.copy(config)
                adder.config.range = ""
            else:
                adder.config = config
            drawn = adder.add_label_to_image(drawn, index)
            error = error or adder.last_error
        self._draw_errors[wanted] = error
        return drawn, error

    def apply_labels_to_source_pixmap(self, pixmap, index: int,
                                      scale_cfg=None):
        """Annotated pixmap for the source canvas (labels + scale bar)."""
        image = pixmap_to_cv2(pixmap)
        if image is None:
            return pixmap
        if self._labels.any_enabled("input"):
            image = image.copy()
            image, _err = self._draw_for(image, "input", index)
        if scale_cfg:
            from utils import scalebar
            image = scalebar.burn(image, scale_cfg)
        return cv2_to_pixmap(image)

    def apply_labels_to_registered_image(self, image, index: int):
        """BGR image with registered-stack captions applied."""
        if not self._labels.any_enabled("registered"):
            return image
        drawn, _err = self._draw_for(image.copy(), "registered", index)
        return drawn

    def prepare_bgr_image(self, target: str, image, index: int, output_id=None):
        """Copy of `image` with the captions for `target` burned in.

        `index` is the frame number within the target. For a single output
        result the frame number is always 1, regardless of the row it happens
        to occupy in the output history; that keeps {value} stable across
        "save", "save as", "drag out" and "export all".
        """
        output = image.copy()
        wanted = normalize_target(target)
        if not self._labels.any_enabled(wanted):
            return output
        drawn, error = self._draw_for(output, wanted, index, output_id)
        if error:
            raise ValueError(trans.t('msg_label_draw_failed_title') + ": " + error)
        return drawn

    def display_bgr_image(self, target: str, image, index: int,
                          scale_cfg=None, output_id=None):
        """BGR image ready to be shown, copying only when something draws."""
        wanted = normalize_target(target)
        drawn = None
        if self._labels.any_enabled(wanted):
            drawn, _err = self._draw_for(image.copy(), wanted, index, output_id)
        if scale_cfg:
            from utils import scalebar
            return scalebar.burn(image if drawn is None else drawn, scale_cfg)
        return image if drawn is None else drawn

    def is_enabled_for_target(self, target: str) -> bool:
        return self._labels.any_enabled(normalize_target(target))

    def draw_error_for(self, target: str) -> str:
        """Failure of the most recent draw, cleared by a successful attempt."""
        return self._draw_errors.get(normalize_target(target), "")

    # ------------------------------------------------------------------
    # Dialog
    # ------------------------------------------------------------------
    def show_add_label_dialog(self) -> None:
        """Open the label manager; edits commit only on Apply."""
        window = self.window
        dialog = QDialog(window)
        dialog.setWindowTitle(trans.t('label_manage_title'))
        dialog.resize(1000, 700)
        dialog.setMinimumSize(820, 560)

        from ui.styles import CURRENT_THEME
        dialog.setStyleSheet(get_add_label_dialog_style(CURRENT_THEME))

        root = QVBoxLayout(dialog)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(8)

        header = QLabel(self._header_text())
        header.setObjectName("mutedLabel")
        header.setWordWrap(True)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, 1)

        # ---- left: caption list ------------------------------------------
        left = QVBoxLayout()
        left.setSpacing(6)
        list_title = QLabel(trans.t('label_list'))
        font = list_title.font()
        font.setBold(True)
        list_title.setFont(font)
        left.addWidget(list_title)

        self._dialog_list = QListWidget()
        self._dialog_list.setMinimumWidth(210)
        self._dialog_list.setMaximumWidth(260)
        left.addWidget(self._dialog_list, 1)

        list_buttons = QHBoxLayout()
        add_button = QPushButton(trans.t('label_add'))
        add_button.clicked.connect(self._dialog_add_label)
        delete_button = QPushButton(trans.t('label_delete'))
        delete_button.clicked.connect(self._dialog_delete_label)
        list_buttons.addWidget(add_button)
        list_buttons.addWidget(delete_button)
        left.addLayout(list_buttons)
        body.addLayout(left, 0)

        # ---- middle: form ------------------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        form_host = QWidget()
        form_col = QVBoxLayout(form_host)
        self._form_host = form_host
        form_col.setContentsMargins(0, 0, 8, 0)
        form_col.setSpacing(10)

        def _group(title_key):
            box = QGroupBox(trans.t(title_key))
            lay = QFormLayout(box)
            lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            lay.setHorizontalSpacing(12)
            lay.setVerticalSpacing(8)
            lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            lay.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form_col.addWidget(box)
            return lay

        target_form = _group('label_group_target')
        self.target_stack_combo = QComboBox()
        self.target_stack_combo.addItem(trans.t('label_target_input'), "input")
        self.target_stack_combo.addItem(trans.t('label_target_registered'),
                                        "registered")
        self.target_stack_combo.addItem(trans.t('label_target_output'),
                                        "output")
        if not getattr(window, "registration_results", None):
            item = self.target_stack_combo.model().item(1)
            item.setEnabled(False)
            item.setToolTip(trans.t('label_preview_no_target'))
        target_form.addRow(trans.t('label_target_stack'),
                           self.target_stack_combo)
        self._output_scope = QComboBox()
        self._output_scope.addItem(trans.t('label_outputs_all'), "")
        selected = self._selected_output()
        if selected is not None:
            self._output_scope.addItem(selected.label, selected.output_id)
        target_form.addRow(trans.t('label_target_output'), self._output_scope)
        self._range_mode = QComboBox()
        self._range_mode.addItem(trans.t('label_range_all_placeholder'), "all")
        self._range_mode.addItem(trans.t('label_range_current'), "current")
        self._range_mode.addItem(trans.t('label_range_custom'), "custom")
        target_form.addRow(trans.t('label_range'), self._range_mode)

        self.range_edit = QLineEdit("")
        self.range_edit.setPlaceholderText(trans.t('label_range_all_placeholder'))
        self.range_edit.setToolTip(trans.t('label_range_hint'))
        target_form.addRow(trans.t('label_range'), self.range_edit)
        self._range_status = QLabel("")
        self._range_status.setObjectName("mutedLabel")
        self._range_status.setWordWrap(True)
        target_form.addRow("", self._range_status)
        self._target_form = target_form

        text_form = _group('label_group_text')
        self.format_edit = QLineEdit("{value}")
        self.format_edit.setToolTip(trans.t('label_format_hint'))
        text_form.addRow(trans.t('label_group_text'), self.format_edit)
        insert_button = QPushButton(trans.t('label_insert_value'))
        insert_button.clicked.connect(
            lambda: self.format_edit.insert("{value}"))
        text_form.addRow("", insert_button)

        self.starting_value_spin = StepperSpinBox(
            minimum=0, maximum=9999, decimals=0, step=1)
        self.starting_value_spin.setValue(1)
        text_form.addRow(trans.t('label_start_val'),
                         self.starting_value_spin)
        self.interval_spin = StepperSpinBox(
            minimum=1, maximum=9999, decimals=0, step=1)
        self.interval_spin.setValue(1)
        text_form.addRow(trans.t('label_interval'), self.interval_spin)

        look_form = _group('label_group_look')
        self.font_size_spin = StepperSpinBox(
            minimum=1, maximum=400, decimals=0, step=1)
        self.font_size_spin.setValue(80)
        look_form.addRow(trans.t('label_font_size'), self.font_size_spin)

        self.font_family_combo = QFontComboBox()
        self.font_family_combo.setEditable(True)
        self.font_family_combo.setCurrentFont(QFont("Arial"))
        look_form.addRow(trans.t('label_font_family'),
                         self.font_family_combo)

        position = QHBoxLayout()
        position.setContentsMargins(0, 0, 0, 0)
        self.x_location_spin = StepperSpinBox(
            minimum=0, maximum=20000, decimals=0, step=1)
        self.x_location_spin.setValue(20)
        self.y_location_spin = StepperSpinBox(
            minimum=0, maximum=20000, decimals=0, step=1)
        self.y_location_spin.setValue(80)
        position.addWidget(QLabel("X"))
        position.addWidget(self.x_location_spin)
        position.addSpacing(8)
        position.addWidget(QLabel("Y"))
        position.addWidget(self.y_location_spin)
        position.addStretch()
        look_form.addRow(trans.t('label_position'), position)
        position_hint = QLabel(trans.t('label_position_hint'))
        position_hint.setWordWrap(True)
        look_form.addRow(position_hint)

        self.transparent_bg_checkbox = QCheckBox(
            trans.t('label_transparent_bg'))
        self.transparent_bg_checkbox.setChecked(True)
        look_form.addRow("", self.transparent_bg_checkbox)

        colors = QHBoxLayout()
        colors.setContentsMargins(0, 0, 0, 0)
        self.bg_color_button = QPushButton(trans.t('label_choose_bg'))
        self.bg_color_value = (0, 0, 0)
        self._set_color_button_style(self.bg_color_button, 0, 0, 0)
        self.bg_color_button.clicked.connect(self.choose_background_color)
        self.font_color_button = QPushButton(
            trans.t('label_choose_font_color'))
        self.font_color_value = (255, 255, 255)
        self._set_color_button_style(self.font_color_button, 255, 255, 255)
        self.font_color_button.clicked.connect(self.choose_font_color)
        colors.addWidget(self.bg_color_button)
        colors.addWidget(self.font_color_button)
        colors.addStretch()
        look_form.addRow(trans.t('label_colors'), colors)

        form_col.addStretch()
        scroll.setWidget(form_host)
        body.addWidget(scroll, 1)

        # ---- right: preview ---------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(6)
        preview_title = QLabel(trans.t('label_preview'))
        bold = preview_title.font()
        bold.setBold(True)
        preview_title.setFont(bold)
        right.addWidget(preview_title)

        nav = QHBoxLayout()
        self._preview_prev = QPushButton("<")
        self._preview_prev.setToolTip(trans.t('label_preview_prev'))
        self._preview_prev.setAccessibleName(trans.t('label_preview_prev'))
        self._preview_prev.clicked.connect(lambda: self._step_preview(-1))
        self._preview_next = QPushButton(">")
        self._preview_next.setToolTip(trans.t('label_preview_next'))
        self._preview_next.setAccessibleName(trans.t('label_preview_next'))
        self._preview_next.clicked.connect(lambda: self._step_preview(1))
        self._preview_index = QSpinBox()
        self._preview_index.setRange(1, 9999)
        self._preview_index.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._preview_index.setKeyboardTracking(False)
        self._preview_index.setAccessibleName(trans.t('label_preview'))
        self._preview_index.setValue(1)
        self._preview_index.valueChanged.connect(
            lambda _v: self._refresh_preview())
        self._preview_total = QLabel("/ 0")
        nav.addWidget(self._preview_prev)
        nav.addWidget(self._preview_index)
        nav.addWidget(self._preview_total)
        nav.addWidget(self._preview_next)
        nav.addStretch()
        right.addLayout(nav)

        from widgets.label_preview import LabelPreview
        self.preview = LabelPreview()
        right.addWidget(self.preview, 1)
        preview_tools = QHBoxLayout()
        fit = QPushButton(trans.t('label_fit'))
        fit.clicked.connect(lambda: self.preview.set_zoom_mode("fit"))
        actual = QPushButton("100%")
        actual.clicked.connect(lambda: self.preview.set_zoom_mode("actual"))
        preview_tools.addWidget(fit)
        preview_tools.addWidget(actual)
        right.addLayout(preview_tools)
        self._first_applicable = QPushButton(trans.t('label_first_applicable'))
        self._first_applicable.clicked.connect(self._show_first_applicable)
        right.addWidget(self._first_applicable)

        self.preview_status = QLabel("")
        self.preview_status.setWordWrap(True)
        self.preview_status.setObjectName("mutedLabel")
        right.addWidget(self.preview_status)
        body.addLayout(right, 1)

        # ---- bottom: fixed actions --------------------------------------
        self._dialog_error = QLabel("")
        self._dialog_error.setWordWrap(True)
        self._dialog_error.setStyleSheet("color:#c0392b;")
        root.addWidget(self._dialog_error)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        self._ok_button = buttons.button(
            QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setText(trans.t('label_apply_close'))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(
            trans.t('btn_cancel'))
        buttons.accepted.connect(lambda: dialog.accept() if self._commit_dialog() else None)
        buttons.rejected.connect(dialog.reject)
        root.addWidget(buttons)

        # ---- working copy and wiring ------------------------------------
        self._draft = self._labels.copy()
        self._editing_id: str = ""
        self._dialog_frames: dict = {}
        self._dialog_target = "registered"

        for widget in (self.format_edit, self.range_edit):
            widget.textChanged.connect(self._refresh_preview)
        for spin in (self.font_size_spin, self.starting_value_spin,
                     self.interval_spin, self.x_location_spin,
                     self.y_location_spin):
            spin.valueChanged.connect(self._refresh_preview)
        self.font_family_combo.currentFontChanged.connect(
            self._refresh_preview)
        self.transparent_bg_checkbox.toggled.connect(self._refresh_preview)
        self.target_stack_combo.currentIndexChanged.connect(
            self._on_target_changed)
        self._output_scope.currentIndexChanged.connect(self._on_target_changed)
        self._range_mode.currentIndexChanged.connect(self._on_range_mode_changed)
        self._dialog_list.itemChanged.connect(self._on_label_enabled)
        self._dialog_list.currentRowChanged.connect(self._load_selected_label)

        self._reload_dialog_list()
        if len(self._draft):
            self._dialog_list.setCurrentRow(0)
        else:
            self._dialog_add_label()
        # The preview frame count belongs to the selected target; without this
        # initial refresh the dialog showed "/ 0" until the target changed.
        self._on_target_changed()

        dialog.exec()
        self._clear_dialog_state()

    # ---- dialog helpers ---------------------------------------------------
    def _header_text(self) -> str:
        window = self.window
        frames = len(getattr(window, "raw_images", None) or [])
        registered = len(getattr(window, "registration_results", None) or [])
        width = height = 0
        depth = ""
        images = getattr(window, "raw_images", None) or []
        if images:
            height, width = images[0].shape[:2]
            depth = f"{images[0].dtype.itemsize * 8} bit"
        return trans.t('label_manage_header').format(
            frames=frames, registered=registered,
            width=width, height=height, depth=depth)

    def _reload_dialog_list(self, select_id: str = "") -> None:
        widget = self._dialog_list
        if widget is None:
            return
        widget.blockSignals(True)
        widget.clear()
        for label in self._draft:
            item = QListWidgetItem(self._label_summary(label))
            item.setData(Qt.ItemDataRole.UserRole, label.label_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if label.enabled else Qt.CheckState.Unchecked)
            widget.addItem(item)
            if select_id and label.label_id == select_id:
                widget.setCurrentItem(item)
        widget.blockSignals(False)
        if select_id:
            self._load_selected_label(widget.currentRow())

    def _label_summary(self, label: Label) -> str:
        text = build_label_text(label.config.format, label.config.text, 1)
        text = text or trans.t('label_preview_empty')
        target = {
            "input": trans.t('label_target_input'),
            "registered": trans.t('label_target_registered'),
            "output": trans.t('label_target_output'),
        }.get(label.target, label.target)
        scope = label.config.range or trans.t('label_range_all_placeholder')
        return f"{text}  ·  {target}  ·  {scope}"

    def _dialog_add_label(self) -> None:
        self._save_editor_to_draft()
        label = Label(LabelConfig())
        target = self._current_target()
        if not self._collect_frames(target)["images"]:
            target = "input" if self._collect_frames("input")["images"] else "output"
        label.config.target = target
        label.config.target_stack = 0 if label.target == "input" else 1
        self._draft.add(label)
        self._reload_dialog_list(select_id=label.label_id)
        self._dialog_list.setCurrentRow(self._dialog_list.count() - 1)

    def _dialog_delete_label(self) -> None:
        item = self._dialog_list.currentItem()
        if item is None:
            return
        label_id = item.data(Qt.ItemDataRole.UserRole)
        self._editing_id = ""
        self._draft.remove(label_id)
        self._reload_dialog_list()
        if self._dialog_list.count():
            self._dialog_list.setCurrentRow(0)
        else:
            self._form_host.setEnabled(False)
            self.preview.set_frame(None)
            self._dialog_error.clear()
            self.preview_status.clear()
            self._ok_button.setEnabled(True)

    def _load_selected_label(self, row: int) -> None:
        if row < 0 or self._dialog_list is None:
            return
        item = self._dialog_list.item(row)
        if item is None:
            return
        label = self._draft.get(item.data(Qt.ItemDataRole.UserRole))
        if label is None:
            return
        self._editing_id = label.label_id
        self._loading_editor = True
        self._form_host.setEnabled(True)
        config = label.config
        widgets = (self.target_stack_combo, self.format_edit,
                   self.range_edit, self.starting_value_spin,
                   self.interval_spin, self.font_size_spin,
                   self.x_location_spin, self.y_location_spin,
                   self.transparent_bg_checkbox, self.font_family_combo,
                   self._range_mode, self._output_scope)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            index = self.target_stack_combo.findData(config.target)
            self.target_stack_combo.setCurrentIndex(max(0, index))
            self.format_edit.setText(config.format or config.text)
            stored_range = (config.range or "").strip()
            if config.frame_ids is not None:
                ids = self.frame_ids(config.target)
                stored_range = ",".join(str(i + 1) for i, identity in enumerate(ids)
                                        if identity in config.frame_ids)
            self.range_edit.setText("" if stored_range.lower() == "all" else stored_range)
            self._range_mode.setCurrentIndex(0 if config.frame_ids is None and
                                             parse_label_range(config.range).all_frames else 2)
            scope_index = self._output_scope.findData(config.output_id)
            if scope_index < 0:
                self._output_scope.addItem(trans.t('label_target_missing'), config.output_id)
                scope_index = self._output_scope.count() - 1
            self._output_scope.setCurrentIndex(scope_index)
            self._range_binding = (config.target, self.range_edit.text(), config.frame_ids)
            self.starting_value_spin.setValue(config.starting_value)
            self.interval_spin.setValue(config.interval)
            self.font_size_spin.setValue(config.font_size)
            self.x_location_spin.setValue(config.x_location)
            self.y_location_spin.setValue(config.y_location)
            self.transparent_bg_checkbox.setChecked(
                bool(config.transparent_bg))
            self.font_family_combo.setCurrentFont(
                QFont(config.font_family or "Arial"))
            self.bg_color_value = tuple(config.bg_color)
            self.font_color_value = tuple(config.font_color)
            self._set_color_button_style(self.bg_color_button,
                                         *self._rgb(self.bg_color_value))
            self._set_color_button_style(self.font_color_button,
                                         *self._rgb(self.font_color_value))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
            self._loading_editor = False
        self._on_target_changed()

    @staticmethod
    def _rgb(bgr) -> tuple:
        values = tuple(int(c) for c in bgr)[:3]
        while len(values) < 3:
            values += (0,)
        return values[2], values[1], values[0]

    def _current_target(self) -> str:
        if self.target_stack_combo is None:
            return "registered"
        data = self.target_stack_combo.currentData()
        if data is None:
            # A combo built without userData (legacy callers/tests) still has
            # the documented order: source, registered, output.
            index = max(0, self.target_stack_combo.currentIndex())
            data = ("input", "registered", "output")[
                min(index, 2)]
        return normalize_target(data)

    def _on_target_changed(self, *_args) -> None:
        if self._loading_editor:
            return
        previous_target = self._dialog_target
        self._dialog_target = self._current_target()
        self._dialog_frames = self._collect_frames(self._dialog_target)
        total = len(self._dialog_frames["images"])
        self._preview_total.setText(f"/ {total}")
        self._preview_index.setMaximum(max(1, total))
        if previous_target != self._dialog_target:
            current = (getattr(self.window, "current_display_index", 0)
                       if self._dialog_target == "input" else
                       getattr(self.window, "current_result_index", 0)
                       if self._dialog_target == "registered" else 0)
            self._preview_index.setValue(max(1, min(total, current + 1)))
        self._preview_prev.setEnabled(total > 1)
        self._preview_next.setEnabled(total > 1)
        self._target_form.setRowVisible(
            self._output_scope, self._dialog_target == "output")
        self._target_form.setRowVisible(
            self.range_edit, self._range_mode.currentData() == "custom")
        self.range_edit.setEnabled(self._range_mode.currentData() == "custom")
        self._refresh_preview()

    def _collect_frames(self, target: str, output_id=None) -> dict:
        """Frames the preview may show for a target, with honest labels."""
        window = self.window
        names = list(getattr(window, "image_filenames", None) or [])
        if target == "input":
            images = list(getattr(window, "raw_images", None) or [])
            return {"images": images, "names": names}
        if target == "registered":
            images = list(getattr(window, "registration_results", None) or [])
            if images:
                return {"images": images, "names": names}
            return {"images": [], "names": names}
        record = self._selected_output()
        wanted = output_id
        if wanted is None:
            wanted = self._output_scope.currentData() if self._output_scope is not None else ""
        if wanted and (record is None or record.output_id != wanted):
            manager = getattr(window, "output_manager", None)
            widget = getattr(window, "output_list", None)
            record = None
            if manager is not None and widget is not None:
                for row in range(widget.count()):
                    candidate = manager.record_for_row(row)
                    if candidate is not None and candidate.output_id == wanted:
                        record = candidate
                        break
        if record is not None and record.image is not None:
            return {"images": [record.image], "names": [record.label]}
        return {"images": [], "names": []}

    def _preview_frame_index(self) -> int:
        value = int(round(float(self._preview_index.value())))
        return max(0, value - 1)

    def _step_preview(self, delta: int) -> None:
        total = max(1, int(self._preview_index.maximum()))
        current = int(round(float(self._preview_index.value())))
        self._preview_index.setValue(max(1, min(total, current + delta)))

    def _dialog_config(self) -> dict:
        target = self._current_target()
        range_text = self.range_edit.text()
        if self._range_mode is not None and self._range_mode.currentData() == "all":
            range_text = ""
        parsed = parse_label_range(range_text, len(self._collect_frames(target)["images"]))
        bound_ids = None
        if target != "output" and not parsed.all_frames and parsed.valid:
            ids = self.frame_ids(target)
            bound_ids = [ids[i - 1] for i in parsed.frames if i <= len(ids)]
        binding = self._range_binding
        if binding and binding[:2] == (target, range_text):
            bound_ids = binding[2]
        return {
            "target": self._current_target(),
            "format": self.format_edit.text(),
            "starting_value": int(round(self.starting_value_spin.value())),
            "interval": int(round(self.interval_spin.value())),
            "x_location": int(round(self.x_location_spin.value())),
            "y_location": int(round(self.y_location_spin.value())),
            "font_size": int(round(self.font_size_spin.value())),
            "font_family": self.font_family_combo.currentFont().family(),
            "text": "",
            "range": range_text,
            "transparent_bg": self.transparent_bg_checkbox.isChecked(),
            "bg_color": self.bg_color_value,
            "font_color": self.font_color_value,
            "frame_ids": bound_ids,
            "output_id": (self._output_scope.currentData() or "")
                         if target == "output" and self._output_scope is not None else "",
        }

    def _range_error(self, config: dict) -> str:
        total = len(self._collect_frames(config["target"], config.get("output_id", ""))["images"])
        parsed = parse_label_range(config["range"],
                                   None if config.get("frame_ids") is not None else total)
        if parsed.valid:
            return ""
        key = {
            "range_empty_part": 'label_range_err_empty',
            "range_not_a_number": 'label_range_err_number',
            "range_must_be_positive": 'label_range_err_positive',
            "range_out_of_bounds": 'label_range_err_bounds',
        }.get(parsed.error, 'label_range_err_number')
        return trans.t(key)

    def _on_range_mode_changed(self, *_args) -> None:
        mode = self._range_mode.currentData()
        self._range_binding = None
        self.range_edit.setEnabled(mode == "custom")
        self._target_form.setRowVisible(self.range_edit, mode == "custom")
        if mode == "all":
            self.range_edit.setText("")
        elif mode == "current":
            self.range_edit.setText(str(self._preview_frame_index() + 1))
        self._refresh_preview()

    def _on_label_enabled(self, item) -> None:
        label = self._draft.get(item.data(Qt.ItemDataRole.UserRole))
        if label is not None:
            label.enabled = item.checkState() == Qt.CheckState.Checked
            self._refresh_preview()

    def _save_editor_to_draft(self) -> None:
        if self._draft is None or self._loading_editor or not self._editing_id:
            return
        label = self._draft.get(self._editing_id)
        if label is None:
            return
        label.config.update_config(self._dialog_config())
        label.config.target_stack = 0 if label.target == "input" else 1
        item = self._dialog_list.currentItem()
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == label.label_id:
            self._dialog_list.blockSignals(True)
            item.setText(self._label_summary(label))
            self._dialog_list.blockSignals(False)

    def _applicable_indices(self, config, total):
        ids = config.get("frame_ids")
        if ids is not None and config["target"] != "output":
            return [i for i, identity in enumerate(self.frame_ids(config["target"])[:total])
                    if identity in ids]
        parsed = parse_label_range(config["range"], total)
        return [i for i in range(total) if parsed.matches(i)]

    @staticmethod
    def _draw_error_message(error):
        if error in ("font_unavailable", "font_missing_glyphs"):
            return trans.t('label_font_failed')
        if error == "label_outside_image":
            return trans.t('label_outside_image')
        return trans.t('msg_label_draw_failed_title')

    def _configuration_error(self, config, enabled=True) -> str:
        if not enabled:
            return ""
        error = self._range_error(config)
        if error:
            return error
        if not build_label_text(config["format"], config.get("text", ""), 1):
            return trans.t('label_preview_empty')
        frames = self._collect_frames(config["target"], config.get("output_id", ""))["images"]
        if not frames:
            return trans.t('label_preview_no_target')
        indices = self._applicable_indices(config, len(frames))
        if not indices:
            return trans.t('label_target_missing')
        adder = LabelAdder()
        adder.config.update_config(config)
        adder.config.range = ""
        for index in indices:
            adder.add_label_to_image(frames[index], index, validate_only=True)
            if adder.last_error:
                return self._draw_error_message(adder.last_error)
        return ""

    def _show_first_applicable(self) -> None:
        config = self._dialog_config()
        indices = self._applicable_indices(config, len(self._dialog_frames["images"]))
        if indices:
            self._preview_index.setValue(indices[0] + 1)

    def _refresh_preview(self, *_args) -> None:
        if self.preview is None or self._loading_editor or not self._editing_id:
            return
        self._save_editor_to_draft()
        config = self._dialog_config()
        images = self._dialog_frames.get("images", [])
        index = self._preview_frame_index()
        frame = images[index] if 0 <= index < len(images) else None
        indices = self._applicable_indices(config, len(images))
        preview_config = dict(config)
        preview_config["range"] = "" if index in indices else str(index + 2)
        label = self._draft.get(self._editing_id)
        preview_config["enabled"] = label.enabled
        self.preview.set_preview(frame, preview_config, index)
        error = self._range_error(config) if label.enabled else ""
        if not error and label.enabled:
            if frame is None:
                error = trans.t('label_preview_no_target')
            elif not build_label_text(config["format"], "", 1):
                error = trans.t('label_preview_empty')
            elif index in indices and self.preview.draw_error():
                error = self._draw_error_message(self.preview.draw_error())
        self._range_status.setText(trans.t('label_range_matches').format(
            n=len(indices), total=len(images)))
        self._dialog_error.setText(error)
        self._ok_button.setEnabled(not error)
        self._preview_prev.setEnabled(index > 0)
        self._preview_next.setEnabled(index + 1 < len(images))
        self._first_applicable.setVisible(bool(indices) and index not in indices)
        if error:
            self.preview_status.setText(error)
        elif index not in indices:
            self.preview_status.setText(trans.t('label_preview_outside_range'))
        elif self.preview.substituted_font:
            self.preview_status.setText(trans.t('label_font_substituted').format(
                name=self.preview.substituted_font))
        else:
            self.preview_status.setText(trans.t('label_preview_hint').format(
                n=build_label_text(config["format"], "",
                                   config["starting_value"] + index * config["interval"])))

    def _replace_labels(self, draft) -> bool:
        if draft.to_dict() == self._labels.to_dict():
            return True
        previous = self.export_state()
        window = self.window
        if hasattr(window, "push_label_undo"):
            window.push_label_undo(previous)
        self._labels = draft.copy()
        self._after_model_change()
        self._refresh_canvases()
        window.statusBar().showMessage(trans.t('msg_config_saved_text'), 6000)
        return True

    def _commit_dialog(self) -> bool:
        """Validate every draft label before committing or closing the dialog."""
        for spin in (self.starting_value_spin, self.interval_spin, self.font_size_spin,
                     self.x_location_spin, self.y_location_spin):
            if isinstance(spin, StepperSpinBox):
                spin.lineEdit().editingFinished.emit()
        self._save_editor_to_draft()
        for label in self._draft:
            original = self._labels.get(label.label_id)
            if original is not None and original.to_dict() == label.to_dict():
                continue
            error = self._configuration_error(vars(label.config), label.enabled)
            if error:
                self._reload_dialog_list(label.label_id)
                self._dialog_error.setText(error)
                return False
        return self._replace_labels(self._draft)

    def _refresh_canvases(self) -> None:
        window = self.window
        if hasattr(window, "refresh_annotation_views"):
            window.refresh_annotation_views()
            return
        try:
            if getattr(window, "stack_images", None) \
                    and getattr(window, "current_display_index", -1) >= 0:
                window.update_source_view(window.current_display_index)
        except Exception:
            pass
        try:
            item = getattr(window, "output_list", None)
            item = item.currentItem() if item is not None else None
            if item is not None:
                window.output_manager.display_output_image_in_result_view(
                    item)
            elif getattr(window, "fusion_result", None) is not None:
                window.display_fusion_result()
        except Exception:
            pass

    # Legacy entry point: some tests (and older call sites) fill the form
    # widgets directly and then call this. It now commits one caption.
    def _apply_label_configuration(self) -> None:
        config = self._dialog_config()
        error = self._range_error(config)
        if error:
            show_warning_box(self.window,
                             trans.t('msg_label_draw_failed_title'), error)
            return
        label = Label(LabelConfig())
        label.config.update_config(config)
        label.config.target = config["target"]
        label.enabled = True
        draft = self._labels.copy()
        draft.add(label)
        self._replace_labels(draft)

    # ---- colour pickers ---------------------------------------------------
    def choose_background_color(self) -> None:
        color = QColorDialog.getColor(parent=self.window)
        if color.isValid() and self.bg_color_button is not None:
            self.bg_color_value = (color.blue(), color.green(), color.red())
            self._set_color_button_style(self.bg_color_button,
                                         color.red(), color.green(),
                                         color.blue())
            self._refresh_preview()

    def choose_font_color(self) -> None:
        color = QColorDialog.getColor(parent=self.window)
        if color.isValid() and self.font_color_button is not None:
            self.font_color_value = (color.blue(), color.green(), color.red())
            self._set_color_button_style(self.font_color_button,
                                         color.red(), color.green(),
                                         color.blue())
            self._refresh_preview()

    def _set_color_button_style(self, button: QPushButton, red: int,
                                green: int, blue: int) -> None:
        brightness = (red * 299 + green * 587 + blue * 114) / 1000
        text_color = "black" if brightness > 150 else "white"
        button.setStyleSheet(
            f"background-color: rgb({red}, {green}, {blue}); "
            f"color: {text_color}; border: 1px solid #555; "
            "padding: 6px 12px; border-radius: 4px;")
        button.setToolTip(f"RGB ({red}, {green}, {blue})")

    def _clear_dialog_state(self) -> None:
        self._loading_editor = False
        self._editing_id = ""
        self._range_mode = None
        self._output_scope = None
        self._range_binding = None
        self.target_stack_combo: Optional[QComboBox] = None
        self.format_edit: Optional[QLineEdit] = None
        self.starting_value_spin: Optional[QSpinBox] = None
        self.interval_spin: Optional[QSpinBox] = None
        self.x_location_spin: Optional[QSpinBox] = None
        self.y_location_spin: Optional[QSpinBox] = None
        self.font_size_spin: Optional[QSpinBox] = None
        self.font_family_combo: Optional[QFontComboBox] = None
        self.text_edit: Optional[QLineEdit] = None
        self.range_edit: Optional[QLineEdit] = None
        self.transparent_bg_checkbox: Optional[QCheckBox] = None
        self.bg_color_button: Optional[QPushButton] = None
        self.font_color_button: Optional[QPushButton] = None
        self.bg_color_value = (0, 0, 0)
        self.font_color_value = (255, 255, 255)
        self.preview: Optional["object"] = None
        self.preview_status: Optional[QLabel] = None
        self._refresh_label_preview = None
        self._dialog_list = None
        self._dialog_error = None
        self._range_status = None
        self._ok_button = None
        self._draft = None
        self._preview_prev = None
        self._preview_next = None
        self._preview_index = None
        self._preview_total = None
