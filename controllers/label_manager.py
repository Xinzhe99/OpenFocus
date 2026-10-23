from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QFontComboBox,
    QGroupBox,
    QLabel,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.styles import WHITE_COMBOBOX_STYLE
from utils import (
    LabelAdder,
    build_label_text,
    cv2_to_pixmap,
    pixmap_to_cv2,
    show_success_box,
    show_warning_box,
)
from locales import trans
from widgets.stepper import StepperSpinBox


class LabelManager:
    """Encapsulates label configuration dialogs and state toggles."""

    def __init__(self, window: Any) -> None:
        self.window = window

        self._input_adder = LabelAdder()
        self._input_adder.config.target_stack = 0
        self._registered_adder = LabelAdder()
        self._registered_adder.config.target_stack = 1

        self._input_enabled = False
        self._registered_enabled = False

        self._clear_dialog_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def delete_registered_labels(self) -> None:
        window = self.window

        if not self._registered_enabled:
            show_warning_box(window, trans.t('msg_no_reg_labels'), trans.t('msg_reg_labels_disabled'))
            return

        self._registered_enabled = False
        self._sync_delete_actions()

        if window.fusion_result is not None:
            window.display_fusion_result()
        elif window.registration_results and window.current_result_index >= 0:
            window.update_result_view(window.current_result_index)

        show_success_box(window, trans.t('msg_success'), trans.t('msg_reg_labels_removed'))

    def delete_input_labels(self) -> None:
        window = self.window

        if not self._input_enabled:
            show_warning_box(window, trans.t('msg_no_input_labels'), trans.t('msg_input_labels_disabled'))
            return

        self._input_enabled = False
        self._sync_delete_actions()

        if window.stack_images and window.current_display_index >= 0:
            window.update_source_view(window.current_display_index)

        show_success_box(window, trans.t('msg_success'), trans.t('msg_input_labels_removed'))

    def _sync_delete_actions(self) -> None:
        """Mirror the enabled flags into the three Remove menu entries.

        The "remove all" entry has to follow every state change: it used to be
        enabled only when a project was loaded, so adding a label left it
        greyed out and there was no one-click removal.
        """
        window = self.window
        if hasattr(window, "del_input_label_action"):
            window.del_input_label_action.setEnabled(self._input_enabled)
        if hasattr(window, "del_reg_label_action"):
            window.del_reg_label_action.setEnabled(self._registered_enabled)
        if hasattr(window, "del_all_labels_action"):
            window.del_all_labels_action.setEnabled(
                self._input_enabled or self._registered_enabled)

    def delete_all_labels(self) -> None:
        """One click for "remove from both stacks" (the two single deletes
        stay for targeted removal)."""
        window = self.window
        had_any = self._input_enabled or self._registered_enabled
        self._input_enabled = False
        self._registered_enabled = False
        self._sync_delete_actions()
        if not had_any:
            show_warning_box(window, trans.t('msg_no_input_labels'),
                             trans.t('msg_input_labels_disabled'))
            return
        if window.fusion_result is not None:
            window.display_fusion_result()
        elif window.registration_results and window.current_result_index >= 0:
            window.update_result_view(window.current_result_index)
        if window.stack_images and window.current_display_index >= 0:
            window.update_source_view(window.current_display_index)
        show_success_box(window, trans.t('msg_success'),
                         trans.t('msg_all_labels_removed'))

    def reset_labels(self) -> None:
        """Disable all labels and update menu state."""
        self._input_enabled = False
        self._registered_enabled = False
        self._sync_delete_actions()

    def export_state(self) -> dict:
        """Serialize both label configurations for project files."""
        def _dump(adder, enabled):
            cfg = dict(vars(adder.config)) if adder is not None else {}
            # strip non-JSON-safe values (none expected, but be safe)
            return {"enabled": enabled, "config": cfg}
        return {
            "input": _dump(self._input_adder, self._input_enabled),
            "registered": _dump(self._registered_adder, self._registered_enabled),
        }

    def import_state(self, state: dict) -> None:
        """Restore label configurations from a project file."""
        try:
            for key, adder, attr in (("input", self._input_adder, "_input_enabled"),
                                     ("registered", self._registered_adder, "_registered_enabled")):
                block = state.get(key) or {}
                cfg = block.get("config", {})
                if cfg:
                    adder.config.update_config(cfg)
                adder.config.target_stack = 0 if key == "input" else 1
                setattr(self, attr, bool(block.get("enabled", False)))
            self._sync_delete_actions()
            if (self._input_enabled or self._registered_enabled) \
                    and self.window.stack_images \
                    and self.window.current_display_index >= 0:
                self.window.update_source_view(self.window.current_display_index)
        except Exception as exc:
            print(f"label import failed: {exc}")

    def apply_labels_to_source_pixmap(self, pixmap, index: int,
                                      scale_cfg=None):
        """Return a pixmap annotated for the source view.

        Labels apply when enabled; the scale bar applies whenever it is
        enabled in the dialog, mirroring the fused view so both canvases show
        the annotations the export will carry.
        """
        image = pixmap_to_cv2(pixmap)
        if image is None:
            return pixmap

        if self._input_enabled:
            image = self._input_adder.add_label_to_image(image.copy(), index)
        if scale_cfg:
            from utils import scalebar
            image = scalebar.burn(image, scale_cfg)
        return cv2_to_pixmap(image)

    def apply_labels_to_registered_image(self, image, index: int):
        """Return a BGR image with registered labels applied when enabled."""
        if not self._registered_enabled:
            return image

        return self._registered_adder.add_label_to_image(image.copy(), index)

    def prepare_bgr_image(self, target: str, image, index: int):
        """Prepare a copy of the image with labels applied for saving."""
        output = image.copy()
        if target == "registered":
            if self._registered_enabled:
                return self._registered_adder.add_label_to_image(output, index)
            return output
        if target == "input":
            if self._input_enabled:
                return self._input_adder.add_label_to_image(output, index)
            return output
        raise ValueError(f"Unknown label target: {target}")

    def display_bgr_image(self, target: str, image, index: int, scale_cfg=None):
        """BGR image ready to be shown, without a defensive copy.

        `prepare_bgr_image` copies because exports may draw into the result;
        the display path only reads, and copying a 24 MP frame per repaint
        cost 13 ms on its own. LabelAdder draws in place, so a copy is still
        made when labels are actually enabled — never otherwise.

        `scale_cfg` draws the scale bar on the DISPLAY image as well, so the
        user sees the annotation they will get on export (previously the bar
        only existed in saved files, which read as "the feature does
        nothing").
        """
        drawn = None
        if target == "registered":
            if self._registered_enabled:
                drawn = self._registered_adder.add_label_to_image(
                    image.copy(), index)
        elif target == "input":
            if self._input_enabled:
                drawn = self._input_adder.add_label_to_image(image.copy(), index)
        else:
            raise ValueError(f"Unknown label target: {target}")
        if scale_cfg:
            from utils import scalebar
            return scalebar.burn(image if drawn is None else drawn, scale_cfg)
        return image if drawn is None else drawn

    def is_enabled_for_target(self, target: str) -> bool:
        """Expose current label enabled state for the requested stack."""
        if target == "registered":
            return self._registered_enabled
        if target == "input":
            return self._input_enabled
        raise ValueError(f"Unknown label target: {target}")

    def show_add_label_dialog(self) -> None:
        """Configure a label, previewed live before confirming."""
        window = self.window

        dialog = QDialog(window)
        dialog.setWindowTitle(trans.t('add_label_title'))
        dialog.resize(920, 620)

        from ui.styles import CURRENT_THEME, get_add_label_dialog_style
        dialog.setStyleSheet(get_add_label_dialog_style(CURRENT_THEME))

        root = QHBoxLayout(dialog)
        root.setSpacing(16)
        root.setContentsMargins(16, 16, 16, 16)

        # ---- left column: the form, grouped by what it affects ------------
        form_host = QWidget()
        form_col = QVBoxLayout(form_host)
        form_col.setContentsMargins(0, 0, 0, 0)
        form_col.setSpacing(10)

        def _group(title_key):
            box = QGroupBox(trans.t(title_key))
            lay = QFormLayout(box)
            lay.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            lay.setHorizontalSpacing(14)
            lay.setVerticalSpacing(8)
            lay.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            lay.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form_col.addWidget(box)
            return lay

        # 1. Where the label goes.
        target_form = _group('label_group_target')
        self.target_stack_combo = QComboBox()
        self.target_stack_combo.addItems([
            trans.t('label_target_input'),
            trans.t('label_target_registered')])
        self.target_stack_combo.setCurrentIndex(1)
        self.target_stack_combo.setMinimumWidth(200)
        self.target_stack_combo.setStyleSheet(WHITE_COMBOBOX_STYLE)
        target_form.addRow(trans.t('label_target_stack'),
                           self.target_stack_combo)

        self.range_edit = QLineEdit("All")
        self.range_edit.setStyleSheet(
            "background-color: #fff; color: #000; font-weight: normal;")
        self.range_edit.setToolTip(trans.t('label_range_hint'))
        target_form.addRow(trans.t('label_range'), self.range_edit)

        # 2. What it says.
        text_form = _group('label_group_text')
        self.format_edit = QLineEdit("{value}")
        self.format_edit.setStyleSheet(
            "background-color: #fff; color: #000; font-weight: normal;")
        self.format_edit.setToolTip(trans.t('label_format_hint'))
        text_form.addRow(trans.t('label_format'), self.format_edit)

        self.text_edit = QLineEdit("")
        self.text_edit.setStyleSheet(
            "background-color: #fff; color: #000; font-weight: normal;")
        self.text_edit.setToolTip(trans.t('label_custom_text_hint'))
        text_form.addRow(trans.t('label_custom_text'), self.text_edit)

        self.starting_value_spin = StepperSpinBox(
            minimum=0, maximum=9999, decimals=0, step=1)
        self.starting_value_spin.setValue(1)
        self.starting_value_spin.setMinimumWidth(90)
        self.starting_value_spin.setStyleSheet(
            "background-color: #fff; color: #000; font-weight: normal;")
        self.interval_spin = StepperSpinBox(
            minimum=1, maximum=9999, decimals=0, step=1)
        self.interval_spin.setValue(1)
        self.interval_spin.setMinimumWidth(90)
        self.interval_spin.setStyleSheet(
            "background-color: #fff; color: #000; font-weight: normal;")
        # One label per line keeps words intact: "起始值" and "步长" next to two
        # 90 px steppers were squeezed into a single clipped chunk.
        numbering = QVBoxLayout()
        numbering.setContentsMargins(0, 0, 0, 0)
        numbering.setSpacing(4)
        for caption_key, spin in (('label_start_val',
                                   self.starting_value_spin),
                                  ('label_interval', self.interval_spin)):
            line = QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            caption = QLabel(trans.t(caption_key))
            caption.setMinimumWidth(84)
            line.addWidget(caption)
            line.addWidget(spin, 1)
            numbering.addLayout(line)
        text_form.addRow("", numbering)

        # 3. How it looks.
        look_form = _group('label_group_look')
        self.font_size_spin = StepperSpinBox(
            minimum=6, maximum=400, decimals=0, step=1)
        self.font_size_spin.setValue(80)
        self.font_size_spin.setMinimumWidth(90)
        self.font_size_spin.setStyleSheet(
            "background-color: #fff; color: #000; font-weight: normal;")
        look_form.addRow(trans.t('label_font_size'), self.font_size_spin)

        self.font_family_combo = QFontComboBox()
        self.font_family_combo.setEditable(True)
        self.font_family_combo.setCurrentFont(QFont("Arial"))
        self.font_family_combo.setStyleSheet(WHITE_COMBOBOX_STYLE)
        look_form.addRow(trans.t('label_font_family'), self.font_family_combo)

        position = QHBoxLayout()
        position.setContentsMargins(0, 0, 0, 0)
        self.x_location_spin = StepperSpinBox(
            minimum=0, maximum=20000, decimals=0, step=1)
        self.x_location_spin.setValue(20)
        self.x_location_spin.setMinimumWidth(90)
        self.x_location_spin.setStyleSheet(
            "background-color: #fff; color: #000; font-weight: normal;")
        self.y_location_spin = StepperSpinBox(
            minimum=0, maximum=20000, decimals=0, step=1)
        self.y_location_spin.setValue(80)
        self.y_location_spin.setMinimumWidth(90)
        self.y_location_spin.setStyleSheet(
            "background-color: #fff; color: #000; font-weight: normal;")
        position.addWidget(QLabel("X"))
        position.addWidget(self.x_location_spin)
        position.addSpacing(10)
        position.addWidget(QLabel("Y"))
        position.addWidget(self.y_location_spin)
        position.addStretch()
        look_form.addRow(trans.t('label_position'), position)

        self.transparent_bg_checkbox = QCheckBox(
            trans.t('label_transparent_bg'))
        self.transparent_bg_checkbox.setChecked(True)
        look_form.addRow("", self.transparent_bg_checkbox)

        colors = QHBoxLayout()
        colors.setContentsMargins(0, 0, 0, 0)
        colors.setSpacing(10)
        self.bg_color_button = QPushButton(trans.t('label_choose_bg'))
        self.bg_color_button.setFixedHeight(30)
        self.bg_color_value = (0, 0, 0)
        self._set_color_button_style(self.bg_color_button, 0, 0, 0)
        self.bg_color_button.clicked.connect(self.choose_background_color)
        self.font_color_button = QPushButton(trans.t('label_choose_font_color'))
        self.font_color_button.setFixedHeight(30)
        self.font_color_value = (255, 255, 255)
        self._set_color_button_style(self.font_color_button, 255, 255, 255)
        self.font_color_button.clicked.connect(self.choose_font_color)
        colors.addWidget(self.bg_color_button)
        colors.addWidget(self.font_color_button)
        colors.addStretch()
        look_form.addRow(trans.t('label_colors'), colors)

        form_col.addStretch()
        root.addWidget(form_host, 0)

        # ---- right column: live preview ----------------------------------
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)
        preview_title = QLabel(trans.t('label_preview'))
        f = preview_title.font()
        f.setBold(True)
        preview_title.setFont(f)
        right.addWidget(preview_title)

        from widgets.label_preview import LabelPreview
        self.preview = LabelPreview()
        right.addWidget(self.preview, 1)

        self.preview_status = QLabel("")
        self.preview_status.setWordWrap(True)
        self.preview_status.setObjectName("mutedLabel")
        right.addWidget(self.preview_status)
        root.addLayout(right, 1)

        # ---- buttons ------------------------------------------------------
        buttons = QHBoxLayout()
        buttons.addStretch()
        ok_button = QPushButton(trans.t('btn_ok'))
        ok_button.setMinimumWidth(90)
        ok_button.setDefault(True)
        cancel_button = QPushButton(trans.t('btn_cancel'))
        cancel_button.setMinimumWidth(90)
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        form_col.addLayout(buttons)

        # Seed the preview with a real frame so proportions are honest.
        frame = None
        try:
            if window.registration_results:
                frame = window.registration_results[0]
            elif getattr(window, "raw_images", None):
                frame = window.raw_images[0]
        except Exception:
            frame = None
        self.preview.set_frame(frame)

        def _current_config() -> dict:
            return {
                "target_stack": self.target_stack_combo.currentIndex(),
                "format": self.format_edit.text(),
                "starting_value": self.starting_value_spin.value(),
                "interval": self.interval_spin.value(),
                "x_location": self.x_location_spin.value(),
                "y_location": self.y_location_spin.value(),
                "font_size": self.font_size_spin.value(),
                "font_family": self.font_family_combo.currentFont().family(),
                "text": self.text_edit.text(),
                "range": self.range_edit.text(),
                "transparent_bg": self.transparent_bg_checkbox.isChecked(),
                "bg_color": self.bg_color_value,
                "font_color": self.font_color_value,
            }

        def _refresh_preview(*_args):
            self.preview.set_config(_current_config())
            if self.preview.message() == "empty":
                self.preview_status.setText(trans.t('label_preview_empty'))
            else:
                self.preview_status.setText(
                    trans.t('label_preview_hint').format(
                        n=build_label_text(
                            self.format_edit.text(), self.text_edit.text(),
                            int(self.starting_value_spin.value()))))

        for widget in (self.format_edit, self.text_edit, self.range_edit):
            widget.textChanged.connect(_refresh_preview)
        for spin in (self.font_size_spin, self.starting_value_spin,
                     self.interval_spin, self.x_location_spin,
                     self.y_location_spin):
            spin.valueChanged.connect(_refresh_preview)
        self.font_family_combo.currentFontChanged.connect(_refresh_preview)
        self.transparent_bg_checkbox.toggled.connect(_refresh_preview)
        self._refresh_label_preview = _refresh_preview
        _refresh_preview()

        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        if accepted:
            self._apply_label_configuration()
        self.preview = None
        self.preview_status = None
        self._refresh_label_preview = None
        self._clear_dialog_state()

    def choose_background_color(self) -> None:
        color = QColorDialog.getColor(parent=self.window)

        if color.isValid() and self.bg_color_button is not None:
            self.bg_color_value = (color.blue(), color.green(), color.red())
            self._set_color_button_style(self.bg_color_button, color.red(), color.green(), color.blue())

    def choose_font_color(self) -> None:
        color = QColorDialog.getColor(parent=self.window)

        if color.isValid() and self.font_color_button is not None:
            self.font_color_value = (color.blue(), color.green(), color.red())
            self._set_color_button_style(self.font_color_button, color.red(), color.green(), color.blue())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_color_button_style(self, button: QPushButton, red: int, green: int, blue: int) -> None:
        brightness = (red * 299 + green * 587 + blue * 114) / 1000
        text_color = "black" if brightness > 150 else "white"
        button.setStyleSheet(
            f"background-color: rgb({red}, {green}, {blue}); "
            f"color: {text_color}; border: 1px solid #555; padding: 6px 12px; border-radius: 4px;"
        )

    def _apply_label_configuration(self) -> None:
        window = self.window

        config_dict = {
            "target_stack": self.target_stack_combo.currentIndex(),
            "format": self.format_edit.text(),
            "starting_value": self.starting_value_spin.value(),
            "interval": self.interval_spin.value(),
            "x_location": self.x_location_spin.value(),
            "y_location": self.y_location_spin.value(),
            "font_size": self.font_size_spin.value(),
            "font_family": self.font_family_combo.currentFont().family(),
            "text": self.text_edit.text(),
            "range": self.range_edit.text(),
            "transparent_bg": self.transparent_bg_checkbox.isChecked(),
            "bg_color": self.bg_color_value,
            "font_color": self.font_color_value,
        }

        target_stack_index = config_dict["target_stack"]

        if target_stack_index == 1:
            self._registered_adder.config.update_config(config_dict)
            self._registered_enabled = True
            self._sync_delete_actions()

            if window.registration_results and window.current_result_index >= 0:
                window.update_result_view(window.current_result_index)
            elif window.fusion_result is not None:
                window.display_fusion_result()
        else:
            self._input_adder.config.update_config(config_dict)
            self._input_enabled = True
            self._sync_delete_actions()

            if window.stack_images and window.current_display_index >= 0:
                window.update_source_view(window.current_display_index)

        # No modal "saved" box: the label is already visible on the canvas, so
        # a dialog on top of it only demands an extra click (and it blocked
        # scripted/automated runs outright, since nothing dismissed it).
        window.statusBar().showMessage(trans.t('msg_config_saved_text'), 6000)
        # Draw failure is otherwise indistinguishable from "nothing set".
        adder_error = (self._registered_adder.last_error
                       if target_stack_index == 1 else self._input_adder.last_error)
        if adder_error:
            show_warning_box(window, trans.t('msg_label_draw_failed_title'),
                             trans.t('msg_label_draw_failed_text'),
                             adder_error)

    def _clear_dialog_state(self) -> None:
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