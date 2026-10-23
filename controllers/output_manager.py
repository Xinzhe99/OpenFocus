import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QPixmap, QImage
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMenu, QMessageBox

from controllers.export_manager import get_imwrite_params
from utils import (show_error_box, show_message_box, show_success_box,
                   show_warning_box)
from utils.settings_store import get_drag_export_format, set_drag_export_format
from utils import scalebar
from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
from utils.image_utils import to_display_uint8, cv2_to_pixmap
from utils.image_utils import imwrite_auto
from locales import trans

# Item-data roles: the folder of an align-only result, and the stable
# provenance record of every output (identity, source context, export state).
ALIGNED_DIR_ROLE = int(Qt.ItemDataRole.UserRole) + 1
OUTPUT_RECORD_ROLE = int(Qt.ItemDataRole.UserRole) + 2


@dataclass
class OutputRecord:
    """Stable identity for one entry in the output history.

    The list row is a view concern and changes when entries are deleted or
    inserted; `output_id` is what a retouch session, a parent link and an
    export decision refer to.  `context` is the immutable render provenance
    (None for batch/legacy/aligned entries, which therefore cannot be
    retouched by guessing).
    """

    output_id: str
    image: Any
    label: str
    context: Any = None
    kind: str = "fusion"
    parent_id: Optional[str] = None
    exported: bool = True
    # Identity of the pixels that were last written. Comparing it with the
    # current image detects "changed since export" without hashing megapixels.
    exported_image: Any = None
    export_path: str = ""

    @classmethod
    def create(cls, image, label: str, *, context=None, kind: str = "fusion",
               parent_id: Optional[str] = None, exported: bool = True):
        return cls(output_id=uuid.uuid4().hex, image=image, label=label,
                   context=context, kind=kind, parent_id=parent_id,
                   exported=exported)

    @property
    def retouchable(self) -> bool:
        return (self.context is not None
                and int(getattr(self.context, "source_count", 0)) > 0
                and self.image is not None)

    #: Outputs that exist only in this session and must not be discarded
    #: silently: retouched buffers and results adopted from the compare view.
    PROTECTED_KINDS = ("retouch", "compare")

    @property
    def protected(self) -> bool:
        """An output whose pixels have not been written to disk yet."""
        return self.kind in self.PROTECTED_KINDS and not self.exported

    @property
    def changed_since_export(self) -> bool:
        """True when the image differs from the one written to disk.

        Fusion results are immutable, so identity is enough and costs nothing;
        a retouched result whose buffer was replaced after export is the case
        this reports.
        """
        if not self.exported:
            return False
        return (self.exported_image is not None
                and self.exported_image is not self.image)


def _thumbnail_bgr(image, size: int):
    """Downscale a result frame to a `size`-px square thumbnail, in uint8.

    Resampling first keeps the colour conversion and the pixmap upload
    proportional to the icon instead of to the full-resolution frame.
    """
    display = to_display_uint8(image)
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    h, w = display.shape[:2]
    if h <= 0 or w <= 0:
        return display
    scale = min(size / float(w), size / float(h))
    if scale >= 1.0:
        return display
    return cv2.resize(display, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


class OutputManager:
    """Encapsulates output panel interactions and state updates."""
    # Every entry keeps a full-resolution copy; unbounded history plus
    # compare-all (8 results per run) exhausted memory on large stacks.
    MAX_HISTORY = 20

    def __init__(self, window: Any):
        self.window = window

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def update_output_count(self) -> None:
        count = self.window.output_list.count()
        self.window.output_label.setText(trans.t("label_output").format(count))

    # ------------------------------------------------------------------
    # Stable output identity
    # ------------------------------------------------------------------
    def record_for_item(self, item) -> Optional[OutputRecord]:
        if item is None:
            return None
        record = item.data(OUTPUT_RECORD_ROLE)
        return record if isinstance(record, OutputRecord) else None

    def record_for_row(self, row: int) -> Optional[OutputRecord]:
        if row < 0:
            return None
        return self.record_for_item(self.window.output_list.item(row))

    def update_record_label(self, row: int, label: str) -> None:
        record = self.record_for_row(row)
        if record is not None:
            record.label = label

    def mark_exported(self, row: Optional[int] = None,
                      path: str = "") -> None:
        """Record that the output at `row` was written to a file."""
        if row is None:
            row = self.window.output_list.currentRow()
        record = self.record_for_row(row)
        if record is not None:
            record.exported = True
            record.exported_image = record.image
            if path:
                record.export_path = path
            self._refresh_output_tooltip(row)

    def mark_selected_exported(self, path: str = "") -> None:
        row = self.window.output_list.currentRow()
        self.mark_exported(row if row >= 0 else 0, path=path)

    def _refresh_output_tooltip(self, row: int) -> None:
        """Keep the tooltip in step with the record's export state."""
        window = self.window
        item = window.output_list.item(row) if row >= 0 else None
        record = self.record_for_row(row) if row >= 0 else None
        if item is None or record is None:
            return
        lines = []
        if record.export_path:
            lines.append(record.export_path)
        if record.protected:
            lines.append(trans.t("retouch_unexported"))
        elif record.changed_since_export:
            lines.append(trans.t("msg_export_after_change"))
        if lines:
            item.setToolTip("\n".join(lines))

    def protected_records(self) -> list[OutputRecord]:
        records = []
        for row in range(self.window.output_list.count()):
            record = self.record_for_row(row)
            if record is not None and record.protected:
                records.append(record)
        return records

    def _insert_output(self, image, label: str, record: Optional[OutputRecord],
                       tooltip: str = "", trim: bool = True) -> QListWidgetItem:
        """Insert one entry at row 0, keeping pixels, item and record aligned.

        `trim=False` lets a caller finish its own commit (for example a
        publication that must restore the previous history exactly if the
        display step fails) before older entries are evicted.
        """
        window = self.window
        if image is not None:
            window.fusion_results.insert(0, image)
        if record is not None:
            record.label = label
        item = QListWidgetItem(label)
        if record is not None:
            item.setData(OUTPUT_RECORD_ROLE, record)
        if tooltip:
            item.setToolTip(tooltip)
        if image is not None:
            try:
                # Thumbnail from a 40 px resample, not from the full frame:
                # converting 12 MP just to fill a 40x40 icon cost ~35 ms of
                # copy + colour convert + pixmap upload per render.
                icon = QIcon(cv2_to_pixmap(_thumbnail_bgr(image, 40)))
                item.setIcon(icon)
            except Exception:
                pass
        window.output_list.insertItem(0, item)
        # Select only the new entry: setCurrentRow() extends an existing
        # selection, which left every previous output highlighted too (and
        # disabled the single-item actions — Save As, Retouch).
        window.output_list.clearSelection()
        window.output_list.setCurrentRow(0)
        if trim:
            self._trim_history()
        self.update_output_count()
        return item

    def _remove_row(self, row: int) -> None:
        window = self.window
        if row < 0 or row >= window.output_list.count():
            return
        window.output_list.takeItem(row)
        if 0 <= row < len(window.fusion_results):
            window.fusion_results.pop(row)
        self.update_output_count()

    def _trim_history(self) -> None:
        """Enforce MAX_HISTORY without silently discarding protected work.

        Retouched outputs that were never exported stay in the list even if
        that leaves it above the cap; the user must decide to export or
        discard them.  Unprotected entries are evicted oldest-first.
        """
        window = self.window
        while window.output_list.count() > self.MAX_HISTORY:
            row = window.output_list.count() - 1
            while row >= 0:
                record = self.record_for_row(row)
                if record is None or not record.protected:
                    break
                row -= 1
            if row < 0:
                break
            window.output_list.takeItem(row)
            if 0 <= row < len(window.fusion_results):
                window.fusion_results.pop(row)

    def update_output_list_for_fusion(self, context=None) -> None:
        window = self.window

        fusion_filename = window.export_manager.generate_default_filename()

        image = None
        record = None
        if window.fusion_result is not None:
            image = window.fusion_result.copy()
            record = OutputRecord.create(image, fusion_filename,
                                         context=context, kind="fusion")
        self._insert_output(image, fusion_filename, record)

    def add_batch_outputs(self, paths) -> int:
        """Append finished batch files to the output list.

        A batch used to end in a message box only: the results were on disk but
        the output panel stayed empty, so they could not be inspected, compared
        with the wipe tool, or retouched like a normal render. Each file is
        read back once (at a display-friendly size cap) so every existing
        output action keeps working unchanged.

        Returns how many entries were added.
        """
        import os

        import cv2

        window = self.window
        added = 0
        for path in paths:
            try:
                if not os.path.isfile(path):
                    continue
                image = cv2.imdecode(np.fromfile(path, np.uint8),
                                     cv2.IMREAD_UNCHANGED)
                if image is None:
                    continue
                label = os.path.basename(path)
                # Batch files are already on disk and have no in-memory
                # provenance; mark them as such instead of guessing frames.
                record = OutputRecord.create(image, label, kind="batch",
                                             exported=True)
                self._insert_output(image, label, record, tooltip=path)
                added += 1
            except Exception:
                continue
        if added:
            self.update_output_count()
            self.display_specific_fusion_result(window.fusion_results[0])
        return added

    # ------------------------------------------------------------------
    # Retouched outputs
    # ------------------------------------------------------------------
    def _retouch_label_parts(self):
        """Localized prefix/suffix and regex for "name · retouch N"."""
        suffix = trans.t("retouch_output_suffix")
        prefix, sep, _rest = suffix.partition("{n}")
        if not sep:
            return suffix, None, None
        prefix = prefix.strip()
        pattern = re.compile(
            r"^(.*) · " + re.escape(prefix) + r" (\d+)$")
        return suffix, prefix, pattern

    def next_retouch_label(self, base_label: str) -> str:
        """Name the new output without making the number part of its identity."""
        suffix, prefix, pattern = self._retouch_label_parts()
        root = (base_label or trans.t("retouch_current")).strip()
        if pattern is not None:
            match = pattern.match(root)
            if match:
                root = match.group(1)
        used = set()
        if pattern is not None:
            for row in range(self.window.output_list.count()):
                item = self.window.output_list.item(row)
                text = item.text() if item is not None else ""
                match = pattern.match(text)
                if match and match.group(1) == root:
                    used.add(int(match.group(2)))
        n = 1
        while n in used:
            n += 1
        try:
            return f"{root} · {suffix.format(n=n)}"
        except (KeyError, ValueError, IndexError):
            return f"{root} {n}"

    def add_retouched_output(self, base_record: OutputRecord,
                             image) -> OutputRecord:
        """Publish a retouch buffer as a new output, leaving the base intact."""
        if base_record is None or image is None:
            raise ValueError("a base output and an edited image are required")
        image = np.ascontiguousarray(image.copy())
        label = self.next_retouch_label(base_record.label)
        record = OutputRecord.create(
            image, label, context=base_record.context, kind="retouch",
            parent_id=base_record.output_id, exported=False)
        window = self.window
        previous = getattr(window, "fusion_result", None)
        previous_row = window.output_list.currentRow()
        item = None
        try:
            window.fusion_result = image
            # trim=False: history eviction happens only after the new result
            # is committed. Evicting first meant a failed display could still
            # have dropped an older output permanently.
            item = self._insert_output(
                image, label, record, tooltip=trans.t("retouch_unexported"),
                trim=False)
            if self.display_specific_fusion_result(image) is False:
                raise RuntimeError(trans.t("msg_display_fusion_failed_text"))
        except Exception:
            # Publication is atomic: a failed display must not leave a
            # half-created entry, a switched view, or a changed selection.
            if item is not None:
                row = window.output_list.row(item)
                if row >= 0:
                    self._remove_row(row)
            window.fusion_result = previous
            if previous_row >= 0:
                window.output_list.setCurrentRow(previous_row)
            self.update_output_count()
            raise
        self._trim_history()
        self.update_output_count()
        return record

    def publish_compare_output(self, record: OutputRecord) -> OutputRecord:
        """Add a full-frame result adopted from the method comparison.

        Uses the same transactional insert as a retouch publication: the entry
        and the view commit together, and history eviction happens only after
        both succeeded.
        """
        if record is None or record.image is None:
            raise ValueError("a result image is required")
        window = self.window
        previous = getattr(window, "fusion_result", None)
        previous_row = window.output_list.currentRow()
        item = None
        try:
            window.fusion_result = record.image
            item = self._insert_output(
                record.image, record.label, record,
                tooltip=trans.t("retouch_unexported"), trim=False)
            if self.display_specific_fusion_result(record.image) is False:
                raise RuntimeError(trans.t("msg_display_fusion_failed_text"))
        except Exception:
            if item is not None:
                row = window.output_list.row(item)
                if row >= 0:
                    self._remove_row(row)
            window.fusion_result = previous
            if previous_row >= 0:
                window.output_list.setCurrentRow(previous_row)
            self.update_output_count()
            raise
        self._trim_history()
        self.update_output_count()
        return record

    # ------------------------------------------------------------------
    # Export protection for unexported retouched outputs
    # ------------------------------------------------------------------
    def export_records(self, records) -> bool:
        """Write each record into a chosen folder; True only if all succeeded.

        Existing files on disk are never overwritten silently: a directory
        export picks a free name instead (a protective export used to replace
        an unrelated file that happened to share the generated name). Every
        item's outcome is reported, not just a generic failure.
        """
        records = [r for r in records if r is not None and r.image is not None]
        if not records:
            return True
        window = self.window
        folder = QFileDialog.getExistingDirectory(
            window, trans.t("action_save"), get_last_dialog_dir(),
            QFileDialog.Option.ShowDirsOnly)
        if not folder:
            return False
        set_last_dialog_dir(folder)
        used = set()
        written = []
        failed = []
        for record in records:
            name = window.export_manager.filename_from_label(record.label)
            stem, ext = os.path.splitext(name)
            ext = ext or ".png"
            path = _unique_path(folder, stem, ext, used)
            candidate = os.path.basename(path)
            try:
                image = window.label_manager.prepare_bgr_image(
                    "output", record.image, 0)
                image = scalebar.burn(
                    image, scalebar.export_cfg(window, target="output"))
                success = imwrite_auto(path, image, get_imwrite_params(ext))
                record_error = ""
            except Exception as exc:  # pylint: disable=broad-except
                success = False
                record_error = str(exc)
            if success:
                used.add(candidate)
                record.exported = True
                record.exported_image = record.image
                record.export_path = path
                written.append(path)
            else:
                # A name that failed to write stays unused so a retry of the
                # same batch does not collide with it.
                failed.append((path, record_error))
        self._report_export_result(written, failed, folder)
        return not failed

    def _report_export_result(self, written, failed, folder) -> None:
        """One summary for a multi-file export, with the failing paths."""
        window = self.window
        if not failed:
            window.statusBar().showMessage(
                trans.t("msg_export_count_done").format(
                    n=len(written), path=folder).replace("\n", " "), 8000)
            return
        detail = "\n".join(
            f"{path}: {error or trans.t('msg_save_failed_info_write')}"
            for path, error in failed)
        show_error_box(
            window, trans.t("msg_save_failed_title"),
            trans.t("msg_export_partial_text").format(
                ok=len(written), failed=len(failed)),
            detail)

    def _confirm_protected_removal(self, records) -> bool:
        """Ask what to do with unexported retouched outputs before discarding."""
        protected = [r for r in records if r is not None and r.protected]
        if not protected:
            return True
        from utils.ui_utils import style_message_box
        box = QMessageBox(self.window)
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(trans.t("msg_warning"))
        box.setText(trans.t("retouch_unexported_text").format(
            count=len(protected)))
        export_btn = box.addButton(trans.t("action_save"),
                                   QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton(trans.t("btn_discard_changes"),
                                    QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(trans.t("btn_cancel"),
                                   QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is export_btn:
            return self.export_records(protected)
        return clicked is discard_btn

    def confirm_unexported_on_exit(self) -> bool:
        """Return False when the close must be cancelled to protect pixels."""
        protected = self.protected_records()
        if not protected:
            return True
        return self._confirm_protected_removal(protected)

    def sync_output_slider_from_list(self, row: int) -> None:
        if row >= 0:
            self.window.result_slider.setValue(row)

    # ------------------------------------------------------------------
    # Context menu & list operations
    # ------------------------------------------------------------------
    def show_output_context_menu(self, position: QPoint) -> None:
        window = self.window
        menu = QMenu(window.output_list)

        delete_action = QAction(trans.t("action_delete"), window)
        delete_action.triggered.connect(self.delete_selected_output_images)
        menu.addAction(delete_action)

        target_item = window.output_list.itemAt(position)
        if target_item is not None:
            # The action must follow the entry that was right-clicked, not a
            # different row that happened to stay selected.
            window.output_list.clearSelection()
            target_item.setSelected(True)
            window.output_list.setCurrentItem(target_item)
        else:
            target_item = window.output_list.currentItem()

        save_as_action = QAction(trans.t("action_save_as"), window)
        save_as_action.triggered.connect(
            lambda: self.save_output_image_as(target_item))
        save_as_action.setEnabled(target_item is not None)
        menu.addAction(save_as_action)

        retouch_action = QAction(trans.t("action_retouch"), window)
        retouch_action.triggered.connect(
            lambda: window.show_retouch_dialog(target_item))
        retouch_action.setEnabled(target_item is not None)
        menu.addAction(retouch_action)

        # Drag-out export format submenu (persisted per user)
        drag_menu = menu.addMenu(trans.t("menu_drag_format"))
        current_fmt = get_drag_export_format(window)
        group = QActionGroup(drag_menu)
        group.setExclusive(True)
        for fmt, label in ((".jpg", "JPG"), (".png", "PNG"), (".webp", "WebP"), (".tif", "TIFF")):
            act = QAction(label, drag_menu)
            act.setCheckable(True)
            act.setChecked(fmt == current_fmt)
            act.triggered.connect(lambda _checked, f=fmt: set_drag_export_format(window, f))
            group.addAction(act)
            drag_menu.addAction(act)

        menu.exec(window.output_list.mapToGlobal(position))

    def delete_output_image(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.output_list.row(item)
        if row < 0:
            return
        record = self.record_for_item(item)
        if record is not None and not self._confirm_protected_removal([record]):
            return

        self._remove_row(row)

        # Output history changed: resync the wipe B-side slider range
        if getattr(window, "wipe_active", False) and hasattr(window, "refresh_wipe_controls"):
            window.refresh_wipe_controls()

        if window.fusion_results:
            new_index = min(row, len(window.fusion_results) - 1)
            window.fusion_result = window.fusion_results[new_index]
            window.output_list.setCurrentRow(new_index)
            self.display_specific_fusion_result(window.fusion_result)
        else:
            window.fusion_result = None
            window.lbl_result_img.clear()
            window.lbl_result_img.setText(trans.t("result_hint"))
            window.result_control_bar.setVisible(False)

    def export_all_results(self) -> None:
        """Save every result in the output history to a user-chosen folder."""
        window = self.window
        if window.fusion_result is None and not window.fusion_results:
            show_warning_box(window, trans.t("msg_warning"), trans.t("msg_export_all_none"))
            return

        folder = QFileDialog.getExistingDirectory(
            window, trans.t("menu_export_all"),
            get_last_dialog_dir(), QFileDialog.Option.ShowDirsOnly)
        if not folder:
            return
        set_last_dialog_dir(folder)

        base = window.export_manager.generate_default_filename()
        base = os.path.splitext(os.path.basename(base))[0] or "OpenFocus"
        count = 0
        failed = []
        results = list(window.fusion_results)  # newest first
        total = len(results)
        used = set()
        for i, image in enumerate(results):
            # A single output is one frame: its caption number must not
            # depend on where it sits in the history (it was the row index,
            # so "Fig 1" became "Fig 2" in Export All).
            labeled = window.label_manager.prepare_bgr_image(
                "output", image, 0)
            path = _unique_path(folder, f"{base}_{total - i:02d}", ".png",
                                used)
            from utils.image_utils import imwrite_auto
            if imwrite_auto(path, scalebar.burn(
                    labeled, scalebar.export_cfg(window, target="output"))):
                used.add(os.path.basename(path))
                self.mark_exported(i, path=path)
                count += 1
            else:
                failed.append((path, ""))
        if failed:
            self._report_export_result(
                [None] * count, failed, folder)
        else:
            show_success_box(
                window, trans.t("msg_success"),
                trans.t("msg_export_all_done_text").format(
                    count=count, folder=folder))

    def save_output_image_as(self, item: QListWidgetItem | None) -> None:
        window = self.window
        if item is None:
            return

        # An align-only entry stages a FOLDER: "Save as" must copy the whole
        # stack. It used to fall through to the single-image path and offered
        # a name like "已对齐图像栈（13 帧）.png".
        folder = item.data(ALIGNED_DIR_ROLE)
        if folder and os.path.isdir(folder):
            window.export_manager.save_aligned_stack_as(folder, item.text())
            return

        row = window.output_list.row(item)
        if row < 0:
            return

        default_filename = window.export_manager.filename_from_label(item.text())
        file_path, selected_filter = QFileDialog.getSaveFileName(
            window,
            trans.t("action_save"),
            default_filename,
            "PNG Files (*.png);;JPG Files (*.jpg);;WebP Files (*.webp);;"
            "Bitmap Files (*.bmp);;TIFF Files (*.tif *.tiff);;"
            "All Files (*)",
        )

        if not file_path:
            return

        fallback_ext = os.path.splitext(default_filename)[1]
        file_path = window.export_manager.normalize_export_path(
            file_path,
            fallback_extension=fallback_ext if fallback_ext else None,
            selected_filter=selected_filter,
        )

        try:
            if row < len(window.fusion_results):
                image_to_save = window.label_manager.prepare_bgr_image(
                    "output", window.fusion_results[row], 0
                )

                ext = os.path.splitext(file_path)[1].lower()
                params = get_imwrite_params(ext)
                success = imwrite_auto(file_path, scalebar.burn(
                    image_to_save,
                    scalebar.export_cfg(window, target="output")), params)
                if success:
                    self.mark_exported(row, path=file_path)
                    window.statusBar().showMessage(
                        trans.t("msg_image_saved_info").format(path=file_path).replace("\n", " "),
                        8000)
                else:
                    show_error_box(
                        window,
                        trans.t("msg_save_failed_title"),
                        trans.t("msg_save_failed_text"),
                        trans.t("msg_save_failed_info_write"),
                    )
            elif window.fusion_result is not None and row == 0:
                image_to_save = window.label_manager.prepare_bgr_image(
                    "output", window.fusion_result, 0
                )

                ext = os.path.splitext(file_path)[1].lower()
                params = get_imwrite_params(ext)
                success = imwrite_auto(file_path, scalebar.burn(
                    image_to_save,
                    scalebar.export_cfg(window, target="output")), params)
                if success:
                    self.mark_exported(row, path=file_path)
                    window.statusBar().showMessage(
                        trans.t("msg_image_saved_info").format(path=file_path).replace("\n", " "),
                        8000)
                else:
                    show_error_box(
                        window,
                        trans.t("msg_save_failed_title"),
                        trans.t("msg_save_failed_text"),
                        trans.t("msg_save_failed_info_write"),
                    )
            else:
                show_warning_box(
                    window,
                    trans.t("msg_warning"),
                    trans.t("msg_no_valid_image_text"),
                    trans.t("msg_no_valid_image_info"),
                )
        except cv2.error as exc:
            show_error_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_opencv").format(error=str(exc)),
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_unexpected").format(error=str(exc)),
            )

    def delete_selected_output_images(self) -> None:
        window = self.window
        selected_items = window.output_list.selectedItems()
        if not selected_items:
            return

        confirm = QMessageBox(window)
        from utils.ui_utils import style_message_box
        style_message_box(confirm)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle(trans.t("msg_confirm_delete_title"))
        confirm.setText(trans.t("msg_confirm_delete_output_text").format(count=len(selected_items)))
        confirm.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return

        records = [self.record_for_item(item) for item in selected_items]
        if not self._confirm_protected_removal(records):
            return

        rows = [window.output_list.row(item) for item in selected_items]
        rows.sort(reverse=True)

        for row in rows:
            self._remove_row(row)

        if window.fusion_results:
            min_row = min(rows) if rows else 0
            new_index = min(min_row, len(window.fusion_results) - 1)
            window.fusion_result = window.fusion_results[new_index]
            window.output_list.setCurrentRow(new_index)
            self.display_specific_fusion_result(window.fusion_result)
        else:
            window.fusion_result = None
            window.lbl_result_img.clear()
            window.lbl_result_img.setText(trans.t("result_hint"))
            window.result_control_bar.setVisible(False)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def display_specific_fusion_result(self, fusion_image) -> bool:
        if fusion_image is None:
            return False

        window = self.window

        try:
            display_image = to_display_uint8(
                window.label_manager.display_bgr_image(
                    "output", fusion_image, 0,
                    scale_cfg=scalebar.export_cfg(window,
                                                 target="output")))
            window.lbl_result_img.set_display_pixmap(cv2_to_pixmap(display_image))
            window.result_control_bar.setVisible(False)
            window.lbl_result_info.setText("-- / --")

            # Keep the wipe view's right side following the latest result
            if getattr(window, "wipe_active", False):
                window._update_wipe_images()
            return True
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_display_error_title"),
                trans.t("msg_display_fusion_failed_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
            return False

    def display_output_image_in_result_view(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.output_list.row(item)
        if row < 0:
            return

        if row < len(window.fusion_results):
            self.display_specific_fusion_result(window.fusion_results[row])
        elif window.fusion_result is not None:
            self.show_fusion_result()

    def show_fusion_result(self) -> None:
        window = self.window
        if window.fusion_result is None:
            return

        self.display_specific_fusion_result(window.fusion_result)

    def show_registration_result(self, index: int) -> None:
        window = self.window

        if window.fusion_result is not None:
            return
        if not window.registration_results:
            return
        if not 0 <= index < len(window.registration_results):
            return

        window.current_result_index = index

        try:
            image = window.registration_results[index]
            image = to_display_uint8(
                window.label_manager.apply_labels_to_registered_image(image, index))

            window.lbl_result_img.set_display_pixmap(cv2_to_pixmap(image))
            window.lbl_result_info.setText(f"{index + 1} / {len(window.registration_results)}")
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_display_error_title"),
                trans.t("msg_display_registration_failed_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )

def publish_aligned_stack(window, frames) -> None:
    """Stage aligned frames in a dated folder and list it as one entry.

    An align-only render produces no fused image, so the output panel stayed
    empty even though the aligned stack IS the result. The frames are written
    once with labels and the scale bar burned in, exactly as a manual export
    would; the folder name carries the stack, the alignment method and the
    time, following the same convention as the fused-image names.

    The first frame is also inserted into `fusion_results` so the existing
    output actions keep working on it, and the entry carries the folder path
    so "Save As" copies the whole stack.
    """
    import datetime
    import os
    import tempfile

    import cv2

    from utils import scalebar
    from utils.image_utils import imwrite_auto

    if not frames:
        return

    label_manager = getattr(window, "label_manager", None)
    names = list(getattr(window, "image_filenames", []) or [])

    # Same naming convention as the fused-image folders:
    # <stack>_<timestamp>_<fusion>_<alignment>
    stack_name = "Stack"
    src = getattr(window, "current_folder_path", None)
    if src:
        stack_name = os.path.basename(src.rstrip("/\\")) or "Stack"
    align = getattr(window, "current_alignment_method", lambda: "none")()
    align_tag = {"homography": "Homo", "ecc": "ECC",
                 "openfocus": "OF"}.get(align, "NoAlign")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Name carries everything the user asked to identify a staged stack:
    # frame count, source stack, alignment method, time. The output entry
    # shows this exact name, and "Save As" creates a sub-folder with it.
    folder_name = (f"OpenFocus_{stack_name}_{align_tag}_"
                   f"{len(frames):03d}frames_{stamp}")
    folder = os.path.join(tempfile.gettempdir(), folder_name)
    os.makedirs(folder, exist_ok=True)

    saved = 0
    for index, frame in enumerate(frames):
        # Staged WITHOUT annotations: this folder is a working copy that the
        # user later saves as an aligned stack. Baking the current captions
        # here meant editing or deleting a label never reached those files,
        # because the save path only copied them.
        image = frame
        name = (names[index] if index < len(names)
                else f"aligned_{index + 1:04d}.png")
        if imwrite_auto(os.path.join(folder, name), image):
            saved += 1
    if not saved:
        return

    # The entry text IS the staged folder name: the UI label and the
    # "Save As" sub-folder match exactly, which the user asked for.
    item = QListWidgetItem(folder_name)
    item.setToolTip(trans.t("output_aligned_stack").format(n=saved)
                    + "\n" + folder)
    item.setData(ALIGNED_DIR_ROLE, folder)
    item.setData(OUTPUT_RECORD_ROLE, OutputRecord.create(
        frames[0], folder_name, kind="aligned", exported=True))
    try:
        icon = QIcon(cv2_to_pixmap(_thumbnail_bgr(frames[0], 40)))
        item.setIcon(icon)
    except Exception:
        pass
    window.output_list.insertItem(0, item)
    window.fusion_results.insert(0, frames[0])
    window.output_list.clearSelection()
    window.output_list.setCurrentRow(0)
    # The "输出 (N)" header must count align-only results too, or the counter
    # stayed at 0 while entries appeared.
    window.output_manager.update_output_count()
    window.aligned_stack_dir = folder
    window.statusBar().showMessage(
        trans.t("output_aligned_staged").format(path=folder), 8000)


def _unique_path(folder: str, stem: str, ext: str, used: set) -> str:
    """First free `<stem>[_N]<ext>` in `folder`.

    `used` tracks names chosen during this call; the disk check prevents
    overwriting a file that already exists (including one from a previous
    run of the same protective export).
    """
    import os
    candidate = f"{stem}{ext}"
    counter = 2
    while candidate in used or os.path.exists(os.path.join(folder, candidate)):
        candidate = f"{stem}_{counter}{ext}"
        counter += 1
    return os.path.join(folder, candidate)
