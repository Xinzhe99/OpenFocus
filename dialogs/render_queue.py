"""Standalone non-modal Background Render Queue window for OpenFocus."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.algorithms import get_registration
from core.render_queue import (
    FINISHED_STATUSES,
    RenderQueueItem,
    RenderQueueManager,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
)
from locales import trans
from ui.context_help import register_help_target
from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette


def _status_pill_spec(item: RenderQueueItem) -> tuple[str, str]:
    """Return (text, semantic_role) for a job's status pill."""
    if item.status == STATUS_RUNNING:
        if item.progress > 0:
            text = trans.t("render_queue_status_running_pct").format(pct=item.progress)
        else:
            text = trans.t("render_queue_status_running")
        return text, "running"
    if item.status == STATUS_COMPLETED:
        return trans.t("render_queue_status_done").format(seconds=item.elapsed_seconds), "completed"
    if item.status == STATUS_FAILED:
        return trans.t("render_queue_status_failed"), "failed"
    if item.status == STATUS_CANCELLED:
        return trans.t("render_queue_status_cancelled"), "cancelled"
    return trans.t("render_queue_status_queued"), "queued"


def _item_subtitle(item: RenderQueueItem) -> str:
    """Secondary summary line showing alignment, options, result name, or error."""
    if item.status == STATUS_FAILED and item.error_message:
        return item.error_message

    parts = []
    if item.align_method and item.align_method != "none":
        try:
            align_label = trans.t(get_registration(item.align_method).label_key)
        except Exception:
            align_label = str(item.align_method)
    else:
        align_label = trans.t("check_align_none")
    parts.append(trans.t("render_queue_align_prefix").format(align=align_label))

    opts = item.options or {}
    if opts.get("exposure_match"):
        parts.append(trans.t("chk_exposure_match"))
    if opts.get("base_correct"):
        parts.append(trans.t("chk_base_correct"))
    if opts.get("reverse_order"):
        parts.append(trans.t("chk_reverse_order"))
    if opts.get("sharpen_output"):
        strength = opts.get("sharpen_strength")
        if strength is not None:
            parts.append(f"{trans.t('chk_sharpen_output')} ({int(strength)}%)")
        else:
            parts.append(trans.t("chk_sharpen_output"))

    if item.status == STATUS_COMPLETED and item.result_name:
        parts.insert(0, item.result_name)
    return " · ".join(parts)


class _QueueItemCard(QFrame):
    """Single calm card row inside RenderQueueDialog."""

    def __init__(
        self,
        item: RenderQueueItem,
        *,
        can_move_up: bool,
        can_move_down: bool,
        on_move_up,
        on_move_down,
        on_cancel,
        on_remove,
        on_view_result,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.job_id = item.job_id
        self.setObjectName("queueCardRunning" if item.status == STATUS_RUNNING else "queueCard")
        self.setProperty("controlId", f"queue_card_{item.job_id}")
        self.setFrameShape(QFrame.Shape.NoFrame)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        root.setSpacing(SPACE["sm"])

        # Top row: job title + status pill
        top_row = QHBoxLayout()
        top_row.setSpacing(SPACE["sm"])

        self.title_label = QLabel(item.label)
        self.title_label.setObjectName("queueCardTitle")
        self.title_label.setWordWrap(True)
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        top_row.addWidget(self.title_label, 1)

        pill_text, pill_role = _status_pill_spec(item)
        self.status_pill = QLabel(pill_text)
        self.status_pill.setObjectName(f"statusPill_{pill_role}")
        self.status_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_pill.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        top_row.addWidget(self.status_pill, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(top_row)

        # Subtitle: alignment / options / output filename / error
        subtitle_text = _item_subtitle(item)
        self.subtitle_label = QLabel(subtitle_text)
        self.subtitle_label.setObjectName(
            "queueCardError" if item.status == STATUS_FAILED else "queueCardSubtitle"
        )
        self.subtitle_label.setWordWrap(True)
        root.addWidget(self.subtitle_label)

        # Progress bar & stage text (active running job only)
        self.progress_bar: Optional[QProgressBar] = None
        self.stage_label: Optional[QLabel] = None
        if item.status == STATUS_RUNNING:
            progress_box = QVBoxLayout()
            progress_box.setSpacing(SPACE["xs"])
            self.progress_bar = QProgressBar()
            self.progress_bar.setObjectName("queueProgressBar")
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setFixedHeight(6)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, int(item.progress))))
            progress_box.addWidget(self.progress_bar)

            if item.progress_text:
                self.stage_label = QLabel(item.progress_text)
                self.stage_label.setObjectName("queueStageCaption")
                progress_box.addWidget(self.stage_label)
            else:
                self.stage_label = QLabel("")
                self.stage_label.setObjectName("queueStageCaption")
                self.stage_label.hide()
                progress_box.addWidget(self.stage_label)
            root.addLayout(progress_box)

        # Bottom action row
        actions_row = QHBoxLayout()
        actions_row.setSpacing(SPACE["xs"])
        actions_row.addStretch(1)

        self.btn_up: Optional[QPushButton] = None
        self.btn_down: Optional[QPushButton] = None
        self.btn_cancel: Optional[QPushButton] = None
        self.btn_remove: Optional[QPushButton] = None
        self.btn_view: Optional[QPushButton] = None

        if item.status == STATUS_QUEUED:
            self.btn_up = QPushButton("↑")
            self.btn_up.setObjectName("queueRowIconButton")
            self.btn_up.setToolTip(trans.t("render_queue_move_up"))
            self.btn_up.setAccessibleName(trans.t("render_queue_move_up"))
            self.btn_up.setEnabled(can_move_up)
            self.btn_up.clicked.connect(lambda: on_move_up(item.job_id))
            actions_row.addWidget(self.btn_up)

            self.btn_down = QPushButton("↓")
            self.btn_down.setObjectName("queueRowIconButton")
            self.btn_down.setToolTip(trans.t("render_queue_move_down"))
            self.btn_down.setAccessibleName(trans.t("render_queue_move_down"))
            self.btn_down.setEnabled(can_move_down)
            self.btn_down.clicked.connect(lambda: on_move_down(item.job_id))
            actions_row.addWidget(self.btn_down)

            self.btn_cancel = QPushButton(trans.t("render_queue_cancel"))
            self.btn_cancel.setObjectName("queueRowButton")
            self.btn_cancel.clicked.connect(lambda: on_cancel(item.job_id))
            actions_row.addWidget(self.btn_cancel)
        elif item.status == STATUS_RUNNING:
            self.btn_cancel = QPushButton(trans.t("render_queue_cancel"))
            self.btn_cancel.setObjectName("queueRowDangerButton")
            self.btn_cancel.clicked.connect(lambda: on_cancel(item.job_id))
            actions_row.addWidget(self.btn_cancel)
        elif item.status == STATUS_COMPLETED:
            self.btn_view = QPushButton(trans.t("render_queue_view_result"))
            self.btn_view.setObjectName("queueRowAccentButton")
            self.btn_view.clicked.connect(lambda: on_view_result(item.job_id))
            actions_row.addWidget(self.btn_view)

            self.btn_remove = QPushButton(trans.t("render_queue_remove"))
            self.btn_remove.setObjectName("queueRowButton")
            self.btn_remove.clicked.connect(lambda: on_remove(item.job_id))
            actions_row.addWidget(self.btn_remove)
        else:
            self.btn_remove = QPushButton(trans.t("render_queue_remove"))
            self.btn_remove.setObjectName("queueRowButton")
            self.btn_remove.clicked.connect(lambda: on_remove(item.job_id))
            actions_row.addWidget(self.btn_remove)

        root.addLayout(actions_row)

    def update_progress(self, percent: int, stage_text: str = "") -> None:
        pct = max(0, min(100, int(percent)))
        if self.progress_bar is not None:
            self.progress_bar.setValue(pct)
        if pct > 0:
            self.status_pill.setText(
                trans.t("render_queue_status_running_pct").format(pct=pct)
            )
        else:
            self.status_pill.setText(trans.t("render_queue_status_running"))
        if self.stage_label is not None:
            if stage_text:
                self.stage_label.setText(stage_text)
                self.stage_label.show()
            else:
                self.stage_label.hide()


def _build_dialog_stylesheet(theme: str) -> str:
    p = palette(theme)
    return f"""
    QDialog#renderQueueDialog {{
        background-color: {p['bg']};
        color: {p['text']};
    }}
    QFrame#queueHeaderBar {{
        background-color: {p['surface']};
        border-bottom: 1px solid {p['border']};
    }}
    QLabel#queueTitle {{
        color: {p['text']};
        font-size: {FONT['title']}pt;
        font-weight: {WEIGHT['semibold']};
        background: transparent;
    }}
    QLabel#queueSummary {{
        color: {p['textMuted']};
        font-size: {FONT['caption'] + 1}pt;
        background: transparent;
    }}
    QScrollArea#queueScroll {{
        background-color: {p['bg']};
        border: none;
    }}
    QWidget#queueListContainer {{
        background-color: {p['bg']};
    }}
    QFrame#queueCard {{
        background-color: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: {RADIUS['md']}px;
    }}
    QFrame#queueCardRunning {{
        background-color: {p['surface']};
        border: 1.5px solid {p['accent']};
        border-radius: {RADIUS['md']}px;
    }}
    QLabel#queueCardTitle {{
        color: {p['text']};
        font-size: {FONT['label']}pt;
        font-weight: {WEIGHT['semibold']};
        background: transparent;
    }}
    QLabel#queueCardSubtitle {{
        color: {p['textMuted']};
        font-size: {FONT['caption'] + 1}pt;
        background: transparent;
    }}
    QLabel#queueCardError {{
        color: {p['danger']};
        font-size: {FONT['caption'] + 1}pt;
        background: transparent;
    }}
    QLabel#queueStageCaption {{
        color: {p['textMuted']};
        font-size: {FONT['caption']}pt;
        background: transparent;
    }}
    QLabel#statusPill_queued {{
        background-color: {p['surfaceHover']};
        color: {p['textMuted']};
        border: 1px solid {p['border']};
        border-radius: {RADIUS['md']}px;
        padding: 2px 10px;
        font-size: {FONT['caption'] + 1}pt;
        font-weight: {WEIGHT['medium']};
    }}
    QLabel#statusPill_running {{
        background-color: {p['accentSoft']};
        color: {p['accentSoftText']};
        border: 1px solid {p['accent']};
        border-radius: {RADIUS['md']}px;
        padding: 2px 10px;
        font-size: {FONT['caption'] + 1}pt;
        font-weight: {WEIGHT['semibold']};
    }}
    QLabel#statusPill_completed {{
        background-color: {p['successSoft']};
        color: {p['success']};
        border: 1px solid {p['success']};
        border-radius: {RADIUS['md']}px;
        padding: 2px 10px;
        font-size: {FONT['caption'] + 1}pt;
        font-weight: {WEIGHT['semibold']};
    }}
    QLabel#statusPill_failed {{
        background-color: {p['dangerSoft']};
        color: {p['danger']};
        border: 1px solid {p['danger']};
        border-radius: {RADIUS['md']}px;
        padding: 2px 10px;
        font-size: {FONT['caption'] + 1}pt;
        font-weight: {WEIGHT['semibold']};
    }}
    QLabel#statusPill_cancelled {{
        background-color: {p['surfaceActive']};
        color: {p['textMuted']};
        border: 1px solid {p['border']};
        border-radius: {RADIUS['md']}px;
        padding: 2px 10px;
        font-size: {FONT['caption'] + 1}pt;
        font-weight: {WEIGHT['medium']};
    }}
    QProgressBar#queueProgressBar {{
        background-color: {p['surfaceActive']};
        border: none;
        border-radius: 3px;
    }}
    QProgressBar#queueProgressBar::chunk {{
        background-color: {p['accent']};
        border-radius: 3px;
    }}
    QPushButton#queuePrimaryButton {{
        background-color: {p['accent']};
        color: {p['textOnAccent']};
        border: 1px solid {p['accent']};
        border-radius: {RADIUS['sm']}px;
        padding: 6px 14px;
        font-weight: {WEIGHT['semibold']};
    }}
    QPushButton#queuePrimaryButton:hover {{
        background-color: {p['accentHover']};
        border-color: {p['accentHover']};
    }}
    QPushButton#queuePrimaryButton:pressed {{
        background-color: {p['accentPressed']};
    }}
    QPushButton#queuePrimaryButton:disabled {{
        background-color: {p['controlDisabled']};
        color: {p['textFaint']};
        border-color: {p['border']};
    }}
    QPushButton#queueToolbarButton {{
        background-color: {p['controlBg']};
        color: {p['text']};
        border: 1px solid {p['controlBorder']};
        border-radius: {RADIUS['sm']}px;
        padding: 6px 12px;
        font-weight: {WEIGHT['medium']};
    }}
    QPushButton#queueToolbarButton:hover {{
        background-color: {p['surfaceHover']};
    }}
    QPushButton#queueToolbarButton:pressed {{
        background-color: {p['surfaceActive']};
    }}
    QPushButton#queueToolbarButton:disabled {{
        background-color: {p['controlDisabled']};
        color: {p['textFaint']};
        border-color: {p['border']};
    }}
    QPushButton#queueRowButton, QPushButton#queueRowIconButton {{
        background-color: {p['controlBg']};
        color: {p['text']};
        border: 1px solid {p['borderStrong']};
        border-radius: {RADIUS['sm']}px;
        padding: 4px 10px;
        font-size: {FONT['caption'] + 1}pt;
    }}
    QPushButton#queueRowIconButton {{
        min-width: 28px;
        padding: 4px 6px;
    }}
    QPushButton#queueRowButton:hover, QPushButton#queueRowIconButton:hover {{
        background-color: {p['surfaceHover']};
    }}
    QPushButton#queueRowButton:disabled, QPushButton#queueRowIconButton:disabled {{
        background-color: {p['controlDisabled']};
        color: {p['textFaint']};
        border-color: {p['border']};
    }}
    QPushButton#queueRowAccentButton {{
        background-color: {p['accentSoft']};
        color: {p['accentSoftText']};
        border: 1px solid {p['accent']};
        border-radius: {RADIUS['sm']}px;
        padding: 4px 12px;
        font-size: {FONT['caption'] + 1}pt;
        font-weight: {WEIGHT['semibold']};
    }}
    QPushButton#queueRowAccentButton:hover {{
        background-color: {p['accent']};
        color: {p['textOnAccent']};
    }}
    QPushButton#queueRowDangerButton {{
        background-color: {p['dangerSoft']};
        color: {p['danger']};
        border: 1px solid {p['danger']};
        border-radius: {RADIUS['sm']}px;
        padding: 4px 10px;
        font-size: {FONT['caption'] + 1}pt;
        font-weight: {WEIGHT['medium']};
    }}
    QPushButton#queueRowDangerButton:hover {{
        background-color: {p['danger']};
        color: {p['textOnAccent']};
    }}
    QFrame#queueEmptyCard {{
        background-color: {p['surface']};
        border: 1px dashed {p['borderStrong']};
        border-radius: {RADIUS['md']}px;
    }}
    QLabel#queueEmptyTitle {{
        color: {p['text']};
        font-size: {FONT['subtitle']}pt;
        font-weight: {WEIGHT['semibold']};
        background: transparent;
    }}
    QLabel#queueEmptyHint {{
        color: {p['textMuted']};
        font-size: {FONT['body'] - 1}pt;
        background: transparent;
    }}
    """


class RenderQueueDialog(QDialog):
    """Non-modal dialog listing queued, running, completed, and failed jobs."""

    enqueue_current_requested = pyqtSignal()
    view_result_requested = pyqtSignal(str)

    def __init__(
        self,
        queue: RenderQueueManager,
        parent: Optional[QWidget] = None,
        *,
        theme: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setModal(False)
        self.setObjectName("renderQueueDialog")
        self.setProperty("controlId", "render_queue_dialog")
        self.setMinimumSize(640, 420)
        self.resize(740, 500)

        self._queue = queue
        self._cards: dict[str, _QueueItemCard] = {}
        self._theme = self._resolve_theme(theme)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header & toolbar
        header = QFrame()
        header.setObjectName("queueHeaderBar")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(SPACE["lg"], SPACE["md"], SPACE["lg"], SPACE["md"])
        header_layout.setSpacing(SPACE["sm"])

        title_row = QHBoxLayout()
        title_row.setSpacing(SPACE["sm"])
        self.lbl_title = QLabel()
        self.lbl_title.setObjectName("queueTitle")
        self.lbl_summary = QLabel()
        self.lbl_summary.setObjectName("queueSummary")
        title_row.addWidget(self.lbl_title)
        title_row.addStretch(1)
        title_row.addWidget(self.lbl_summary)
        header_layout.addLayout(title_row)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACE["sm"])

        self.btn_enqueue_current = QPushButton()
        self.btn_enqueue_current.setObjectName("queuePrimaryButton")
        self.btn_enqueue_current.setProperty("controlId", "btn_queue_enqueue_current")
        self.btn_enqueue_current.clicked.connect(self._on_enqueue_current)
        toolbar.addWidget(self.btn_enqueue_current)

        self.btn_pause_resume = QPushButton()
        self.btn_pause_resume.setObjectName("queueToolbarButton")
        self.btn_pause_resume.setProperty("controlId", "btn_queue_pause_resume")
        self.btn_pause_resume.clicked.connect(self._on_toggle_pause)
        toolbar.addWidget(self.btn_pause_resume)

        toolbar.addStretch(1)

        self.btn_clear_finished = QPushButton()
        self.btn_clear_finished.setObjectName("queueToolbarButton")
        self.btn_clear_finished.setProperty("controlId", "btn_queue_clear_finished")
        self.btn_clear_finished.clicked.connect(self._on_clear_finished)
        toolbar.addWidget(self.btn_clear_finished)

        header_layout.addLayout(toolbar)
        root.addWidget(header)

        # Scrollable job list
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("queueScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._list_container = QWidget()
        self._list_container.setObjectName("queueListContainer")
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(
            SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["lg"]
        )
        self._list_layout.setSpacing(SPACE["sm"])
        self._list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self._list_container)
        root.addWidget(self.scroll_area, 1)

        self._queue.queue_changed.connect(self.refresh)
        self._queue.job_progress.connect(self._on_job_progress)

        self.apply_theme(self._theme)
        self.retranslate()
        register_help_target(self, "render_queue")

    def _resolve_theme(self, explicit: Optional[str]) -> str:
        if explicit:
            return explicit
        parent = self.parentWidget()
        if parent is not None and getattr(parent, "ui_theme", None):
            return parent.ui_theme
        try:
            from ui.styles import current_theme
            return current_theme()
        except Exception:
            return "light"

    def apply_theme(self, theme: Optional[str] = None) -> None:
        self._theme = self._resolve_theme(theme)
        self.setStyleSheet(_build_dialog_stylesheet(self._theme))

    def retranslate(self) -> None:
        self.setWindowTitle(trans.t("render_queue_title"))
        self.lbl_title.setText(trans.t("render_queue_title"))
        self.btn_enqueue_current.setText(trans.t("render_queue_enqueue_current"))
        self.btn_clear_finished.setText(trans.t("render_queue_clear_finished"))
        self.refresh()

    def refresh(self) -> None:
        """Rebuild job cards and toolbar state from the queue model."""
        items = self._queue.items()
        paused = self._queue.is_paused()
        self.btn_pause_resume.setText(
            trans.t("render_queue_resume") if paused else trans.t("render_queue_pause")
        )
        has_finished = any(it.status in FINISHED_STATUSES for it in items)
        self.btn_clear_finished.setEnabled(has_finished)

        running_n = sum(1 for it in items if it.status == STATUS_RUNNING)
        queued_n = sum(1 for it in items if it.status == STATUS_QUEUED)
        done_n = sum(1 for it in items if it.status == STATUS_COMPLETED)
        if not items:
            summary = trans.t("render_queue_summary_idle")
        else:
            summary = trans.t("render_queue_summary_counts").format(
                running=running_n, queued=queued_n, done=done_n
            )
            if paused:
                summary = f"{trans.t('render_queue_summary_paused')} · {summary}"
        self.lbl_summary.setText(summary)

        self._cards.clear()
        while self._list_layout.count():
            child = self._list_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        if not items:
            empty = QFrame()
            empty.setObjectName("queueEmptyCard")
            empty_layout = QVBoxLayout(empty)
            empty_layout.setContentsMargins(
                SPACE["xl"], SPACE["xl"], SPACE["xl"], SPACE["xl"]
            )
            empty_layout.setSpacing(SPACE["xs"])
            title = QLabel(trans.t("render_queue_empty_title"))
            title.setObjectName("queueEmptyTitle")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint = QLabel(trans.t("render_queue_empty_hint"))
            hint.setObjectName("queueEmptyHint")
            hint.setWordWrap(True)
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_layout.addWidget(title)
            empty_layout.addWidget(hint)
            self._list_layout.addWidget(empty)
            self._list_layout.addStretch(1)
            return

        queued_ids = [it.job_id for it in items if it.status == STATUS_QUEUED]
        first_queued = queued_ids[0] if queued_ids else None
        last_queued = queued_ids[-1] if queued_ids else None

        for item in items:
            card = _QueueItemCard(
                item,
                can_move_up=item.status == STATUS_QUEUED and item.job_id != first_queued,
                can_move_down=item.status == STATUS_QUEUED and item.job_id != last_queued,
                on_move_up=lambda jid: self._queue.move_job(jid, -1),
                on_move_down=lambda jid: self._queue.move_job(jid, 1),
                on_cancel=self._queue.cancel_job,
                on_remove=self._queue.remove_job,
                on_view_result=self._on_view_result,
                parent=self._list_container,
            )
            self._cards[item.job_id] = card
            self._list_layout.addWidget(card)

        self._list_layout.addStretch(1)

    def _on_job_progress(self, job_id: str, percent: int, stage_text: str) -> None:
        card = self._cards.get(job_id)
        if card is not None:
            card.update_progress(percent, stage_text)

    def _on_enqueue_current(self) -> None:
        parent = self.parentWidget()
        enqueue_fn = getattr(parent, "enqueue_render_job", None)
        if callable(enqueue_fn):
            enqueue_fn()
        self.enqueue_current_requested.emit()

    def _on_toggle_pause(self) -> None:
        if self._queue.is_paused():
            self._queue.resume()
        else:
            self._queue.pause()

    def _on_clear_finished(self) -> None:
        self._queue.clear_finished()

    def _on_view_result(self, job_id: str) -> None:
        self.view_result_requested.emit(job_id)

    def card_for(self, job_id: str) -> Optional[_QueueItemCard]:
        """Return the card widget for a job id (used by tests & callers)."""
        return self._cards.get(job_id)
