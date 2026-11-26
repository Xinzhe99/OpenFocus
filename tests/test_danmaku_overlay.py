"""Offscreen Qt tests for widgets.danmaku_overlay."""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest

from widgets.danmaku_overlay import (
    DANMAKU_COOLDOWN_SEC,
    DANMAKU_MAX_CHARS,
    REPORT_REASONS,
    DanmakuOverlay,
    DanmakuSettings,
    is_floating_danmaku_comment,
)


@dataclass
class _StubComment:
    id: str
    work_id: str
    content: str
    author_id: str = "u1"
    author_nickname: str = "Alice"
    is_reply: bool = False
    parent_id: str | None = None
    is_danmaku: bool = True
    upvotes: int = 2
    downvotes: int = 0
    viewer_vote: int = 0


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def test_danmaku_settings_normalization_and_serialization():
    raw = DanmakuSettings(
        enabled=True,
        position="invalid",
        font_size=99,
        opacity=2.5,
        density="ultra",
    )
    norm = raw.normalized()
    assert norm.position == "top"
    assert norm.font_size == 32
    assert norm.opacity == 1.0
    assert norm.density == "medium"

    rt = DanmakuSettings.from_dict(norm.to_dict())
    assert rt == norm


def test_author_replies_never_float_as_danmaku(qapp):
    overlay = DanmakuOverlay()
    overlay.resize(640, 360)

    comments = [
        _StubComment(id="c1", work_id="w1", content="Crisp antennae detail!", is_reply=False),
        _StubComment(
            id="c2",
            work_id="w1",
            content="Thank you, shot with 10x objective!",
            is_reply=True,
            parent_id="c1",
            is_danmaku=False,
        ),
        {
            "id": "c3",
            "work_id": "w1",
            "content": "Another nested reply without is_reply flag",
            "parent_id": "c1",
        },
        {
            "id": "c4",
            "work_id": "w1",
            "content": "Non-danmaku top comment",
            "is_reply": False,
            "is_danmaku": False,
        },
        {
            "id": "c5",
            "work_id": "w1",
            "content": "Awesome lighting",
            "is_reply": False,
        },
    ]

    assert is_floating_danmaku_comment(comments[0]) is True
    assert is_floating_danmaku_comment(comments[1]) is False
    assert is_floating_danmaku_comment(comments[2]) is False
    assert is_floating_danmaku_comment(comments[3]) is False
    assert is_floating_danmaku_comment(comments[4]) is True
    # Defensive rejection of is_author_reply=True even if parent_id/is_reply omitted
    assert is_floating_danmaku_comment(
        {"id": "c_author_reply", "work_id": "w1", "content": "Author reply", "is_author_reply": True}
    ) is False

    overlay.set_comments(comments, work_id="w1")
    all_ids = {b.comment_id for b in overlay.active_bullets + overlay.queued_comments}
    assert all_ids == {"c1", "c5"}

    # add_comment must also strictly ignore replies
    rejected = overlay.add_comment(
        _StubComment(id="c6", work_id="w1", content="Author reply", is_reply=True, parent_id="c1")
    )
    assert rejected is None
    assert "c6" not in {b.comment_id for b in overlay.active_bullets + overlay.queued_comments}
    overlay.deleteLater()


def test_hover_pauses_only_hovered_bullet_and_click_opens_popover(qapp):
    overlay = DanmakuOverlay(settings=DanmakuSettings(density="high"))
    overlay.resize(720, 400)
    overlay.show()
    qapp.processEvents()

    overlay.set_comments(
        [
            _StubComment(id="c1", work_id="w1", content="First floating bullet", upvotes=3, downvotes=1),
            _StubComment(id="c2", work_id="w1", content="Second floating bullet", upvotes=0, downvotes=0),
        ],
        work_id="w1",
    )
    assert len(overlay.active_bullets) == 2
    b1, b2 = overlay.active_bullets[0], overlay.active_bullets[1]
    b1.x, b1.y = 120.0, 40.0
    b2.x, b2.y = 260.0, 120.0

    # Hover over b1 -> pauses b1 only
    hover_pos = QPointF(b1.x + 12.0, b1.y + 10.0)
    move_ev = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        hover_pos,
        hover_pos,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    overlay.mouseMoveEvent(move_ev)
    assert b1.hovered is True
    assert b2.hovered is False

    x1_before, x2_before = b1.x, b2.x
    overlay.step_animation(0.5)
    assert b1.x == x1_before
    assert b2.x < x2_before

    # Click b1 -> opens action popover, pins b1 paused, emits popover_opened
    opened_events: list[bool] = []
    closed_events: list[bool] = []
    votes: list[tuple[str, int]] = []
    reports: list[tuple[str, str]] = []
    overlay.popover_opened.connect(lambda: opened_events.append(True))
    overlay.popover_closed.connect(lambda: closed_events.append(True))
    overlay.comment_voted.connect(lambda cid, v: votes.append((cid, v)))
    overlay.comment_reported.connect(lambda cid, r: reports.append((cid, r)))

    QTest.mouseClick(overlay.windowHandle(), Qt.MouseButton.LeftButton, pos=hover_pos.toPoint())
    assert overlay.popover.isVisible() is True
    assert b1.paused is True
    assert len(opened_events) == 1

    # Upvote -> updates live count + feedback animation
    overlay.popover.btn_upvote.click()
    assert votes[-1] == ("c1", 1)
    assert b1.upvotes == 4
    assert b1.viewer_vote == 1
    assert b1.feedback_text == "+1"
    assert b1.feedback_alpha == 1.0
    assert "4" in overlay.popover.btn_upvote.text()

    # Switch to downvote -> decrements upvote, increments downvote
    overlay.popover.btn_downvote.click()
    assert votes[-1] == ("c1", -1)
    assert b1.upvotes == 3
    assert b1.downvotes == 1 + 1
    assert b1.viewer_vote == -1
    assert b1.feedback_text == "-1"

    overlay.confirm_vote("c1", 2, 8, 4)
    assert (b1.viewer_vote, b1.upvotes, b1.downvotes) == (1, 8, 4)
    assert overlay.popover.isVisible() is True
    assert b1.paused is True
    assert overlay.popover.btn_upvote.isChecked() is True
    assert overlay.popover.btn_downvote.isChecked() is False
    assert "8" in overlay.popover.btn_upvote.text()
    assert "4" in overlay.popover.btn_downvote.text()

    # Report -> uses canonical English snake_case keys and closes popover
    overlay.popover.btn_report.click()
    assert overlay.popover.report_panel.isVisible() is True
    assert tuple(overlay.popover.reason_buttons.keys()) == REPORT_REASONS
    overlay.popover.reason_buttons["spam_advertising"].click()
    assert reports == [("c1", "spam_advertising")]
    assert overlay.popover.isVisible() is False
    assert b1.paused is False
    assert len(closed_events) == 1

    # Paint check offscreen
    pm = overlay.grab()
    assert not pm.isNull()
    overlay.deleteLater()


def test_composer_50_char_limit_and_10s_cooldown(qapp):
    clock = _FakeClock(start=1000.0)
    overlay = DanmakuOverlay(clock=clock)
    overlay.resize(640, 360)
    overlay.show()
    qapp.processEvents()

    opened: list[bool] = []
    closed: list[bool] = []
    submitted: list[str] = []
    errors: list[str] = []
    overlay.composer_opened.connect(lambda: opened.append(True))
    overlay.composer_closed.connect(lambda: closed.append(True))
    overlay.comment_submitted.connect(submitted.append)
    overlay.composer.validation_failed.connect(errors.append)

    composer = overlay.composer
    assert composer.max_chars == DANMAKU_MAX_CHARS == 50
    assert composer.cooldown_sec == DANMAKU_COOLDOWN_SEC == 10
    assert composer.input_edit.objectName() == "gallery_danmaku_input"
    assert composer.input_edit.property("controlId") == "gallery_danmaku_input"
    assert composer.input_edit.maxLength() == 50

    # Open composer
    overlay.open_composer()
    assert composer.isVisible() is True
    assert len(opened) == 1
    assert composer.lbl_counter.text() == "0/50"
    assert composer.btn_send.isEnabled() is False

    # Live counter updates
    text_50 = "a" * 50
    composer.input_edit.setText(text_50)
    assert composer.lbl_counter.text() == "50/50"
    assert composer.btn_send.isEnabled() is True

    # Programmatic >50 chars is rejected
    assert composer.submit_text("b" * 51) is False
    assert errors[-1] == "danmaku_too_long"
    assert submitted == []

    # Valid send succeeds, closes composer, starts 10s cooldown
    assert composer.submit() is True
    assert submitted == [text_50]
    assert composer.isVisible() is False
    assert len(closed) == 1
    assert composer.cooldown_remaining() == 10.0

    # Re-open during cooldown: cooldown display shown, send disabled
    clock.advance(4.0)
    overlay.open_composer()
    assert len(opened) == 2
    assert composer.lbl_cooldown.isVisible() is True
    assert "6" in composer.lbl_cooldown.text()
    composer.input_edit.setText("Too soon")
    assert composer.btn_send.isEnabled() is False
    assert composer.submit() is False
    assert errors[-1] == "danmaku_cooldown"
    assert len(submitted) == 1

    # Cancel via ESC closes composer and emits composer_closed
    assert qapp.focusWidget() is composer.input_edit
    QTest.keyClick(composer.input_edit, Qt.Key.Key_Escape)
    assert composer.isVisible() is False
    assert len(closed) == 2

    # Advance past 10s cooldown -> send succeeds
    clock.advance(6.2)
    overlay.open_composer()
    assert composer.cooldown_remaining() == 0.0
    assert composer.lbl_cooldown.isVisible() is False
    composer.input_edit.setText("Focus stack looks sharp!")
    assert composer.btn_send.isEnabled() is True
    composer.btn_send.click()
    assert submitted[-1] == "Focus stack looks sharp!"
    assert len(closed) == 3

    overlay.deleteLater()


def test_danmaku_positions_and_enabled_toggle(qapp):
    overlay = DanmakuOverlay(settings=DanmakuSettings(enabled=True, position="top", density="low"))
    overlay.resize(640, 400)
    overlay.set_comments(
        [_StubComment(id="c1", work_id="w1", content="Top lane")],
        work_id="w1",
    )
    assert len(overlay.active_bullets) == 1
    y_top = overlay.active_bullets[0].y

    overlay.set_settings(DanmakuSettings(enabled=True, position="middle", density="medium"))
    y_mid = overlay.active_bullets[0].y
    overlay.set_settings(DanmakuSettings(enabled=True, position="bottom", density="high"))
    y_bottom = overlay.active_bullets[0].y
    assert y_top < y_mid < y_bottom

    # Disabling danmaku stops animation step and paints cleanly
    x_before = overlay.active_bullets[0].x
    overlay.set_enabled(False)
    overlay.step_animation(0.5)
    assert overlay.active_bullets[0].x == x_before
    pm = overlay.grab()
    assert not pm.isNull()
    overlay.deleteLater()


def test_switching_popover_between_bullets_emits_opened_once(qapp):
    overlay = DanmakuOverlay(settings=DanmakuSettings(density="high"))
    overlay.resize(640, 360)
    overlay.show()
    qapp.processEvents()

    overlay.set_comments(
        [
            _StubComment(id="c1", work_id="w1", content="Bullet A"),
            _StubComment(id="c2", work_id="w1", content="Bullet B"),
        ],
        work_id="w1",
    )
    opened: list[bool] = []
    closed: list[bool] = []
    overlay.popover_opened.connect(lambda: opened.append(True))
    overlay.popover_closed.connect(lambda: closed.append(True))

    assert overlay.open_popover_for("c1") is True
    assert overlay.open_popover_for("c2") is True
    # Transitioned closed -> open once; switching bullet must not double-emit
    assert len(opened) == 1
    assert overlay.active_bullets[0].paused is False
    assert overlay.active_bullets[1].paused is True

    overlay.popover.dismiss()
    assert len(closed) == 1
    assert overlay.active_bullets[1].paused is False
    overlay.deleteLater()


def test_short_canvas_queues_comments_instead_of_overlapping_lanes(qapp):
    overlay = DanmakuOverlay(settings=DanmakuSettings(position="top", density="high"))
    overlay.resize(720, 420)
    overlay.show()
    qapp.processEvents()
    comments = [
        _StubComment(id=f"c{i}", work_id="w1", content="A wide comment across the showcase")
        for i in range(5)
    ]
    overlay.set_comments(comments, work_id="w1")
    assert len(overlay.active_bullets) >= 2
    for item in overlay.active_bullets:
        item.x = 40

    overlay.resize(320, 200)
    qapp.processEvents()
    active, = overlay.active_bullets
    assert active.y >= 14
    assert active.y + active.height <= 14 + (200 - 102) * 0.42
    assert len(overlay.queued_comments) == 4
    next_comment = overlay.queued_comments[0]
    overlay.step_animation(0.6)
    assert overlay.active_bullets == [active]
    assert overlay.queued_comments[0] is next_comment

    overlay.step_animation(20)
    assert overlay.active_bullets == [next_comment]
    assert len(overlay.queued_comments) == 4
    overlay.deleteLater()


def test_danmaku_translation_modes_hover_tooltip_and_popover(qapp):
    from PyQt6.QtGui import QMouseEvent
    from locales import trans

    raw = DanmakuSettings(translation_mode="unknown_mode").normalized()
    assert raw.translation_mode == "auto"
    assert DanmakuSettings.from_dict({"translation_mode": "same_language"}).translation_mode == "same_language"
    assert DanmakuSettings(translation_mode="original").to_dict()["translation_mode"] == "original"

    prev_lang = trans.current_lang
    try:
        trans.set_language("zh")
        overlay = DanmakuOverlay(settings=DanmakuSettings(density="high", translation_mode="auto"))
        overlay.resize(720, 420)
        overlay.show()
        qapp.processEvents()

        comments = [
            {
                "id": "c_foreign",
                "work_id": "w1",
                "content": "Incredible compound eye detail!",
                "src_lang": "en",
                "translated_content": "复眼细节太绝了！",
                "translated_lang": "zh",
                "author_nickname": "MacroPro",
                "upvotes": 2,
            },
            {
                "id": "c_native",
                "work_id": "w1",
                "content": "堆栈非常干净",
                "src_lang": "zh",
            },
            {
                "id": "c_unknown",
                "work_id": "w1",
                "content": "Unknown origin keeps visible",
            },
        ]
        overlay.set_comments(comments, work_id="w1")
        bullets = {item.comment_id: item for item in overlay.active_bullets}
        assert len(bullets) == 3
        text_foreign, is_trans = bullets["c_foreign"].bullet_text("auto", "zh")
        assert is_trans is True
        assert text_foreign == "复眼细节太绝了！"
        assert "\U0001f310" not in text_foreign

        # Smooth 60fps PreciseTimer & DPR pixmap raster cache
        assert overlay._anim_timer.interval() <= 16
        assert overlay._anim_timer.timerType() == Qt.TimerType.PreciseTimer
        pm = overlay._bullet_pixmap(bullets["c_native"], "堆栈非常干净", 2.0)
        assert not pm.isNull()
        assert pm.devicePixelRatio() == 2.0

        # Hover over translated bullet sets original tooltip; moving off clears it.
        b_foreign = bullets["c_foreign"]
        b_foreign.x = 60.0
        hover_pt = QPointF(b_foreign.x + 12.0, b_foreign.y + b_foreign.height / 2.0)
        move_ev = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            hover_pt,
            overlay.mapToGlobal(hover_pt),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        overlay.mouseMoveEvent(move_ev)
        assert "Incredible compound eye detail!" in overlay.toolTip()

        # Inline action rects inside the hovered pill allow direct upvote/downvote/comment/report
        actions = overlay._inline_action_rects(b_foreign)
        assert set(actions.keys()) == {"upvote", "downvote", "comment", "report"}
        votes_direct: list[tuple[str, int]] = []
        drawer_requests: list[bool] = []
        overlay.comment_voted.connect(lambda cid, v: votes_direct.append((cid, v)))
        overlay.comments_drawer_requested.connect(lambda: drawer_requests.append(True))

        up_center = actions["upvote"].center()
        overlay.mouseMoveEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseMove,
                up_center,
                overlay.mapToGlobal(up_center),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        assert overlay._hovered_action == "upvote"
        overlay.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                up_center,
                overlay.mapToGlobal(up_center),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        assert votes_direct == [("c_foreign", 1)]
        assert b_foreign.upvotes == 3
        assert overlay.popover.isVisible() is False

        cmt_center = overlay._inline_action_rects(b_foreign)["comment"].center()
        overlay.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                cmt_center,
                overlay.mapToGlobal(cmt_center),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        assert drawer_requests == [True]

        rep_center = overlay._inline_action_rects(b_foreign)["report"].center()
        overlay.mousePressEvent(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                rep_center,
                overlay.mapToGlobal(rep_center),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        assert overlay.popover.isVisible() is True
        assert overlay.popover.report_panel.isVisible() is True
        overlay.popover.dismiss()
        # Reset vote back to original state for the remaining assertions
        overlay._on_popover_vote("c_foreign", 0)
        overlay.mouseMoveEvent(move_ev)

        assert overlay.open_popover_for("c_foreign") is True
        assert "复眼细节太绝了！" in overlay.popover.lbl_preview.text()
        assert overlay.popover.lbl_original.isVisible() is True
        assert "Incredible compound eye detail!" in overlay.popover.lbl_original.text()

        empty_pt = QPointF(5.0, 5.0)
        leave_move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            empty_pt,
            overlay.mapToGlobal(empty_pt),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        overlay.mouseMoveEvent(leave_move)
        assert overlay.toolTip() == ""

        # A locale change dismisses the pinned preview and its hover tooltip.
        overlay.mouseMoveEvent(move_ev)
        assert overlay.toolTip()
        assert b_foreign.paused is True
        trans.set_language("en")
        assert overlay.toolTip() == ""
        assert overlay.popover.isVisible() is False
        assert b_foreign.paused is False
        assert overlay.open_popover_for("c_foreign") is True
        assert "Incredible compound eye" in overlay.popover.lbl_preview.text()
        assert overlay.popover.lbl_original.isVisible() is False

        trans.set_language("zh")
        votes: list[tuple[str, int]] = []
        overlay.comment_voted.connect(lambda cid, vote: votes.append((cid, vote)))
        assert overlay.open_popover_for("c_foreign") is True
        overlay.popover.btn_upvote.click()
        assert votes == [("c_foreign", 1)]
        assert (b_foreign.viewer_vote, b_foreign.upvotes) == (1, 3)

        # Switch live to "original" -> preserves all 3 bullets, disables translated decoration.
        overlay.set_settings(DanmakuSettings(density="high", translation_mode="original"))
        orig_map = {item.comment_id: item for item in overlay.active_bullets}
        assert set(orig_map.keys()) == {"c_foreign", "c_native", "c_unknown"}
        assert orig_map["c_foreign"].bullet_text("original", "zh") == ("Incredible compound eye detail!", False)
        assert (orig_map["c_foreign"].viewer_vote, orig_map["c_foreign"].upvotes) == (1, 3)

        # Switch live to "same_language" -> filters out known foreign "en", keeps "zh" + unknown.
        overlay.set_settings(DanmakuSettings(density="high", translation_mode="same_language"))
        same_ids = {item.comment_id for item in overlay.active_bullets}
        assert same_ids == {"c_native", "c_unknown"}
        assert overlay.open_popover_for("c_foreign") is False

        overlay.set_settings(DanmakuSettings(density="high", translation_mode="auto"))
        restored = next(item for item in overlay.active_bullets if item.comment_id == "c_foreign")
        assert (restored.viewer_vote, restored.upvotes) == (1, 3)
        assert overlay.open_popover_for("c_foreign") is True
        assert overlay.popover.btn_upvote.isChecked() is True
        overlay.popover.btn_upvote.click()
        assert votes == [("c_foreign", 1), ("c_foreign", 0)]
        assert (restored.viewer_vote, restored.upvotes) == (0, 2)
        overlay.deleteLater()
    finally:
        trans.set_language(prev_lang)


def test_danmaku_sent_toast_and_forced_own_spawn(qapp):
    clock_now = [1.0]
    overlay = DanmakuOverlay(settings=DanmakuSettings(density="low"), clock=lambda: clock_now[0])
    overlay.resize(640, 360)
    overlay.show()
    # Fill both lanes of low density
    overlay.set_comments(
        [
            {"id": "d1", "content": "First lane occupied"},
            {"id": "d2", "content": "Second lane occupied"},
        ],
        work_id="w1",
    )
    assert len(overlay.active_bullets) == 2

    # User submits their own comment: forced spawn makes it appear inside viewport immediately with is_own=True
    own = overlay.add_comment({"id": "local-mine", "content": "刚发送的弹幕！"}, immediate=True, is_own=True)
    assert own is not None
    assert own.is_own is True
    assert own in overlay.active_bullets
    assert 0.0 <= own.x < overlay.width()

    overlay.show_toast("弹幕已发送", duration_sec=2.0)
    assert overlay._toast_remaining == 2.0
    assert overlay._anim_timer.isActive()
    overlay.step_animation(1.2)
    assert abs(overlay._toast_remaining - 0.8) < 1e-6
    overlay.step_animation(1.0)
    assert overlay._toast_remaining == 0.0
    assert overlay._toast_text == ""
    overlay.deleteLater()
