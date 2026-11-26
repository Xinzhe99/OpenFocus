"""Offscreen Qt tests for GalleryShowcaseWidget and GalleryScreensaverWindow."""
from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QFrame, QPushButton

from ui.control_surface import control_id
from widgets.danmaku_overlay import REPORT_REASONS, DanmakuSettings
from widgets.gallery_screensaver import GalleryScreensaverWindow
from widgets.gallery_showcase import (
    GalleryShowcaseWidget,
    ShowcaseWork,
    StaticBlurBackgroundCache,
    format_exif_summary,
)


class _FakeClock:
    def __init__(self, start: float = 500.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def _make_pixmap(width: int, height: int, color: QColor) -> QPixmap:
    img = QImage(width, height, QImage.Format.Format_RGBA8888)
    img.fill(color)
    return QPixmap.fromImage(img)


def _sample_works() -> list[ShowcaseWork]:
    portrait = _make_pixmap(180, 320, QColor(40, 140, 220))
    landscape = _make_pixmap(320, 180, QColor(210, 110, 45))
    return [
        ShowcaseWork(
            id="w1",
            title="Dewdrop Beetle",
            author_id="author_1",
            author_nickname="MacroMaster",
            source_badge="following",
            openfocus_badge=True,
            allow_download=True,
            like_count=12,
            viewer_liked=False,
            comment_count=2,
            exif={
                "camera": "OM-1",
                "lens": "M.Zuiko 90mm Macro",
                "aperture": "f/5.6",
                "shutter": "1/160s",
                "iso": "200",
            },
            pixmap=portrait,
            comments=[
                {
                    "id": "c1",
                    "work_id": "w1",
                    "author_nickname": "ViewerA",
                    "content": "Incredible compound eye sharpness!",
                    "is_reply": False,
                    "is_danmaku": True,
                },
                {
                    "id": "c2",
                    "work_id": "w1",
                    "author_nickname": "MacroMaster",
                    "content": "Thanks! 45 frames stacked in OpenFocus.",
                    "is_reply": True,
                    "parent_id": "c1",
                    "is_danmaku": False,
                    "is_author": True,
                    "can_delete": True,
                },
            ],
        ),
        ShowcaseWork(
            id="w2",
            title="Crystal Cluster",
            author_id="author_2",
            author_nickname="MineralLab",
            source_badge="ad",
            openfocus_badge=False,
            allow_download=False,
            like_count=5,
            viewer_liked=True,
            comment_count=0,
            exif={"make": "Sony", "model": "A7R V", "lens": "FE 90mm Macro"},
            pixmap=landscape,
        ),
        ShowcaseWork(
            id="w3",
            title="Remote Uncached Work",
            author_id="author_3",
            author_nickname="FieldWalker",
            source_badge=None,
            openfocus_badge=False,
            allow_download=True,
            image_url="https://cdn.openfocus.top/works/w3.webp",
            pixmap=None,
        ),
    ]


def _click_canvas_control(showcase, control):
    point = control.mapTo(showcase, control.rect().center())
    assert showcase.childAt(point) is control, control.objectName()
    QTest.mouseClick(showcase.windowHandle(), Qt.MouseButton.LeftButton, pos=point)


def test_static_blur_cache_reused_across_repaints_and_zero_black_bars(qapp):
    cache = StaticBlurBackgroundCache(max_entries=8)
    showcase = GalleryShowcaseWidget(blur_cache=cache)
    showcase.resize(640, 360)
    showcase.show()
    qapp.processEvents()

    works = _sample_works()
    showcase.set_works([works[0]])
    qapp.processEvents()

    assert cache.compute_count == 1

    # Multiple repaints at the same size must NEVER recompute Gaussian blur
    grab1 = showcase.grab()
    showcase.repaint()
    grab2 = showcase.grab()
    showcase.repaint()
    qapp.processEvents()
    assert cache.compute_count == 1
    assert not grab1.isNull() and not grab2.isNull()

    # Zero black bars on left/right edges of portrait image in 16:9 canvas:
    # portrait image is vivid blue (40, 140, 220), so blurred side margin is bluish, not black
    edge_img = grab1.toImage()
    left_pixel = edge_img.pixelColor(8, edge_img.height() // 2)
    assert left_pixel.blue() > 80
    assert left_pixel.blue() > left_pixel.red()

    # Resize to a new size computes once for the new (work_id, size)
    showcase.resize(800, 450)
    qapp.processEvents()
    assert cache.compute_count == 2

    # Resizing back hits the existing static blur cache without recomputing
    hits_before = cache.cache_hits
    showcase.resize(640, 360)
    qapp.processEvents()
    assert cache.compute_count == 2
    assert cache.cache_hits > hits_before

    # Replacing the pixmap at another size must also invalidate the previous backdrop.
    showcase.resize(800, 450)
    assert showcase.set_work_pixmap("w1", _make_pixmap(180, 320, QColor(220, 45, 30)))
    showcase.resize(640, 360)
    qapp.processEvents()
    replacement = showcase.grab().toImage().pixelColor(8, 180)
    assert replacement.red() > 100
    assert replacement.red() > replacement.blue()

    showcase.deleteLater()


def test_fusion_busy_pauses_prefetch_and_rotates_only_cached_slides(qapp):
    prefetched: list[str] = []
    showcase = GalleryShowcaseWidget(prefetch_callback=prefetched.append)
    showcase.resize(640, 360)
    works = _sample_works()

    # Start with fusion busy BEFORE loading works -> zero prefetch calls
    showcase.set_fusion_busy(True)
    assert showcase.prefetch_paused is True
    showcase.set_works(works, start_index=2)
    assert prefetched == []

    # Rotating while fusion busy skips uncached w3 and stays on locally cached slides (w1, w2)
    assert showcase.current_work is not None and showcase.current_work.id == "w1"
    showcase.next_slide()
    assert showcase.current_work.id == "w2"
    showcase.next_slide()
    assert showcase.current_work.id == "w1"
    showcase.next_slide()
    assert showcase.current_work.id == "w2"
    assert prefetched == []

    # Clearing fusion busy resumes prefetching for upcoming uncached slide w3
    showcase.set_fusion_busy(False)
    assert showcase.prefetch_paused is False
    assert prefetched == ["w3"]

    showcase.show_slide(2)
    assert showcase.current_work.id == "w3"
    showcase.set_fusion_busy(True)
    assert showcase.current_work.id == "w1"
    assert prefetched == ["w3"]

    showcase.deleteLater()


def test_danmaku_composer_pauses_5s_timer_and_send_or_cancel_resumes(qapp):
    clock = _FakeClock(start=200.0)
    showcase = GalleryShowcaseWidget(clock=clock)
    showcase.resize(720, 420)
    showcase.show()
    qapp.processEvents()

    works = _sample_works()[:2]
    showcase.set_works(works)
    assert showcase.slide_interval_sec == 5.0
    assert showcase.slide_timer.interval() == 5000
    assert showcase.slide_timer.isActive() is True

    submitted: list[tuple[str, str, str]] = []
    showcase.comment_submitted.connect(lambda wid, text, pid: submitted.append((wid, text, pid)))

    # Clicking [Comment/Danmaku] opens composer and strictly pauses the 5s timer
    showcase.btn_comment.click()
    assert showcase.danmaku_overlay.is_composing() is True
    assert showcase.slide_timer.isActive() is False
    assert "danmaku_compose" in showcase.active_pause_reasons()

    # Sending a danmaku closes composer, resumes timer, emits signal, floats top-level bullet
    showcase.danmaku_overlay.composer.input_edit.setText("Super crisp stack!")
    showcase.danmaku_overlay.composer.btn_send.click()
    assert showcase.danmaku_overlay.is_composing() is False
    assert showcase.slide_timer.isActive() is True
    assert "danmaku_compose" not in showcase.active_pause_reasons()
    assert submitted == [("w1", "Super crisp stack!", "")]
    assert any(
        b.content == "Super crisp stack!"
        for b in showcase.danmaku_overlay.active_bullets
    )

    # Escape from the embedded input cancels the draft and resumes rotation.
    clock.advance(12.0)
    showcase.btn_comment.click()
    assert showcase.slide_timer.isActive() is False
    input_edit = showcase.danmaku_overlay.composer.input_edit
    input_edit.setText("Cancel this draft")
    QTest.keyClick(input_edit, Qt.Key.Key_Escape)
    assert input_edit.text() == ""
    assert showcase.danmaku_overlay.is_composing() is False
    assert showcase.slide_timer.isActive() is True

    # Reference-counted overlapping pause reasons: composer + comment drawer
    showcase.open_danmaku_composer()
    showcase.open_comment_drawer()
    assert showcase.active_pause_reasons() == frozenset({"danmaku_compose", "comment_drawer"})
    assert showcase.slide_timer.isActive() is False
    showcase.close_danmaku_composer()
    assert showcase.slide_timer.isActive() is False
    showcase.close_comment_drawer()
    assert showcase.slide_timer.isActive() is True

    # Custom slide interval
    showcase.set_slide_interval(7.5)
    assert showcase.slide_timer.interval() == 7500

    showcase.deleteLater()


def test_open_danmaku_draft_retains_work_on_feed_refresh_and_cancels_when_removed(qapp):
    clock = _FakeClock()
    showcase = GalleryShowcaseWidget(clock=clock)
    showcase.set_works(_sample_works()[:2])
    showcase.show()
    qapp.processEvents()
    composer = showcase.danmaku_overlay.composer
    submitted = []
    showcase.comment_submitted.connect(lambda *args: submitted.append(args))

    showcase.open_danmaku_composer()
    composer.input_edit.setText("Draft for the beetle")
    w1, w2 = _sample_works()[:2]
    w1.pixmap = None
    showcase.set_works([w2, w1], start_index=0)
    assert showcase.current_work.id == "w1"
    assert composer.input_edit.text() == "Draft for the beetle"
    showcase.set_fusion_busy(True)
    assert showcase.current_work.id == "w2"
    qapp.processEvents()
    _click_canvas_control(showcase, composer.btn_send)
    assert submitted == [("w1", "Draft for the beetle", "")]
    assert w1.comments[-1]["content"] == "Draft for the beetle"
    assert w2.comments == []

    clock.advance(11)
    showcase.set_fusion_busy(False)
    for remaining in ([w2], []):
        showcase.set_works([w1, w2])
        showcase.open_danmaku_composer()
        composer.input_edit.setText("Draft for a removed work")
        showcase.set_works(remaining)
        assert not composer.is_open()
        assert composer.input_edit.text() == ""
        assert not composer.submit_text("Draft for a removed work")
        assert composer.cooldown_remaining() == 0
        assert len(submitted) == 1
        assert "danmaku_compose" not in showcase.active_pause_reasons()
    showcase.deleteLater()


def test_canvas_clicks_reach_composer_and_popover_over_drawer_at_narrow_sizes(qapp):
    showcase = GalleryShowcaseWidget(clock=_FakeClock())
    showcase.set_danmaku_limits(cooldown_sec=0)
    showcase.show()
    overlay = showcase.danmaku_overlay
    composer = overlay.composer
    popover = overlay.popover
    submitted = []
    votes = []
    reports = []
    showcase.comment_submitted.connect(lambda *args: submitted.append(args))
    showcase.comment_voted.connect(lambda *args: votes.append(args))
    showcase.comment_reported.connect(lambda *args: reports.append(args))

    for width, height in ((640, 360), (320, 200)):
        showcase.resize(width, height)
        showcase.set_works(_sample_works()[:2])
        showcase.open_danmaku_composer()
        qapp.processEvents()
        _click_canvas_control(showcase, composer.btn_all_comments)
        assert showcase.comment_drawer.is_open()
        composer.input_edit.setText(f"Canvas {width}")
        _click_canvas_control(showcase, composer.btn_send)
        assert submitted[-1] == ("w1", f"Canvas {width}", "")
        assert not composer.is_open()
        assert showcase.comment_drawer.is_open()

        showcase.open_danmaku_composer()
        composer.input_edit.setText("Cancel this draft")
        qapp.processEvents()
        _click_canvas_control(showcase, composer.btn_cancel)
        assert not composer.is_open()
        assert composer.input_edit.text() == ""

        bullet = next(item for item in overlay.active_bullets if item.comment_id == "c1")
        bullet.x = 30
        point = QPoint(49, int(bullet.y + bullet.height / 2))
        assert showcase.childAt(point) is overlay
        QTest.mouseClick(showcase.windowHandle(), Qt.MouseButton.LeftButton, pos=point)
        assert popover.current_item is bullet
        _click_canvas_control(showcase, popover.btn_upvote)
        assert votes[-1] == ("c1", 1)
        _click_canvas_control(showcase, popover.btn_report)
        qapp.processEvents()
        reason = popover.reason_buttons["other"]
        popover.report_panel.ensureWidgetVisible(reason)
        qapp.processEvents()
        assert showcase.rect().contains(popover.geometry())
        _click_canvas_control(showcase, reason)
        assert reports[-1] == ("c1", "other")
        assert not popover.isVisible()
        assert showcase.comment_drawer.is_open()
        showcase.close_comment_drawer()

    assert len(submitted) == len(votes) == len(reports) == 2
    showcase.deleteLater()


def test_pending_danmaku_is_visible_but_only_server_comments_can_vote_or_report(qapp):
    showcase = GalleryShowcaseWidget()
    showcase.resize(640, 360)
    showcase.set_works([ShowcaseWork(id="w1")])
    showcase.show()
    qapp.processEvents()
    votes = []
    reports = []
    showcase.comment_voted.connect(lambda *args: votes.append(args))
    showcase.comment_reported.connect(lambda *args: reports.append(args))

    showcase.open_danmaku_composer()
    overlay = showcase.danmaku_overlay
    overlay.composer.input_edit.setText("Pending comment")
    qapp.processEvents()
    _click_canvas_control(showcase, overlay.composer.btn_send)
    pending, = overlay.active_bullets
    assert pending.content == "Pending comment"
    assert pending.comment_id.startswith("local-")

    showcase.open_comment_drawer()
    card = showcase.comment_drawer.findChild(QFrame, "commentCard")
    assert card is not None
    actions = card.findChildren(QPushButton)
    assert len(actions) == 2
    assert all(not button.isEnabled() for button in actions)
    for button in actions:
        button.click()

    assert overlay.open_popover_for(pending)
    popover = overlay.popover
    for button in (popover.btn_upvote, popover.btn_downvote, popover.btn_report):
        assert not button.isEnabled()
        button.click()
    assert not popover.report_panel.isVisible()
    assert votes == reports == []

    showcase.set_work_comments("w1", [{"id": "c-confirmed", "content": pending.content}])
    assert overlay.open_popover_for("c-confirmed")
    assert popover.btn_upvote.isEnabled()
    assert popover.btn_report.isEnabled()
    popover.btn_upvote.click()
    assert votes == [("c-confirmed", 1)]
    showcase.deleteLater()


def test_spec_3_3_hud_layout_order_allow_download_gating_and_control_ids(qapp):
    showcase = GalleryShowcaseWidget()
    showcase.resize(800, 450)
    showcase.show()
    qapp.processEvents()

    works = _sample_works()
    showcase.set_works(works, start_index=0)
    qapp.processEvents()

    # Stable objectName and controlId properties for openfocus ctl automation
    expected_ids = [
        (showcase, "gallery_showcase"),
        (showcase.btn_like, "btn_gallery_like"),
        (showcase.btn_comment, "btn_gallery_comment"),
        (showcase.btn_download, "btn_gallery_download"),
        (showcase.btn_fullscreen, "btn_gallery_fullscreen"),
        (showcase.btn_report, "btn_gallery_report"),
        (showcase.danmaku_overlay.composer.input_edit, "gallery_danmaku_input"),
    ]
    for widget, cid in expected_ids:
        assert widget.objectName() == cid
        assert control_id(widget) == cid

    # Fixed bottom-right horizontal order:
    # [Like] [Comment/Danmaku] [Download] [Fullscreen Screensaver] [Report]
    buttons = showcase.action_buttons
    assert [b.objectName() for b in buttons] == [
        "btn_gallery_like",
        "btn_gallery_comment",
        "btn_gallery_download",
        "btn_gallery_fullscreen",
        "btn_gallery_report",
    ]
    xs = [b.x() for b in buttons]
    assert xs == sorted(xs) and len(set(xs)) == 5

    # Top-right source badge, bottom-left author, OpenFocus badge, EXIF summary
    assert showcase.lbl_source_badge.isVisible() is True
    assert showcase.lbl_source_badge.property("sourceBadge") == "following"
    assert showcase.lbl_openfocus_badge.isVisible() is True
    assert showcase.lbl_openfocus_badge.text() == "OpenFocus Studio"
    assert "OM-1" in showcase.lbl_exif.text()
    assert "M.Zuiko 90mm Macro" in showcase.lbl_exif.text()
    assert "ISO 200" in format_exif_summary(works[0].exif)

    # Author replies are excluded from floating danmaku, but present in comment drawer
    floating_ids = {
        b.comment_id
        for b in showcase.danmaku_overlay.active_bullets + showcase.danmaku_overlay.queued_comments
    }
    assert floating_ids == {"c1"}

    deleted: list[str] = []
    showcase.comment_deleted.connect(deleted.append)
    showcase.open_comment_drawer()
    delete_btn = showcase.comment_drawer.findChild(type(showcase.btn_like), "btn_delete_comment_c2")
    assert delete_btn is not None
    delete_btn.click()
    assert deleted == ["c2"]
    showcase.close_comment_drawer()

    # Signals on w1 (allow_download=True)
    likes: list[tuple[str, bool]] = []
    downloads: list[str] = []
    fullscreens: list[bool] = []
    authors: list[str] = []
    work_reports: list[tuple[str, str]] = []
    showcase.like_toggled.connect(lambda wid, liked: likes.append((wid, liked)))
    showcase.download_requested.connect(downloads.append)
    showcase.fullscreen_requested.connect(lambda: fullscreens.append(True))
    showcase.author_clicked.connect(authors.append)
    showcase.report_work_requested.connect(lambda wid, r: work_reports.append((wid, r)))

    assert showcase.btn_download.isVisible() is True
    assert showcase.btn_download.isEnabled() is True

    showcase.btn_like.click()
    assert likes == [("w1", True)]
    assert showcase.btn_like.isChecked() is True
    assert "13" in showcase.btn_like.text()

    showcase.btn_download.click()
    assert downloads == ["w1"]

    showcase.btn_fullscreen.click()
    assert fullscreens == [True]

    showcase.btn_author.click()
    assert authors == ["author_1"]

    assert showcase.report_current_work(REPORT_REASONS[0]) is True
    assert work_reports == [("w1", "nsfw_explicit")]

    # Switch to w2 (source_badge="ad", openfocus_badge=False, allow_download=False)
    showcase.show_slide(1)
    qapp.processEvents()
    assert showcase.lbl_source_badge.isVisible() is True
    assert showcase.lbl_source_badge.property("sourceBadge") == "ad"
    assert showcase.lbl_openfocus_badge.isVisible() is False
    assert showcase.btn_download.isVisible() is False
    assert showcase.btn_download.isEnabled() is False

    # Switch to w3 (source_badge=None) -> badge strictly hidden
    showcase.show_slide(2)
    qapp.processEvents()
    assert showcase.lbl_source_badge.isVisible() is False

    showcase.deleteLater()


def test_fullscreen_screensaver_always_on_top_toast_and_esc_exit(qapp):
    source = GalleryShowcaseWidget()
    source.set_works(_sample_works()[:2], start_index=1)
    source.set_danmaku_settings(DanmakuSettings(position="bottom", font_size=18))
    source.set_fusion_busy(True)

    win = GalleryScreensaverWindow(toast_hold_ms=20, toast_fade_ms=50)
    win.sync_from_showcase(source)
    assert not win.showcase.slide_timer.isActive()

    flags = win.windowFlags()
    assert bool(flags & Qt.WindowType.WindowStaysOnTopHint)
    assert bool(flags & Qt.WindowType.FramelessWindowHint)
    assert win.objectName() == "gallery_screensaver"
    assert control_id(win) == "gallery_screensaver"
    assert win.showcase.current_work is not None and win.showcase.current_work.id == "w2"
    assert win.showcase.fusion_busy is True
    assert win.showcase.danmaku_overlay.settings.position == "bottom"

    closed_count = 0

    def _on_closed() -> None:
        nonlocal closed_count
        closed_count += 1

    win.closed.connect(_on_closed)

    win.launch()
    qapp.processEvents()
    assert win.isVisible() is True
    assert win.showcase.slide_timer.isActive()
    assert win.lbl_exit_toast.isVisible() is True
    assert "ESC" in win.lbl_exit_toast.text()

    # Wait for auto-fade toast to hide cleanly
    QTest.qWait(140)
    qapp.processEvents()
    assert win.lbl_exit_toast.isVisible() is False

    # Escape must exit even when the composer input owns keyboard focus.
    win.showcase.open_danmaku_composer()
    composer = win.showcase.danmaku_overlay.composer
    composer.input_edit.setText("Fullscreen draft")
    assert qapp.focusWidget() is composer.input_edit
    QTest.keyClick(composer.input_edit, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert win.isVisible() is False
    assert closed_count == 1
    win.exit_screensaver()
    assert closed_count == 1
    assert win.showcase.active_pause_reasons() == frozenset({"hidden", "screensaver_closed"})

    # Re-launching resets closed guard and clicking fullscreen button also exits cleanly
    win.launch()
    qapp.processEvents()
    assert win.isVisible() is True
    assert win.showcase.slide_timer.isActive()
    assert not composer.is_open()
    assert composer.input_edit.text() == ""
    win.showcase.btn_fullscreen.click()
    qapp.processEvents()
    assert win.isVisible() is False
    assert closed_count == 2

    win.deleteLater()
    source.deleteLater()


def test_popover_and_visibility_pause_resume_slideshow(qapp):
    showcase = GalleryShowcaseWidget()
    showcase.resize(720, 420)
    showcase.set_works(_sample_works()[:2], start_index=0)
    overlay = showcase.danmaku_overlay
    assert not showcase.slide_timer.isActive()
    assert not overlay._anim_timer.isActive()
    showcase.show()
    qapp.processEvents()
    assert overlay._anim_timer.isActive()
    assert showcase.slide_timer.isActive() is True

    # Opening a danmaku action popover pauses slideshow via "hover_popover"
    assert showcase.danmaku_overlay.open_popover_for("c1") is True
    assert "hover_popover" in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is False

    pinned = overlay.popover.current_item
    assert pinned is not None and pinned.paused
    showcase.resize(320, 200)
    qapp.processEvents()
    assert not overlay.popover.isVisible()
    assert not pinned.paused
    assert "hover_popover" not in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is True

    # Hiding widget pauses slideshow via "hidden"; showing resumes
    showcase.hide()
    qapp.processEvents()
    assert "hidden" in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is False
    assert not overlay._anim_timer.isActive()

    showcase.show()
    qapp.processEvents()
    assert "hidden" not in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is True
    assert overlay._anim_timer.isActive()

    showcase.deleteLater()


def test_switching_danmaku_popover_does_not_freeze_slideshow(qapp):
    showcase = GalleryShowcaseWidget()
    showcase.resize(720, 420)
    showcase.show()
    qapp.processEvents()

    works = _sample_works()[:2]
    works[0].comments = [
        {"id": "c1", "work_id": "w1", "content": "Bullet A", "is_reply": False},
        {"id": "c2", "work_id": "w1", "content": "Bullet B", "is_reply": False},
    ]
    showcase.set_works(works, start_index=0)
    assert showcase.slide_timer.isActive() is True

    # Open popover on bullet A, then directly switch to bullet B while open
    assert showcase.danmaku_overlay.open_popover_for("c1") is True
    assert showcase.danmaku_overlay.open_popover_for("c2") is True
    assert showcase.slide_timer.isActive() is False
    assert showcase._pause_reasons.get("hover_popover") == 1

    # Dismissing once must cleanly resume the 5s slideshow
    showcase.danmaku_overlay.popover.dismiss()
    assert "hover_popover" not in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is True
    showcase.deleteLater()


def test_nested_replies_and_is_author_reply_badge_in_drawer(qapp):
    from PyQt6.QtWidgets import QFrame, QLabel

    showcase = GalleryShowcaseWidget()
    showcase.resize(720, 420)
    works = _sample_works()[:1]
    works[0].comments = [
        {
            "id": "top1",
            "work_id": "w1",
            "author_nickname": "ViewerOne",
            "content": "How many frames?",
            "is_reply": False,
            "is_danmaku": True,
            "replies": [
                {
                    "id": "rep1",
                    "work_id": "w1",
                    "author_nickname": "MacroMaster",
                    "content": "64 frames stacked!",
                    "parent_id": "top1",
                    "is_reply": True,
                    "is_danmaku": False,
                    "is_author_reply": True,
                }
            ],
        }
    ]
    showcase.set_works(works, start_index=0)

    # Only top-level comment floats as danmaku
    floating_ids = {
        b.comment_id
        for b in showcase.danmaku_overlay.active_bullets + showcase.danmaku_overlay.queued_comments
    }
    assert floating_ids == {"top1"}

    # Comment drawer unpacks nested replies and renders the Author badge
    showcase.open_comment_drawer()
    reply_cards = showcase.comment_drawer.findChildren(QFrame, "replyCard")
    assert len(reply_cards) == 1
    author_tags = reply_cards[0].findChildren(QLabel, "commentAuthorTag")
    assert len(author_tags) == 1
    body_labels = [lbl.text() for lbl in reply_cards[0].findChildren(QLabel, "commentBody")]
    assert "64 frames stacked!" in body_labels
    showcase.deleteLater()


def test_blur_cache_evicts_lru_and_previous_slide_single_cached_guard(qapp):
    cache = StaticBlurBackgroundCache(max_entries=2)
    showcase = GalleryShowcaseWidget(blur_cache=cache)
    showcase.resize(400, 300)

    w1 = ShowcaseWork(id="w1", pixmap=_make_pixmap(100, 100, QColor(10, 80, 200)))
    w2 = ShowcaseWork(id="w2", image_url="https://cdn.openfocus.top/w2.webp", pixmap=None)
    w3 = ShowcaseWork(id="w3", image_url="https://cdn.openfocus.top/w3.webp", pixmap=None)

    # Single cached slide under fusion_busy: previous_slide must be a no-op
    showcase.set_fusion_busy(True)
    showcase.set_works([w1, w2, w3], start_index=0)
    changes: list[str] = []
    showcase.slide_changed.connect(changes.append)
    assert showcase.previous_slide() is False
    assert changes == []

    # Fill cache beyond max_entries with pixmap updates.
    showcase.set_work_pixmap("w1", _make_pixmap(100, 100, QColor(20, 90, 210)))
    showcase.show_slide(1)
    showcase.set_work_pixmap("w2", _make_pixmap(100, 100, QColor(30, 100, 220)))
    showcase.show_slide(2)
    showcase.set_work_pixmap("w3", _make_pixmap(100, 100, QColor(40, 110, 230)))
    assert len(cache) == 2
    showcase.deleteLater()


def test_comment_drawer_renders_translation_with_original_toggle(qapp):
    from PyQt6.QtWidgets import QLabel
    from locales import trans

    prev_lang = trans.current_lang
    try:
        trans.set_language("zh")
        showcase = GalleryShowcaseWidget()
        showcase.resize(640, 360)
        showcase.set_danmaku_settings(DanmakuSettings(translation_mode="auto"))
        showcase.set_works(
            [
                ShowcaseWork(
                    id="w1",
                    comment_count=85,
                    pixmap=_make_pixmap(120, 80, QColor(30, 80, 140)),
                    comments=[
                        {
                            "id": "c1",
                            "author_nickname": "Hiro",
                            "content": "Sharp compound eyes!",
                            "src_lang": "en",
                            "translated_content": "锐利的复眼！",
                            "translated_lang": "zh",
                        }
                    ],
                )
            ]
        )
        comments = showcase.current_work.comments
        assert showcase.set_work_comments("w1", comments)
        assert showcase.current_work.comment_count == 85
        assert showcase.btn_comment.text().endswith("85")
        assert "85" in showcase.danmaku_overlay.composer.btn_all_comments.text()
        showcase.open_comment_drawer()
        qapp.processEvents()

        card = showcase.comment_drawer.findChild(QFrame, "commentCard")
        assert card is not None
        body = card.findChild(QLabel, "commentBody")
        toggle = card.findChild(QPushButton, "btn_toggle_comment_translation")
        assert body is not None and toggle is not None
        assert body.textFormat() == Qt.TextFormat.PlainText
        assert body.text() == "锐利的复眼！"
        toggle.click()
        assert body.text() == "Sharp compound eyes!"
        toggle.click()
        assert body.text() == "锐利的复眼！"
        showcase.deleteLater()
    finally:
        trans.set_language(prev_lang)


def test_showcase_distinguishes_studio_badge_from_author_name(qapp):
    showcase = GalleryShowcaseWidget()
    showcase.resize(640, 360)
    showcase.show()
    qapp.processEvents()

    showcase.set_works(
        [
            ShowcaseWork(
                id="w_official",
                author_id="openfocus_official",
                author_nickname="OpenFocus",
                openfocus_badge=True,
                exif={"lens": "Macro 60mm f/4"},
                pixmap=_make_pixmap(120, 80, QColor(40, 90, 150)),
            )
        ]
    )
    qapp.processEvents()
    assert showcase.btn_author.text() == "OpenFocus"
    assert showcase.lbl_openfocus_badge.isVisible() is True
    assert showcase.lbl_openfocus_badge.text() == "OpenFocus Studio"
    assert showcase.lbl_exif.text() == "Macro 60mm f/4"

    showcase.set_works(
        [
            ShowcaseWork(
                id="w_studio_dup",
                author_id="openfocus_studio",
                author_nickname="OpenFocus Studio",
                openfocus_badge=True,
                exif={"lens": "Macro 90mm f/2.8"},
                pixmap=_make_pixmap(120, 80, QColor(40, 90, 150)),
            )
        ]
    )
    qapp.processEvents()
    assert showcase.btn_author.text() == "OpenFocus Studio"
    assert showcase.lbl_openfocus_badge.isVisible()
    assert showcase.lbl_openfocus_badge.text() == '◆'
    work = showcase.current_work
    work.openfocus_badge = False
    showcase.show_slide(0)
    assert showcase.btn_author.text() == 'OpenFocus Studio'
    assert not showcase.lbl_openfocus_badge.isVisible()
    showcase.deleteLater()


def test_standalone_showcase_window_fusion_bar_and_return_flow(qapp):
    from widgets.gallery_showcase import GalleryShowcaseWindow

    win = GalleryShowcaseWindow()
    win.set_works([ShowcaseWork(id="w1", title="Beetle", pixmap=_make_pixmap(100, 80, QColor(40, 80, 120)))])
    assert win.fusion_bar.isVisible() is False

    toggled = []
    returned = []
    win.auto_show_toggled.connect(toggled.append)
    win.return_to_workspace_requested.connect(lambda: returned.append(True))

    win.show()
    qapp.processEvents()
    win.set_fusion_in_progress("Fusing 2/4", done=2, total=4)
    assert win.fusion_bar.isVisible() is True
    assert win.fusion_progress.maximum() == 4
    assert win.fusion_progress.value() == 2
    assert "2/4" in win.lbl_fusion_status.text()

    win.set_fusion_completed()
    assert win.fusion_progress.value() == 1

    win.chk_auto_show.setChecked(False)
    assert toggled == [False]

    win.btn_return.click()
    assert returned == [True]
    assert win.isVisible() is False
    win.deleteLater()
