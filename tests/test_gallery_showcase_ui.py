"""Offscreen Qt tests for GalleryShowcaseWidget and GalleryScreensaverWindow."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtTest import QTest

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

    showcase.deleteLater()


def test_fusion_busy_pauses_prefetch_and_rotates_only_cached_slides(qapp):
    prefetched: list[str] = []
    showcase = GalleryShowcaseWidget(prefetch_callback=prefetched.append)
    showcase.resize(640, 360)
    works = _sample_works()

    # Start with fusion busy BEFORE loading works -> zero prefetch calls
    showcase.set_fusion_busy(True)
    assert showcase.prefetch_paused is True
    showcase.set_works(works, start_index=1)
    assert prefetched == []

    # Rotating while fusion busy skips uncached w3 and stays on locally cached slides (w1, w2)
    assert showcase.current_work is not None and showcase.current_work.id == "w2"
    showcase.next_slide()
    assert showcase.current_work.id == "w1"
    showcase.next_slide()
    assert showcase.current_work.id == "w2"
    assert prefetched == []

    # Clearing fusion busy resumes prefetching for upcoming uncached slide w3
    showcase.set_fusion_busy(False)
    assert showcase.prefetch_paused is False
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

    # Re-open composer -> pauses timer; cancel -> resumes timer
    clock.advance(12.0)
    showcase.btn_comment.click()
    assert showcase.slide_timer.isActive() is False
    showcase.danmaku_overlay.composer.btn_cancel.click()
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
    assert win.lbl_exit_toast.isVisible() is True
    assert "ESC" in win.lbl_exit_toast.text()

    # Wait for auto-fade toast to hide cleanly
    QTest.qWait(140)
    qapp.processEvents()
    assert win.lbl_exit_toast.isVisible() is False

    # Pressing ESC cleanly exits screensaver and emits closed once
    QTest.keyClick(win, Qt.Key.Key_Escape)
    qapp.processEvents()
    assert win.isVisible() is False
    assert closed_count == 1

    # Re-launching resets closed guard and clicking fullscreen button also exits cleanly
    win.launch()
    qapp.processEvents()
    assert win.isVisible() is True
    win.showcase.btn_fullscreen.click()
    qapp.processEvents()
    assert win.isVisible() is False
    assert closed_count == 2

    win.deleteLater()
    source.deleteLater()


def test_popover_and_visibility_pause_resume_slideshow(qapp):
    showcase = GalleryShowcaseWidget()
    showcase.resize(720, 420)
    showcase.show()
    qapp.processEvents()

    showcase.set_works(_sample_works()[:2], start_index=0)
    assert showcase.slide_timer.isActive() is True

    # Opening a danmaku action popover pauses slideshow via "hover_popover"
    assert showcase.danmaku_overlay.open_popover_for("c1") is True
    assert "hover_popover" in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is False

    showcase.danmaku_overlay.popover.dismiss()
    assert "hover_popover" not in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is True

    # Hiding widget pauses slideshow via "hidden"; showing resumes
    showcase.hide()
    qapp.processEvents()
    assert "hidden" in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is False

    showcase.show()
    qapp.processEvents()
    assert "hidden" not in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is True

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


def test_blur_cache_force_put_evicts_lru_and_previous_slide_single_cached_guard(qapp):
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

    # Fill cache beyond max_entries via force updates (set_work_pixmap)
    showcase.set_work_pixmap("w1", _make_pixmap(100, 100, QColor(20, 90, 210)))
    showcase.show_slide(1)
    showcase.set_work_pixmap("w2", _make_pixmap(100, 100, QColor(30, 100, 220)))
    showcase.show_slide(2)
    showcase.set_work_pixmap("w3", _make_pixmap(100, 100, QColor(40, 110, 230)))
    assert len(cache) == 2
    showcase.deleteLater()
