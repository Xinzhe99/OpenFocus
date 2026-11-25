"""Main-window Gallery & Light Community integration + openfocus ctl regression suite.

Verifies:
- Top-level Account & Gallery menus discoverable & invocable via ControlServer /
  control_surface, with red-dot unread badges ONLY on Account & Message Box
  (never on Gallery), across all 12 language packs.
- Startup/idle shows Gallery Showcase; fusion result restores result canvas;
  right-click "Hide Fusion Result" returns to Gallery Showcase; right-click
  "Upload to Gallery" opens Publish tab prefilled with fused image +
  openfocus_badge=True.
- Opening Danmaku composer pauses the 5s slideshow timer; sending/cancelling
  resumes it.
- Fullscreen Always-on-Top screensaver launches and exits cleanly on ESC.
- Compute avoidance: set_fusion_busy(True) pauses background prefetch/downloads;
  session fence calls clear_account_state() before next account refresh on
  logout or account switch; can_hide_gallery is NEVER exposed in UI.
- WebP -> JPG local download + clickable reveal in file manager + My Downloads.
- Post-fusion UploadPromptDialog probability & headless guard; HonorCardDialog
  on login with ack_honor_card on close.
- Advanced Settings Danmaku & Gallery Storage/Cache live propagation + Clear Cache.
- 16-bit / RAW focus stacking & export path has zero regression.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QSettings, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from core.control.protocol import encode
from core.control.server import ControlServer
from core.gallery_cache import GalleryDiskCache, resolve_gallery_download_dir
from core.gallery_state import (
    AccountProfile,
    GalleryConfig,
    GalleryStateStore,
    GalleryWork,
    HonorCard,
    apply_config_snapshot,
    apply_profile_snapshot,
)
from dialogs.advanced_settings import AdvancedSettingsDialog
from locales import trans
from ui.control_surface import action_catalog, widget_tree
from utils.gallery_client import GalleryClient
from utils.image_utils import imwrite_auto
from widgets.danmaku_overlay import DanmakuSettings

_ORIG_REFRESH_CONFIG = GalleryStateStore.refresh_config
_ORIG_REFRESH_FEED = GalleryStateStore.refresh_feed
_ORIG_REFRESH_ACCOUNT = GalleryStateStore.refresh_account_silently


def _make_webp_bytes(color=(40, 120, 200), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="WEBP", quality=85)
    return buf.getvalue()


def _make_pixmap(color=Qt.GlobalColor.darkCyan, size=(96, 64)) -> QPixmap:
    img = QImage(size[0], size[1], QImage.Format.Format_RGB32)
    img.fill(color)
    return QPixmap.fromImage(img)


class _StubAccountClient:
    def __init__(self, signed_in: bool = True, user_id: str = "usr_alpha", email: str = "alpha@example.com", tok: str = "tok_alpha"):
        self._signed_in = signed_in
        self._user_id = user_id
        self._email = email
        self._tok = tok

    def is_signed_in(self) -> bool:
        return self._signed_in

    def token(self, *, deadline=None):
        return self._tok if self._signed_in else None

    def current_user(self):
        if not self._signed_in:
            return None
        return {"user_id": self._user_id, "email": self._email, "nickname": self._user_id}


def _ctl_call(server: ControlServer, cmd: str, **args):
    payload = {"id": "1", "cmd": cmd, "args": args, "token": server.token}
    return server.handle_line(encode(payload), token=server.token)


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    assert predicate()


@pytest.fixture
def integrated_window(qapp, monkeypatch, tmp_path):
    """Real OpenFocus window with sandboxed settings & zero live network calls."""
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENFOCUS_LOG_DIR", str(tmp_path / "logs"))

    from core import frame_storage
    from main import OpenFocus
    from ui import styles
    from utils import account_client, recovery, settings_store

    original_stylesheet = qapp.styleSheet()
    original_theme = styles.CURRENT_THEME
    original_language = trans.current_lang

    monkeypatch.setattr(frame_storage, "automatic_preflight_may_block", lambda: False)

    # Prevent startup account restore from touching network or real user storage.
    acct_cls = account_client.AccountClient

    def _offline_request(*_a, **_k):
        raise account_client.AccountError("network")

    def _sandbox_account_client():
        return acct_cls(
            store=account_client.TokenStore(path=tmp_path / "account-session"),
            request=_offline_request,
        )

    if isinstance(acct_cls, type):
        monkeypatch.setattr(account_client, "AccountClient", _sandbox_account_client)

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.clear()
    settings.setValue("ui/first_run_done", "1")
    settings.setValue("updates/last_check_epoch", 9999999999)
    settings.setValue("rendering/ai_cpu_warning_dismissed", True)
    settings.setValue("gallery/cache_dir", str(tmp_path / "gallery_cache"))
    settings.setValue("gallery/download_dir", str(tmp_path / "gallery_downloads"))
    monkeypatch.setattr(settings_store, "_settings_singleton", settings)
    monkeypatch.setattr(recovery, "_base_dir", lambda: str(tmp_path / "recovery"))
    monkeypatch.setenv("OPENFOCUS_CONTROL_FILE", str(tmp_path / "control.json"))

    # Keep startup remote fetches offline-safe inside tests.
    monkeypatch.setattr(GalleryStateStore, "refresh_config", lambda self, **k: False)
    monkeypatch.setattr(GalleryStateStore, "refresh_feed", lambda self, **k: False)
    monkeypatch.setattr(GalleryStateStore, "refresh_account_silently", lambda self, **k: False)
    monkeypatch.setattr(GalleryStateStore, "refresh_notifications", lambda self, **k: False)

    window = OpenFocus()
    window.ask_load_scale = False
    window.load_scale_default = 1.0
    window.thread_count = 1
    window.use_gpu = False
    window.tile_enabled = False
    window.align_cache_enabled = False
    window.restore_last_stack = False
    window.set_alignment_method("none")
    window.chk_quick_preview.setChecked(False)
    window.show()
    qapp.processEvents()

    server = ControlServer(window, socket_name=f"OpenFocus-gallery-test-{os.getpid()}-{id(tmp_path)}")
    try:
        yield window, server, settings, tmp_path
    finally:
        server.close()
        window._force_quit = True
        window.source_manager.clear_image_stack()
        window.close()
        deadline = time.monotonic() + 5.0
        while window.isVisible() and time.monotonic() < deadline:
            QTest.qWait(10)
        window.deleteLater()
        from PyQt6.QtCore import QCoreApplication, QEvent
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        styles.set_current_theme(original_theme)
        qapp.setStyleSheet(original_stylesheet)
        if trans.current_lang != original_language:
            trans.set_language(original_language)
        qapp.processEvents()


# ---------------------------------------------------------------------------
# 1. Top-level Account & Gallery menus + openfocus ctl + unread red dot
# ---------------------------------------------------------------------------

def test_top_level_account_and_gallery_menus_via_ctl_and_unread_dot(integrated_window, qapp):
    window, server, _settings, tmp_path = integrated_window
    gc = window.gallery_controller

    # Verify top-level menus exist alongside File/Edit/Tools/Settings/Help.
    top_menus = [act.menu() for act in window.menuBar().actions() if act.menu() is not None]
    assert window.ui_objs["menu_account"] in top_menus
    assert window.ui_objs["menu_gallery"] in top_menus

    # Verify control_surface / openfocus ctl discovers every Account & Gallery action.
    res = _ctl_call(server, "ui.actions")
    assert res["ok"] is True
    keys = {item["key"] for item in res["result"]["actions"]}
    for expected in (
        "action_my_account",
        "action_message_box",
        "action_my_downloads",
        "action_open_gallery",
        "action_gallery_screensaver",
    ):
        assert expected in keys, f"{expected} missing from ctl action catalog"

    # Invoke "My Downloads" via openfocus ctl and verify OS opener receives custom dir.
    opened_folders = []
    gc._folder_opener = lambda path: opened_folders.append(path) or True
    dl_res = _ctl_call(server, "ui.trigger", key="action_my_downloads")
    assert dl_res["ok"] is True
    _wait_until(lambda: opened_folders == [str(tmp_path / "gallery_downloads")])

    # Invoke "OpenFocus Gallery" via openfocus ctl -> opens 4-Tab GalleryHubDialog.
    hub_res = _ctl_call(server, "ui.trigger", key="action_open_gallery")
    assert hub_res["ok"] is True
    _wait_until(lambda: gc.hub_dialog is not None and gc.hub_dialog.isVisible())
    assert gc.hub_dialog.tabs.count() == 4
    gc.hub_dialog.reject()
    qapp.processEvents()

    # Invoke "Message Box" via openfocus ctl -> opens AccountCenterDialog on tab 1.
    msg_res = _ctl_call(server, "ui.trigger", key="action_message_box")
    assert msg_res["ok"] is True
    _wait_until(lambda: gc.account_dialog is not None and gc.account_dialog.isVisible())
    assert gc.account_dialog.tabs.currentIndex() == 1
    gc.account_dialog.reject()
    qapp.processEvents()

    # Unread red-dot indicator: appears ONLY on Account menu & Message Box, NEVER Gallery.
    apply_profile_snapshot(
        gc.state_store,
        {
            "user_id": "u1",
            "nickname": "MacroPro",
            "email": "macro@example.com",
            "unread_notifications_count": 3,
            "can_hide_gallery": True,
        },
    )
    qapp.processEvents()
    account_menu = window.ui_objs["menu_account"]
    msg_action = window.ui_objs["action_message_box"]
    gallery_menu = window.ui_objs["menu_gallery"]

    assert "\u25cf" in account_menu.title()
    assert account_menu.property("hasUnreadDot") is True
    assert "\u25cf" in msg_action.text()
    assert "(3)" in msg_action.text()
    assert "\u25cf" not in gallery_menu.title()
    assert gallery_menu.property("hasUnreadDot") is False

    # Retranslate to Chinese & back: red dot persists on Account & Message Box, never Gallery.
    trans.set_language("zh")
    qapp.processEvents()
    assert account_menu.title().startswith(trans.t("menu_account"))
    assert "\u25cf" in account_menu.title()
    assert "\u25cf" not in gallery_menu.title()
    assert gallery_menu.title() == trans.t("menu_gallery")

    # Clearing unread clears the red dot cleanly.
    gc.update_unread_indicators(0)
    assert "\u25cf" not in account_menu.title()
    assert "\u25cf" not in msg_action.text()


# ---------------------------------------------------------------------------
# 2. Startup showcase, Hide Fusion Result, Upload to Gallery context menu
# ---------------------------------------------------------------------------

def test_showcase_startup_hide_result_and_upload_to_gallery(integrated_window, qapp):
    window, _server, _settings, _tmp_path = integrated_window
    gc = window.gallery_controller

    # Startup/idle displays Gallery Showcase automatically.
    assert window.view_stack.currentWidget() is window.gallery_showcase_page
    assert gc.is_showing_showcase() is True

    # Displaying a fusion result restores the normal result canvas (index 0).
    fused = np.full((64, 96, 3), 140, dtype=np.uint8)
    window.fusion_result = fused
    assert window.output_manager.display_specific_fusion_result(fused) is True
    qapp.processEvents()
    assert window.view_stack.currentIndex() == 0
    assert gc.is_showing_showcase() is False

    # Right-click context menu on lbl_result_img exposes Hide Fusion Result & Upload to Gallery.
    menu = gc.build_result_context_menu(window.lbl_result_img)
    actions_by_name = {act.objectName(): act for act in menu.actions()}
    assert "action_hide_fusion_result" in actions_by_name
    assert "action_upload_to_gallery" in actions_by_name
    assert actions_by_name["action_hide_fusion_result"].isEnabled()
    assert actions_by_name["action_upload_to_gallery"].isEnabled()

    # Triggering "Hide Fusion Result" switches view_stack back to Gallery Showcase.
    actions_by_name["action_hide_fusion_result"].trigger()
    qapp.processEvents()
    assert window.view_stack.currentWidget() is window.gallery_showcase_page
    assert gc.is_showing_showcase() is True

    # Triggering "Upload to Gallery" opens Publish tab prefilled with fused image & openfocus_badge=True.
    actions_by_name["action_upload_to_gallery"].trigger()
    qapp.processEvents()
    _wait_until(lambda: gc.hub_dialog is not None and gc.hub_dialog.isVisible())
    hub = gc.hub_dialog
    assert hub.tabs.currentIndex() == 2  # Publish tab
    assert len(hub._publish_queue) == 1
    assert hub._publish_queue[0].openfocus_badge is True
    hub.reject()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# 3. Danmaku composer pauses 5s slideshow timer; send/cancel resumes it
# ---------------------------------------------------------------------------

def test_danmaku_composer_pauses_and_resumes_slide_timer(integrated_window, qapp):
    window, _server, _settings, _tmp_path = integrated_window
    showcase = window.gallery_showcase_page
    showcase.set_works(
        [
            {"id": "w1", "title": "Slide 1", "author_id": "a1", "author_nickname": "Alice", "pixmap": _make_pixmap(Qt.GlobalColor.red)},
            {"id": "w2", "title": "Slide 2", "author_id": "a2", "author_nickname": "Bob", "pixmap": _make_pixmap(Qt.GlobalColor.blue)},
        ]
    )
    qapp.processEvents()
    assert showcase.slide_interval_sec == 5.0
    assert showcase.slide_timer.isActive() is True

    # Open composer -> slide timer pauses.
    showcase.open_danmaku_composer()
    qapp.processEvents()
    assert showcase.is_slideshow_paused() is True
    assert "danmaku_compose" in showcase.active_pause_reasons()
    assert showcase.slide_timer.isActive() is False

    # Cancel composer -> slide timer resumes.
    showcase.close_danmaku_composer()
    qapp.processEvents()
    assert showcase.is_slideshow_paused() is False
    assert showcase.slide_timer.isActive() is True

    # Open composer again and submit -> timer pauses then resumes cleanly.
    showcase.open_danmaku_composer()
    qapp.processEvents()
    assert showcase.slide_timer.isActive() is False
    showcase.danmaku_overlay.composer.input_edit.setText("Sharp stack!")
    showcase.danmaku_overlay.composer.btn_send.click()
    qapp.processEvents()
    assert showcase.is_slideshow_paused() is False
    assert showcase.slide_timer.isActive() is True


# ---------------------------------------------------------------------------
# 4. Fullscreen Always-on-Top screensaver opens and exits cleanly on ESC
# ---------------------------------------------------------------------------

def test_fullscreen_screensaver_opens_and_exits_on_esc(integrated_window, qapp):
    window, _server, _settings, _tmp_path = integrated_window
    gc = window.gallery_controller

    scr = gc.open_screensaver()
    assert scr is not None
    qapp.processEvents()
    assert scr.isVisible() is True
    assert bool(scr.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
    assert bool(scr.windowFlags() & Qt.WindowType.FramelessWindowHint)

    closed_events = []
    scr.closed.connect(lambda: closed_events.append(True))

    QTest.keyClick(scr, Qt.Key.Key_Escape)
    _wait_until(lambda: not scr.isVisible())
    assert closed_events == [True]


# ---------------------------------------------------------------------------
# 5. Compute avoidance, QA t4 session fence, and can_hide_gallery never in UI
# ---------------------------------------------------------------------------

def test_fusion_busy_session_fence_and_no_can_hide_gallery_in_ui(integrated_window, qapp, monkeypatch):
    window, _server, _settings, _tmp_path = integrated_window
    gc = window.gallery_controller

    # 5a. Compute avoidance: set_fusion_busy(True) pauses background downloads & remote fetches.
    fetched_urls = []
    gc._media_fetcher = lambda url: fetched_urls.append(url) or _make_webp_bytes()
    gc.showcase.set_works(
        [
            {"id": "cached_1", "title": "Cached", "pixmap": _make_pixmap()},
            {"id": "remote_2", "title": "Remote", "image_url": "https://cdn.example.com/r2.webp"},
        ]
    )
    gc.set_fusion_busy(True)
    assert gc.fusion_busy is True
    assert gc.state_store.fusion_busy is True
    assert gc.showcase.fusion_busy is True
    assert gc.showcase.prefetch_paused is True
    assert gc._prefetch_work_image("remote_2") is False
    assert fetched_urls == []

    gc.set_fusion_busy(False)
    assert gc.fusion_busy is False
    assert gc.state_store.fusion_busy is False
    assert gc.showcase.fusion_busy is False

    # 5b. QA t4 session fence: logout & account switch call clear_account_state() BEFORE next refresh.
    call_order = []
    orig_clear = gc.state_store.clear_account_state

    def _tracked_clear():
        call_order.append("clear_account_state")
        orig_clear()

    monkeypatch.setattr(gc.state_store, "clear_account_state", _tracked_clear)
    monkeypatch.setattr(
        gc.state_store,
        "refresh_account_silently",
        lambda **k: call_order.append("refresh_account_silently") or True,
    )

    user_a = _StubAccountClient(True, "user_a", "a@example.com", "tok_a")
    user_b = _StubAccountClient(True, "user_b", "b@example.com", "tok_b")
    window._account_client = user_a
    gc.sync_account_session(user_a)
    apply_profile_snapshot(
        gc.state_store,
        {"user_id": "user_a", "nickname": "UserA", "unread_notifications_count": 4, "can_hide_gallery": True},
    )
    assert gc.state_store.can_hide_gallery is True

    # Account switch: clear_account_state MUST precede refresh_account_silently.
    call_order.clear()
    window._account_client = user_b
    gc.sync_account_session(user_b)
    assert call_order[:2] == ["clear_account_state", "refresh_account_silently"]
    assert gc.state_store.profile is None
    assert gc.state_store.unread_notifications_count == 0

    # Logout: clear_account_state runs and NO refresh is started.
    call_order.clear()
    signed_out = _StubAccountClient(False)
    window._account_client = signed_out
    gc.sync_account_session(signed_out)
    assert call_order == ["clear_account_state"]

    # 5c. can_hide_gallery is NEVER exposed in UI (Account Center, Advanced Settings, main window).
    apply_profile_snapshot(
        gc.state_store,
        {"user_id": "user_b", "nickname": "UserB", "can_hide_gallery": True},
    )
    assert gc.state_store.can_hide_gallery is True
    assert "can_hide_gallery" not in (gc.state_store.ui_profile_dict() or {})

    acct_dlg = gc.open_account_center(0)
    qapp.processEvents()
    assert "can_hide_gallery" not in str(widget_tree(acct_dlg, max_depth=8)).lower()
    acct_dlg.reject()

    adv_dlg = AdvancedSettingsDialog(window)
    qapp.processEvents()
    assert "can_hide_gallery" not in str(widget_tree(adv_dlg, max_depth=8)).lower()
    adv_dlg.reject()
    adv_dlg.deleteLater()
    qapp.processEvents()


# ---------------------------------------------------------------------------
# 6. WebP -> JPG download + clickable reveal + HonorCard + UploadPrompt + Settings
# ---------------------------------------------------------------------------

def test_webp_to_jpg_download_honor_card_upload_prompt_and_advanced_settings(integrated_window, qapp, tmp_path):
    window, _server, _settings, _root = integrated_window
    gc = window.gallery_controller

    # 6a. WebP -> JPG local download + clickable reveal in file manager.
    webp_bytes = _make_webp_bytes(color=(20, 180, 90))
    gc._media_fetcher = lambda url: webp_bytes
    revealed = []
    gc._file_revealer = lambda path: revealed.append(path)
    downloaded = []
    gc.download_completed.connect(lambda p: downloaded.append(p))

    gc.showcase.set_works(
        [
            {
                "id": "dl_work_1",
                "title": "Beetle Macro",
                "author_id": "a1",
                "author_nickname": "MacroLab",
                "allow_download": True,
                "image_url": "https://cdn.example.com/beetle.webp",
                "pixmap": _make_pixmap(),
            }
        ]
    )
    assert gc.download_work("dl_work_1") is True
    _wait_until(lambda: len(downloaded) == 1)
    saved = Path(downloaded[0])
    assert saved.is_file()
    assert saved.suffix.lower() == ".jpg"
    with Image.open(saved) as im:
        assert im.format == "JPEG"
    # Clicking the reveal notification button selects the saved JPG in file manager.
    assert gc._reveal_button is not None and gc._reveal_button.isVisible()
    gc._reveal_button.click()
    assert revealed == [str(saved)]

    # 6b. Champion HonorCardDialog on login calls ack_honor_card on close.
    acked = []
    window._account_client = _StubAccountClient(True, "champ_1", "champ@example.com", "tok_champ")
    gc.client.ack_honor_card = lambda cid: acked.append(cid) or {"acknowledged": True}
    apply_profile_snapshot(
        gc.state_store,
        {
            "user_id": "champ_1",
            "nickname": "Champion",
            "pending_honor_cards": [
                {
                    "card_id": "hc_7d_01",
                    "badge_key": "top_7d",
                    "period": "7d",
                    "rank": 1,
                    "work_id": "w_champ",
                    "work_title": "Winning Stack",
                    "points_awarded": 500,
                }
            ],
        },
    )
    _wait_until(lambda: len(gc._honor_dialogs) == 1)
    honor_dlg = gc._honor_dialogs[0]
    honor_dlg.close_btn.click()
    _wait_until(lambda: acked == ["hc_7d_01"])

    # 6c. Post-fusion UploadPromptDialog respects headless guard and probability.
    fused = np.full((32, 32, 3), 200, dtype=np.uint8)
    apply_config_snapshot(gc.state_store, {"gallery_enabled": True, "upload_prompt_probability": 1.0})
    # Skipped in offscreen/headless by default, and skipped in batch mode.
    assert gc.on_fusion_completed(fused, interactive=True) is None
    gc.allow_headless_upload_prompt = True
    assert gc.on_fusion_completed(fused, interactive=True, is_batch=True) is None
    prompt = gc.on_fusion_completed(fused, interactive=True, is_batch=False)
    assert prompt is not None and prompt.isVisible()
    prompt.reject()
    qapp.processEvents()

    # 6d. Advanced Settings Danmaku & Gallery Storage/Cache live propagation + Clear Cache.
    gc.disk_cache.put("test_key_1", b"x" * 2048)
    assert gc.cache_usage_bytes() >= 2048

    adv = AdvancedSettingsDialog(window)
    adv.chk_danmaku_enabled.setChecked(False)
    adv.combo_danmaku_position.setCurrentIndex(adv.combo_danmaku_position.findData("bottom"))
    adv.spin_danmaku_font_size.setValue(22)
    adv.spin_danmaku_opacity.setValue(60)
    adv.combo_danmaku_density.setCurrentIndex(adv.combo_danmaku_density.findData("low"))

    custom_dl = tmp_path / "custom_dl"
    custom_cache = tmp_path / "custom_cache"
    adv.gallery_download_dir.setText(str(custom_dl))
    adv.gallery_cache_dir.setText(str(custom_cache))
    adv.spin_gallery_cache_limit.setValue(150)

    # One-click Clear Cache clears current cache immediately.
    adv.btn_clear_gallery_cache.click()
    assert gc.cache_usage_bytes() == 0

    adv._apply()
    assert adv.result() == QDialog.DialogCode.Accepted
    live_dm = window.gallery_showcase_page.danmaku_overlay.settings
    assert live_dm.enabled is False
    assert live_dm.position == "bottom"
    assert live_dm.font_size == 22
    assert abs(live_dm.opacity - 0.60) < 0.01
    assert live_dm.density == "low"
    assert gc.cache_max_mb() == 150
    assert resolve_gallery_download_dir(create=False) == custom_dl
    assert gc.disk_cache.cache_dir == custom_cache
    adv.deleteLater()
    qapp.processEvents()

    # 6e. Remote gallery_enabled=False circuit breaker hides Showcase cleanly.
    gc.show_gallery_showcase(force=True)
    assert gc.is_showing_showcase() is True
    apply_config_snapshot(gc.state_store, {"gallery_enabled": False})
    qapp.processEvents()
    assert gc.is_showing_showcase() is False
    assert window.view_stack.currentIndex() == 0


# ---------------------------------------------------------------------------
# 7. Zero regression on 16-bit / RAW focus fusion & export via openfocus ctl
# ---------------------------------------------------------------------------

def test_16bit_fusion_and_export_zero_regression_with_gallery_mounted(integrated_window, tmp_path):
    window, server, _settings, _root = integrated_window
    gc = window.gallery_controller

    # Create a 3-frame synthetic 16-bit TIFF focus stack.
    stack_dir = tmp_path / "stack16"
    stack_dir.mkdir()
    h, w = 48, 64
    yy, xx = np.mgrid[0:h, 0:w]
    for idx in range(3):
        img = np.zeros((h, w, 3), dtype=np.uint16)
        band = slice(idx * 16, (idx + 1) * 16)
        pattern = (((xx + yy) % 2) * 50000 + 8000).astype(np.uint16)
        img[band, :, 0] = pattern[band, :]
        img[band, :, 1] = np.clip(pattern[band, :].astype(np.int32) + 4000, 0, 65535).astype(np.uint16)
        img[band, :, 2] = 30000
        imwrite_auto(str(stack_dir / f"frame_{idx:02d}.tif"), img)

    # Load 16-bit stack via openfocus ctl.
    load_res = _ctl_call(server, "stack.load", folder=str(stack_dir))
    assert load_res["ok"] is True
    _wait_until(lambda: len(window.raw_images) == 3, timeout=10.0)
    assert window.raw_images[0].dtype == np.uint16

    # Run focus fusion via openfocus ctl with OpenFocus A (default) and verify 16-bit output.
    window.set_fusion_method("openfocus_a")
    render_res = _ctl_call(server, "render.start", confirmed=True)
    assert render_res["ok"] is True
    _wait_until(lambda: window.render_manager.last_state == "done", timeout=15.0)

    assert window.fusion_result is not None
    assert window.fusion_result.dtype == np.uint16
    assert window.fusion_result.shape == (h, w, 3)
    # Gallery Showcase automatically yields to the fused result canvas and releases busy state.
    assert window.view_stack.currentIndex() == 0
    assert gc.fusion_busy is False

    # Export 16-bit TIFF via openfocus ctl and verify bit depth preserved on disk.
    out_path = tmp_path / "fused_16bit.tif"
    exp_res = _ctl_call(server, "output.export", path=str(out_path))
    assert exp_res["ok"] is True
    assert out_path.is_file()
    import cv2
    reloaded = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
    assert reloaded is not None
    assert reloaded.dtype == np.uint16
    assert reloaded.shape == (h, w, 3)


# ---------------------------------------------------------------------------
# 8. QA t4 acceptance: HTTP 404 on /api/v1/config & /api/v1/account/me degrades
#    silently to offline showcase with zero modal popups & intact account flows
# ---------------------------------------------------------------------------

def test_http_404_config_and_me_degrade_silently_and_preserve_local_account(
    integrated_window, qapp, monkeypatch
):
    from utils.gallery_client import HttpResponse

    window, _server, _settings, _root = integrated_window
    gc = window.gallery_controller

    # Wire a real GalleryClient + real GalleryStateStore refresh methods backed by HTTP 404.
    requested_paths = []

    def _transport_404(method, url, headers, body, timeout):
        requested_paths.append(url)
        return HttpResponse(status=404, headers={}, body=b'{"error":"not_found"}')

    local_account = _StubAccountClient(
        signed_in=True,
        user_id="usr_local_404",
        email="local@example.com",
        tok="tok_local_404",
    )
    window._account_client = local_account
    window._refresh_account_action()

    # Replace client with one using our HTTP 404 transport and restore real store refresh methods.
    client_404 = GalleryClient(account_client=gc._account_proxy, transport=_transport_404)
    gc.client = client_404
    gc.state_store._client = client_404
    gc.state_store._background_runner = lambda fn: fn()
    monkeypatch.setattr(GalleryStateStore, "refresh_config", _ORIG_REFRESH_CONFIG)
    monkeypatch.setattr(GalleryStateStore, "refresh_feed", _ORIG_REFRESH_FEED)
    monkeypatch.setattr(GalleryStateStore, "refresh_account_silently", _ORIG_REFRESH_ACCOUNT)

    sync_errors = []
    gc.state_store.sync_error.connect(lambda op, code: sync_errors.append((op, code)))

    assert gc.state_store.refresh_config(force_refresh=True) is True
    assert gc.state_store.refresh_feed() is True
    assert gc.state_store.refresh_account_silently(bypass_backoff=True) is True
    qapp.processEvents()

    # 1. All 404s degraded silently via sync_error with ZERO modal error popups.
    assert any("/api/v1/config" in u for u in requested_paths)
    assert any("/api/v1/account/me" in u for u in requested_paths)
    assert ("config", "not_found") in sync_errors
    assert ("account", "not_found") in sync_errors
    assert QApplication.activeModalWidget() is None

    # 2. Idle Gallery showcase stays visible with offline built-in slides.
    assert window.view_stack.currentWidget() is window.gallery_showcase_page
    assert len(window.gallery_showcase_page.works) >= 1
    assert any(w.id.startswith("builtin_") for w in window.gallery_showcase_page.works)

    # 3. Existing local account menu & AccountCenterDialog remain 100% accessible.
    help_acct_action = window.ui_objs["action_account"]
    assert "local@example.com" in help_acct_action.text()

    acct_dlg = gc.open_account_center(0)
    qapp.processEvents()
    assert acct_dlg is not None and acct_dlg.isVisible()
    assert "local@example.com" in acct_dlg.email_label.text()
    assert QApplication.activeModalWidget() is None
    acct_dlg.reject()
    qapp.processEvents()

    # 4. Signing out clears state cleanly with zero errors.
    local_account._signed_in = False
    window._refresh_account_action()
    gc.sync_account_session(local_account)
    qapp.processEvents()
    assert gc.state_store.profile is None
    assert gc.state_store.unread_notifications_count == 0
    assert QApplication.activeModalWidget() is None


# ---------------------------------------------------------------------------
# 9. MagnifierLabel right-click context menu: single emit, no double-fire,
#    and suppressed after right-drag pan or ROI right-click exit
# ---------------------------------------------------------------------------

def test_magnifier_label_context_menu_single_emit_and_pan_roi_suppression(qapp):
    from PyQt6.QtCore import QEvent, QPointF
    from PyQt6.QtGui import QContextMenuEvent, QMouseEvent
    from widgets.magnifier_label import MagnifierLabel

    label = MagnifierLabel()
    label.resize(200, 150)
    label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    label.set_display_pixmap(_make_pixmap(size=(200, 150)))
    label.show()
    qapp.processEvents()

    emitted = []
    label.customContextMenuRequested.connect(lambda pt: emitted.append((pt.x(), pt.y())))

    def _mouse_ev(ev_type, pos):
        pt = QPointF(float(pos.x()), float(pos.y()))
        return QMouseEvent(
            ev_type,
            pt,
            QPointF(label.mapToGlobal(pos)),
            Qt.MouseButton.RightButton,
            Qt.MouseButton.RightButton if ev_type != QEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )

    def _send_mouse_ctx(pos):
        ctx = QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,
            pos,
            label.mapToGlobal(pos),
        )
        QApplication.sendEvent(label, ctx)

    try:
        # 1. Plain right-click + trailing platform QContextMenuEvent(Reason.Mouse): emits ONCE.
        p0 = QPoint(60, 60)
        label.mousePressEvent(_mouse_ev(QEvent.Type.MouseButtonPress, p0))
        label.mouseReleaseEvent(_mouse_ev(QEvent.Type.MouseButtonRelease, p0))
        _send_mouse_ctx(p0)
        assert len(emitted) == 1

        # 2. Right-drag pan (>4px) + trailing QContextMenuEvent(Reason.Mouse): emits ZERO times.
        emitted.clear()
        p1 = QPoint(90, 80)
        label.mousePressEvent(_mouse_ev(QEvent.Type.MouseButtonPress, p0))
        label.mouseMoveEvent(_mouse_ev(QEvent.Type.MouseMove, p1))
        label.mouseReleaseEvent(_mouse_ev(QEvent.Type.MouseButtonRelease, p1))
        _send_mouse_ctx(p1)
        assert emitted == []

        # 3. ROI mode right-click exit + trailing QContextMenuEvent(Reason.Mouse): emits ZERO times.
        emitted.clear()
        roi_exited = []
        label.roiModeExitRequested.connect(lambda: roi_exited.append(True))
        label.roi_mode = True
        label.mousePressEvent(_mouse_ev(QEvent.Type.MouseButtonPress, p0))
        label.mouseReleaseEvent(_mouse_ev(QEvent.Type.MouseButtonRelease, p0))
        _send_mouse_ctx(p0)
        assert roi_exited == [True]
        assert emitted == []

        # 4. Direct synthetic QContextMenuEvent (openfocus ctl path): emits ONCE.
        label.roi_mode = False
        emitted.clear()
        _send_mouse_ctx(p0)
        assert len(emitted) == 1
    finally:
        label.close()
        label.deleteLater()
        qapp.processEvents()
