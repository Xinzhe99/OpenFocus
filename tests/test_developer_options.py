"""Comprehensive tests for Developer Options and default-OFF Community gating.

Verifies:
1. Default-locked behavior: Community is strictly disabled by default; top-level
   menu_gallery and menu_account are hidden while Help -> Account remains visible
   and functional; menu_quick_start is removed while menu_guided_tour stays visible;
   gallery_settings_tab is hidden in AdvancedSettingsDialog; no startup polling,
   no fusion showcase popup, no UploadPromptDialog, no HonorCardDialog, no
   right-click "Upload to Community" menu entry.
2. Remote verification (`POST /api/v1/developer/verify-key`): valid key, invalid key,
   revoked key, expired key, rate limiting, and unreachable/network errors.
3. Startup background revalidation: revokes local grant when server explicitly says
   invalid/revoked; preserves offline grant when network is unreachable.
4. End-to-end GUI flow in AdvancedSettingsDialog Developer Options tab:
   masked key input + show/hide toggle, non-blocking async Verify & Unlock,
   live show/hide of gallery_settings_tab, live show/hide of main-window menus &
   GalleryController on Apply and on Lock & Clear Key.
5. 12-language i18n completeness for all Developer Options strings.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import threading
import time
import urllib.error

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QLineEdit

from core import developer_options as dev_opts
from core.developer_options import (
    DEVELOPER_VERIFY_ROUTE,
    FEATURE_COMMUNITY,
    DeveloperGrant,
    DeveloperKeyVerificationResult,
    clear_developer_grant,
    has_developer_grant,
    is_community_enabled,
    is_developer_feature_enabled,
    load_developer_grant,
    mask_key_prefix,
    revalidate_saved_developer_grant,
    save_developer_grant,
    set_developer_feature_enabled,
    verify_developer_key_remote,
)
from core.gallery_state import GalleryStateStore, apply_config_snapshot, apply_profile_snapshot
from dialogs.advanced_settings import AdvancedSettingsDialog
from locales import _DEVELOPER_OPTIONS_KEYS, trans
from utils.gallery_client import HttpResponse


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    assert predicate()


@pytest.fixture
def sandbox_settings(tmp_path, monkeypatch):
    from utils import settings_store

    settings = QSettings(str(tmp_path / "dev_settings.ini"), QSettings.Format.IniFormat)
    settings.clear()
    monkeypatch.setattr(settings_store, "_settings_singleton", settings)
    return settings


@pytest.fixture
def default_locked_window(qapp, monkeypatch, tmp_path):
    """Real OpenFocus main window with clean fresh settings (default locked state)."""
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

    # Track any accidental remote Gallery calls in default-locked state.
    remote_calls: list[str] = []
    monkeypatch.setattr(
        GalleryStateStore,
        "refresh_config",
        lambda self, **k: remote_calls.append("config") or False,
    )
    monkeypatch.setattr(
        GalleryStateStore,
        "refresh_feed",
        lambda self, **k: remote_calls.append("feed") or False,
    )
    monkeypatch.setattr(
        GalleryStateStore,
        "refresh_account_silently",
        lambda self, **k: remote_calls.append("account") or False,
    )
    monkeypatch.setattr(
        GalleryStateStore,
        "refresh_notifications",
        lambda self, **k: remote_calls.append("notifications") or False,
    )

    window = OpenFocus()
    window.show()
    qapp.processEvents()
    try:
        yield window, settings, remote_calls
    finally:
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
# 1. Core persistence, expiry, and remote verification
# ---------------------------------------------------------------------------

def test_default_persistence_is_strictly_locked(sandbox_settings):
    assert load_developer_grant(settings=sandbox_settings) is None
    assert has_developer_grant(settings=sandbox_settings) is False
    assert has_developer_grant(FEATURE_COMMUNITY, settings=sandbox_settings) is False
    assert is_developer_feature_enabled(FEATURE_COMMUNITY, settings=sandbox_settings) is False
    assert is_community_enabled(settings=sandbox_settings) is False

    # Cannot flip feature ON without a verified grant.
    assert set_developer_feature_enabled(FEATURE_COMMUNITY, True, settings=sandbox_settings) is False
    assert is_community_enabled(settings=sandbox_settings) is False


def test_grant_persistence_expiry_and_clear(sandbox_settings):
    future_iso = (datetime.now(timezone.utc) + timedelta(days=30)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    grant = DeveloperGrant(
        key_prefix=mask_key_prefix("ofdev_secret_12345678"),
        label="Internal QA",
        features=["Community", "community"],
        verified_at="2026-11-27T08:00:00Z",
        expires_at=future_iso,
        server_validated=True,
    )
    assert grant.features == ["community"]
    assert grant.is_active() is True
    assert grant.has_feature(FEATURE_COMMUNITY) is True

    save_developer_grant(grant, key="ofdev_secret_12345678", settings=sandbox_settings)
    loaded = load_developer_grant(settings=sandbox_settings)
    assert loaded is not None
    assert loaded.key_prefix == "ofdev_se..."
    assert loaded.label == "Internal QA"
    assert loaded.features == ["community"]
    assert loaded.server_validated is True
    # Grant alone does NOT enable Community until toggled ON explicitly.
    assert is_community_enabled(settings=sandbox_settings) is False

    assert set_developer_feature_enabled(FEATURE_COMMUNITY, True, settings=sandbox_settings) is True
    assert is_community_enabled(settings=sandbox_settings) is True

    # Expired grant immediately revokes effective access even if toggle was True.
    past_time = datetime.now(timezone.utc) + timedelta(days=60)
    assert loaded.is_expired(now=past_time) is True
    assert is_community_enabled(settings=sandbox_settings, now=past_time) is False

    # Clear removes grant, secret, and feature flag.
    clear_developer_grant(settings=sandbox_settings)
    assert load_developer_grant(settings=sandbox_settings) is None
    assert dev_opts.load_developer_key_secret(settings=sandbox_settings) == ""
    assert is_community_enabled(settings=sandbox_settings) is False


def test_verify_developer_key_remote_outcomes():
    captured = []

    def _ok_transport(method, url, headers, body, timeout):
        captured.append((method, url, dict(headers), json.loads(body.decode("utf-8")), timeout))
        return HttpResponse(
            status=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps({
                "ok": True,
                "valid": True,
                "key_prefix": "ofdev_9a3f...",
                "label": "Core Maintainer",
                "features": ["community"],
                "expires_at": "2099-12-31T23:59:59Z",
            }).encode("utf-8"),
        )

    res = verify_developer_key_remote(
        "ofdev_9a3f_real_key_value",
        base_url="https://account.openfocus.top",
        transport=_ok_transport,
        bearer_token="tok_session_123",
    )
    assert res.ok is True and res.valid is True and res.status == "valid"
    assert res.revoked is False and res.network_error is False
    assert res.grant is not None
    assert res.grant.key_prefix == "ofdev_9a3f..."
    assert res.grant.label == "Core Maintainer"
    assert res.grant.features == ["community"]
    assert res.grant.server_validated is True
    assert captured[0][0] == "POST"
    assert captured[0][1] == f"https://account.openfocus.top{DEVELOPER_VERIFY_ROUTE}"
    assert captured[0][2]["Authorization"] == "Bearer tok_session_123"
    assert captured[0][3] == {"key": "ofdev_9a3f_real_key_value"}

    # Empty key -> immediate local rejection.
    empty = verify_developer_key_remote("   ", transport=_ok_transport)
    assert empty.valid is False and empty.error_code == "empty_key"

    # Explicit server revocation (401/403 or 200 valid=false).
    revoked = verify_developer_key_remote(
        "ofdev_revoked",
        transport=lambda *_a: HttpResponse(
            status=403,
            headers={},
            body=b'{"ok":false,"valid":false,"error":"key_revoked"}',
        ),
    )
    assert revoked.valid is False and revoked.revoked is True and revoked.status == "revoked"

    invalid_200 = verify_developer_key_remote(
        "ofdev_bad",
        transport=lambda *_a: HttpResponse(
            status=200,
            headers={},
            body=b'{"ok":false,"valid":false,"error":"invalid_key"}',
        ),
    )
    assert invalid_200.valid is False and invalid_200.revoked is True and invalid_200.status == "invalid"

    # Rate limited -> not revoked, not network error.
    limited = verify_developer_key_remote(
        "ofdev_rl",
        transport=lambda *_a: HttpResponse(status=429, headers={}, body=b'{"error":"rate_limited"}'),
    )
    assert limited.valid is False and limited.status == "rate_limited" and limited.revoked is False

    # Network outage / 503 -> unreachable, preserves offline grant during revalidation.
    def _net_fail(*_a):
        raise urllib.error.URLError("offline")

    offline = verify_developer_key_remote("ofdev_any", transport=_net_fail)
    assert offline.valid is False and offline.network_error is True and offline.revoked is False

    err503 = verify_developer_key_remote(
        "ofdev_any",
        transport=lambda *_a: HttpResponse(status=503, headers={}, body=b"unavailable"),
    )
    assert err503.valid is False and err503.network_error is True and err503.revoked is False


def test_background_revalidation_revokes_on_explicit_reject_and_keeps_offline(sandbox_settings):
    grant = DeveloperGrant(
        key_prefix="ofdev_keep...",
        label="Dev",
        features=[FEATURE_COMMUNITY],
        verified_at="2026-11-27T00:00:00Z",
        expires_at="2099-01-01T00:00:00Z",
        server_validated=True,
    )
    save_developer_grant(grant, key="ofdev_keep_secret", settings=sandbox_settings)
    set_developer_feature_enabled(FEATURE_COMMUNITY, True, settings=sandbox_settings)
    assert is_community_enabled(settings=sandbox_settings) is True

    # 1. Network unreachable -> offline grant MUST remain intact!
    res_offline = revalidate_saved_developer_grant(
        transport=lambda *_a: (_ for _ in ()).throw(urllib.error.URLError("no network")),
        settings=sandbox_settings,
    )
    assert res_offline is not None and res_offline.network_error is True
    assert load_developer_grant(settings=sandbox_settings) is not None
    assert is_community_enabled(settings=sandbox_settings) is True

    # 2. Server explicitly revokes -> local grant and feature toggle MUST be cleared!
    res_revoked = revalidate_saved_developer_grant(
        transport=lambda *_a: HttpResponse(
            status=200,
            headers={},
            body=b'{"ok":false,"valid":false,"error":"key_revoked"}',
        ),
        settings=sandbox_settings,
    )
    assert res_revoked is not None and res_revoked.revoked is True
    assert load_developer_grant(settings=sandbox_settings) is None
    assert is_community_enabled(settings=sandbox_settings) is False


# ---------------------------------------------------------------------------
# 2. Main window default-locked state & Help menu integrity
# ---------------------------------------------------------------------------

def test_main_window_default_locked_hides_all_community_and_keeps_help_account(
    default_locked_window, qapp
):
    window, _settings, remote_calls = default_locked_window
    gc = window.gallery_controller

    # Top-level Community & Account menus MUST be hidden from menubar.
    assert window.ui_objs["menu_gallery"].menuAction().isVisible() is False
    assert window.ui_objs["menu_account"].menuAction().isVisible() is False
    assert window.ui_objs["action_open_gallery"].isVisible() is False
    assert window.ui_objs["action_my_account"].isVisible() is False

    # CRITICAL: Help -> Account MUST remain visible & enabled!
    help_account = window.ui_objs["action_account"]
    assert help_account in window.ui_objs["menu_help"].actions()
    assert help_account.isVisible() is True
    assert help_account.isEnabled() is True

    # Help menu cleanup: menu_quick_start removed; menu_guided_tour intact.
    assert "menu_quick_start" not in window.ui_objs
    assert window.ui_objs["menu_guided_tour"] in window.ui_objs["menu_help"].actions()
    assert window.ui_objs["menu_guided_tour"].isVisible() is True

    # No startup Gallery polling timer or background remote fetches.
    assert window._gallery_start_timer.isActive() is False
    gc.start(fetch_remote=True)
    assert gc._started is False
    assert gc.state_store._poll_timer.isActive() is False
    assert remote_calls == []

    # Fusion lifecycle does NOT open GalleryShowcaseWindow or UploadPromptDialog.
    apply_config_snapshot(
        gc.state_store,
        {
            "gallery_enabled": True,
            "upload_prompt_probability": 1.0,
            "gallery_terms_version": "2026-11-26",
        },
    )
    gc.allow_headless_upload_prompt = True
    gc.on_render_started(show_fusion=True)
    qapp.processEvents()
    assert gc.is_showing_showcase() is False
    assert gc.showcase_window is None

    fused = np.full((32, 32, 3), 180, dtype=np.uint8)
    window.fusion_result = fused
    assert gc.on_fusion_completed(fused, interactive=True) is None
    assert gc._upload_prompt_dialog is None

    # HonorCardDialog is suppressed when locked.
    apply_profile_snapshot(
        gc.state_store,
        {
            "user_id": "u1",
            "nickname": "User",
            "pending_honor_cards": [
                {
                    "card_id": "hc_locked_1",
                    "badge_key": "top_7d",
                    "period": "7d",
                    "rank": 1,
                    "work_id": "w1",
                    "work_title": "Test",
                    "points_awarded": 100,
                }
            ],
        },
    )
    qapp.processEvents()
    assert gc._honor_dialogs == []

    # Right-click context menu on result canvas is suppressed and has no Upload action.
    ctx_menu = gc.build_result_context_menu(window.lbl_result_img)
    assert ctx_menu.actions() == []
    assert gc.show_result_context_menu(QPoint(10, 10)) is None

    # Direct launch helpers refuse while locked.
    assert window.open_gallery_hub() is None
    assert window.open_account_center() is None
    assert window.open_gallery_showcase() is False
    assert window.open_gallery_screensaver() is None

    # Language switch keeps Community menus hidden and Help -> Account visible.
    trans.set_language("zh")
    qapp.processEvents()
    assert window.ui_objs["menu_gallery"].menuAction().isVisible() is False
    assert window.ui_objs["menu_account"].menuAction().isVisible() is False
    assert window.ui_objs["action_account"].isVisible() is True


# ---------------------------------------------------------------------------
# 3. AdvancedSettingsDialog Developer Options live unlock / enable / lock flow
# ---------------------------------------------------------------------------

def test_developer_options_dialog_async_unlock_enable_and_lock_live(
    default_locked_window, qapp
):
    window, _settings, _remote_calls = default_locked_window
    gc = window.gallery_controller
    main_tid = threading.get_ident()
    verify_threads: list[int] = []

    def _mock_verifier(key: str, *, bearer_token=None):
        verify_threads.append(threading.get_ident())
        if key == "ofdev_valid_community_key":
            return DeveloperKeyVerificationResult(
                ok=True,
                valid=True,
                status="valid",
                grant=DeveloperGrant(
                    key_prefix="ofdev_va...",
                    label="Maintainer Dev",
                    features=[FEATURE_COMMUNITY],
                    verified_at="2026-11-27T08:00:00Z",
                    expires_at="2099-12-31T23:59:59Z",
                    server_validated=True,
                ),
            )
        return DeveloperKeyVerificationResult(
            ok=True,
            valid=False,
            status="revoked",
            error_code="key_revoked",
            revoked=True,
        )

    dialog = AdvancedSettingsDialog(window)
    dialog._dev_key_verifier = _mock_verifier
    dialog.show()
    qapp.processEvents()
    try:
        # Locked default: gallery_settings_tab is hidden (5 tabs), key is masked.
        assert dialog.tabs.count() == 5
        assert dialog._is_gallery_tab_visible() is False
        assert dialog.tabs.tabText(4) == trans.t("developer_options_tab")
        dialog.tabs.setCurrentIndex(4)
        qapp.processEvents()

        assert dialog.edit_dev_key.echoMode() == QLineEdit.EchoMode.Password
        dialog.btn_dev_key_toggle.click()
        assert dialog.edit_dev_key.echoMode() == QLineEdit.EchoMode.Normal
        dialog.btn_dev_key_toggle.click()
        assert dialog.edit_dev_key.echoMode() == QLineEdit.EchoMode.Password

        assert dialog.chk_dev_community.isEnabled() is False
        assert dialog.chk_dev_community.isChecked() is False
        assert dialog.btn_dev_lock.isEnabled() is False
        assert dialog.lbl_dev_status_badge.text() == trans.t("dev_status_locked")

        # Invalid key -> async rejection, stays locked.
        dialog.edit_dev_key.setText("ofdev_wrong_key")
        dialog.btn_dev_verify.click()
        _wait_until(lambda: not dialog._dev_verifying)
        assert verify_threads[-1] != main_tid
        assert dialog.chk_dev_community.isEnabled() is False
        assert dialog.lbl_dev_feedback.isVisible() is True
        assert dialog.lbl_dev_feedback.text() == trans.t("dev_key_err_invalid")

        # Valid key -> async unlock on worker thread, clears input, enables switch.
        dialog.edit_dev_key.setText("ofdev_valid_community_key")
        dialog.btn_dev_verify.click()
        _wait_until(lambda: not dialog._dev_verifying and dialog.chk_dev_community.isEnabled())
        assert verify_threads[-1] != main_tid
        assert dialog.edit_dev_key.text() == ""
        assert dialog.btn_dev_lock.isEnabled() is True
        assert dialog.lbl_dev_status_badge.text() == trans.t("dev_status_unlocked")
        assert "ofdev_va..." in dialog.lbl_dev_status_detail.text()
        assert "Maintainer Dev" in dialog.lbl_dev_status_detail.text()
        # Still hidden until Community toggle checkbox is checked.
        assert dialog._is_gallery_tab_visible() is False
        assert dialog.tabs.count() == 5

        # Checking Community switch LIVE-SHOWS gallery_settings_tab while staying on Developer Options.
        dialog.chk_dev_community.setChecked(True)
        qapp.processEvents()
        assert dialog._is_gallery_tab_visible() is True
        assert dialog.tabs.count() == 6
        assert dialog.tabs.tabText(4) == trans.t("gallery_settings_tab")
        assert dialog.tabs.tabText(5) == trans.t("developer_options_tab")
        assert dialog.tabs.currentWidget() is dialog._developer_scroll_page

        # Apply -> live-shows top-level menus & starts GalleryController!
        dialog._apply()
        assert dialog.result() == QDialog.DialogCode.Accepted
    finally:
        dialog.deleteLater()
        qapp.processEvents()

    assert is_community_enabled() is True
    assert window.ui_objs["menu_gallery"].menuAction().isVisible() is True
    assert window.ui_objs["menu_account"].menuAction().isVisible() is True
    assert window.ui_objs["action_account"].isVisible() is True
    assert gc._started is True
    assert gc.state_store._poll_timer.isActive() is True

    # Showcase & context menu now work live.
    assert gc.show_gallery_showcase(force=True) is True
    assert gc.is_showing_showcase() is True
    ctx_menu = gc.build_result_context_menu(window.lbl_result_img)
    assert [a.objectName() for a in ctx_menu.actions()] == ["action_upload_to_gallery"]

    # Re-open dialog and click "Lock & Clear Key" -> re-hides everything live!
    dialog2 = AdvancedSettingsDialog(window)
    dialog2.show()
    qapp.processEvents()
    try:
        assert dialog2._is_gallery_tab_visible() is True
        assert dialog2.tabs.count() == 6
        dialog2.tabs.setCurrentIndex(5)
        dialog2.btn_dev_lock.click()
        qapp.processEvents()

        assert dialog2._is_gallery_tab_visible() is False
        assert dialog2.tabs.count() == 5
        assert dialog2.chk_dev_community.isEnabled() is False
        assert dialog2.chk_dev_community.isChecked() is False
        assert dialog2.btn_dev_lock.isEnabled() is False
        assert dialog2.lbl_dev_status_badge.text() == trans.t("dev_status_locked")
    finally:
        dialog2.reject()
        dialog2.deleteLater()
        qapp.processEvents()

    assert load_developer_grant() is None
    assert is_community_enabled() is False
    assert window.ui_objs["menu_gallery"].menuAction().isVisible() is False
    assert window.ui_objs["menu_account"].menuAction().isVisible() is False
    assert window.ui_objs["action_account"].isVisible() is True
    assert gc.is_showing_showcase() is False
    assert gc.state_store._poll_timer.isActive() is False


def test_startup_background_revalidation_revokes_live_on_main_window(
    default_locked_window, qapp, monkeypatch
):
    window, settings, _remote_calls = default_locked_window
    save_developer_grant(
        DeveloperGrant(
            key_prefix="ofdev_rv...",
            label="Revoked Later",
            features=[FEATURE_COMMUNITY],
            verified_at="2026-11-27T00:00:00Z",
            server_validated=True,
        ),
        key="ofdev_rv_secret",
        settings=settings,
    )
    set_developer_feature_enabled(FEATURE_COMMUNITY, True, settings=settings)
    window.apply_community_feature_state(start_remote=True)
    assert window.ui_objs["menu_gallery"].menuAction().isVisible() is True
    assert window.gallery_controller.state_store._poll_timer.isActive() is True

    # Simulate background revalidation discovering server revocation.
    def _fake_revalidate(**_kwargs):
        clear_developer_grant(settings=settings)
        return DeveloperKeyVerificationResult(
            ok=True,
            valid=False,
            status="revoked",
            error_code="key_revoked",
            revoked=True,
        )

    monkeypatch.setattr(dev_opts, "revalidate_saved_developer_grant", _fake_revalidate)
    window._start_developer_grant_revalidation()
    _wait_until(lambda: not window.ui_objs["menu_gallery"].menuAction().isVisible())
    assert window.ui_objs["menu_account"].menuAction().isVisible() is False
    assert window.ui_objs["action_account"].isVisible() is True
    assert window.gallery_controller.state_store._poll_timer.isActive() is False


# ---------------------------------------------------------------------------
# 4. 12-language i18n completeness
# ---------------------------------------------------------------------------

def test_developer_options_translations_complete_in_all_12_languages():
    assert len(trans.PACK_ORDER) == 12
    for code in trans.PACK_ORDER:
        table = trans.translations[code]
        for key in _DEVELOPER_OPTIONS_KEYS:
            val = table.get(key)
            assert isinstance(val, str) and val.strip(), f"{code}: missing {key}"
