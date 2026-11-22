"""Tests for multi-page TIFF support and crash auto-recovery."""
import os
import shutil
import tempfile

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


class TestMultiPageTiffRead:
    def test_multipage_expands_to_frames(self, tmp_path):
        from core.image_loader import ImageStackLoader
        pages = [(np.arange(i, i + 30 * 40 * 3).reshape(30, 40, 3) % 255).astype(np.uint8)
                 for i in range(3)]
        mp = str(tmp_path / "zstack.tif")
        assert cv2.imwritemulti(mp, pages)

        ok, msg, imgs, names = ImageStackLoader().load_from_folder(str(tmp_path))
        assert ok, msg
        assert len(imgs) == 3
        assert names[0] == "zstack_page_001.tif"
        assert names[2] == "zstack_page_003.tif"

    def test_multipage_via_filepaths(self, tmp_path):
        from core.image_loader import ImageStackLoader
        pages = [np.zeros((20, 30, 3), np.uint8) + i * 40 for i in range(2)]
        mp = str(tmp_path / "stack.tif")
        assert cv2.imwritemulti(mp, pages)

        ok, _msg, imgs, names = ImageStackLoader().load_from_filepaths([mp])
        assert ok and len(imgs) == 2 and "page_001" in names[0]

    def test_single_page_tiff_unchanged(self, tmp_path):
        from core.image_loader import ImageStackLoader, _read_tiff_pages
        single = str(tmp_path / "one.tif")
        cv2.imwrite(single, np.zeros((10, 10, 3), np.uint8))
        assert _read_tiff_pages(single) is None  # falls through to normal path
        ok, _m, imgs, names = ImageStackLoader().load_from_filepaths([single])
        assert ok and len(imgs) == 1 and names[0] == "one.tif"

    def test_16bit_pages_preserved(self, tmp_path):
        from core.image_loader import ImageStackLoader
        pages = [(np.random.rand(20, 30, 3) * 65535).astype(np.uint16) for _ in range(3)]
        mp = str(tmp_path / "u16.tif")
        assert cv2.imwritemulti(mp, pages)
        ok, _m, imgs, _names = ImageStackLoader().load_from_folder(str(tmp_path))
        assert ok and imgs[0].dtype == np.uint16


class TestMultiPageTiffWrite:
    def test_roundtrip_unicode_path(self, tmp_path):
        from utils.image_utils import imwrite_multi_tiff
        pages = [(np.random.rand(30, 40, 3) * 255).astype(np.uint8) for _ in range(4)]
        target = str(tmp_path / "导出栈.tif")
        assert imwrite_multi_tiff(target, pages)
        ok, back = cv2.imdecodemulti(
            np.fromfile(target, dtype=np.uint8),
            flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        assert ok and len(back) == 4 and back[0].dtype == np.uint8

    def test_16bit_roundtrip(self, tmp_path):
        from utils.image_utils import imwrite_multi_tiff
        pages = [(np.random.rand(20, 30, 3) * 65535).astype(np.uint16) for _ in range(2)]
        target = str(tmp_path / "u16.tif")
        assert imwrite_multi_tiff(target, pages)
        ok, back = cv2.imreadmulti(target, flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        assert ok and back[0].dtype == np.uint16

    def test_empty_rejected(self, tmp_path):
        from utils.image_utils import imwrite_multi_tiff
        assert not imwrite_multi_tiff(str(tmp_path / "x.tif"), [])


class TestRecovery:
    @pytest.fixture(autouse=True)
    def recovery_mod(self, tmp_path, monkeypatch):
        import utils.recovery as rec
        base = str(tmp_path / "data")
        os.makedirs(base, exist_ok=True)
        monkeypatch.setattr(rec, "_base_dir", lambda: base)
        return rec

    def _fake_window(self, stack_dir, n=2):
        """Minimal object satisfying utils.project_file._collect_state."""
        from utils.validators import LabelAdder

        class _Mgr:
            export_state = lambda self: {}
            import_state = lambda self, s: None

        class _W:
            pass

        w = _W()
        w.raw_images = [np.zeros((10, 10, 3), np.uint8) for _ in range(n)]
        # 下拉单选契约（与真实窗口一致）
        w._fusion, w._alignment = "guided_filter", "ecc"
        w.current_fusion_method = lambda: w._fusion
        w.current_alignment_method = lambda: w._alignment
        w.set_fusion_method = lambda code: setattr(w, "_fusion", code)
        w.set_alignment_method = lambda code: setattr(w, "_alignment", code)
        from core.algorithms import resolve_fusion_parameters
        w.fusion_parameters_for = lambda method, depth=False: resolve_fusion_parameters(
            method, {"kernel_size": 21} if method == "guided_filter" and not depth else None,
            depth=depth)
        w.chk_quick_preview = type("Q", (), {"isChecked": lambda s: False})()
        w.current_display_index = 0
        w.image_filenames = [f"f{i}.png" for i in range(n)]
        w.current_folder_path = stack_dir
        w.current_scale_factor = 1.0
        w.thread_count, w.tile_enabled = 4, True
        w.tile_block_size, w.tile_overlap, w.tile_threshold = 1024, 256, 2048
        w.reg_downscale_width, w.ai_batch_size = 1024, 2
        w.label_manager = _Mgr()
        # the real files must exist for validate_project
        for name in w.image_filenames:
            cv2.imwrite(os.path.join(stack_dir, name), np.zeros((5, 5, 3), np.uint8))
        return w

    def test_snapshot_lock_cycle(self, tmp_path, recovery_mod):
        rec = recovery_mod
        stack = str(tmp_path / "stack"); os.makedirs(stack)
        w = self._fake_window(stack)
        w.sharpen_strength = 145

        assert rec.crash_snapshot_path() is None   # clean: no lock
        assert rec.write_snapshot(w)              # snapshot saved
        rec.init_lock()                           # session goes live
        path = rec.crash_snapshot_path()
        assert path == rec.session_path()
        state = rec.load_snapshot_state(path)
        assert rec.snapshot_frame_count(state) == 2
        assert state['settings']['sharpen_strength'] == 145
        assert state['settings']['sharpen_output'] is False
        rec.mark_clean_exit()                     # normal close
        assert rec.crash_snapshot_path() is None  # lock cleared, snapshot kept
        assert os.path.isfile(rec.session_path())

    @pytest.mark.parametrize("project_path,dirty", [("A.ofproj", True), ("A.ofproj", False), (None, True)])
    def test_snapshot_records_document_state(self, tmp_path, recovery_mod, project_path, dirty):
        stack = str(tmp_path / "stack")
        os.makedirs(stack)
        window = self._fake_window(stack)
        window.project_path = project_path
        window._project_dirty = dirty
        assert recovery_mod.write_snapshot(window)
        assert recovery_mod.load_snapshot_state()["recovery_document"] == {
            "version": 1, "project_path": project_path or "", "dirty": dirty}

    def test_override_avoids_portable_probe(self, tmp_path, monkeypatch):
        import utils.recovery as rec
        monkeypatch.undo()
        monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "isolated"))
        monkeypatch.setattr(rec.app_paths, "is_portable_mode", lambda: pytest.fail("portable probe"))
        from utils import settings_store
        monkeypatch.setattr(settings_store, "_settings_singleton", None)
        assert os.path.normpath(settings_store.get_settings().fileName()) == os.path.normpath(
            str(tmp_path / "isolated" / "settings.ini"))
        assert rec._base_dir() == str(tmp_path / "isolated")
        assert rec.session_path().startswith(str(tmp_path / "isolated"))
        assert rec.lock_path().startswith(str(tmp_path / "isolated"))
        assert rec.pending_restore_path().startswith(str(tmp_path / "isolated"))

    def test_pending_ignores_missing_sources(self, tmp_path, recovery_mod):
        rec = recovery_mod
        stack = str(tmp_path / "gone"); os.makedirs(stack)
        w = self._fake_window(stack)
        assert rec.write_snapshot(w)
        rec.init_lock()
        # A missing source does not destroy the candidate or produce valid state.
        shutil.rmtree(stack)
        assert rec.crash_snapshot_path() == rec.session_path()
        assert rec.load_snapshot_state(rec.crash_snapshot_path()) is None


    def test_cleared_workspace_is_not_offered_after_crash(self, tmp_path, recovery_mod):
        from types import SimpleNamespace
        from controllers.source_manager import SourceManager

        rec = recovery_mod
        stack = str(tmp_path / "stack")
        os.makedirs(stack)
        window = self._fake_window(stack)
        window.transform_manager = SimpleNamespace(
            invalidate_processing_results=lambda **kwargs: None,
            reload_image_stack=lambda **kwargs: None)
        window.label_manager.reset_labels = lambda: None
        assert rec.write_snapshot(window)
        rec.init_lock()
        assert rec.snapshot_frame_count(rec.load_snapshot_state(rec.crash_snapshot_path())) == 2

        SourceManager(window).clear_image_stack()

        assert window.raw_images == []
        assert rec.crash_snapshot_path() is None
        assert not os.path.exists(rec.session_path())

    @pytest.mark.parametrize('restore', [False, True], ids=['start-fresh', 'restore'])
    def test_recovery_choice_during_source_probe(self, qapp, tmp_path, recovery_mod, monkeypatch, restore):
        from pathlib import Path
        from threading import Event, current_thread

        from PyQt6.QtCore import QTimer, Qt
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QMessageBox, QProgressBar, QWidget

        from locales import trans
        from ui.session_recovery import prompt_crash_recovery

        rec = recovery_mod
        stack = tmp_path / 'stack'
        stack.mkdir()
        saved = self._fake_window(str(stack))
        saved.project_path = str(tmp_path / 'original.ofproj')
        saved._project_dirty = True
        assert rec.write_snapshot(saved)
        rec.init_lock()
        snapshot = Path(rec.session_path())
        original_bytes = snapshot.read_bytes()

        # Stall the actual filesystem probe for a saved source, simulating
        # an unresponsive share without relying on network timing.
        slow_path = os.path.join(stack, saved.image_filenames[0])
        release = Event()
        returned = Event()
        workers = []
        isfile = os.path.isfile

        def delayed_isfile(path):
            if os.fspath(path) == slow_path:
                workers.append(current_thread())
                release.wait(5)
                returned.set()
            return isfile(path)

        monkeypatch.setattr(os.path, 'isfile', delayed_isfile)
        parent = QWidget()
        observed = []
        clicked = []
        driver = QTimer()
        watchdog = QTimer()
        watchdog.setSingleShot(True)

        def close_timed_out_prompt():
            box = qapp.activeModalWidget()
            if isinstance(box, QMessageBox) and box.parentWidget() is parent:
                box.reject()

        def drive_prompt():
            box = qapp.activeModalWidget()
            if not workers or not isinstance(box, QMessageBox) or box.parentWidget() is not parent:
                return
            buttons = {button.text(): button for button in box.buttons()}
            restore_btn = buttons.get(trans.t('btn_restore'))
            fresh_btn = buttons.get(trans.t('btn_start_fresh'))
            if restore_btn is None or fresh_btn is None:
                return
            if not observed:
                progress = box.findChild(QProgressBar)
                observed.append((box.text(), restore_btn.isEnabled(), fresh_btn.isEnabled(), returned.is_set(),
                                 progress is not None and progress.isVisible() and progress.maximum() == 0))
                if restore:
                    release.set()
            button = restore_btn if restore else fresh_btn
            if button.isEnabled():
                clicked.append(button.text())
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)

        watchdog.timeout.connect(close_timed_out_prompt)
        watchdog.start(5000)
        driver.timeout.connect(drive_prompt)
        driver.start(10)
        try:
            state = prompt_crash_recovery(parent)
            assert observed == [(trans.t('recovery_checking'), False, True, False, True)]
            assert clicked == [trans.t('btn_restore' if restore else 'btn_start_fresh')]
            if restore:
                assert rec.snapshot_frame_count(state) == 2
                assert state['recovery_document']['project_path'] == saved.project_path
                assert state['recovery_document']['dirty'] is True
            else:
                assert state is None
                assert not returned.is_set()
        finally:
            driver.stop()
            watchdog.stop()
            release.set()
            for worker in workers:
                if worker is not current_thread():
                    worker.join(5)
                    assert not worker.is_alive()

        qapp.processEvents()
        assert qapp.activeModalWidget() is None
        assert snapshot.read_bytes() == original_bytes
        assert rec.crash_snapshot_path() == str(snapshot)

    @pytest.mark.parametrize('cancel', [False, True], ids=['complete', 'cancel'])
    def test_startup_replay_reports_progress_and_keeps_snapshot(self, qapp, tmp_path, recovery_mod,
                                                              monkeypatch, cancel):
        from pathlib import Path
        from threading import Event, current_thread
        from time import monotonic

        from PyQt6.QtCore import QSettings, QTimer, Qt
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QMessageBox

        from core import source_frames
        from locales import trans
        from main import OpenFocus
        from ui.session_recovery import restore_crash_session
        from utils import settings_store

        stack = tmp_path / 'stack'
        stack.mkdir()
        saved = self._fake_window(str(stack))
        saved.project_path = str(tmp_path / 'original.ofproj')
        saved._project_dirty = True
        saved.source_transforms = [{'operation': 'flip', 'code': 0}]
        assert recovery_mod.write_snapshot(saved)
        recovery_mod.init_lock()
        snapshot = Path(recovery_mod.session_path())
        original_bytes = snapshot.read_bytes()

        settings = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
        monkeypatch.setattr(settings_store, '_settings_singleton', settings)
        window = OpenFocus()
        manager = window.source_manager
        previous_images = window.image_loader.images
        release = Event()
        replay_threads = []
        gui_thread = current_thread()
        transform_frame = source_frames.transform_frame

        def delayed_transform(frame, operation, **kwargs):
            if operation == 'flip':
                replay_threads.append(current_thread())
                release.wait(15)
            return transform_frame(frame, operation, **kwargs)

        monkeypatch.setattr(source_frames, 'transform_frame', delayed_transform)
        driver = QTimer()
        choice_deadline = monotonic() + 10

        def choose_restore():
            box = qapp.activeModalWidget()
            if isinstance(box, QMessageBox) and box.parentWidget() is window:
                if monotonic() >= choice_deadline:
                    box.reject()
                    return
                for button in box.buttons():
                    if button.text() == trans.t('btn_restore') and button.isEnabled():
                        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
                        break

        def wait_until(predicate):
            deadline = monotonic() + 10
            while not predicate() and monotonic() < deadline:
                QTest.qWait(10)
            assert predicate()

        driver.timeout.connect(choose_restore)
        driver.start(10)
        try:
            assert restore_crash_session(window)
            driver.stop()
            window.show()
            status = manager._load_status_label
            wait_until(lambda: replay_threads and status.text() == trans.t('update_restore_running'))
            assert len(replay_threads) == 1 and replay_threads[0] is not gui_thread
            assert qapp.activeModalWidget() is None and not window.raw_images
            progress = manager._load_progress_bar
            assert progress.isVisible() and (progress.minimum(), progress.maximum()) == (0, 0)
            button = manager._load_cancel_button
            assert button.isVisible() and button.isEnabled()
            if cancel:
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
                assert not button.isEnabled() and status.text() == trans.t('depth_cancelling')

            release.set()
            wait_until(lambda: manager._load_worker is None)
            assert not progress.isVisible() and qapp.activeModalWidget() is None
            if cancel:
                assert len(replay_threads) == 1
                assert not window.raw_images and getattr(window, 'project_path', None) is None
                assert window.image_loader.images is previous_images
            else:
                assert window.image_filenames == saved.image_filenames
                assert window.source_transforms == saved.source_transforms
                assert window.project_path == saved.project_path and window._project_dirty
            assert snapshot.read_bytes() == original_bytes
        finally:
            driver.stop()
            release.set()
            window._force_quit = True
            window.close()
            window.deleteLater()
            qapp.processEvents()
