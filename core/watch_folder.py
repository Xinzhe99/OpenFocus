"""Non-blocking Watch Folder / Folder Monitor engine for OpenFocus.

Continuously watches an input directory (hybrid ``QFileSystemWatcher`` +
``QTimer`` polling for reliability across local, USB, and network volumes),
enforces a write-stabilization settle delay so tethered camera transfers never
trigger partial stacks, deduplicates completed stacks via an in-memory and
optional persisted JSON ledger, and renders complete stacks sequentially in a
background worker thread.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
import tempfile
import time
from typing import Callable, Optional, Sequence

from PyQt6.QtCore import QFileSystemWatcher, QObject, QThread, QTimer, pyqtSignal

from core.algorithms import FUSION_DEFAULT, alignment_steps, get_fusion, get_registration
from core.filename_template import apply_output_naming, build_template_context
from core.image_loader import ImageStackLoader
from utils.export_paths import derived_aligned_source, publish_unique
from utils.logging_setup import get_logger
from utils.platform_utils import configure_native_worker_stack
from utils.sorting import natural_sort_key

logger = get_logger(__name__)

_TEMP_SUFFIXES = frozenset({
    ".tmp", ".temp", ".part", ".partial", ".crdownload", ".download", ".bak",
})


@dataclass(frozen=True)
class WatchFolderConfig:
    """Configuration snapshot for a FolderMonitorEngine session."""

    watch_dir: str
    output_dir: str
    fusion_method: str = FUSION_DEFAULT
    alignment_method: str = "none"
    output_format: str = "png"
    settle_seconds: float = 2.0
    poll_interval_ms: int = 1000
    min_frames: int = 2
    filename_template: str = ""
    output_name_prefix: str = ""
    play_sound: bool = False
    ledger_path: Optional[str] = None
    thread_count: int = 0
    resource_preference: str = "balanced"
    use_gpu: bool = True


@dataclass(frozen=True)
class StackCandidate:
    """A stabilized stack ready for background rendering."""

    stack_id: str
    name: str
    source_dir: str
    file_paths: tuple[str, ...]
    signature: str
    is_root_batch: bool = False


def _is_ignored_filename(name: str) -> bool:
    if not name or name.startswith(".") or name.startswith("~"):
        return True
    lower = name.lower()
    _, ext = os.path.splitext(lower)
    if ext in _TEMP_SUFFIXES:
        return True
    if derived_aligned_source(name):
        return True
    return False


def _supported_extensions() -> frozenset[str]:
    return frozenset(ext.lower() for ext in ImageStackLoader.SUPPORTED_FORMATS)


def _list_stack_files(
    directory: str,
    *,
    excluded_paths: frozenset[str] = frozenset(),
) -> tuple[list[tuple[str, str, int, int]], bool]:
    """Return sorted ``(name, full_path, size, mtime_ns)`` and a ``writing`` flag.

    If any candidate file is zero-byte or temporarily unreadable (locked while
    being written), ``writing`` is set to ``True`` so the caller waits for
    stabilization before accepting the folder.
    """
    exts = _supported_extensions()
    entries: list[tuple[str, str, int, int]] = []
    writing = False
    try:
        names = os.listdir(directory)
    except OSError:
        return [], True

    for name in sorted(names, key=natural_sort_key):
        if name.startswith(".") or name.startswith("~"):
            continue
        full = os.path.join(directory, name)
        if not os.path.isfile(full):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in _TEMP_SUFFIXES:
            # Active temporary transfer file in directory: hold off stabilization.
            writing = True
            continue
        if ext.lower() not in exts or derived_aligned_source(name):
            continue
        norm_full = os.path.normcase(os.path.realpath(full))
        if norm_full in excluded_paths:
            continue
        try:
            st = os.stat(full)
        except OSError:
            writing = True
            continue
        if st.st_size <= 0:
            writing = True
            continue
        entries.append((name, full, int(st.st_size), int(st.st_mtime_ns)))

    return entries, writing


def _snapshot_digest(entries: Sequence[tuple[str, str, int, int]]) -> str:
    digest = hashlib.sha256()
    for name, _full, size, mtime_ns in entries:
        digest.update(f"{name}|{size}|{mtime_ns}\n".encode("utf-8", errors="replace"))
    return digest.hexdigest()


def _default_render_stack(
    candidate: StackCandidate,
    config: WatchFolderConfig,
    should_cancel: Callable[[], bool],
) -> str:
    """Run a detected stack through the standard OpenFocus alignment + fusion pipeline."""
    from core.frame_storage import scratch_directory
    from core.multi_focus_fusion import MultiFocusFusion
    from core.registration import ImageRegistration, RegistrationCancelled
    from core.source_frames import resolve_frame
    from utils.image_utils import get_imwrite_params, imwrite_auto, prepare_fusion_input, quantize_fuse_output

    if should_cancel():
        raise RegistrationCancelled()

    os.makedirs(config.output_dir, exist_ok=True)
    loader = ImageStackLoader()
    loader.should_cancel = should_cancel

    ext = (config.output_format or "png").lstrip(".").lower()
    with scratch_directory(separate=True):
        if ext == "dng":
            ok, message, images, filenames = loader.load_linear_raw_stack(
                list(candidate.file_paths),
                thread_count=config.thread_count,
                resource_preference=config.resource_preference,
            )
        else:
            ok, message, images, filenames = loader.load_from_filepaths(
                list(candidate.file_paths),
                thread_count=config.thread_count,
                resource_preference=config.resource_preference,
            )
        if should_cancel():
            raise RegistrationCancelled()
        if not ok or not images:
            raise ValueError(message or f"Failed to load stack: {candidate.name}")
        if ext == "dng" and getattr(loader, "linear_dng_metadata", None) is None:
            raise ValueError(message or f"Linear DNG export requires matching camera RAW frames ({candidate.name})")
        if len(images) < 2 and config.fusion_method != "none":
            raise ValueError(f"Fusion requires at least 2 frames ({candidate.name})")

        steps = alignment_steps(config.alignment_method)
        processed = list(images)
        if steps:
            reg = ImageRegistration(
                method=config.alignment_method,
                use_gpu=config.use_gpu,
                defer_warps=(
                    config.fusion_method == "none"
                    or get_fusion(config.fusion_method).streaming_sources
                ),
            )
            processed = reg.process(
                processed,
                thread_count=config.thread_count,
                resource_preference=config.resource_preference,
                should_cancel=should_cancel,
            )

        if should_cancel():
            raise RegistrationCancelled()

        if config.fusion_method and config.fusion_method != "none":
            fusion = MultiFocusFusion(
                algorithm=config.fusion_method,
                use_gpu=config.use_gpu,
                resource_preference=config.resource_preference,
            )
            fusion_input, is_16bit = prepare_fusion_input(processed, should_cancel)
            fused = fusion.fuse(
                fusion_input,
                thread_count=config.thread_count,
                should_cancel=should_cancel,
            )
            result_img = quantize_fuse_output(fused, is_16bit)
        else:
            result_img = resolve_frame(processed[0])

        if should_cancel():
            raise RegistrationCancelled()

        result_img = resolve_frame(result_img)
        h, w = result_img.shape[:2]
        method_tag = (
            "None"
            if not config.fusion_method or config.fusion_method == "none"
            else get_fusion(config.fusion_method).file_tag
        )
        align_tag = (
            "NoAlign"
            if not config.alignment_method or config.alignment_method == "none"
            else get_registration(config.alignment_method).file_tag
        )
        ctx = build_template_context(
            filenames=filenames or [os.path.basename(p) for p in candidate.file_paths],
            folder_path=candidate.name,
            method=method_tag,
            align=align_tag,
            frame_count=len(processed),
            width=w,
            height=h,
        )
        stem = apply_output_naming(
            candidate.name,
            template=config.filename_template,
            prefix=config.output_name_prefix,
            context=ctx,
        )
        target = os.path.join(config.output_dir, f"{stem}.{ext}")
        exif = getattr(loader, "source_exif", b"") or b""
        icc = b"" if ext == "dng" else (getattr(loader, "source_icc_profile", b"") or b"")
        linear_meta = getattr(loader, "linear_dng_metadata", None) if ext == "dng" else None
        linear_px = getattr(loader, "px_size_um", None) if ext == "dng" else None
        params = get_imwrite_params(f".{ext}")
        written_path = publish_unique(
            target,
            lambda staging: imwrite_auto(
                staging,
                result_img,
                params,
                exif=exif,
                icc_profile=icc,
                linear_dng_metadata=linear_meta,
                linear_dng_px_size_um=linear_px,
            ),
            should_cancel=should_cancel,
        )
        return written_path


class _WatchRenderWorker(QThread):
    """Background worker executing one watched stack render."""

    completed = pyqtSignal(object, str, float)  # candidate, output_path, elapsed
    failed = pyqtSignal(object, str)            # candidate, error_message

    def __init__(
        self,
        candidate: StackCandidate,
        config: WatchFolderConfig,
        render_fn: Callable[[StackCandidate, WatchFolderConfig, Callable[[], bool]], str],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        configure_native_worker_stack(self)
        self.candidate = candidate
        self.config = config
        self._render_fn = render_fn
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        started = time.monotonic()
        try:
            output_path = self._render_fn(
                self.candidate, self.config, lambda: self._cancelled
            )
            if self._cancelled:
                self.failed.emit(self.candidate, "CANCELLED")
                return
            self.completed.emit(self.candidate, str(output_path), time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.candidate, str(exc) or type(exc).__name__)


class FolderMonitorEngine(QObject):
    """Watches a folder for completed focus stacks and renders them automatically."""

    started = pyqtSignal(str)
    stopped = pyqtSignal()
    paused_changed = pyqtSignal(bool)
    stack_detected = pyqtSignal(str, int)
    stack_completed = pyqtSignal(str, str, float)
    stack_failed = pyqtSignal(str, str)
    log_message = pyqtSignal(str, str)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QObject] = None,
        *,
        render_fn: Optional[
            Callable[[StackCandidate, WatchFolderConfig, Callable[[], bool]], str]
        ] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        self._render_fn = render_fn or _default_render_stack
        self._clock = clock
        self._config: Optional[WatchFolderConfig] = None
        self._running = False
        self._paused = False
        self._status = "idle"
        self._processed_count = 0

        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_fs_event)
        self._watcher.fileChanged.connect(self._on_fs_event)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_poll_tick)

        # stack_id -> (signature, first_stable_monotonic)
        self._pending_stability: dict[str, tuple[str, float]] = {}
        # Dedup ledgers
        self._processed_stacks: dict[str, str] = {}
        self._processed_root_files: set[str] = set()
        self._in_flight_ids: set[str] = set()
        self._queue: deque[StackCandidate] = deque()
        self._active_worker: Optional[_WatchRenderWorker] = None
        self._retiring_workers: set[_WatchRenderWorker] = set()
        self._root_batch_index = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def status(self) -> str:
        return self._status

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def config(self) -> Optional[WatchFolderConfig]:
        return self._config

    def _set_status(self, state: str) -> None:
        if state != self._status:
            self._status = state
            self.status_changed.emit(state)

    def _load_ledger(self, ledger_path: Optional[str]) -> None:
        self._processed_stacks.clear()
        self._processed_root_files.clear()
        if not ledger_path or not os.path.isfile(ledger_path):
            return
        try:
            with open(ledger_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                stacks = payload.get("stacks")
                if isinstance(stacks, dict):
                    self._processed_stacks.update(
                        {str(k): str(v) for k, v in stacks.items()}
                    )
                root_files = payload.get("root_files")
                if isinstance(root_files, list):
                    self._processed_root_files.update(str(p) for p in root_files)
                self._root_batch_index = int(payload.get("root_batch_index", 0) or 0)
        except Exception:
            logger.warning("Could not load watch-folder ledger %s", ledger_path, exc_info=True)

    def _save_ledger(self) -> None:
        cfg = self._config
        if cfg is None or not cfg.ledger_path:
            return
        ledger_dir = os.path.dirname(os.path.abspath(cfg.ledger_path))
        try:
            os.makedirs(ledger_dir, exist_ok=True)
            data = {
                "stacks": dict(self._processed_stacks),
                "root_files": sorted(self._processed_root_files),
                "root_batch_index": self._root_batch_index,
            }
            fd, tmp_path = tempfile.mkstemp(prefix=".openfocus-ledger-", suffix=".tmp", dir=ledger_dir)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, sort_keys=True)
                os.replace(tmp_path, cfg.ledger_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception:
            logger.warning("Could not persist watch-folder ledger", exc_info=True)

    def _reset_watcher(self) -> None:
        dirs = self._watcher.directories()
        if dirs:
            self._watcher.removePaths(dirs)
        files = self._watcher.files()
        if files:
            self._watcher.removePaths(files)

    def _sync_watcher_dirs(self, subdirs: Sequence[str]) -> None:
        cfg = self._config
        if cfg is None or not self._running:
            return
        wanted = {os.path.normpath(cfg.watch_dir)}
        for path in subdirs:
            wanted.add(os.path.normpath(path))
        current = {os.path.normpath(p) for p in self._watcher.directories()}
        to_remove = sorted(current - wanted)
        if to_remove:
            self._watcher.removePaths(to_remove)
        to_add = [p for p in sorted(wanted - current) if os.path.isdir(p)]
        if to_add:
            self._watcher.addPaths(to_add)

    def start(self, config: WatchFolderConfig) -> None:
        """Start monitoring ``config.watch_dir``."""
        watch_dir = os.path.abspath(config.watch_dir or "")
        output_dir = os.path.abspath(config.output_dir or "")
        if not watch_dir or not os.path.isdir(watch_dir):
            raise ValueError(f"Watch folder does not exist: {config.watch_dir}")
        if not output_dir:
            raise ValueError("Output folder must be specified")
        os.makedirs(output_dir, exist_ok=True)

        if self._running:
            self.stop()

        self._config = WatchFolderConfig(
            watch_dir=watch_dir,
            output_dir=output_dir,
            fusion_method=config.fusion_method,
            alignment_method=config.alignment_method,
            output_format=config.output_format,
            settle_seconds=max(0.0, float(config.settle_seconds)),
            poll_interval_ms=max(100, int(config.poll_interval_ms)),
            min_frames=max(1, int(config.min_frames)),
            filename_template=config.filename_template or "",
            output_name_prefix=config.output_name_prefix or "",
            play_sound=bool(config.play_sound),
            ledger_path=config.ledger_path,
            thread_count=int(config.thread_count),
            resource_preference=config.resource_preference,
            use_gpu=bool(config.use_gpu),
        )
        self._pending_stability.clear()
        self._in_flight_ids.clear()
        self._queue.clear()
        self._processed_count = 0
        self._load_ledger(self._config.ledger_path)

        self._running = True
        self._paused = False
        self._reset_watcher()
        self._watcher.addPath(watch_dir)
        self._timer.start(self._config.poll_interval_ms)

        self._set_status("watching")
        self.started.emit(watch_dir)
        self.log_message.emit("info", f"Monitoring started: {watch_dir}")
        # Perform an initial scan immediately to baseline existing entries.
        self.scan_once()

    def pause(self) -> None:
        if not self._running or self._paused:
            return
        self._paused = True
        self._set_status("paused")
        self.paused_changed.emit(True)
        self.log_message.emit("warning", "Monitoring paused")

    def resume(self) -> None:
        if not self._running or not self._paused:
            return
        self._paused = False
        self._set_status("processing" if self._active_worker is not None else "watching")
        self.paused_changed.emit(False)
        self.log_message.emit("info", "Monitoring resumed")
        self.scan_once()
        self._drain_queue()

    def stop(self) -> None:
        if not self._running and self._active_worker is None:
            return
        self._running = False
        self._paused = False
        self._timer.stop()
        self._reset_watcher()
        self._pending_stability.clear()
        self._queue.clear()
        self._in_flight_ids.clear()

        worker = self._active_worker
        self._active_worker = None
        if worker is not None:
            worker.cancel()
            if worker.isRunning():
                self._retiring_workers.add(worker)
                worker.finished.connect(lambda w=worker: self._retire_worker(w))
            else:
                worker.deleteLater()

        self._set_status("idle")
        self.stopped.emit()
        self.log_message.emit("info", "Monitoring stopped")

    def shutdown(self, timeout_ms: int = 3000) -> None:
        """Stop monitoring and wait briefly for any active worker before exit."""
        self.stop()
        for worker in tuple(self._retiring_workers):
            if worker.isRunning():
                worker.wait(timeout_ms)

    def _retire_worker(self, worker: _WatchRenderWorker) -> None:
        self._retiring_workers.discard(worker)
        worker.deleteLater()

    def _on_fs_event(self, _path: str) -> None:
        if self._running and not self._paused:
            self.scan_once()

    def _on_poll_tick(self) -> None:
        if self._running and not self._paused:
            self.scan_once()

    def scan_once(self, now: Optional[float] = None) -> list[StackCandidate]:
        """Scan the watch directory once and enqueue newly stabilized stacks.

        Accepts an explicit monotonic timestamp *now* for deterministic testing.
        """
        cfg = self._config
        if cfg is None or not self._running or self._paused:
            return []
        if not os.path.isdir(cfg.watch_dir):
            return []

        clock = self._clock() if now is None else float(now)
        output_real = os.path.normcase(os.path.realpath(cfg.output_dir))

        active_ids: set[str] = set()
        subdirs_to_watch: list[str] = []
        enqueued: list[StackCandidate] = []

        try:
            root_names = sorted(os.listdir(cfg.watch_dir), key=natural_sort_key)
        except OSError:
            return []

        # 1. Inspect child subdirectories as individual stacks.
        for name in root_names:
            if name.startswith(".") or name.startswith("~"):
                continue
            sub_path = os.path.join(cfg.watch_dir, name)
            if not os.path.isdir(sub_path):
                continue
            sub_real = os.path.normcase(os.path.realpath(sub_path))
            if sub_real == output_real or output_real.startswith(sub_real + os.sep):
                continue
            subdirs_to_watch.append(sub_path)
            stack_id = f"dir:{name}"
            entries, writing = _list_stack_files(sub_path)
            if len(entries) < cfg.min_frames:
                self._pending_stability.pop(stack_id, None)
                continue
            active_ids.add(stack_id)
            sig = _snapshot_digest(entries)
            if self._processed_stacks.get(stack_id) == sig or stack_id in self._in_flight_ids:
                self._pending_stability.pop(stack_id, None)
                continue
            if writing:
                self._pending_stability[stack_id] = (sig, clock)
                continue
            if self._is_stabilized(stack_id, sig, clock, cfg.settle_seconds):
                candidate = StackCandidate(
                    stack_id=stack_id,
                    name=name,
                    source_dir=sub_path,
                    file_paths=tuple(item[1] for item in entries),
                    signature=sig,
                    is_root_batch=False,
                )
                self._enqueue_candidate(candidate)
                enqueued.append(candidate)

        # 2. Inspect loose incoming images directly inside watch_dir.
        root_entries, root_writing = _list_stack_files(
            cfg.watch_dir,
            excluded_paths=frozenset(self._processed_root_files),
        )
        root_id = "root:incoming"
        if len(root_entries) >= cfg.min_frames:
            active_ids.add(root_id)
            sig = _snapshot_digest(root_entries)
            if root_id not in self._in_flight_ids and self._processed_stacks.get(root_id) != sig:
                if root_writing:
                    self._pending_stability[root_id] = (sig, clock)
                elif self._is_stabilized(root_id, sig, clock, cfg.settle_seconds):
                    self._root_batch_index += 1
                    root_base = os.path.basename(cfg.watch_dir) or "WatchStack"
                    batch_name = f"{root_base}_batch_{self._root_batch_index:03d}"
                    candidate = StackCandidate(
                        stack_id=f"root:{sig[:12]}",
                        name=batch_name,
                        source_dir=cfg.watch_dir,
                        file_paths=tuple(item[1] for item in root_entries),
                        signature=sig,
                        is_root_batch=True,
                    )
                    # Record root files as claimed immediately so the next tick
                    # does not double-queue the exact same loose files.
                    for item in root_entries:
                        self._processed_root_files.add(
                            os.path.normcase(os.path.realpath(item[1]))
                        )
                    self._pending_stability.pop(root_id, None)
                    self._enqueue_candidate(candidate)
                    enqueued.append(candidate)
        else:
            self._pending_stability.pop(root_id, None)

        # Prune vanished folders from stability map.
        for stale_id in list(self._pending_stability):
            if stale_id not in active_ids:
                self._pending_stability.pop(stale_id, None)

        self._sync_watcher_dirs(subdirs_to_watch)
        if enqueued:
            self._drain_queue()
        return enqueued

    def _is_stabilized(
        self, stack_id: str, signature: str, now: float, settle_seconds: float
    ) -> bool:
        prev = self._pending_stability.get(stack_id)
        if prev is None or prev[0] != signature:
            self._pending_stability[stack_id] = (signature, now)
            return settle_seconds <= 0.0
        first_seen = prev[1]
        return (now - first_seen) >= settle_seconds

    def _enqueue_candidate(self, candidate: StackCandidate) -> None:
        self._pending_stability.pop(candidate.stack_id, None)
        self._in_flight_ids.add(candidate.stack_id)
        self._queue.append(candidate)
        self.stack_detected.emit(candidate.name, len(candidate.file_paths))
        self.log_message.emit(
            "info",
            f"Detected stack '{candidate.name}' ({len(candidate.file_paths)} frames)",
        )

    def _drain_queue(self) -> None:
        if not self._running or self._paused or self._active_worker is not None:
            return
        if not self._queue or self._config is None:
            if self._running and not self._paused:
                self._set_status("watching")
            return
        candidate = self._queue.popleft()
        self._set_status("processing")
        worker = _WatchRenderWorker(candidate, self._config, self._render_fn, parent=self)
        worker.completed.connect(self._on_worker_completed)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self._active_worker = worker
        worker.start()

    def _on_worker_completed(
        self, candidate: StackCandidate, output_path: str, elapsed: float
    ) -> None:
        self._in_flight_ids.discard(candidate.stack_id)
        self._processed_stacks[candidate.stack_id] = candidate.signature
        self._processed_count += 1
        self._save_ledger()
        self.stack_completed.emit(candidate.name, output_path, float(elapsed))
        self.log_message.emit(
            "success",
            f"Completed '{candidate.name}' -> {output_path} ({elapsed:.1f}s)",
        )
        if self._config is not None and self._config.play_sound:
            from utils.sound_notify import play_completion_sound
            play_completion_sound(True)

    def _on_worker_failed(self, candidate: StackCandidate, error_message: str) -> None:
        self._in_flight_ids.discard(candidate.stack_id)
        if error_message == "CANCELLED":
            if candidate.is_root_batch:
                for p in candidate.file_paths:
                    self._processed_root_files.discard(os.path.normcase(os.path.realpath(p)))
            return
        # Mark signature as seen so a broken folder does not spin infinitely.
        self._processed_stacks[candidate.stack_id] = candidate.signature
        self._save_ledger()
        self.stack_failed.emit(candidate.name, error_message)
        self.log_message.emit(
            "error",
            f"Failed '{candidate.name}': {error_message}",
        )

    def _on_worker_finished(self, worker: _WatchRenderWorker) -> None:
        if self._active_worker is worker:
            self._active_worker = None
        worker.deleteLater()
        if self._running and not self._paused:
            self._drain_queue()
        elif not self._running:
            self._set_status("idle")
