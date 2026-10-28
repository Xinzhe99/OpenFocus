"""Drive every enabled menu action in the live GUI and close what it opens.

Complements ``agent_gui_smoke.py``: the smoke covers the end-to-end pipeline,
this sweep proves that every action in ``ui.actions`` remains callable and
that its dialog/file picker can be resolved through the generic UI layer.

    python tools/agent_action_sweep.py [--launch] [--timeout 120]

Exit code 0 means no action timed out or left an unresolved window. Actions
that are disabled in the current state, or that would change the installation
(exit, update, open logs, GPU), are reported as skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.control.client import (  # noqa: E402
    ControlError, Session, launch_gui, tool_session,
)
from locales import trans  # noqa: E402


def _texts(key, *fallbacks):
    """Every loaded translation of *key*, plus explicit fallbacks."""
    values = [trans.t(key)]
    for table in getattr(trans, "translations", {}).values():
        value = table.get(key)
        if value:
            values.append(value)
    values.extend(fallbacks)
    return list(dict.fromkeys(values))

SKIP = {"action_exit", "action_install_update", "menu_check_updates",
        "menu_open_logs", "action_gpu_accel"}
LANGUAGE_SKIP = {f"action_lang_{code}" for code in
                 ("de", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "ru", "zh")}


class Sweep:
    def __init__(self, timeout: float, session: Session):
        self.timeout = timeout
        self.session = session
        self.work = Path(tempfile.mkdtemp(prefix="openfocus_sweep_"))
        self.demo = ROOT / "assets" / "demo_stack"
        self.video = self._make_video()
        self.project = self.work / "source.ofp"
        self.save_dir = self.work / "actions"
        self.save_dir.mkdir(exist_ok=True)

    def call(self, command, **args):
        return self.session.call(command, args, timeout=self.timeout)

    def windows(self):
        return self.call("dialog.list")["dialogs"]

    def click_text(self, root, *labels):
        for label in labels:
            for button in self.call("ui.find", window=root,
                                    **{"class": "QPushButton"})["matches"]:
                text = (button["text"] or "").strip().replace("&", "").casefold()
                if text == label.casefold() and button["enabled"]:
                    self.call("ui.click", path=button["path"])
                    return button["text"]
        return None

    def dismiss(self, window, startup=False):
        klass, root = window["class"], window["root"]
        if klass == "QFileDialog":
            self.call("dialog.file", index=window["index"], cancel=True)
        elif klass == "QColorDialog":
            self.call("dialog.color", index=window["index"], color="#336699")
        elif klass == "QMenu":
            self.call("dialog.close", index=window["index"])
        elif klass == "QMessageBox":
            labels = tuple(_texts("btn_start_fresh", "Start Fresh")
                           + _texts("btn_discard_changes", "Discard changes")
                           + ["No", "否", "Discard", "OK", "Close", "Cancel"])
            if startup:
                labels = (_texts("btn_start_fresh", "Start Fresh")[0],) + labels
            if not self.click_text(root, *labels):
                self.call("dialog.accept", index=window["index"])
        else:
            try:
                self.call("dialog.reject", index=window["index"])
            except ControlError:
                self.call("dialog.close", index=window["index"])

    def clean_all(self):
        for _ in range(12):
            open_windows = self.windows()
            if not open_windows:
                return
            for window in open_windows:
                try:
                    self.dismiss(window, startup=True)
                except ControlError:
                    pass
            time.sleep(0.1)

    def answer_file(self, window, key):
        index = window["index"]
        if key == "action_open_folder":
            path = self.demo
        elif key == "action_open_video":
            path = self.video
        elif key == "menu_open_project":
            path = self.project
        elif key in ("action_save_folder_input", "action_save_folder_reg"):
            path = self.save_dir / f"{key}_dir"
            path.mkdir(exist_ok=True)
        elif "project" in key:
            path = self.save_dir / f"{key}.ofp"
        else:
            suffix = ".gif" if "gif" in key else ".tif" if "mtiff" in key else ".png"
            path = self.save_dir / f"{key}{suffix}"
        self.call("dialog.file", index=index, path=str(path))

    def settle(self, key, seen):
        empty_rounds = 0
        for _ in range(20):
            open_windows = self.windows()
            if not open_windows:
                empty_rounds += 1
                if empty_rounds >= 2:
                    return []
                time.sleep(0.25)
                continue
            empty_rounds = 0
            open_windows.sort(key=lambda item: (not item["active"], item["index"]))
            window = open_windows[0]
            seen.append(f"{window['class']}:{window['title']}")
            try:
                if window["class"] == "QFileDialog":
                    self.answer_file(window, key)
                else:
                    self.dismiss(window)
            except ControlError:
                # The window may have closed itself between list and action.
                pass
            time.sleep(0.1)
        return ["UNRESOLVED"]

    def _make_video(self):
        import cv2
        import numpy as np
        path = self.work / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 10, (64, 48))
        for index in range(5):
            writer.write(np.full((48, 64, 3), index * 40, np.uint8))
        writer.release()
        return path

    def ensure_ready(self):
        frames = self.call("stack.frames")
        if frames["count"] == 0:
            self.call("stack.load", folder=str(self.demo))
            self.wait(lambda: self.call("stack.frames"),
                      lambda value: value["count"] == 6, 60, "frames")
        if self.call("output.list")["count"] == 0:
            self.call("method.set", id="dct")
            self.call("render.start")
            self.wait(lambda: self.call("render.state"),
                      lambda value: not value.get("busy")
                      and value.get("state") in ("done", "error"), 300, "render")

    @staticmethod
    def wait(producer, predicate, timeout, what):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = producer()
            if predicate(last):
                return last
            time.sleep(0.15)
        raise TimeoutError(f"{what}: {last}")

    def run(self):
        self.clean_all()
        self.ensure_ready()
        self.call("project.save", path=str(self.project))
        self.clean_all()

        results = []
        for action in self.call("ui.actions")["actions"]:
            key = action["key"]
            if key in SKIP or key in LANGUAGE_SKIP:
                results.append({"key": key, "status": "skip", "reason": "unsafe"})
                continue
            if action.get("source") == "menu":
                results.append({"key": key, "status": "skip",
                                "reason": "dynamic recent menu"})
                continue
            if not action["enabled"]:
                results.append({"key": key, "status": "skip",
                                "reason": "disabled"})
                continue
            seen = []
            try:
                self.ensure_ready()
                self.clean_all()
                current = next(
                    (item for item in self.call("ui.actions")["actions"]
                     if item["key"] == key), None)
                if current is None or not current["enabled"]:
                    results.append({"key": key, "status": "skip",
                                    "reason": "disabled after setup"})
                    continue
                self.call("ui.trigger", key=key)
                time.sleep(0.2)
                unresolved = self.settle(key, seen)
                if unresolved:
                    raise ControlError("internal", f"unresolved: {seen}")
                status = "pass"
                error = None
            except Exception as exc:  # noqa: BLE001 - reported per action
                status = "fail"
                error = f"{type(exc).__name__}: {exc}"
                try:
                    self.clean_all()
                except Exception:
                    pass
            results.append({"key": key, "status": status,
                            "windows": seen, "error": error})
            print(f"{status:4} {key:34} {seen} {error or ''}")
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true",
                        help="start a private GUI (isolated discovery file "
                             "and socket; never touches a live session)")
    parser.add_argument("--allow-existing", action="store_true",
                        help="attach to an already-running session; the "
                             "sweep triggers every action destructively")
    parser.add_argument("--quit", action="store_true",
                        help="quit the GUI when the run finishes")
    parser.add_argument("--timeout", type=float, default=120)
    options = parser.parse_args()

    session = None
    try:
        session = tool_session(
            timeout=options.timeout,
            launch=options.launch,
            allow_existing=options.allow_existing,
        )
        results = Sweep(options.timeout, session).run()
    except Exception as error:  # noqa: BLE001 - command-line report
        print(f"SWEEP FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if options.quit and session is not None:
            try:
                session.call("app.quit", {"force": True}, timeout=10)
            except Exception:
                pass
        if session is not None:
            session.close()

    failures = [item for item in results if item["status"] == "fail"]
    skipped = [item for item in results if item["status"] == "skip"]
    print(json.dumps({
        "ok": not failures,
        "passed": sum(1 for item in results if item["status"] == "pass"),
        "skipped": len(skipped),
        "failed": failures,
        "results": results,
    }, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
