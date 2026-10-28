"""Closed-loop agent smoke test for the live OpenFocus GUI.

Runs against an already-running window (`--control`) or launches one with
`--launch`. It exercises the same commands an agent uses: stack load, render,
export, depth/compare/retouch dialogs, canvas gestures, a menu action with a
file picker, project round-trip and batch. Exit code 0 means every assertion
passed.

    python tools/agent_gui_smoke.py [--launch] [--quit] [--keep]

This is the executable version of the workflow in
`.claude/skills/openfocus-control/SKILL.md`.
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
    ControlError, Session, launch_gui, tool_session, wait_for,
)


class Smoke:
    def __init__(self, timeout: float, session: Session):
        self.timeout = timeout
        self.session = session
        self.checks: list[dict] = []
        self.work = Path(tempfile.mkdtemp(prefix="openfocus_agent_"))
        self.demo = ROOT / "assets" / "demo_stack"

    def call(self, command, **args):
        return self.session.call(command, args, timeout=self.timeout)

    def check(self, name, function):
        started = time.monotonic()
        try:
            detail = function()
            self.checks.append({"check": name, "ok": True,
                                "seconds": round(time.monotonic() - started, 2),
                                "detail": detail})
            print(f"PASS {name}: {detail or ''}")
            return detail
        except Exception as error:  # noqa: BLE001 - reported as a check
            self.checks.append({"check": name, "ok": False,
                                "seconds": round(time.monotonic() - started, 2),
                                "error": f"{type(error).__name__}: {error}"})
            print(f"FAIL {name}: {type(error).__name__}: {error}")
            raise

    # --------------------------------------------------------------- helpers
    def windows(self):
        return self.call("dialog.list")["dialogs"]

    def wait_windows(self, predicate, timeout=15.0):
        deadline = time.monotonic() + timeout
        last = []
        while time.monotonic() < deadline:
            last = self.windows()
            if predicate(last):
                return last
            time.sleep(0.15)
        raise TimeoutError(f"windows: {last}")

    def click_text(self, root, labels):
        for label in labels:
            for button in self.call("ui.find", window=root,
                                    **{"class": "QPushButton"})["matches"]:
                text = (button["text"] or "").strip().replace("&", "").casefold()
                if text == label.casefold() and button["enabled"]:
                    self.call("ui.click", path=button["path"])
                    return button["text"]
        raise AssertionError(f"no button {labels} in {root}")

    def close_all(self):
        for _ in range(12):
            windows = self.windows()
            if not windows:
                return
            window = sorted(windows, key=lambda item: (not item["active"],
                                                       item["index"]))[0]
            if window["class"] == "QFileDialog":
                self.call("dialog.file", index=window["index"], cancel=True)
            elif window["class"] == "QMessageBox":
                labels = ("Start Fresh", "Discard changes", "No", "Discard",
                          "OK", "Close", "Cancel")
                try:
                    self.click_text(window["root"], labels)
                except AssertionError:
                    self.call("dialog.reject", index=window["index"])
            elif window["class"] == "QColorDialog":
                self.call("dialog.color", index=window["index"], color="#336699")
            else:
                try:
                    self.call("dialog.reject", index=window["index"])
                except ControlError:
                    self.call("dialog.close", index=window["index"])
            time.sleep(0.1)

    # ---------------------------------------------------------------- checks
    def startup(self):
        info = self.call("app.info")
        assert info["version"]
        self.close_all()
        context = self.call("ui.context")
        assert context["level"] == "main", context["level"]
        assert context["roots"] == ["main"], context["roots"]
        return f"v{info['version']} commands={len(info['control']['commands'])}"

    def load(self):
        self.call("stack.load", folder=str(self.demo))
        frames = wait_for("stack.frames",
                          lambda value: value.get("count") == 6,
                          timeout=120)
        assert frames["dtype"] == "uint8"
        return f"{frames['count']} frames {frames['shape']}"

    def configure(self):
        method = self.call("method.set", id="dct")
        assert method["id"] == "dct"
        self.call("option.set", name="thread_count", value=4)
        self.call("option.set", name="preview", value=False)
        self.call("align.set", id="none")
        return "dct, cpu, 4 threads"

    def render(self):
        state = self.call("render.start")
        assert state["started"]
        final = wait_for("render.state",
                         lambda value: not value.get("busy")
                         and value.get("state") in ("done", "error"),
                         timeout=300)
        assert final["state"] == "done", final
        result = self.call("render.result")
        assert result["shape"][0] > 0
        return f"{result['shape']} {result['dtype']}"

    def export(self):
        target = self.work / "result.png"
        self.call("output.export", path=str(target), index=0)
        assert target.is_file() and target.stat().st_size > 1000
        return f"{target.stat().st_size} bytes"

    def depth(self):
        self.call("dialog.open", name="depth")
        window = self.wait_windows(
            lambda items: any(item["class"] == "DepthMapDialog"
                              for item in items))[0]
        button = self.call("ui.find", window=window["root"],
                           object_name="compute_btn")["matches"][0]
        self.call("ui.click", path=button["path"])
        self.wait_windows(lambda items: not items[0]["busy"], timeout=300)
        shot = self.work / "depth.png"
        self.call("ui.grab", path=window["root"], output=str(shot))
        assert shot.stat().st_size > 1000
        self.call("dialog.accept")
        return "computed + screenshot"

    def compare(self):
        self.call("dialog.open", name="compare")
        window = self.wait_windows(
            lambda items: any(item["class"] == "CompareDialog"
                              for item in items))[0]
        for control in ("method_dct", "method_openfocus_a"):
            match = self.call("ui.find", window=window["root"],
                              object_name=control)["matches"][0]
            self.call("ui.set", path=match["path"], value=True)
        mode = self.call("ui.find", window=window["root"],
                         object_name="mode_combo")["matches"][0]
        self.call("ui.set", path=mode["path"], value=False)
        run = self.call("ui.find", window=window["root"],
                        object_name="btn_run")["matches"][0]
        self.call("ui.click", path=run["path"])
        self.wait_windows(lambda items: not items[0]["busy"], timeout=300)
        self.call("dialog.reject")
        return "dct vs openfocus_a"

    def retouch(self):
        self.call("dialog.open", name="retouch")
        window = self.wait_windows(
            lambda items: any(item["class"] == "RetouchDialog"
                              for item in items))[0]
        canvas = self.call("ui.find", window=window["root"],
                           object_name="retouch_result")["matches"][0]
        geometry = canvas["geometry"]
        self.call("ui.mouse", path=canvas["path"], action="drag",
                  x=int(geometry["width"] * 0.3),
                  y=int(geometry["height"] * 0.3),
                  to_x=int(geometry["width"] * 0.55),
                  to_y=int(geometry["height"] * 0.55), steps=16)
        undo = self.call("ui.find", window=window["root"],
                         object_name="retouch_undo")["matches"][0]
        assert self.call("ui.click", path=undo["path"])["clicked"]
        self.call("ui.mouse", path=canvas["path"], action="drag",
                  x=int(geometry["width"] * 0.3),
                  y=int(geometry["height"] * 0.3),
                  to_x=int(geometry["width"] * 0.55),
                  to_y=int(geometry["height"] * 0.55), steps=16)
        before = self.call("output.list")["count"]
        apply_button = self.call("ui.find", window=window["root"],
                                 object_name="retouch_apply")["matches"][0]
        self.call("ui.click", path=apply_button["path"])
        self.wait_windows(lambda items: not items, timeout=20)
        after = self.call("output.list")["count"]
        assert after > before, (before, after)
        return f"brush + undo + apply ({before}->{after} outputs)"

    def gestures(self):
        wipe = self.call("ui.find", window="main",
                         object_name="btn_wipe")["matches"][0]
        if self.call("app.state")["view"]["wipe_active"]:
            self.call("ui.click", path=wipe["path"])
        self.call("ui.click", path=wipe["path"])
        time.sleep(0.2)
        view = self.call("app.state")["view"]
        assert view["wipe_active"] and view["result_page"] == 1, view
        canvas = self.call("ui.find", window="main",
                            object_name="wipe_compare")["matches"][0]
        geometry = canvas["geometry"]
        self.call("ui.mouse", path=canvas["path"], action="drag",
                  x=geometry["width"] // 4, y=geometry["height"] // 2,
                  to_x=geometry["width"] * 3 // 4,
                  to_y=geometry["height"] // 2, steps=10)
        self.call("ui.mouse", path=canvas["path"], action="wheel",
                  x=geometry["width"] // 2, y=geometry["height"] // 2,
                  delta=120)
        self.call("ui.click", path=wipe["path"])
        return "wipe drag + wheel"

    def file_action(self):
        self.call("ui.trigger", key="action_open_folder")
        windows = self.wait_windows(
            lambda items: any(item["class"] == "QFileDialog"
                              for item in items))
        window = next(item for item in windows
                      if item["class"] == "QFileDialog")
        self.call("dialog.file", index=window["index"], path=str(self.demo))
        frames = wait_for("stack.frames",
                          lambda value: value.get("count") == 6, timeout=120)
        return f"picked folder -> {frames['count']} frames"

    def labels(self):
        self.call("ui.trigger", key="action_add_label")
        window = self.wait_windows(
            lambda items: any(item["title"] == "Manage Labels"
                              for item in items))[0]
        edits = self.call("ui.find", window=window["root"],
                          **{"class": "QLineEdit"})["matches"]
        self.call("ui.set", path=edits[0]["path"], value="AGENT")
        self.click_text(window["root"], ["Add"])
        color_button = self.call("ui.find", window=window["root"],
                                 text="Choose Background Color")["matches"][0]
        self.call("ui.click", **{"path": color_button["path"], "async": True})
        color_window = next(
            item for item in self.wait_windows(
                lambda items: any(item["class"] == "QColorDialog"
                                  for item in items))
            if item["class"] == "QColorDialog")
        self.call("dialog.color", index=color_window["index"], color="#336699")
        self.click_text(window["root"], ["Apply and Close"])
        actions = {item["key"]: item["enabled"]
                   for item in self.call("ui.actions")["actions"]}
        assert actions["action_del_input_label"]
        self.call("ui.trigger", key="action_del_input_label")
        return "added, colored, applied, removed"

    def project(self):
        path = self.work / "roundtrip.ofp"
        self.call("project.save", path=str(path))
        self.call("stack.clear")
        assert self.call("stack.frames")["count"] == 0
        self.call("project.open", path=str(path))
        frames = wait_for("stack.frames",
                          lambda value: value.get("count") == 6, timeout=120)
        return f"{frames['count']} frames restored"

    def batch(self):
        outdir = self.work / "batch"
        self.call("dialog.open", name="batch", folders=[str(self.demo)],
                  output_dir=str(outdir))
        window = self.wait_windows(
            lambda items: any(item["class"] == "BatchProcessingDialog"
                              for item in items))[0]
        self.wait_windows(lambda items: not items[0]["busy"], timeout=180)
        combo = self.call("ui.find", window=window["root"],
                          object_name="fusion_combo")["matches"][0]
        self.call("ui.set", path=combo["path"], value="dct")
        self.call("dialog.accept")
        state = wait_for("batch.state",
                         lambda value: not value.get("busy"), timeout=600)
        assert state["error"] is None, state["error"]
        assert state["results"]["success"] >= 1, state["results"]
        files = list(outdir.rglob("*.png"))
        assert files, outdir
        return f"{len(files)} file(s)"

    def screenshot(self):
        target = self.work / "window.png"
        self.call("ui.grab", output=str(target))
        assert target.is_file() and target.stat().st_size > 1000
        return f"{target.stat().st_size} bytes"

    # ------------------------------------------------------------------ run
    def run(self):
        self.check("startup/context", self.startup)
        self.check("stack load", self.load)
        self.check("configure", self.configure)
        self.check("render", self.render)
        self.check("export", self.export)
        self.check("depth dialog", self.depth)
        self.check("compare dialog", self.compare)
        self.check("retouch brush", self.retouch)
        self.check("wipe gestures", self.gestures)
        self.check("menu action + file picker", self.file_action)
        self.check("labels + color dialog", self.labels)
        self.check("project round-trip", self.project)
        self.check("batch dialog", self.batch)
        self.check("screenshot", self.screenshot)
        return self.checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true",
                        help="start a private GUI (isolated discovery file "
                             "and socket; never touches a live session)")
    parser.add_argument("--allow-existing", action="store_true",
                        help="attach to an already-running session; the "
                             "smoke clears stacks and answers prompts, so "
                             "this is destructive")
    parser.add_argument("--quit", action="store_true",
                        help="quit the GUI when the run finishes")
    parser.add_argument("--timeout", type=float, default=600,
                        help="per-command timeout in seconds")
    options = parser.parse_args()

    session = None
    smoke = None
    try:
        session = tool_session(
            timeout=options.timeout,
            launch=options.launch,
            allow_existing=options.allow_existing,
        )
        smoke = Smoke(options.timeout, session)
        smoke.call("ping")
        checks = smoke.run()
    except Exception as error:  # noqa: BLE001 - command-line report
        print(f"SMOKE FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if options.quit and session is not None:
            try:
                session.call("app.quit", {"force": True}, timeout=10)
            except Exception:
                pass
        if session is not None:
            session.close()
    print(json.dumps({"ok": all(check["ok"] for check in checks),
                      "checks": checks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
