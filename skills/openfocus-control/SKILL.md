---
name: openfocus-control
description: Operate the live OpenFocus desktop GUI as a user would. Use for loading focus stacks, fusing, comparing methods, retouching, batch jobs, exports, screenshots and any menu action, including dialogs and file pickers. Prefer this over guessing CLI flags when the task needs the running application.
allowed-tools: Read, Grep, Glob, Bash(openfocus *), Bash(python main.py *), Bash(python -c *), Bash(python tools/agent_gui_smoke.py *), Bash(python tools/agent_action_sweep.py *)
---

# OpenFocus GUI control

You drive a real `OpenFocus` window through its local control channel. Every
command runs on the Qt GUI thread and calls the same controllers, dialogs and
buttons a person uses. You can therefore inspect real state, fill dialogs,
paint with the retouch brush, answer file pickers and take screenshots.

## 1. Start and connect

```bash
openfocus gui                     # launch the desktop build / checkout
openfocus ctl ping --pretty       # verify the window
openfocus ctl ping --launch       # launch only if no window is running
```

From a checkout use `python main.py` and `python main.py ctl <command>`.
The control server starts automatically with the desktop GUI.

The PyPI wheel is headless and cannot launch the desktop app. To use its
`openfocus ctl` against a separately running desktop window, first install
the transport dependency with `pip install PyQt6`.

Never connect to a window you did not start unless the user asked for it. The
token in `<user-data>/control.json` grants full GUI input. Keep one agent per
window; parallel mutating commands race with each other.

## 2. Know where you are

Call `ui.context` before acting and after every dialog transition:

```json
{
  "level": "dialog",
  "active": {"root": "dialog[0]", "path": "dialog[0]/...", "title": "Depth Map", "class": "DepthMapDialog"},
  "focus": {"path": "dialog[0]/.../compute_btn", "control_id": "compute_btn"},
  "roots": ["main", "dialog[0]"],
  "dialogs": [{"root": "dialog[0]", "class": "DepthMapDialog", "active": true, "busy": false}],
  "view": {"source_zoom": 1.0, "result_page": 0, "wipe_active": false},
  "stack": {"count": 6, "included": [0,1,2,3,4,5]},
  "render": {"state": "done", "busy": false}
}
```

- `level`: `main`, `dialog`, `popup` or `modal`.
- `active.root`: where input would go now.
- `focus.path`: exact widget that has keyboard focus.
- `dialog[0]`, `dialog[1]`: every visible top-level dialog, **including**
  file pickers, message boxes and Qt input dialogs.
- `popup[0]`: an open menu / combo popup. Use `ui.menu` for its actions.
- `dialog.list` adds `managed`, `popup`, `active` and `busy`.

`app.state` is the same picture plus method/options/outputs; `app.info`
describes methods, capabilities and the exact command list.

## 3. Address widgets

```bash
openfocus ctl ui.tree --window main --depth 3 --pretty
openfocus ctl ui.find --window main --object-name combo_fusion
openfocus ctl ui.find --window dialog[0] --text "Compute"
openfocus ctl ui.find --window main --class QPushButton
```

Copy a returned `path` verbatim. Paths are hierarchical:

```
main/<controlId | objectName | Class[n]>/...
dialog[0]/<controlId | objectName | Class[n]>/...
popup[0]/...
```

- Duplicate sibling names get an occurrence suffix: the source and result
  panels are `panelRoot[0]` and `panelRoot[1]`.
- Prefer `controlId`/`objectName`; `class` and `text` are language-dependent.
- Nodes include `geometry {x,y,width,height}` (parent-relative),
  `global_origin`, `visible`, `enabled`, `focus`, `value`, `checked`.
- If a path stops resolving, the widget tree changed: call `ui.find` again.
  Do not guess a path or a sibling index.

## 4. Operate like a user

```bash
# semantic actions
openfocus ctl ui.click --path 'main/.../btn_render'
openfocus ctl ui.set --path 'main/.../combo_fusion' --value openfocus_b
openfocus ctl ui.set --path 'main/.../chk_quick_preview' --value true
openfocus ctl ui.focus --path 'main/.../view_source'

# real mouse events, coordinates relative to the widget
openfocus ctl ui.mouse --path 'main/.../view_source' --action click --x 120 --y 80
openfocus ctl ui.mouse --path 'main/.../view_source' --action drag \
  --x 100 --y 100 --to-x 180 --to-y 150 --steps 12
openfocus ctl ui.mouse --path 'main/.../view_source' --action wheel --delta 120
openfocus ctl ui.mouse --path 'main/.../view_result' --action right_click --x 40 --y 40

# keyboard and text
openfocus ctl ui.key --key Escape
openfocus ctl ui.key --key 'Ctrl+S'
openfocus ctl ui.type --path 'dialog[0]/.../fileNameEdit' --text 'F:/out.tif' --submit
```

`ui.mouse` actions: `click`, `double_click`, `right_click`, `middle_click`,
`press`, `release`, `move`, `drag`, `wheel`. Add `--async` to `ui.click` when
the slot opens a modal dialog; otherwise that command waits inside the
dialog's event loop. After an async click, poll `ui.context`/`dialog.list`.
A `right_click` posts the context-menu event and returns before the menu
opens: poll `ui.context` for `popup[N]`, then use `ui.menu`. `ui.key` actions: `click`,
`press`, `release`. `ui.type` appends `--submit` to press Enter afterwards.
These are the tools for zooming/panning canvases, ROI drawing, wipe handles,
retouch strokes, list selection and 3D/depth surfaces.
Text supplied to `ui.type` or `ui.key` turns LF, CRLF and lone CR into one
Return each; Tab retains native behavior. Other C0 controls and DEL reject the
entire request with `invalid_argument` before focus or visibility changes.
Only `ui.type --submit` sends an additional Return after the text.

`ui.set` supports combo boxes (by data or text), checkboxes/radios,
spin boxes, sliders, line edits, text areas, tabs, stacked pages and lists.
For a tab widget, `value` is the active index. Its `tabs` array lists the
localized labels and whether each tab is visible and enabled. Pass the desired
zero-based index to `ui.set`, then inspect the controls on the selected page.

## 5. Core workflows

This workflow works with a fresh default configuration (OpenFocus B and ECC):

```bash
# load -> configure -> render -> inspect -> export
openfocus ctl stack.load --folder 'F:/stacks/session'   # add --linear-raw for camera-linear RAW
# poll stack.frames until load.finished for the returned load_id
openfocus ctl method.set --id openfocus_b
openfocus ctl align.set --id ecc
openfocus ctl option.set --name resource_preference --value balanced
openfocus ctl option.set --name sharpen_output --value true        # sharpen_strength: 0..200 (default 100)
openfocus ctl render.start --wait --timeout 300
openfocus ctl render.state
openfocus ctl output.list
openfocus ctl output.export --path 'F:/out/result.tif' --index 0
```

Open Advanced Settings with `openfocus ctl dialog.open --name settings`.
Find `advanced_settings_tabs` with `ui.find --control-id advanced_settings_tabs`
in the settings dialog, then use `ui.set` on the returned path. The indices are
stable across languages: 0 General, 1 Fusion, 2 Registration, 3 Startup & updates.
Registration initially shows the downscale width. To edit the OpenFocus
reference or search limits, find `registration_options_disclosure` on page 2.
If `checked` is false, use `ui.click` on its path or send `ui.key --key Space`
to that path. Inspect the revealed controls with `ui.find`.

On Fusion, `effective_tiling_status` explains the current method. With tiling
enabled, you can save block size and overlap for a later method while OpenFocus
A/B/C uses the full image. DT-CWT determines its need for tiles from memory
and disables the size threshold field.

DCT, AI and other experimental desktop methods require a persistent opt-in.
On General, find `chk_experimental_features`, set it checked and accept the
dialog. Confirm the save before selecting the method. AI also needs its runtime
and bundled model.

To profile the next render through the control channel, use the typed options
below. Detailed logging also records plans, tile sizes and directories.

```bash
openfocus ctl option.set --name profile_performance --value true
openfocus ctl option.set --name debug_logging --value true     # optional, DEBUG log detail
openfocus ctl render.start --wait --timeout 300
openfocus ctl render.result     # .performance: stages, tiling, threads, cache
```

`render.start --wait` exits when the returned `render_id` reaches `done`,
`error`, `cancelled` or `discarded`, or when a newer render supersedes it;
only completion of the requested render is success. Without it poll
`render.state` until `busy` is false and compare `render_id`. `output.export`
bypasses the file picker and writes through the normal export path (labels,
scale bar, bit depth).

Projects: `project.save --path …` (no path: saves the open project), then
`project.open --path …` (add `--wait` to wait for frames). `project.open`
refuses with `confirm_required` while the current project has unsaved
changes; pass `force=true` only when discarding them is intended. Both
commands keep `project_path` and the dirty marker in sync with the GUI.

Batch: `dialog.open --name batch
--args-json '{"folders":["F:/a","F:/b"],"output_dir":"F:/out"}'`, set
`fusion_combo`, then `dialog.accept`; poll `batch.state` until `busy` is
false and inspect `results`/`error`. `batch.cancel` requests cancellation.
For grouped processing, set `chk_grouping` to true before setting
`group_size_spin` and `group_overlap_spin`. Grouped fusion is approximate,
uses temporary disk space, and does not produce original-frame depth maps.

For desktop sensor defect repair and denoising, select Fusion (page 1) and
expand `sensor_cleanup_disclosure` using the same click or Space action.
Batch jobs have their own sensor controls in the batch dialog. Let mask
inspection finish before accepting. For an alignment report from a single, non-grouped CLI run,
select an alignment method and pass `--alignment-report FILE`. In the desktop
app, **Help → Open Logs Folder** opens the session logs.

For a layered TIFF, first check that the result has source-frame provenance and
no active watermark. Right-click the desired output, list the context menu via
`ui.menu`, and select its returned Save As action. Wait for the file picker;
discover `fileTypeCombo` using `ui.find` or `ui.tree`, select the returned
localized layered TIFF filter with `ui.set`, and submit the path with
`dialog.file`. Ordinary TIFF shares the extension, so choosing the filter is
essential. Picker acceptance does not prove export success: wait for export
progress to finish or an explicit error. Cancellation before commit preserves
the existing target and unsaved protection; encoding may finish before the
cancel check, and cancellation cannot undo an already committed file.
`output.export` writes the ordinary flattened output. In the retouch dialog,
`retouch_save_checkpoint` and `retouch_load_checkpoint` use file pickers and
may run asynchronously. To replay headlessly, use `openfocus retouch replay
CHECKPOINT --base BASE --sources FRAME1 FRAME2 ... --output RESULT.tif` with
the original fused base and processed frames in checkpoint order. GUI depth
exports include color, raw depth, source indices and other products; CLI
`--depth-map` exports a colorized focus-position map only for supported methods.

## 6. Dialogs, menus and file pickers

Managed dialogs (`dialog.open`) are modeless, so commands keep flowing:

```bash
openfocus ctl dialog.open --name depth
openfocus ctl ui.find --window dialog[0] --control-id compute_btn   # object-name works too
openfocus ctl ui.click --path 'dialog[0]/.../compute_btn'
# poll dialog.list until busy is false
openfocus ctl ui.grab --path 'dialog[0]' --output 'F:/tmp/depth.png'
openfocus ctl dialog.accept
```

Dialog names: `settings`, `batch`, `compare`, `depth`, `feedback`, `scalebar`,
`watermark`, `converter`, `retouch`, `welcome`, `method_help`, `registration_help`,
`environment`, `contact`. (`dust` and `mosaic` remain disabled in development and
return `invalid_argument`.)

File helper actions triggered through `ui.trigger` or supported control clicks
use addressable Qt pickers. Wait for `QFileDialog` in `dialog.list`, discover
its actual controls and answer it by path. The running control server alone
does not change ordinary desktop interactions, which use native helper pickers.
An enumerated outer `QFileDialog` does not guarantee its native internal UI is
addressable. Finish an already-open native interaction normally before reopening
it from a control action, or use an applicable typed command such as
`stack.load` or `output.export`; rediscover the current dialog and controls.

```bash
openfocus ctl ui.trigger --key action_open_folder
openfocus ctl dialog.list
openfocus ctl dialog.file --path 'F:/stacks/session'          # directory picker
openfocus ctl dialog.file --path 'F:/out/result.png'          # save/open file
openfocus ctl dialog.file --index 0 --path 'F:/out/result.png' # stacked pickers
openfocus ctl dialog.file --cancel
```

If several dialogs are stacked, pass `index` or `title` to `dialog.file`,
`dialog.accept` or `dialog.reject`. Message boxes and `QInputDialog` are
ordinary roots: find the button by text/class and click it. For color dialogs
use `dialog.color --color '#336699'`.

Use `ui.trigger` for menu actions: it schedules the action and returns before
it runs. Then poll `ui.context`/`dialog.list`. `ui.actions` lists every menu
action with `key`, `text`, `enabled`, `checked` and `shortcut`. Disabled
actions are not usable in that state; do not force them.

Context menus are `popup[N]`; list and trigger their entries with:

```bash
openfocus ctl ui.menu
openfocus ctl ui.menu --index 2
openfocus ctl ui.menu --action 'Copy'
```

## 7. Screenshots and proof

```bash
openfocus ctl ui.grab --output 'F:/tmp/window.png'
openfocus ctl ui.grab --path 'dialog[0]' --output 'F:/tmp/dialog.png'
```

`ui.grab` captures the Qt surface, not the OS desktop. A screenshot plus
`ui.context`/`app.state` is the normal proof for a UI action.

## 8. Async, busy and failure rules

- Long operations return immediately. Poll state; never assume completion
  from the acknowledgement.
- `busy` on a dialog means its worker (depth, compare, batch import, export)
  is still running. `dialog.accept` returns `busy` instead of closing it.
- Error responses are `{ok:false, error:{code,message,hint}}`; `not_running`
  means the window is gone, `unauthorized` means a stale token file.
- Do not call `dialog.reject` on retouch after painting unless you intend the
  discard confirmation; answer its message box button explicitly.
- Never kill the process to cancel work; use `render.cancel`/`batch.cancel`
  and poll the state.

## 9. Automation / CI

The same commands are an automation API. A test can start an offscreen or
visible window normally, then drive it through `openfocus ctl` or
`core.control.client.request`, asserting on returned JSON. Discovery and
tokens can be isolated with `OPENFOCUS_CONTROL_FILE` and
`OPENFOCUS_INSTANCE_NAME`. `OPENFOCUS_DATA_DIR` redirects application data and
settings to an explicit directory, including in portable mode. Set these
before starting the GUI. Prefer `output.export`, `project.save/open` and
`dialog.file` over OS automation.

## 10. Scope and safety

Control mode adds local GUI input only; it does not add privileges beyond the
user. `app.quit` quits the app (it may prompt for unexported/unsaved work; use
`app.quit --force` only when you intend to discard it);
`action_install_update` can restart it; update
checks and "open logs folder" touch the OS/network. Treat those as destructive
and ask the user unless the task explicitly requires them. Do not expose the
token or enable control on a shared account.

## 11. Verification tools

- `python tools/agent_gui_smoke.py --launch` runs the pipeline end to end
  (load, render, export, depth, compare, retouch apply, wipe, labels/color,
  project, grouped batch, file picker) and exits non-zero on failure.
- `python tools/agent_action_sweep.py --launch` triggers enabled fixed catalog
  actions and resolves their dialogs/pickers; disabled, updater, exit,
  non-English language switches and dynamic recent items are skipped. Context menus are
  outside its scope.
- Both tools are destructive: `--launch` starts a private instance with its
  own data and settings directory, discovery file, control socket and
  single-instance name. Attaching to an already-running window requires
  `--allow-existing`. Add `--quit` so an isolated instance exits after the run.
- The tools match dialog titles and buttons in every loaded UI language, so
  they work against a localized window.
- Python clients that issue many commands should reuse
  `core.control.client.Session` instead of opening a socket per request.
