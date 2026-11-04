# Agent control surface (GUI + CLI)

Status: implemented. Protocol id `openfocus/control/1`.

## 1. Goal

Let a local agent or script drive **the running OpenFocus GUI** the way a
person does: load a stack, choose a method, render, inspect outputs, open a
dialog, change its controls, export, compare, batch-process and save a
project. The headless CLI remains the path for CI; this channel adds the live
session.

Non-goals: network access, multi-user authentication, remote hosts, a second
fusion engine, or a plugin API. The channel is local IPC only.

## 2. Why an in-process command server

The GUI already owns every capability in its controllers and dialogs. Driving
the widgets through accessibility or synthetic mouse events is slow, depends
on screen layout, and cannot read state reliably. Instead the application
hosts a small command server on a local socket, executed on the GUI thread, so
commands call the same managers the buttons call. A generic UI layer covers
dialogs that have no dedicated command, which is what makes the surface
complete.

```
agent ── openfocus ctl … ──QLocalSocket──▶ ControlServer (GUI thread)
                                             ├─ typed commands → controllers
                                             └─ ui.* → widgets/actions
```

## 3. Lifecycle and discovery

The control channel starts automatically with the desktop application:

```bash
openfocus gui                           # detached
python main.py                          # from a checkout
```

`openfocus gui --control` also waits for discovery and prints connection info.

At startup the server writes `<user-data>/control.json`:

```json
{
  "schema": "openfocus/control/1",
  "version": "1.75.0",
  "pid": 12345,
  "socket": "OpenFocus-control-12345",
  "token": "…",
  "started_at": "2026-10-27T12:00:00"
}
```

`openfocus ctl` reads this file, connects to the named socket and sends the
token. A stale file (process gone) is reported as `not_running`; `--launch`
starts a new GUI first. The file is removed on a clean quit.

The socket is per-user local IPC (Windows named pipe, Unix-domain socket).
Requests without the token are rejected with `unauthorized`. The discovery
file is created with owner-only permissions from the first byte (a restricted
ACL on Windows, mode `0600` elsewhere). On Windows the server also compares
the named-pipe client's computer with `GetComputerNameW` and rejects a remote
or unidentifiable client; `ERROR_PIPE_LOCAL` is treated as local.

## 4. Framing

One request per line, one response per line, UTF-8, max 1 MiB per line.
Input is read in bounded chunks and queued replies are capped, so an oversize
line or a burst of pipelined requests cannot grow memory without limit; an
oversize response is returned as a small `internal` error. Clients match the
response `id` and discard a late reply after a timeout.

```json
{"id":"1","token":"…","cmd":"app.info","args":{}}
{"id":"1","ok":true,"result":{…},"elapsed_ms":2.1}
{"id":"1","ok":false,"error":{"code":"not_found","message":"…","hint":"…"}}
```

Responses always carry the request `id`. `ui.trigger` schedules the action and
returns before it runs, because menu actions may open a modal dialog.

Errors: `bad_request`, `unauthorized`, `not_found`, `invalid_argument`,
`busy`, `confirm_required`, `unsupported`, `internal`, `not_running`.

## 5. Commands

All commands run on the GUI thread. Long operations (render, batch, project
load, stack decode) return immediately; poll `app.state`/`render.state`/
`batch.state`, or use `--wait` in the client.

### Application

| Command | Args | Result |
|---|---|---|
| `ping` | — | `{pong:true, version}` |
| `app.info` | — | version, pid, platform, python, methods, registrations, capabilities, control |
| `app.state` | — | stack, method, alignment, options, render, outputs, dialogs, view, context |
| `app.quit` | `force?` | `{quitting:true, force}`; `force:true` discards prompts/work |

`app.info.methods` mirrors the registry (id, label, category, neural,
order_sensitive, depth_map, weights, parameters). `capabilities` reports torch,
gpu, heic, raw, dtcwt and pyqt without importing the optional packages. `gpu`
is `null` until a background device probe or OS hardware query has completed.

### Stack

| Command | Args | Result |
|---|---|---|
| `stack.load` | `folder` \| `files` \| `video`, `scale`, `append` | `{loading:true, source, load_id}` |
| `stack.frames` | — | `{count, names, included, folder, scale, shape, dtype, load:{id,active,finished,error}}` |
| `stack.clear` | — | `{cleared:true}` |
| `stack.include` | `indices`, `include` | current stack info |

`stack.load` uses the same worker as the Open button. Loads started from the
menu or a drag-drop advance the same `load_id`, so a wait can bind to one
request instead of any non-empty stack. Poll `stack.frames` until
`load.finished` is true and inspect `load.error`; a failed or cancelled decode
reports `finished:true` with the reason and leaves the previous frames alone.

### Method, alignment and options

| Command | Args | Result |
|---|---|---|
| `method.set` | `id`, `parameters?` | current method + resolved parameters |
| `method.parameters` | `id?` | `{id, parameters, specs, depth}` |
| `align.set` | `id`, `parameters?` | current alignment + parameters |
| `option.set` | `name`, `value` | `{name, value}` |

`option.set` names: `base_correct`, `exposure_match`, `reverse_order`,
`thread_count`, `use_gpu`, `tile_enabled`, `tile_block_size`, `tile_overlap`,
`tile_threshold`, `preview`, `ai_batch_size`, `resource_preference`,
`reg_downscale_width`.

### Render

| Command | Args | Result |
|---|---|---|
| `render.start` | `preview?`, `confirm?`, `roi_mode?`, `roi_base_index?` | `{started, render_id, method, alignment, preview}` |
| `render.cancel` | — | `{cancelling:true}` |
| `render.state` | — | `{state, render_id, busy, progress:{done,total}, method, result, error}` |
| `render.result` | — | result metadata + selected output record |

`state` is `idle`, `rendering`, `done`, `cancelled`, `discarded` or `error`.
A cancelled render and a result discarded because the stack changed while it
ran are terminal failures, not `done`, even when an older `fusion_result`
still exists. `render.start` answers `confirm_required` instead of opening the
CPU-AI or ROI modal; retry with `confirm:true` or `roi_mode` plus
`roi_base_index`. Renders started by the render button advance the same
`render_id`, so `--wait` never accepts a previous render's completion.

### Outputs and export

| Command | Args | Result |
|---|---|---|
| `output.list` | — | records with index, label, kind, exported, retouchable, shape, dtype, export_path |
| `output.select` | `index` | `{selected, label}` |
| `output.display` | `index` | `{displayed:true}` |
| `output.export` | `path`, `index?` | `{path, written:true}` |
| `output.delete` | `index`, `force?` | `{deleted:true}`; protected unexported retouch/compare outputs require `force:true` |

`output.export` writes the selected (or given) output without a file dialog,
through the export manager's normal writer (labels, scale bar, bit depth). For
an aligned-stack output it writes the frame folder and returns
`kind:"aligned"`; layered TIFF is not part of this command (use the output
list's context menu through `ui.menu`).

### Project

| Command | Args | Result |
|---|---|---|
| `project.save` | `path?` | `{saved:true, path}`; without `path` saves the open project |
| `project.open` | `path`, `force?` | `{loading:true, path, load_id, discarded_unsaved}` |

Both commands share the GUI's project lifecycle: `project.save` updates
`project_path` and clears the dirty marker, `project.open` refuses with
`confirm_required` while the current project has unsaved changes and only
proceeds with `force:true` (which discards them), then tracks the new path
and clears the dirty marker once loaded.

### Dialogs

| Command | Args | Result |
|---|---|---|
| `dialog.open` | `name`, plus name-specific arguments | `{opened, title}` |
| `dialog.list` | — | every visible top-level window: `index`, `root`, `title`, `class`, `visible`, `modal`, `popup`, `managed`, `active`, `busy` |
| `dialog.close` | `index?`, `title?`, `root?` | `{closed:true}` |
| `dialog.accept` | `index?`, `title?` | `{accepted:true}`; applies pending edits and clicks the dialog's own OK/Apply action instead of raw `accept()` — refuses when busy or the commit action is disabled |
| `dialog.reject` | `index?`, `title?` | `{rejected:true}` |
| `dialog.file` | `path`, `index?`, `title?`, `cancel?` | answers a `QFileDialog` by path (or cancels it) |
| `dialog.color` | `color`, `index?` | answers a `QColorDialog` with `#rrggbb` |

Names: `settings`, `batch`, `compare`, `depth`, `feedback`, `scalebar`,
`dust`, `retouch`, `welcome`, `method_help`, `registration_help`,
`environment`, `contact`.

`dialog.list` is not limited to these named dialogs: it includes **every
visible Qt top-level window**, including file pickers, message boxes,
`QInputDialog`, `QColorDialog`, label/scale/ROI dialogs opened by actions, and
menus (`popup[N]`). Entries carry `root` (`dialog[N]`/`popup[N]`), `active`
(the window that would receive input) and `managed` (opened through
`dialog.open`). Startup prompts such as Session Recovery appear here too;
handle them before starting a workflow.

Managed dialogs and `dialog.open` are modeless so commands continue to flow.
Dialogs opened by a menu action with `exec()` are reachable in the same way:
`ui.trigger` returns, then `dialog.list`, `ui.find`, `ui.click` and
`dialog.accept` operate their widgets from inside the nested Qt loop.

Dialog controls carry `controlId` values such as `method_combo`,
`compute_btn`, `btn_save_3d`, `btn_run`, `btn_adopt_output`, `btn_start`,
`fusion_combo` and `method_<id>` checkboxes. `dialog.accept` drives the normal
accept path, so an accepted batch dialog starts its batch exactly like the
menu action.

### UI surface (any capability)

| Command | Args | Result |
|---|---|---|
| `ui.actions` | — | every menu action: key, text, enabled, checkable, checked, shortcut, source, menu |
| `ui.trigger` | `key` | `{scheduled:true}` (runs after a short flush delay) |
| `ui.tree` | `window?`, `depth?` | nested widgets: path, class, objectName, controlId, text, enabled, visible, focus, checked, value, geometry, global origin, children |
| `ui.find` | `object_name?` (also matches controlId), `control_id?`, `class?`, `text?`, `window?` | matching nodes |
| `ui.click` | `path`, `async?` | `{clicked:true}`; with `async:true` returns `{scheduled:true}` before the slot runs |
| `ui.set` | `path`, `value` | written value |
| `ui.grab` | `path?`, `output` | `{path, width, height}` |
| `ui.context` | — | level, active window, focused widget, roots, dialogs, view, stack, render |
| `ui.focus` | `path` | `{focused:true}` |
| `ui.mouse` | `path`, `action`, `x?`, `y?`, `to_x?`, `to_y?`, `steps?`, `button?`, `modifiers?`, `delta?` | real mouse click/drag/wheel events |
| `ui.key` | `path?`, `key`, `action?`, `modifiers?` | real key press/click/release |
| `ui.type` | `path?`, `text`, `submit?` | types text; optional Enter |
| `ui.menu` | `action?` or `index?` | list/trigger actions in the open context menu |

Paths are stable within a snapshot: `main/<controlId or objectName or
Class[n]>/…`; dialog roots are `dialog[0]`, `dialog[1]`, … in title order and
menu roots are `popup[0]`, … `controlId` is a dynamic property used for the
right-panel controls whose `objectName` is reserved by the stylesheet
(`btn_render`, `combo_fusion`, `output_list`, …). `ui.find` matches
`object_name`/`control_id` against either identifier.
`ui.actions` keys are the `ui_objs` keys (`action_open_folder`, `action_save`,
…) for fixed actions, which do not change with language. Dynamic leaves such
as recent files/projects (and any action added directly to a menu) get
`menu_action_<n>` keys with `source:"menu"` and a `menu` path; the index is
stable for the current menu. `ui.trigger` resolves both kinds.

Duplicate sibling identifiers are disambiguated with an occurrence suffix, so
the source and result panels are `panelRoot[0]` and `panelRoot[1]`. Always
copy a path from `ui.find`/`ui.tree`; never guess a bare duplicate name.

`ui.tree`/`ui.find` nodes include `geometry` and `global_origin`. Use these
to aim `ui.mouse` at a canvas: coordinates in `ui.mouse` are relative to the
addressed widget. `ui.context` is the level indicator: `level` is `main`,
`dialog`, `popup` or `modal`; `active.root` is the topmost input target; `focus.path`
is the keyboard target. This is how an agent knows whether it is typing into
the main window, a dialog, or a message box.

Because every menu action and every dialog widget is addressable, the surface
covers capabilities that have no dedicated command (labels, retouch, dust map,
multi-page TIFF, updates, help, …).

### Batch

| Command | Args | Result |
|---|---|---|
| `batch.state` | — | `{busy, results, error}` |
| `batch.cancel` | — | `{cancelling:true}` |

Start a batch by opening the dialog (`dialog.open --name batch`), filling it
through `ui.set`, and clicking Start (or `dialog.accept`); both paths run the
dialog's validation before the server forwards settings to `BatchManager`.
`dialog.open --name batch`
also accepts `folders` (a list of stack folders) and `output_dir` (switches to
the custom output folder). For grouped processing, set `chk_grouping` to true
before `group_size_spin`/`group_overlap_spin`; those controls are hidden while
the checkbox is off. Grouped fusion is approximate, uses temporary disk space,
and produces no original-frame depth, provenance or aligned-frame export.

## 6. Security

- The desktop GUI starts the control channel automatically.
- Local IPC only; no TCP port.
- Every request carries a 128-bit token stored in the per-user data directory.
- The token file has owner-only permissions. Any process running as the same
  user can read it, so a shared account also shares control of the GUI.
- The channel exposes file reads/writes the GUI itself can perform; it does not
  add privileges beyond the user's own.
- Install scope: the pip wheel is headless-only and cannot launch the desktop
  app. Use the packaged desktop build (download it from
  **https://openfocus.top**) or a source checkout to start the GUI. To use the
  wheel's `ctl` command against that running GUI, install PyQt6 separately
  with `pip install PyQt6`. The windowed desktop executable can return CLI
  output through inherited standard handles, such as pipes supplied by a
  subprocess; without usable handles it discards output and opens no console.

## 7. CLI additions

```
openfocus version
openfocus describe                 # JSON capabilities/methods/commands
openfocus gui [--control] [--socket NAME] [paths…]
openfocus ctl <command> [flags] [--json|--pretty] [--launch] [--timeout N]
openfocus ctl raw '<json request>'
openfocus ctl render.start --wait
openfocus ctl ui.tree --pretty
openfocus ctl ui.find --object-name combo_fusion
openfocus ctl ui.click --path 'main/…/QPushButton[0]'
openfocus ctl ui.trigger --key action_open_folder
```

`--json-progress` keeps stdout pure NDJSON; human progress and engine prints
are redirected to stderr, and events are written to the stream saved at CLI
startup. Batch folder events and result summaries are emitted with
`json.dumps`, so names with quotes or backslashes cannot corrupt the stream.
`--version` and `--describe` work without an input.
`describe` also lists available CLI subcommands and long options, including
grouped fusion, cleanup, depth and calibration exports.

`ctl --launch` starts the GUI when no live discovery file exists.
`OPENFOCUS_CONTROL_FILE` isolates the discovery file for tests or parallel
sessions; `OPENFOCUS_INSTANCE_NAME` separates single-instance guards.

## 8. Completeness and verification

Completeness argument: typed commands cover the pipeline (load → method →
options → render → outputs/export → project) and batch/depth/compare dialogs;
the generic UI layer covers every menu action and every widget in every dialog,
including state mutation and screenshots. Nothing in the GUI is unreachable.

Verification:

- Protocol tests: framing, token rejection, unknown command, invalid args.
- Server tests on a real offscreen `OpenFocus` window: load, method/params,
  render + poll, export to a path, project save/open, dialog open/close,
  `ui.actions/tree/find/click/set/trigger`, screenshot.
- Client tests: discovery parsing, request/response round-trip, `--launch`.
- Native session: an agent command sequence drives a visible window end to end.
- Cross-process tests exercise default GUI startup and authenticated control.

Completeness has since been extended to human input: real Qt mouse
click/drag/wheel and key/text events (`ui.mouse`, `ui.key`, `ui.type`), all
top-level dialogs/popups (`dialog.list`, `dialog.file`, `dialog.color`,
`ui.menu`), duplicate-safe paths, and `ui.context` level reporting. The
menu-action sweep drives every enabled action (including save/open project,
GIF, multi-page TIFF, stack folders, labels, resize, scale bar, depth/dust,
compare, retouch and settings) and resolves the dialogs each one opens.

Known limits: `ui.trigger` is asynchronous by design (the response is flushed
before the action runs). `ui.click` is synchronous, so a button whose slot
opens a modal `exec()` dialog blocks that command until the dialog closes —
use `ui.click --async` or `ui.trigger`. Set `OPENFOCUS_CONTROL_WATCHDOG=1`
to have the app dump the GUI stack if the thread stalls. Screenshots capture
Qt-rendered pixels, not the OS desktop. `ui.mouse` targets Qt widgets;
external OS drag-and-drop from Explorer is not simulated (use `stack.load`).

See `skills/openfocus-control/SKILL.md` for the agent-facing workflow.
