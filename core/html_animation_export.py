"""Self-contained single-file HTML5 interactive focus-stack animation exporter.

Produces a standalone offline ``.html`` viewer with zero external CDN or network
dependencies. Frames are decoded, resized, and WebP/JPEG base64-encoded in
parallel across CPU threads (preserving exact frame order and using fast IDCT
scaled JPEG decoding for large ``SourceFrame`` inputs). The generated viewer
uses a DPR-aware 60fps ``<canvas>`` stage with pointer/touch drag scrubbing,
mouse-wheel focus stepping, subtle auto-fading HUD, range slider, play/pause,
FPS selector, ping-pong/loop playback, keyboard navigation, and optional
fused-result comparison.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import html
import io
import json
import os
import tempfile
from typing import Callable, Optional, Sequence

import cv2
import numpy as np


_JPEG_EXTS = frozenset((".jpg", ".jpeg", ".jpe", ".jfif"))


class HtmlAnimationExportCancelled(Exception):
    """Raised when an HTML focus-stack animation export is cancelled."""


@dataclass(frozen=True)
class HtmlAnimationOptions:
    """Configuration for generating an interactive HTML5 focus-stack viewer."""

    max_width: int = 1280
    quality: int = 85
    image_format: str = "webp"  # "webp" or "jpeg"
    fps: float = 10.0
    ping_pong: bool = True
    autoplay: bool = False
    include_fused_frame: bool = False
    title: str = "OpenFocus Interactive Focus Stack"
    show_footer: bool = True
    theme: str = "auto"  # "auto", "dark", "light"

    def normalized(self) -> "HtmlAnimationOptions":
        fmt = str(self.image_format or "webp").strip().lower()
        if fmt in ("jpg", "jpeg"):
            fmt = "jpeg"
        else:
            fmt = "webp"
        theme = str(self.theme or "auto").strip().lower()
        if theme not in ("auto", "dark", "light"):
            theme = "auto"
        max_w = int(self.max_width)
        if max_w < 0:
            max_w = 0
        elif 0 < max_w < 64:
            max_w = 64
        quality = max(1, min(100, int(self.quality)))
        fps = max(1.0, min(60.0, float(self.fps)))
        title = (self.title or "").strip() or "OpenFocus Interactive Focus Stack"
        return HtmlAnimationOptions(
            max_width=max_w,
            quality=quality,
            image_format=fmt,
            fps=fps,
            ping_pong=bool(self.ping_pong),
            autoplay=bool(self.autoplay),
            include_fused_frame=bool(self.include_fused_frame),
            title=title,
            show_footer=bool(self.show_footer),
            theme=theme,
        )


def _try_fast_reduced_jpeg(frame, max_width: int) -> Optional[np.ndarray]:
    """Decode an uncached large JPEG ``SourceFrame`` via libjpeg IDCT scaling.

    Preserves >= 1.25x oversampling above ``max_width`` so the subsequent
    ``cv2.INTER_AREA`` resize matches full-resolution quality while avoiding
    full 45MP decode and memory overhead. Returns ``None`` to fall back to the
    standard ``resolve_frame`` path whenever any precondition is not met.
    """
    if max_width <= 0:
        return None
    from core.source_frames import SourceFrame

    if not isinstance(frame, SourceFrame):
        return None
    if (
        frame.transforms
        or frame.linear_metadata is not None
        or frame._rgb_tiff_offset is not None
        or frame.scale != 1.0
        or frame.dtype != np.uint8
        or len(frame.shape) < 2
    ):
        return None
    ext = os.path.splitext(frame.path)[1].lower()
    if ext not in _JPEG_EXTS:
        return None

    frame.validate_source()
    cached = frame.cached_pixels()
    if cached is not None:
        return cached

    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    if src_w >= max_width * 10:
        flag, factor = cv2.IMREAD_REDUCED_COLOR_8, 8
    elif src_w >= max_width * 5:
        flag, factor = cv2.IMREAD_REDUCED_COLOR_4, 4
    elif src_w >= max_width * 2.5:
        flag, factor = cv2.IMREAD_REDUCED_COLOR_2, 2
    else:
        return None

    try:
        raw = np.fromfile(frame.path, dtype=np.uint8)
        if raw.size == 0:
            return None
        decoded = cv2.imdecode(raw, flag)
    except Exception:
        return None
    if decoded is None or decoded.ndim < 2 or decoded.size == 0:
        return None

    exp_h = max(1, (src_h + factor - 1) // factor)
    exp_w = max(1, (src_w + factor - 1) // factor)
    dh, dw = decoded.shape[:2]
    if abs(dh - exp_h) > 4 or abs(dw - exp_w) > 4:
        return None
    return decoded


def _to_bgr_uint8(
    frame,
    *,
    max_width: int = 0,
    linear_dng_metadata=None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Resolve a frame or ndarray into a contiguous 3-channel BGR uint8 array."""
    img = _try_fast_reduced_jpeg(frame, max_width)
    if img is None:
        from core.source_frames import resolve_frame

        img = resolve_frame(frame, should_cancel=should_cancel)
    if img is None:
        raise ValueError("Frame image is None")
    arr = np.asarray(img)
    if arr.ndim < 2 or arr.size == 0:
        raise ValueError("Frame image must be a non-empty 2D or 3D array")
    if (
        linear_dng_metadata is not None
        and arr.dtype == np.uint16
        and arr.ndim == 3
        and arr.shape[2] == 3
    ):
        from utils.linear_dng import develop_linear

        arr = develop_linear(arr, linear_dng_metadata)
    if arr.dtype != np.uint8:
        from utils.image_utils import to_display_uint8

        arr = to_display_uint8(arr)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(arr)


def _resize_for_web(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_bgr


def encode_frame_data_uri(
    frame,
    *,
    max_width: int = 1280,
    quality: int = 85,
    image_format: str = "webp",
    linear_dng_metadata=None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> tuple[str, int, int]:
    """Encode a single BGR/grayscale/16-bit frame into a base64 data URI.

    Returns ``(data_uri, width, height)``. Falls back to JPEG if WebP encoding
    is unavailable in the local OpenCV and Pillow builds.
    """
    max_w = int(max_width)
    bgr = _resize_for_web(
        _to_bgr_uint8(
            frame,
            max_width=max_w,
            linear_dng_metadata=linear_dng_metadata,
            should_cancel=should_cancel,
        ),
        max_w,
    )
    h, w = bgr.shape[:2]
    q = max(1, min(100, int(quality)))
    fmt = "jpeg" if str(image_format).lower() in ("jpg", "jpeg") else "webp"

    encoded_bytes: Optional[bytes] = None
    mime = "image/webp" if fmt == "webp" else "image/jpeg"

    if fmt == "webp":
        try:
            ok, buf = cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), q])
            if ok and buf is not None and buf.size > 0:
                encoded_bytes = buf.tobytes()
        except Exception:
            encoded_bytes = None
        if encoded_bytes is None:
            try:
                from PIL import Image

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bio = io.BytesIO()
                Image.fromarray(rgb).save(bio, format="WEBP", quality=q)
                encoded_bytes = bio.getvalue()
            except Exception:
                encoded_bytes = None
        if encoded_bytes is None:
            fmt = "jpeg"
            mime = "image/jpeg"

    if fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok and buf is not None and buf.size > 0:
            encoded_bytes = buf.tobytes()
        else:
            from PIL import Image

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bio = io.BytesIO()
            Image.fromarray(rgb).save(bio, format="JPEG", quality=q)
            encoded_bytes = bio.getvalue()

    b64 = base64.b64encode(encoded_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}", w, h


def _safe_json_for_script(obj) -> str:
    """Serialize JSON safe to embed inside an HTML ``<script>`` tag."""
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")


def _default_html_workers(task_count: int, max_workers: Optional[int] = None) -> int:
    if max_workers is not None:
        return max(1, min(int(max_workers), task_count))
    cpus = os.cpu_count() or 4
    return max(1, min(16, cpus, task_count))


def build_html_animation(
    frames: Sequence,
    *,
    labels: Optional[Sequence[str]] = None,
    fused_image=None,
    options: Optional[HtmlAnimationOptions] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    max_workers: Optional[int] = None,
    linear_dng_metadata=None,
) -> str:
    """Build a 100% self-contained single-file HTML5 focus-stack viewer string.

    Encodes frames concurrently via :class:`concurrent.futures.ThreadPoolExecutor`
    while strictly preserving exact input frame order.
    """
    frame_list = list(frames or ())
    if not frame_list:
        raise ValueError("At least one focus-stack frame is required for HTML export.")

    if should_cancel is not None and should_cancel():
        raise HtmlAnimationExportCancelled("cancelled")

    opts = (options or HtmlAnimationOptions()).normalized()
    include_fused = bool(opts.include_fused_frame and fused_image is not None)
    n_frames = len(frame_list)
    total_steps = n_frames + (1 if include_fused else 0)

    def _encode_one(frame):
        if should_cancel is not None and should_cancel():
            raise HtmlAnimationExportCancelled("cancelled")
        return encode_frame_data_uri(
            frame,
            max_width=opts.max_width,
            quality=opts.quality,
            image_format=opts.image_format,
            linear_dng_metadata=linear_dng_metadata,
            should_cancel=should_cancel,
        )

    encoded_results: list[Optional[tuple[str, int, int]]] = [None] * n_frames
    fused_result: Optional[tuple[str, int, int]] = None

    workers = _default_html_workers(total_steps, max_workers=max_workers)
    completed = 0

    if workers <= 1:
        for idx, frame in enumerate(frame_list):
            if should_cancel is not None and should_cancel():
                raise HtmlAnimationExportCancelled("cancelled")
            encoded_results[idx] = _encode_one(frame)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total_steps)
        if include_fused:
            if should_cancel is not None and should_cancel():
                raise HtmlAnimationExportCancelled("cancelled")
            fused_result = _encode_one(fused_image)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total_steps)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {}
            for idx, frame in enumerate(frame_list):
                future_map[pool.submit(_encode_one, frame)] = idx
            if include_fused:
                future_map[pool.submit(_encode_one, fused_image)] = -1

            try:
                for fut in as_completed(future_map):
                    if should_cancel is not None and should_cancel():
                        raise HtmlAnimationExportCancelled("cancelled")
                    slot = future_map[fut]
                    res = fut.result()
                    if slot < 0:
                        fused_result = res
                    else:
                        encoded_results[slot] = res
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total_steps)
                    if should_cancel is not None and should_cancel():
                        raise HtmlAnimationExportCancelled("cancelled")
            except Exception:
                for fut in future_map:
                    fut.cancel()
                raise

    if should_cancel is not None and should_cancel():
        raise HtmlAnimationExportCancelled("cancelled")

    encoded_frames = []
    width, height = 0, 0
    for idx, res in enumerate(encoded_results):
        uri, w, h = res
        if idx == 0:
            width, height = w, h
        if labels is not None and idx < len(labels) and labels[idx]:
            label = str(labels[idx])
        else:
            label = f"Plane {idx + 1}"
        encoded_frames.append({"src": uri, "label": label})

    fused_payload = None
    if include_fused and fused_result is not None:
        fused_payload = {"src": fused_result[0], "label": "Fused Result"}

    safe_title = html.escape(opts.title)
    payload = {
        "frames": encoded_frames,
        "fused": fused_payload,
        "fps": opts.fps,
        "pingPong": opts.ping_pong,
        "autoplay": opts.autoplay,
        "width": width,
        "height": height,
        "theme": opts.theme,
    }
    payload_json = _safe_json_for_script(payload)

    fps_choices = sorted({4, 8, 10, 15, 24, 30, int(round(opts.fps))})
    fps_options_html = "\n".join(
        f'          <option value="{v}"{" selected" if v == int(round(opts.fps)) else ""}>{v} FPS</option>'
        for v in fps_choices
    )

    fused_btn_html = (
        '        <button id="btnFused" class="btn" type="button" '
        'title="Toggle Fused Result (F)" aria-pressed="false">Fused Result</button>\n'
        if fused_payload is not None
        else ""
    )

    footer_html = (
        '    <footer class="footer">'
        '<span>OpenFocus Interactive Focus Stack</span>'
        '<span class="sep">&middot;</span>'
        '<span>Drag, wheel, or &larr;/&rarr;/Space to scrub &middot; M or hold Right-Click/Alt for Loupe (wheel=zoom, Ctrl+wheel=size)</span>'
        "</footer>\n"
        if opts.show_footer
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="{html.escape(opts.theme)}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #0f1115;
      --surface: #161a20;
      --surface-hover: #1e232b;
      --border: #262c35;
      --text: #e6e9ef;
      --muted: #9aa4b2;
      --accent: #4c8dff;
      --accent-soft: #16243d;
      --canvas: #0b0d10;
      --badge-bg: rgba(15, 17, 21, 0.76);
      --shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
    }}
    @media (prefers-color-scheme: light) {{
      html[data-theme="auto"] {{
        --bg: #f5f6f8;
        --surface: #ffffff;
        --surface-hover: #f3f4f6;
        --border: #e5e7eb;
        --text: #111827;
        --muted: #6b7280;
        --accent: #2563eb;
        --accent-soft: #eff6ff;
        --canvas: #edeff2;
        --badge-bg: rgba(255, 255, 255, 0.86);
        --shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
      }}
    }}
    html[data-theme="light"] {{
      --bg: #f5f6f8;
      --surface: #ffffff;
      --surface-hover: #f3f4f6;
      --border: #e5e7eb;
      --text: #111827;
      --muted: #6b7280;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --canvas: #edeff2;
      --badge-bg: rgba(255, 255, 255, 0.86);
      --shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .app {{
      min-height: 100%;
      display: flex;
      flex-direction: column;
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px;
      gap: 12px;
    }}
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .title {{
      margin: 0;
      font-size: 1.05rem;
      font-weight: 600;
      letter-spacing: -0.01em;
    }}
    .header-meta {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .stage-wrap {{
      position: relative;
      flex: 1 1 auto;
      min-height: 280px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--canvas);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: var(--shadow);
      overflow: hidden;
      cursor: ew-resize;
      user-select: none;
      -webkit-user-select: none;
      touch-action: pan-y;
    }}
    .stage-wrap.loupe-on {{ cursor: crosshair; touch-action: none; }}
    .stage-wrap.dragging {{ cursor: grabbing; }}
    .stage-canvas {{
      display: block;
      max-width: 100%;
      max-height: 75vh;
      width: auto;
      height: auto;
      pointer-events: none;
    }}
    .hud {{
      transition: opacity 0.35s ease, transform 0.35s ease;
    }}
    .stage-wrap.hud-idle .badge {{
      opacity: 0.22;
    }}
    .stage-wrap.hud-idle .hint-pill {{
      opacity: 0;
      transform: translateY(4px);
    }}
    .stage-wrap:hover .badge {{
      opacity: 1;
    }}
    .badge {{
      position: absolute;
      top: 12px;
      left: 12px;
      background: var(--badge-bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 5px 12px;
      font-size: 0.82rem;
      font-variant-numeric: tabular-nums;
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      pointer-events: none;
    }}
    .hint-pill {{
      position: absolute;
      bottom: 12px;
      right: 12px;
      background: var(--badge-bg);
      color: var(--muted);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.76rem;
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
      pointer-events: none;
    }}
    .controls {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .scrubber-row {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .scrubber {{
      flex: 1 1 auto;
      accent-color: var(--accent);
      cursor: pointer;
      height: 6px;
    }}
    .counter {{
      min-width: 72px;
      text-align: right;
      font-size: 0.84rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .btn-group {{
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .btn, .select {{
      appearance: none;
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 0.85rem;
      cursor: pointer;
      transition: background 0.15s ease, border-color 0.15s ease;
    }}
    .btn:hover, .select:hover {{
      background: var(--surface-hover);
      border-color: var(--accent);
    }}
    .btn.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
      font-weight: 600;
    }}
    .btn.active {{
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 600;
    }}
    .footer {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 0.78rem;
      color: var(--muted);
      padding-bottom: 4px;
    }}
    .sep {{ opacity: 0.5; }}
    @media (max-width: 640px) {{
      .app {{ padding: 10px; gap: 8px; }}
      .stage-wrap {{ min-height: 220px; border-radius: 10px; }}
      .controls {{ padding: 10px 12px; border-radius: 10px; }}
      .btn, .select {{ padding: 6px 9px; font-size: 0.8rem; }}
      .hint-pill {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main class="app">
    <header class="header">
      <h1 class="title">{safe_title}</h1>
      <div class="header-meta">
        <span id="metaDims">{width}&times;{height} &middot; {len(encoded_frames)} frames</span>
        <button id="btnTheme" class="btn" type="button" title="Toggle light/dark theme">Theme</button>
      </div>
    </header>

    <section id="stage" class="stage-wrap" aria-label="Interactive focus stack viewer">
      <canvas id="viewerCanvas" class="stage-canvas" width="{max(1, width)}" height="{max(1, height)}" aria-label="Focus stack frame"></canvas>
      <div id="frameBadge" class="badge hud">1 / {len(encoded_frames)}</div>
      <div id="dragHint" class="hint-pill hud">&larr; Drag or wheel to focus &middot; M or Right-Click for Loupe &rarr;</div>
    </section>

    <section class="controls" aria-label="Playback controls">
      <div class="scrubber-row">
        <input id="scrubber" class="scrubber" type="range" min="0" max="{max(0, len(encoded_frames) - 1)}" value="0" step="1" aria-label="Focus plane slider">
        <span id="frameCounter" class="counter">1 / {len(encoded_frames)}</span>
      </div>
      <div class="toolbar">
        <div class="btn-group">
          <button id="btnPrev" class="btn" type="button" title="Previous focus plane (Left Arrow)">&larr;</button>
          <button id="btnPlay" class="btn primary" type="button" title="Play / Pause (Space)">Play</button>
          <button id="btnNext" class="btn" type="button" title="Next focus plane (Right Arrow)">&rarr;</button>
        </div>
        <div class="btn-group">
          <button id="btnLoupe" class="btn" type="button" title="Toggle Magnifier Loupe (M) or hold Right-Click / Alt (Wheel: zoom, Ctrl+Wheel or [ ]: size, Shift+Wheel: focus)" aria-pressed="false">Loupe</button>
          <button id="btnLoupeShape" class="btn" type="button" title="Toggle Square / Circle Loupe (L)" hidden>Square</button>
          <button id="btnMode" class="btn" type="button" title="Toggle Ping-Pong / Loop playback">Ping-Pong</button>
          <select id="fpsSelect" class="select" aria-label="Playback speed (FPS)">
{fps_options_html}
          </select>
{fused_btn_html}        </div>
      </div>
    </section>

{footer_html}  </main>

  <script>
  (function() {{
    "use strict";
    var DATA = {payload_json};
    var frames = DATA.frames || [];
    var fused = DATA.fused || null;
    var count = frames.length;
    var currentIndex = 0;
    var showingFused = false;
    var playing = false;
    var pingPong = !!DATA.pingPong;
    var direction = 1;
    var fps = Number(DATA.fps) || 10;
    var rafId = 0;
    var paintRaf = 0;
    var lastTick = 0;
    var hudTimer = 0;
    var hintDismissed = false;

    var stage = document.getElementById("stage");
    var canvas = document.getElementById("viewerCanvas");
    var ctx = canvas && canvas.getContext ? canvas.getContext("2d", {{ alpha: false }}) : null;
    var badge = document.getElementById("frameBadge");
    var counter = document.getElementById("frameCounter");
    var dragHint = document.getElementById("dragHint");
    var scrubber = document.getElementById("scrubber");
    var btnPlay = document.getElementById("btnPlay");
    var btnPrev = document.getElementById("btnPrev");
    var btnNext = document.getElementById("btnNext");
    var btnMode = document.getElementById("btnMode");
    var btnFused = document.getElementById("btnFused");
    var btnTheme = document.getElementById("btnTheme");
    var btnLoupe = document.getElementById("btnLoupe");
    var btnLoupeShape = document.getElementById("btnLoupeShape");
    var fpsSelect = document.getElementById("fpsSelect");

    var nativeW = Math.max(1, Number(DATA.width) || 1);
    var nativeH = Math.max(1, Number(DATA.height) || 1);

    // Desktop-parity Magnifier Loupe state
    var LOUPE_MIN_SIZE = 120;
    var LOUPE_MAX_SIZE = 520;
    var LOUPE_MIN_ZOOM = 0.5;
    var LOUPE_MAX_ZOOM = 16.0;
    var loupePinned = false;
    var loupePeek = false;
    var loupeRightDown = false;
    var loupeShape = "square";
    var loupeSize = 280;
    var loupeZoom = 2.0;
    var loupeZoomInitialized = false;
    var loupeHasPos = false;
    var loupeCssX = 0;
    var loupeCssY = 0;
    var lastPointerClientX = null;
    var lastPointerClientY = null;

    function isLoupeActive() {{
      return loupePinned || loupePeek;
    }}

    function ensureNativeLoupe(srcImg) {{
      if (loupeZoomInitialized || !srcImg || !srcImg.naturalWidth) return;
      var cssW = (canvas && (canvas.clientWidth || canvas.width)) || nativeW;
      var fitScale = cssW / srcImg.naturalWidth;
      var nativeFloor = 1.0 / Math.max(1e-6, fitScale);
      loupeZoom = Math.min(Math.max(loupeZoom, nativeFloor), LOUPE_MAX_ZOOM);
      loupeZoomInitialized = true;
    }}

    function syncLoupePosition(clientX, clientY) {{
      if (!canvas) return;
      var rect = canvas.getBoundingClientRect();
      if (!rect || rect.width <= 0 || rect.height <= 0) return;
      loupeCssX = clientX - rect.left;
      loupeCssY = clientY - rect.top;
      loupeHasPos = true;
    }}

    function updateLoupeUi() {{
      var active = isLoupeActive();
      stage.classList.toggle("loupe-on", active);
      if (btnLoupe) {{
        btnLoupe.classList.toggle("active", loupePinned);
        btnLoupe.setAttribute("aria-pressed", loupePinned ? "true" : "false");
      }}
      if (btnLoupeShape) {{
        btnLoupeShape.hidden = !loupePinned;
        btnLoupeShape.textContent = loupeShape === "circle" ? "Circle" : "Square";
      }}
      if (dragHint && active) {{
        dragHint.textContent = "Loupe: Wheel=Zoom \u00b7 Ctrl+Wheel=Size \u00b7 Shift+Wheel=Focus \u00b7 L=Shape \u00b7 M/Esc=Exit";
      }} else if (dragHint) {{
        dragHint.textContent = "\u2190 Drag or wheel to focus \u00b7 M or Right-Click for Loupe \u2192";
      }}
    }}

    function setLoupePinned(next) {{
      loupePinned = !!next;
      if (loupePinned && !loupeHasPos && canvas) {{
        var r = canvas.getBoundingClientRect();
        loupeCssX = (r.width || canvas.width) * 0.5;
        loupeCssY = (r.height || canvas.height) * 0.5;
        loupeHasPos = true;
      }}
      updateLoupeUi();
      schedulePaint();
    }}

    function resizeCanvas() {{
      if (!canvas) return;
      var dpr = Math.min(2, window.devicePixelRatio || 1);
      var rect = stage.getBoundingClientRect();
      var availW = Math.max(1, rect.width || nativeW);
      var availH = Math.max(1, Math.min(window.innerHeight * 0.75, rect.height || nativeH));
      var scale = Math.min(availW / nativeW, availH / nativeH, 1.0);
      var cssW = Math.max(1, Math.round(nativeW * scale));
      var cssH = Math.max(1, Math.round(nativeH * scale));
      canvas.style.width = cssW + "px";
      canvas.style.height = cssH + "px";
      var targetW = Math.min(nativeW, Math.max(1, Math.round(cssW * dpr)));
      var targetH = Math.min(nativeH, Math.max(1, Math.round(cssH * dpr)));
      if (canvas.width !== targetW || canvas.height !== targetH) {{
        canvas.width = targetW;
        canvas.height = targetH;
      }}
      schedulePaint();
    }}

    function wakeHud() {{
      stage.classList.remove("hud-idle");
      if (hudTimer) window.clearTimeout(hudTimer);
      hudTimer = window.setTimeout(function() {{
        stage.classList.add("hud-idle");
      }}, 2400);
    }}

    var decodedImages = new Array(count);
    var fusedImage = null;

    function loadImage(src, onReady) {{
      var el = new Image();
      el.decoding = "async";
      el.onload = function() {{ onReady(el); }};
      el.src = src;
      if (el.complete && el.naturalWidth > 0) {{
        onReady(el);
      }}
      return el;
    }}

    for (var i = 0; i < count; i++) {{
      (function(idx) {{
        decodedImages[idx] = loadImage(frames[idx].src, function(el) {{
          if (idx === currentIndex && !showingFused) schedulePaint();
        }});
      }})(i);
    }}
    if (fused) {{
      fusedImage = loadImage(fused.src, function() {{
        if (showingFused) schedulePaint();
      }});
    }}

    function drawLoupe(srcImg) {{
      if (!ctx || !srcImg || !srcImg.complete || !srcImg.naturalWidth || !isLoupeActive() || !loupeHasPos) return;
      var dispW = Math.max(1, canvas.clientWidth || canvas.width);
      var dispH = Math.max(1, canvas.clientHeight || canvas.height);
      var dpr = canvas.width / dispW;
      var lensCss = Math.min(loupeSize, Math.max(96, Math.min(dispW, dispH) * 0.92));
      if (loupeCssX < -lensCss * 0.35 || loupeCssY < -lensCss * 0.35 ||
          loupeCssX > dispW + lensCss * 0.35 || loupeCssY > dispH + lensCss * 0.35) {{
        return;
      }}
      ensureNativeLoupe(srcImg);

      var natW = srcImg.naturalWidth;
      var natH = srcImg.naturalHeight;
      var fitScale = dispW / natW;
      var srcCx = (loupeCssX / dispW) * natW;
      var srcCy = (loupeCssY / dispH) * natH;
      var patchSrc = lensCss / (loupeZoom * fitScale);
      var sx = srcCx - patchSrc * 0.5;
      var sy = srcCy - patchSrc * 0.5;

      var lensPx = lensCss * dpr;
      var tx = loupeCssX * dpr - lensPx * 0.5;
      var ty = loupeCssY * dpr - lensPx * 0.5;
      var nativeScalePx = lensPx / Math.max(1e-6, patchSrc);
      var deviceScale = nativeScalePx;

      ctx.save();
      if (loupeShape === "circle") {{
        ctx.beginPath();
        ctx.arc(tx + lensPx * 0.5, ty + lensPx * 0.5, lensPx * 0.5, 0, Math.PI * 2);
        ctx.clip();
      }} else {{
        ctx.beginPath();
        ctx.rect(tx, ty, lensPx, lensPx);
        ctx.clip();
      }}

      ctx.fillStyle = "#181818";
      ctx.fillRect(tx, ty, lensPx, lensPx);

      var cx0 = Math.max(0, sx);
      var cy0 = Math.max(0, sy);
      var cx1 = Math.min(natW, sx + patchSrc);
      var cy1 = Math.min(natH, sy + patchSrc);
      if (cx1 > cx0 && cy1 > cy0) {{
        var dx = tx + (cx0 - sx) * nativeScalePx;
        var dy = ty + (cy0 - sy) * nativeScalePx;
        var dw = (cx1 - cx0) * nativeScalePx;
        var dh = (cy1 - cy0) * nativeScalePx;
        ctx.imageSmoothingEnabled = deviceScale < 1.0;
        if (ctx.imageSmoothingEnabled) ctx.imageSmoothingQuality = "high";
        ctx.drawImage(srcImg, cx0, cy0, cx1 - cx0, cy1 - cy0, dx, dy, dw, dh);
      }}

      // Subtle center crosshair matching desktop inspection loupe
      var mx = loupeCssX * dpr;
      var my = loupeCssY * dpr;
      var arm = 6 * dpr;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
      ctx.lineWidth = Math.max(1, dpr);
      ctx.beginPath();
      ctx.moveTo(mx - arm, my);
      ctx.lineTo(mx + arm, my);
      ctx.moveTo(mx, my - arm);
      ctx.lineTo(mx, my + arm);
      ctx.stroke();

      // Bottom magnification readout caption inside lens
      var caption = Math.round(deviceScale * 100) + "% (" + loupeZoom.toFixed(1) + "\u00d7)";
      var fontPx = Math.round(11 * dpr);
      ctx.font = "600 " + fontPx + "px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      var tw = ctx.measureText(caption).width;
      var padX = 8 * dpr;
      var pillW = tw + padX * 2;
      var pillH = 20 * dpr;
      var pillX = tx + (lensPx - pillW) * 0.5;
      var pillY = ty + lensPx - pillH - 6 * dpr;
      ctx.fillStyle = "rgba(18, 20, 26, 0.84)";
      if (ctx.roundRect) {{
        ctx.beginPath();
        ctx.roundRect(pillX, pillY, pillW, pillH, 999);
        ctx.fill();
      }} else {{
        ctx.fillRect(pillX, pillY, pillW, pillH);
      }}
      ctx.fillStyle = "#ffffff";
      ctx.fillText(caption, pillX + pillW * 0.5, pillY + pillH * 0.52);

      ctx.restore();

      // Crisp border ring
      ctx.save();
      ctx.beginPath();
      if (loupeShape === "circle") {{
        ctx.arc(tx + lensPx * 0.5, ty + lensPx * 0.5, lensPx * 0.5 - dpr, 0, Math.PI * 2);
      }} else {{
        ctx.rect(tx + dpr * 0.5, ty + dpr * 0.5, lensPx - dpr, lensPx - dpr);
      }}
      ctx.strokeStyle = "rgba(0, 0, 0, 0.45)";
      ctx.lineWidth = 3 * dpr;
      ctx.stroke();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
      ctx.lineWidth = 1.75 * dpr;
      ctx.stroke();
      ctx.restore();
    }}

    function drawCurrentFrame() {{
      paintRaf = 0;
      if (!ctx || count === 0) return;
      var srcImg = (showingFused && fusedImage) ? fusedImage : decodedImages[currentIndex];
      if (!srcImg || !srcImg.complete || srcImg.naturalWidth === 0) return;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(srcImg, 0, 0, canvas.width, canvas.height);
      drawLoupe(srcImg);
    }}

    function schedulePaint() {{
      if (!paintRaf) {{
        paintRaf = window.requestAnimationFrame(drawCurrentFrame);
      }}
    }}

    function clampIndex(idx) {{
      if (count <= 1) return 0;
      return Math.max(0, Math.min(count - 1, idx));
    }}

    function render() {{
      if (count === 0) return;
      if (showingFused && fused) {{
        badge.textContent = fused.label || "Fused Result";
        counter.textContent = "Fused";
      }} else {{
        var item = frames[currentIndex];
        var label = item.label ? (" \u00b7 " + item.label) : "";
        badge.textContent = (currentIndex + 1) + " / " + count + label;
        counter.textContent = (currentIndex + 1) + " / " + count;
      }}
      scrubber.value = String(currentIndex);
      if (btnFused) {{
        btnFused.classList.toggle("active", showingFused);
        btnFused.setAttribute("aria-pressed", showingFused ? "true" : "false");
      }}
      schedulePaint();
    }}

    function setIndex(idx) {{
      showingFused = false;
      var next = clampIndex(idx);
      if (next !== currentIndex || showingFused) {{
        currentIndex = next;
      }}
      render();
    }}

    function stepFrame(delta) {{
      if (count <= 1) return;
      showingFused = false;
      var next = currentIndex + delta;
      if (next < 0) next = count - 1;
      if (next >= count) next = 0;
      currentIndex = next;
      render();
    }}

    function advancePlayback() {{
      if (count <= 1) return;
      if (pingPong) {{
        var next = currentIndex + direction;
        if (next >= count) {{
          direction = -1;
          next = Math.max(0, count - 2);
        }} else if (next < 0) {{
          direction = 1;
          next = Math.min(count - 1, 1);
        }}
        currentIndex = next;
      }} else {{
        currentIndex = (currentIndex + 1) % count;
      }}
      render();
    }}

    function tick(ts) {{
      if (!playing) return;
      var interval = 1000 / Math.max(1, fps);
      if (!lastTick) lastTick = ts;
      if (ts - lastTick >= interval) {{
        lastTick = ts - ((ts - lastTick) % interval);
        advancePlayback();
      }}
      rafId = window.requestAnimationFrame(tick);
    }}

    function setPlaying(state) {{
      playing = !!state && count > 1;
      if (playing) {{
        showingFused = false;
        lastTick = 0;
        btnPlay.textContent = "Pause";
        rafId = window.requestAnimationFrame(tick);
      }} else {{
        btnPlay.textContent = "Play";
        if (rafId) {{
          window.cancelAnimationFrame(rafId);
          rafId = 0;
        }}
      }}
    }}

    function updateModeButton() {{
      btnMode.textContent = pingPong ? "Ping-Pong" : "Loop";
      btnMode.classList.toggle("active", pingPong);
    }}

    btnPlay.addEventListener("click", function() {{ wakeHud(); setPlaying(!playing); }});
    btnPrev.addEventListener("click", function() {{ wakeHud(); setPlaying(false); stepFrame(-1); }});
    btnNext.addEventListener("click", function() {{ wakeHud(); setPlaying(false); stepFrame(1); }});
    btnMode.addEventListener("click", function() {{
      pingPong = !pingPong;
      updateModeButton();
    }});
    fpsSelect.addEventListener("change", function() {{
      fps = Math.max(1, Math.min(60, Number(fpsSelect.value) || 10));
    }});
    scrubber.addEventListener("input", function() {{
      wakeHud();
      setPlaying(false);
      setIndex(Number(scrubber.value));
    }});
    if (btnFused) {{
      btnFused.addEventListener("click", function() {{
        wakeHud();
        setPlaying(false);
        showingFused = !showingFused;
        render();
      }});
    }}
    btnTheme.addEventListener("click", function() {{
      var root = document.documentElement;
      var cur = root.getAttribute("data-theme") || "auto";
      var isLight = cur === "light" || (cur === "auto" && window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches);
      root.setAttribute("data-theme", isLight ? "dark" : "light");
    }});

    if (btnLoupe) {{
      btnLoupe.addEventListener("click", function() {{
        wakeHud();
        setLoupePinned(!loupePinned);
      }});
    }}
    if (btnLoupeShape) {{
      btnLoupeShape.addEventListener("click", function() {{
        wakeHud();
        loupeShape = (loupeShape === "square") ? "circle" : "square";
        updateLoupeUi();
        schedulePaint();
      }});
    }}

    // Mouse-wheel: zoom/resize loupe when active, otherwise scrub focus planes (Shift+wheel always scrubs focus)
    var wheelAccum = 0;
    stage.addEventListener("wheel", function(e) {{
      var activeLoupe = isLoupeActive();
      if (!activeLoupe && (e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      wakeHud();
      syncLoupePosition(e.clientX, e.clientY);

      if (activeLoupe && !e.shiftKey) {{
        var rawDelta = Math.abs(e.deltaY) >= Math.abs(e.deltaX) ? e.deltaY : e.deltaX;
        var normSteps = e.deltaMode === 1 ? (rawDelta / 3.0) : (rawDelta / 120.0);
        if (normSteps === 0) return;
        normSteps = Math.max(-4.0, Math.min(4.0, normSteps));
        if (e.ctrlKey || e.metaKey) {{
          loupeSize = Math.max(LOUPE_MIN_SIZE, Math.min(LOUPE_MAX_SIZE, loupeSize * Math.pow(1.1, -normSteps)));
        }} else {{
          loupeZoom = Math.max(LOUPE_MIN_ZOOM, Math.min(LOUPE_MAX_ZOOM, loupeZoom * Math.pow(1.2, -normSteps)));
        }}
        schedulePaint();
        return;
      }}

      if (count <= 1) return;
      setPlaying(false);
      var dy = Math.abs(e.deltaY) >= Math.abs(e.deltaX) ? e.deltaY : e.deltaX;
      var threshold = e.deltaMode === 1 ? 1 : 28;
      wheelAccum += dy;
      var steps = Math.trunc(wheelAccum / threshold);
      if (steps !== 0) {{
        wheelAccum -= steps * threshold;
        setIndex(currentIndex + (steps > 0 ? 1 : -1) * Math.min(Math.abs(steps), 4));
      }}
    }}, {{ passive: false }});

    // Pointer & touch horizontal scrub with gesture axis lock + right-click / pinned Loupe tracking
    var dragging = false;
    var dragStartX = 0;
    var dragStartY = 0;
    var dragStartIndex = 0;
    var touchAxis = null;

    function beginDrag(clientX, clientY) {{
      if (count <= 1) return;
      dragging = true;
      dragStartX = clientX;
      dragStartY = clientY || 0;
      dragStartIndex = currentIndex;
      touchAxis = null;
      setPlaying(false);
      stage.classList.add("dragging");
      wakeHud();
      if (!hintDismissed && dragHint) {{
        hintDismissed = true;
        dragHint.style.opacity = "0";
      }}
    }}

    function moveDrag(clientX) {{
      if (!dragging || count <= 1) return;
      var rect = stage.getBoundingClientRect();
      var span = Math.max(120, rect.width * 0.72);
      var dx = clientX - dragStartX;
      var deltaFrames = Math.round((dx / span) * (count - 1));
      setIndex(dragStartIndex + deltaFrames);
      wakeHud();
    }}

    function endDrag() {{
      if (!dragging) return;
      dragging = false;
      touchAxis = null;
      stage.classList.remove("dragging");
    }}

    stage.addEventListener("contextmenu", function(e) {{
      if (isLoupeActive() || loupeRightDown) {{
        e.preventDefault();
      }}
    }});
    stage.addEventListener("pointerdown", function(e) {{
      if (e.pointerType === "touch") return;
      lastPointerClientX = e.clientX;
      lastPointerClientY = e.clientY;
      syncLoupePosition(e.clientX, e.clientY);
      if (e.button === 2) {{
        e.preventDefault();
        loupeRightDown = true;
        loupePeek = true;
        updateLoupeUi();
        wakeHud();
        schedulePaint();
        if (stage.setPointerCapture) {{
          try {{ stage.setPointerCapture(e.pointerId); }} catch (_err) {{}}
        }}
        return;
      }}
      if (loupePinned) {{
        wakeHud();
        schedulePaint();
        return;
      }}
      if (e.button !== 0 && e.pointerType === "mouse") return;
      beginDrag(e.clientX, e.clientY);
      if (stage.setPointerCapture) {{
        try {{ stage.setPointerCapture(e.pointerId); }} catch (_err) {{}}
      }}
    }});
    stage.addEventListener("pointermove", function(e) {{
      if (e.pointerType !== "touch") {{
        wakeHud();
        lastPointerClientX = e.clientX;
        lastPointerClientY = e.clientY;
        if (isLoupeActive()) {{
          syncLoupePosition(e.clientX, e.clientY);
          schedulePaint();
        }}
      }}
      if (dragging && e.pointerType !== "touch") moveDrag(e.clientX);
    }});
    function endPointer(e) {{
      if (e && e.button === 2 && loupeRightDown) {{
        loupeRightDown = false;
        loupePeek = !!(e.altKey);
        updateLoupeUi();
        schedulePaint();
      }}
      endDrag();
    }}
    stage.addEventListener("pointerup", endPointer);
    stage.addEventListener("pointercancel", endPointer);
    stage.addEventListener("pointerleave", function() {{
      lastPointerClientX = null;
      lastPointerClientY = null;
      loupeHasPos = false;
      if (loupePeek && !loupeRightDown) {{
        loupePeek = false;
        updateLoupeUi();
      }}
      if (isLoupeActive() || loupeRightDown) schedulePaint();
    }});

    stage.addEventListener("touchstart", function(e) {{
      if (e.touches && e.touches.length === 1) {{
        if (loupePinned) {{
          syncLoupePosition(e.touches[0].clientX, e.touches[0].clientY);
          wakeHud();
          schedulePaint();
          return;
        }}
        beginDrag(e.touches[0].clientX, e.touches[0].clientY);
      }}
    }}, {{ passive: true }});
    stage.addEventListener("touchmove", function(e) {{
      if (loupePinned && e.touches && e.touches.length === 1) {{
        e.preventDefault();
        syncLoupePosition(e.touches[0].clientX, e.touches[0].clientY);
        wakeHud();
        schedulePaint();
        return;
      }}
      if (!dragging || !e.touches || e.touches.length !== 1) return;
      var tx = e.touches[0].clientX;
      var ty = e.touches[0].clientY;
      if (touchAxis === null) {{
        var adx = Math.abs(tx - dragStartX);
        var ady = Math.abs(ty - dragStartY);
        if (adx > 6 || ady > 6) {{
          touchAxis = adx >= ady ? "x" : "y";
        }}
      }}
      if (touchAxis === "x") {{
        e.preventDefault();
        moveDrag(tx);
      }} else if (touchAxis === "y") {{
        endDrag();
      }}
    }}, {{ passive: false }});
    stage.addEventListener("touchend", endDrag, {{ passive: true }});
    stage.addEventListener("touchcancel", endDrag, {{ passive: true }});

    // Keyboard navigation & Loupe shortcuts
    window.addEventListener("keydown", function(e) {{
      var tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
      if (tag === "select" || tag === "input" || tag === "textarea") return;
      if (e.key === "Alt" && !loupePeek) {{
        if (lastPointerClientX !== null && lastPointerClientY !== null) {{
          syncLoupePosition(lastPointerClientX, lastPointerClientY);
          loupePeek = true;
          updateLoupeUi();
          schedulePaint();
        }}
        return;
      }}
      if (e.key === "m" || e.key === "M") {{
        e.preventDefault();
        wakeHud();
        setLoupePinned(!loupePinned);
      }} else if (e.key === "l" || e.key === "L") {{
        e.preventDefault();
        loupeShape = (loupeShape === "square") ? "circle" : "square";
        updateLoupeUi();
        schedulePaint();
      }} else if (e.key === "Escape" && (loupePinned || loupePeek)) {{
        e.preventDefault();
        loupePinned = false;
        loupePeek = false;
        loupeRightDown = false;
        updateLoupeUi();
        schedulePaint();
      }} else if (isLoupeActive() && (e.key === "[" || e.key === "]")) {{
        e.preventDefault();
        loupeSize = Math.max(LOUPE_MIN_SIZE, Math.min(LOUPE_MAX_SIZE, loupeSize * (e.key === "]" ? 1.1 : 1 / 1.1)));
        schedulePaint();
      }} else if (isLoupeActive() && (e.key === "+" || e.key === "=" || e.key === "-" || e.key === "_")) {{
        e.preventDefault();
        var zoomIn = (e.key === "+" || e.key === "=");
        loupeZoom = Math.max(LOUPE_MIN_ZOOM, Math.min(LOUPE_MAX_ZOOM, loupeZoom * (zoomIn ? 1.2 : 1 / 1.2)));
        schedulePaint();
      }} else if (e.key === "ArrowLeft") {{
        e.preventDefault();
        wakeHud();
        setPlaying(false);
        stepFrame(-1);
      }} else if (e.key === "ArrowRight") {{
        e.preventDefault();
        wakeHud();
        setPlaying(false);
        stepFrame(1);
      }} else if (e.key === " " || e.code === "Space") {{
        e.preventDefault();
        wakeHud();
        setPlaying(!playing);
      }} else if (e.key === "Home") {{
        e.preventDefault();
        wakeHud();
        setPlaying(false);
        setIndex(0);
      }} else if (e.key === "End") {{
        e.preventDefault();
        wakeHud();
        setPlaying(false);
        setIndex(count - 1);
      }} else if ((e.key === "f" || e.key === "F") && fused) {{
        e.preventDefault();
        wakeHud();
        setPlaying(false);
        showingFused = !showingFused;
        render();
      }}
    }});
    window.addEventListener("keyup", function(e) {{
      if (e.key === "Alt" && loupePeek && !loupeRightDown) {{
        loupePeek = false;
        updateLoupeUi();
        schedulePaint();
      }}
    }});

    window.addEventListener("resize", resizeCanvas);
    if (window.ResizeObserver) {{
      try {{
        new window.ResizeObserver(resizeCanvas).observe(stage);
      }} catch (_err) {{}}
    }}

    updateModeButton();
    updateLoupeUi();
    resizeCanvas();
    render();
    wakeHud();
    if (DATA.autoplay && count > 1) {{
      setPlaying(true);
    }}
  }})();
  </script>
</body>
</html>
"""


def export_html_animation(
    path: str,
    frames: Sequence,
    *,
    labels: Optional[Sequence[str]] = None,
    fused_image=None,
    options: Optional[HtmlAnimationOptions] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    max_workers: Optional[int] = None,
    linear_dng_metadata=None,
) -> str:
    """Export a self-contained single-file HTML5 focus-stack viewer to ``path``."""
    target = os.path.abspath(str(path))
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    html_content = build_html_animation(
        frames,
        labels=labels,
        fused_image=fused_image,
        options=options,
        progress_callback=progress_callback,
        should_cancel=should_cancel,
        max_workers=max_workers,
        linear_dng_metadata=linear_dng_metadata,
    )

    if should_cancel is not None and should_cancel():
        raise HtmlAnimationExportCancelled("cancelled")

    fd, part = tempfile.mkstemp(suffix=".html", dir=parent or None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(html_content)
        if should_cancel is not None and should_cancel():
            raise HtmlAnimationExportCancelled("cancelled")
        os.replace(part, target)
        part = None
        return target
    finally:
        if part and os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
