"""Self-contained single-file HTML5 interactive focus-stack animation exporter.

Produces a standalone offline ``.html`` viewer with zero external CDN or network
dependencies. Frames are embedded as base64 WebP or JPEG data URIs and driven by
vanilla JavaScript with pointer drag scrubbing, mobile touch support, range
slider, play/pause, FPS selector, ping-pong/loop playback, keyboard navigation,
and optional fused-result comparison.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import html
import io
import json
import os
import tempfile
from typing import Callable, Optional, Sequence

import cv2
import numpy as np


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


def _to_bgr_uint8(frame) -> np.ndarray:
    """Resolve a frame or ndarray into a contiguous 3-channel BGR uint8 array."""
    from core.source_frames import resolve_frame

    img = resolve_frame(frame)
    if img is None:
        raise ValueError("Frame image is None")
    arr = np.asarray(img)
    if arr.ndim < 2 or arr.size == 0:
        raise ValueError("Frame image must be a non-empty 2D or 3D array")
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
) -> tuple[str, int, int]:
    """Encode a single BGR/grayscale/16-bit frame into a base64 data URI.

    Returns ``(data_uri, width, height)``. Falls back to JPEG if WebP encoding
    is unavailable in the local OpenCV and Pillow builds.
    """
    bgr = _resize_for_web(_to_bgr_uint8(frame), int(max_width))
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


def build_html_animation(
    frames: Sequence,
    *,
    labels: Optional[Sequence[str]] = None,
    fused_image=None,
    options: Optional[HtmlAnimationOptions] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """Build a 100% self-contained single-file HTML5 focus-stack viewer string."""
    frame_list = list(frames or ())
    if not frame_list:
        raise ValueError("At least one focus-stack frame is required for HTML export.")

    opts = (options or HtmlAnimationOptions()).normalized()
    include_fused = bool(opts.include_fused_frame and fused_image is not None)
    total_steps = len(frame_list) + (1 if include_fused else 0)

    encoded_frames = []
    width = 0
    height = 0

    for idx, frame in enumerate(frame_list):
        if should_cancel is not None and should_cancel():
            raise HtmlAnimationExportCancelled("cancelled")
        uri, w, h = encode_frame_data_uri(
            frame,
            max_width=opts.max_width,
            quality=opts.quality,
            image_format=opts.image_format,
        )
        if idx == 0:
            width, height = w, h
        label = ""
        if labels is not None and idx < len(labels) and labels[idx]:
            label = str(labels[idx])
        else:
            label = f"Plane {idx + 1}"
        encoded_frames.append({"src": uri, "label": label})
        if progress_callback is not None:
            progress_callback(idx + 1, total_steps)

    fused_payload = None
    if include_fused:
        if should_cancel is not None and should_cancel():
            raise HtmlAnimationExportCancelled("cancelled")
        fused_uri, _, _ = encode_frame_data_uri(
            fused_image,
            max_width=opts.max_width,
            quality=opts.quality,
            image_format=opts.image_format,
        )
        fused_payload = {"src": fused_uri, "label": "Fused Result"}
        if progress_callback is not None:
            progress_callback(total_steps, total_steps)

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
        '<span>Drag image, scrub slider, or use &larr; / &rarr; / Space</span>'
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
      --badge-bg: rgba(15, 17, 21, 0.78);
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
      overflow: hidden;
      cursor: ew-resize;
      user-select: none;
      -webkit-user-select: none;
      touch-action: pan-y;
    }}
    .stage-wrap.dragging {{ cursor: grabbing; }}
    .stage-img {{
      display: block;
      max-width: 100%;
      max-height: 74vh;
      width: auto;
      height: auto;
      object-fit: contain;
      pointer-events: none;
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
      pointer-events: none;
      transition: opacity 0.3s ease;
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
    @media (max-width: 600px) {{
      .app {{ padding: 10px; }}
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
      <img id="viewerImg" class="stage-img" alt="Focus stack frame" draggable="false">
      <div id="frameBadge" class="badge">1 / {len(encoded_frames)}</div>
      <div id="dragHint" class="hint-pill">&larr; Drag horizontally to focus &rarr;</div>
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
    var lastTick = 0;

    var stage = document.getElementById("stage");
    var img = document.getElementById("viewerImg");
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
    var fpsSelect = document.getElementById("fpsSelect");

    var preloaded = frames.map(function(item) {{
      var el = new Image();
      el.src = item.src;
      return el;
    }});
    if (fused) {{
      var fusedEl = new Image();
      fusedEl.src = fused.src;
    }}

    function clampIndex(idx) {{
      if (count <= 1) return 0;
      return Math.max(0, Math.min(count - 1, idx));
    }}

    function render() {{
      if (count === 0) return;
      if (showingFused && fused) {{
        img.src = fused.src;
        badge.textContent = fused.label || "Fused Result";
        counter.textContent = "Fused";
      }} else {{
        var item = frames[currentIndex];
        img.src = item.src;
        var label = item.label ? (" \u00b7 " + item.label) : "";
        badge.textContent = (currentIndex + 1) + " / " + count + label;
        counter.textContent = (currentIndex + 1) + " / " + count;
      }}
      scrubber.value = String(currentIndex);
      if (btnFused) {{
        btnFused.classList.toggle("active", showingFused);
        btnFused.setAttribute("aria-pressed", showingFused ? "true" : "false");
      }}
    }}

    function setIndex(idx) {{
      showingFused = false;
      currentIndex = clampIndex(idx);
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
        lastTick = ts;
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

    btnPlay.addEventListener("click", function() {{ setPlaying(!playing); }});
    btnPrev.addEventListener("click", function() {{ setPlaying(false); stepFrame(-1); }});
    btnNext.addEventListener("click", function() {{ setPlaying(false); stepFrame(1); }});
    btnMode.addEventListener("click", function() {{
      pingPong = !pingPong;
      updateModeButton();
    }});
    fpsSelect.addEventListener("change", function() {{
      fps = Math.max(1, Math.min(60, Number(fpsSelect.value) || 10));
    }});
    scrubber.addEventListener("input", function() {{
      setPlaying(false);
      setIndex(Number(scrubber.value));
    }});
    if (btnFused) {{
      btnFused.addEventListener("click", function() {{
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

    // Pointer & touch horizontal scrub
    var dragging = false;
    var dragStartX = 0;
    var dragStartIndex = 0;

    function beginDrag(clientX) {{
      if (count <= 1) return;
      dragging = true;
      dragStartX = clientX;
      dragStartIndex = currentIndex;
      setPlaying(false);
      stage.classList.add("dragging");
      if (dragHint) dragHint.style.opacity = "0";
    }}

    function moveDrag(clientX) {{
      if (!dragging || count <= 1) return;
      var rect = stage.getBoundingClientRect();
      var span = Math.max(120, rect.width * 0.75);
      var dx = clientX - dragStartX;
      var deltaFrames = Math.round((dx / span) * (count - 1));
      setIndex(dragStartIndex + deltaFrames);
    }}

    function endDrag() {{
      if (!dragging) return;
      dragging = false;
      stage.classList.remove("dragging");
    }}

    stage.addEventListener("pointerdown", function(e) {{
      if (e.button !== 0 && e.pointerType === "mouse") return;
      beginDrag(e.clientX);
      if (stage.setPointerCapture) {{
        try {{ stage.setPointerCapture(e.pointerId); }} catch (_err) {{}}
      }}
    }});
    stage.addEventListener("pointermove", function(e) {{
      if (dragging) moveDrag(e.clientX);
    }});
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);

    stage.addEventListener("touchstart", function(e) {{
      if (e.touches && e.touches.length === 1) beginDrag(e.touches[0].clientX);
    }}, {{ passive: true }});
    stage.addEventListener("touchmove", function(e) {{
      if (dragging && e.touches && e.touches.length === 1) moveDrag(e.touches[0].clientX);
    }}, {{ passive: true }});
    stage.addEventListener("touchend", endDrag, {{ passive: true }});

    // Keyboard navigation
    window.addEventListener("keydown", function(e) {{
      var tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
      if (tag === "select" || tag === "input" || tag === "textarea") return;
      if (e.key === "ArrowLeft") {{
        e.preventDefault();
        setPlaying(false);
        stepFrame(-1);
      }} else if (e.key === "ArrowRight") {{
        e.preventDefault();
        setPlaying(false);
        stepFrame(1);
      }} else if (e.key === " " || e.code === "Space") {{
        e.preventDefault();
        setPlaying(!playing);
      }} else if (e.key === "Home") {{
        e.preventDefault();
        setPlaying(false);
        setIndex(0);
      }} else if (e.key === "End") {{
        e.preventDefault();
        setPlaying(false);
        setIndex(count - 1);
      }} else if ((e.key === "f" || e.key === "F") && fused) {{
        e.preventDefault();
        setPlaying(false);
        showingFused = !showingFused;
        render();
      }}
    }});

    updateModeButton();
    render();
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
