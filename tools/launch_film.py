"""Render the OpenFocus launch film entirely in code.

Every frame is drawn with PIL/NumPy (typography, easing, glow, grain) and the
soundtrack is synthesised from oscillators and filtered noise. Real product
imagery is composited in: the nine source frames of the heliconsoft7 stack, a
genuine sharpness-selection fusion of that stack, and the feature captures.

Usage:  python tools/launch_film.py [--out launch.mp4] [--preview 120]
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
STACK = Path(r"C:\Users\dell\Pictures\Helicon Focus\StackMFF V2 Used\heliconsoft7")
SHOTS = ROOT / "assets" / "readme"

W, H = 1920, 1080
FPS = 30

BG = (8, 10, 14)
INK = (240, 244, 250)
MUTED = (136, 148, 166)
DIM = (86, 96, 112)
ACCENT = (47, 129, 247)
ACCENT_SOFT = (28, 62, 120)
GOLD = (227, 179, 65)
GREEN = (63, 185, 80)

FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_REG = "C:/Windows/Fonts/arial.ttf"
FONT_MONO = "C:/Windows/Fonts/consolab.ttf"
FONT_MONO_R = "C:/Windows/Fonts/consola.ttf"


# ----------------------------------------------------------------- easing
def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp(t)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_expo(t):
    t = clamp(t)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def smooth(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return a + (b - a) * t


def mixc(c1, c2, t):
    return tuple(int(round(mix(a, b, clamp(t)))) for a, b in zip(c1, c2))


# --------------------------------------------------------------- drawing
_fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(path: str, size: int):
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


def text_tracked(draw, xy, text, fnt, fill, tracking=0.0, anchor_left=True):
    """Draw text with letter-spacing; returns the advance width."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill, anchor="lm" if not anchor_left else "lm")
        x += fnt.getlength(ch) + tracking
    return x - xy[0]


def text_width(fnt, text, tracking=0.0):
    return sum(fnt.getlength(c) for c in text) + tracking * max(0, len(text) - 1)


def radial_glow(size, color, radius_ratio=0.5, strength=1.0):
    """A soft circular glow that reaches zero exactly at the image edge.

    A plain gaussian leaves ~12% alpha in the corners of its bounding box,
    which showed up as a visible square behind the logo. The falloff is
    multiplied by a radial window that is 0 at the border, so the glow can be
    composited anywhere without a seam.
    """
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sigma = max(1.0, min(w, h) * radius_ratio)
    a = np.exp(-(r ** 2) / (2 * sigma ** 2)) * strength
    # Raised-cosine window: value 1 in the middle, exactly 0 at the border
    # with a matching first derivative, so there is no visible circular seam.
    norm = np.clip(r / (min(w, h) / 2.0), 0, 1)
    edge = 0.5 * (1.0 + np.cos(np.pi * norm))
    a = np.clip(a * edge, 0, 1)
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = color
    alpha = (a * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), "RGBA")


def rounded_panel(size, radius=18, fill=(18, 22, 30, 235), border=(255, 255, 255, 34),
                  border_w=2):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=fill,
                        outline=border, width=border_w)
    return img


def vignette(size, strength=0.55):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    r = np.sqrt(nx ** 2 + ny ** 2) / math.sqrt(2)
    a = np.clip((r ** 2.2) * strength, 0, 1)
    rgb = np.zeros((h, w, 3), np.uint8)
    return Image.fromarray(np.dstack([rgb, (a * 255).astype(np.uint8)]), "RGBA")


def make_grain(size, seed=7, amount=7):
    rng = np.random.default_rng(seed)
    n = rng.normal(0, amount, (size[1], size[0], 1)).repeat(3, axis=2)
    base = np.full((size[1], size[0], 3), 128, np.float32) + n
    alpha = np.full((size[1], size[0], 1), 26, np.uint8)
    arr = np.dstack([np.clip(base, 0, 255).astype(np.uint8), alpha])
    return Image.fromarray(arr, "RGBA")


def paste_center(base, img, center):
    x = int(center[0] - img.width / 2)
    y = int(center[1] - img.height / 2)
    base.alpha_composite(img, (x, y))


def fit(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    scale = min(box_w / img.width, box_h / img.height)
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                      Image.LANCZOS)


def shadowed(img, blur=26, opacity=150, offset=(0, 18)):
    pad = blur * 2
    canvas = Image.new("RGBA", (img.width + pad * 2, img.height + pad * 2), (0, 0, 0, 0))
    sil = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sil.paste((0, 0, 0, opacity), (pad + offset[0], pad + offset[1],
                                   pad + offset[0] + img.width,
                                   pad + offset[1] + img.height))
    canvas = Image.alpha_composite(canvas, sil.filter(ImageFilter.GaussianBlur(blur)))
    canvas.alpha_composite(img, (pad, pad))
    return canvas


# ------------------------------------------------------------- media prep
def load_stack_frames(width=1180):
    paths = sorted(STACK.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        frames.append(fit(im, width, int(width * 1.4)))
    return frames


def sharpen_mask(img: Image.Image) -> np.ndarray:
    """Local high-frequency energy, used to pick the sharpest source per pixel."""
    a = np.asarray(img.convert("L"), np.float32)
    pad = np.pad(a, 2, mode="edge")
    lap = (-4 * pad[2:-2, 2:-2] + pad[1:-3, 2:-2] + pad[3:-1, 2:-2]
           + pad[2:-2, 1:-3] + pad[2:-2, 3:-1])
    # Box-blur the energy so neighbouring pixels agree (no salt-and-pepper).
    k = 9
    cum = np.cumsum(np.cumsum(np.abs(lap), axis=0), axis=1)
    cum = np.pad(cum, ((1, 0), (1, 0)))
    h, w = lap.shape
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - k // 2, 0, h)
    y1 = np.clip(ys + k // 2 + 1, 0, h)
    x0 = np.clip(xs - k // 2, 0, w)
    x1 = np.clip(xs + k // 2 + 1, 0, w)
    s = (cum[np.ix_(y1, x1)] - cum[np.ix_(y0, x1)] - cum[np.ix_(y1, x0)]
         + cum[np.ix_(y0, x0)])
    return s / ((y1 - y0)[:, None] * (x1 - x0)[None, :] + 1e-6)


def fuse_stack(frames):
    """Pick, per pixel, the frame with the most local detail: a real fusion."""
    energies = np.stack([sharpen_mask(f) for f in frames])
    pick = energies.argmax(axis=0)
    stack = np.stack([np.asarray(f.convert("RGB")) for f in frames])
    h, w = pick.shape
    out = stack[pick, np.arange(h)[:, None], np.arange(w)[None, :]]
    return Image.fromarray(out.astype(np.uint8))


def load_gif_frames(path, width, limit=None):
    im = Image.open(path)
    out = []
    n = getattr(im, "n_frames", 1)
    step = 1
    if limit and n > limit:
        step = max(1, n // limit)
    for i in range(0, n, step):
        im.seek(i)
        out.append(fit(im.convert("RGB"), width, width))
    return out


# ------------------------------------------------------------------ audio
def synth_audio(duration, path):
    sr = 48000
    n = int(duration * sr)
    t = np.arange(n) / sr
    left = np.zeros(n, np.float32)
    right = np.zeros(n, np.float32)

    def note(freq, start, length, amp, detune=0.0, shape=0.0):
        i0, i1 = int(start * sr), min(n, int((start + length) * sr))
        if i1 <= i0:
            return
        tt = np.arange(i1 - i0) / sr
        env = np.minimum(1.0, tt / 0.08) * np.exp(-tt / (length * 0.55))
        env = env * np.clip((length - tt) / 0.25, 0, 1)
        wave_l = np.sin(2 * np.pi * freq * tt)
        wave_r = np.sin(2 * np.pi * (freq * (1 + detune)) * tt)
        if shape:
            wave_l = wave_l + shape * np.sin(4 * np.pi * freq * tt) * 0.35
            wave_r = wave_r + shape * np.sin(4 * np.pi * freq * (1 + detune) * tt) * 0.35
        left[i0:i1] += amp * env * wave_l
        right[i0:i1] += amp * env * wave_r

    # Slow pad: A minor add9, one chord per scene.
    chords = [
        (0.0, 4.2, [110.0, 164.81, 220.0, 329.63]),
        (3.8, 4.4, [110.0, 164.81, 261.63, 329.63]),
        (7.6, 5.6, [98.0, 146.83, 196.0, 293.66]),
        (12.6, 6.2, [110.0, 164.81, 246.94, 329.63]),
        (18.2, 7.4, [87.31, 130.81, 174.61, 261.63]),
        (25.0, 5.0, [98.0, 146.83, 220.0, 293.66]),
        (29.4, 4.6, [110.0, 164.81, 246.94, 329.63]),
        (33.6, 4.0, [146.83, 220.0, 293.66, 440.0]),
    ]
    for start, length, freqs in chords:
        for k, f in enumerate(freqs):
            note(f, start, length, 0.075 / (1 + 0.35 * k), detune=0.0022 * (k + 1),
                 shape=0.5 if k < 2 else 0.0)

    # Heartbeat pulse for forward motion.
    beat = 1.0
    tt = 0.0
    while tt < duration - 1:
        note(55.0, tt, 0.42, 0.10, detune=0.004)
        tt += beat

    # Scene-transition whooshes: filtered noise with a fast attack.
    rng = np.random.default_rng(3)
    for start in (3.4, 7.2, 12.2, 18.0, 24.8, 29.2, 33.4):
        length = 0.75
        i0, i1 = int(start * sr), min(n, int((start + length) * sr))
        if i1 <= i0:
            continue
        seg = rng.normal(0, 1, i1 - i0).astype(np.float32)
        # One-pole low-pass sweep, bright to dark.
        alpha = np.linspace(0.55, 0.05, i1 - i0)
        out = np.empty_like(seg)
        prev = 0.0
        for i in range(seg.size):
            prev = prev + alpha[i] * (seg[i] - prev)
            out[i] = prev
        env = np.exp(-np.arange(i1 - i0) / (0.22 * sr))
        left[i0:i1] += 0.16 * env * out
        right[i0:i1] += 0.16 * env * out[::-1]

    # Final shimmer.
    for k, f in enumerate((587.33, 880.0, 1174.66)):
        note(f, duration - 3.4, 3.2, 0.045 / (1 + k), detune=0.003, shape=0.2)

    mixdown = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(mixdown)) or 1.0)
    mixdown = mixdown / peak * 0.82
    # Gentle fade in/out.
    fade = int(0.35 * sr)
    mixdown[:fade] *= np.linspace(0, 1, fade)[:, None]
    mixdown[-int(1.2 * sr):] *= np.linspace(1, 0, int(1.2 * sr))[:, None]
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes((mixdown * 32767).astype(np.int16).tobytes())


# ----------------------------------------------------------------- scenes
class Film:
    def __init__(self):
        self.stack = load_stack_frames(1180)
        self.fused = fuse_stack(self.stack)
        self.logo = Image.open(ROOT / "assets" / "OpenFocus.png").convert("RGBA")
        self.logo = fit(self.logo, 460, 460)
        self.glow = radial_glow((1900, 1900), (34, 74, 132), 0.34, 0.42)
        self.glow_gold = radial_glow((1900, 1900), (128, 104, 44), 0.30, 0.17)
        self.vig = vignette((W, H), 0.62)
        self.grain = [make_grain((W, H), seed=s) for s in (11, 12, 13)]
        self.compare = load_gif_frames(SHOTS / "method-comparison.gif", 1180, limit=48)
        self.depth = load_gif_frames(SHOTS / "depth.gif", 760, limit=36)
        self.retouch = load_gif_frames(SHOTS / "retouch.gif", 900, limit=36)
        self.batch = load_gif_frames(SHOTS / "batch.gif", 760, limit=30)
        self.scalebar = load_gif_frames(SHOTS / "scalebar.gif", 760, limit=30)
        self.wipe = load_gif_frames(SHOTS / "wipe.gif", 1100, limit=40)

    # ---------------------------------------------------------- helpers
    def background(self, t):
        base = Image.new("RGBA", (W, H), BG + (255,))
        d = ImageDraw.Draw(base)
        # Slow drifting accent wash.
        for i in range(3):
            cx = W * (0.2 + 0.3 * i) + math.sin(t * 0.22 + i) * 120
            cy = H * (0.3 + 0.2 * i) + math.cos(t * 0.18 + i * 1.7) * 90
            r = 620 - i * 120
            a = 10 + 4 * math.sin(t * 0.5 + i)
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=(int(ACCENT[0] * 0.5), int(ACCENT[1] * 0.5), int(ACCENT[2] * 0.6),
                            max(0, int(a))))
        return base.filter(ImageFilter.GaussianBlur(70))

    def caption(self, img, text, sub=None, t_in=1.0, y=None):
        """Lower-left caption block with an accent rule."""
        if t_in <= 0:
            return img
        d = ImageDraw.Draw(img)
        x = 150
        yy = y if y is not None else H - 172
        f = font(FONT_BOLD, 26)
        track = 5.0
        a = int(255 * ease_out(t_in))
        rule = int(54 * ease_out(t_in))
        d.rectangle([x, yy - 14, x + rule, yy - 11],
                    fill=ACCENT + (a,))
        text_tracked(d, (x + rule + 18, yy - 12), text.upper(), f,
                     INK + (a,), track)
        if sub:
            f2 = font(FONT_REG, 27)
            d.text((x, yy + 34), sub, font=f2, fill=MUTED + (int(215 * ease_out(t_in)),))
        return img

    # ----------------------------------------------------------- scenes
    def scene_logo(self, lt):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        # Glow blooms first.
        g = self.glow.copy()
        ga = int(255 * mix(0.45, 1.0, ease_out(clamp(lt / 1.4))))
        g.putalpha(g.getchannel("A").point(lambda v: v * ga // 255))
        paste_center(img, g, (W / 2, H / 2 - 40))
        # Logo rises and settles.
        p = ease_out_expo(clamp(lt / 1.5))
        size = mix(0.72, 1.0, p)
        lg = self.logo.resize((int(self.logo.width * size), int(self.logo.height * size)),
                              Image.LANCZOS)
        a = int(255 * ease_out(clamp(lt / 1.1)))
        lg.putalpha(lg.getchannel("A").point(lambda v: v * a // 255))
        paste_center(img, lg, (W / 2, H / 2 - 40 - (1 - p) * 40))
        # Wordmark with tracking that tightens.
        if lt > 0.55:
            q = ease_out(clamp((lt - 0.55) / 1.1))
            f = font(FONT_BOLD, 92)
            track = mix(26, 3, q)
            text = "OpenFocus"
            wdt = text_width(f, text, track)
            d = ImageDraw.Draw(img)
            text_tracked(d, (W / 2 - wdt / 2, H / 2 + 168), text, f,
                         INK + (int(255 * q),), track)
        if lt > 1.5:
            q = ease_out(clamp((lt - 1.5) / 1.0))
            d = ImageDraw.Draw(img)
            f = font(FONT_REG, 30)
            line = "OPEN-SOURCE  ·  MULTI-FOCUS  FUSION"
            wdt = text_width(f, line, 7)
            text_tracked(d, (W / 2 - wdt / 2, H / 2 + 246), line, f,
                         MUTED + (int(230 * q),), 7)
        return img

    def scene_tagline(self, lt):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        q = ease_out(clamp(lt / 0.9))
        lines = ["Every focus plane.", "One all-in-focus image."]
        f1 = font(FONT_BOLD, 74)
        f2 = font(FONT_BOLD, 74)
        y0 = H / 2 - 60
        d.text((W / 2, y0), lines[0], font=f1, fill=INK + (int(255 * q),), anchor="mm")
        q2 = ease_out(clamp((lt - 0.42) / 0.9))
        if q2 > 0:
            d.text((W / 2, y0 + 96), lines[1], font=f2,
                   fill=mixc(MUTED, GOLD, 0.55) + (int(255 * q2),), anchor="mm")
        # Thin rule that draws itself.
        q3 = ease_out(clamp((lt - 0.9) / 0.7))
        if q3 > 0:
            half = 210 * q3
            d.rectangle([W / 2 - half, y0 + 168, W / 2 + half, y0 + 171],
                        fill=ACCENT + (int(190 * q3),))
        return img

    def scene_stack(self, lt):
        """Nine real frames sweep, then wipe into the fused result."""
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        panel_w, panel_h = 1120, 760
        card = rounded_panel((panel_w + 8, panel_h + 8), 20,
                             fill=(16, 20, 28, 255), border=(255, 255, 255, 40))
        px, py = W / 2 - (panel_w + 8) / 2, H / 2 - (panel_h + 8) / 2 + 40

        # Sweep 0..1 over the first 3.4 s, then wipe 3.4..5.4.
        sweep = clamp(lt / 3.4)
        idx_f = sweep * (len(self.stack) - 1)
        i0 = int(math.floor(idx_f))
        i1 = min(len(self.stack) - 1, i0 + 1)
        blend = idx_f - i0
        a = self.stack[i0]
        b = self.stack[i1]
        frame = Image.blend(a, b, smooth(blend)) if a.size == b.size else a

        wipe = clamp((lt - 3.4) / 2.0)
        if wipe > 0:
            fused = self.fused.resize(frame.size, Image.LANCZOS)
            mask = Image.new("L", frame.size, 0)
            md = ImageDraw.Draw(mask)
            edge = int(frame.width * (1 - ease_in_out(wipe)))
            md.rectangle([edge, 0, frame.width, frame.height], fill=255)
            frame = Image.composite(fused, frame, mask)

        frame = frame.resize((panel_w, panel_h), Image.LANCZOS)
        inner = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 255))
        inner.paste(frame, (0, 0))
        inner = inner.filter(ImageFilter.GaussianBlur(0.3))
        card.alpha_composite(inner, (4, 4))
        img.alpha_composite(shadowed(card, 34, 160, (0, 26)), (int(px), int(py)))

        d = ImageDraw.Draw(img)
        # Frame counter, monospace, like a timecode.
        f = font(FONT_MONO, 24)
        if wipe < 0.5:
            label = f"FRAME {i0 + 1:02d} / 09"
            col = MUTED
        else:
            label = "ALL-IN-FOCUS RESULT"
            col = mixc(MUTED, GREEN, clamp((wipe - 0.5) * 3))
        d.text((px + 22, py + panel_h + 34), label, font=f, fill=col + (235,))

        # Focus-plane ticks under the card during the sweep.
        if wipe < 0.35:
            q = clamp(lt / 3.4)
            x0 = px + 22
            span = panel_w - 44
            d.rectangle([x0, py + panel_h + 74, x0 + span, py + panel_h + 76],
                        fill=(255, 255, 255, 26))
            d.rectangle([x0, py + panel_h + 73, x0 + span * q, py + panel_h + 77],
                        fill=ACCENT + (220,))
            d.ellipse([x0 + span * q - 6, py + panel_h + 69,
                       x0 + span * q + 6, py + panel_h + 81],
                      fill=INK + (240,))
        return img

    def scene_methods(self, lt):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        q = ease_out(clamp(lt / 0.9))
        idx = int(clamp(lt / 5.0) * (len(self.compare) - 1))
        frame = self.compare[idx]
        frame = fit(frame, 1620, 900)
        card = rounded_panel((frame.width + 8, frame.height + 8), 18,
                             fill=(16, 20, 28, 255), border=(255, 255, 255, 40))
        card.alpha_composite(frame.convert("RGBA"), (4, 4))
        # Entrance: rise + slight scale.
        sc = mix(0.97, 1.0, q)
        card = card.resize((int(card.width * sc), int(card.height * sc)), Image.LANCZOS)
        if q < 1:
            card.putalpha(card.getchannel("A").point(lambda v: int(v * q)))
        paste_center(img, shadowed(card, 32, 150, (0, 24)),
                     (W / 2, H / 2 + 30))
        return img

    def scene_features(self, lt, total=7.4):
        """Three cards cross-slide: depth, retouch, batch."""
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        items = [
            (self.depth, "DEPTH MAP & 3D", "focus position per pixel"),
            (self.retouch, "SOURCE-LOCKED RETOUCH", "paint from the frames that made it"),
            (self.batch, "BATCH PROCESSING", "one folder per stack"),
        ]
        per = total / len(items)
        for i, (frames, title, sub) in enumerate(items):
            start = i * per
            local = lt - start
            if local < -0.55 or local > per + 0.6:
                continue
            # Slide in, hold, slide out.
            if local < 0:
                continue
            enter = ease_out(clamp(local / 0.65))
            exit_p = ease_in_out(clamp((local - (per - 0.45)) / 0.6))
            alpha = enter * (1 - exit_p)
            if alpha <= 0.01:
                continue
            idx = int(clamp(local / max(0.01, per - 0.5)) * (len(frames) - 1))
            frame = frames[idx]
            frame = fit(frame, 880, 620)
            card = rounded_panel((frame.width + 8, frame.height + 8), 18,
                                 fill=(16, 20, 28, 255), border=(255, 255, 255, 40))
            card.alpha_composite(frame.convert("RGBA"), (4, 4))
            card.putalpha(card.getchannel("A").point(lambda v: int(v * alpha)))
            dx = (1 - enter) * 90 - exit_p * 90
            cx = W / 2 + 210 + dx
            paste_center(img, shadowed(card, 30, 140, (0, 22)), (cx, H / 2 + 40))

            d = ImageDraw.Draw(img)
            ta = int(255 * alpha)
            f = font(FONT_BOLD, 40)
            d.text((150, H / 2 - 60), title, font=f, fill=INK + (ta,))
            f2 = font(FONT_REG, 28)
            d.text((150, H / 2 + 2), sub, font=f2, fill=MUTED + (ta,))
            d.rectangle([150, H / 2 - 104, 150 + 54, H / 2 - 100],
                        fill=ACCENT + (ta,))
        return img

    def scene_stats(self, lt):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        stats = [("8", "FUSION METHODS"), ("16", "BIT END TO END"),
                 ("12", "UI LANGUAGES"), ("0", "COST · MIT")]
        n = len(stats)
        f_num = font(FONT_BOLD, 132)
        f_lab = font(FONT_BOLD, 21)
        span = W - 300
        for i, (num, lab) in enumerate(stats):
            start = i * 0.34
            q = ease_out_expo(clamp((lt - start) / 0.9))
            if q <= 0:
                continue
            cx = 150 + span * (i + 0.5) / n
            rise = (1 - q) * 44
            a = int(255 * clamp((lt - start) / 0.5))
            d.text((cx, H / 2 - 10 + rise), num, font=f_num,
                   fill=INK + (a,), anchor="mm")
            lw = text_width(f_lab, lab, 4)
            text_tracked(d, (cx - lw / 2, H / 2 + 92 + rise), lab, f_lab,
                         MUTED + (a,), 4)
            if i < n - 1:
                d.rectangle([150 + span * (i + 1) / n - 1, H / 2 - 90,
                             150 + span * (i + 1) / n, H / 2 + 110],
                            fill=(255, 255, 255, int(26 * q)))
        return img

    def scene_cli(self, lt):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        q = ease_out(clamp(lt / 0.7))
        box_w, box_h = 1180, 460
        card = rounded_panel((box_w, box_h), 16, fill=(13, 16, 22, 246),
                             border=(255, 255, 255, 38))
        d = ImageDraw.Draw(card)
        for i, col in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            d.ellipse([24 + i * 26, 22, 38 + i * 26, 36], fill=col + (230,))

        lines = [
            ("$ ", "pip install openfocus", 0.35),
            ("", "Installing collected packages: torch, opencv-python-headless, ...", 1.0),
            ("", "Successfully installed openfocus-1.68.1", 1.75),
            ("$ ", "openfocus -i ./stack -o fused.tif -m stackmffv3", 2.35),
            ("", "Aligning 9 frames (ECC) .......... done", 2.95),
            ("", "Fusing 1190x790  tiled 2/2 ....... done  5.7s", 3.3),
            ("", "Wrote fused.tif  (16-bit, 1200x797)", 3.6),
        ]
        fm = font(FONT_MONO_R, 25)
        y = 84
        for prompt, body, start in lines:
            local = lt - start
            if local < 0:
                break
            # 0.14 s per character keeps every line complete before the next
            # one starts (the gaps above are 0.35-0.65 s).
            typed = int(local / 0.014)
            shown = body[: max(0, min(len(body), typed))]
            x = 44
            if prompt:
                d.text((x, y), prompt, font=fm, fill=GREEN + (255,))
                x += fm.getlength(prompt)
            col = INK if prompt else MUTED
            d.text((x, y), shown, font=fm, fill=col + (240,))
            caret_x = x + fm.getlength(shown)
            y += 46
        # Blinking caret right after the last typed character.
        if int(lt * 2.2) % 2 == 0 and y > 84:
            d.rectangle([caret_x + 3, y - 74, caret_x + 15, y - 32],
                        fill=INK + (200,))

        card.putalpha(card.getchannel("A").point(lambda v: int(v * q)))
        paste_center(img, shadowed(card, 30, 160, (0, 22)), (W / 2, H / 2 + 10))
        return img

    def scene_end(self, lt):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        q = ease_out(clamp(lt / 0.9))
        g = self.glow_gold.copy()
        ga = int(190 * q)
        g.putalpha(g.getchannel("A").point(lambda v: v * ga // 255))
        paste_center(img, g, (W / 2, H / 2 - 20))
        lg = self.logo.resize((190, 190), Image.LANCZOS)
        lg.putalpha(lg.getchannel("A").point(lambda v: int(v * q)))
        paste_center(img, lg, (W / 2, H / 2 - 190))
        d = ImageDraw.Draw(img)
        f = font(FONT_BOLD, 68)
        d.text((W / 2, H / 2 - 40), "OpenFocus", font=f,
               fill=INK + (int(255 * q),), anchor="mm")
        q2 = ease_out(clamp((lt - 0.35) / 0.8))
        if q2 > 0:
            f2 = font(FONT_REG, 31)
            d.text((W / 2, H / 2 + 30), "Free. Open source. MIT.",
                   font=f2, fill=(206, 214, 226) + (int(250 * q2),), anchor="mm")
        q3 = ease_out(clamp((lt - 0.7) / 0.8))
        if q3 > 0:
            f3 = font(FONT_BOLD, 34)
            d.text((W / 2, H / 2 + 106), "github.com/Xinzhe99/OpenFocus",
                   font=f3, fill=ACCENT + (int(255 * q3),), anchor="mm")
        q4 = ease_out(clamp((lt - 1.1) / 0.8))
        if q4 > 0:
            f4 = font(FONT_REG, 25)
            d.text((W / 2, H / 2 + 168),
                   "Windows 10/11  ·  macOS Apple Silicon  ·  pip install openfocus",
                   font=f4, fill=(158, 170, 188) + (int(250 * q4),), anchor="mm")
        return img

    # ------------------------------------------------------------ compose
    def frame(self, t):
        img = self.background(t)
        # Scene windows with overlapping cross-fades.
        scenes = [
            (0.0, 4.0, self.scene_logo),
            (3.2, 3.6, self.scene_tagline),
            (6.2, 6.6, self.scene_stack),
            (12.2, 6.0, self.scene_methods),
            (17.8, 7.4, lambda lt: self.scene_features(lt)),
            (24.8, 4.8, self.scene_stats),
            (28.2, 6.2, self.scene_cli),
            (34.0, 4.0, self.scene_end),
        ]
        for start, length, fn in scenes:
            if start <= t <= start + length:
                local = t - start
                layer = fn(local)
                if layer is None:
                    continue
                fade_out = clamp((length - local) / 0.45)
                if fade_out < 1.0:
                    layer.putalpha(layer.getchannel("A").point(
                        lambda v: int(v * fade_out)))
                img.alpha_composite(layer)
        img.alpha_composite(self.vig)
        g = self.grain[int(t * FPS) % len(self.grain)]
        img.alpha_composite(g)
        return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets" / "launch.mp4"))
    ap.add_argument("--preview", type=int, default=0,
                    help="render only N stills for inspection")
    args = ap.parse_args()

    film = Film()
    total = 38.0
    if args.preview:
        out_dir = Path(tempfile.mkdtemp(prefix="of_film_"))
        for i in range(args.preview):
            t = total * i / max(1, args.preview - 1)
            film.frame(t).save(out_dir / f"p{i:03d}.png")
        print("preview frames:", out_dir)
        return

    out_dir = Path(tempfile.mkdtemp(prefix="of_film_"))
    frames = int(total * FPS)
    print(f"rendering {frames} frames at {W}x{H}...")
    for i in range(frames):
        film.frame(i / FPS).save(out_dir / f"f{i:05d}.png")
        if i % 90 == 0:
            print(f"  {i}/{frames}", flush=True)

    audio = out_dir / "score.wav"
    synth_audio(total, audio)

    ffmpeg = shutil.which("ffmpeg") or r"D:\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-framerate", str(FPS), "-i", str(out_dir / "f%05d.png"),
           "-i", str(audio),
           "-c:v", "libx264", "-preset", "slow", "-crf", "19",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
           "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)
    print("wrote", args.out, Path(args.out).stat().st_size // 1024 // 1024, "MB")
    shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
