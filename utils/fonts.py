"""System-font resolution shared by labels and scale bars.

Both features draw user-visible text into image pixels, so they must agree on
how a requested family is resolved and must be able to report when the
request could not be honoured. Guessing a file name from the family
(``Arial.ttf``) silently substituted another typeface for every family the
guess missed, which is invisible in the UI and wrong in the export.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

# Fallbacks used when the requested family cannot be found at all. Order is
# deliberate: a font with CJK coverage first on Windows so a Chinese caption
# never degrades into tofu boxes.
_FALLBACK_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)

_FONT_SUFFIXES = (".ttf", ".ttc", ".otf")
_MAX_SCANNED_FONTS = 600

_font_cache: Dict[tuple, Tuple[Optional[object], str]] = {}
_family_index: Optional[Dict[str, str]] = None


def _font_dirs() -> List[str]:
    """Directories holding installed fonts on this platform."""
    dirs: List[str] = []
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(os.path.join(windir, "Fonts"))
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    elif sys.platform == "darwin":
        dirs += ["/System/Library/Fonts", "/Library/Fonts",
                 os.path.expanduser("~/Library/Fonts")]
    else:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts",
                 os.path.expanduser("~/.fonts"),
                 os.path.expanduser("~/.local/share/fonts")]
    return [d for d in dirs if os.path.isdir(d)]


def _pil_font():
    try:
        from PIL import ImageFont
        return ImageFont
    except Exception:
        return None


def _scan_family_index() -> Dict[str, str]:
    """Map lowercase family name -> font file, by reading each font's names.

    PIL on Windows resolves some families through GDI but not others (CJK
    families such as Microsoft YaHei commonly fail by bare name). Reading the
    name table gives an exact answer on every platform, and the result is
    cached for the process lifetime.
    """
    global _family_index
    if _family_index is not None:
        return _family_index
    index: Dict[str, str] = {}
    ImageFont = _pil_font()
    if ImageFont is not None:
        scanned = 0
        for directory in _font_dirs():
            try:
                names = sorted(os.listdir(directory))
            except OSError:
                continue
            for name in names:
                if not name.lower().endswith(_FONT_SUFFIXES):
                    continue
                path = os.path.join(directory, name)
                scanned += 1
                if scanned > _MAX_SCANNED_FONTS:
                    break
                try:
                    face = ImageFont.truetype(path, 12)
                    family, style = face.getname()
                except Exception:
                    continue
                key = (family or "").strip().lower()
                if not key:
                    continue
                bold = "bold" in (style or "").lower()
                # First regular face wins; a bold-only family still registers.
                if key not in index or not bold:
                    index.setdefault(key, path)
                    if not bold:
                        index[key] = path
    _family_index = index
    return index


def _fallback_font(size: int):
    """First readable fallback face, with its family name."""
    ImageFont = _pil_font()
    if ImageFont is None:
        return None, ""
    for path in _FALLBACK_CANDIDATES:
        try:
            face = ImageFont.truetype(path, size)
        except Exception:
            continue
        try:
            family = face.getname()[0]
        except Exception:
            family = os.path.splitext(os.path.basename(path))[0]
        return face, family
    return None, ""


def load_font(size: int, family: str = None,
              bold: bool = False) -> Tuple[Optional[object], str]:
    """Return ``(PIL font, actual family name)`` for the requested style.

    The actual name is ``""`` when no font could be loaded. Callers surface a
    substitution when it differs from the request instead of pretending the
    requested family was used.
    """
    wanted = (family or "").strip()
    key = (int(size), wanted.lower(), bool(bold))
    cached = _font_cache.get(key)
    if cached is not None:
        return cached

    ImageFont = _pil_font()
    face = None
    actual = ""
    if ImageFont is not None and wanted:
        index = _scan_family_index()
        path = index.get(wanted.lower())
        if path:
            try:
                face = ImageFont.truetype(path, size)
                actual = wanted
            except Exception:
                face = None
        if face is None:
            # Some platforms (macOS, Linux fontconfig) resolve a bare family.
            try:
                face = ImageFont.truetype(wanted, size)
                actual = wanted
            except Exception:
                face = None
    if face is None:
        face, actual = _fallback_font(size)
    if face is None and ImageFont is not None:
        try:
            face = ImageFont.load_default()
            actual = "default"
        except Exception:
            face, actual = None, ""

    result = (face, actual)
    _font_cache[key] = result
    return result


_coverage_cache: Dict[tuple, bool] = {}


def _glyph_signature(font, char: str):
    """Bitmap of one character, or None when it cannot be rendered."""
    try:
        mask = font.getmask(char)
        return (mask.size, bytes(mask))
    except Exception:
        return None


def covers_text(font, text: str) -> bool:
    """True when every non-space character has a real glyph in `font`.

    FreeType substitutes the .notdef box for missing characters and Pillow
    exposes no coverage API, so each character is compared against a private
    use code point that no normal font defines. A CJK caption drawn with a
    Latin-only face therefore reads as boxes; this detects that instead of
    silently exporting boxes into the image.
    """
    if font is None or not text:
        return True
    key = (id(font), text)
    cached = _coverage_cache.get(key)
    if cached is not None:
        return cached
    sentinel = _glyph_signature(font, "\ue000")
    ok = True
    for char in set(text):
        if char.isspace():
            continue
        signature = _glyph_signature(font, char)
        if signature is None:
            ok = False
            break
        if sentinel is not None and signature == sentinel:
            ok = False
            break
    _coverage_cache[key] = ok
    return ok


def _cjk_capable_candidates():
    """Fallback faces that usually carry CJK glyphs, best first."""
    return (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simsun.ttf",
        "C:/Windows/Fonts/msgothic.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )


def load_font_for_text(size: int, family: str, text: str,
                       bold: bool = False):
    """Resolve a font that can actually draw `text`.

    Returns ``(font, actual_family, substituted)``. When the requested family
    lacks glyphs for the caption (a Chinese title in Arial, for example) a
    covering face is used and ``substituted`` is the requested name, so the UI
    can say which family was replaced instead of showing boxes.
    """
    font, actual = load_font(size, family, bold)
    if font is None or covers_text(font, text):
        return font, actual, ""
    for path in _cjk_capable_candidates():
        try:
            from PIL import ImageFont
            candidate = ImageFont.truetype(path, size)
        except Exception:
            continue
        if not covers_text(candidate, text):
            continue
        try:
            candidate_family = candidate.getname()[0]
        except Exception:
            candidate_family = os.path.splitext(os.path.basename(path))[0]
        return candidate, candidate_family, (family or "").strip()
    return font, actual, ""


def clear_font_cache() -> None:
    """Drop cached faces (used by tests and after a font install)."""
    _coverage_cache.clear()
    """Drop cached faces (used by tests and after a font install)."""
    global _family_index
    _font_cache.clear()
    _family_index = None
