"""Release update checker backed by the OpenFocus download mirror.

Runs network calls on daemon threads and reports back through callbacks so
the caller (UI) can decide how to present progress and results.
"""
import json
import re
import sys
import threading
import urllib.request
from typing import Callable, Optional, Tuple
from utils.release_notes import LANGUAGES, RELEASE_NOTES, localized_bodies

RELEASES_API_URL = "https://openfocus.top/api/updates"
RELEASES_PAGE_URL = "https://openfocus.top/#download"
RELEASE_DOWNLOAD_URL = "https://openfocus.top/download/releases"
_ASSET_SUFFIXES = {
    "windows-setup": "setup.exe",
    "windows-zip": "windows-x64.zip",
    "macos-dmg": "macos.dmg",
    "macos-zip": "macos-arm64.zip",
}
REQUEST_TIMEOUT_S = 10


def parse_version(tag: str) -> Optional[Tuple[int, ...]]:
    """Turn 'v1.10' / '1.10.2' into a comparable tuple; None if unparsable."""
    tag = tag.strip().lstrip("vV")
    parts = tag.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def is_newer(latest_tag: str, current_version: str) -> bool:
    latest = parse_version(latest_tag)
    current = parse_version(current_version)
    if latest is None or current is None:
        return False
    width = max(len(latest), len(current))
    latest += (0,) * (width - len(latest))
    current += (0,) * (width - len(current))
    return latest > current


def _mirror_asset(release: dict, kind: str) -> Optional[dict]:
    """Select a version-pinned mirror asset; never use remote URLs in metadata."""
    if not isinstance(release, dict):
        return None
    tag = release.get("tag")
    if not isinstance(tag, str) or not re.fullmatch(r"v[0-9]+(?:\.[0-9]+)*", tag):
        return None
    suffix = _ASSET_SUFFIXES.get(kind)
    assets = release.get("assets")
    asset = assets.get(kind) if isinstance(assets, dict) else None
    if not suffix or not isinstance(asset, dict):
        return None
    name = f"OpenFocus-{tag}-{suffix}"
    size = asset.get("size")
    if asset.get("name") != name or type(size) is not int or size <= 0:
        return None
    digest = str(asset.get("digest") or "")
    # Never downgrade self-update integrity when an old or incomplete mirror
    # manifest lacks the SHA-256 supplied by the release workflow.
    if kind.endswith("-zip") and not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
        return None
    return {"name": name, "url": f"{RELEASE_DOWNLOAD_URL}/{tag}/{kind}",
            "size": size, "digest": digest}


def find_portable_zip_asset(release: dict) -> Optional[dict]:
    """Platform portable zip used by the in-app one-click update flow."""
    if sys.platform == "win32":
        return _mirror_asset(release, "windows-zip")
    if sys.platform == "darwin":
        import platform
        if platform.machine().lower() in ("arm64", "aarch64"):
            return _mirror_asset(release, "macos-zip")
    return None


def find_installer_asset(release: dict) -> Optional[dict]:
    """Pick the installer for this platform from mirrored release metadata."""
    if sys.platform == "win32":
        return _mirror_asset(release, "windows-setup")
    if sys.platform == "darwin":
        import platform
        if platform.machine().lower() in ("arm64", "aarch64"):
            return _mirror_asset(release, "macos-dmg")
    return None


def fetch_latest_release() -> Tuple[bool, str, str, Optional[dict], Optional[dict], list]:
    """Return the release, platform assets, and notes from one mirror request."""
    data = _api_get(RELEASES_API_URL)
    tag = data.get("tag") if isinstance(data, dict) else None
    if not isinstance(tag, str) or not re.fullmatch(r"v[0-9]+(?:\.[0-9]+)*", tag):
        raise ValueError("Invalid update metadata")
    releases = data.get("releases")
    if not isinstance(releases, list):
        releases = []
    return (True, tag, RELEASES_PAGE_URL, find_installer_asset(data),
            find_portable_zip_asset(data), releases)


def _md_to_plain(md: str) -> str:
    """Tiny markdown -> plain text for the update dialog's rich label.

    Release notes are markdown; QLabel(setText) would show the raw source.
    This is not a full renderer: headings/lists/inline code/table pipes are
    neutralized, line breaks and indentation kept.
    """
    out = []
    in_code = False
    for line in (md or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            out.append("    " + line.strip())
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)   # ATX headings
        line = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", line)  # inline code
        line = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r"\1 (\2)", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)        # bold
        line = re.sub(r"__([^_]+)__", r"\1", line)            # bold (alt)
        line = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", line)  # italic
        line = re.sub(r"^\s*[-*+]\s+", "• ", line)      # bullets
        line = re.sub(r"^\s*\d+\.\s+", lambda m: m.group().strip(), line)
        line = re.sub(r"^\s*>\s?", "", line)            # quotes
        line = line.replace("|", " ")                   # tables
        line = re.sub(r" {2,}", "  ", line)
        out.append(line)
    text = "\n".join(out).strip()
    return text if text else (md or "").strip()


def _api_get(url: str, timeout: int = REQUEST_TIMEOUT_S) -> Optional[dict]:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "OpenFocus"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _note_version(tag: str) -> tuple:
    version = parse_version(tag) or ()
    while len(version) > 1 and version[-1] == 0:
        version = version[:-1]
    return version


def release_note_entries(current_version: str, latest_tag: str, releases,
                         max_releases: int = 10) -> list:
    """Keep bounded, sorted language-neutral notes for later UI selection.

    The mirror may contain explicit ``body_i18n`` dictionaries or GitHub
    language blocks. Bundled translations cover the current shipped release
    even when an older mirror still has only its English GitHub body.
    """
    if not parse_version(latest_tag) or max_releases <= 0:
        return []
    candidates = {}
    bundled_by_version = {_note_version(version): bodies
                          for version, bodies in RELEASE_NOTES.items()}
    for item in releases if isinstance(releases, list) else []:
        if not isinstance(item, dict):
            continue
        tag = item.get("tag_name")
        if not isinstance(tag, str) or not re.fullmatch(r"[vV]?[0-9]+(?:\.[0-9]+)*", tag):
            continue
        if ((parse_version(current_version) and not is_newer(tag, current_version))
                or is_newer(tag, latest_tag)):
            continue
        body = item.get("body", "")
        body = body if isinstance(body, str) else ""
        translations = localized_bodies(body)
        supplied = item.get("body_i18n")
        if isinstance(supplied, dict):
            translations.update({lang: text.strip() for lang, text in supplied.items()
                                 if lang in LANGUAGES and isinstance(text, str)
                                 and text.strip()})
        bundled = bundled_by_version.get(_note_version(tag), {})
        translations = {**bundled, **translations}
        # Never expose other languages' blocks as the legacy fallback body.
        if localized_bodies(body):
            body = translations.get("en", "")
        entry = {"tag_name": tag, "body": body.strip()[:12000],
                 "body_i18n": {lang: text[:12000]
                               for lang, text in translations.items()}}
        key = _note_version(tag)
        if entry["body"] or entry["body_i18n"]:
            candidates.setdefault(key, entry)
    for version, translations in RELEASE_NOTES.items():
        if ((parse_version(current_version) and not is_newer(version, current_version))
                or is_newer(version, latest_tag)):
            continue
        key = _note_version(version)
        candidates.setdefault(key, {"tag_name": "v" + version, "body": "",
                                    "body_i18n": dict(translations)})
    return [candidates[key] for key in sorted(candidates, reverse=True)
            if candidates[key]["body"] or candidates[key]["body_i18n"]][:max_releases]


def format_release_notes(current_version: str, latest_tag: str, releases,
                         language: str = "en", max_releases: int = 10) -> str:
    """Render retained release data without network access or Qt imports."""
    language = str(language).lower().replace("-", "_").split("_")[0]
    blocks = []
    for item in release_note_entries(current_version, latest_tag, releases, max_releases):
        bodies = item["body_i18n"]
        body = bodies.get(language) or bodies.get("en") or item["body"]
        if body:
            source = "https://github.com/Xinzhe99/OpenFocus/releases/tag/v" + item["tag_name"].lstrip("vV")
            blocks.append(item["tag_name"].lstrip("vV") + "\n" + source
                          + "\n\n" + _md_to_plain(body))
    return "\n\n".join(blocks)[:12000]


def fetch_release_notes(current_version: str, latest_tag: str,
                        max_releases: int = 10, timeout: int = 3,
                        releases=None, language: str = "en") -> str:
    """Fetch if necessary, then select actual text in the requested language."""
    if not parse_version(latest_tag) or max_releases <= 0:
        return ""
    if releases is None:
        try:
            data = _api_get(RELEASES_API_URL, timeout)
        except Exception:
            return ""
        releases = data.get("releases", []) if isinstance(data, dict) else []
    return format_release_notes(current_version, latest_tag, releases,
                                language, max_releases)


class DownloadCancelled(Exception):
    """Raised out of download_to_file when should_cancel() asked to stop."""


def download_to_file(url: str, dest_path: str,
                     on_progress: Callable[[int, int], None] = None,
                     should_cancel: Callable[[], bool] = None) -> bool:
    """Stream a download to dest_path on the CALLING thread (blocking).

    Intended to run inside a QThread. on_progress(received, total) fires
    after every chunk in this thread — wrap it in a signal before touching
    any UI. Raises on network/IO errors.

    should_cancel() is polled between chunks (a background update must not
    pin a thread for the whole ~350 MB just because the user quit or hit
    cancel). A cancelled download removes its .part file and raises
    DownloadCancelled, so a half file can never be promoted.
    """
    import os
    part_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "OpenFocus"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            done = 0
            with open(part_path, "wb") as f:
                while True:
                    if should_cancel is not None and should_cancel():
                        raise DownloadCancelled("download cancelled")
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        try:
                            on_progress(done, total)
                        except Exception:
                            pass
        # 无 Content-Length 的响应 total=0，跳过校验；否则字节数必须吻合，
        # 代理/杀软截断会"干净地"提前 EOF，晋升半截 zip 只会在解压时白费
        if total and done != total:
            raise RuntimeError(
                f"download truncated: {done} of {total} bytes")
        if should_cancel is not None and should_cancel():
            raise DownloadCancelled("download cancelled")
        os.replace(part_path, dest_path)
    except BaseException:
        # ANY failure (socket reset, disk full, truncation, cancel) must take
        # the partial file with it: a leftover .part only wasted disk and made
        # the next attempt look like it had already downloaded something.
        try:
            os.unlink(part_path)
        except OSError:
            pass
        raise
    return True


def verify_file_digest(path: str, digest: str,
                       should_cancel: Callable[[], bool] = None) -> Optional[bool]:
    """Check a downloaded file against its published digest ("sha256:<hex>").

    Returns True/False on a definite result, None when no usable digest was
    published (older releases) — callers skip verification then rather than
    blocking the update.
    """
    import hashlib
    import re
    digest = str(digest or "").strip()
    if not digest:
        return None
    m = re.match(r"^sha256[:=]?([0-9a-fA-F]{64})$", digest)
    bare = re.match(r"^([0-9a-fA-F]{64})$", digest)
    if m:
        expected = m.group(1).lower()
    elif bare:
        expected = bare.group(1).lower()
    else:
        return None  # unknown algorithm (sha384/md5/…): skip, don't fail
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if should_cancel is not None and should_cancel():
                raise DownloadCancelled("verification cancelled")
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest() == expected


def download_async(url: str, dest_path: str,
                   on_progress: Callable[[int, int], None],
                   on_done: Callable[[bool, str], None]) -> None:
    """Daemon-thread wrapper kept for non-UI callers.

    UI code must prefer download_to_file inside a QThread: the callbacks
    here run on the worker thread and may not touch widgets.
    """

    def worker():
        try:
            download_to_file(url, dest_path, on_progress)
            on_done(True, dest_path)
        except Exception as exc:
            on_done(False, str(exc))

    threading.Thread(target=worker, daemon=True, name="update-download").start()


def check_async(current_version: str, on_result: Callable[..., None], quiet: bool = False) -> None:
    """Check for updates off-thread.

    on_result receives (state, tag, download_url) where state is:
      - "update": a newer release exists (tag holds the new version)
      - "latest": the current version is the newest
      - "error":  the check could not be completed (offline, API failure)

    With quiet=True the "latest" and "error" states are not reported (used
    for the automatic startup check, which must stay silent unless there is
    something to install).

    For an update the callback also receives the mirror ZIP and installer
    URLs, the ZIP's SHA-256 digest, and language-neutral release-note entries.
    """

    def worker():
        try:
            ok, tag, url, asset, portable_zip, releases = fetch_latest_release()
            if not ok:
                raise ValueError("Invalid update metadata")
            setup_url = (asset or {}).get("url", "")
            zip_url = (portable_zip or {}).get("url", "")
            zip_digest = (portable_zip or {}).get("digest", "")
            if ok and is_newer(tag, current_version):
                notes = release_note_entries(current_version, tag, releases)
                on_result("update", tag, url, zip_url, setup_url, zip_digest,
                          notes)
            elif not quiet:
                on_result("latest", tag or "", url)
        except Exception:
            if not quiet:
                on_result("error", "", RELEASES_PAGE_URL)

    threading.Thread(target=worker, daemon=True, name="update-check").start()
