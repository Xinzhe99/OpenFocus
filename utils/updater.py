"""Release update checker backed by the GitHub releases API.

Runs network calls on daemon threads and reports back through callbacks so
the caller (UI) can decide how to present progress and results.
"""
import json
import re
import sys
import threading
import urllib.request
from typing import Callable, Optional, Tuple

RELEASES_API_URL = "https://api.github.com/repos/Xinzhe99/OpenFocus/releases/latest"
# The LIST endpoint (newest first) — "/latest?per_page=N" still returns one
# release, so the per-version notes need this URL instead.
RELEASES_LIST_API_URL = ("https://api.github.com/repos/Xinzhe99/OpenFocus"
                         "/releases")
RELEASES_PAGE_URL = "https://github.com/Xinzhe99/OpenFocus/releases"
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


def _asset_fields(a: dict) -> dict:
    """Normalize one API asset entry (name/url/size plus its digest)."""
    return {"name": a.get("name", ""), "url": a.get("browser_download_url", ""),
            "size": int(a.get("size", 0)),
            "digest": str(a.get("digest", "") or "")}


def find_asset_by_keyword(release: dict, keyword: str) -> Optional[dict]:
    assets = release.get("assets", []) or []
    for a in assets:
        name = str(a.get("name", "")).lower()
        if keyword in name:
            return _asset_fields(a)
    return None


def _find_zip_asset(release: dict, keywords) -> Optional[dict]:
    """First zip asset whose name contains one of keywords (best match first)."""
    zips = [a for a in (release.get("assets") or [])
            if str(a.get("name", "")).lower().endswith(".zip")]
    for keyword in keywords:
        for a in zips:
            if keyword in str(a.get("name", "")).lower():
                return _asset_fields(a)
    return None


def find_portable_zip_asset(release: dict) -> Optional[dict]:
    """Platform portable zip used by the in-app one-click update flow."""
    if sys.platform == "win32":
        return _find_zip_asset(release, ("windows-x64.zip",))
    if sys.platform == "darwin":
        import platform
        # Only arm64 builds are published. Matching the exact arch keeps an
        # Intel Mac from swapping in a binary it cannot launch.
        arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"
        return _find_zip_asset(release, (f"macos-{arch}.zip",))
    return None


def find_installer_asset(release: dict) -> Optional[dict]:
    """Pick the right installer asset for this platform from a release dict."""
    if sys.platform == "win32":
        return find_asset_by_keyword(release, "setup.exe")
    if sys.platform == "darwin":
        import platform
        arch = platform.machine().lower()
        if arch in ("arm64", "aarch64"):
            return (find_asset_by_keyword(release, "macos-arm64.dmg")
                    or find_asset_by_keyword(release, "macos.dmg"))
        return find_asset_by_keyword(release, "macos-x86_64.dmg")
    return None


def fetch_latest_release() -> Tuple[bool, str, str, Optional[dict], Optional[dict]]:
    """Return (ok, tag_name, page_url, installer_asset, portable_zip) where
    portable_zip is the platform portable-zip asset used by the one-click
    self-update flow."""

    req = urllib.request.Request(
        RELEASES_API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "OpenFocus"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    tag = str(data.get("tag_name", "")).strip()
    url = str(data.get("html_url", "")) or RELEASES_PAGE_URL
    if not tag:
        return False, "", url, None, None
    return True, tag, url, find_installer_asset(data), find_portable_zip_asset(data)


def _md_to_plain(md: str) -> str:
    """Tiny markdown -> plain text for the update dialog's rich label.

    GitHub release bodies are markdown; QLabel(setText) would show the
    raw source (##/```/| tables) which reads as garbage. This is not a
    full renderer: headings/lists/inline code/table pipes are neutralized,
    line breaks and indentation kept.
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
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)  # links
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
        headers={"Accept": "application/vnd.github+json", "User-Agent": "OpenFocus"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_release_notes(current_version: str, latest_tag: str,
                        max_releases: int = 10, timeout: int = 3) -> str:
    """Release notes for every version newer than the installed one.

    Best-effort and cheap: bounded by ``timeout`` (seconds) per call and at
    most two requests, so a slow/offline network returns '' quickly and the
    update notification is never held up by the notes fetch. Lists the
    GitHub releases (newest first), keeps everything strictly newer than
    ``current_version`` (cap ``max_releases``), renders each body and joins
    with ``## tag`` headings. Returns '' when nothing could be fetched.
    """
    latest = parse_version(latest_tag) or ()
    if not latest:
        return ""
    current = parse_version(current_version) or ()
    try:
        data = _api_get(RELEASES_LIST_API_URL + "?per_page=%d"
                        % max(3, min(30, max_releases + 2)), timeout)
    except Exception:
        return ""
    blocks = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag_name", "")).strip()
        ver = parse_version(tag)
        if not tag or not ver:
            continue
        if current and ver <= current:
            continue
        if latest and ver > latest:
            continue
        body = str(item.get("body", "") or "").strip()
        if not body:
            continue
        blocks.append("## " + tag.lstrip("vV") + "\n\n" + _md_to_plain(body))
        if len(blocks) >= max_releases:
            break
    return ("\n\n".join(blocks))[:12000]


def _notes_text(tag: str) -> str:
    """Best-effort notes for a just-published release; '' on any failure."""
    from constants import APP_VERSION
    try:
        return fetch_release_notes(APP_VERSION, tag)
    except Exception:
        return ""


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
    """Check a downloaded file against a GitHub asset digest ("sha256:<hex>").

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

    For the "update" state two extra arguments follow the page URL: the
    portable-zip URL and the SHA-256 digest GitHub publishes for that asset,
    so the caller can verify the download instead of swapping in whatever a
    proxy handed it; a seventh argument carries the plain-text release
    notes of every version between the installed one and the latest.
    """

    def worker():
        try:
            ok, tag, url, asset, portable_zip = fetch_latest_release()
            setup_url = (asset or {}).get("url", "")
            zip_url = (portable_zip or {}).get("url", "")
            zip_digest = (portable_zip or {}).get("digest", "")
            if ok and is_newer(tag, current_version):
                notes = ""
                try:
                    notes = fetch_release_notes(current_version, tag)
                except Exception:
                    notes = ""
                on_result("update", tag, url, zip_url, setup_url, zip_digest,
                          notes)
            elif not quiet:
                on_result("latest", tag or "", url)
        except Exception:
            if not quiet:
                on_result("error", "", RELEASES_PAGE_URL)

    threading.Thread(target=worker, daemon=True, name="update-check").start()
