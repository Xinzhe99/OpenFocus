"""Release update checker backed by the GitHub releases API.

Runs the network call on a daemon thread and reports back through a
callback so the caller (UI) can decide how to present the result.
"""
import json
import threading
import urllib.request
from typing import Callable, Optional, Tuple

RELEASES_API_URL = "https://api.github.com/repos/Xinzhe99/OpenFocus/releases/latest"
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


def fetch_latest_release() -> Tuple[bool, str, str]:
    """Return (ok, tag_name, html_url) of the latest published release."""
    req = urllib.request.Request(
        RELEASES_API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "OpenFocus"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    tag = str(data.get("tag_name", "")).strip()
    url = str(data.get("html_url", "")) or RELEASES_PAGE_URL
    if not tag:
        return False, "", url
    return True, tag, url


def check_async(current_version: str, on_result: Callable[[str, str, str], None], quiet: bool = False) -> None:
    """Check for updates off-thread.

    on_result receives (state, tag, download_url) where state is:
      - "update": a newer release exists (tag holds the new version)
      - "latest": the current version is the newest
      - "error":  the check could not be completed (offline, API failure)

    With quiet=True the "latest" and "error" states are not reported (used
    for the automatic startup check, which must stay silent unless there is
    something to install).
    """

    def worker():
        try:
            ok, tag, url = fetch_latest_release()
            if ok and is_newer(tag, current_version):
                on_result("update", tag, url)
            elif not quiet:
                on_result("latest", tag or "", url)
        except Exception:
            if not quiet:
                on_result("error", "", RELEASES_PAGE_URL)

    threading.Thread(target=worker, daemon=True, name="update-check").start()
