"""Report closed community GitHub issues missing from site/contributors.json.

Stdlib-only helper for release maintainers:

    python tools/sync_contributors.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = "Xinzhe99/OpenFocus"
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "site" / "contributors.json"


def load_recorded() -> tuple[set[str], set[int]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    excluded = {item.lower() for item in data.get("excluded_authors", [])}
    recorded: set[int] = set()
    for contributor in data.get("contributors", []):
        for entry in contributor.get("contributions", []):
            recorded.add(int(entry["issue"]))
    return excluded, recorded


def fetch_closed_issues() -> list[dict]:
    cmd = [
        "gh",
        "issue",
        "list",
        "-R",
        REPO,
        "--state",
        "closed",
        "--limit",
        "200",
        "--json",
        "number,title,author,closedAt,url",
    ]
    raw = subprocess.check_output(cmd, text=True, encoding="utf-8")
    return json.loads(raw)


def main() -> int:
    excluded, recorded = load_recorded()
    try:
        issues = fetch_closed_issues()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Failed to query GitHub issues via gh CLI: {exc}", file=sys.stderr)
        return 1

    missing = []
    for issue in issues:
        login = ((issue.get("author") or {}).get("login") or "").strip()
        if not login or login.lower() in excluded or login.lower().endswith("[bot]"):
            continue
        number = int(issue["number"])
        if number not in recorded:
            missing.append((number, login, (issue.get("closedAt") or "")[:10], issue.get("title", "")))

    if not missing:
        print("site/contributors.json is up to date with all closed external GitHub issues.")
        return 0

    print(f"Found {len(missing)} closed external issue(s) not yet in site/contributors.json:")
    for number, login, closed_date, title in sorted(missing, reverse=True):
        print(f"  #{number}  @{login}  ({closed_date})  {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
