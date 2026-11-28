"""Synchronize site/contributors.json with closed community GitHub issues.

Stdlib-only helper for release maintainers and GitHub Actions:

    python tools/sync_contributors.py             # check for updates (read-only)
    python tools/sync_contributors.py --write     # update site/contributors.json
    python tools/sync_contributors.py --validate  # validate schema offline
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "Xinzhe99/OpenFocus"
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "site" / "contributors.json"
DEFAULT_EXCLUDED = ["Xinzhe99"]
STABLE_TAG_RE = re.compile(r"^v[0-9]+(?:\.[0-9]+)*$")
GITHUB_HANDLE_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")


def load_data(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_issue_title(title: str) -> str:
    text = re.sub(r"\s+", " ", str(title or "")).strip()
    while True:
        stripped = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
        if stripped == text:
            break
        text = stripped
    return text or "Community feedback"


def infer_kind(issue: dict) -> str:
    labels = [
        str(lbl.get("name", "")).lower()
        for lbl in (issue.get("labels") or [])
        if isinstance(lbl, dict)
    ]
    title_l = str(issue.get("title") or "").lower()
    if any(k in label for label in labels for k in ("enhancement", "feature")) or "[feature]" in title_l or "ux:" in title_l:
        return "feature"
    if any(k in label for label in labels for k in ("bug", "crash", "regression")) or any(
        w in title_l for w in ("bug", "crash", "fail", "error", "fix", "stuck", "broken", "unable")
    ):
        return "bug"
    return "feedback"


def stable_releases(releases: list[dict]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft") or rel.get("prerelease"):
            continue
        tag = str(rel.get("tag_name") or "").strip()
        published_at = str(rel.get("published_at") or "").strip()
        if STABLE_TAG_RE.fullmatch(tag) and published_at:
            items.append((published_at, tag))
    items.sort(key=lambda pair: pair[0])
    return items


def resolve_version(closed_at: str, sorted_releases: list[tuple[str, str]]) -> str:
    if not closed_at:
        return "next"
    for published_at, tag in sorted_releases:
        if published_at >= closed_at:
            return tag
    return "next"


def validate_payload(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError("contributors.json must be a JSON object")
    excluded = {str(item).strip().lower() for item in data.get("excluded_authors", [])}
    opt_out = {str(item).strip().lower() for item in data.get("opt_out_authors", [])}
    blocked = excluded | opt_out
    excluded_issues = {int(item) for item in data.get("excluded_issues", [])}
    contributors = data.get("contributors")
    if not isinstance(contributors, list):
        raise ValueError("contributors.json: 'contributors' must be a list")
    seen_handles: set[str] = set()
    seen_issues: set[int] = set()
    for person in contributors:
        if not isinstance(person, dict):
            raise ValueError("contributors.json: contributor entry must be an object")
        handle = str(person.get("handle") or "").strip()
        if not handle or not GITHUB_HANDLE_RE.fullmatch(handle):
            raise ValueError(f"contributors.json: invalid handle {handle!r}")
        lowered = handle.lower()
        if lowered in blocked or lowered.endswith("[bot]"):
            raise ValueError(f"contributors.json: excluded contributor {handle!r}")
        if lowered in seen_handles:
            raise ValueError(f"contributors.json: duplicate contributor {handle!r}")
        seen_handles.add(lowered)
        url = str(person.get("url") or "").strip()
        if not url.startswith("https://github.com/"):
            raise ValueError(f"contributors.json: invalid URL {url!r}")
        entries = person.get("contributions")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"contributors.json: {handle!r} has no contributions")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"contributors.json: invalid contribution under {handle!r}")
            issue_num = int(entry["issue"])
            if issue_num <= 0 or issue_num in excluded_issues:
                raise ValueError(f"contributors.json: forbidden issue #{issue_num}")
            if issue_num in seen_issues:
                raise ValueError(f"contributors.json: duplicate issue #{issue_num}")
            seen_issues.add(issue_num)
            ver = str(entry.get("version") or "").strip()
            if ver != "next" and not STABLE_TAG_RE.fullmatch(ver):
                raise ValueError(f"contributors.json: invalid version {ver!r} on #{issue_num}")
            for field in ("date", "en", "zh"):
                if not str(entry.get(field) or "").strip():
                    raise ValueError(f"contributors.json: missing {field} on #{issue_num}")


def merge_contributors(existing: dict, issues: list[dict], releases: list[dict]) -> dict:
    excluded_authors = list(existing.get("excluded_authors") or DEFAULT_EXCLUDED)
    opt_out_authors = list(existing.get("opt_out_authors") or [])
    excluded_issues = [int(n) for n in (existing.get("excluded_issues") or [])]

    blocked_handles = {item.strip().lower() for item in excluded_authors + opt_out_authors}
    blocked_issues = set(excluded_issues)

    existing_by_issue: dict[int, tuple[dict, dict]] = {}
    for person in existing.get("contributors", []):
        for entry in person.get("contributions", []):
            existing_by_issue[int(entry["issue"])] = (person, entry)

    sorted_releases = stable_releases(releases)
    grouped: dict[str, dict] = {}

    for issue in issues:
        if not isinstance(issue, dict) or "pull_request" in issue:
            continue
        if str(issue.get("state") or "").lower() != "closed":
            continue
        if str(issue.get("state_reason") or "").lower() in {"not_planned", "duplicate"}:
            continue
        issue_num = int(issue["number"])
        if issue_num in blocked_issues:
            continue
        author = issue.get("author") if "author" in issue else issue.get("user")
        login = str((author or {}).get("login") or "").strip()
        if not login or login.lower() in blocked_handles or login.lower().endswith("[bot]"):
            continue

        closed_at = str(issue.get("closed_at") or issue.get("closedAt") or "").strip()
        prev_pair = existing_by_issue.get(issue_num)
        if prev_pair is not None:
            prev_person, prev_entry = prev_pair
            if str(prev_person.get("handle") or "").strip().lower() != login.lower():
                # Guard against stale attribution if handle metadata differs.
                prev_person = {}
            prev_ver = str(prev_entry.get("version") or "next").strip()
            version = (
                prev_ver
                if STABLE_TAG_RE.fullmatch(prev_ver)
                else resolve_version(closed_at, sorted_releases)
            )
            merged_entry = {
                "issue": issue_num,
                "version": version,
                "date": str(prev_entry.get("date") or closed_at[:10]).strip(),
                "kind": str(prev_entry.get("kind") or infer_kind(issue)).strip(),
                "en": str(prev_entry.get("en") or clean_issue_title(issue.get("title", ""))).strip(),
                "zh": str(prev_entry.get("zh") or clean_issue_title(issue.get("title", ""))).strip(),
            }
            profile_url = str(prev_person.get("url") or f"https://github.com/{login}").strip()
            platform = str(prev_person.get("platform") or "GitHub").strip()
        else:
            title_clean = clean_issue_title(issue.get("title", ""))
            merged_entry = {
                "issue": issue_num,
                "version": resolve_version(closed_at, sorted_releases),
                "date": closed_at[:10] or "1970-01-01",
                "kind": infer_kind(issue),
                "en": title_clean,
                "zh": title_clean,
            }
            profile_url = f"https://github.com/{login}"
            platform = "GitHub"

        bucket = grouped.setdefault(
            login.lower(),
            {
                "handle": login,
                "platform": platform,
                "url": profile_url,
                "contributions": [],
            },
        )
        bucket["contributions"].append(merged_entry)

    contributors_list = list(grouped.values())
    for person in contributors_list:
        person["contributions"].sort(key=lambda item: (item["date"], int(item["issue"])), reverse=True)
    contributors_list.sort(
        key=lambda p: (
            p["contributions"][0]["date"],
            max(int(item["issue"]) for item in p["contributions"]),
        ),
        reverse=True,
    )

    result = {
        "excluded_authors": excluded_authors,
        "opt_out_authors": opt_out_authors,
        "excluded_issues": excluded_issues,
        "contributors": contributors_list,
    }
    validate_payload(result)
    return result


def gh_api_list(endpoint: str) -> list[dict]:
    cmd = ["gh", "api", "--paginate", "--slurp", endpoint]
    raw = subprocess.check_output(cmd, text=True, encoding="utf-8")
    pages = json.loads(raw)
    items: list[dict] = []
    for page in pages:
        if isinstance(page, list):
            items.extend(page)
        elif isinstance(page, dict):
            items.append(page)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync OpenFocus site/contributors.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="Write updates back to site/contributors.json")
    group.add_argument("--validate", action="store_true", help="Validate site/contributors.json offline")
    args = parser.parse_args(argv)

    existing = load_data(DATA_PATH)
    if args.validate:
        validate_payload(existing)
        print(f"Validated {len(existing['contributors'])} contributor(s) in {DATA_PATH.relative_to(ROOT)}.")
        return 0

    try:
        issues = gh_api_list(f"repos/{REPO}/issues?state=closed&per_page=100")
        releases = gh_api_list(f"repos/{REPO}/releases?per_page=100")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Failed to query GitHub API via gh CLI: {exc}", file=sys.stderr)
        return 1

    updated = merge_contributors(existing, issues, releases)
    serialized = json.dumps(updated, ensure_ascii=False, indent=2) + "\n"
    current_raw = DATA_PATH.read_text(encoding="utf-8")

    if serialized == current_raw:
        print("site/contributors.json is up to date with all closed external GitHub issues.")
        return 0

    if args.write:
        DATA_PATH.write_text(serialized, encoding="utf-8", newline="\n")
        total = sum(len(p["contributions"]) for p in updated["contributors"])
        print(f"Updated {DATA_PATH.relative_to(ROOT)}: {len(updated['contributors'])} contributors, {total} entries.")
        return 0

    old_nums = {
        int(e["issue"]): e["version"]
        for p in existing.get("contributors", [])
        for e in p.get("contributions", [])
    }
    new_nums = {
        int(e["issue"]): (p["handle"], e["version"])
        for p in updated["contributors"]
        for e in p["contributions"]
    }
    added = sorted(set(new_nums) - set(old_nums), reverse=True)
    promoted = sorted(n for n in (set(new_nums) & set(old_nums)) if new_nums[n][1] != old_nums[n])
    print(f"Pending contributors.json updates: +{len(added)} new issue(s), {len(promoted)} version update(s).")
    for n in added:
        handle, ver = new_nums[n]
        print(f"  + #{n} @{handle} -> {ver}")
    for n in promoted:
        handle, ver = new_nums[n]
        print(f"  ~ #{n} @{handle}: {old_nums[n]} -> {ver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
