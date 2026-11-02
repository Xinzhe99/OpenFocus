"""Check translated manuals for missing chapters and broken local references.

Translations may use different examples and tables, so matching raw heading or
row counts does not establish content parity. A maintainer must review wording
and feature coverage separately.
"""
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

DOCS = Path(__file__).resolve().parent.parent / "docs"
SOURCE = DOCS / "USER_MANUAL_EN.md"
LANGS = ("ZH", "JA", "ES", "DE", "FR", "RU", "KO", "PT", "IT", "NL", "PL")

_CHAPTER = re.compile(r"^##\s+(\d+)\.\s+")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LINK = re.compile(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)')


def structure(path):
    chapters = []
    missing_links = []
    fence = None
    for line in path.read_text(encoding="utf-8").splitlines():
        marker = _FENCE.match(line)
        if marker:
            token, suffix = marker.groups()
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence) and not suffix.strip():
                fence = None
            continue
        if fence is not None:
            continue
        chapter = _CHAPTER.match(line)
        if chapter:
            chapters.append(int(chapter.group(1)))
        for link in _LINK.findall(line):
            url = urlsplit(link.strip("<>"))
            if url.scheme or url.netloc or not url.path:
                continue
            if not (path.parent / unquote(url.path)).exists():
                missing_links.append(link)
    return {"chapters": chapters, "unclosed_fence": fence,
            "missing_links": missing_links}


def main():
    src = structure(SOURCE)
    failures = []
    expected = src["chapters"]
    if not expected or expected != list(range(1, len(expected) + 1)):
        failures.append("EN: chapters are missing or out of order")
    if src["unclosed_fence"]:
        failures.append("EN: unclosed code fence")
    failures.extend(f"EN: missing local file {link}" for link in src["missing_links"])
    print(f"source: {len(expected)} numbered chapters")
    for code in LANGS:
        path = DOCS / f"USER_MANUAL_{code}.md"
        if not path.exists():
            failures.append(f"{code}: missing file")
            continue
        cur = structure(path)
        problems = []
        if cur["chapters"] != expected:
            problems.append("numbered chapters differ from English")
        if cur["unclosed_fence"]:
            problems.append("unclosed code fence")
        problems.extend(f"missing local file {link}" for link in cur["missing_links"])
        status = "FAIL" if problems else "OK  "
        print(f"  {status} {code}: {len(cur['chapters'])} numbered chapters")
        for problem in problems:
            print(f"       {problem}")
            failures.append(f"{code}: {problem}")
    if failures:
        print(f"\n{len(failures)} problem(s)")
        return 1
    print("\nchapter order, code fences and local file links are valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
