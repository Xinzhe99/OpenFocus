"""Check every translated user manual against the English source.

Translations drift: a missing section or a broken table is invisible until a
user cannot find the feature. This compares structure, not wording.
"""
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
SOURCE = DOCS / "USER_MANUAL_EN.md"
LANGS = ("ZH", "JA", "ES", "DE", "FR", "RU", "KO", "PT", "IT", "NL", "PL")

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
_FENCE = re.compile(r"^```", re.M)


def structure(path):
    text = path.read_text(encoding="utf-8")
    headings = _HEADING.findall(text)
    levels = [len(h[0]) for h in headings]
    fences = len(_FENCE.findall(text))
    # table rows: lines that start and end with a pipe
    rows = [ln for ln in text.splitlines()
            if ln.strip().startswith("|") and ln.strip().endswith("|")]
    return {"levels": levels, "fences": fences, "rows": len(rows),
            "headings": len(headings)}


def main():
    src = structure(SOURCE)
    failures = []
    print(f"source: {src['headings']} headings, {src['fences']} fences, "
          f"{src['rows']} table rows")
    for code in LANGS:
        path = DOCS / f"USER_MANUAL_{code}.md"
        if not path.exists():
            failures.append(f"{code}: missing file")
            continue
        cur = structure(path)
        problems = []
        if cur["headings"] != src["headings"]:
            problems.append(f"headings {cur['headings']} != {src['headings']}")
        if cur["levels"] != src["levels"]:
            first = next((i for i, (a, b) in
                          enumerate(zip(cur["levels"], src["levels"])) if a != b),
                         None)
            problems.append(f"heading structure differs at index {first}")
        if cur["fences"] != src["fences"]:
            problems.append(f"code fences {cur['fences']} != {src['fences']}")
        if cur["rows"] != src["rows"]:
            problems.append(f"table rows {cur['rows']} != {src['rows']}")
        status = "OK  " if not problems else "FAIL"
        print(f"  {status} {code}: {cur['headings']} headings, "
              f"{cur['fences']} fences, {cur['rows']} rows")
        for p in problems:
            print(f"       {p}")
            failures.append(f"{code}: {p}")
    if failures:
        print(f"\n{len(failures)} problem(s)")
        return 1
    print("\nall manuals match the English structure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
