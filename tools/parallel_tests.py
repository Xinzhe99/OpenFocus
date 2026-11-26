"""Run the test suite as consecutive slices in parallel processes.

The suite is cut into runs of neighbouring test files. Each slice is an
ordinary pytest process that runs its files in the usual order, so a test
keeps the predecessors it has in a single-process run and the result does not
depend on which process finished first.

    python tools/parallel_tests.py [--jobs N] [--slices M] [pytest options]

Options this script does not know are passed to every pytest process.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TESTS = "tests"
# The same cuts on every machine, whatever its core count, so a result can be
# reproduced elsewhere. Slices differ several-fold in duration; this many keep
# four to eight processes busy.
SLICES = 16
# pytest lists one test per line ("file.py::test") or, when quieter, one file
# per line with its number of tests ("file.py: 12").
_NODE = re.compile(r"^(" + TESTS + r"/[^:\s]+\.py)(?:::|: (\d+)$)")


def split_files(counts, slices):
    """Cut an ordered {file: test count} mapping into consecutive groups.

    Groups keep the mapping's order, cover every file once, are never empty,
    and hold roughly equal numbers of tests.
    """
    files = list(counts)
    slices = max(1, min(int(slices), len(files)))
    total = sum(max(1, counts[name]) for name in files)
    groups, current, seen = [], [], 0
    for position, name in enumerate(files):
        current.append(name)
        seen += max(1, counts[name])
        left = len(files) - position - 1
        open_groups = slices - len(groups) - 1
        if open_groups and (seen >= total * (len(groups) + 1) / slices or left == open_groups):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def collect(pytest_options):
    """Return {file: test count} in collection order, uncollected files last."""
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
               *pytest_options]
    listing = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    counts = {}
    for line in listing.stdout.splitlines():
        match = _NODE.match(line.replace("\\", "/"))
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + int(match.group(2) or 1)
    if not counts:
        sys.stdout.write(listing.stdout + listing.stderr)
        raise SystemExit(listing.returncode or 1)
    # A file that failed to import has no tests listed. Keep it, so the slice
    # that runs it reports the error.
    for path in sorted((ROOT / TESTS).rglob("test_*.py")):
        counts.setdefault(path.relative_to(ROOT).as_posix(), 0)
    return counts


def exit_code(returncode):
    """Map a process result to a shell exit code; a signal becomes 128 + n."""
    if returncode < 0:
        return 128 - returncode
    return returncode if returncode < 256 else 128 + (returncode & 0x7F)


def run(groups, jobs, pytest_options):
    """Run every group, at most ``jobs`` at a time; return the worst exit code."""
    grouped = os.environ.get("GITHUB_ACTIONS") == "true"
    pending = list(enumerate(groups, 1))
    running = {}
    worst = 0
    with tempfile.TemporaryDirectory(prefix="openfocus-tests-") as scratch:
        while pending or running:
            while pending and len(running) < jobs:
                index, files = pending.pop(0)
                output = open(os.path.join(scratch, f"slice-{index}.txt"), "w+",
                              encoding="utf-8", errors="replace")
                command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
                           *pytest_options, *files]
                process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                           stderr=subprocess.STDOUT)
                running[index] = (process, output, files, time.monotonic())
            time.sleep(0.2)
            for index in sorted(running):
                process, output, files, started = running[index]
                if process.poll() is None:
                    continue
                del running[index]
                code = exit_code(process.returncode)
                worst = max(worst, code)
                title = (f"tests {index} of {len(groups)}: {files[0]} to {files[-1]}, "
                         f"exit code {code}, {time.monotonic() - started:.0f} s")
                output.seek(0)
                print(f"::group::{title}" if grouped else f"\n===== {title} =====", flush=True)
                sys.stdout.write(output.read())
                output.close()
                if grouped:
                    print("::endgroup::", flush=True)
                if code:
                    print(f"FAILED: {title}", flush=True)
    return worst


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="pytest processes to run at once (default: CPU count)")
    parser.add_argument("--slices", type=int, default=SLICES,
                        help=f"consecutive slices to cut the suite into (default: {SLICES})")
    options, pytest_options = parser.parse_known_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # test output may not fit the console encoding
    jobs = max(1, options.jobs)
    counts = collect(pytest_options)
    groups = split_files(counts, options.slices)
    print(f"{sum(counts.values())} tests in {len(counts)} files, {len(groups)} slices, "
          f"{jobs} at a time", flush=True)
    worst = run(groups, jobs, pytest_options)
    print(f"\nworst exit code: {worst}", flush=True)
    return worst


if __name__ == "__main__":
    sys.exit(main())
