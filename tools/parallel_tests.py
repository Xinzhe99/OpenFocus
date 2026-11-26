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
_NODE = re.compile(r"^(" + TESTS + r"/[^:\r\n]+\.py)(?:::|: (\d+)$)")


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


def parse_listing(text):
    """Read pytest's collection listing as a mapping of files to test counts."""
    counts = {}
    for line in text.splitlines():
        line = line.replace("\\", "/")
        match = _NODE.match(line)
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + int(match.group(2) or 1)
    return counts


def without_option(options, name):
    """Drop ``name VALUE`` and ``name=VALUE`` from a list of pytest options."""
    kept, skip = [], False
    for option in options:
        if skip:
            skip = False
        elif option == name:
            skip = True
        elif not option.startswith(name + "="):
            kept.append(option)
    return kept


def slice_options(options, index):
    """The pytest options for one slice.

    pytest empties ``--basetemp`` when it starts, so slices that shared one
    would delete each other's temporary files. Each gets a folder inside it.
    """
    result, take_value = [], False
    for option in options:
        if take_value:
            result.append(os.path.join(option, f"slice-{index}"))
            take_value = False
        elif option == "--basetemp":
            result.append(option)
            take_value = True
        elif option.startswith("--basetemp="):
            result.append("--basetemp=" + os.path.join(option.split("=", 1)[1], f"slice-{index}"))
        else:
            result.append(option)
    return result


def collect(pytest_options):
    """Collect the full selection before starting any test process."""
    # The last verbosity option wins, so the listing has one test per line
    # whatever -v or -q the caller passes for the runs themselves.
    command = [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider",
               *without_option(pytest_options, "--basetemp"), "--verbosity=-1"]
    listing = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    if listing.returncode:
        sys.stdout.write(listing.stdout + listing.stderr)
        raise SystemExit(exit_code(listing.returncode))
    counts = parse_listing(listing.stdout)
    if not counts:
        sys.stdout.write(listing.stdout + listing.stderr)
        raise SystemExit(1)
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
                           *slice_options(pytest_options, index), *files]
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
    for position, option in enumerate(pytest_options):
        # pytest creates a slice's base temp folder but not its parent.
        if option.startswith("--basetemp="):
            os.makedirs(option.split("=", 1)[1], exist_ok=True)
        elif option == "--basetemp" and position + 1 < len(pytest_options):
            os.makedirs(pytest_options[position + 1], exist_ok=True)
    counts = collect(pytest_options)
    groups = split_files(counts, options.slices)
    print(f"{sum(counts.values())} tests in {len(counts)} files, {len(groups)} slices, "
          f"{jobs} at a time", flush=True)
    worst = run(groups, jobs, pytest_options)
    print(f"\nworst exit code: {worst}", flush=True)
    return worst


if __name__ == "__main__":
    sys.exit(main())
