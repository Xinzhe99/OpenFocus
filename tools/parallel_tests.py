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
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
# The same cuts on every machine, whatever its core count, so a result can be
# reproduced elsewhere. Slices differ several-fold in duration; this many keep
# four to eight processes busy.
SLICES = 16
_COLLECTION_FILE = "OPENFOCUS_PYTEST_COLLECTION_FILE"
_SLICE_FILE = "OPENFOCUS_PYTEST_SLICE_FILE"


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


def pytest_configure(config):
    selection = os.environ.pop(_SLICE_FILE, None)
    if selection:
        # Pytest has already parsed flags and selectors. Replace only the
        # collection targets, so explicit paths are not added to every slice.
        config.args = json.loads(Path(selection).read_text(encoding="utf-8"))


def pytest_collection_finish(session):
    report = os.environ.pop(_COLLECTION_FILE, None)
    if not report:
        return
    counts, targets = {}, {}
    has_node_selectors = any("::" in arg for arg in session.config.args)
    for item in session.items:
        name, separator, node = item.nodeid.partition("::")
        counts[name] = counts.get(name, 0) + 1
        if has_node_selectors:
            targets.setdefault(name, []).append(f"{item.path}{separator}{node}")
        elif name not in targets:
            targets[name] = [str(item.path)]
    basetemp = session.config.getoption("basetemp")
    Path(report).write_text(json.dumps([counts, targets, basetemp]), encoding="utf-8")


def _pytest_command(options, *extra):
    end = options.index("--") if "--" in options else len(options)
    return [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "tools.parallel_tests",
            *options[:end], *extra, *options[end:]]


def collect(pytest_options):
    """Collect the full selection before starting any test process."""
    with tempfile.TemporaryDirectory(prefix="openfocus-collect-") as scratch:
        report = Path(scratch) / "collection.json"
        environment = {**os.environ, _COLLECTION_FILE: str(report)}
        environment.pop(_SLICE_FILE, None)
        listing = subprocess.run(_pytest_command(pytest_options, "--collect-only"), cwd=ROOT,
                                 env=environment, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
        if listing.returncode or not report.is_file():
            sys.stdout.write(listing.stdout + listing.stderr)
            raise SystemExit(exit_code(listing.returncode) or 1)
        return json.loads(report.read_text(encoding="utf-8"))


def exit_code(returncode):
    """Map a process result to a shell exit code; a signal becomes 128 + n."""
    if returncode < 0:
        return 128 - returncode
    return returncode if returncode < 256 else 128 + (returncode & 0x7F)


def run(groups, jobs, pytest_options, targets, basetemp, crash_selection):
    """Run every group, at most ``jobs`` at a time; return the worst exit code."""
    grouped = os.environ.get("GITHUB_ACTIONS") == "true"
    pending = list(enumerate(groups, 1))
    running = {}
    worst = 0
    if basetemp is not None:
        basetemp = ROOT / basetemp
        basetemp.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openfocus-tests-") as scratch:
        while pending or running:
            while pending and len(running) < jobs:
                index, files = pending.pop(0)
                selection = Path(scratch) / f"slice-{index}.json"
                selection.write_text(json.dumps([target for name in files for target in targets[name]]),
                                     encoding="utf-8")
                extra = () if basetemp is None else ("--basetemp", str(basetemp / f"slice-{index}"))
                output = open(os.path.join(scratch, f"slice-{index}.txt"), "w+",
                              encoding="utf-8", errors="replace")
                process = subprocess.Popen(_pytest_command(pytest_options, *extra), cwd=ROOT,
                                           env={**os.environ, _SLICE_FILE: str(selection)},
                                           stdout=output, stderr=subprocess.STDOUT)
                running[index] = (process, output, files, time.monotonic(), selection)
            time.sleep(0.2)
            for index in sorted(running):
                process, output, files, started, selection = running[index]
                if process.poll() is None:
                    continue
                del running[index]
                code = exit_code(process.returncode)
                worst = max(worst, code)
                if code >= 128 and crash_selection is not None:
                    shutil.copyfile(selection, crash_selection)
                    crash_selection = None
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
    parser.add_argument("--crash-selection", type=Path,
                        help="save the first crashing slice's targets for a debugger")
    options, pytest_options = parser.parse_known_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # test output may not fit the console encoding
    jobs = max(1, options.jobs)
    counts, targets, basetemp = collect(pytest_options)
    groups = split_files(counts, options.slices)
    print(f"{sum(counts.values())} tests in {len(counts)} files, {len(groups)} slices, "
          f"{jobs} at a time", flush=True)
    worst = run(groups, jobs, pytest_options, targets, basetemp, options.crash_selection)
    print(f"\nworst exit code: {worst}", flush=True)
    return worst


if __name__ == "__main__":
    sys.exit(main())
